# DFlash — Speculative Decoding with Target KV Cache Sharing

This repository contains a forked vLLM (`vllm/`) and the DFlash speculator training library (`speculators/`).
The main modification introduced here extends DFlash so that the draft model can consume **K/V tensors
computed directly by the target model** instead of re-projecting target hidden states through the
draft's own projection weights.

---

## Repository Structure

```
vllm_dflash/
├── vllm/           # Forked vLLM with DFlash inference support
│   └── vllm/
│       ├── model_executor/models/qwen3_dflash.py   # Draft model (inference)
│       └── v1/worker/gpu/spec_decode/dflash/
│           └── speculator.py                        # DFlashSpeculator
├── speculators/    # Training library for DFlash draft models
│   └── src/speculators/models/dflash/
│       └── model_definitions.py                     # Draft attention layer (training)
└── vllm.txt        # Quick-reference commands
```

---

## Background: Original KV Cache Flow

In the original DFlash design the draft model never creates K/V cache by reading from the
target's paged KV tensors directly. Instead it runs its own projection:

```
Target forward pass
  → aux hidden states at target_layer_ids (e.g. layers 2, 18, 33)
  → combined via draft fc layer  →  context_states  [num_tokens, hidden_size]

precompute_and_store_context_kv(context_states)
  → RMSNorm(context_states)
  → fused GEMM:  context_states @ draft_kv_weight  →  draft K/V projections
  → per-layer k_norm + RoPE
  → write into draft KV cache at context slot positions
```

Each draft layer therefore holds K/V values **derived from target hidden states through the
draft's own K/V projection weights** — a separate computation from what the target stored.

---

## Change: Bring K/V Directly from the Target Model

The new path lets the draft skip its own KV projection entirely and instead
receive the **exact same K/V values** that the target already computed and
stored in its paged KV cache.

```
Target forward pass
  → for each target layer in kv_target_layer_ids:
      qkv_proj → q, k, v
      k_norm(k) → k                   ← target's own k_norm applied
      RoPE(q, k) → q, k               ← target's own RoPE applied
      self.attn(q, k, v) ←── HOOK captures k, v HERE (post-norm+RoPE)
      (target writes these k, v into its own paged KV cache)

precompute_and_store_context_kv_from_target(target_k_layers, target_v_layers)
  → NO k_norm, NO RoPE   (already applied by target)
  → write directly into draft KV cache at context slot positions
```

The draft's context KV cache now contains identical values to what the target
stored — no separate projection, no separate norm, no separate RoPE pass.

Both paths remain available. The system falls back to the original hidden-states
path when `kv_target_layer_ids` is not set in the drafter config.

---

## Modified Files

### 1. `speculators/src/speculators/models/dflash/model_definitions.py`

#### `Qwen3DFlashAttention.forward`

Two mutually exclusive ways to supply context K/V are now supported:

| Parameter | Type | Description |
|---|---|---|
| `target_hidden` | `Tensor \| None` | **Original path.** Project K/V from target hidden states via draft `k_proj` / `v_proj`. |
| `target_k` + `target_v` | `Tensor \| None` | **New path.** Raw K/V projections already computed by the corresponding target attention layer. Skips `k_proj(target_hidden)` and `v_proj(target_hidden)`. |

When `target_k` / `target_v` are given, `k_norm` and `RoPE` are still applied to the
full concatenated K sequence (context + noise) exactly as before — only the projection step
is replaced.

```python
# Original call (no change required for existing code)
layer(hidden_states=..., target_hidden=fc_output, ...)

# New call — pass target K/V directly
layer(hidden_states=..., target_k=k_from_target, target_v=v_from_target, ...)
```

#### `Qwen3DFlashDecoderLayer.forward`

The `target_k` / `target_v` parameters are forwarded to `self_attn`. Calling with
`target_hidden` still works as before.

---

### 2. `vllm/vllm/model_executor/models/qwen3_dflash.py`

#### New method: `DFlashQwen3Model.precompute_and_store_context_kv_from_target`

```python
model.precompute_and_store_context_kv_from_target(
    target_k_layers,       # list of Tensor [num_ctx, num_kv_heads, head_dim], one per draft layer
    target_v_layers,       # list of Tensor [num_ctx, num_kv_heads, head_dim], one per draft layer
    context_positions,     # [num_ctx]
    context_slot_mapping,  # [num_ctx] — None during dummy runs
)
```

**What it does** compared to `precompute_and_store_context_kv`:

| Step | Original | New |
|---|---|---|
| RMSNorm on hidden states | ✅ | ❌ skipped |
| Fused KV projection (GEMM) | ✅ | ❌ skipped |
| Per-layer K-norm | ✅ | ✅ |
| Fused RoPE | ✅ | ✅ |
| KV cache write | ✅ | ✅ |

The original `precompute_and_store_context_kv` is unchanged and still available.

---

### 3. `vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`

#### `DFlashSpeculator.propose`

A new optional parameter is added:

```python
aux_kv_states: list[tuple[Tensor, Tensor]] | None = None
```

- Each element is `(k, v)` for one draft layer, shape `[num_tokens, num_kv_heads, head_dim]`.
- When provided, `propose` calls `precompute_and_store_context_kv_from_target`.
- When `None` (default), the original `precompute_and_store_context_kv` path is used.

This keeps full backward compatibility — no caller needs to change unless it wants
to enable the new path.

---

## Three KV-Sharing Modes

The mode is selected by **one config change** in the drafter's `config.json`.
No changes to `model_runner.py` or the target model code are needed.

### All four modes at a glance

| | `hidden_states` | `raw_copy` | `copy` | `alias` |
|---|---|---|---|---|
| **Config** | *(default)* | `"kv_mode": "raw_copy"` | `"kv_mode": "copy"` | `"kv_mode": "alias"` |
| **Hook point** | none | `qkv_proj` output | `self_attn.attn` input | `self_attn.attn` input |
| **Captured at** | — | pre-k_norm, pre-RoPE | post-k_norm, post-RoPE | post-k_norm, post-RoPE |
| **k_norm applied** | draft's own | **draft's own** (on target K) | none (already applied) | none |
| **RoPE applied** | draft's own | **draft's own** (on target K) | none (already applied) | none |
| **Precompute write** | yes | yes | yes | **skipped** |
| **Draft KV cache** | separate | separate | separate | **aliased = target's** |
| **GPU memory** | full draft | full draft | full draft | **zero extra** |
| **Requires same kv dims** | ✗ | ✓ | ✓ | ✓ |

For DFlash Qwen3-8B: target=8 KV heads, head_dim=128; draft=8 KV heads, head_dim=128 ✓

---

### Mode 0 — `hidden_states` (original, default)

No config change needed. Works with any model regardless of dimension differences.

---

### Mode 1 — `raw_copy` (draft's own norm/RoPE on target K/V)

```json
{
  "dflash_config": {
    "kv_target_layer_ids": [0, 9, 18, 27, 35],
    "kv_mode": "raw_copy"
  }
}
```

Hook fires on `self_attn.qkv_proj` — captures K/V **before** k_norm and RoPE.
The draft's own k_norm weights and RoPE are then applied (same as original path
but the *source* K/V comes from the target instead of target hidden states).
Useful when the draft has different norm/RoPE parameters than the target.

---

### Mode 2 — `copy` (exact target K/V, separate cache)

```json
{
  "dflash_config": {
    "kv_target_layer_ids": [0, 9, 18, 27, 35],
    "kv_mode": "copy"
  }
}
```

Hook fires on `self_attn.attn` — captures K/V **after** k_norm and RoPE.
These are the exact values the target writes to its paged cache.  Written
directly to the draft's own (separate) KV cache with no further transformation.

---

### Mode 3 — `alias` (shared physical cache, zero overhead)

```json
{
  "dflash_config": {
    "kv_target_layer_ids": [0, 9, 18, 27, 35],
    "kv_mode": "alias"
  }
}
```

Same hook as `copy`, **plus** after `set_attn`:
- `draft_layer[i].self_attn.attn.kv_cache` is replaced with a reference to
  `target_layer[kv_target_layer_ids[i]].self_attn.attn.kv_cache`
- The draft KV cache allocation is eliminated entirely (shared tensor)
- In `propose()`, the precompute step is **skipped** — the target's forward pass
  already wrote K/V into the shared tensor at the target's slot positions
- During speculative generation, the draft reads context K/V from the target's
  physical pages and writes new speculative K/V to its own allocated blocks

---

### Choosing layer IDs

```python
M, N = 36, 5   # target layers, draft layers (Qwen3-8B example)
ids = [round(i * (M - 1) / (N - 1)) for i in range(N)]
# → [0, 9, 18, 27, 35]
```

**Choosing layer IDs:**  Pick one target layer per draft layer such that the mapping
makes semantic sense.  A common heuristic for a target with `M` layers and a draft
with `N` layers is evenly spaced indices:

```python
import math
target_layers = sorted(
    round(i * (M - 1) / (N - 1)) for i in range(N)
)
```

For example, with a 32-layer target and a 4-layer draft: `[0, 10, 21, 31]`.

### Training with mode 3 — K/V supplied directly from the target

**All four components are now implemented end-to-end.**
Training with `--kv-mode alias` (or `copy`) means the draft model receives the
target's post-processed K/V at every training step — exactly matching what happens
at inference time.  No manual hooks or code changes are needed.

#### How it works (data → model)

```
vLLM generates a sequence
  ExampleHiddenStatesConnector
    → extracts hidden states  (as before)
    → extracts K and V from real target attention layers at kv_target_layer_ids
    → saves to disk: { hidden_states, token_ids, k_0, v_0, k_9, v_9, ... }

data.py loads the file
  → detects k_*/v_* keys
  → stacks into target_k_all [seq, num_layers, nkv*hd]
              and target_v_all [seq, num_layers, nkv*hd]

core.py forward()
  → receives target_k_all / target_v_all
  → skips fc() projection
  → builds per-layer (k, v) pairs and routes to each attention layer

model_definitions.py Qwen3DFlashAttention.forward()
  target_kv_processed=True path:
    context K/V  → used as-is  (no k_norm, no RoPE — already applied by target)
    noise K/V    → draft k_proj/v_proj → k_norm → RoPE  (same as always)
```

This means the draft learns to predict tokens **given the target's exact K/V at
context positions**, so there is no distribution gap between training and alias-mode
inference.

#### Activation

Pass `--kv-mode` and `--kv-target-layer-ids` to `train.py`:

```bash
python scripts/train.py \
  ...  \
  --kv-mode alias \
  --kv-target-layer-ids 0 9 18 27 35
```

The vLLM server must be running normally (it does not need any special flag — the
connector reads `dflash_config.kv_target_layer_ids` from the draft checkpoint's
`config.json` automatically).

#### Modified files for K/V training

| File | Change |
|---|---|
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` | Reads `kv_target_layer_ids` from `dflash_config`; registers target-layer KV caches; extracts K and V and saves them alongside hidden states |
| `speculators/src/speculators/train/data.py` | Detects `k_{id}` / `v_{id}` keys in the safetensors file; stacks into `target_k_all` / `target_v_all` tensors |
| `speculators/src/speculators/models/dflash/core.py` | Accepts `target_k_all` / `target_v_all`; skips `fc()` projection when present; builds per-layer K/V list |
| `speculators/src/speculators/models/dflash/model_definitions.py` | New `target_kv_processed=True` path in `Qwen3DFlashAttention.forward` — applies k_norm+RoPE only to draft noise tokens; uses context K/V directly |

---

## Quick-Reference Commands

```bash
# 1. Prepare training data
python scripts/prepare_data.py \
  --model Qwen/Qwen3-8B \
  --data sharegpt \
  --output ./output/dflash_qwen3_8b_sharegpt \
  --max-samples 1000 \
  --seq-length 1024

# 2. Launch vLLM server (tensor-parallel across 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py Qwen/Qwen3-8B \
  --target-layer-ids 2 18 33 \
  -- --tensor-parallel-size 2 --port 8013

# 3. Train the DFlash draft model (mode 0 — original hidden-states path)
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output/dflash_qwen3_8b_sharegpt \
  --vllm-endpoint http://localhost:8013/v1 \
  --save-path ./output/dflash_qwen3_8b_sharegpt/checkpoints \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 3072 \
  --num-layers 5 \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --lr 3e-4 \
  --total-seq-len 512 \
  --on-missing generate \
  --on-generate delete

# 3b. Train with mode 3 — draft uses target K/V directly (alias training)
#     The saved checkpoints will have kv_mode=alias in config.json automatically.
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output/dflash_qwen3_8b_sharegpt \
  --vllm-endpoint http://localhost:8013/v1 \
  --save-path ./output/dflash_qwen3_8b_sharegpt/checkpoints \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 3072 \
  --num-layers 5 \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --lr 3e-4 \
  --total-seq-len 512 \
  --on-missing generate \
  --on-generate delete \
  --kv-mode alias \
  --kv-target-layer-ids 0 9 18 27 35 \
  --no-resume-from-checkpoint \
```


python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output/dflash_qwen3_8b_sharegpt \
  --vllm-endpoint http://localhost:8013/v1 \
  --save-path ./output/dflash_qwen3_8b_sharegpt/checkpoints \
  --speculator-type dflash \
  --block-size 16 \
  --max-anchors 3072 \
  --num-layers 5 \
  --draft-vocab-size 32000 \
  --epochs 1 --lr 3e-4 \
  --total-seq-len 3072 \
  --on-missing generate \
  --on-generate delete \
  --target-layer-ids 0 9 18 27 35 \
  --logger wandb \
  --run-name dflash-qwen3-8b-mode3_nocheck \
