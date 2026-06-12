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

The new path lets the draft skip its own KV projection entirely and instead receive
raw K/V projections that were already computed inside the target attention layers.

```
Target forward pass
  → raw K/V projections captured at target_layer_ids
      (pre k_norm, pre RoPE — one (k, v) pair per draft layer)

precompute_and_store_context_kv_from_target(target_k_layers, target_v_layers)
  → per-layer k_norm + RoPE   (projection step is removed)
  → write into draft KV cache at context slot positions
```

Both paths remain available and the system falls back to the original hidden-states path
when `aux_kv_states` is not provided.

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

## Wiring Up the New Path

To activate the target-KV path end-to-end you need to supply `aux_kv_states` from
the target model forward pass. The steps are:

### Step 1 — Capture raw K/V in the target model

Add hooks (or modify the target model's `forward`) to capture **raw K projections**
(before `k_norm` and `RoPE`) from the layers listed in `target_layer_ids`.
One `(k, v)` tensor pair is needed per draft layer; if `num_draft_layers > len(target_layer_ids)`
you can repeat or interpolate.

### Step 2 — Propagate through the model runner

In `vllm/v1/worker/gpu/model_runner.py`, unpack the captured K/V alongside
`aux_hidden_states` and pass them to `speculator.propose`:

```python
speculator.propose(
    ...,
    aux_hidden_states=aux_hidden_states,
    aux_kv_states=aux_kv_states,   # new
)
```

### Step 3 — Training

Pass `target_k` / `target_v` to each `Qwen3DFlashDecoderLayer` call in `core.py`
instead of (or alongside) `target_hidden`. The raw K/V tensors should come from
forward hooks on the target model's attention layers at `target_layer_ids`.

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

# 3. Train the DFlash draft model
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
  --target-layer-ids 2 18 33 \
  --epochs 5 \
  --lr 3e-4 \
  --total-seq-len 512 \
  --on-missing generate \
  --on-generate delete
```
