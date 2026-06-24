# DFlash Inference Guide

This document describes how DFlash speculative decoding runs at **inference time** in this repo (vLLM V2 GPU path), with emphasis on **KV cache generation** and how data flows from the **target model** to the **drafter**.

---

## Overview

Each decode step runs two models:

| Model | Role |
|-------|------|
| **Target** (`Qwen/Qwen3-8B`) | Verifies tokens; runs a full forward pass |
| **Draft** (DFlash checkpoint) | Proposes up to `speculative_tokens` (e.g. 15) draft tokens in parallel |

The benchmark client (`test_bench_all.py`) only sends HTTP requests. All KV logic lives **inside vLLM**.

---

## End-to-end pipeline (one decode step)

```
Scheduler
    │
    ▼
① execute_model()  ──►  Target Qwen3-8B forward
    │                      • qkv_proj → k_norm → RoPE → attn
    │                      • K/V written to TARGET kv_cache
    │                      • aux hidden states at layers 1,9,17,25,33
    │
    ▼
② sample_tokens()  ──►  Target samples verified/bonus token
    │
    ▼
③ speculator.propose()  ──►  DFlashSpeculator
    │                      • prepare_dflash_inputs (slots, mask tokens)
    │                      • combine_hidden_states (draft fc)
    │                      • fill DRAFT context KV (kv_mode-dependent)
    │                      • _generate_draft() forward
    │
    ▼
④ Draft token proposals returned to scheduler for target verification
```

### Entry points in code

| Step | File | Function |
|------|------|----------|
| Target forward | `vllm/vllm/v1/worker/gpu/model_runner.py` | `execute_model()` |
| Target attention / KV write | `vllm/vllm/model_executor/models/qwen3.py` | `Qwen3Attention.forward()` |
| Draft proposal | `vllm/vllm/v1/worker/gpu/model_runner.py` | `sample_tokens()` → `speculator.propose()` |
| DFlash logic | `vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | `DFlashSpeculator.propose()` |
| Draft model | `vllm/vllm/model_executor/models/qwen3_dflash.py` | `DFlashQwen3Model`, `DFlashQwen3Attention` |

---

## Two separate KV streams

DFlash uses **two different KV computations**. Do not confuse them.

### 1. Context KV (prefix / history tokens)

Populated in `propose()` **before** the draft forward. Controlled by `dflash_config.kv_mode` in the checkpoint `config.json`.

| `kv_mode` | Source of context K/V | Code path |
|-----------|----------------------|-----------|
| `hidden_states` (default) | Draft **re-projects** K/V from target hidden states via draft weights | `precompute_and_store_context_kv()` |
| `raw_copy` | Target K/V (pre-norm) captured by hooks → draft applies its own norm+RoPE | hooks + `precompute_and_store_context_kv_from_target(skip_norm_and_rope=False)` |
| `copy` | Target K/V (post-norm+RoPE) captured by hooks → written directly to draft cache | hooks + `precompute_and_store_context_kv_from_target(skip_norm_and_rope=True)` |
| `alias` | **Shared** physical KV cache with target; no separate context write | `_alias_kv_caches()` at load time; skip precompute in `propose()` |

**Location in `propose()`:**

```text
vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
  ~620–671  Context K/V population (mode-driven)
```

**Draft-side context KV write (hidden_states / copy paths):**

```text
vllm/vllm/model_executor/models/qwen3_dflash.py
  precompute_and_store_context_kv()              # hidden_states mode
  precompute_and_store_context_kv_from_target()  # copy / raw_copy mode
```

**Target-side KV capture (copy / raw_copy hooks):**

```text
vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
  _KVCaptureHooks  — hooks on target layers at kv_target_layer_ids
```

**Alias wiring:**

```text
vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
  load_draft_model() → _alias_kv_caches()  (in set_attn)
```

### 2. Query KV (draft / mask tokens)

**Always** computed by the draft's own `qkv_proj` during `_generate_draft()`, regardless of `kv_mode`.

```text
vllm/vllm/model_executor/models/qwen3_dflash.py
  DFlashQwen3Attention.forward()
    qkv = linear(hidden_states, qkv_proj.weight)
    q, k, v = split → norm → RoPE → attn(q, k, v)
```

The draft forward assumes context K/V is **already** in the draft KV cache; it only computes K/V for the speculative query tokens (bonus + mask tokens).

---

## Target forward: where target KV is born

```text
vllm/vllm/model_executor/models/qwen3.py — Qwen3Attention.forward()

  qkv_proj(hidden_states)
    → q_norm, k_norm
    → rotary_emb(positions, q, k)
    → attn(q, k, v)   ← writes into TARGET paged kv_cache
```

During the same forward, **auxiliary hidden states** are recorded at layers listed in `aux_hidden_state_layer_ids` (e.g. `[1, 9, 17, 25, 33]`). These are **not** KV tensors; they are features passed to the draft `fc` layer.

---

## Inside `propose()`: hidden states → draft cache

### Hidden-state path (always runs)

```text
aux_hidden_states (from target layers)
    → torch.cat
    → combine_hidden_states()  [draft fc layer]
    → self.hidden_states buffer
```

Used for draft token prediction and (in `hidden_states` kv_mode) as input to context KV projection.

### Slot mapping

```text
prepare_dflash_inputs()  [Triton kernel]
    → context_positions
    → context_slot_mapping   (where context K/V lives in DRAFT cache)
    → query input_ids        (bonus token + mask tokens)
    → query positions / slot mappings
```

---

## KV mode configuration

Add to checkpoint `config.json`:

```json
"dflash_config": {
  "kv_mode": "alias",
  "kv_target_layer_ids": [1, 9, 17, 25, 33]
}
```

| Field | Meaning |
|-------|---------|
| `aux_hidden_state_layer_ids` | Target layers whose **hidden states** feed the draft `fc` layer |
| `kv_target_layer_ids` | Target layers whose **K/V** are shared/copied (copy/alias/raw_copy only) |
| `kv_mode` | How context K/V reaches the draft cache |

vLLM merges `dflash_config` from the checkpoint in:

```text
vllm/vllm/transformers_utils/configs/speculators/algos.py — update_dflash()
```

Mode is read at draft load time in:

```text
vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py — load_draft_model()
```

### Mode selection notes

| Mode | Best for | Caveat |
|------|----------|--------|
| `hidden_states` | Default; no extra config | Draft K/V ≠ target K/V; train/inference aligned if trained this way |
| `copy` | Exact target K/V in separate draft cache | Forward **hooks** on target; may not fire under CUDA graphs → silent fallback to `hidden_states` |
| `alias` | Zero extra KV memory; uses target cache directly | Requires same `num_kv_heads` / `head_dim`; recommended for inference with CUDA graphs |
| `raw_copy` | Target pre-norm K/V + draft norm/RoPE | Same hook caveat as `copy` |

If `copy`/`raw_copy` hooks return empty, vLLM falls back to `precompute_and_store_context_kv(hidden_states)`.

---

## Serving a trained checkpoint

**Do not** use `launch_vllm.py` for inference — that script is for **training** (hidden-state extraction).

```bash
conda activate vllm_ali   # or vllm_speculator

bash test_bench/run_server.sh
```

Equivalent command:

```bash
vllm serve \
  ./speculators/output/dflash_qwen3_8b_sharegpt/checkpoints/checkpoint_best \
  --port 8013 \
  --tensor-parallel-size 1 \
  --served-model-name "Qwen/Qwen3-8B" \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256
```

DFlash with 15 speculative tokens needs raised batch limits on H100 (default `8192 × 1024 seqs` yields negative `max_num_scheduled_tokens`).

### Benchmark

```bash
python test_bench/test_bench_all.py \
  --base-url http://localhost:8013 \
  --model "Qwen/Qwen3-8B" \
  --dataset gsm8k \
  --num-prompts 100 \
  --concurrency 8
```

Use `--base-url http://localhost:8013` (no `/v1` suffix). The script appends `/v1/completions` itself.

---

## Verify inference is working

### Server startup logs

Look for:

```text
DFlash KV mode: alias — will alias draft kv_cache tensors to target layers [1, 9, 17, 25, 33]
DFlash KV mode: alias — aliased 5 draft layer(s) to target kv_cache.
```

Or for `hidden_states`:

```text
DFlash KV mode: hidden_states (original projection path).
```

### Prometheus metrics (`/metrics`)

```text
vllm:spec_decode_num_drafts_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_accepted_tokens_per_pos_total
```

### Acceptance length

| Accept length | Meaning |
|---------------|---------|
| ~1.0 | Draft almost never matches target (no speedup) |
| >1.5 | Meaningful speculative decoding |

Low acceptance is usually **draft quality** (undertrained checkpoint), not necessarily KV mode. KV mode affects **context attention input**; draft **token** correctness is a separate training issue.

---

## Data-flow diagram

```
TARGET FORWARD (execute_model)
──────────────────────────────
input tokens
    │
    ▼
Qwen3 layers 0…35
    │
    ├─► Target kv_cache (all layers, target block table slots)
    │
    └─► aux hidden states @ layers 1,9,17,25,33
              │
              ▼
PROPOSE (DFlashSpeculator.propose)
──────────────────────────────
aux hiddens ──fc──► draft hidden states
                         │
         ┌───────────────┼────────────────┐
         │               │                │
   hidden_states      copy/raw_copy      alias
   draft qkv_proj      hook → copy        shared kv_cache
         │               │                │
         └───────────────┴────────────────┘
                         │
                         ▼
              DRAFT kv_cache (context slots)
                         │
DRAFT FORWARD (_generate_draft)
──────────────────────────────
mask/bonus tokens ──qkv_proj──► query K/V
                         │
              attn: read context KV + query KV
                         │
                         ▼
              draft token logits → verification by target
```

---

## Related files

| File | Purpose |
|------|---------|
| `test_bench/run_server.sh` | Serve DFlash checkpoint on port 8013 |
| `test_bench/test_bench_all.py` | Multi-dataset benchmark client |
| `speculators/output/.../checkpoints/checkpoint_best/config.json` | Draft config (`speculators_config`, `dflash_config`) |
| `vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | Core DFlash inference orchestration |
| `vllm/vllm/model_executor/models/qwen3_dflash.py` | Draft model + context KV precompute |
| `vllm/vllm/model_executor/models/qwen3.py` | Target attention + target KV write |
| `README.md` | KV mode design (modes 0–3) and training notes |

---

## Training vs inference

| | Training (`launch_vllm.py`) | Inference (`vllm serve checkpoint`) |
|--|----------------------------|-------------------------------------|
| Purpose | Extract hidden states (+ optional K/V) to disk | Speculative decoding |
| Speculative method | `extract_hidden_states` | `dflash` |
| Loads draft checkpoint | No | Yes |
| KV connector | `ExampleHiddenStatesConnector` | None |

Do not benchmark inference against a `launch_vllm.py` server — that is the training data pipeline, not production DFlash serving.
