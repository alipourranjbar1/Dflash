#!/bin/bash
# Serve trained DFlash speculator + Qwen/Qwen3-8B target for inference/benchmarking.
#
# NOT launch_vllm.py — that script is only for online training (hidden-state extraction).
#
# Usage (from anywhere):
#   conda activate vllm_ali   # or vllm_speculator
#   bash test_bench/run_server.sh
#
# Then benchmark:
#   python test_bench/test_bench_all.py \
#     --base-url http://localhost:8013 \

#     --model "Qwen/Qwen3-8B" \
#     --dataset gsm8k --num-prompts 100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${SCRIPT_DIR}/../speculators/output/dflash_qwen3_8b_sharegpt/checkpoints/checkpoint_best"
PORT=8013
export CUDA_VISIBLE_DEVICES=4
# Optional: pin GPU
# export CUDA_VISIBLE_DEVICES=0

if [[ ! -e "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

echo "Serving DFlash checkpoint: ${CHECKPOINT}"
echo "Target model (from speculators_config): Qwen/Qwen3-8B"
echo "aux_hidden_state_layer_ids: 1 9 17 25 33 (already in checkpoint config.json)"
echo "KV mode (from checkpoint dflash_config): copy — uses target K/V after k-norm+RoPE"
echo "Port: ${PORT}"

# DFlash uses parallel drafting with 15 speculative tokens. On H100-class GPUs vLLM
# defaults to max_num_batched_tokens=8192 and max_num_seqs=1024, which leaves
# max_num_scheduled_tokens negative (8192 - 14*1024 = -6144). Raise batch limits.
vllm serve "${CHECKPOINT}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --served-model-name "Qwen/Qwen3-8B" \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256 \
  --enforce-eager