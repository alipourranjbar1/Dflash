"""
DFlash benchmark CLIENT for a *running* vLLM server (V1).

This script talks to the OpenAI-compatible /v1/completions endpoint for
generation, and reads speculative decoding counters from the Prometheus
/metrics endpoint with a before/after delta.

Supported datasets:
  - gsm8k
  - math500
  - humaneval
  - mbpp
  - mt-bench
  - all

For mt-bench, this script follows the official benchmark.py format:
each sample contains a list of user turns. The script sends the turns
sequentially, appends the generated assistant answer to the chat history,
and then sends the next turn.

Example:
  python test_bench_all.py \
    --base-url http://localhost:8013  \
    --model "Qwen/Qwen3-8B" \
    --dataset all \
    --num-prompts 100 \
    --concurrency 8 \
    --warmup-steps 10 \
    --max-new-tokens 1024
"""

import argparse
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            f"{x['question']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            f"{x['problem']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{x['prompt']}\n```"
        ),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------


def load_data(name, n):
    """
    Returns a list of samples.

    Each sample has:
      {
        "turns": [user_turn_1, user_turn_2, ...]
      }

    For normal single-turn datasets, turns has length 1.
    For mt-bench, turns can have length > 1.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[name]
    ds = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])

    items = []

    for row in ds:
        if cfg.get("multi_turn"):
            turns = cfg["format"](row)

            if not isinstance(turns, list):
                turns = [str(turns)]

            items.append({"turns": [str(t) for t in turns]})
        else:
            items.append({"turns": [str(cfg["format"](row))]})

    random.seed(42)
    random.shuffle(items)

    return items if n is None else items[:n]


# ---------------------------------------------------------------------
# Spec-decode counters from Prometheus /metrics
# ---------------------------------------------------------------------

_NUM_RE = re.compile(r"\}?\s+([0-9eE.+-]+)\s*$")


def _sum_counter(text, base_name):
    """Sum all label-series for a Prometheus counter."""
    total = 0.0
    found = False

    for line in text.splitlines():
        if line.startswith("#"):
            continue

        if line.startswith(base_name + "{") or line.split("{")[0] == base_name:
            m = _NUM_RE.search(line)

            if m:
                total += float(m.group(1))
                found = True

    return total if found else None


def _per_pos(text, base_name):
    """Return {position_index: count} for a per-position counter."""
    out = {}

    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(base_name):
            continue

        pos_m = re.search(r'position="(\d+)"', line)
        val_m = _NUM_RE.search(line)

        if pos_m and val_m:
            i = int(pos_m.group(1))
            out[i] = out.get(i, 0.0) + float(val_m.group(1))

    return out


def get_spec_metrics(base_url):
    """
    Reads speculative decoding metrics from /metrics.

    Expected counters:
      - vllm:spec_decode_num_drafts_total
      - vllm:spec_decode_num_draft_tokens_total
      - vllm:spec_decode_num_accepted_tokens_total
      - vllm:spec_decode_num_accepted_tokens_per_pos_total
    """
    try:
        r = requests.get(f"{base_url}/metrics", timeout=10)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        print(f"WARN: could not fetch /metrics: {e}")
        return None

    return {
        "num_drafts": _sum_counter(text, "vllm:spec_decode_num_drafts_total"),
        "num_draft_tokens": _sum_counter(text, "vllm:spec_decode_num_draft_tokens_total"),
        "num_accepted_tokens": _sum_counter(text, "vllm:spec_decode_num_accepted_tokens_total"),
        "per_pos": _per_pos(text, "vllm:spec_decode_num_accepted_tokens_per_pos_total"),
    }


# ---------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------


def send_one(
    base_url,
    model,
    prompt,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    timeout,
):
    """
    Sends one streaming /v1/completions request.

    Captures:
      - generated text
      - end-to-end latency
      - TTFT
      - ITL
      - output token count from final usage chunk
    """
    import json as _json

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k

    start_time = time.perf_counter()
    ttft = 0.0
    most_recent_ts = start_time
    itls = []
    output_len = 0
    text_out = []

    with requests.post(
        f"{base_url}/v1/completions",
        json=payload,
        timeout=timeout,
        stream=True,
    ) as resp:
        resp.raise_for_status()

        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue

            line = raw.strip()

            if line.startswith("data:"):
                line = line[5:].lstrip()

            if line == "[DONE]":
                continue

            try:
                data = _json.loads(line)
            except Exception:
                continue

            ts = time.perf_counter()
            choices = data.get("choices") or []

            if choices and choices[0].get("text"):
                text_out.append(choices[0]["text"])

                if ttft == 0.0:
                    ttft = ts - start_time
                else:
                    itls.append((ts - most_recent_ts) * 1000.0)

                most_recent_ts = ts

            if data.get("usage"):
                output_len = int(data["usage"].get("completion_tokens", 0))

    e2e = time.perf_counter() - start_time

    return {
        "text": "".join(text_out),
        "e2e_latency": e2e,
        "ttft_ms": ttft * 1000.0,
        "itls": itls,
        "output_len": output_len,
    }


def apply_chat_template_safe(tokenizer, messages, enable_thinking):
    """
    Applies Qwen-style chat template.

    Some tokenizers support enable_thinking, some do not. This helper first
    tries with enable_thinking, then falls back without it.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def send_sample(
    base_url,
    model,
    turns,
    tokenizer,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    enable_thinking,
    timeout,
):
    """
    Sends one benchmark sample.

    For single-turn datasets:
      turns = [prompt]

    For mt-bench:
      turns = [turn_1, turn_2, ...]

    For multi-turn samples, the generated assistant answer from turn N is
    appended to the message history before turn N+1.
    """
    messages = []

    total_output_len = 0
    total_e2e_latency = 0.0
    all_itls = []
    all_ttfts = []
    outputs = []

    for user_content in turns:
        messages.append({"role": "user", "content": user_content})

        prompt = apply_chat_template_safe(
            tokenizer=tokenizer,
            messages=messages,
            enable_thinking=enable_thinking,
        )

        out = send_one(
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            timeout=timeout,
        )

        assistant_text = out["text"]

        messages.append({"role": "assistant", "content": assistant_text})
        outputs.append(assistant_text)

        total_output_len += out["output_len"]
        total_e2e_latency += out["e2e_latency"]
        all_itls.extend(out["itls"])

        if out["ttft_ms"] > 0:
            all_ttfts.append(out["ttft_ms"])

    return {
        "texts": outputs,
        "e2e_latency": total_e2e_latency,
        "ttfts": all_ttfts,
        "itls": all_itls,
        "output_len": total_output_len,
        "num_turns": len(turns),
    }


# ---------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------


def run_one_dataset(args, dataset_name, tokenizer):
    warmup_steps = max(args.warmup_steps, 0)

    if args.num_prompts is None:
        print(
            f"\nLoading FULL {dataset_name} dataset "
            f"(plus {warmup_steps} warmup, if enough samples exist)..."
        )
        samples = load_data(dataset_name, None)
    else:
        print(
            f"\nLoading {args.num_prompts} samples from {dataset_name} "
            f"(plus {warmup_steps} warmup)..."
        )
        samples = load_data(dataset_name, args.num_prompts + warmup_steps)

    print(f"  Loaded samples:   {len(samples)}")
    print(f"  Loaded turns:     {sum(len(x['turns']) for x in samples)}")

    # Flush prefix cache before each dataset so datasets do not help each other.
    try:
        requests.post(f"{args.base_url}/reset_prefix_cache", timeout=30)
    except Exception:
        pass

    # -------------------------
    # Warmup
    # -------------------------
    if warmup_steps > 0:
        actual_warmup = min(warmup_steps, len(samples))

        if actual_warmup > 0:
            warmup_workers = min(max(args.concurrency, 1), actual_warmup)

            print(f"Warmup ({actual_warmup} samples)...")

            with ThreadPoolExecutor(max_workers=warmup_workers) as pool:
                list(
                    pool.map(
                        lambda s: send_sample(
                            base_url=args.base_url,
                            model=args.model,
                            turns=s["turns"],
                            tokenizer=tokenizer,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            enable_thinking=args.enable_thinking,
                            timeout=args.timeout_s,
                        ),
                        samples[:actual_warmup],
                    )
                )

            samples = samples[actual_warmup:]

    # Snapshot acceptance counters AFTER warmup.
    m_before = get_spec_metrics(args.base_url)

    if m_before:
        print(
            f"Spec metrics BEFORE: drafts={m_before['num_drafts']} "
            f"accepted={m_before['num_accepted_tokens']}"
        )

    total_expected_turns = sum(len(x["turns"]) for x in samples)

    print(
        f"Benchmarking {len(samples)} samples, "
        f"{total_expected_turns} total turns, "
        f"concurrency={args.concurrency}..."
    )

    start = time.perf_counter()

    total_tokens = 0
    total_turns = 0
    e2e_latencies = []
    all_itls = []
    all_ttfts = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                send_sample,
                args.base_url,
                args.model,
                sample["turns"],
                tokenizer,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                args.enable_thinking,
                args.timeout_s,
            ): i
            for i, sample in enumerate(samples)
        }

        for fut in tqdm(as_completed(futures), total=len(samples), desc=dataset_name):
            out = fut.result()

            total_tokens += out["output_len"]
            total_turns += out["num_turns"]
            e2e_latencies.append(out["e2e_latency"])
            all_itls.extend(out["itls"])
            all_ttfts.extend(out["ttfts"])

    elapsed = time.perf_counter() - start

    m_after = get_spec_metrics(args.base_url)

    # -------------------------
    # Acceptance length
    # -------------------------
    accept_length = float("nan")
    accept_rate = float("nan")

    d_drafts = 0.0
    d_acc = 0.0
    d_draft_tok = 0.0

    if m_before and m_after and m_after["num_drafts"] is not None:
        d_drafts = m_after["num_drafts"] - m_before["num_drafts"]
        d_acc = m_after["num_accepted_tokens"] - m_before["num_accepted_tokens"]

        if m_after["num_draft_tokens"] is not None:
            d_draft_tok = (
                m_after["num_draft_tokens"]
                - m_before["num_draft_tokens"]
            )

        if d_drafts > 0:
            # Same convention as your original script:
            # accepted bonus token + accepted draft tokens per draft.
            accept_length = 1.0 + d_acc / d_drafts

        if d_draft_tok > 0:
            accept_rate = d_acc / d_draft_tok

    throughput = total_tokens / elapsed if elapsed > 0 else float("nan")

    # -------------------------
    # Per-dataset result
    # -------------------------
    print()
    print("=" * 70)
    print(f"Dataset:                    {dataset_name}")
    print(f"Samples:                    {len(samples)}")
    print(f"Turns:                      {total_turns}")
    print(f"Concurrency:                {args.concurrency}")
    print(f"Warmup steps:               {warmup_steps}")
    print(f"Wall clock:                 {elapsed:.2f} s")
    print(f"Total output tokens:        {total_tokens}")
    print()
    print(f"** Throughput:              {throughput:.2f} tok/s **")
    print(f"** Accept length:           {accept_length:.3f} **")
    print(f"   Accept rate:             {100 * accept_rate:.2f}%")
    print(f"   num_drafts:              {int(d_drafts)}")
    print(f"   num_draft_tokens:        {int(d_draft_tok)}")
    print(f"   num_accepted_tokens:     {int(d_acc)}")

    if e2e_latencies:
        print(f"   E2E latency mean/sample: {statistics.mean(e2e_latencies):.2f} s")

    if all_ttfts:
        print(f"   Mean TTFT:               {statistics.mean(all_ttfts):.2f} ms")

    if all_itls:
        print(f"   Mean ITL:                {statistics.mean(all_itls):.2f} ms")
        print(f"   Median ITL:              {statistics.median(all_itls):.2f} ms")

    if (
        m_before
        and m_after
        and m_after.get("per_pos")
        and m_before.get("per_pos")
        and d_drafts > 0
    ):
        print("   Per-position accept rate:")

        for i in sorted(m_after["per_pos"]):
            c = m_after["per_pos"].get(i, 0) - m_before["per_pos"].get(i, 0)
            print(f"     pos {i}: {100 * c / d_drafts:.2f}%")

    if m_after and m_after["num_drafts"] in (None, 0):
        print()
        print("NOTE: no spec-decode counters from /metrics.")
        print("      Server may not be on the V1 engine, or this vllm-ascend")
        print("      build may expose speculative metrics differently.")

    print("=" * 70)

    return {
        "dataset": dataset_name,
        "samples": len(samples),
        "turns": total_turns,
        "tokens": total_tokens,
        "elapsed": elapsed,
        "throughput": throughput,
        "accept_length": accept_length,
        "accept_rate": accept_rate,
        "num_drafts": d_drafts,
        "num_draft_tokens": d_draft_tok,
        "num_accepted_tokens": d_acc,
    }


def print_final_summary(results):
    print()
    print("=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)
    print(
        f"{'Dataset':<12} "
        f"{'Samples':>10} "
        f"{'Turns':>10} "
        f"{'Tokens':>12} "
        f"{'Time(s)':>12} "
        f"{'Throughput(tok/s)':>20} "
        f"{'Accept Len':>14} "
        f"{'Accept Rate':>14}"
    )
    print("-" * 120)

    for r in results:
        accept_rate_pct = 100 * r["accept_rate"]

        print(
            f"{r['dataset']:<12} "
            f"{r['samples']:>10} "
            f"{r['turns']:>10} "
            f"{r['tokens']:>12} "
            f"{r['elapsed']:>12.2f} "
            f"{r['throughput']:>20.2f} "
            f"{r['accept_length']:>14.3f} "
            f"{accept_rate_pct:>13.2f}%"
        )

    print("=" * 120)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument(
        "--model",
        required=True,
        help="Path/name as passed to `vllm serve`.",
    )

    ap.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        required=True,
        help="Dataset name, or 'all' to run all datasets.",
    )

    ap.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help=(
            "Number of benchmark samples per dataset. "
            "If omitted, benchmarks the full dataset."
        ),
    )
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup-steps", type=int, default=10)

    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=3600)

    args = ap.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.dataset == "all":
        dataset_names = list(DATASETS.keys())
    else:
        dataset_names = [args.dataset]

    all_results = []

    for dataset_name in dataset_names:
        result = run_one_dataset(args, dataset_name, tokenizer)
        all_results.append(result)

    print_final_summary(all_results)


if __name__ == "__main__":
    main()