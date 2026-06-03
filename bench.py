"""Warmup + steady-state sweep for one (model, mode) configuration.

WHAT: phase A runs one full generate() per (input_len × chunk_size)
cell in largest-bucket-first order. Phase B measures TTFT × N (via a
StoppingCriteria truncating to 1 new token) plus one full-decode
call per cell. Snapshots torch._dynamo.utils.counters and the
Inductor cache directory around each cell to detect silent recompiles.
JSON summary emitted to stdout.

WHY:  the harness pins two things across every generate() call —
`max_new_tokens` (the decode budget) and the input_len bucket
warming order (biggest first) — so the StaticCache buffer never
grows mid-run. That keeps shapes stable, so any recompile we see
during steady state is a real surprise, not a setup artifact.

RESULT: per-cell warmup/TTFT/decode/recompile numbers. See
logs/<model>-<mode>.json + results.md.

RUN: see README.md "Reproduce".
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

# These envs must be set *before* importing torch so they take effect.
def _setup_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Want to measure actual compile cost, not cache hits from prior runs.
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")


def _count_cache_artifacts(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix in {".cubin", ".so"} or p.name.endswith(".kernel.json")):
            n += 1
    return n


def _snapshot_counters() -> dict:
    import torch
    counters = torch._dynamo.utils.counters
    return {k: dict(v) for k, v in counters.items()}


def _diff_counters(before: dict, after: dict) -> dict:
    diff = {}
    keys = set(before) | set(after)
    for k in keys:
        b = before.get(k, {})
        a = after.get(k, {})
        sub = {}
        for kk in set(a) | set(b):
            if a.get(kk, 0) != b.get(kk, 0):
                sub[kk] = a.get(kk, 0) - b.get(kk, 0)
        if sub:
            diff[k] = sub
    return diff


def _build_inputs(tokenizer, bucket_len: int, batch_size: int, device, seed: int):
    """Random ids of exact shape (B, bucket_len). Padded with BOS at position 0."""
    rng = random.Random(seed)
    vocab_lo, vocab_hi = 1000, min(tokenizer.vocab_size or 32000, 30000)
    import torch
    ids = torch.tensor(
        [[rng.randint(vocab_lo, vocab_hi) for _ in range(bucket_len)] for _ in range(batch_size)],
        dtype=torch.long,
        device=device,
    )
    bos = tokenizer.bos_token_id
    if bos is not None:
        ids[:, 0] = bos
    mask = torch.ones_like(ids)
    return ids, mask


class _StopAfterNNewTokens:
    """A StoppingCriteria that fires after `n` new tokens, but pretends to be a
    `transformers.StoppingCriteria` instance via duck typing."""

    def __init__(self, prompt_len: int, n: int):
        self.prompt_len = prompt_len
        self.n = n

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        new = input_ids.shape[1] - self.prompt_len
        # Need to return a tensor of bool per-batch row (since transformers v5).
        done = new >= self.n
        return torch.full(
            (input_ids.shape[0],),
            done,
            dtype=torch.bool,
            device=input_ids.device,
        )


def _time_generate(model, ids, mask, gen_cfg, stopping_criteria=None):
    import torch
    from transformers import StoppingCriteriaList
    sc_list = StoppingCriteriaList([stopping_criteria]) if stopping_criteria is not None else None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        generation_config=gen_cfg,
        stopping_criteria=sc_list,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_new = int(out.shape[1] - ids.shape[1])
    del out
    return dt, n_new


def run(args):
    cache_root = Path(args.cache_root) / f"{args.run_tag}"
    _setup_env(cache_root)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print(f"[load] {args.model_id} dtype=bf16 on {device}", file=sys.stderr)
    t_load = time.perf_counter()
    load_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    model = None
    last_err = None
    for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            from transformers import AutoModelForImageTextToText  # noqa: F401
        except ImportError:
            if loader_name == "AutoModelForImageTextToText":
                continue
        try:
            cls = {"AutoModelForCausalLM": AutoModelForCausalLM}
            try:
                from transformers import AutoModelForImageTextToText
                cls["AutoModelForImageTextToText"] = AutoModelForImageTextToText
            except ImportError:
                pass
            model = cls[loader_name].from_pretrained(args.model_id, **load_kwargs).to(device)
            print(f"[load] used {loader_name} -> {type(model).__name__}", file=sys.stderr)
            break
        except Exception as e:
            last_err = e
            print(f"[load] {loader_name} failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    if model is None:
        raise RuntimeError(f"No loader worked. Last error: {last_err!r}")
    model.eval()
    load_s = time.perf_counter() - t_load
    print(f"[load] done in {load_s:.1f}s", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Order cells so the LARGEST input_len is warmed first → static cache
    # is allocated at its final, biggest size and reused thereafter.
    cells = []
    for input_len in sorted(set(args.input_lens), reverse=True):
        for chunk_size in args.chunk_sizes:
            cells.append((int(input_len), int(chunk_size)))

    # Single decode budget for the whole experiment. Every generate() call
    # asks for this many new tokens; stopping criteria truncates earlier
    # when we just want TTFT. Cache size therefore stays constant.
    decode_budget = args.decode_sanity_tokens

    def make_gen_cfg() -> GenerationConfig:
        return GenerationConfig(
            do_sample=False,
            num_beams=1,
            cache_implementation="static",
            compile_config=CompileConfig(mode=args.mode, fullgraph=False, dynamic=False),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=decode_budget,
            min_new_tokens=decode_budget,  # actual stopping is via stopping_criteria
        )

    # --- Phase A: warmup. One full generate per cell (largest bucket first). ---
    cells_warm: list[dict] = []
    for cell_idx, (input_len, chunk_size) in enumerate(cells):
        ids, mask = _build_inputs(tokenizer, input_len, args.batch_size, device, seed=1000 + cell_idx)
        gen_cfg = make_gen_cfg()
        gen_cfg.prefill_chunk_size = chunk_size

        artifacts_before = _count_cache_artifacts(cache_root)
        counters_before = _snapshot_counters()
        # Force full decode budget to compile decode kernel too.
        dt, n_new = _time_generate(model, ids, mask, gen_cfg)
        artifacts_after = _count_cache_artifacts(cache_root)
        counters_after = _snapshot_counters()

        cells_warm.append(
            {
                "input_len": input_len,
                "chunk_size": chunk_size,
                "warmup_total_s": dt,
                "warmup_n_new": n_new,
                "warmup_artifacts_delta": artifacts_after - artifacts_before,
                "warmup_counters_delta": _diff_counters(counters_before, counters_after),
            }
        )
        print(
            f"[warmup] ({input_len:>5},{chunk_size:>4})  total={dt:6.2f}s  "
            f"artifacts+={artifacts_after - artifacts_before:>4}  new_toks={n_new}",
            file=sys.stderr,
        )

    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase B: steady state. ---
    # For each cell: 10 prefill-only calls (TTFT, via StoppingCriteria stopping
    # after 1 new token) + 1 full-decode call for tok/s sanity. We separately
    # snapshot Inductor cache / Dynamo counters around each cell to detect
    # silent recompiles.
    results: list[dict] = []
    # Restore natural cell order for reporting.
    cells_report_order = sorted(cells)
    for cell_idx, (input_len, chunk_size) in enumerate(cells_report_order):
        gen_cfg = make_gen_cfg()
        gen_cfg.prefill_chunk_size = chunk_size

        ttfts: list[float] = []
        artifacts_before = _count_cache_artifacts(cache_root)
        counters_before = _snapshot_counters()
        for i in range(args.steady_calls):
            ids, mask = _build_inputs(
                tokenizer, input_len, args.batch_size, device, seed=2000 + cell_idx * 100 + i
            )
            sc = _StopAfterNNewTokens(prompt_len=ids.shape[1], n=1)
            dt, _ = _time_generate(model, ids, mask, gen_cfg, stopping_criteria=sc)
            ttfts.append(dt)
        artifacts_after = _count_cache_artifacts(cache_root)
        counters_after = _snapshot_counters()
        ttft_cdiff = _diff_counters(counters_before, counters_after)

        # Decode-throughput sanity: one full max_new_tokens call.
        ids, mask = _build_inputs(tokenizer, input_len, args.batch_size, device, seed=9000 + cell_idx)
        artifacts_before_dec = _count_cache_artifacts(cache_root)
        counters_before_dec = _snapshot_counters()
        dec_dt, dec_new = _time_generate(model, ids, mask, gen_cfg)
        artifacts_after_dec = _count_cache_artifacts(cache_root)
        counters_after_dec = _snapshot_counters()

        ttfts_sorted = sorted(ttfts)
        n = len(ttfts_sorted)
        p50 = ttfts_sorted[n // 2]
        p99 = ttfts_sorted[min(n - 1, int(round(0.99 * (n - 1))))]
        mean_ttft = sum(ttfts) / n
        decode_only_s = dec_dt - p50  # subtract median prefill
        decode_tok_per_s = (dec_new - 1) / decode_only_s if dec_new > 1 and decode_only_s > 0 else None

        cell_result = {
            "input_len": input_len,
            "chunk_size": chunk_size,
            "mode": args.mode,
            "model_id": args.model_id,
            "ttft_s_calls": ttfts,
            "ttft_s_median": p50,
            "ttft_s_p99": p99,
            "ttft_s_mean": mean_ttft,
            "ttft_p99_over_p50": p99 / p50 if p50 > 0 else None,
            "decode_total_s": dec_dt,
            "decode_n_new": dec_new,
            "decode_only_s": decode_only_s,
            "decode_tok_per_s": decode_tok_per_s,
            "ss_artifacts_delta": artifacts_after - artifacts_before,
            "ss_counters_delta": ttft_cdiff,
            "dec_artifacts_delta": artifacts_after_dec - artifacts_before_dec,
            "dec_counters_delta": _diff_counters(counters_before_dec, counters_after_dec),
        }
        for w in cells_warm:
            if w["input_len"] == input_len and w["chunk_size"] == chunk_size:
                cell_result["warmup_total_s"] = w["warmup_total_s"]
                cell_result["warmup_artifacts_delta"] = w["warmup_artifacts_delta"]
                cell_result["warmup_counters_delta"] = w["warmup_counters_delta"]
                break
        results.append(cell_result)
        dec_str = f"{decode_tok_per_s:.1f}" if decode_tok_per_s else "n/a"
        print(
            f"[steady] ({input_len:>5},{chunk_size:>4})  p50={p50*1000:7.1f}ms  "
            f"p99={p99*1000:7.1f}ms  ss_artifacts+={cell_result['ss_artifacts_delta']:>4}  "
            f"dec_artifacts+={cell_result['dec_artifacts_delta']:>4}  dec_tok/s={dec_str}",
            file=sys.stderr,
        )

    summary = {
        "model_id": args.model_id,
        "mode": args.mode,
        "load_s": load_s,
        "warmup_order": [list(c) for c in cells],
        "warmup_total_s": sum(w["warmup_total_s"] for w in cells_warm),
        "decode_budget": decode_budget,
        "batch_size": args.batch_size,
        "steady_calls": args.steady_calls,
        "cells": results,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument(
        "--mode",
        required=True,
        choices=["default", "max-autotune-no-cudagraphs", "reduce-overhead"],
    )
    p.add_argument("--input-lens", type=int, nargs="+", default=[1024, 8192])
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--steady-calls", type=int, default=10)
    p.add_argument("--decode-sanity-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-root", default="/tmp/inductor-bench")
    p.add_argument("--run-tag", default="run")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
