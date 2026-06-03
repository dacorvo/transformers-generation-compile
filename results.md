# Per-cell scenario data

Appendix to [README.md](README.md). All numbers below are from
`transformers 5.10.0.dev0` (upstream `main` at commit `595721c`,
2026-06-03) on `torch 2.7.0+cu126`, single A10G.

## Scenario sweep

Produced by [bench_scenarios.py](bench_scenarios.py). Each row is one
(mode, scenario) cell; the orchestrator emits these as JSONL from each
mode subprocess and writes the final table to
[`logs/scenarios.tsv`](logs/scenarios.tsv).

| mode           | scenario        | total_s | tps    | +artifacts | color |
|----------------|-----------------|--------:|-------:|-----------:|-------|
| vanilla        | cold            |   28.99 |   4.41 |         56 |       |
| vanilla        | warm            |   14.58 |   8.78 |         18 |       |
| vanilla        | warm-diff-mnt   |   26.87 |   9.53 |         21 | 🔴    |
| vanilla        | warm-diff-in    |   39.76 |   3.22 |         30 | 🔴    |
| diy            | cold            |   29.17 |   4.39 |         56 |       |
| diy            | warm            |   15.05 |   8.51 |         18 |       |
| diy            | warm-diff-mnt   |    2.66 |  96.23 |          0 | 🟢    |
| diy            | warm-diff-in    |   14.57 |   8.78 |          0 | 🟢    |
| static_tensors | cold            |   26.91 |   4.76 |         56 |       |
| static_tensors | warm            |   14.00 |   9.14 |         20 |       |
| static_tensors | warm-diff-mnt   |    2.39 | 107.21 |          0 | 🟢    |
| static_tensors | warm-diff-in    |   13.48 |   9.49 |          0 | 🟢    |

`+artifacts` counts new Inductor artifact files (`.cubin` / `.so` /
`.kernel.json`) added during the cell. A non-zero count in a warm-diff
row means a real recompile happened.

Color is computed against the same mode's `warm` total_s, with
CUDA-tuned thresholds: 🟢 ≤ 1.2×, 🟡 1.2–1.5×, 🔴 > 1.5×. Looser
Neuron-style thresholds (≤1.5 / 1.5–10 / >10) would call the vanilla
deltas "partial reuse"; on CUDA they're real recompiles that add
12–25 s of latency per first occurrence.

## Decode-loop overhead microbenchmark

Llama-3.2-1B, prompt=256, decode=256, cache=544, mode=default,
N_ITER=5 (warm calls only). From
[evidence/decode_overhead.py](evidence/decode_overhead.py).

| measurement | per step | tok/s |
|---|--:|--:|
| `model.generate()` (subtracting ~52 ms prefill) | 7.85 ms | 127.4 |
| compiled-forward direct call, all tensors pre-allocated | 6.71 ms | 149.0 |
| gap (Python plumbing + `torch.cat` + mask rebuild) | **1.13 ms (~14 %)** | – |

This is the smaller-cache, smaller-prompt version of the
`diy → static_tensors` win in the scenario table (96 → 107 tok/s on
warm-diff-mnt). The relative gain shrinks as the cache and prompt grow
because the compiled forward starts dominating wallclock.

## Raw output schema

`logs/scenarios.tsv` columns:

```
mode    scenario    total_s    tps    artifacts_delta    color
```

`tps = max_new_tokens / total_s` — wallclock-effective tokens per
second for the cell, *including* compile and any lazy-late-compile
overhead. It is not a steady-state decode throughput; for that, look
at the `warm-diff-mnt` row of the diy/static_tensors modes (large
`max_new`, no recompile, dominated by actual decode).
