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
| vanilla        | cold            |   29.58 |   4.33 |         56 |       |
| vanilla        | warm            |   14.89 |   8.59 |         18 |       |
| vanilla        | warm-diff-mnt   |   27.46 |   9.32 |         21 | 🟡    |
| vanilla        | warm-diff-in    |   40.50 |   3.16 |         30 | 🟡    |
| diy            | cold            |   29.31 |   4.37 |         56 |       |
| diy            | warm            |   15.22 |   8.41 |         18 |       |
| diy            | warm-diff-mnt   |    2.67 |  96.00 |          0 | 🟢    |
| diy            | warm-diff-in    |   14.84 |   8.62 |          0 | 🟢    |
| static_tensors | cold            |   26.66 |   4.80 |         56 |       |
| static_tensors | warm            |   13.64 |   9.38 |         20 |       |
| static_tensors | warm-diff-mnt   |    2.39 | 107.25 |          0 | 🟢    |
| static_tensors | warm-diff-in    |   13.20 |   9.70 |          0 | 🟢    |

`+artifacts` counts new Inductor artifact files (`.cubin` / `.so` /
`.kernel.json`) added during the cell. A non-zero count in a warm-diff
row means a real recompile happened.

Color is computed against the same mode's `warm` total_s:
🟢 ≤ 1.5×, 🟡 1.5–10×, 🔴 > 10×.

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
