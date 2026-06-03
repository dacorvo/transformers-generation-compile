# Per-cell scenario data

Appendix to [README.md](README.md). All numbers below are from
`transformers 5.10.0.dev0` (upstream `main` at commit `595721c`,
2026-06-03) on `torch 2.7.0+cu126`, single A10G.

## Scenario sweep

Produced by [bench_scenarios.py](bench_scenarios.py). Each row is one
(mode, scenario) cell; the orchestrator emits these as JSONL from each
mode subprocess and writes the final table to
[`logs/scenarios.tsv`](logs/scenarios.tsv).

Both `diy` and `static_tensors` modes call
`cache.early_initialization(...)` once before the first generate(),
which keeps `warm` clean (see README takeaway #4). Vanilla can't apply
this fix.

| mode           | scenario        | total_s | tps    | +artifacts | color |
|----------------|-----------------|--------:|-------:|-----------:|-------|
| vanilla        | cold            |   29.66 |   4.32 |         56 |       |
| vanilla        | warm            |   14.90 |   8.59 |         18 |       |
| vanilla        | warm-diff-mnt   |   27.55 |   9.29 |         21 | 🔴    |
| vanilla        | warm-diff-in    |   40.75 |   3.14 |         30 | 🔴    |
| diy            | cold            |   28.86 |   4.44 |         54 |       |
| diy            | warm            |    1.35 |  94.99 |          0 |       |
| diy            | warm-diff-mnt   |    2.65 |  96.59 |          0 | 🟢    |
| diy            | warm-diff-in    |   14.70 |   8.71 |          0 | 🟢    |
| static_tensors | cold            |   26.60 |   4.81 |         54 |       |
| static_tensors | warm            |    1.22 | 105.07 |          0 |       |
| static_tensors | warm-diff-mnt   |    2.39 | 107.25 |          0 | 🟢    |
| static_tensors | warm-diff-in    |   13.60 |   9.41 |          0 | 🟢    |

`+artifacts` counts new Inductor artifact files (`.cubin` / `.so` /
`.kernel.json`) added during the cell. A non-zero count means a real
Inductor recompile happened; zero means the cache absorbed the delta.
Color is on this column alone (🟢 = +0, 🔴 = +>0).

Note on `diy / warm-diff-in` and `static_tensors / warm-diff-in`:
+0 artifacts but ~14 s wallclock. The cache absorbed the delta (no
Inductor recompile), but Dynamo likely re-traced for the new outer
`input_ids` shape (1, 2048 → 2 chunks of 1024) and paid tracing time
without producing new kernels. See README takeaway #5.

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
`diy → static_tensors` win in the scenario table (95 → 105 tok/s on
the warm cell, 96 → 107 tok/s on warm-diff-mnt). The relative gain
shrinks as the cache and prompt grow because the compiled forward
starts dominating wallclock.

## Raw output schema

`logs/scenarios.tsv` columns:

```
mode    scenario    total_s    tps    artifacts_delta    color
```

`tps = max_new_tokens / total_s` — wallclock-effective tokens per
second for the cell. For the `diy` and `static_tensors` modes, the
`warm` cell is now a true steady-state measurement (~95 / 105 tps).
For `vanilla`, `warm` still carries the lazy compile (~13 s), so
its tps is meaningless — `warm-diff-mnt` won't reveal vanilla's
clean steady-state either, because that cell adds its own recompile.
