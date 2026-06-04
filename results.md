# Per-cell scenario data

Appendix to [README.md](README.md). All numbers below are from
`transformers 5.10.1` (release tag) on `torch 2.7.0+cu126`, single A10G.

## Scenario sweep

Produced by [bench_scenarios.py](bench_scenarios.py). Each row is one
(mode, scenario) cell; the orchestrator emits these as JSONL from each
mode subprocess and writes the final table to
[`logs/scenarios.tsv`](logs/scenarios.tsv).

`vanilla_patched` runs stock 5.10.1 with a runtime monkey patch on
`GenerationMixin._prepare_static_cache` that calls
`cache.early_initialization(...)` on the freshly-allocated cache before
returning it — the proposed upstream fix for the lazy-init recompile
(see issue draft). `diy` and `static_tensors` apply the same fix
manually on a DIY-constructed cache. Plain `vanilla` does not.

| mode            | scenario        | total_s | tps    | +artifacts | color |
|-----------------|-----------------|--------:|-------:|-----------:|-------|
| vanilla         | cold            |   29.65 |   4.32 |         56 |       |
| vanilla         | warm            |   14.93 |   8.57 |         18 | 🔴    |
| vanilla         | warm-diff-mnt   |   27.80 |   9.21 |         21 | 🔴    |
| vanilla         | warm-diff-in    |   41.78 |   3.06 |         30 | 🔴    |
| vanilla_patched | cold            |   29.14 |   4.39 |         54 |       |
| vanilla_patched | warm            |    1.15 | 111.16 |          0 | 🟢    |
| vanilla_patched | warm-diff-mnt   |   27.51 |   9.31 |         19 | 🔴    |
| vanilla_patched | warm-diff-in    |   27.09 |   4.73 |         19 | 🔴    |
| diy             | cold            |   29.09 |   4.40 |         54 |       |
| diy             | warm            |    1.35 |  94.66 |          0 | 🟢    |
| diy             | warm-diff-mnt   |    2.66 |  96.31 |          0 | 🟢    |
| diy             | warm-diff-in    |   14.64 |   8.74 |          0 | 🟢    |
| static_tensors  | cold            |   26.68 |   4.80 |         54 |       |
| static_tensors  | warm            |    1.22 | 105.04 |          0 | 🟢    |
| static_tensors  | warm-diff-mnt   |    2.39 | 107.20 |          0 | 🟢    |
| static_tensors  | warm-diff-in    |   13.41 |   9.55 |          0 | 🟢    |

`vanilla_patched` shows the upstream fix is scoped exactly to `warm`:
the lazy-init recompile is gone (1.15 s / +0). `warm-diff-mnt` and
`warm-diff-in` are still red because they trigger a separate issue —
the auto-allocated `StaticCache` reallocates whenever `max_cache_len`
grows. The DIY/static_tensors recipe sidesteps that by pre-sizing the
cache to the worst case.

`+artifacts` counts new Inductor artifact files (`.cubin` / `.so` /
`.kernel.json`) added during the cell. A non-zero count means a real
Inductor recompile happened; zero means the cache absorbed the delta.
Color is on this column alone (🟢 = +0, 🔴 = +>0). `cold` is left
uncolored — it's the baseline first-time compile and would always be
🔴 by definition.

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
