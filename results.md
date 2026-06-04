# Per-cell scenario data

Appendix to [README.md](README.md). All numbers below are from
`transformers 5.10.1` (release tag) on `torch 2.7.0+cu126`, single A10G.

## Scenario sweep

Produced by [bench_scenarios.py](bench_scenarios.py). Each row is one
(mode, scenario) cell; the orchestrator emits these as JSONL from each
mode subprocess and writes the final table to
[`logs/scenarios.tsv`](logs/scenarios.tsv).

The four `vanilla*` rows are a 2×2 ablation over two proposed upstream
fixes, each applied as a runtime monkey patch on stock 5.10.1
(`_apply_monkey_patches` in [bench_scenarios.py](bench_scenarios.py)):

- **p1** — auto-call `cache.early_initialization(...)` from
  `_prepare_static_cache`. Fixes the lazy-init recompile that the
  chunked-prefill loop trips when `update`'s `is_initialized` branch
  mutates inside a compiled region. See [ISSUE_DRAFT.md](ISSUE_DRAFT.md).
- **p2** — honor `generation_config.cache_config["max_cache_len"]` on
  the static path of `_prepare_cache_for_generation`. The
  `GenerationConfig` docstring claims it's used but the static branch
  silently ignores it; verified in
  [evidence/cache_config_max_cache_len.py](evidence/cache_config_max_cache_len.py).

`diy` and `static_tensors` apply both fixes manually on a user-constructed
`StaticCache`.

| mode           | scenario        | total_s | tps    | +artifacts | color |
|----------------|-----------------|--------:|-------:|-----------:|-------|
| vanilla        | cold            |   29.55 |   4.33 |         56 |       |
| vanilla        | warm            |   14.96 |   8.55 |         18 | 🔴    |
| vanilla        | warm-diff-mnt   |   28.34 |   9.03 |         21 | 🔴    |
| vanilla        | warm-diff-in    |   41.16 |   3.11 |         30 | 🔴    |
| vanilla_p1     | cold            |   28.71 |   4.46 |         54 |       |
| vanilla_p1     | warm            |    1.15 | 111.18 |          0 | 🟢    |
| vanilla_p1     | warm-diff-mnt   |   26.98 |   9.49 |         19 | 🔴    |
| vanilla_p1     | warm-diff-in    |   26.43 |   4.84 |         19 | 🔴    |
| vanilla_p2     | cold            |   29.34 |   4.36 |         56 |       |
| vanilla_p2     | warm            |   14.82 |   8.64 |         18 | 🔴    |
| vanilla_p2     | warm-diff-mnt   |    2.66 |  96.16 |          0 | 🟢    |
| vanilla_p2     | warm-diff-in    |   14.42 |   8.88 |          0 | 🟢    |
| vanilla_p1_p2  | cold            |   28.94 |   4.42 |         54 |       |
| vanilla_p1_p2  | warm            |    1.35 |  94.77 |          0 | 🟢    |
| vanilla_p1_p2  | warm-diff-mnt   |    2.66 |  96.41 |          0 | 🟢    |
| vanilla_p1_p2  | warm-diff-in    |   14.51 |   8.82 |          0 | 🟢    |
| diy            | cold            |   28.77 |   4.45 |         54 |       |
| diy            | warm            |    1.35 |  94.88 |          0 | 🟢    |
| diy            | warm-diff-mnt   |    2.65 |  96.44 |          0 | 🟢    |
| diy            | warm-diff-in    |   14.36 |   8.91 |          0 | 🟢    |
| static_tensors | cold            |   26.87 |   4.76 |         54 |       |
| static_tensors | warm            |    1.22 | 105.04 |          0 | 🟢    |
| static_tensors | warm-diff-mnt   |    2.39 | 107.21 |          0 | 🟢    |
| static_tensors | warm-diff-in    |   13.46 |   9.51 |          0 | 🟢    |

The ablation is clean: p1 alone clears only `warm`; p2 alone clears
only `warm-diff-mnt` / `warm-diff-in`; together they recover full
DIY-equivalent behavior. The two fixes are independent — neither
subsumes the other.

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
