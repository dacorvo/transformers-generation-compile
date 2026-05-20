# Raw sweep results

Detailed per-cell JSON output for each (model × mode) combination. See
[RECIPE.md](RECIPE.md) for analysis.

## Files

- `logs/llama-default.json`  — Llama-3.2-1B, mode=default, 4 cells
- `logs/llama-mautotune.json` — Llama-3.2-1B, mode=max-autotune-no-cudagraphs, 4 cells
- `logs/llama-default-1k-only.json` — Llama-3.2-1B, mode=default, 1024-bucket only (baseline for "cost of pinning cache to bigger bucket")
- `logs/gemma-default.json` — Gemma-4-E4B, mode=default (in progress)
- `logs/gemma-mautotune.json` — Gemma-4-E4B, mode=max-autotune-no-cudagraphs (in progress)

## Summary table (so far)

| model | mode | bucket | chunk | warmup s | TTFT p50 ms | p99/p50 | decode tok/s | ss recompiles |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Llama-1B | default | 1024 |  512 | 14.4 |  86.6 | 1.016 |  52.3 | 0 |
| Llama-1B | default | 1024 | 1024 | 15.2 |  76.2 | 1.010 |  52.3 | 0 |
| Llama-1B | default | 8192 |  512 | 42.9 | 676.5 | 1.001 |  52.4 | 0 |
| Llama-1B | default | 8192 | 1024 | 16.2 | 594.2 | 1.001 |  52.3 | 0 |
| Llama-1B | autotune | 1024 |  512 | 36.6 |  80.6 | 1.020 |  54.8 | 0 |
| Llama-1B | autotune | 1024 | 1024 | 36.5 |  69.8 | 1.016 |  54.9 | 0 |
| Llama-1B | autotune | 8192 |  512 | 140.0 | 627.5 | 1.002 | 54.9 | 0 |
| Llama-1B | autotune | 8192 | 1024 |  61.6 | 544.3 | 1.003 | 54.8 | 0 |
| Llama-1B | default | 1024 |  512 | 41.1 |  52.3 | 1.019 | 114.7 | 0 | *(cache pinned to 1024+128, not 8192+128)*
| Llama-1B | default | 1024 | 1024 | 14.3 |  47.7 | 1.010 | 115.1 | 0 | *(cache pinned to 1024+128)*
| Gemma-4-E4B | default | 1024 |  512 |  10.7 |  285.6 | 1.006 | 12.2 | 0 |
| Gemma-4-E4B | default | 1024 | 1024 |  10.6 |  283.1 | 1.001 | 12.2 | 0 |
| Gemma-4-E4B | default | 8192 |  512 | 2725.8 | 1770.2 | 1.000 | 14.8 | 0 |
| Gemma-4-E4B | default | 8192 | 1024 |  12.0 | 2298.1 | 1.000 | 14.8 | 0 |
| Gemma-4-E4B | autotune | 1024 |  512 |  10.7 |  285.5 | 1.008 | 12.1 | 0 |
| Gemma-4-E4B | autotune | 1024 | 1024 |  10.7 |  282.5 | 1.003 | 12.1 | 0 |
| Gemma-4-E4B | autotune | 8192 |  512 | 4002.3 | 1745.6 | 1.001 | 15.1 | 0 |
| Gemma-4-E4B | autotune | 8192 | 1024 |  15.0 | 2292.7 | 1.000 | 15.1 | 0 |

Decode-loop overhead (Llama-1B, prompt=256, decode=256, cache=544):

| measurement | per-step | tok/s |
|---|--:|--:|
| `model.generate()` | 7.85 ms | 127.4 |
| compiled-forward direct call, no growing tensors | 6.71 ms | 149.0 |
| gap (Python plumbing + tensor concat) | **1.13 ms (~14 %)** | – |

"ss recompiles" = number of new Inductor artifacts (cubin/so) added
during the 10-call steady-state phase.

## Where the raw data lives

Each cell JSON includes:
- `ttft_s_calls`: the 10 individual TTFT timings.
- `ttft_s_median`, `ttft_s_p99`, `ttft_s_mean`, `ttft_p99_over_p50`.
- `decode_total_s`, `decode_n_new`, `decode_tok_per_s`.
- `ss_artifacts_delta`: Inductor cache file count change during
  steady-state TTFT calls.
- `ss_counters_delta`: `torch._dynamo.utils.counters` diff (empty
  when steady-state added no graphs, frames, or aten ops).
- `warmup_*` parallel fields.

Sample inspection:

```sh
jq '.cells[] | {input_len, chunk_size, ttft_s_median, ss_artifacts_delta}' \
   logs/llama-default.json
```
