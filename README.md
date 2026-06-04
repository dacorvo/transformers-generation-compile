# Generation loop on transformers + `torch.compile` — findings

Cache and recompile behavior of `model.generate()` on transformers v5
with `torch.compile` + chunked prefill, in a single-request access
pattern with varying prompt and decode-budget sizes across calls.

Setup: 1× NVIDIA A10G (23 GiB), bf16, torch 2.7.0+cu126,
transformers 5.10.0.dev0 (upstream `main` at commit 595721c),
`Llama-3.2-1B-Instruct`.

Each scenario cell is a single timed call (N=1). The `+artifacts`
column is the unambiguous "did Inductor recompile" signal; wallclock
is informational.

## TL;DR

Three approaches to running `generate()` with `torch.compile` + chunked
prefill, in ascending order of stability:

1. **vanilla** — set `cache_implementation="static"` and call
   `generate()`. Works for repeated identical calls. Silently recompiles
   any time `max_length = max_new_tokens + input_ids_length` grows: a
   longer prompt costs ~2.7× warm wallclock, a larger `max_new_tokens`
   costs ~1.8×. The auto-allocated StaticCache is the root cause.
2. **diy** — construct the StaticCache yourself sized for the worst
   case, call `cache.early_initialization(...)` *once* before the
   first generate() (mandatory whenever the prefill itself is compiled —
   which `prefill_chunk_size` triggers; otherwise the second call eats
   ~13 s retracing — see takeaway #2), pass the cache via
   `past_key_values=`, and *do not* set `cache_implementation`. Both
   warm-diff cells absorb cleanly: +0 new Inductor artifacts. The
   provisional recipe.
3. **static_tensors** — DIY cache (with the same `early_initialization`
   step) *plus* pre-allocate the decode loop's tensors (input_ids,
   position_ids, cache_position, 4D mask) and call
   `model.get_compiled_call()` directly with manual argmax sampling.
   Same cache behavior as DIY, plus ~11 % higher decode tok/s
   (107 vs 95 on a clean warm cell) by skipping `generate()`'s
   Python plumbing.

See [scenario sweep](#scenario-sweep) for the data. One finding sits
outside the sweep: **`generate()`'s Python decode loop costs ~1.1 ms /
step on CUDA** (~14 % of wallclock on a 1B model). `static_tensors`
removes it. See [evidence/decode_overhead.py](evidence/decode_overhead.py).

## Scenario sweep

Driven by [bench_scenarios.py](bench_scenarios.py). Raw output in
[`logs/scenarios.tsv`](logs/scenarios.tsv); appendix in
[results.md](results.md).

### Config

| Parameter | Value |
|---|---|
| model | `meta-llama/Llama-3.2-1B-Instruct` |
| `max_seq_len` | 2048 |
| `prefill_chunk_size` | 1024 |
| short prompt | 1024 tokens (1 chunk) |
| long prompt | 2048 tokens (2 chunks) |
| `max_new_tokens` (default) | 128 |
| `max_new_tokens` (warm-diff-mnt) | 256 |
| DIY cache `max_cache_len` | 2304 |

### Four scenarios

- **cold** — empty Inductor cache. First call pays the full compile cost.
- **warm** — same prompt + `max_new_tokens` as cold. Should be a clean
  cache hit.
- **warm-diff-mnt** — warm cache, same prompt, larger `max_new_tokens`.
- **warm-diff-in** — warm cache, longer prompt.

Each cell is `total_s / tps / +artifacts`. Color is on the
`+artifacts` column alone, applied to every cell except `cold` (the
compile baseline, by definition non-zero):

- 🟢 +0 artifacts (cache absorbed the delta; no Inductor compile)
- 🔴 +artifacts > 0 (real Inductor recompile)

Wallclock is context, not the color basis — more tokens take more
wallclock even with no recompile.

### Results

| mode            | cold | warm | warm-diff-mnt | warm-diff-in |
|---|---|---|---|---|
| vanilla         | 29.6 s / 4.3 tps / +56 | 🔴 15.0 s / 8.6 tps / +18  | 🔴 28.3 s / 9.0 tps / +21 | 🔴 41.2 s / 3.1 tps / +30 |
| vanilla_p1      | 28.7 s / 4.5 tps / +54 | 🟢  1.1 s / 111.2 tps / +0 | 🔴 27.0 s / 9.5 tps / +19 | 🔴 26.4 s / 4.9 tps / +19 |
| vanilla_p2      | 29.3 s / 4.4 tps / +56 | 🔴 14.8 s / 8.6 tps / +18  | 🟢  2.7 s / 96.2 tps / +0 | 🟢 14.4 s / 8.9 tps / +0  |
| vanilla_p1_p2   | 28.9 s / 4.4 tps / +54 | 🟢  1.4 s / 94.8 tps / +0  | 🟢  2.7 s / 96.4 tps / +0 | 🟢 14.5 s / 8.8 tps / +0  |
| diy             | 28.8 s / 4.5 tps / +54 | 🟢  1.4 s / 94.9 tps / +0  | 🟢  2.6 s / 96.4 tps / +0 | 🟢 14.4 s / 8.9 tps / +0  |
| static_tensors  | 26.9 s / 4.8 tps / +54 | 🟢  1.2 s / 105.0 tps / +0 | 🟢  2.4 s / 107.2 tps / +0 | 🟢 13.5 s / 9.5 tps / +0  |

The four `vanilla*` rows are an ablation across two proposed upstream
fixes, each applied as a runtime monkey patch on stock 5.10.1
(`_apply_monkey_patches` in [bench_scenarios.py](bench_scenarios.py)):

- **p1** — auto-call `cache.early_initialization(...)` from
  `_prepare_static_cache`. Defuses the lazy-init recompile (see
  [ISSUE_DRAFT.md](ISSUE_DRAFT.md)). Visible effect: `warm` clears.
- **p2** — honor `generation_config.cache_config["max_cache_len"]` on the
  static path of `_prepare_cache_for_generation` (the `GenerationConfig`
  docstring implies it's honored, but it is silently ignored — verified
  in [evidence/cache_config_max_cache_len.py](evidence/cache_config_max_cache_len.py)).
  Visible effect: `warm-diff-mnt` and `warm-diff-in` clear because the
  cache no longer reallocates when `max_length` grows.

`vanilla_p1_p2` recovers DIY-equivalent behavior without the user
constructing a `StaticCache` themselves. The two fixes are orthogonal:
neither alone clears all three warm cells, and both together do.

### Takeaways

1. **Vanilla recompiles on every delta.** Cache key is
   `max_cache_len = max_new_tokens + input_length - 1`
   any growth in either dimension forces a realloc. Both warm-diff
   cells add 21–30 artifacts and 13–26 s of wallclock — the per-turn
   TTFT spike an agent sees.
2. **Chunked prefill + any static cache: `is_initialized` is a Dynamo
   footgun.** With `prefill_chunk_size` set, `generate()` compiles the
   prefill (via `model.get_compiled_call()` per chunk). The first chunk
   traces with `is_initialized=False`; `lazy_initialization` then flips
   it to `True`. On the next call Dynamo's `___check_obj_id` guard on
   that Python bool fails (`id(False) != id(True)`) and the prefill
   re-traces. Affects every static-cache flavour we tested
   (auto-allocated by `cache_implementation="static"` or DIY-constructed
   and passed via `past_key_values=`). It does *not*
   happen with unchunked prefill — there the flip happens in eager code
   that no compiled graph guards on (verified in
   [evidence/unchunked_no_early_init.py](evidence/unchunked_no_early_init.py):
   cold → warm shows +0 artifacts without any `early_initialization`
   call). Fix: call `cache.early_initialization(...)` before the first
   generate() — collapses warm from 15 s / +18 to 1.4 s / +0.
3. **`static_tensors` buys ~12 % decode tok/s over DIY** (107 vs 96
   tps on warm-diff-mnt). The win is `generate()`'s Python decode
   plumbing — `torch.cat` of growing tensors and the per-step mask
   rebuild — that direct `compiled_call()` skips. Isolated
   microbenchmark: [evidence/decode_overhead.py](evidence/decode_overhead.py).

## Worth flagging upstream

**`cache_implementation="static"` conflicts with DIY cache if both are set**

## Reproduce

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python transformers accelerate
```

Then run the scenario sweep (orchestrator spawns one subprocess per
mode, each with a fresh Inductor cache):

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_scenarios.py
```

The three orthogonal evidence scripts in [evidence/](evidence/) each
test a separate question and print results to stdout:

- [`padding.py`](evidence/padding.py) — tail-chunk shape recompile
  (input length must be a multiple of `prefill_chunk_size`).
- [`interleave_buckets.py`](evidence/interleave_buckets.py) —
  cross-bucket dispatch survives a random-order loop with zero
  recompiles.
- [`decode_overhead.py`](evidence/decode_overhead.py) — isolated
  measurement of generate()'s Python loop overhead vs a direct
  compiled-forward loop.

Every script reads its purpose from the first ~10 lines of its file.
