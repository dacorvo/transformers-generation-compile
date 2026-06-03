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
   first generate() (otherwise you eat ~13 s on the second call from a
   Dynamo guard miss on `is_initialized` — see takeaway #2), pass the
   cache via `past_key_values=`, and *do not* set
   `cache_implementation`. Both warm-diff cells absorb cleanly:
   +0 new Inductor artifacts. The provisional recipe.
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

| mode | cold | warm | warm-diff-mnt | warm-diff-in |
|---|---|---|---|---|
| vanilla        | 29.7 s / 4.3 tps / +56 | 🔴 14.9 s / 8.6 tps / +18 | 🔴 27.6 s / 9.3 tps / +21  | 🔴 40.8 s / 3.1 tps / +30 |
| diy            | 28.9 s / 4.4 tps / +54 | 🟢  1.4 s / 95.0 tps / +0 | 🟢  2.6 s / 96.6 tps / +0  | 🟢 14.7 s / 8.7 tps / +0  |
| static_tensors | 26.6 s / 4.8 tps / +54 | 🟢  1.2 s / 105.1 tps / +0 | 🟢  2.4 s / 107.2 tps / +0 | 🟢 13.6 s / 9.4 tps / +0  |

### Takeaways

1. **Vanilla recompiles on every delta.** Cache key is
   `max_cache_len = max_new_tokens + input_length - 1`
   ([generation/utils.py:2495](src/transformers/generation/utils.py#L2495));
   any growth in either dimension forces a realloc. Both warm-diff
   cells add 21–30 artifacts and 13–26 s of wallclock — the per-turn
   TTFT spike an agent sees.
2. **Warm's recompile is a Dynamo guard miss on `is_initialized`.**
   The flag is a Python bool that flips False → True during the
   first prefill's `lazy_initialization`
   ([cache_utils.py:336](src/transformers/cache_utils.py#L336));
   Dynamo's `___check_obj_id` guard fails on the second call (True
   is a different object than the original False), so the prefill
   graph re-traces. `cache.reset()` doesn't restore the original
   object. Fix: call `cache.early_initialization(...)` before the
   first generate()
   ([cache_utils.py:302](src/transformers/cache_utils.py#L302));
   in this bench it collapses warm from 15 s / +18 to 1.4 s / +0.
3. **`static_tensors` buys ~12 % decode tok/s over DIY** (107 vs 96
   tps on warm-diff-mnt). The win is `generate()`'s Python decode
   plumbing — `torch.cat` of growing tensors and the per-step mask
   rebuild — that direct `compiled_call()` skips. Isolated
   microbenchmark: [evidence/decode_overhead.py](evidence/decode_overhead.py).
4. **`diy / warm-diff-in` is slow with +0 artifacts.** 14.7 s
   wallclock, no new Inductor compile — Dynamo likely re-traces the
   outer prefill graph for the new `input_ids` shape (1, 2048) and
   hits the FX-graph cache for the per-chunk (1, 1024) kernel.
   Tracing time paid, codegen time not. The cache-absorption claim
   stands.
5. **Scenario order matters for vanilla.** `warm-diff-mnt` runs
   before `warm-diff-in` so each delta forces a fresh realloc.
   Reversed, `warm-diff-in`'s bigger cache absorbs `warm-diff-mnt`
   and hides the footgun. DIY/static_tensors are order-independent
   (cache pre-sized).

## Worth flagging upstream

1. **`cache_implementation="static"` reallocates on any `max_length`
   growth** and has no documented escape. The DIY workaround conflicts
   with `cache_implementation` at runtime
   ([generation/utils.py:1822](src/transformers/generation/utils.py#L1822))
   with no prose explaining why a user might want either. Fixes:
   (a) doc-note the DIY pattern in the `cache_implementation` docstring,
   or (b) honor `cache_config["max_cache_len"]` in
   `_prepare_static_cache` (3-line change).
2. **`CacheLayerMixin.is_initialized` is Dynamo-guarded by object id**
   (mechanism in takeaway #2). Vanilla users can't apply
   `early_initialization()` themselves because the cache is built
   inside generate(). Fixes, ascending in invasiveness: doc the knob
   for agentic loops (existing
   [cache_utils.py:302](src/transformers/cache_utils.py#L302) frames
   it as export-only); auto-call from `_prepare_static_cache`;
   replace the bool with a tensor/class-level constant so the guard
   specializes on value.
3. **`prefill_chunk_size` is undocumented public API.** Set via
   `kwargs.pop` at
   [configuration_utils.py:465](src/transformers/generation/configuration_utils.py#L465);
   absent from the class docstring.
4. **No bucket-padding helper.** Recompile-free dispatch needs
   exact-shape input; every team rolls its own pad-to-next-bucket loop.

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
