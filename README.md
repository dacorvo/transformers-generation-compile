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
   Dynamo guard miss on `is_initialized` — see takeaway #4), pass the
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

1. **Vanilla is unstable across both deltas.** The auto-allocated
   StaticCache uses `max_cache_len = max_new_tokens + input_length - 1`
   as its key
   ([generation/utils.py:2495](src/transformers/generation/utils.py#L2495))
   — any change in either dimension forces a realloc and recompile.
   Both warm-diff cells add 21–30 new Inductor artifacts (the +artifacts
   column) and 13–26 s of wallclock per first occurrence — the per-turn
   TTFT spike an agent would see.
2. **DIY + `early_initialization` absorbs both deltas.** Construct the
   cache once at the worst-case size, call
   `cache.early_initialization(...)`, pass via `past_key_values=`, drop
   `cache_implementation` (the two can't coexist —
   [generation/utils.py:1822](src/transformers/generation/utils.py#L1822)).
   The auto-compile criterion checks `cache.is_compileable`, not the
   config field, so the compile path still kicks in. Both warm-diff
   cells show 0 new artifacts — the cache absorbs the delta.
3. **`static_tensors` adds ~12 % decode tok/s on top of DIY** (107 vs 96
   tps on warm-diff-mnt). The win is generate()'s Python overhead —
   `torch.cat` of growing tensors plus the mask rebuild per step — that
   the direct `compiled_call()` path skips. See
   [evidence/decode_overhead.py](evidence/decode_overhead.py) for the
   isolated microbenchmark; the relative win shrinks at larger model
   sizes where decode is bandwidth-bound.
4. **The "warm" recompile in vanilla is a Dynamo guard miss on
   `is_initialized`, and `cache.early_initialization()` fixes it.**
   The StaticCache layer's `is_initialized` flag is a plain Python
   bool that flips False → True during the first prefill's
   `lazy_initialization` ([cache_utils.py:336](src/transformers/cache_utils.py#L336)).
   Dynamo's `___check_obj_id` guard captures the *object identity* of
   the False at trace time and fires on the second call because True
   is a different Python object — forcing the prefill graph to
   re-trace. `cache.reset()` doesn't restore the original False
   object. Cost: ~18 redundant Inductor artifacts and ~13 s wallclock
   on every second-call-after-cold transition.

   The fix is documented at [cache_utils.py:302](src/transformers/cache_utils.py#L302):
   call `cache.early_initialization(batch_size, num_heads, head_dim,
   dtype, device)` once before the first generate(). The flip then
   happens before Dynamo traces; subsequent calls hit the cache
   cleanly. In our bench, applying this to diy and static_tensors
   collapses warm from ~15 s / +18 artifacts to ~1.4 s / +0 artifacts.

   **Vanilla can't use this fix** — `cache_implementation="static"`
   makes generate() construct the cache internally, so the user has
   no handle on it before tracing. That's a second strike against
   vanilla beyond the cache realloc footgun. Closing this would need
   an upstream change in `_prepare_static_cache` (call
   `early_initialization()` on the freshly-allocated cache before
   returning).
5. **One Dynamo retrace anomaly on `diy / warm-diff-in`.** The cell
   shows 14.7 s wallclock with **+0 new Inductor artifacts** — no
   kernel was compiled, but something inside Dynamo still took
   ~13 s. Likely cause: the longer prompt forces Dynamo to re-trace
   the prefill graph for the new outer `input_ids` shape (1, 2048),
   the per-chunk shape (1, 1024) is unchanged so the FX-graph cache
   serves the chunk kernel — tracing time is paid, codegen time is
   not. Not investigated further. The cache-absorption claim
   (0 artifacts → no Inductor recompile) holds regardless.
6. **Scenario order matters for vanilla.** `warm-diff-mnt` is run
   before `warm-diff-in` so each delta forces a fresh cache realloc.
   With the reverse order, `warm-diff-in` grows the auto-cache to 2175
   slots and `warm-diff-mnt` (needing only 1279) silently hits the
   larger cache — hiding the footgun. Diy and static_tensors are
   order-independent because their cache is pre-sized.

## Worth flagging upstream

1. **`cache_implementation="static"` is a TTFT-bomb-shaped default.**
   The auto-cache reallocates on any `max_length` growth and there's
   no documented escape. The DIY pattern works but conflicts with
   `cache_implementation` at runtime
   ([generation/utils.py:1822](src/transformers/generation/utils.py#L1822))
   without prose explaining why a user might want either. Two fixes:
   (a) doc note pointing to the DIY pattern under the
   `cache_implementation` docstring entry, or (b) honor
   `cache_config["max_cache_len"]` in `_prepare_static_cache` (3-line
   change).
2. **`CacheLayerMixin.is_initialized` triggers a Dynamo recompile on
   the second call** (takeaway #4). The flag is a plain Python bool
   that flips False → True inside the compiled forward, and Dynamo
   guards on it by object id via `___check_obj_id`. Result: ~18
   redundant Inductor artifacts and ~13 s wallclock on every
   warm-after-cold call until the user calls `early_initialization`.
   Three possible fixes, increasing in invasiveness:
   - **Doc**: surface `cache.early_initialization()` as the
     recommended workaround for agentic loops with chunked prefill,
     not just for torch.export. The existing docstring at
     [cache_utils.py:302](src/transformers/cache_utils.py#L302)
     mentions it but frames it as an export-only knob.
   - **Auto-call**: have `_prepare_static_cache` call
     `early_initialization()` on the freshly-allocated cache before
     returning — closes the gap for `cache_implementation="static"`
     users too.
   - **Avoid the guard**: replace the Python bool with a tensor or
     class-level constant so Dynamo specializes on value, not on
     mutable object identity. Removes the trap entirely.
3. **`prefill_chunk_size` is undocumented public API.** Defined via
   `kwargs.pop` at
   [configuration_utils.py:465](src/transformers/generation/configuration_utils.py#L465)
   and absent from the class docstring.
4. **No bucket-padding helper.** The whole point of buckets is
   recompile-free dispatch, and that requires exact-shape input. Every
   team using this pattern rolls their own pad-to-next-bucket loop.

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
