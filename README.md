# Generation loop on transformers + `torch.compile` — findings

Scope: [SCOPE.md](SCOPE.md). One-process, batch-1 generation loop on
transformers v5 with chunked prefill, no prefix caching.

Test bed: 1× NVIDIA A10G (23 GiB), bf16, torch 2.7.0+cu126,
transformers 5.10.0.dev0 (upstream `main` at commit 595721c),
`Llama-3.2-1B-Instruct` only. Gemma-4-E4B was in SCOPE.md but was
dropped during simplification; the hybrid / sliding-window open
question therefore stays open (see caveats).

Each scenario cell below is a single timed call (N=1). SCOPE asked
for N≥10 with TTFT variance; the new bench traded that for
side-by-side mode comparison. Treat warm-diff cells as "this delta
does/doesn't recompile" indicators (the +artifacts column is the
unambiguous signal), not as steady-state performance numbers.

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

See [scenario sweep](#scenario-sweep) for the data. Two findings sit
outside the sweep:

- **Cache over-provisioning costs decode tok/s.** Pinning a 1024-prompt
  call to a cache sized 8192+128 halves decode throughput. See
  [cache pinning vs over-provisioning](#cache-pinning-vs-over-provisioning).
- **`generate()`'s Python decode loop costs ~1.1 ms / step on CUDA.**
  ~14 % of wallclock on a 1B model. `static_tensors` removes it. See
  [evidence/decode_overhead.py](evidence/decode_overhead.py).

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

Each cell is `total_s / tps / +artifacts`. Color is on the +artifacts
column alone — the unambiguous "did anything recompile" signal:

- 🟢 +0 artifacts (cache absorbed the delta cleanly)
- 🔴 +artifacts > 0 (real Inductor recompile)

Wallclock ratios stopped being a useful color basis once the diy /
static_tensors warms became clean steady-state (more tokens = more
wallclock even with no recompile). They still appear in the table as
context.

### Results

| mode | cold | warm | warm-diff-mnt | warm-diff-in |
|---|---|---|---|---|
| vanilla        | 29.7 s / 4.3 tps / +56 | 14.9 s / 8.6 tps / +18 | 🔴 27.6 s / 9.3 tps / +21  | 🔴 40.8 s / 3.1 tps / +30 |
| diy            | 28.9 s / 4.4 tps / +54 |  1.4 s / 95.0 tps / +0 | 🟢 2.6 s / 96.6 tps / +0   | 🟢 14.7 s / 8.7 tps / +0  |
| static_tensors | 26.6 s / 4.8 tps / +54 |  1.2 s / 105.1 tps / +0 | 🟢 2.4 s / 107.2 tps / +0 | 🟢 13.6 s / 9.4 tps / +0  |

### Takeaways

1. **Vanilla is unstable across both deltas.** The auto-allocated
   StaticCache uses `max_cache_len = max_new_tokens + input_length - 1`
   as its key
   ([generation/utils.py:2495](src/transformers/generation/utils.py#L2495))
   — any change in either dimension forces a realloc and recompile.
   Both warm-diff cells add 21–30 new Inductor artifacts (the +artifacts
   column) and 13–26 s of wallclock per first occurrence — the per-turn
   TTFT spike an agent would see.
2. **DIY + `early_initialization` absorbs both deltas, one-shot.**
   Construct the cache once at the worst-case size, call
   `cache.early_initialization(...)` once, pass via `past_key_values=`,
   drop `cache_implementation` (the two can't coexist —
   [generation/utils.py:1822](src/transformers/generation/utils.py#L1822)).
   The auto-compile criterion checks `cache.is_compileable`, not the
   config field, so the compile path still kicks in. Both warm-diff
   cells show 0 new artifacts — the cache absorbs the delta. Each cell
   is still a single timed call, so this is "no recompile on this
   delta", not a 10-call steady-state validation; but the +artifacts
   signal is mechanically unambiguous.
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
   shows 14.7 s wallclock with **+0 new Inductor artifacts** — i.e.
   no kernel was compiled, but something inside Dynamo still took
   ~13 s. The most likely culprit: the longer prompt forces Dynamo
   to re-trace the prefill graph for the new outer `input_ids` shape
   (1, 2048), discover the per-chunk shape (1, 1024) is unchanged,
   and hit the FX-graph cache for the chunk kernel — paying the
   tracing time but not the codegen time. We did not chase this; the
   cache absorption claim (0 artifacts → no recompile) still holds.
6. **Scenario order matters for vanilla.** `warm-diff-mnt` is run
   before `warm-diff-in` so each delta forces a fresh cache realloc.
   With the reverse order, `warm-diff-in` grows the auto-cache to 2175
   slots and `warm-diff-mnt` (needing only 1279) silently hits the
   larger cache — hiding the footgun. Diy and static_tensors are
   order-independent because their cache is pre-sized.

## Cache pinning vs over-provisioning

Separate from the scenario sweep above. When a single process serves
prompts of very different sizes (e.g. a 1024 bucket and an 8192 bucket),
the StaticCache must be sized for the largest. The small-bucket calls
then attend over a mostly-empty buffer:

| 1024-bucket measurement | cache=1024+128 | cache=8192+128 |
|---|---|---|
| TTFT p50 (chunk=1024) | 47.7 ms | 76.2 ms (1.60× slower) |
| decode tok/s | 115 | 52 (**2.2× slower**) |

Numbers from a separate run with a different config; not regenerated.
The actionable rule: if prompt-length distribution is bimodal, run two
compiled processes (one per bucket family), not one shared.

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

## Caveats and what we didn't test

- **Left-pad output quality**: not end-to-end tested. The bench uses
  random BOS-prefilled ids; before serving real traffic, verify that
  the model produces the same tokens with and without padding.
- **`fullgraph=True` with sdpa**: not tested.
- **Hybrid / sliding-window attention models** (Gemma, Mistral-SWA,
  etc.): not measured in this iteration. `prefill_chunk_size >
  sliding_window` is anecdotally known to interact poorly with kernel
  choice on those models, but we have no fresh data.
- **Compile mode (`default` vs `max-autotune-no-cudagraphs`)**: not
  varied in the scenario bench. A separate run on Llama-1B showed
  autotune buys ~8–9 % TTFT for 3× the warmup wallclock; `default` is
  the right production default.
- **Batch > 1** is out of scope.

## Reproduce

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -e /path/to/transformers accelerate
```

Then run the scenario sweep (orchestrator spawns one subprocess per
mode, each with a fresh Inductor cache):

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_scenarios.py
```

Or run a single mode directly (used internally by the orchestrator):

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_scenarios.py --mode diy
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
