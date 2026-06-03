# Generation loop on transformers + `torch.compile` — findings

Scope: [SCOPE.md](SCOPE.md). One-process, batch-1 generation loop on
transformers v5 with chunked prefill, no prefix caching.

Test bed: 1× NVIDIA A10G (23 GiB), bf16, torch 2.7.0+cu126,
transformers 5.10.0.dev0 (upstream `main` at commit 595721c),
`Llama-3.2-1B-Instruct`.

## TL;DR

Three approaches to running `generate()` with `torch.compile` + chunked
prefill, in ascending order of stability:

1. **vanilla** — set `cache_implementation="static"` and call
   `generate()`. Works for repeated identical calls. Silently recompiles
   any time `max_length = max_new_tokens + input_ids_length` grows: a
   longer prompt costs ~2.7× warm wallclock, a larger `max_new_tokens`
   costs ~1.8×. The auto-allocated StaticCache is the root cause.
2. **diy** — construct the StaticCache yourself sized for the worst
   case, pass via `past_key_values=`, and *do not* set
   `cache_implementation`. Recompile-free across both deltas. The
   recipe.
3. **static_tensors** — DIY cache *plus* pre-allocate the decode loop's
   tensors (input_ids, position_ids, cache_position, 4D mask) and call
   `model.get_compiled_call()` directly with manual argmax sampling.
   Same cache properties as DIY, plus ~12 % higher steady-state tok/s
   by skipping `generate()`'s Python plumbing.

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

Each cell is `total_s / tps / +artifacts`. Color rates each warm-diff
cell against the same mode's warm wallclock:

- 🟢 ≤ 1.5× warm (essentially a cache hit)
- 🟡 1.5–10× warm (partial reuse)
- 🔴 > 10× warm (effectively a recompile)

### Results

| mode | cold | warm | warm-diff-mnt | warm-diff-in |
|---|---|---|---|---|
| vanilla        | 29.6 s / 4.3 tps / +56 | 14.9 s / 8.6 tps / +18 | 🟡 27.5 s / 9.3 tps / +21  | 🟡 40.5 s / 3.2 tps / +30 |
| diy            | 29.3 s / 4.4 tps / +56 | 15.2 s / 8.4 tps / +18 | 🟢 2.7 s / 96.0 tps / +0   | 🟢 14.8 s / 8.6 tps / +0  |
| static_tensors | 26.7 s / 4.8 tps / +56 | 13.6 s / 9.4 tps / +20 | 🟢 2.4 s / 107.2 tps / +0  | 🟢 13.2 s / 9.7 tps / +0  |

### Takeaways

1. **Vanilla is unstable across both deltas.** The auto-allocated
   StaticCache uses `max_cache_len = max_new_tokens + input_length - 1`
   as its key
   ([generation/utils.py:2495](src/transformers/generation/utils.py#L2495))
   — any change in either dimension forces a realloc and recompile. On
   CUDA the absolute cost is 13–26 s per first occurrence (vs minutes
   on Neuron, but the per-turn TTFT spike is still very visible in an
   agent loop).
2. **DIY rescues vanilla completely.** Construct the cache once at the
   worst-case size, pass via `past_key_values=`, drop
   `cache_implementation` (the two can't coexist —
   [generation/utils.py:1822](src/transformers/generation/utils.py#L1822)).
   The auto-compile criterion checks `cache.is_compileable`, not the
   config field, so the compile path still kicks in.
3. **`static_tensors` adds ~12 % decode tok/s on top of DIY** (107 vs 96
   tps on warm-diff-mnt). The win is generate()'s Python overhead —
   `torch.cat` of growing tensors plus the mask rebuild per step — that
   the direct `compiled_call()` path skips. See
   [evidence/decode_overhead.py](evidence/decode_overhead.py) for the
   isolated microbenchmark; the relative win shrinks at larger model
   sizes where decode is bandwidth-bound.
4. **The first warm call adds ~18 artifacts in every mode.** A lazy
   late compile inside `generate()` fires on the second call after
   cold. It costs ~10–12 s wallclock and is mode-independent, so it
   doesn't affect the diff measurements — they're rated against the
   warm wallclock, which already includes the lazy cost.
5. **Scenario order matters for vanilla.** `warm-diff-mnt` is run
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
2. **`prefill_chunk_size` is undocumented public API.** Defined via
   `kwargs.pop` at
   [configuration_utils.py:465](src/transformers/generation/configuration_utils.py#L465)
   and absent from the class docstring.
3. **No bucket-padding helper.** The whole point of buckets is
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
