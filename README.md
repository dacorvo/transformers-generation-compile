# Generation loop on transformers + `torch.compile` — findings

Scope: [SCOPE.md](SCOPE.md). One-process, batch-1 generation loop on
transformers v5 with `cache_implementation="static"` + chunked
prefill, no prefix caching.

Test bed: 1× NVIDIA A10G (23 GiB), bf16, torch 2.7.0+cu126,
transformers 5.8.0.dev0 (local editable), `Llama-3.2-1B-Instruct` and
`google/gemma-4-E4B-it`.

## TL;DR

Recompile-free, low-variance steady-state TTFT (p99/p50 ≤ 1.02 on
Llama, ≤ 1.006 on Gemma) is achievable today, but only if four
behaviors are baked into the generate() calls. None of them surface
in the public knobs.

1. **Pad inputs to a fixed bucket** that is a clean multiple of
   `prefill_chunk_size`. Otherwise the tail chunk has a unique shape
   and gets compiled on first occurrence.
2. **Pin the StaticCache size yourself.** Construct
   `StaticCache(config=..., max_cache_len=N)`, pass it via
   `past_key_values=`, and *do not* set `cache_implementation`. The
   built-in auto-cache reallocates the buffer whenever a later call
   has a bigger `max_length`, which forces a recompile.
3. **Warm the largest bucket first.** With the DIY cache, ordering
   only matters for Inductor's shape set; warming the worst case
   first means smaller buckets reuse those kernels.
4. **Use `mode="default"`, not the library default `"reduce-overhead"`.**
   `reduce-overhead` uses CUDA Graphs, which conflict with chunked
   prefill. Among the two CUDA-Graphs-free options, `default` and
   `max-autotune-no-cudagraphs`, autotune buys 8–9 % TTFT on Llama
   and ≤ 1.4 % on Gemma at 1.5–3× the warmup cost.

For hybrid-attention models a fifth rule applies: **`prefill_chunk_size
≤ sliding_window`**. Cross by 2× on Gemma-4-E4B and you eat a 30 %
TTFT regression at the 8K bucket.

Separately: generate()'s Python decode loop carries ~1.1 ms/step of
overhead from growing `input_ids`/`attention_mask`/`position_ids` —
14–17 % of decode wallclock on a 1B model. Not a recompile on CUDA
(unlike Neuron/TPU), but real allocator/dispatch cost. Shrinks
proportionally as model size grows.

## Findings, with the script that proves each

### Silent recompiles

| # | Finding | Reproducer |
|---|---|---|
| 1 | Tail chunk: `L % chunk_size ≠ 0` adds 25–31 Inductor artifacts and ~26 s per first-seen tail length. | [evidence/padding.py](evidence/padding.py) |
| 2 | Cache realloc: `max_new_tokens` exceeding any seen-so-far reallocates the StaticCache and triggers a full ~15–30 s recompile silently. | [evidence/cache_realloc.py](evidence/cache_realloc.py) |
| 2-fix | DIY-StaticCache (size once, pass as `past_key_values=`, drop `cache_implementation`) makes `max_new_tokens` free across calls. | [evidence/cache_realloc_fix.py](evidence/cache_realloc_fix.py) |
| – | Cross-bucket dispatch is genuinely recompile-free after warmup: 20 random-bucket calls, 0 new artifacts, p99/p50 = 1.001. | [evidence/interleave_buckets.py](evidence/interleave_buckets.py) |

### Cheap wins (Llama-1B unless noted)

1. **Warm largest bucket first.** With this order the four warmup
   cells add 76 + 29 + 0 + 0 = 105 Inductor artifacts; the small-bucket
   cells reuse the prefill kernels compiled at the large-bucket cache
   size. Reverse-order warmup is inferred to ~double the artifacts;
   we didn't quantify.
2. **`prefill_chunk_size ≤ sliding_window` is mandatory on hybrid
   attention.** On Llama (no sliding window) chunk=1024 beats
   chunk=512 by ~13 %. On Gemma-4-E4B (sliding_window=512, 35/42
   layers sliding), chunk=1024 loses by 30 % at the 8K bucket because
   per-chunk attention becomes "rolling" inside the chunk and
   Inductor produces 6 artifacts instead of 401.
3. **Cache-pinning has a real decode cost on the small bucket.**
   1024-prompt request with cache pinned at 8192+128 decodes at
   52 tok/s; pinned at 1024+128 it decodes at 115 tok/s — **2.2×
   slowdown** from over-provisioning. If prompt-length distribution is
   bimodal, run two compiled processes, not one shared.
4. **`mode="default"` is the production choice.** Autotune wins
   8–9 % TTFT on Llama for 3× the warmup; on Gemma it wins ≤ 1.4 % at
   1.5× the warmup, because attention (cuDNN/efficient-SDPA) dominates
   and isn't autotuned by Inductor.
5. **The generate()-loop convenience-tensor cost is ~1.1 ms/step on
   CUDA** (~14–17 % of decode wallclock on a 1B model). Not a
   recompile; just `torch.cat` of growing tensors + the rebuild of the
   4D causal mask per step. Removing it (pre-allocate, fill in-place,
   call the compiled forward directly) bought 127 → 149 tok/s.
   See [evidence/decode_overhead.py](evidence/decode_overhead.py).

## Worth flagging upstream

1. **`cache_implementation="static"` is a TTFT-bomb-shaped default.**
   The auto-cache reallocates on any `max_length` growth and there's
   no documented escape. The DIY-StaticCache + `past_key_values=`
   pattern works but conflicts with `cache_implementation` at runtime
   ([generation/utils.py:1822](src/transformers/generation/utils.py#L1822)),
   so users discover the conflict before discovering the workaround.
   Two fixes: (a) doc note pointing to the DIY pattern under the
   `cache_implementation` docstring entry, or (b) honor
   `cache_config["max_cache_len"]` in `_prepare_static_cache` (3-line
   change).
2. **`prefill_chunk_size` is undocumented public API.** Defined via
   `kwargs.pop` at
   [configuration_utils.py:465](src/transformers/generation/configuration_utils.py#L465)
   and not mentioned in the class docstring. The user can only learn
   about it by reading
   [generation/utils.py:3766](src/transformers/generation/utils.py#L3766).
3. **No bucket-padding helper.** The whole point of buckets is
   recompile-free dispatch, and that requires exact-shape input. Every
   team using this pattern rolls their own pad-to-next-bucket loop.

## Caveats and what we didn't test

- **Left-pad output quality**: not end-to-end tested. The bench uses
  random BOS-prefilled ids; before serving real traffic, verify that
  the model produces the same tokens with and without padding.
- **`fullgraph=True` with sdpa**: not tested (`_can_compile_fullgraph`
  is set on both Llama and Gemma).
- **Batch > 1**: out of scope — multi-request batching is
  `generate_batch`'s problem.
- **Attention-mask shape variation within a fixed bucket**: not
  tested. The recipe pads, so the 2D mask shape is fixed.
- **Decode tok/s** is measured from a single full-decode call per
  cell, not averaged, so cross-cell decode comparisons within ~5–10 %
  should be taken with a grain of salt. The TTFT numbers are from
  N=10 calls per cell and are tight.

## Reproduce

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -e /path/to/transformers accelerate
```

Then run any of the standalone evidence scripts (each is
self-contained and prints its result to stdout), or [bench.py](bench.py)
for the full warmup + steady-state sweep:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct --mode default \
    --input-lens 1024 8192 --chunk-sizes 512 1024 \
    --steady-calls 10 --decode-sanity-tokens 128 \
    --cache-root /tmp/inductor-bench --run-tag llama
```

Every script reads its own purpose from the first ten lines of its
file. Sweep JSON outputs and per-cell warmup logs are in `logs/`;
see [results.md](results.md) for the full per-cell numbers.
