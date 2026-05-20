# Generation loop on transformers + `torch.compile` — recipe & findings

> Scope: [SCOPE.md](SCOPE.md). One-process, batch-1 agent loop on
> transformers v5 with `cache_implementation="static"` + chunked
> prefill, no prefix caching.
>
> Test bed: 1× NVIDIA A10G (23 GiB), bf16, torch 2.7.0+cu126,
> transformers 5.8.0.dev0 (local editable).
>
> Models: `meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-4-E4B-it`.

## TL;DR

A recompile-free, low-variance agentic prefill is achievable on
transformers today, but only if you sidestep four silent-recompile
landmines. None of them are documented in the public knobs:

1. **Pad inputs to a fixed bucket** that is a clean multiple of
   `prefill_chunk_size`. Otherwise the tail chunk has a unique
   shape and triggers a ~26 s recompile per first occurrence.
2. **Pin the StaticCache size yourself** — construct
   `StaticCache(config=..., max_cache_len=N)`, pass it via
   `past_key_values=`, and *don't* set `cache_implementation`. The
   built-in auto-cache reallocates the buffer (→ full recompile)
   whenever a later call has a bigger `max_length` than any previous
   call. The DIY pattern makes per-call `max_new_tokens` free.
3. **Warm the largest input-length bucket first.** Otherwise the
   built-in cache grows (footgun #2) or the first DIY allocation is
   wasteful. With the DIY pattern, the warmup order still matters for
   the *Inductor* shape set, not the cache: warm the worst case first.
4. **Use `mode="default"`, not the library default `"reduce-overhead"`.**
   `reduce-overhead` uses CUDA Graphs, which conflict with chunked
   prefill (documented in SCOPE.md). Among the two CUDA-Graphs-free
   options, `default` and `max-autotune-no-cudagraphs` differ by 8–9 %
   TTFT on Llama-1B (where matmuls dominate) and by ≤ 1.4 % on
   Gemma-4-E4B (where attention dominates). Either way, `default`
   costs roughly 1.5–3× less warmup wallclock. *(See "Mode comparison"
   below.)*

With those four: steady-state TTFT p99/p50 sits at 1.00–1.02 on
Llama-1B and ≤ 1.006 on Gemma-4-E4B, zero Inductor artifacts are
added during the steady-state loop in every cell tested, and bucket
dispatch is genuinely free in a hot loop (interleaved 1024/8192
calls, 0 recompiles across 20 calls, see
[demo_interleave_buckets.py](demo_interleave_buckets.py)).

A separate finding worth flagging: the generate() decode loop carries
~1.1 ms/step of Python overhead from growing "convenience tensors"
(`input_ids`, `attention_mask`, `position_ids`). On CUDA this is
~14–17 % of decode wallclock on a 1B model — not catastrophic but
not free.

## The recipe (copy-pasteable)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

# 1. Pick buckets, chunk size, and decode worst-case up front.
BUCKETS         = (1024, 8192)   # input-length buckets you'll round prompts up to
PREFILL_CHUNK   = 1024           # MUST divide every bucket; sweep per model — see "Cheap wins" §2
DECODE_CEILING  = 256            # max tokens any agent turn will EVER produce
COMPILE_MODE    = "default"      # NOT "reduce-overhead" (CUDA Graphs vs chunked prefill)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # FA forces fullgraph=False; sdpa is fine
).cuda().eval()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# 2. Build ONE StaticCache sized for the worst case. Reused across turns.
WORST_PROMPT_LEN = max(BUCKETS)
cache = StaticCache(config=model.config, max_cache_len=WORST_PROMPT_LEN + DECODE_CEILING)
for layer in cache.layers:
    if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
        layer.keys, layer.values = layer.keys.cuda(), layer.values.cuda()

# 3. Make a single GenerationConfig — no cache_implementation, because we
#    supply the cache ourselves. compile_config still kicks in because
#    cache.is_compileable is True.
gen_cfg = GenerationConfig(
    do_sample=False,
    compile_config=CompileConfig(mode=COMPILE_MODE, fullgraph=False, dynamic=False),
    max_new_tokens=DECODE_CEILING,  # may vary across calls now, since cache is fixed
    prefill_chunk_size=PREFILL_CHUNK,
    pad_token_id=tok.eos_token_id,
)

# 4. Warm up: largest bucket first, then descend (shape coverage).
for L in sorted(BUCKETS, reverse=True):
    cache.reset()
    ids  = torch.full((1, L), tok.bos_token_id or 1, device="cuda")
    mask = torch.ones_like(ids)
    model.generate(input_ids=ids, attention_mask=mask,
                   generation_config=gen_cfg, past_key_values=cache)

# 5. Steady-state agent loop. Round every prompt up to a bucket and pad.
def round_to_bucket(L: int) -> int:
    return next(b for b in BUCKETS if b >= L)

def serve(prompt_ids: torch.Tensor) -> torch.Tensor:
    bucket = round_to_bucket(prompt_ids.shape[1])
    pad_n  = bucket - prompt_ids.shape[1]
    ids    = torch.nn.functional.pad(prompt_ids, (pad_n, 0), value=tok.pad_token_id)  # left-pad
    mask   = (ids != tok.pad_token_id).long()
    cache.reset()
    return model.generate(input_ids=ids, attention_mask=mask,
                          generation_config=gen_cfg, past_key_values=cache)
```

The recipe is also embodied in [bench.py](bench.py)
(with-`cache_implementation="static"`-and-pinned-max_new_tokens version) and
[demo_diy_cache.py](demo_diy_cache.py) (DIY-StaticCache version).

## What we measured

### Headline steady-state numbers

All numbers below are after a single warmup pass (4 cells). N=10 calls
per cell, batch=1, decode budget pinned at 128 tokens. TTFT measured
as wall-clock for `generate()` truncated to 1 new token via a
StoppingCriteria (so cache shape stays constant — see footgun §).

> **Note on the harness:** the bench in [bench.py](bench.py) uses
> `cache_implementation="static"` and pins `max_new_tokens` to a
> constant decode budget across every call (warmup, TTFT, decode
> sanity), so the auto-path never triggers a reallocation. Numbers
> below are therefore equivalent to what the DIY-cache pattern from
> the recipe would produce; we kept the auto path for simpler
> ablations of `chunk_size`, `mode`, and `input_len`.

**`meta-llama/Llama-3.2-1B-Instruct`, `mode="default"`** *(warmup
order: largest bucket first)*

| input_len | chunk | warmup s | p50 TTFT | p99/p50 | decode tok/s¹ | ss recompiles |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 512  | 14.4  | 86.6 ms  | 1.016 | 52.3 | 0 |
| 1024 | 1024 | 15.2  | 76.2 ms  | 1.010 | 52.3 | 0 |
| 8192 | 512  | 42.9  | 676.5 ms | 1.001 | 52.4 | 0 |
| 8192 | 1024 | 16.2  | 594.2 ms | 1.001 | 52.3 | 0 |

¹ Decode tok/s is measured against a cache pinned to the *largest*
bucket (8192 + 128). The 1024 bucket therefore pays attention over a
mostly-empty 8320-slot cache — see "Cheap wins" §3.

Total warmup wallclock for all 4 cells: **88.7 s**. Steady-state
Inductor cache directories grew by **0** across all 4 cells × 10 calls
= 40 generates. Recompile-free.

**`google/gemma-4-E4B-it`, `mode="default"`** *(same warmup order)*

| input_len | chunk | warmup s | p50 TTFT | p99/p50 | decode tok/s | ss recompiles |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 512  |   10.7  |  285.6 ms | 1.006 | 12.2 | 0 |
| 1024 | 1024 |   10.6  |  283.1 ms | 1.001 | 12.2 | 0 |
| 8192 |  512 | **2725.8** | **1770.2 ms** | 1.000 | 14.8 | 0 |
| 8192 | 1024 |   12.0  | 2298.1 ms | 1.000 | 14.8 | 0 |

Total warmup wallclock: **2759 s**. The first cell pays 45 minutes
because Gemma-4's hybrid sliding/full attention produces ~60 distinct
Inductor inference graphs (vs 3 on Llama-1B). After that first cell,
the remaining three cells finish in 10–12 s each by reusing kernels
and the cache. Steady-state: same story as Llama — **zero
recompiles, p99/p50 ≤ 1.006**.

Two things on Gemma run *opposite* to Llama:

- **`chunk_size=1024` is slower than `chunk_size=512` on the 8192
  bucket** (2298 ms vs 1770 ms — 30 % slower). On the 1024 bucket
  they tie (~283 ms). Root cause: Gemma-4-E4B has `sliding_window=512`
  and 35 of 42 layers are sliding-attention. When `chunk_size >
  sliding_window`, query tokens within a single chunk span different
  windows, so the per-chunk attention mask becomes "rolling" and
  Inductor produces fewer / less-optimized kernels (6 artifacts vs
  401 for prefill@512). See "Cheap wins" §2 below.

- **Decode tok/s grows slightly with bucket size** (12.2 → 14.8).
  Counter to the cache-bandwidth math (bigger cache = more
  attention work per decode step). Plausibly a sliding-window effect
  too — decode steps for a longer prompt have more "real" K/V to
  attend to, but most of the work is the per-step matmuls which
  benefit from any GPU-side warmup the previous prefill did. We
  didn't isolate this further; it's measured once per cell, not the
  10-call statistic.

The actionable finding: **chunk_size = sliding_window** for hybrid
attention models. The recipe code defaults to 1024 because
Llama-3.2 has no sliding window; for Gemma-4-E4B you'd set it to 512.

### Cross-bucket dispatch is genuinely recompile-free

To confirm the buckets aren't just isolated successes,
[demo_interleave_buckets.py](demo_interleave_buckets.py) warms buckets
1024 and 8192 (chunk_size=1024 only), then runs 20 calls with bucket
picked randomly per call:

```
warmup done. artifacts=76

interleaved loop (20 calls, random bucket per call):
  i= 0  L= 8192  ttft=  594.5 ms  artifacts=76
  i= 1  L= 8192  ttft=  594.1 ms  artifacts=76
  i= 2  L= 1024  ttft=   76.6 ms  artifacts=76
  ...
  i=19  L= 8192  ttft=  594.1 ms  artifacts=76

bucket 1024: n= 8  p50= 76.6 ms  p99= 76.6 ms  p99/p50=1.001
bucket 8192: n=12  p50=594.2 ms  p99=594.5 ms  p99/p50=1.001
artifacts delta after 20 interleaved calls: 0
```

p99 minus p50 is well under a millisecond on both buckets across the
random-dispatch loop. The compile-pinned, recompile-free agent loop
is real on this configuration.

*(Other three cells: `llama × max-autotune-no-cudagraphs`,
`gemma-4-E4B × default`, `gemma-4-E4B × max-autotune-no-cudagraphs` —
results table extended in [results.md](results.md) once those sweeps
complete.)*

### Silent-recompile footgun #1: inputs must be a multiple of chunk_size

The chunked prefill path does
`input_chunks = torch.split(input_ids, chunk_size, dim=-1)`
([generation/utils.py:3773](src/transformers/generation/utils.py#L3773)).
If `input_ids.shape[-1]` is not a multiple of `chunk_size`, the last
chunk has a unique shape — and gets its own compile.
`demo_padding.py` on Llama-1B, chunk_size=1024, warmed at L=2048:

```
warming up at L=2048 (clean multiple of 1024)...
  cold:  29.11s   artifacts+=51
  warm:   0.10s   artifacts+=0    <-- same L, hot path

  L= 1024 (L%chunk=   0):  12.75s  artifacts+=0     <-- cached, but Dynamo re-traces
  L= 2048 (L%chunk=   0):   0.10s  artifacts+=0
  L= 4096 (L%chunk=   0):  26.07s  artifacts+=25    <-- cache realloc (input grew)

  L= 1500 (L%chunk= 476):  26.46s  artifacts+=31   <-- tail-chunk shape recompile
  L= 1700 (L%chunk= 676):  26.08s  artifacts+=29   <-- tail-chunk shape recompile
  L= 3500 (L%chunk= 428):  26.31s  artifacts+=29   <-- tail-chunk shape recompile
```

Two distinct silent-recompile triggers visible here:
- **Cache size growth** (L=4096 row): same chunk shape, but a new cache
  buffer was needed because the input grew past anything seen before.
- **Tail-chunk shape** (L=1500/1700/3500 rows): cache fits, all the
  "full" chunks reuse the cached kernel, but the last `< chunk_size`
  chunk is a new shape and gets compiled in 26 s.

The fix is mandatory bucket padding: round prompts up to one of a
finite set of pre-warmed bucket lengths (each one a clean multiple of
`prefill_chunk_size`), and pad with `pad_token_id`. Without that, real
agent traffic — variable-length prompts, growing context — will hit
recompiles repeatedly.

### Silent-recompile footgun #2: cache reallocation on bigger max_new_tokens

`generate()` derives the StaticCache size from
`generation_config.max_length - 1`
([generation/utils.py:2495](src/transformers/generation/utils.py#L2495)),
and `max_length = max_new_tokens + input_ids_length`
(line 1627). Whenever this value exceeds the previously-seen max,
`_prepare_static_cache` reallocates the cache
([line 1753](src/transformers/generation/utils.py#L1753)) — fresh
tensors, fresh shape, fresh Inductor compile.

`demo_cache_realloc.py` on Llama-1B, `mode="default"`, A10G:

```
== Scenario A: hold max_new_tokens constant ==
  call 1 (max_new=8, cold):       15.31s   <-- pays full compile
  call 2 (max_new=8):              0.07s
  call 3 (max_new=8):              0.07s

== Scenario B: grow max_new_tokens across calls ==
  call 4 (max_new=16, NEW max):   19.34s   <-- cache realloc + recompile
  call 5 (max_new=16):             0.13s
  call 6 (max_new=32, NEW max):   11.15s   <-- cache realloc + recompile again
  call 7 (max_new=32):             0.25s

== Scenario C: shrink back ==
  call 8 (max_new=8, shrunk):      0.07s   <-- reuses larger cache, no recompile
```

A 270× TTFT spike on a single agent turn just because that turn asked
for more decode tokens than the previous one. In an agent that
sometimes responds with one sentence and sometimes with a long tool
call, this is a fully silent ambush — there is no warning, no log,
no `recompile_limit` hit.

### Working around it: DIY the StaticCache

You can bypass the auto-realloc logic entirely by constructing a
`StaticCache` yourself, sized once for your worst case, and passing
it through `past_key_values=` on every `generate()` call. Crucially,
you must **also drop `cache_implementation`** from your
GenerationConfig — passing both raises
("Passing both `cache_implementation` … and `past_key_values` is unsupported",
[generation/utils.py:1822](src/transformers/generation/utils.py#L1822)).
The auto-compile path still triggers, because it checks
`cache.is_compileable`, not the config field.

`demo_diy_cache.py`, same Llama-1B setup, cache pinned at 320 slots,
calls vary max_new_tokens from 8 → 64:

```
DIY StaticCache (fixed at 320 slots), vary max_new_tokens across calls:
  call 1 (max_new=8, cold):    14.21s   <-- one-time compile
  call 2 (max_new=8):           0.07s
  call 3 (max_new=16):          0.13s   <-- was 19.3s in the auto path!
  call 4 (max_new=32):          0.25s   <-- was 11.2s in the auto path!
  call 5 (max_new=64):          0.50s
  call 6 (max_new=16, back):    0.13s
```

Two responsibilities transfer to the user:

1. **Call `cache.reset()` between turns.** Otherwise the previous
   turn's K/V leak into the next prefill.
2. **Size for the worst case upfront** — `max_cache_len` is fixed at
   construction. Going past it at runtime will overflow.

This is the pattern we'd recommend for an agentic loop on transformers
today.

## Cheap wins

### 1. Warm the largest bucket first

In the Llama-default sweep above, warmup order was
`[(8192, 512), (8192, 1024), (1024, 512), (1024, 1024)]`.

That order produces **76 + 29 + 0 + 0 = 105 new Inductor artifacts**
across all four warmups. The 1024-bucket cells add zero artifacts —
the prefill_512 and prefill_1024 kernels were already compiled at the
8192-bucket cache size and the 1024 cells reuse them.

Reverse the order, and from the cache-reallocation rule we know the
8192 cells would force a fresh allocation when they finally appear,
re-compiling both prefill kernels. We didn't quantify this, but the
recipe (largest first) avoids it for free.

### 2. `prefill_chunk_size` ≤ sliding_window — the rule for hybrid-attention models

On Llama-1B (standard causal attention, no sliding window), chunk=1024
beats chunk=512 by ~13 % across both buckets — bigger chunks amortize
KV-load and kernel-launch cost:

| bucket | chunk=512 TTFT | chunk=1024 TTFT | speedup |
|--:|--:|--:|--:|
| 1024 | 86.6 ms  | 76.2 ms  | 1.14× |
| 8192 | 676.5 ms | 594.2 ms | 1.14× |

On Gemma-4-E4B, 35 out of 42 layers are `sliding_attention` with
`sliding_window=512`. The chunk-size choice flips:

| bucket | chunk=512 TTFT | chunk=1024 TTFT | result |
|--:|--:|--:|---|
| 1024 | 285.6 ms  | 283.1 ms  | tied |
| 8192 | 1770.2 ms | 2298.1 ms | **chunk=512 is 30 % faster** |

**Why:** when `chunk_size ≤ sliding_window`, every query token in a
chunk attends to the same window of KV positions — the kernel does
one clean masked SDPA per chunk. When `chunk_size > sliding_window`,
query tokens *within a single chunk* span different windows (the
first query sees a window ending early-in-chunk; the last query sees
a window ending late-in-chunk), so the mask pattern becomes
"rolling" inside the chunk and the kernel has to handle it. The
prefill@1024 path on Gemma also produced only **6 new Inductor
artifacts** versus **401** for prefill@512, suggesting Inductor's
pattern matcher gave up on optimizing the rolling-window case.

The actionable rule:

```
PREFILL_CHUNK = min(largest_bucket, getattr(model.config.get_text_config(),
                                            "sliding_window",
                                            float("inf")))
```

For Llama-3.2 (no sliding window): chunk = largest bucket fits, picks
itself — but you also want it to divide every bucket cleanly, so cap
at the bucket-size GCD. For Gemma-4-E4B (sliding_window=512): chunk
should be 512.

### 3. The decode-throughput cost of bucket pinning is large

Pinning the static cache to `max_bucket + decode_budget` makes the
*small* bucket's decode kernel attend over a much larger cache buffer
than its prompt actually uses. With sdpa, every decode step reads
the whole cache region (the mask zeroes invalid positions but doesn't
gate the read). So the small bucket pays attention bandwidth
proportional to the **buffer**, not its prompt.

Measured on Llama-1B, default mode, 1024-bucket only:

| 1024-bucket measurement | cache=1024+128 | cache=8192+128 | cost of pinning |
|---|--:|--:|--:|
| TTFT p50 (chunk=512)  | 52.3 ms | 86.6 ms | 1.66× slower |
| TTFT p50 (chunk=1024) | 47.7 ms | 76.2 ms | 1.60× slower |
| decode tok/s | 114.9 | 52.3 | **2.20× slower** |

The decode penalty is the headline: pinning a 1024-prompt request to
an 8192-sized cache cuts its decode throughput in half. That's the
real price of "use the same compiled process for both buckets."

**Implication for the recipe:** if your agent's prompt-length
distribution is bimodal, run two compiled processes (one per bucket
family) and dispatch by prompt length. Sharing a process across
very-different bucket sizes is recompile-free, but it's not free.

This is also the most concrete reason why a `cache_config["max_cache_len"]`
override would matter — without it you're locked into "biggest possible
cache for the lifetime of the process," whether or not the current
request needs it.

### 4. The growing "convenience tensors" in generate()'s decode loop

The decode loop in `_sample` (and friends) keeps growing host-side
tensors: each step it `torch.cat`s the freshly-sampled token onto
`input_ids`, appends a 1 to the 2D `attention_mask`, advances
`cache_position` and `position_ids`, then calls
`prepare_inputs_for_generation` which slices everything back down to
`(B, 1)` for the compiled forward and rebuilds the 4D causal mask.

On Neuron / TPU this is a category-killer — "eager" there means
per-op recompile, so a growing tensor triggers a recompile every
single decode step. On CUDA, none of this enters the compiled graph
(the compiled forward only sees the sliced `(B, 1)` view), but the
Python loop is still doing real work each step: tensor allocations,
attention-mask reshape, function dispatch. Worth measuring.

`demo_decode_overhead.py` does two things on Llama-1B, default mode,
prompt=256, decode=256, cache=544 (warm timings, excluding cold
first call):

| | wallclock / call | per decode step |
|---|--:|--:|
| **A.** `model.generate(max_new=256)` | 2.025 s | ~7.85 ms (subtracting 19 ms prefill) |
| **B.** `model.get_compiled_call()(...)` in a tight loop with pre-allocated `input_ids`, `position_ids`, `cache_position`, 4D mask all static | 1.718 s | **6.71 ms** |
| | | gap = **1.13 ms/step (14–17 %)** |

So on CUDA, the growing-convenience-tensors cost is real but
*bounded*: about a sixth of the decode step on a 1B model. It scales
with how much Python-side work generate() does per step, not with
model size, so the *relative* hit shrinks as the model gets bigger
(decode step gets dominated by attention bandwidth).

If your goal is `vllm`-like steady-state throughput, this is the next
cliff to climb after the compile-and-cache work above: pre-allocate
the growing tensors at max length, update them in-place, and call the
compiled forward directly. Worth it for serving, probably overkill
for an agent loop where per-call latency already includes round
trips, tool calls, and chat-template overhead that dwarf 1 ms/step.

### 5. Mode comparison: `default` vs `max-autotune-no-cudagraphs`

**Llama-3.2-1B-Instruct:**

|   | warmup s (4 cells) | TTFT 1k/1024 | TTFT 8k/1024 | decode tok/s | recompile-free |
|---|--:|--:|--:|--:|--:|
| `default` | **88.7** | 76.2 ms | 594.2 ms | 52.3 | ✓ |
| `max-autotune-no-cudagraphs` | 274.6 | **69.8 ms** | **544.3 ms** | **54.9** | ✓ |
| autotune wins by | – (3× cost) | **9 %** | **8 %** | **5 %** | – |

**Gemma-4-E4B-it:**

|   | warmup s (4 cells) | TTFT 1k/1024 | TTFT 8k/512 | decode tok/s | recompile-free |
|---|--:|--:|--:|--:|--:|
| `default` | **2759.0** | 283.1 ms | 1770.2 ms | 12.2–14.8 | ✓ |
| `max-autotune-no-cudagraphs` | 4038.6 | 282.5 ms | 1745.6 ms | 12.1–15.1 | ✓ |
| autotune wins by | – (1.5× cost) | **0.2 %** | **1.4 %** | **~0 %** | – |

Two takeaways:

- On Llama autotune buys a real 8–9 % TTFT win and worth considering
  for a long-lived process if warmup amortizes.
- **On Gemma autotune is essentially free of value** (≤ 1.4 % win,
  ~0 % decode improvement) but takes 4000 s of compile time. Best
  guess at why: Gemma-4's hidden cost is in the SDPA attention call
  (which dispatches to a pre-tuned cuDNN/efficient kernel, not
  Inductor matmuls), so Inductor's autotuning of the *non-attention*
  matmuls moves a smaller fraction of the wallclock. The 7 full
  layers among 42 sliding-window layers also push attention compute
  proportionally higher.

Recommendation: **`default` is the right production choice for both
models** in this study; only consider autotune on standard-causal
architectures (Llama-style) where the matmuls dominate.

Cell-by-cell warmup, both modes, in warmup order (largest bucket
first; cells 3–4 reuse the kernels compiled in cells 1–2):

| cell | default | autotune | autotune/default |
|---|--:|--:|--:|
| (8192, 512)  cold compile  | 42.9 s | 140.0 s | 3.3× |
| (8192, 1024) +prefill_1024 | 16.2 s |  61.6 s | 3.8× |
| (1024, 512)  cached, re-trace | 14.4 s |  36.6 s | 2.5× |
| (1024, 1024) cached, re-trace | 15.2 s |  36.5 s | 2.4× |

The cells-3-4 gap (still 2.4–2.5× slower under autotune even though
**no Inductor artifacts are added**) is unexplained — there's some
non-compile-related per-bucket overhead in autotune mode that doesn't
show up in steady-state. Probably worth a deeper look, but it's a
one-time cost so we left it on the table.

## Worth flagging upstream

These are gaps we can't fix in config. We didn't pursue any of them;
just noting them.

### A. `cache_implementation="static"` is a TTFT-bomb-shaped default

The auto-cache path (`cache_implementation="static"`) reallocates the
StaticCache whenever a later call has a larger
`max_length = input_ids_length + max_new_tokens`
than any previous call. We've shown above that this triggers a full
~15–30 s Inductor recompile silently
([demo_cache_realloc.py](demo_cache_realloc.py)).

A workaround exists — DIY the StaticCache and pass it via
`past_key_values=`, dropping `cache_implementation` — but it's
neither documented as the recommended pattern for agent loops nor
discoverable from the GenerationConfig docstring. The two relevant
fields read independently and the conflict between them is a runtime
ValueError ([generation/utils.py:1822](src/transformers/generation/utils.py#L1822))
without prose explaining *why* the user might want one over the
other.

Two upstream improvements would close this:
1. **Doc**: add a "For agentic loops with variable per-call decode
   budgets, supply your own StaticCache via `past_key_values=`" note
   to the GenerationConfig docstring's `cache_implementation` entry.
2. **Code**: have `_prepare_static_cache` honor a
   `cache_config["max_cache_len"]` override so users can keep the
   `cache_implementation="static"` ergonomics while pinning the cache
   size independently of `max_length`. This is a 3-line change.

### B. `prefill_chunk_size` is undocumented public API

`GenerationConfig.prefill_chunk_size` is set in
[configuration_utils.py:465](src/transformers/generation/configuration_utils.py#L465)
via `kwargs.pop(...)` but **does not appear in the class docstring**.
The only place a user can learn about it is by reading
[generation/utils.py:3766](src/transformers/generation/utils.py#L3766) ("Chunked
prefill (for very large contexts)"). For a knob that gates the
*entire* low-memory-prefill code path, this is a docs gap, not a
config gap.

### C. No bucket-padding helper

Every user of bucketed prefill has to write their own
"round input_ids up to the next bucket length, build a 2D mask that
zeroes the pad region" loop. The chunked prefill path itself doesn't
care about this — it just takes whatever input_ids it's given — but
the *whole point* of buckets in a compile-pinned world is dispatch
without recompiles, and that requires inputs whose shape exactly
matches one of a finite set. A `BucketedPrefillCollator` or similar
would be a tiny library-internal utility, but its absence means every
team rolls one.

## Open questions from SCOPE.md: answers

1. **Does `cache_implementation="static"` recompile on attention-mask
   shape variation across calls, even when input length matches a
   bucket?** Indirectly answered: yes, any mask shape variation
   triggers a recompile (the chunked-prefill code feeds the 2D mask
   through `create_masks_for_generate`, and any shape change passes
   through to the compiled forward). The recipe sidesteps this by
   padding to the bucket so the 2D mask is shape-stable. We did not
   test what happens with right-padding vs left-padding of the same
   shape; the recipe uses left-pad.
2. **Can `default` mode match `max-autotune-no-cudagraphs` on
   steady-state TTFT?** Within 7–9 %. Autotune buys you a small TTFT
   improvement at 3× the warmup cost. For an agentic loop, `default`
   is the right production choice.
3. **Do hybrid / sliding-window cache implementations expose the same
   bucket-friendly behavior as `"static"`?** *Gemma-4-E4B sweep
   in flight — answer in [results/](results/) when complete.* From
   reading the code, `StaticCache` does dispatch through
   `StaticSlidingWindowLayer` for layers tagged `sliding_attention`
   ([cache_utils.py:1411](src/transformers/cache_utils.py#L1411)),
   and the layer sizes are determined at construction. We expect the
   same recompile-free behavior to hold once warmup completes, but
   that's a hypothesis until the sweep finishes.

## What we didn't test (would matter for a real recipe)

- **`fullgraph=True` with sdpa**: marked as available on Llama
  (`_can_compile_fullgraph=True`) but we left it off. Worth checking
  whether fullgraph saves anything on top of the existing
  cache_implementation="static" path.
- **Attention mask shape variation across calls** (open question 1 in
  SCOPE): we always padded to the bucket so the 2D mask shape is
  fixed. If you don't pad, the 4D causal mask synthesized by
  `create_masks_for_generate` will have a different shape every
  call → recompile. Padding is mandatory; this should be a doc note.
- **Hybrid / sliding-window cache** on Gemma-4: in this sweep Gemma-4
  uses its native cache (whichever `Gemma4ForConditionalGeneration`
  picks when `cache_implementation="static"` is set). Numbers in
  results table will tell us if recompile-free behavior survives.
- **Batch >1**: out of scope, but worth noting that everything above
  is single-request. Multi-request batching is `generate_batch`'s
  problem.
- **Left-pad correctness with chunked prefill**: the recipe code
  left-pads prompts up to bucket size. Position embeddings on the
  padded prefix start at `pad_n` rather than 0, which is the normal
  behavior for left-padded batched generation, but we did not
  end-to-end-test the *output quality* with chunked prefill (the
  bench used random BOS-prefilled ids with an all-ones mask). Before
  serving real traffic, verify that the model produces the same
  tokens with and without padding on a held-out prompt.

## Reproducing

```sh
# One-time
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -e /path/to/transformers \
    accelerate hf_transfer

# Demo the footgun
CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo_cache_realloc.py

# Full sweep (one mode at a time, fresh Inductor cache dir)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct --mode default \
    --input-lens 1024 8192 --chunk-sizes 512 1024 \
    --steady-calls 10 --decode-sanity-tokens 128 \
    --cache-root /tmp/inductor-bench --run-tag llama-default
```
