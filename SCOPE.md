# Scope — transformers generation × torch.compile throughput

How do you configure `model.generate()` so an agentic loop on
transformers + `torch.compile` stays predictable and avoids the
worst surprises (silent recompiles, eager fallbacks, CUDA Graphs
clashes)? The goal is to identify optimizations but not necessarily
to chase AOT-compiler parity.

## Load-bearing assumption: no prefix caching

transformers has no prefix caching — no detection that this turn's
prompt shares a prefix with the previous turn. Every agent step
pays the full prefill cost. With prefix caching (vLLM, SGLang, TGI,
JetStream), the second turn onward prefills only the delta and most
of this experiment becomes irrelevant. We're characterizing the
no-prefix-caching path because that's what transformers offers today.

## Properties we'd like to approximate

The properties that would matter most for an agentic loop on transformers:

| Property | Why it matters for agentic | Cheap to approximate? |
|---|---|---|
| **No silent recompile.** Every step uses an already-compiled kernel. | A recompile mid-loop spikes one step's TTFT by seconds. | Probably — depends on whether static-cache shape pinning is reliable across calls. |
| **Chunked prefill on long inputs.** Splits the prefill compute across smaller forward passes. | Cuts peak memory; enables compile when full prefill won't fit. | Already a config knob. Just need to find the right chunk size. |
| **Deterministic TTFT.** Low variance across calls. | Lets the agent's wall-clock be predictable, not "usually fast, sometimes 10× slower". | Falls out of the above if they work. |

## What we already know

- `cache_implementation="static"` is what triggers transformers'
  automatic compile path. Without it, `generate()` runs eager.
- The default mode for the static-cache compile path
  (`reduce-overhead`) uses CUDA Graphs. CUDA Graphs **conflict** with
  chunked prefill — the shared output buffer gets overwritten across
  chunks and you get "accessing tensor output of CUDAGraphs that has
  been overwritten" at runtime. With chunked prefill on, the only
  usable modes are `default` and `max-autotune-no-cudagraphs`.
- Compile time scales with the number of unique kernel shapes
  Inductor sees. Chunk size, batch size, input length, and dtype each
  multiply the shape set. Static cache pins KV layout so the decode
  loop shouldn't add new shapes per step.

## Parameters

### Held fixed

- `cache_implementation="static"` — required for the compile path.
- `dtype=bfloat16` — what modern models support best.
- `batch_size=1` — agentic is per-request. Batching is
  `generate_batch`'s concern; separate experiment.
- Chunked prefill **on** — the regime we care about.

### Swept

| Knob | Values | Why |
|---|---|---|
| `prefill_chunk_size` | 512, 1024 | Each is a distinct compiled prefill shape. TTFT vs memory. |
| `CompileConfig.mode` | `default`, `max-autotune-no-cudagraphs` | The two CUDA-Graphs-free modes. Trade warmup time for steady-state speed. |
| `input_len` (bucket) | 1024, 8192 | The bucket set. Recompile-free dispatch across buckets is one of the things we want to verify. |

## Metrics

| Metric | Definition |
|---|---|
| **Steady-state TTFT** | Median TTFT across N≥10 calls after warmup, per bucket. The headline. |
| **Warmup time** | Wallclock to compile the full bucket set up front. The price of admission. |
| **Recompile-free hot path** | Number of new Inductor cache dirs created during the steady-state calls. Want 0. |
| **TTFT variance** | p99 / p50 of steady-state TTFT. Tight = predictable. |
| **Decode throughput** | Tokens/sec on the (short) decode phase. Sanity check, not headline. |

If recompile-free isn't zero, the steady-state TTFT numbers
underestimate real-world variance — that cell isn't deployable
without finding the cause.

## Test plan

Two models:
- `meta-llama/Llama-3.2-1B-Instruct`
- `google/gemma-4-E4B-it`

Per model:

**Phase A — Warmup.** Compile the configurations set:
2 input lengths × 2 chunk sizes × 2 modes = 8 unique compile shapes.
Record warmup wallclock.

**Phase B — Steady state.** 10 `generate()` calls per cell, batch=1,
4096-token input padded down to the bucket, 128 decode tokens,
randomized input content within the bucket. Record TTFT per call,
measure cache-dir growth.

**Phase C — Find cheap wins.** For each cell where steady-state TTFT
is good *and* recompile-free passes: that's a recipe. For each cell
that fails: identify the cause and classify the fix:
- Config workaround (set X to Y).
- Documentation gap (knob exists, no one knows).
- Upstream change needed — note it, don't pursue.

## Open questions

- Does `cache_implementation="static"` recompile on attention-mask
  shape variation across calls, even when input length matches a
  bucket?
- Can `default` mode (no CUDA Graphs) match
  `max-autotune-no-cudagraphs` on steady-state TTFT, or does
  autotuning materially help on prefill kernels?
- Do hybrid / sliding-window cache implementations expose the same
  bucket-friendly behavior as `"static"`?

## Deliverable

A short doc with:

1. **Recipe**: minimum viable config for agentic on transformers,
   with the bucket sizes we tested.
2. **Cheap wins**: config changes that moved a metric meaningfully.
3. **Worth-flagging-upstream**: gaps we can't fix in config but
   would matter if closed. Brief, not a roadmap.
