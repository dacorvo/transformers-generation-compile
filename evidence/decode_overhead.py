"""Measure generate()'s Python-loop overhead per decode step on CUDA.

WHAT: compare two paths producing 256 decode tokens from the same
compiled forward against the same StaticCache:
  (A) `model.generate(max_new_tokens=256)` — full generate() loop.
  (B) `model.get_compiled_call()(...)` in a tight loop with
      pre-allocated input_ids / position_ids / cache_position /
      4D attention mask. No `torch.cat`, no mask rebuild per step.

WHY:  on CUDA the growing input_ids/attention_mask don't enter the
compiled graph (the forward only sees a sliced (B, 1) view), but
generate()'s Python loop still runs `torch.cat`, mask reshape, and
prepare_inputs_for_generation per step. On Neuron/TPU the same
growing tensors trigger a recompile every step; CUDA only pays the
allocator/dispatch cost.

RESULT (A10G, Llama-3.2-1B, mode=default, prompt=256, decode=256):
  A = 7.85 ms / step (127 tok/s),
  B = 6.71 ms / step (149 tok/s),
  gap = 1.13 ms / step (~14 % of decode wallclock).

RUN: CUDA_VISIBLE_DEVICES=0 .venv/bin/python evidence/decode_overhead.py
"""
# ── env (must precede `import torch`) ──
import os
CACHE_DIR = "/tmp/inductor-overhead"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── imports ──
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig
from transformers.masking_utils import create_masks_for_generate

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

# ── config ──
MID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = torch.device("cuda:0")
PROMPT_LEN = 256
DECODE_STEPS = 256
CACHE_LEN = PROMPT_LEN + DECODE_STEPS + 32  # +32 slack
N_ITER = 5
PREFILL_EST_S = 0.052  # rough prefill subtraction in the (A) timing

# ── setup ──
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MID)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

# Pre-built prompt + StaticCache reused across both methods.
prompt_ids = torch.full((1, PROMPT_LEN), tok.bos_token_id or 1, device=DEVICE)
prompt_mask = torch.ones_like(prompt_ids)
cache = StaticCache(config=model.config, max_cache_len=CACHE_LEN)
for layer in cache.layers:
    if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
        layer.keys, layer.values = layer.keys.to(DEVICE), layer.values.to(DEVICE)

gen_cfg = GenerationConfig(
    do_sample=False,
    compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
    max_new_tokens=DECODE_STEPS, min_new_tokens=DECODE_STEPS,
    prefill_chunk_size=PROMPT_LEN,  # one prefill chunk
    pad_token_id=tok.eos_token_id,
)

# Warmup once so the compile cost doesn't leak into the timed runs.
print("warming up compile...", file=sys.stderr)
cache.reset()
model.generate(input_ids=prompt_ids, attention_mask=prompt_mask,
               generation_config=gen_cfg, past_key_values=cache)
torch.cuda.synchronize()
print("compile done.", file=sys.stderr)

# ── helpers ──
def time_generate(n_iter: int = N_ITER) -> list[float]:
    """Method A: full generate() loop. Includes one prefill per call."""
    times = []
    for _ in range(n_iter):
        cache.reset()
        torch.cuda.synchronize(); t = time.perf_counter()
        model.generate(input_ids=prompt_ids, attention_mask=prompt_mask,
                       generation_config=gen_cfg, past_key_values=cache)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return times

def time_raw_decode_static(n_iter: int = N_ITER) -> list[float]:
    """Method B: compiled forward called directly, with pre-built inputs.

    Decodes only — prefill is done outside the timed region.
    """
    compiled_call = model.get_compiled_call(gen_cfg.compile_config)
    one_tok_buf = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    pos_buf = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    cache_pos_buf = torch.zeros((1,), dtype=torch.long, device=DEVICE)
    mask_4d = create_masks_for_generate(
        config=model.config,
        inputs_embeds=torch.empty((1, 1, 0), dtype=torch.bfloat16, device=DEVICE),
        attention_mask=torch.ones((1, CACHE_LEN), dtype=torch.long, device=DEVICE),
        past_key_values=cache,
        position_ids=pos_buf,
    )
    times = []
    for _ in range(n_iter):
        cache.reset()
        # Prefill (outside the timed region).
        prefill_mask = create_masks_for_generate(
            config=model.config,
            inputs_embeds=torch.empty((1, PROMPT_LEN, 0), dtype=torch.bfloat16, device=DEVICE),
            attention_mask=torch.ones((1, CACHE_LEN), dtype=torch.long, device=DEVICE),
            past_key_values=cache,
            position_ids=torch.arange(PROMPT_LEN, device=DEVICE).unsqueeze(0),
        )
        compiled_call(
            input_ids=prompt_ids,
            attention_mask=prefill_mask,
            position_ids=torch.arange(PROMPT_LEN, device=DEVICE).unsqueeze(0),
            cache_position=torch.arange(PROMPT_LEN, device=DEVICE),
            past_key_values=cache,
            return_dict=True, use_cache=True,
        )
        # Timed decode loop — DECODE_STEPS calls with NO growing tensors.
        torch.cuda.synchronize(); t = time.perf_counter()
        for step in range(DECODE_STEPS):
            pos_buf.fill_(PROMPT_LEN + step)
            cache_pos_buf.fill_(PROMPT_LEN + step)
            compiled_call(
                input_ids=one_tok_buf,
                attention_mask=mask_4d,
                position_ids=pos_buf,
                cache_position=cache_pos_buf,
                past_key_values=cache,
                return_dict=True, use_cache=True,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return times

# ── main ──
print(f"\nPROMPT_LEN={PROMPT_LEN}, DECODE_STEPS={DECODE_STEPS}, CACHE_LEN={CACHE_LEN}\n")

print("Method A: model.generate() — full decode loop")
t_a = time_generate()
mean_a = sum(t_a) / len(t_a)
print(f"  per-call: {[round(x,3) for x in t_a]}")
print(f"  per-step: {mean_a / DECODE_STEPS * 1000:.3f} ms ({DECODE_STEPS/mean_a:.1f} tok/s)")

print("\nMethod B: compiled-forward called directly in a loop, NO growing tensors")
t_b = time_raw_decode_static()
mean_b = sum(t_b) / len(t_b)
print(f"  per-call (decode-only): {[round(x,3) for x in t_b]}")
print(f"  per-step: {mean_b / DECODE_STEPS * 1000:.3f} ms ({DECODE_STEPS/mean_b:.1f} tok/s)")

# (A) includes one prefill, (B) does not. Subtract a rough prefill estimate
# from (A) to compare per-step.
gap_ms = ((mean_a - PREFILL_EST_S) - mean_b) / DECODE_STEPS * 1000
print(f"\n(Note: A includes one prefill, B doesn't; rough prefill estimate ≈ {PREFILL_EST_S*1000:.0f} ms)")
print(f"Per-step gap A−B (with prefill subtracted): {gap_ms:.3f} ms/step")
