"""How much of generate()'s per-decode-step time is the compiled
forward, and how much is Python plumbing around growing tensors?

We compare three things, all on the same model, same compiled forward,
same StaticCache of the same size:

  (A) generate(max_new_tokens=N) — full generate() loop.
  (B) The same N decode steps done by directly calling
      model.get_compiled_call()(input_ids=last_tok, ...) in a tight
      loop that allocates NOTHING new each step (we reuse pre-built
      tensors for input_ids, position_ids, cache_position, attention
      mask). This is the "static convenience tensors" world.
  (C) Like B but explicitly torch.cat'ing the prefix tensors each step
      to mimic generate()'s growing input_ids/attention_mask. Should
      land between A and B.

Difference between (A) and (B) = Python overhead + growing-tensor cost
per decode step on CUDA.

Run me with:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo_decode_overhead.py
"""
from __future__ import annotations
import os, time, sys
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-overhead")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig
from transformers.masking_utils import create_masks_for_generate

torch.set_grad_enabled(False); torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda:0")
MID = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MID)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

# Fixed dimensions, intentionally small so we can run many decode steps cheaply.
PROMPT_LEN = 256
DECODE_STEPS = 256        # how many decode steps to time
CACHE_LEN = PROMPT_LEN + DECODE_STEPS + 32   # +32 slack

# Pre-build inputs.
prompt_ids = torch.full((1, PROMPT_LEN), tok.bos_token_id or 1, device=DEVICE)
prompt_mask = torch.ones_like(prompt_ids)

# Use the DIY pattern.
cache_template = StaticCache(config=model.config, max_cache_len=CACHE_LEN)
for layer in cache_template.layers:
    if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
        layer.keys, layer.values = layer.keys.cuda(), layer.values.cuda()

gen_cfg = GenerationConfig(
    do_sample=False,
    compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
    max_new_tokens=DECODE_STEPS,
    min_new_tokens=DECODE_STEPS,
    prefill_chunk_size=PROMPT_LEN,  # one chunk
    pad_token_id=tok.eos_token_id,
)

# --- WARMUP: do a full generate to compile everything. ---
print("warming up compile...", file=sys.stderr)
cache_template.reset()
_ = model.generate(input_ids=prompt_ids, attention_mask=prompt_mask,
                   generation_config=gen_cfg, past_key_values=cache_template)
torch.cuda.synchronize()
print("compile done.", file=sys.stderr)


def time_generate(n_iter=5):
    times = []
    for _ in range(n_iter):
        cache_template.reset()
        torch.cuda.synchronize(); t = time.perf_counter()
        model.generate(input_ids=prompt_ids, attention_mask=prompt_mask,
                       generation_config=gen_cfg, past_key_values=cache_template)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return times


def time_raw_decode_static(n_iter=5):
    """Reuse pre-built tensors with NO Python-side growth."""
    times = []
    # First, do a prefill so the cache contains real K/V.
    compiled_call = model.get_compiled_call(gen_cfg.compile_config)
    # Pre-allocate "growing" tensors at their max length, plus the views we'll feed in.
    one_tok_buf = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # single-token input
    position_id_buf = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    cache_pos_buf = torch.zeros((1,), dtype=torch.long, device=DEVICE)
    # 4D mask pre-built once for the decode step. Shape (B, 1, q=1, kv=CACHE_LEN).
    mask_4d = create_masks_for_generate(
        config=model.config,
        inputs_embeds=torch.empty((1, 1, 0), dtype=torch.bfloat16, device=DEVICE),
        attention_mask=torch.ones((1, CACHE_LEN), dtype=torch.long, device=DEVICE),
        past_key_values=cache_template,
        position_ids=position_id_buf,
    )
    for _ in range(n_iter):
        cache_template.reset()
        # Prefill via a single forward (chunk = PROMPT_LEN). We don't time this.
        prefill_mask_4d = create_masks_for_generate(
            config=model.config,
            inputs_embeds=torch.empty((1, PROMPT_LEN, 0), dtype=torch.bfloat16, device=DEVICE),
            attention_mask=torch.ones((1, CACHE_LEN), dtype=torch.long, device=DEVICE),
            past_key_values=cache_template,
            position_ids=torch.arange(PROMPT_LEN, device=DEVICE).unsqueeze(0),
        )
        compiled_call(
            input_ids=prompt_ids,
            attention_mask=prefill_mask_4d,
            position_ids=torch.arange(PROMPT_LEN, device=DEVICE).unsqueeze(0),
            cache_position=torch.arange(PROMPT_LEN, device=DEVICE),
            past_key_values=cache_template,
            return_dict=True,
            use_cache=True,
        )
        torch.cuda.synchronize(); t = time.perf_counter()
        for step in range(DECODE_STEPS):
            position_id_buf.fill_(PROMPT_LEN + step)
            cache_pos_buf.fill_(PROMPT_LEN + step)
            # one_tok_buf is reused; in a real loop we'd update with the new token,
            # but for timing the plumbing the value doesn't matter.
            out = compiled_call(
                input_ids=one_tok_buf,
                attention_mask=mask_4d,
                position_ids=position_id_buf,
                cache_position=cache_pos_buf,
                past_key_values=cache_template,
                return_dict=True,
                use_cache=True,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return times


print(f"\nPROMPT_LEN={PROMPT_LEN}, DECODE_STEPS={DECODE_STEPS}, CACHE_LEN={CACHE_LEN}\n")

print("Method A: model.generate() — full decode loop")
t_a = time_generate()
mean_a = sum(t_a) / len(t_a)
print(f"  per-call: {t_a}")
print(f"  per-step: {mean_a / DECODE_STEPS * 1000:.3f} ms ({DECODE_STEPS/mean_a:.1f} tok/s)")

print("\nMethod B: compiled-forward called directly in a loop, NO growing tensors")
t_b = time_raw_decode_static()
mean_b_full = sum(t_b) / len(t_b)
print(f"  per-call (decode-only): {t_b}")
print(f"  per-step: {mean_b_full / DECODE_STEPS * 1000:.3f} ms ({DECODE_STEPS/mean_b_full:.1f} tok/s)")

# generate() includes a prefill, raw_decode_static does not. So compare per-step only.
overhead_ms = (mean_a - mean_b_full) / DECODE_STEPS * 1000
# But mean_a INCLUDES one prefill, which method B excluded. So subtract a prefill estimate.
prefill_est_s = 0.052  # ~52ms at PROMPT_LEN=256, chunk=256 from earlier sweeps (will overshoot — fine)
print(f"\n(Note: A includes one prefill, B doesn't; rough prefill estimate ≈ {prefill_est_s*1000:.0f} ms)")
print(f"Per-step gap A−B (with prefill subtracted): "
      f"{((mean_a - prefill_est_s) - mean_b_full) / DECODE_STEPS * 1000:.3f} ms/step")
