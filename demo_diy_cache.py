"""Workaround for the cache-realloc footgun: pre-build a StaticCache
of your chosen size, pass it via `past_key_values=`, and DON'T set
`cache_implementation`. This keeps the auto-compile path (the cache is
still `.is_compileable`) but bypasses the auto-realloc-on-growth
logic in `_prepare_static_cache`.

The user is responsible for calling `cache.reset()` between calls.

Run me with:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo_diy_cache.py
"""
from __future__ import annotations
import os, time
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-diy")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

torch.set_grad_enabled(False); torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda:0")
MID = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MID)

prompt_len = 256
ids  = tok("Hello, this is a benchmark of " * 30, return_tensors="pt").input_ids[:, :prompt_len].to(DEVICE)
mask = torch.ones_like(ids)

# Build ONE StaticCache that fits the worst case.
WORST_CACHE_LEN = prompt_len + 64   # we will never decode more than 64 tokens
cache = StaticCache(config=model.config, max_cache_len=WORST_CACHE_LEN)
# Move cache to device:
for layer in cache.layers:
    if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
        layer.keys = layer.keys.to(DEVICE)
        layer.values = layer.values.to(DEVICE)


def gen_diy(max_new_tokens: int) -> float:
    cache.reset()  # MUST reset between calls — otherwise stale K/V from prior turn
    gc = GenerationConfig(
        do_sample=False,
        # Crucially: no cache_implementation. The compile path triggers on cache.is_compileable.
        compile_config=CompileConfig(mode="default", fullgraph=False),
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask, generation_config=gc, past_key_values=cache)
    torch.cuda.synchronize()
    return time.perf_counter() - t


print("DIY StaticCache (fixed at 320 slots), vary max_new_tokens across calls:")
print(f"  call 1 (max_new=8, cold):   {gen_diy(8):6.2f}s   <-- one-time compile")
print(f"  call 2 (max_new=8):         {gen_diy(8):6.2f}s")
print(f"  call 3 (max_new=16):        {gen_diy(16):6.2f}s   <-- was 19.3s in the auto path!")
print(f"  call 4 (max_new=32):        {gen_diy(32):6.2f}s   <-- was 11.2s in the auto path!")
print(f"  call 5 (max_new=64):        {gen_diy(64):6.2f}s")
print(f"  call 6 (max_new=16, back):  {gen_diy(16):6.2f}s")
