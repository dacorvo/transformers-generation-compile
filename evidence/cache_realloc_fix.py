"""Workaround for the cache-realloc footgun: own the StaticCache.

WHAT: build `StaticCache(config, max_cache_len=N)` once, pass via
`past_key_values=`, drop `cache_implementation`. Then vary
`max_new_tokens` across calls and confirm no recompile.

WHY:  the auto-compile criterion checks `cache.is_compileable`, not
`generation_config.cache_implementation`. User-supplied caches keep
the compile path while skipping the auto-realloc logic. The user
takes on two responsibilities: call `cache.reset()` between turns,
and size the cache for the worst case at construction.

RESULT (A10G, Llama-3.2-1B, mode=default):
  cold first call 14 s, subsequent calls 0.07–0.50 s regardless of
  `max_new_tokens` (8 → 16 → 32 → 64 → back to 16, all fast).

RUN: CUDA_VISIBLE_DEVICES=0 .venv/bin/python evidence/cache_realloc_fix.py
"""
# ── env (must precede `import torch`) ──
import os
CACHE_DIR = "/tmp/inductor-diy"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── imports ──
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

# ── config ──
MID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = torch.device("cuda:0")
PROMPT_LEN = 256
WORST_DECODE_BUDGET = 64  # the cache is sized for this many new tokens; never decode more
WORST_CACHE_LEN = PROMPT_LEN + WORST_DECODE_BUDGET

# ── setup ──
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MID)
ids = tok("Hello, this is a benchmark of " * 30, return_tensors="pt").input_ids[:, :PROMPT_LEN].to(DEVICE)
mask = torch.ones_like(ids)

# Build ONE StaticCache, move its tensors to device.
cache = StaticCache(config=model.config, max_cache_len=WORST_CACHE_LEN)
for layer in cache.layers:
    if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
        layer.keys = layer.keys.to(DEVICE)
        layer.values = layer.values.to(DEVICE)

# ── helpers ──
def gen_diy(max_new_tokens: int) -> float:
    cache.reset()  # MUST reset between calls — otherwise stale K/V from prior turn
    cfg = GenerationConfig(
        do_sample=False,
        # Crucially: no cache_implementation. The compile path triggers on cache.is_compileable.
        compile_config=CompileConfig(mode="default", fullgraph=False),
        max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask,
                   generation_config=cfg, past_key_values=cache)
    torch.cuda.synchronize()
    return time.perf_counter() - t

# ── main ──
print(f"DIY StaticCache (fixed at {WORST_CACHE_LEN} slots), vary max_new_tokens across calls:")
print(f"  call 1 (max_new=8, cold):   {gen_diy(8):6.2f}s   <-- one-time compile")
print(f"  call 2 (max_new=8):         {gen_diy(8):6.2f}s")
print(f"  call 3 (max_new=16):        {gen_diy(16):6.2f}s   <-- was 19.3s in the auto path!")
print(f"  call 4 (max_new=32):        {gen_diy(32):6.2f}s   <-- was 11.2s in the auto path!")
print(f"  call 5 (max_new=64):        {gen_diy(64):6.2f}s")
print(f"  call 6 (max_new=16, back):  {gen_diy(16):6.2f}s")
