"""Show that growing `max_new_tokens` reallocates the StaticCache and
forces a full Inductor recompile.

WHAT: 8 generate() calls with `max_new_tokens` stepped 8 → 16 → 32 → 8.

WHY:  `max_cache_length = generation_config.max_length - 1`
(transformers/generation/utils.py:2495) and `max_length =
max_new_tokens + input_ids_length`. Any call whose computed
`max_cache_length` exceeds anything seen so far reallocates the
StaticCache (line 1753) → new tensor shape → recompile.

RESULT (A10G, Llama-3.2-1B, mode=default):
  each new-max value pays 11–19 s of recompile;
  same-as-before or smaller values stay at ~70 ms / call.

RUN: CUDA_VISIBLE_DEVICES=0 .venv/bin/python evidence/cache_realloc.py
"""
# ── env (must precede `import torch`) ──
import os
CACHE_DIR = "/tmp/inductor-realloc"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── imports ──
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

# ── config ──
MID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = torch.device("cuda:0")
PROMPT_LEN = 256

# ── setup ──
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MID)
ids = tok("Hello, this is a benchmark of " * 30, return_tensors="pt").input_ids[:, :PROMPT_LEN].to(DEVICE)
mask = torch.ones_like(ids)

# ── helpers ──
def time_generate(max_new_tokens: int) -> float:
    cfg = GenerationConfig(
        do_sample=False,
        cache_implementation="static",
        compile_config=CompileConfig(mode="default", fullgraph=False),
        max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
    torch.cuda.synchronize()
    return time.perf_counter() - t

# ── main ──
print("== Scenario A: hold max_new_tokens constant ==")
print(f"  call 1 (max_new=8, cold):      {time_generate(8):6.2f}s   <-- pays full compile")
print(f"  call 2 (max_new=8):            {time_generate(8):6.2f}s")
print(f"  call 3 (max_new=8):            {time_generate(8):6.2f}s")

print("\n== Scenario B: grow max_new_tokens across calls ==")
print(f"  call 4 (max_new=16, NEW max):  {time_generate(16):6.2f}s   <-- cache realloc + recompile")
print(f"  call 5 (max_new=16):           {time_generate(16):6.2f}s")
print(f"  call 6 (max_new=32, NEW max):  {time_generate(32):6.2f}s   <-- cache realloc + recompile again")
print(f"  call 7 (max_new=32):           {time_generate(32):6.2f}s")

print("\n== Scenario C: shrink back ==")
print(f"  call 8 (max_new=8, shrunk):    {time_generate(8):6.2f}s   <-- reuses larger cache, no recompile")
