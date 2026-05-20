"""Demonstrate the silent-recompile bomb when max_new_tokens grows.

`generate()` derives `max_cache_length = generation_config.max_length - 1`
(see transformers/generation/utils.py:2495) and `max_length` is itself
`max_new_tokens + input_ids_length` (line 1627). When a subsequent
generate() call asks for a bigger `max_new_tokens` than any seen so far,
the StaticCache is reallocated (lines 1749-1770) — which changes the
shape of the KV tensors fed to the compiled forward and forces a full
Inductor recompile. In an agent loop where each turn may need a
different decode budget, this is a TTFT bomb.

Run me with:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo_cache_realloc.py
"""
from __future__ import annotations
import os, time
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-demo")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

MID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = torch.device("cuda:0")

model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
model.eval()
tok = AutoTokenizer.from_pretrained(MID)

ids = tok("Hello, this is a benchmark of " * 30, return_tensors="pt").input_ids[:, :256].to(DEVICE)
mask = torch.ones_like(ids)


def time_generate(max_new_tokens: int) -> float:
    gc = GenerationConfig(
        do_sample=False,
        cache_implementation="static",
        compile_config=CompileConfig(mode="default", fullgraph=False),
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    torch.cuda.synchronize()
    t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask, generation_config=gc)
    torch.cuda.synchronize()
    return time.perf_counter() - t


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
