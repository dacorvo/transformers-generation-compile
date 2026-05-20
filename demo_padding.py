"""Demonstrate: if input length isn't a multiple of prefill_chunk_size,
the tail chunk has a different shape and triggers a recompile.

The chunked prefill code does:
    input_chunks = torch.split(input_ids, chunk_size, dim=-1)
([transformers/generation/utils.py:3773](src/transformers/generation/utils.py#L3773))

torch.split returns the last chunk with whatever's left, even if that
shape has never been compiled. Inductor sees a new shape -> recompile.

Run me with:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo_padding.py
"""
from __future__ import annotations
import os, time
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-padding")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

torch.set_grad_enabled(False); torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda:0")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16, attn_implementation="sdpa",
).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

CHUNK = 1024
cfg = GenerationConfig(
    do_sample=False,
    cache_implementation="static",
    compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
    max_new_tokens=64, min_new_tokens=64,
    prefill_chunk_size=CHUNK,
    pad_token_id=tok.eos_token_id,
)

def fake(L: int):
    ids = torch.full((1, L), tok.bos_token_id or 1, device=DEVICE)
    return ids, torch.ones_like(ids)

def count_artifacts(root="/tmp/inductor-padding"):
    from pathlib import Path
    p = Path(root)
    if not p.exists(): return 0
    return sum(1 for f in p.rglob("*") if f.is_file()
               and (f.suffix in {".cubin", ".so"} or f.name.endswith(".kernel.json")))

class StopAfterOne:
    def __init__(self, n): self.n = n
    def __call__(self, input_ids, scores, **k):
        return torch.full((input_ids.shape[0],), input_ids.shape[1] > self.n,
                          dtype=torch.bool, device=input_ids.device)


def time_call(L: int) -> tuple[float, int]:
    ids, mask = fake(L)
    a_before = count_artifacts()
    sc = StoppingCriteriaList([StopAfterOne(L)])
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg, stopping_criteria=sc)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return dt, count_artifacts() - a_before


# Step 0: warm up at L=2048 (multiple of CHUNK=1024).
print("warming up at L=2048 (clean multiple of 1024)...")
dt, da = time_call(2048)
print(f"  cold:  {dt:6.2f}s   artifacts+={da}")
dt, da = time_call(2048)
print(f"  warm:  {dt:6.2f}s   artifacts+={da}")

print("\n=== well-padded vs ragged input ===")
for L in [1024, 2048, 4096]:
    dt, da = time_call(L)
    print(f"  L={L:>5} (L%chunk={L%CHUNK:>4}):  {dt:6.2f}s  artifacts+={da}")

print()
for L in [1500, 1700, 3500]:
    dt, da = time_call(L)
    print(f"  L={L:>5} (L%chunk={L%CHUNK:>4}):  {dt:6.2f}s  artifacts+={da}   <-- TAIL CHUNK")
