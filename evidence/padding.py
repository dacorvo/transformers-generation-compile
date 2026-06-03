"""Show that `prefill_chunk_size` must divide the input length.

WHAT: warm at L=2048 (clean multiple of chunk=1024), then call
generate() at clean and ragged lengths. Tracks Inductor cache
directory growth and wallclock per call.

WHY:  the chunked-prefill path does `torch.split(input_ids, chunk_size)`
in transformers/generation/utils.py:3773 — a ragged tail is a new
input shape that Inductor compiles on first occurrence.

RESULT (A10G, Llama-3.2-1B, mode=default):
  ragged lengths add 25–31 artifacts and ~26 s per first occurrence;
  clean multiples reuse cached kernels at 0 artifacts.

RUN: CUDA_VISIBLE_DEVICES=0 .venv/bin/python evidence/padding.py
"""
# ── env (must precede `import torch`) ──
import os
CACHE_DIR = "/tmp/inductor-padding"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── imports ──
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

# ── config ──
MID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = torch.device("cuda:0")
CHUNK = 1024
DECODE_TOKENS = 64

# ── setup ──
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MID)
gen_cfg = GenerationConfig(
    do_sample=False,
    cache_implementation="static",
    compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
    max_new_tokens=DECODE_TOKENS, min_new_tokens=DECODE_TOKENS,
    prefill_chunk_size=CHUNK,
    pad_token_id=tok.eos_token_id,
)

# ── helpers ──
def fake_inputs(L: int):
    ids = torch.full((1, L), tok.bos_token_id or 1, device=DEVICE)
    return ids, torch.ones_like(ids)

def count_artifacts() -> int:
    p = Path(CACHE_DIR)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.is_file()
               and (f.suffix in {".cubin", ".so"} or f.name.endswith(".kernel.json")))

class _StopAfter:
    def __init__(self, prompt_len: int): self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **_):
        return torch.full((input_ids.shape[0],), input_ids.shape[1] > self.prompt_len,
                          dtype=torch.bool, device=input_ids.device)

def time_call(L: int) -> tuple[float, int]:
    ids, mask = fake_inputs(L)
    a_before = count_artifacts()
    stop = StoppingCriteriaList([_StopAfter(L)])
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask,
                   generation_config=gen_cfg, stopping_criteria=stop)
    torch.cuda.synchronize()
    return time.perf_counter() - t, count_artifacts() - a_before

# ── main ──
print(f"warming up at L=2048 (clean multiple of {CHUNK})...")
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
