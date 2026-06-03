"""Confirm recompile-free dispatch across input-length buckets.

WHAT: warm buckets 1024 and 8192 with chunk_size=1024 and a pinned
decode budget, then run 20 generate() calls with bucket picked
uniformly at random. Track Inductor cache directory and per-call TTFT.

WHY:  the "recompile-free dispatch across buckets" claim from
SCOPE.md is only useful if the buckets actually share the compiled
kernels at runtime — not just in isolation. Random-order calls are
the proof.

RESULT (A10G, Llama-3.2-1B, mode=default):
  0 new Inductor artifacts across 20 interleaved calls;
  p99/p50 = 1.001 for both buckets.

RUN: CUDA_VISIBLE_DEVICES=0 .venv/bin/python evidence/interleave_buckets.py
"""
# ── env (must precede `import torch`) ──
import os
CACHE_DIR = "/tmp/inductor-interleave"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── imports ──
import random
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
BUCKETS = (1024, 8192)
PREFILL_CHUNK = 1024
DECODE_BUDGET = 128
N_INTERLEAVED_CALLS = 20

# ── setup ──
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MID)
gen_cfg = GenerationConfig(
    do_sample=False,
    cache_implementation="static",
    compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
    max_new_tokens=DECODE_BUDGET, min_new_tokens=DECODE_BUDGET,
    prefill_chunk_size=PREFILL_CHUNK,
    pad_token_id=tok.eos_token_id,
)

# ── helpers ──
def fake_prompt(L: int, seed: int):
    rng = random.Random(seed)
    ids = torch.tensor(
        [[rng.randint(1000, 30000) for _ in range(L)]], dtype=torch.long, device=DEVICE,
    )
    if tok.bos_token_id is not None:
        ids[:, 0] = tok.bos_token_id
    return ids, torch.ones_like(ids)

class _StopAfter:
    def __init__(self, prompt_len: int): self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **_):
        return torch.full((input_ids.shape[0],), input_ids.shape[1] > self.prompt_len,
                          dtype=torch.bool, device=input_ids.device)

def count_artifacts() -> int:
    p = Path(CACHE_DIR)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.is_file()
               and (f.suffix in {".cubin", ".so"} or f.name.endswith(".kernel.json")))

# ── main ──
# 1. Warmup, biggest bucket first.
print("warming up...")
for L in sorted(BUCKETS, reverse=True):
    ids, mask = fake_prompt(L, seed=10 + L)
    model.generate(input_ids=ids, attention_mask=mask, generation_config=gen_cfg)
artifacts_after_warmup = count_artifacts()
print(f"  warmup done. artifacts={artifacts_after_warmup}")

# 2. Interleaved loop: random bucket per call.
random.seed(0)
ttfts_by_bucket: dict[int, list[float]] = {L: [] for L in BUCKETS}
print(f"\ninterleaved loop ({N_INTERLEAVED_CALLS} calls, random bucket per call):")
for i in range(N_INTERLEAVED_CALLS):
    L = random.choice(BUCKETS)
    ids, mask = fake_prompt(L, seed=2000 + i)
    stop = StoppingCriteriaList([_StopAfter(prompt_len=ids.shape[1])])
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask,
                   generation_config=gen_cfg, stopping_criteria=stop)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    ttfts_by_bucket[L].append(dt)
    print(f"  i={i:2d}  L={L:>5}  ttft={dt*1000:7.1f} ms  artifacts={count_artifacts()}")

# 3. Summary.
print("\n=== summary ===")
artifacts_after_loop = count_artifacts()
print(f"artifacts before interleave: {artifacts_after_warmup}")
print(f"artifacts after  interleave: {artifacts_after_loop}   delta={artifacts_after_loop - artifacts_after_warmup}")
for L, lst in ttfts_by_bucket.items():
    if not lst:
        continue
    s = sorted(lst)
    p50 = s[len(s) // 2]
    p99 = s[min(len(s) - 1, int(round(0.99 * (len(s) - 1))))]
    print(f"bucket {L}: n={len(lst):2d}  p50={p50*1000:7.1f} ms  p99={p99*1000:7.1f} ms  p99/p50={p99/p50:.3f}")
