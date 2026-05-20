"""Verify: after warming buckets 1024 and 8192 with chunk_size=1024
and a pinned decode budget, can we INTERLEAVE bucket calls in a hot
loop with zero recompiles?

This is the "recompile-free dispatch across buckets" claim from
SCOPE.md. If it holds, the same agent process can serve any prompt
that fits one of the warmed buckets without TTFT spikes.

Run me with:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo_interleave_buckets.py
"""
from __future__ import annotations
import os, time, random
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-interleave")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda:0")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=torch.bfloat16, attn_implementation="sdpa",
).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

BUCKETS = (1024, 8192)
PREFILL_CHUNK = 1024
DECODE_BUDGET = 128


def make_cfg() -> GenerationConfig:
    return GenerationConfig(
        do_sample=False,
        cache_implementation="static",
        compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
        max_new_tokens=DECODE_BUDGET,
        min_new_tokens=DECODE_BUDGET,
        prefill_chunk_size=PREFILL_CHUNK,
        pad_token_id=tok.eos_token_id,
    )


def fake_prompt(L: int, seed: int):
    rng = random.Random(seed)
    ids = torch.tensor(
        [[rng.randint(1000, 30000) for _ in range(L)]], dtype=torch.long, device=DEVICE,
    )
    if tok.bos_token_id is not None:
        ids[:, 0] = tok.bos_token_id
    return ids, torch.ones_like(ids)


class StopAfterOne:
    def __init__(self, prompt_len): self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **k):
        return torch.full((input_ids.shape[0],), input_ids.shape[1] > self.prompt_len,
                          dtype=torch.bool, device=input_ids.device)


def count_artifacts(root="/tmp/inductor-interleave"):
    from pathlib import Path
    p = Path(root)
    if not p.exists(): return 0
    return sum(1 for f in p.rglob("*") if f.is_file()
               and (f.suffix in {".cubin", ".so"} or f.name.endswith(".kernel.json")))


cfg = make_cfg()

# WARMUP: biggest first.
print("warming up...")
for L in sorted(BUCKETS, reverse=True):
    ids, mask = fake_prompt(L, seed=10 + L)
    model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
a0 = count_artifacts()
print(f"  warmup done. artifacts={a0}")

# INTERLEAVED loop: 20 calls, bucket chosen randomly per call.
random.seed(0)
ttfts_by_bucket = {L: [] for L in BUCKETS}
print("\ninterleaved loop (20 calls, random bucket per call):")
for i in range(20):
    L = random.choice(BUCKETS)
    ids, mask = fake_prompt(L, seed=2000 + i)
    sc = StoppingCriteriaList([StopAfterOne(prompt_len=ids.shape[1])])
    torch.cuda.synchronize()
    t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg, stopping_criteria=sc)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    ttfts_by_bucket[L].append(dt)
    print(f"  i={i:2d}  L={L:>5}  ttft={dt*1000:7.1f} ms  artifacts={count_artifacts()}")

print("\n=== summary ===")
print(f"artifacts before interleave: {a0}")
print(f"artifacts after  interleave: {count_artifacts()}   delta={count_artifacts()-a0}")
for L, lst in ttfts_by_bucket.items():
    if lst:
        p50 = sorted(lst)[len(lst)//2]
        p99 = sorted(lst)[min(len(lst)-1, int(round(0.99*(len(lst)-1))))]
        print(f"bucket {L}: n={len(lst):2d}  p50={p50*1000:7.1f} ms  p99={p99*1000:7.1f} ms  p99/p50={p99/p50:.3f}")
