"""Empirically test the claim: the `is_initialized` Dynamo guard miss
happens only when prefill itself is compiled — i.e. only on the chunked
prefill path. With unchunked prefill, the flip occurs in eager code and
no compiled graph guards on it, so cold -> warm should be clean.

Run against a vanilla `cache_implementation="static"` config with
`prefill_chunk_size` left unset, no `cache.early_initialization(...)`.
"""
import os, sys, time, tempfile, pathlib
import torch

CACHE = tempfile.mkdtemp(prefix="unchunked_no_init_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig


def count_artifacts(root):
    p = pathlib.Path(root)
    if not p.exists(): return 0
    n = 0
    for f in p.rglob("*"):
        if f.is_file() and (f.suffix in (".cubin", ".so") or f.name.endswith(".kernel.json")):
            n += 1
    return n


def main():
    print(f"inductor cache: {CACHE}")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16
    ).to("cuda").eval()

    import random
    rng = random.Random(0)
    ids = torch.tensor(
        [[rng.randint(1000, 30000) for _ in range(1024)]], dtype=torch.long, device="cuda"
    )
    ids[:, 0] = tok.bos_token_id or 1
    mask = torch.ones_like(ids)

    def cfg():
        return GenerationConfig(
            do_sample=False,
            cache_implementation="static",
            compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
            max_new_tokens=128, min_new_tokens=128,
            # NB: NO prefill_chunk_size — unchunked path
            pad_token_id=tok.eos_token_id,
        )

    def call(label):
        before = count_artifacts(CACHE)
        torch.cuda.synchronize(); t = time.perf_counter()
        model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        after = count_artifacts(CACHE)
        print(f"  {label}: {dt:.2f} s  +{after-before} artifacts")

    print("\n-- vanilla static cache, UNCHUNKED prefill, no early_init --")
    call("cold")
    call("warm")


if __name__ == "__main__":
    main()
