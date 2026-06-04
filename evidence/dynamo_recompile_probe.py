"""Investigate the actual Dynamo retrace mechanism for chunked-prefill +
static cache. Run the cold call, then a warm call, while logging recompile
reasons and guards. Read the raw output -- no inferences.

Set --case to pick a variant:
  baseline       : DIY cache, no early_init               (expect retrace)
  early_init     : DIY cache, early_initialization()      (expect clean)
  flag_only      : DIY cache, manually set is_initialized=True on each
                   layer (no K/V allocation, no shape attrs set)
                   -- isolates whether the flag flip alone is the trigger
  shape_only     : DIY cache, allocate keys/values + set shape attrs but
                   leave is_initialized=False
                   -- isolates whether the tensor allocation / shape attrs
                   matter independently
"""
import os, sys, time, tempfile, pathlib, argparse, logging

p = argparse.ArgumentParser()
p.add_argument("--case", choices=["baseline", "early_init", "flag_only", "shape_only"],
               required=True)
args = p.parse_args()

CACHE = tempfile.mkdtemp(prefix=f"probe_{args.case}_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ.setdefault("TORCH_LOGS", "recompiles,guards")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
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
    sys.stderr.write(f"\n========= case={args.case}  cache={CACHE} =========\n")
    sys.stderr.flush()

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16
    ).to("cuda").eval()

    cache = StaticCache(config=model.config, max_cache_len=2304)

    if args.case == "early_init":
        tc = model.config.get_text_config()
        cache.early_initialization(
            batch_size=1,
            num_heads=tc.num_key_value_heads,
            head_dim=tc.hidden_size // tc.num_attention_heads,
            dtype=torch.bfloat16,
            device="cuda",
        )
    elif args.case == "flag_only":
        # Flip only the bool; do NOT allocate K/V, do NOT set shape attrs.
        for layer in cache.layers:
            layer.is_initialized = True
    elif args.case == "shape_only":
        # Allocate K/V and set shape attrs but leave is_initialized = False.
        tc = model.config.get_text_config()
        nh = tc.num_key_value_heads
        hd = tc.hidden_size // tc.num_attention_heads
        for layer in cache.layers:
            # Mimic what lazy_initialization does, EXCEPT the final flag flip.
            layer.dtype = torch.bfloat16
            layer.device = torch.device("cuda")
            layer.max_batch_size = 1
            layer.num_heads = nh
            layer.v_head_dim = hd
            layer.k_head_dim = hd
            shape = (1, nh, layer.max_cache_len, hd)
            layer.keys = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
            layer.values = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
            if hasattr(layer, "cumulative_length") and torch.is_tensor(layer.cumulative_length):
                layer.cumulative_length = layer.cumulative_length.to("cuda")

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
            compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
            max_new_tokens=8, min_new_tokens=8,         # tiny decode to keep logs short
            prefill_chunk_size=1024,
            pad_token_id=tok.eos_token_id,
        )

    def call(label):
        before = count_artifacts(CACHE)
        torch.cuda.synchronize(); t = time.perf_counter()
        model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, generation_config=cfg())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        after = count_artifacts(CACHE)
        sys.stderr.write(f"\n>>> {label}: {dt:.2f} s  +{after-before} artifacts\n")
        sys.stderr.flush()

    sys.stderr.write("\n----- COLD -----\n"); sys.stderr.flush()
    call("cold")
    sys.stderr.write("\n----- WARM -----\n"); sys.stderr.flush()
    call("warm")


if __name__ == "__main__":
    main()
