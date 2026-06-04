"""Test: does `cache_config={"max_cache_len": N}` actually size the static cache
when used with `cache_implementation="static"`, or is it silently ignored?

Calls generate() once with a small max_new_tokens and a small prompt but a
deliberately-large cache_config["max_cache_len"]. Inspects the resulting
cache afterwards to see what size was actually allocated.
"""
import os, tempfile, torch
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="cc_test_")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16
).to("cuda").eval()

ids = torch.tensor([[tok.bos_token_id or 1] + [42] * 31], dtype=torch.long, device="cuda")
mask = torch.ones_like(ids)

USER_REQUESTED = 2048
NATURAL = 32 + 16 - 1   # input_length=32, max_new_tokens=16 → max_cache_len would be 47

cfg = GenerationConfig(
    do_sample=False,
    cache_implementation="static",
    cache_config={"max_cache_len": USER_REQUESTED},
    compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
    max_new_tokens=16, min_new_tokens=16,
    pad_token_id=tok.eos_token_id,
)
model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)

cache = model._cache
print(f"User requested cache_config['max_cache_len'] = {USER_REQUESTED}")
print(f"Natural max_length-1                          = {NATURAL}")
print(f"Actual cache.max_cache_len                    = {cache.max_cache_len}")
print(f"Layer 0 keys shape                            = {cache.layers[0].keys.shape}")
if cache.max_cache_len == USER_REQUESTED:
    print("→ cache_config WAS honored")
else:
    print(f"→ cache_config IGNORED on static path (got {cache.max_cache_len}, not {USER_REQUESTED})")
