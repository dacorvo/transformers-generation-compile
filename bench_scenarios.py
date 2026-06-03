"""Three-modes × four-scenarios cache-reuse sweep on CUDA.

Mirrors the Neuron team's `bench_4_scenarios.sh` pattern. Each cell
reports total_s / tps / Inductor artifact delta. Cache-absorption is
color-coded against the mode's own warm baseline:

  🟢 ≤ 1.5× warm     (essentially a cache hit)
  🟡 1.5–10× warm    (partial reuse)
  🔴 > 10× warm      (effectively a recompile)

Three modes:
  vanilla         cache_implementation="static" — auto-allocated cache
  diy             StaticCache pre-built, passed via past_key_values=
  static_tensors  diy + pre-allocated input_ids/position_ids/cache_position
                  + direct compiled_call() decode loop (no torch.cat)

Four scenarios:
  cold            fresh process, short prompt, default max_new_tokens
  warm            same call repeated (clean cache hit)
  warm-diff-in    warm cache, LONG prompt (2× chunks)
  warm-diff-mnt   warm cache, short prompt, LARGER max_new_tokens

RUN (orchestrator):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_scenarios.py

RUN (single mode subprocess, used by orchestrator):
    .venv/bin/python bench_scenarios.py --mode {vanilla,diy,static_tensors}
"""
from __future__ import annotations

# ── env (orchestrator and child both set TORCHINDUCTOR_CACHE_DIR per-mode) ──
import os
import sys

# ── imports ──
import argparse
import json
import subprocess
import time
from pathlib import Path

# ── config ──
MID = "meta-llama/Llama-3.2-1B-Instruct"
MAX_SEQ_LEN = 2048
PREFILL_CHUNK = 1024
SHORT_PROMPT_LEN = 1024
LONG_PROMPT_LEN = 2048
MAX_NEW_DEFAULT = 128
MAX_NEW_BIG = 256
# DIY cache must cover both warm-diff cells.
DIY_MAX_CACHE_LEN = max(LONG_PROMPT_LEN + MAX_NEW_DEFAULT,
                        SHORT_PROMPT_LEN + MAX_NEW_BIG) + 64  # 64 slack

MODES = ("vanilla", "diy", "static_tensors")
# Order matters for `vanilla`: its auto-cache grows monotonically, so running
# warm-diff-mnt before warm-diff-in ensures BOTH deltas force a realloc
# (otherwise warm-diff-in's bigger total cache would absorb warm-diff-mnt).
SCENARIOS = ("cold", "warm", "warm-diff-mnt", "warm-diff-in")
# (prompt_len, max_new_tokens) per scenario
SCENARIO_PARAMS = {
    "cold":          (SHORT_PROMPT_LEN, MAX_NEW_DEFAULT),
    "warm":          (SHORT_PROMPT_LEN, MAX_NEW_DEFAULT),
    "warm-diff-mnt": (SHORT_PROMPT_LEN, MAX_NEW_BIG),
    "warm-diff-in":  (LONG_PROMPT_LEN,  MAX_NEW_DEFAULT),
}
CACHE_ROOT = "/tmp/inductor-scenarios"


# ─────────────────────────── child-process side ───────────────────────────


def run_mode(mode: str) -> list[dict]:
    """Run all 4 scenarios for one mode in this process. Emit list of rows."""
    cache_dir = f"{CACHE_ROOT}/{mode}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Ensure cold actually means cold.
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    from transformers.generation.configuration_utils import GenerationConfig, CompileConfig
    from transformers.masking_utils import create_masks_for_generate

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    print(f"[{mode}] loading model...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        MID, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Pre-build the two prompt tensors so neither path pays tokenization cost
    # in the timed region. Random ids over a safe vocab range; BOS at pos 0.
    import random
    rng = random.Random(0)
    def make_ids(L: int) -> "torch.Tensor":
        ids = torch.tensor([[rng.randint(1000, 30000) for _ in range(L)]],
                           dtype=torch.long, device=device)
        if tok.bos_token_id is not None:
            ids[:, 0] = tok.bos_token_id
        return ids
    short_ids = make_ids(SHORT_PROMPT_LEN)
    long_ids = make_ids(LONG_PROMPT_LEN)
    short_mask = torch.ones_like(short_ids)
    long_mask = torch.ones_like(long_ids)

    # Mode-specific runners. Each returns wallclock for one call.
    if mode == "vanilla":
        run_call = _make_vanilla_runner(model, tok)
    elif mode == "diy":
        run_call = _make_diy_runner(model, tok, device)
    elif mode == "static_tensors":
        run_call = _make_static_tensors_runner(model, tok, device,
                                               create_masks_for_generate)
    else:
        raise ValueError(mode)

    rows: list[dict] = []
    for scenario in SCENARIOS:
        prompt_len, max_new = SCENARIO_PARAMS[scenario]
        ids = short_ids if prompt_len == SHORT_PROMPT_LEN else long_ids
        mask = short_mask if prompt_len == SHORT_PROMPT_LEN else long_mask
        artifacts_before = _count_artifacts(cache_dir)
        dt = run_call(ids, mask, max_new)
        artifacts_after = _count_artifacts(cache_dir)
        tps = max_new / dt if dt > 0 else 0.0
        row = {
            "mode": mode,
            "scenario": scenario,
            "total_s": round(dt, 2),
            "tps": round(tps, 2),
            "artifacts_delta": artifacts_after - artifacts_before,
        }
        rows.append(row)
        print(f"[{mode}/{scenario}] total={dt:.2f}s  tps={tps:.2f}  "
              f"artifacts+={artifacts_after - artifacts_before}",
              file=sys.stderr)

    return rows


def _make_vanilla_runner(model, tok):
    import torch
    from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

    def run(ids, mask, max_new):
        cfg = GenerationConfig(
            do_sample=False,
            cache_implementation="static",
            compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
            max_new_tokens=max_new, min_new_tokens=max_new,
            prefill_chunk_size=PREFILL_CHUNK,
            pad_token_id=tok.eos_token_id,
        )
        torch.cuda.synchronize(); t = time.perf_counter()
        model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    return run


def _make_diy_runner(model, tok, device):
    import torch
    from transformers import StaticCache
    from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

    cache = StaticCache(config=model.config, max_cache_len=DIY_MAX_CACHE_LEN)
    # Pre-fire the cache layers' `is_initialized` flag and allocate the K/V
    # tensors BEFORE Dynamo traces. Without this, Dynamo guards on the
    # is_initialized Python bool by object id (___check_obj_id), the flag
    # flips during cold's first forward, and the second call's prefill
    # recompiles. cache_utils.py:302 documents this knob.
    tc = model.config.get_text_config()
    cache.early_initialization(
        batch_size=1,
        num_heads=tc.num_key_value_heads,
        head_dim=tc.hidden_size // tc.num_attention_heads,
        dtype=torch.bfloat16,
        device=device,
    )

    def run(ids, mask, max_new):
        cache.reset()
        cfg = GenerationConfig(
            do_sample=False,
            # NO cache_implementation — would conflict with past_key_values=
            compile_config=CompileConfig(mode="default", fullgraph=False, dynamic=False),
            max_new_tokens=max_new, min_new_tokens=max_new,
            prefill_chunk_size=PREFILL_CHUNK,
            pad_token_id=tok.eos_token_id,
        )
        torch.cuda.synchronize(); t = time.perf_counter()
        model.generate(input_ids=ids, attention_mask=mask,
                       generation_config=cfg, past_key_values=cache)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    return run


def _make_static_tensors_runner(model, tok, device, create_masks_for_generate):
    """Lifted from evidence/decode_overhead.py — DIY cache + pre-allocated
    decode tensors + direct compiled_call() loop with manual argmax."""
    import torch
    from transformers import StaticCache
    from transformers.generation.configuration_utils import GenerationConfig, CompileConfig

    cache = StaticCache(config=model.config, max_cache_len=DIY_MAX_CACHE_LEN)
    tc = model.config.get_text_config()
    cache.early_initialization(
        batch_size=1,
        num_heads=tc.num_key_value_heads,
        head_dim=tc.hidden_size // tc.num_attention_heads,
        dtype=torch.bfloat16,
        device=device,
    )

    compile_config = CompileConfig(mode="default", fullgraph=False, dynamic=False)
    compiled_call = model.get_compiled_call(compile_config)

    # Pre-allocate decode-step buffers (single-token input + position + cache_pos).
    one_tok_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
    pos_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
    cache_pos_buf = torch.zeros((1,), dtype=torch.long, device=device)

    def _decode_4d_mask(cache_len: int):
        return create_masks_for_generate(
            config=model.config,
            inputs_embeds=torch.empty((1, 1, 0), dtype=torch.bfloat16, device=device),
            attention_mask=torch.ones((1, cache_len), dtype=torch.long, device=device),
            past_key_values=cache,
            position_ids=pos_buf,
        )

    def run(ids, mask, max_new):
        cache.reset()
        prompt_len = ids.shape[1]
        cache_len = cache.max_cache_len  # always DIY_MAX_CACHE_LEN

        torch.cuda.synchronize(); t = time.perf_counter()

        # --- Prefill, chunked. ---
        prefill_mask = create_masks_for_generate(
            config=model.config,
            inputs_embeds=torch.empty((1, prompt_len, 0), dtype=torch.bfloat16, device=device),
            attention_mask=torch.ones((1, cache_len), dtype=torch.long, device=device),
            past_key_values=cache,
            position_ids=torch.arange(prompt_len, device=device).unsqueeze(0),
        )
        for start in range(0, prompt_len, PREFILL_CHUNK):
            end = start + PREFILL_CHUNK
            chunk_ids = ids[:, start:end]
            chunk_pos = torch.arange(start, end, device=device).unsqueeze(0)
            chunk_cache_pos = torch.arange(start, end, device=device)
            # Slice the 4D mask for this chunk.
            if isinstance(prefill_mask, dict):
                chunk_mask = {k: v[..., start:end, :] for k, v in prefill_mask.items()}
            else:
                chunk_mask = prefill_mask[..., start:end, :]
            out = compiled_call(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                position_ids=chunk_pos,
                cache_position=chunk_cache_pos,
                past_key_values=cache,
                return_dict=True, use_cache=True,
            )

        # Sample first token from prefill's last-position logits.
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_mask = _decode_4d_mask(cache_len)

        # --- Decode loop, no growing tensors. ---
        for step in range(max_new - 1):  # already emitted 1 token above
            one_tok_buf.copy_(next_token)
            pos_buf.fill_(prompt_len + step)
            cache_pos_buf.fill_(prompt_len + step)
            out = compiled_call(
                input_ids=one_tok_buf,
                attention_mask=decode_mask,
                position_ids=pos_buf,
                cache_position=cache_pos_buf,
                past_key_values=cache,
                return_dict=True, use_cache=True,
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        torch.cuda.synchronize()
        return time.perf_counter() - t

    return run


def _count_artifacts(root: str) -> int:
    p = Path(root)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.is_file()
               and (f.suffix in {".cubin", ".so"} or f.name.endswith(".kernel.json")))


# ─────────────────────────── orchestrator side ────────────────────────────


def orchestrate() -> None:
    all_rows: list[dict] = []
    for mode in MODES:
        print(f"\n=== launching {mode} ===", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, __file__, "--mode", mode],
            check=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=sys.stderr,
        )
        rows = [json.loads(line) for line in proc.stdout.decode().splitlines() if line.strip()]
        all_rows.extend(rows)

    print_table(all_rows)
    tsv = Path("logs/scenarios.tsv")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open("w") as f:
        f.write("mode\tscenario\ttotal_s\ttps\tartifacts_delta\tcolor\n")
        for r in colorize_rows(all_rows):
            f.write(f"{r['mode']}\t{r['scenario']}\t{r['total_s']}\t{r['tps']}\t{r['artifacts_delta']}\t{r['color']}\n")
    print(f"\n[wrote {tsv}]")


def colorize_rows(rows: list[dict]) -> list[dict]:
    """Add a 'color' field per row, based on Inductor artifact delta.

    Previously the color was a wallclock-vs-warm ratio. That made sense when
    `warm` carried ~13 s of one-shot late compile (since fixed by
    `cache.early_initialization()` in the diy/static_tensors runners): then
    a "fast" warm-diff cell was the giveaway that no recompile happened.
    With warm now clean, more tokens just take more wallclock — ratio gets
    confused by the workload size.

    The +artifacts column is the unambiguous "did Inductor recompile"
    signal: zero new artifacts means the cache absorbed the delta, any
    positive count means a real recompile happened.
    """
    out = []
    for r in rows:
        scenario = r["scenario"]
        # `cold` is the compile baseline — always has artifacts, coloring it
        # would be meaningless. Every other cell (warm + warm-diff-*) is rated
        # on its artifact delta.
        if scenario == "cold":
            color = ""
        elif r["artifacts_delta"] == 0:
            color = "🟢"
        else:
            color = "🔴"
        out.append({**r, "color": color})
    return out


def print_table(rows: list[dict]) -> None:
    rows = colorize_rows(rows)
    by_mode: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], {})[r["scenario"]] = r
    header = f"{'mode':<16} | " + " | ".join(f"{s:^22}" for s in SCENARIOS)
    print("\n" + header)
    print("-" * len(header))
    for mode in MODES:
        cells = []
        for scenario in SCENARIOS:
            r = by_mode.get(mode, {}).get(scenario)
            if r is None:
                cells.append(" " * 22)
                continue
            color = r["color"] + " " if r["color"] else ""
            txt = f"{color}{r['total_s']:.1f}s/{r['tps']:.1f}tps/+{r['artifacts_delta']}a"
            cells.append(f"{txt:^22}")
        print(f"{mode:<16} | " + " | ".join(cells))


# ─────────────────────────── entry point ─────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, default=None,
                   help="If set, run only this mode and emit JSONL rows to stdout.")
    args = p.parse_args()
    if args.mode:
        # Child process: emit one JSON row per scenario to stdout.
        rows = run_mode(args.mode)
        for r in rows:
            print(json.dumps(r))
    else:
        orchestrate()


if __name__ == "__main__":
    main()
