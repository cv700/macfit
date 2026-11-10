#!/usr/bin/env python3
"""macfit: will this model fit on my Mac? Estimates the peak memory of mlx-lm before you download the weights.

Usage:
    python3 macfit.py MODEL [--context 4096,8192,32768] [--max-tokens 512] [--kv-bits 16|8|4]
                            [--prefill-step 2048] [--limit-gb N] [--json]

MODEL is a local folder (config.json and *.safetensors) or a Hugging Face repo id such as
mlx-community/Qwen3-8B-4bit. For a repo id, macfit reads only config.json and the file sizes. It downloads no weights.
For a gated or private repo, set HF_TOKEN. macfit sends it only to huggingface.co.

It needs only the Python that ships with macOS. It reads the GPU memory limit from Metal.

The estimate (see VALIDATION.md):
    peak = weights (file sizes) + KV cache (from config.json, mlx-lm allocates in steps of 256 tokens)
           + command buffers in flight (0.40 to 0.45 GiB at prefill chunks of 512 to 2,048 tokens)
           + score matrices, only with a quantized KV cache (that path writes them out)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

__version__ = "0.1.0"
GIB = 2**30
TIGHT = 0.90  # above this share of the limit, the verdict is "tight": estimates have run up to 4% low


# Command buffers in flight during prefill, measured on Qwen3-4B (default settings, 1K and 8K prompts):
# 0.40 GiB at 512-token chunks, 0.43 at 1,024, 0.45 at 2,048. Below 512 tokens, assumed to shrink in proportion.
def inflight_bytes(chunk):
    if chunk >= 512:
        return (0.40 + 0.025 * math.log2(chunk / 512)) * GIB
    return 0.40 * GIB * chunk / 512


KV_STEP = 256               # mlx-lm KVCache grows in steps of 256 tokens
KV_BYTES = {16: 2.0, 8: 1.0 + 4 / 64, 4: 0.5 + 4 / 64}  # per element; quantized caches carry bf16 scale and bias per 64
HUB = "https://huggingface.co"


class Unsupported(Exception):
    pass


def fetch_json(url):
    headers = {"User-Agent": f"macfit/{__version__}"}
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith(HUB + "/"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            sys.exit(f"The Hub refused access (HTTP {e.code}). Check the repo id. If the repo is gated or private, "
                     "set HF_TOKEN, or download config.json and the weights and pass the folder.")
        if e.code == 404:
            sys.exit("Not found on the Hub. Check the repo id, for example mlx-community/Qwen3-8B-4bit.")
        sys.exit(f"The Hub returned HTTP {e.code} for {url}")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach huggingface.co ({e.reason}). Pass a local folder instead.")


def weight_files(names):
    """The files mlx-lm loads: model*.safetensors in the top folder (else any *.safetensors there)."""
    top = [n for n in names if "/" not in n and n.endswith(".safetensors")]
    return [n for n in top if n.startswith("model")] or top


def load_model(spec):
    """Return (config dict, weight bytes, source description)."""
    p = Path(spec).expanduser()
    if p.is_dir():
        sizes = {f.name: f.stat().st_size for f in p.glob("*.safetensors")}
        names = weight_files(list(sizes))
        if not names:
            sys.exit(f"No *.safetensors files in {p}")
        if not (p / "config.json").exists():
            sys.exit(f"No config.json in {p}")
        return json.load(open(p / "config.json")), sum(sizes[n] for n in names), str(p)
    if spec.count("/") != 1 or spec.startswith((".", "~", "/")):
        sys.exit(f"{spec} is not a folder here, and it does not look like a Hub repo id (owner/name).")
    info = fetch_json(f"{HUB}/api/models/{spec}?blobs=true")
    sizes = {s["rfilename"]: s.get("size") or 0 for s in info.get("siblings", [])}
    names = weight_files(list(sizes))
    if not names:
        sys.exit(f"{spec}: no .safetensors files on the Hub. macfit supports MLX and safetensors models.")
    rev = info.get("sha", "main")
    config = fetch_json(f"{HUB}/{spec}/resolve/{rev}/config.json")
    return config, sum(sizes[n] for n in names), f"{spec} at {rev[:8]}"


HYBRID_KEYS = ("full_attention_interval", "linear_num_value_heads", "linear_conv_kernel_dim", "layers_block_type",
               "hybrid_override_pattern", "attn_layer_indices", "mamba_d_state", "ssm_state_size", "conv_L_cache")
HYBRID_TYPES = ("qwen3_next", "qwen3_5", "qwen3_5_moe", "falcon_h1", "granitemoehybrid", "nemotron_h", "jamba", "lfm2",
                "mamba", "mamba2", "rwkv", "rwkv7", "recurrent_gemma", "zamba2")


def attention_layout(config):
    """Return (per-layer list of 'full' or a window size, kv heads, head size, query heads). Raise Unsupported."""
    c = config.get("text_config", config)
    if c.get("kv_lora_rank"):
        raise Unsupported("this model uses latent attention (MLA)")
    if c.get("model_type") in HYBRID_TYPES or config.get("model_type") in HYBRID_TYPES or any(k in c for k in HYBRID_KEYS):
        raise Unsupported("this model mixes attention with recurrent or linear-attention layers")
    missing = [k for k in ("num_hidden_layers", "num_attention_heads") if k not in c]
    if missing:
        raise Unsupported(f"config.json leaves out {', '.join(missing)} and relies on library defaults")
    L = c["num_hidden_layers"]
    H = c["num_attention_heads"]
    KV = c.get("num_key_value_heads") or H
    d = c.get("head_dim") or c["hidden_size"] // H
    types = c.get("layer_types")
    window = c.get("sliding_window")
    if types:
        unknown = sorted(set(types) - {"full_attention", "sliding_attention"})
        if unknown:
            raise Unsupported(f"it has layer types {unknown}")
        layers = ["full" if t == "full_attention" else window for t in types]
    elif window and c.get("use_sliding_window"):
        layers = [window] * L
    else:
        layers = ["full"] * L
    return layers, KV, d, H


def kv_cache_bytes(layers, KV, d, tokens, kv_bits):
    slots = math.ceil(tokens / KV_STEP) * KV_STEP
    per = 2 * KV * d * KV_BYTES[kv_bits]
    return sum(per * (slots if w == "full" else min(slots, w)) for w in layers)


def peak_bytes(config, weights, prompt, gen, kv_bits=16, step=2048):
    layers, KV, d, H = attention_layout(config)
    chunk = min(prompt, step)
    kv = kv_cache_bytes(layers, KV, d, prompt + gen + 1, kv_bits)
    inflight = inflight_bytes(chunk)
    scores = 0.0
    if kv_bits != 16:  # the quantized-cache attention path writes scores and softmax output (bf16)
        scores = 2 * H * chunk * prompt * 2
    return {"weights": weights, "kv": kv, "inflight": inflight, "scores": scores, "peak": weights + kv + inflight + scores}


def metal_limit():
    """Metal's recommendedMaxWorkingSetSize, read with ctypes. The same value MLX reports. None if unavailable."""
    try:
        import ctypes
        import ctypes.util
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        device = metal.MTLCreateSystemDefaultDevice()
        if not device:
            return None
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        selector = objc.sel_registerName(b"recommendedMaxWorkingSetSize")
        send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        value = int(send(device, selector))
        return value if value > 0 else None
    except (OSError, AttributeError, TypeError, ValueError):
        return None


def gpu_limit():
    """Return (bytes, how it was found)."""
    value = metal_limit()
    if value:
        return value, "from Metal"
    try:
        wired = int(subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True).stdout or 0)
        if wired > 0:
            return wired * 2**20, "from iogpu.wired_limit_mb"
    except (OSError, ValueError):
        pass
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    return int(ram * 2 / 3), "estimated as 2/3 of RAM; Metal did not answer"


def verdict(share):
    if share > 1:
        return "no"
    return "tight" if share > TIGHT else "yes"


def longest_prompt(config, weights, gen, kv_bits, step, budget):
    """Largest prompt length whose estimate stays within budget, by bisection (0 if none)."""
    if peak_bytes(config, weights, 1, gen, kv_bits, step)["peak"] > budget:
        return 0
    lo, hi = 0, 1 << 22
    while hi - lo > 64:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if peak_bytes(config, weights, mid, gen, kv_bits, step)["peak"] <= budget else (lo, mid)
    return lo


def main(argv=None):
    ap = argparse.ArgumentParser(prog="macfit", description="Estimate peak memory for an mlx-lm model on this Mac.")
    ap.add_argument("model", help="local folder or Hugging Face repo id, for example mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--context", default="4096,8192,32768", help="prompt lengths in tokens, comma-separated")
    ap.add_argument("--max-tokens", type=int, default=512, help="tokens to generate (default 512)")
    ap.add_argument("--kv-bits", type=int, choices=[16, 8, 4], default=16, help="KV cache bits (mlx-lm default 16)")
    ap.add_argument("--prefill-step", type=int, default=2048, help="mlx-lm prefill step size (default 2048)")
    ap.add_argument("--limit-gb", type=float, help="GPU memory limit in GiB (default: read from this Mac)")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--version", action="version", version=f"macfit {__version__}")
    a = ap.parse_args(argv)

    config, weights, source = load_model(a.model)
    limit, how = (int(a.limit_gb * GIB), "given") if a.limit_gb else gpu_limit()
    g = lambda b: f"{b / GIB:.2f} GiB"
    try:
        layers, KV, d, H = attention_layout(config)
    except Unsupported as why:
        fit = "fit" if weights <= limit else f"do NOT fit ({g(weights - limit)} over)"
        if a.json:
            print(json.dumps({"model": source, "weights_bytes": weights, "gpu_limit_bytes": limit, "gpu_limit_source": how,
                              "weights_fit": weights <= limit, "unsupported": str(why), "version": __version__}, indent=1))
            return
        print(f"Model:  {source}\nMac:    GPU memory limit {g(limit)} ({how})")
        print(f"The weights ({g(weights)}) {fit}.")
        print(f"macfit cannot estimate the cache: {why}. Treat the weights as a lower bound on the memory needed.")
        return
    rows = []
    for T in [int(x) for x in a.context.split(",")]:
        est = peak_bytes(config, weights, T, a.max_tokens, a.kv_bits, a.prefill_step)
        share = est["peak"] / limit
        rows.append({"prompt_tokens": T, **est, "share_of_limit": share, "verdict": verdict(share), "fits": share <= 1})
    args = (config, weights, a.max_tokens, a.kv_bits, a.prefill_step)
    max_fit, max_room = longest_prompt(*args, limit), longest_prompt(*args, TIGHT * limit)
    result = {"model": source, "weights_bytes": weights, "layers": len(layers),
              "sliding_layers": sum(w != "full" for w in layers), "kv_heads": KV, "head_dim": d,
              "kv_bits": a.kv_bits, "prefill_step": a.prefill_step, "max_tokens": a.max_tokens,
              "gpu_limit_bytes": limit, "gpu_limit_source": how, "rows": rows,
              "max_prompt_that_fits": max_fit, "max_prompt_with_10pct_room": max_room, "version": __version__}
    if a.json:
        print(json.dumps(result, indent=1))
        return
    print(f"Model:  {source}")
    sliding = f" ({result['sliding_layers']} sliding-window)" if result["sliding_layers"] else ""
    print(f"        weights {g(weights)}, {len(layers)} layers{sliding}, {KV} KV heads, head size {d}")
    print(f"Mac:    GPU memory limit {g(limit)} ({how})")
    print(f"Setup:  {a.kv_bits}-bit KV cache, prefill step {a.prefill_step}, {a.max_tokens} new tokens\n")
    print(f"{'Prompt':>9}  {'KV cache':>10}  {'Peak':>10}  Fits?")
    for r in rows:
        pct = f"{100 * r['share_of_limit']:.0f}% of the limit"
        text = {"yes": f"yes, {pct}", "tight": f"tight, {pct}", "no": f"NO, {g(r['peak'] - limit)} over"}[r["verdict"]]
        print(f"{r['prompt_tokens']:>9,}  {g(r['kv']):>10}  {g(r['peak']):>10}  {text}")
    trained = config.get("text_config", config).get("max_position_embeddings")
    if max_fit:
        print(f"\nLongest prompt with 10% to spare: about {max_room:,} tokens (at the limit: about {max_fit:,}).")
        if trained and trained < max_room:
            print(f"The model itself is trained for up to {trained:,} tokens.")
    else:
        print("\nThe weights alone do not fit.")
    if any(r["verdict"] == "tight" for r in rows):
        print("Tight means within 10% of the limit. Estimates have run up to 4% low, so a tight fit may fail.")
    print("Other apps share this memory, so leave some room. This is an estimate, not a guarantee.")


if __name__ == "__main__":
    main()
