#!/usr/bin/env python3
"""Download google/embeddinggemma-2 (text tower) and export fixed-shape ANE artifacts.

The sidecar loads what this writes. Weights are not committed.

Toolchain, pinned in scripts/requirements-embeddinggemma2.txt:

    coreai-torch==0.4.2
    coreai-core==1.0.0b2
    coreai-opt==0.2.1
    torch==2.11.0

That is the stack Anemll used for anemll/anemll-embeddinggemma-2-ane and the
one cowboycoderhq needed (0.4.2; later converters mis-compile the padding
mask). Do not run Apple's `coreai.embedding.export embeddinggemma-2` on the
macOS preset: that preset is 4-bit and the runtime then prefers the GPU.
This script never calls it. Weights stay fp16 (int8 only if you pass
--int8 and coreai-opt actually exposes an int8 path). 4-bit is refused.

What the export does, and why:

- The token table stays on the host (262144 x 512). The graph takes
  inputs_embeds, not ids. A table that size does not belong on the ANE.
- One package per length bucket (64, 128, 256, 512). Fixed shapes are what
  keep the graph on the Neural Engine; a ranged shape falls back.
- Attention is eager matmul, not fused SDPA. The ANE compiler rejects the
  fused op and Core ML / Core AI will not tell you; the op just runs on the
  CPU or the GPU.
- Masks use -1e4, built with torch.where, not -inf. -inf poisons fp16, and
  an arithmetic mask was wrong for padded batches in coreai-torch 0.4.2
  (apple/coreai-torch#137).
- RMSNorm runs in fp32. fp16 RMSNorm overflows and the ANE also flushes
  values under about 6e-5 to zero.
- Optional replacement of nn.Linear with a 1x1 conv on (B, C, 1, S). That is
  the ANE's preferred layout. If the export fails, the script retries without
  it and says so.
- Mean pool, then the 512->768 projection if the backbone has not already
  applied it, then L2. Matryoshka truncation happens in the sidecar.

Run on the Mac (macOS 27, Apple Silicon), not on Linux:

    python3.13 -m venv .venv-eg2
    .venv-eg2/bin/python -m pip install -r scripts/requirements-embeddinggemma2.txt
    .venv-eg2/bin/python scripts/convert_embeddinggemma2.py --out ~/Library/Application\\ Support/FSearch/models/embeddinggemma-2

--also-coreml writes a Core ML fallback (mlprogram, fp16) for the same
buckets. The sidecar loads those with CPU + Neural Engine and not the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

BUCKETS = (64, 128, 256, 512)
MODEL_ID = "google/embeddinggemma-2"
# cowboycoderhq and Anemll both pin this converter. Do not bump it here.
REQUIRED = {
    "coreai_torch": "0.4.2",
    "coreai": "1.0.0b2",
}


def _die(msg: str) -> None:
    print(f"convert_embeddinggemma2: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _check_host() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _die("Core AI conversion runs on an Apple Silicon Mac (this machine is " f"{platform.system()} {platform.machine()}).")


def _need(mod: str, expected: str | None) -> None:
    try:
        imported = __import__(mod)
    except ImportError as exc:
        _die(f"missing {mod} ({exc}). pip install -r scripts/requirements-embeddinggemma2.txt")
    if expected is None:
        return
    got = getattr(imported, "__version__", "")
    if expected not in str(got):
        _die(f"{mod} is {got}, this recipe needs {expected}. A newer coreai-torch mis-compiles the mask.")


def _patch_sdpa(torch) -> None:
    """Eager attention: matmul, -1e4 mask via torch.where, softmax, matmul."""

    def eager(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **_kwargs):
        del dropout_p
        dim = q.shape[-1]
        scale = (dim ** -0.5) if scale is None else scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            s = scores.shape[-1]
            causal = torch.ones(s, s, dtype=scores.dtype, device=scores.device).tril()
            scores = scores + torch.where(causal > 0, torch.zeros_like(scores[..., :s, :s]), torch.full_like(scores[..., :s, :s], -1e4))
        if attn_mask is not None:
            mask = attn_mask
            if mask.dtype.is_floating_point:
                mask = torch.where(torch.isneginf(mask), torch.full_like(mask, -1e4), mask)
                mask = torch.where(mask < -1e4, torch.full_like(mask, -1e4), mask)
            scores = scores + mask
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    torch.nn.functional.scaled_dot_product_attention = eager


def _patch_rmsnorm(torch) -> int:
    """RMSNorm in fp32 so fp16 and the ANE's subnormal flush do not change it."""
    patched = 0
    cls = getattr(torch.nn, "RMSNorm", None)
    if cls is None:
        return 0

    def forward(self, x):
        inner = x.to(torch.float32)
        var = inner.pow(2).mean(dim=-1, keepdim=True)
        inner = inner * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            inner = inner * self.weight.to(torch.float32)
        return inner.to(x.dtype)

    cls.forward = forward
    patched += 1
    return patched


def _linears_to_conv(torch, module) -> int:
    """nn.Linear -> 1x1 conv on (B, C, 1, S). Best-effort; the caller retries."""
    nn = torch.nn
    swapped = 0
    for name, child in list(module.named_modules()):
        if child is module or not isinstance(child, nn.Linear):
            continue
        parent = module
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        conv = nn.Conv2d(child.in_features, child.out_features, kernel_size=1, bias=child.bias is not None)
        with torch.no_grad():
            conv.weight.copy_(child.weight.detach().view(child.out_features, child.in_features, 1, 1))
            if child.bias is not None:
                conv.bias.copy_(child.bias.detach())

        class AsConv(nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv

            def forward(self, x):
                # x is (B, S, C). The ANE wants (B, C, 1, S).
                y = self.conv(x.transpose(1, 2).unsqueeze(2))
                return y.squeeze(2).transpose(1, 2)

        setattr(parent, parts[-1], AsConv(conv))
        swapped += 1
    return swapped


class TextTower:
    """Filled in after torch imports. See _make_tower."""


def _make_tower(torch, backbone, proj, embed_dim: int):
    nn = torch.nn
    F = torch.nn.functional

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj = proj
            self.embed_dim = embed_dim

        def forward(self, inputs_embeds, attention_mask):
            out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if self.proj is not None and pooled.shape[-1] != self.embed_dim:
                pooled = self.proj(pooled)
            return F.normalize(pooled.float(), p=2, dim=-1)

    return Tower().eval()


def _load_text(torch):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model.eval()
    text = getattr(model, "text_model", None) or getattr(model, "model", model)
    hidden = int(getattr(text.config, "hidden_size", 512))
    embed_dim = int(getattr(model.config, "embedding_dim", getattr(text.config, "embedding_dim", 768)))
    proj = None
    for name in ("embed_projection", "projection", "dense"):
        proj = getattr(model, name, None) or getattr(text, name, None)
        if proj is not None:
            break
    table = model.get_input_embeddings().weight.detach().to(torch.float32).cpu()
    return text, proj, hidden, embed_dim, table


def _export_coreai(torch, tower, seq: int, hidden: int, out: Path) -> None:
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision

    embeds = torch.zeros(1, seq, hidden)
    mask = torch.ones(1, seq)
    mask[:, -1] = 0
    ep = torch.export.export(tower, (embeds, mask), strict=False)
    ep = ep.run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(
        ep,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["embedding"],
        entrypoint_name=f"embed_{seq}",
    )
    program = converter.to_coreai()
    program.optimize()
    dest = out / "coreai" / f"embed_{seq}.aimodel"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    program.save_asset(dest)
    nbytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"wrote {dest} ({nbytes / 1e6:.1f} MB)")
    if nbytes > 1 << 30:
        print(f"WARNING: {dest.name} is over 1 GiB. The ANE silently falls off a model that large.", file=sys.stderr)


def _export_coreml(torch, tower, seq: int, hidden: int, out: Path) -> None:
    import coremltools as ct

    embeds = torch.zeros(1, seq, hidden)
    mask = torch.ones(1, seq)
    traced = torch.jit.trace(tower, (embeds, mask))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=embeds.shape, dtype=torch.float32),
            ct.TensorType(name="attention_mask", shape=mask.shape, dtype=torch.float32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    dest = out / "coreml" / f"embed_{seq}.mlpackage"
    dest.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dest))
    print(f"wrote {dest}")
    print("load this package with MLComputeUnits.cpuAndNeuralEngine. Do not use .all: that allows the GPU.")


def _write_table(torch, table, out: Path, hidden: int) -> float:
    bf = table.to(torch.bfloat16).contiguous().view(torch.int16).numpy()
    path = out / "embed_tokens.bf16"
    path.write_bytes(bf.tobytes())
    scale = math.sqrt(hidden)
    print(f"wrote {path} rows={table.shape[0]} hidden={table.shape[1]} scale={scale:.6f} (host multiplies)")
    return scale


def _try_int8() -> str:
    """Never 4-bit. Int8 only if this coreai-opt build has a obvious entry point."""
    try:
        import coreai_opt
    except ImportError:
        return "coreai-opt is not importable; staying on fp16"
    for name in ("quantize_weights_int8", "int8_weights"):
        if hasattr(coreai_opt, name):
            return f"found coreai_opt.{name} but it is not wired: staying on fp16 so a 4-bit preset cannot sneak in"
    return "no int8 entry point in this coreai-opt; staying on fp16 (4-bit is intentionally not used)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=False, help="model directory the sidecar reads")
    parser.add_argument("--buckets", type=int, nargs="+", default=list(BUCKETS))
    parser.add_argument("--also-coreml", action="store_true")
    parser.add_argument("--no-conv", action="store_true", help="do not rewrite Linear as 1x1 conv")
    parser.add_argument("--int8", action="store_true", help="refused unless an int8 API exists; never 4-bit")
    args = parser.parse_args()
    if args.out is None:
        _die("pass --out, for example ~/Library/Application Support/FSearch/models/embeddinggemma-2")
    _check_host()
    _need("torch", "2.11")
    _need("coreai_torch", REQUIRED["coreai_torch"])
    _need("coreai", "1.0.0b2")
    _need("transformers", None)

    import torch

    _patch_sdpa(torch)
    norms = _patch_rmsnorm(torch)
    print(f"RMSNorm patched ({norms}); SDPA replaced with eager matmul and -1e4 masks")
    if args.int8:
        print(_try_int8())

    text, proj, hidden, embed_dim, table = _load_text(torch)
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    scale = _write_table(torch, table, out, hidden)

    def build(conv: bool):
        tower = _make_tower(torch, text, proj, embed_dim)
        if conv:
            n = _linears_to_conv(torch, tower)
            print(f"rewrote {n} Linear modules as 1x1 convs")
        return tower

    for seq in args.buckets:
        if seq > 512:
            _die(f"bucket {seq} is above 512. Longer inputs are chunked by fsearch, not by the ANE graph.")
        try:
            _export_coreai(torch, build(conv=not args.no_conv), seq, hidden, out)
        except Exception as exc:  # noqa: BLE001 - retry without the layout rewrite, then surface the real error
            if args.no_conv:
                _die(f"embed_{seq} failed: {type(exc).__name__}: {exc}")
            print(f"embed_{seq} failed with the conv layout ({type(exc).__name__}: {exc}); retrying without it")
            _export_coreai(torch, build(conv=False), seq, hidden, out)
        if args.also_coreml:
            _export_coreml(torch, build(conv=False), seq, hidden, out)

    manifest = {
        "model": MODEL_ID,
        "revision_note": "text tower only (270M). Vision and audio are not exported.",
        "native_dim": embed_dim,
        "hidden": hidden,
        "embed_scale": scale,
        "embed_table": "embed_tokens.bf16",
        "embed_rows": int(table.shape[0]),
        "buckets": list(args.buckets),
        "mask": "one_is_token",
        "inputs": {"embeds": "inputs_embeds", "mask": "attention_mask"},
        "output": "embedding",
        "dtype": "fp16",
        "quantization": "none",
        "compute": "neuralEngine",
        "coreai_torch": REQUIRED["coreai_torch"],
        "notes": [
            "Host applies embed_scale to the bf16 table, then passes inputs_embeds.",
            "Mask is 1 on real tokens and 0 on padding.",
            "Do not load with the GPU. The sidecar asks for the Neural Engine.",
            "Apple's macOS coreai.embedding.export preset is 4-bit and prefers the GPU. It was not used.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'manifest.json'}")
    print("copy tokenizer.json and tokenizer_config.json from the Hugging Face snapshot into this directory.")
    print("The sidecar refuses to start without them. huggingface-cli download " f"{MODEL_ID} tokenizer.json tokenizer_config.json --local-dir {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
