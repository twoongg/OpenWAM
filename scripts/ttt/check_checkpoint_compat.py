#!/usr/bin/env python
"""Check that an OpenWAM checkpoint loads into a TTT-enabled architecture — without its weights.

Runs the real deploy loader (``load_from_checkpoint_dir``) end to end, but feeds
it meta tensors built from the safetensors *header* (names / shapes / dtypes),
and the model is built on the meta device, so no 25 GB weight file has to be
downloaded or held in RAM. Catches key-name or shape drift between the
checkpoint and the TTT-augmented architecture, and exercises the strict-load
rule that lets ``video_ttt.*`` be absent.

Usage (checkpoint dir holds config.yaml + tokenizer/; weights may be absent):

    # header read remotely from the Hugging Face repo
    python scripts/ttt/check_checkpoint_compat.py <ckpt_dir> \
        --hf-repo OpenWAM/OpenWAM-Alpha-Sim-RoboDojo --filename checkpoint_step_60000.safetensors

    # header read from a local safetensors file inside <ckpt_dir>
    python scripts/ttt/check_checkpoint_compat.py <ckpt_dir> --filename checkpoint_step_60000.safetensors
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import sys
import tempfile
from unittest import mock

import torch
from omegaconf import OmegaConf

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64}


def _header(ckpt_dir: str, filename: str, hf_repo: str | None) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    if hf_repo:
        from huggingface_hub import HfApi

        meta = HfApi().parse_safetensors_file_metadata(hf_repo, filename)
        return {k: (_DTYPES[t.dtype], tuple(t.shape)) for k, t in meta.tensors.items()}
    from safetensors import safe_open

    out = {}
    with safe_open(os.path.join(ckpt_dir, filename), framework="pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            out[k] = (_DTYPES[sl.get_dtype()], tuple(sl.get_shape()))
    return out


def _staged_ckpt_dir(ckpt_dir: str, ttt: dict | None) -> str:
    """Temp copy of the checkpoint dir whose config enables TTT (assets symlinked, weights not needed)."""
    tmp = tempfile.mkdtemp(prefix="ttt_ckpt_check_")
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    if ttt is not None:
        OmegaConf.update(cfg, "model.architecture.ttt", ttt, force_add=True)
    OmegaConf.save(cfg, os.path.join(tmp, "config.yaml"))
    for name in os.listdir(ckpt_dir):
        if name != "config.yaml" and not name.endswith(".safetensors") and not name.startswith("."):
            os.symlink(os.path.abspath(os.path.join(ckpt_dir, name)), os.path.join(tmp, name))
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir")
    ap.add_argument("--filename", required=True, help="checkpoint_step_*.safetensors name")
    ap.add_argument("--hf-repo", default=None, help="read the header remotely from this HF model repo")
    ap.add_argument("--no-ttt", action="store_true", help="check the unmodified architecture instead")
    ap.add_argument("--ttt-layers", default=None, help="comma-separated block indices (default: all)")
    ap.add_argument("--ttt-num-heads", type=int, default=8)
    ap.add_argument("--ttt-head-dim", type=int, default=64)
    ap.add_argument("--ttt-hidden-dim", type=int, default=128)
    args = ap.parse_args()

    ttt = None
    if not args.no_ttt:
        ttt = {
            "enabled": True,
            "layers": None if args.ttt_layers is None else [int(x) for x in args.ttt_layers.split(",")],
            "num_heads": args.ttt_num_heads,
            "head_dim": args.ttt_head_dim,
            "hidden_dim": args.ttt_hidden_dim,
            "weight_norm": True,
        }

    header = _header(args.ckpt_dir, args.filename, args.hf_repo)
    print(
        f"checkpoint header: {len(header)} tensors, {sum(torch.Size(s).numel() for _, s in header.values()) / 1e9:.2f}B params"
    )

    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.model.architectures.base import BaseWAMArchitecture

    report: dict = {}
    original_load = BaseWAMArchitecture.load_checkpoint

    def checked_load(self, path, strict=True):
        own = {k: tuple(v.shape) for k, v in self.state_dict().items()}
        report["missing"] = sorted(set(own) - set(header))
        report["unexpected"] = sorted(set(header) - set(own))
        report["shape_mismatch"] = sorted(k for k in set(own) & set(header) if own[k] != header[k][1])
        ttt_mod = getattr(self, "video_ttt", None)
        if ttt_mod is not None:
            report["ttt_layers"] = list(ttt_mod.layers)
            report["ttt_params"] = sum(p.numel() for p in ttt_mod.parameters())
        return original_load(self, path, strict=strict)

    def meta_load_file(_path, device="cpu"):
        return {k: torch.empty(shape, dtype=dtype, device="meta") for k, (dtype, shape) in header.items()}

    staged = _staged_ckpt_dir(args.ckpt_dir, ttt)
    # Build everything on the meta device: the real deploy path allocates ~25 GB of
    # CPU shells, which swaps a small machine to a halt. Meta keeps the same
    # construction code and key / shape / strictness logic with no weight memory.
    # The video backbone is built with an explicit device="cpu", which overrides the
    # default-device context, so that one call is redirected too.
    import openwam.model.video_backbone as vb_pkg

    real_build_vb = vb_pkg.build_video_backbone

    def meta_build_vb(*a, **kw):
        kw["device"] = "meta"
        return real_build_vb(*a, **kw)

    try:
        with (
            torch.device("meta"),
            mock.patch.object(vb_pkg, "build_video_backbone", meta_build_vb),
            mock.patch.object(BaseWAMArchitecture, "load_checkpoint", checked_load),
            mock.patch("safetensors.torch.load_file", meta_load_file),
        ):
            _, arch = load_from_checkpoint_dir(staged, device="meta", ckpt_name=args.filename)
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    non_ttt_missing = [k for k in report["missing"] if not k.startswith("video_ttt.")]
    print(f"architecture: {type(arch).__name__}")
    print(f"missing keys (non-TTT): {len(non_ttt_missing)} {non_ttt_missing[:5]}")
    print(f"missing keys (TTT, expected): {len(report['missing']) - len(non_ttt_missing)}")
    print(f"unexpected keys: {len(report['unexpected'])} {report['unexpected'][:5]}")
    print(f"shape mismatches: {len(report['shape_mismatch'])} {report['shape_mismatch'][:5]}")
    if "ttt_layers" in report:
        print(f"TTT memory: {len(report['ttt_layers'])} layers, {report['ttt_params'] / 1e6:.1f}M params")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak RSS: {peak / (1e9 if sys.platform == 'darwin' else 1e6):.1f} GB")

    ok = not non_ttt_missing and not report["unexpected"] and not report["shape_mismatch"]
    print("RESULT:", "OK — checkpoint loads into this architecture" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
