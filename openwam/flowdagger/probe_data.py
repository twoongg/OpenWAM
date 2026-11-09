"""Dataset plumbing and aggregation for the inversion probe (scripts/flowdagger/inversion_probe.py)."""

from __future__ import annotations

import math
import os
from statistics import mean, median
from typing import Iterable, Optional

import numpy as np
from omegaconf import OmegaConf


def probe_dataset_config(train_cfg, *, ckpt_dir: str, dataset_dir: str, split: str = "train") -> dict:
    """The checkpoint's own dataloader config, pointed at new demos.

    Window length, video stride, canvas layout and normalization mode stay the
    checkpoint's, and the reader normalizes with the checkpoint's saved stats
    (``<ckpt_dir>/normalization_stats.npy``) so demo actions land in the exact
    space the model was trained in, not in stats recomputed from the new demos.
    """
    dl = OmegaConf.to_container(train_cfg.dataloader, resolve=True)
    if dl.get("type") != "libero":
        raise NotImplementedError(
            f"inversion probe only supports the libero reader (got {dl.get('type')!r}): it is the reader "
            "that accepts normalization_stats_path, which keeps demo actions in the checkpoint's space."
        )
    dl["dataset_dir"] = str(dataset_dir)
    dl["normalization_stats_path"] = os.path.join(ckpt_dir, "normalization_stats.npy")
    dl["split"] = split
    jitter = dl.get("color_jitter") or {}
    dl["color_jitter"] = {**jitter, "enabled": False}
    return dl


def pick_windows(dataset, n: int, *, seed: int) -> list[int]:
    """Up to ``n`` random window indices whose every action step is supervised.

    Windows cut short by an episode end carry padded tail actions; they are
    skipped so every probed chunk is a full demo chunk.
    """
    order = np.random.default_rng(seed).permutation(len(dataset))
    picked: list[int] = []
    for idx in order:
        mask = dataset[int(idx)]["action_mask"]
        if bool(mask.any(-1).all()):
            picked.append(int(idx))
            if len(picked) == n:
                break
    return picked


def _stats(values: Iterable[Optional[float]]) -> dict:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}


def summarize(reports: list[dict]) -> dict:
    """Aggregate per-window probe reports (see ``probe_action_target``)."""
    summary: dict = {"num_samples": len(reports)}
    for key in ("base_mse", "recon_mse", "recon_ratio", "noise_action_shift", "video_drift", "self_recon_mse"):
        summary[key] = _stats(r.get(key) for r in reports)
    summary["seed_best_mse"] = _stats(min(r["seed_mse"]) if r.get("seed_mse") else None for r in reports)
    for stream in ("noise_action", "noise_video"):
        for stat in ("std", "absmax", "frac_abs_gt3"):
            summary[f"{stream}_{stat}"] = _stats(r[stream][stat] for r in reports)
    summary["fp_action_update_last"] = _stats(r["fp_action_update"][-1] for r in reports if r["fp_action_update"])
    summary["fp_video_update_last"] = _stats(r["fp_video_update"][-1] for r in reports if r["fp_video_update"])
    for key in ("base_per_dim", "recon_per_dim"):
        per: dict[str, list[float]] = {}
        for r in reports:
            for dim, value in zip(r["active_dims"], r[key]):
                per.setdefault(str(dim), []).append(float(value))
        summary[key] = {dim: mean(vals) for dim, vals in per.items()}
    return summary


def format_summary(summary: dict, label: str = "") -> str:
    """Human-readable table of the headline numbers."""

    def row(name: str, key: str) -> str:
        s = summary.get(key, {"n": 0})
        if not s.get("n"):
            return f"  {name:<34} (none)"
        return f"  {name:<34} median {s['median']:.4g}   mean {s['mean']:.4g}   max {s['max']:.4g}   (n={s['n']})"

    lines = [f"Inversion probe{f' [{label}]' if label else ''}: {summary['num_samples']} windows"]
    lines += [
        row("deployed vs demo (base_mse)", "base_mse"),
        row("best of extra seeds vs demo", "seed_best_mse"),
        row("inverted noise -> demo (recon_mse)", "recon_mse"),
        row("recon_mse / base_mse", "recon_ratio"),
        row("self-check floor (self_recon_mse)", "self_recon_mse"),
        row("action noise std (unit gauss = 1)", "noise_action_std"),
        row("action noise |w|>3 frac (0.0027)", "noise_action_frac_abs_gt3"),
        row("video noise std", "noise_video_std"),
        row("video drift (rel. MSE)", "video_drift"),
        row("last-step FP update, action", "fp_action_update_last"),
    ]
    dims = summary.get("base_per_dim", {})
    if dims:
        lines.append("  per active dim (base -> recon MSE):")
        for dim in dims:
            lines.append(f"    dim {dim:>3}: {dims[dim]:.4g} -> {summary['recon_per_dim'][dim]:.4g}")
    return "\n".join(lines)


__all__ = ["format_summary", "pick_windows", "probe_dataset_config", "summarize"]
