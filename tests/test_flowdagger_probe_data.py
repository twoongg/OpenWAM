"""Dataset plumbing and summary for the FlowDAgger inversion probe."""

from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from openwam.flowdagger.probe_data import pick_windows, probe_dataset_config, summarize


def _train_cfg(**dl):
    base = {"type": "libero", "dataset_dir": "/train/libero", "num_frames": 33, "color_jitter": {"enabled": True}}
    base.update(dl)
    return OmegaConf.create({"dataloader": base})


def test_probe_config_reads_new_demos_with_the_checkpoint_normalization():
    cfg = probe_dataset_config(_train_cfg(), ckpt_dir="/ckpt", dataset_dir="/data/libero_plus")
    assert cfg["dataset_dir"] == "/data/libero_plus"
    assert cfg["normalization_stats_path"] == "/ckpt/normalization_stats.npy"
    assert cfg["color_jitter"]["enabled"] is False
    assert cfg["num_frames"] == 33  # window/rendering settings stay the checkpoint's


def test_probe_config_refuses_readers_without_a_stats_override():
    with pytest.raises(NotImplementedError, match="libero"):
        probe_dataset_config(_train_cfg(type="robotwin"), ckpt_dir="/ckpt", dataset_dir="/d")


class _Windows:
    """Dataset stub: windows 1 and 4 are cut short by the episode end."""

    def __init__(self, n=8, short=(1, 4)):
        self.n, self.short = n, set(short)
        self.reads = []

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.reads.append(idx)
        mask = torch.zeros(4, 6, dtype=torch.bool)
        mask[:, :2] = True
        if idx in self.short:
            mask[2:] = False
        return {"action_mask": mask}


def test_pick_windows_keeps_only_full_windows_deterministically():
    ds = _Windows()
    picked = pick_windows(ds, 4, seed=0)
    assert len(picked) == 4 and len(set(picked)) == 4
    assert not set(picked) & ds.short
    assert picked == pick_windows(_Windows(), 4, seed=0)


def test_pick_windows_returns_what_exists_when_asked_for_too_many():
    assert sorted(pick_windows(_Windows(), 100, seed=0)) == [0, 2, 3, 5, 6, 7]


def _report(base, recon, *, seeds=(), self_recon=None):
    r = {
        "base_mse": base,
        "recon_mse": recon,
        "recon_ratio": recon / base,
        "active_dims": [0, 1],
        "base_per_dim": [base, base],
        "recon_per_dim": [recon, recon],
        "seed_mse": list(seeds),
        "noise_action": {"std": 1.0, "mean": 0.0, "absmax": 3.0, "frac_abs_gt3": 0.0},
        "noise_video": {"std": 1.0, "mean": 0.0, "absmax": 3.0, "frac_abs_gt3": 0.0},
        "noise_action_shift": 0.5,
        "video_drift": 0.01,
        "fp_action_update": [0.0, 1e-3],
        "fp_video_update": [0.0, 1e-4],
    }
    if self_recon is not None:
        r["self_recon_mse"] = self_recon
    return r


def test_summarize_aggregates_reports():
    s = summarize([_report(1.0, 0.1, seeds=(0.8, 0.5), self_recon=1e-6), _report(3.0, 0.9, seeds=(2.0,))])
    assert s["num_samples"] == 2
    assert s["base_mse"]["median"] == pytest.approx(2.0)
    assert s["recon_ratio"]["mean"] == pytest.approx((0.1 + 0.3) / 2)
    assert s["seed_best_mse"]["median"] == pytest.approx((0.5 + 2.0) / 2)
    assert s["self_recon_mse"]["n"] == 1
    assert s["recon_per_dim"] == {"0": pytest.approx(0.5), "1": pytest.approx(0.5)}
    assert s["fp_action_update_last"]["max"] == pytest.approx(1e-3)


def test_summarize_skips_undefined_ratios():
    r = _report(1.0, 0.0)
    r["base_mse"], r["recon_ratio"] = 0.0, float("nan")
    s = summarize([r, _report(1.0, 0.5)])
    assert s["recon_ratio"]["n"] == 1 and not math.isnan(s["recon_ratio"]["mean"])
