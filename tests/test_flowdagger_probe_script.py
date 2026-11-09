"""End-to-end wiring of scripts/flowdagger/inversion_probe.py on the tiny model with a fake demo set."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from tests.test_flowdagger_inversion import ACTION_DIM, T_ACTION, _arch, _fake_preprocess

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "flowdagger" / "inversion_probe.py"


class _Demos:
    def __init__(self):
        g = torch.Generator().manual_seed(3)
        self.actions = [0.3 * torch.randn(T_ACTION, ACTION_DIM, generator=g) for _ in range(5)]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        mask = torch.zeros(T_ACTION, ACTION_DIM, dtype=torch.bool)
        mask[:, :4] = True
        if idx == 2:
            mask[-2:] = False  # short window: must be skipped
        return {
            "action": self.actions[idx],
            "action_mask": mask,
            "proprio": torch.zeros(1, ACTION_DIM),
            "prompt": "put the bowl on the plate",
            "first_frame_image": [Image.new("RGB", (8, 8))],
        }


def test_probe_script_runs_end_to_end(tmp_path, monkeypatch, capsys):
    import openwam.dataloader.registry as registry
    import openwam.deploy.model_loader as model_loader

    arch = _arch()
    _fake_preprocess(arch.video_backbone)
    train_cfg = OmegaConf.create(
        {
            "dataloader": {
                "type": "libero",
                "dataset_dir": "/train",
                "num_frames": T_ACTION + 1,
                "video_stride": 1,
                "height": 8,
                "width": 8,
            }
        }
    )
    seen = {}

    def fake_load(ckpt_dir, device, ckpt_name=None):
        seen["load"] = (ckpt_dir, device)
        return train_cfg, arch

    def fake_build(cfg, split="train"):
        seen["dataset_cfg"] = cfg
        return _Demos()

    monkeypatch.setattr(model_loader, "load_from_checkpoint_dir", fake_load)
    monkeypatch.setattr(registry, "build_dataset", fake_build)

    spec = importlib.util.spec_from_file_location("inversion_probe", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "probe.json"
    script.main(
        [
            "--ckpt-dir",
            "/ckpt",
            "--dataset-dir",
            "/demos",
            "--num-samples",
            "3",
            "--denoise-steps",
            "3",
            "--extra-seeds",
            "1",
            "--self-check",
            "1",
            "--fp-iters",
            "20",
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text())
    assert seen["dataset_cfg"]["dataset_dir"] == "/demos"
    assert seen["dataset_cfg"]["normalization_stats_path"] == "/ckpt/normalization_stats.npy"
    assert payload["run"]["denoise_steps"] == 3
    assert len(payload["run"]["schedule"]) == 4
    reports = payload["reports"]
    assert len(reports) == 3 and 2 not in {r["index"] for r in reports}
    assert "self_recon_mse" in reports[0] and "self_recon_mse" not in reports[1]
    assert all(len(r["seed_mse"]) == 1 for r in reports)
    assert payload["summary"]["num_samples"] == 3
    assert "Inversion probe" in capsys.readouterr().out
