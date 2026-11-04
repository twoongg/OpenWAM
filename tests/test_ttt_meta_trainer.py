"""OpenWAMTrainer wiring for video-TTT meta-training (training.ttt_meta)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tests.test_ttt_meta import _inputs
from tests.test_video_ttt_architecture import TTT_CFG, _build


def _trainer(monkeypatch, *, ttt_meta: dict | None):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    holder = {}

    def _fake_resolve(model_cfg):  # noqa: ARG001
        return type(
            "Resolved",
            (),
            {
                "registry_name": "dual_system_self_attn",
                "params": {},
                "canonical": type("Canonical", (), {"framework": "dual_system", "variant": "joint_self_attn"})(),
            },
        )()

    def _fake_build(name, params):  # noqa: ARG001
        arch = _build(TTT_CFG)
        arch.set_dtype_device = lambda *a, **k: None
        holder["arch"] = arch
        return arch

    monkeypatch.setattr("openwam.model.resolve_architecture_config", _fake_resolve)
    monkeypatch.setattr("openwam.model.build_architecture", _fake_build)

    training = {
        "initialize_model_on_cpu": False,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
        "lambda_video": 1.0,
        "lambda_action": 1.0,
    }
    if ttt_meta is not None:
        training["ttt_meta"] = ttt_meta
    cfg = OmegaConf.create(
        {
            "training": training,
            "model": {
                "architecture": {"framework": "dual_system", "variant": "joint_self_attn", "ttt": TTT_CFG},
                "freeze": [],
            },
        }
    )
    trainer = OpenWAMTrainer(cfg, accelerator=None, dataset=None)
    arch = holder["arch"]
    calls = []

    def prepare_inputs(samples):
        calls.append([s["seed"] for s in samples])
        return _inputs([s["seed"] for s in samples])

    arch.prepare_inputs = prepare_inputs
    return trainer, arch, calls


PAIR_BATCH = [{"seed": 1, "ttt_context": [{"seed": 101}, {"seed": 102}]}]
PLAIN_BATCH = [{"seed": 1}]


def test_ttt_meta_enabled_trains_only_ttt_and_runs_pair_step(monkeypatch):
    trainer, arch, calls = _trainer(monkeypatch, ttt_meta={"enabled": True, "inner_lr": 1.0, "gain_every": 1})
    trainable = {n for n, p in arch.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("video_ttt.") for n in trainable)

    losses = trainer.compute_loss(PAIR_BATCH)
    assert calls == [[101], [102], [1], [1]]  # 2 context writes, target, target without context (gain)
    assert {"total", "video", "action", "ttt_context_gain"} <= set(losses)
    losses["total"].backward()


def test_plain_training_path_unchanged_without_ttt_meta(monkeypatch):
    trainer, arch, calls = _trainer(monkeypatch, ttt_meta=None)
    assert any(p.requires_grad for n, p in arch.named_parameters() if not n.startswith("video_ttt."))
    losses = trainer.compute_loss(PLAIN_BATCH)
    assert calls == [[1]]
    assert set(losses) == {"total", "video", "action"}


def test_ttt_meta_with_plain_dataloader_is_rejected(monkeypatch):
    trainer, _, _ = _trainer(monkeypatch, ttt_meta={"enabled": True})
    with pytest.raises(ValueError, match="robodojo_ttt_pairs"):
        trainer.compute_loss(PLAIN_BATCH)


def test_pair_dataloader_without_ttt_meta_is_rejected(monkeypatch):
    trainer, _, _ = _trainer(monkeypatch, ttt_meta=None)
    with pytest.raises(ValueError, match="ttt_meta"):
        trainer.compute_loss(PAIR_BATCH)


def test_reduce_ttt_metrics_single_process():
    import torch

    from openwam.train.utils.ttt_meta import reduce_ttt_metrics

    out = reduce_ttt_metrics(None, {"total": torch.tensor(1.0), "ttt_context_gain": torch.tensor(0.25)})
    assert out == {"ttt_context_gain": 0.25}
    assert reduce_ttt_metrics(None, {"total": torch.tensor(1.0)}) == {}
