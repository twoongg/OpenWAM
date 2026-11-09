"""OpenWAMTrainer wiring for LoRA training (model.architecture.lora)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tests.test_ttt_meta import _inputs
from tests.test_video_ttt_architecture import TTT_CFG, _build

LORA_CFG = {"enabled": True, "rank": 4, "alpha": 4, "experts": ["action"], "modules": ["self_attn", "ffn"]}


def _trainer(monkeypatch, *, lora: dict | None, ttt_meta: dict | None = None):
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
        arch = _build(TTT_CFG if ttt_meta else None, lora_cfg=lora)
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
    architecture = {"framework": "dual_system", "variant": "joint_self_attn"}
    if lora is not None:
        architecture["lora"] = lora
    if ttt_meta is not None:
        architecture["ttt"] = TTT_CFG
    cfg = OmegaConf.create({"training": training, "model": {"architecture": architecture, "freeze": []}})
    trainer = OpenWAMTrainer(cfg, accelerator=None, dataset=None)
    arch = holder["arch"]
    arch.prepare_inputs = lambda samples: _inputs([s["seed"] for s in samples])
    return trainer, arch


def test_lora_training_optimizes_only_the_adapters(monkeypatch):
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer, arch = _trainer(monkeypatch, lora=LORA_CFG)
    lora_params = {id(p) for p in arch.lora.parameters()}
    assert lora_params
    assert {id(p) for p in build_trainable_parameters(trainer)} == lora_params

    losses = trainer.compute_loss([{"seed": 1}])
    assert set(losses) == {"total", "video", "action"}
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in arch.lora.parameters())
    assert all(p.grad is None for n, p in arch.named_parameters() if not n.startswith("lora."))


def test_plain_training_without_lora_keeps_backbones_trainable(monkeypatch):
    _, arch = _trainer(monkeypatch, lora=None)
    assert arch.lora is None
    assert any(p.requires_grad for n, p in arch.named_parameters() if n.startswith("action_backbone."))


def test_lora_and_ttt_meta_training_cannot_be_combined(monkeypatch):
    with pytest.raises(ValueError, match="lora"):
        _trainer(monkeypatch, lora=LORA_CFG, ttt_meta={"enabled": True})
