"""LoRA adapters on the dual_system / joint_self_attn experts.

Integration tests run the *real* Wan block code and the real ActionDiT (tiny
random weights, CPU), so the adapters are exercised on the same ``nn.Linear``
call sites that run with OpenWAM-α weights.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from tests.test_video_ttt_real_wan import ACTION_DIM, TEXT_DIM, TTT_CFG, _inputs, _tiny_ti2v_backbone

ACTION_LORA = {"enabled": True, "rank": 4, "alpha": 8, "experts": ["action"], "modules": ["self_attn", "ffn"]}
BOTH_LORA = {**ACTION_LORA, "experts": ["action", "video"]}


def _build(lora_cfg=None, ttt_cfg=None):
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

    class _Arch(DualSystemSelfAttnArchitecture):
        def _init_video_backbone(self, _cfg):
            self.video_backbone = _tiny_ti2v_backbone()

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        "dim": 96,
        "ffn_dim": 192,
        "text_dim": TEXT_DIM,
        "attention_mask_mode": "mutual",
        "video_attention_mask_mode": "first_frame_causal",
    }
    if lora_cfg is not None:
        cfg["lora"] = lora_cfg
    if ttt_cfg is not None:
        cfg["ttt"] = ttt_cfg
    torch.manual_seed(0)
    arch = _Arch(cfg=cfg)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _joint(arch, seed: int, **kw):
    g = torch.Generator().manual_seed(seed + 1)
    actions = torch.randn(1, 5, ACTION_DIM, generator=g)
    return arch(actions, torch.tensor([0.5]), **_inputs(seed), **kw)


def _activate(arch):
    """Stand-in for a trained adapter: non-zero B so the low-rank update matters."""
    torch.manual_seed(1)
    for adapter in arch.lora.adapters.values():
        nn.init.normal_(adapter.lora_B, std=0.5)


def _assert_same(a, b):
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def _expected_targets(prefix: str, layers: int = 2) -> set[str]:
    names = set()
    for i in range(layers):
        names |= {f"{prefix}.blocks.{i}.self_attn.{p}" for p in "qkvo"}
        names |= {f"{prefix}.blocks.{i}.ffn.0", f"{prefix}.blocks.{i}.ffn.2"}
    return names


# --- adapter module -------------------------------------------------------------


def test_adapter_starts_as_exact_zero_update():
    from openwam.model.lora import LoRAAdapter

    adapter = LoRAAdapter(8, 6, rank=2, alpha=4)
    out = adapter(torch.randn(3, 8))
    assert out.shape == (3, 6)
    assert torch.count_nonzero(out) == 0
    assert adapter.scale == 2.0


def test_adapter_output_follows_input_dtype():
    from openwam.model.lora import LoRAAdapter

    adapter = LoRAAdapter(8, 6, rank=2, alpha=2)
    nn.init.normal_(adapter.lora_B)
    assert adapter(torch.randn(3, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16


# --- architecture wiring ----------------------------------------------------------


def test_lora_disabled_by_default():
    arch = _build()
    assert arch.lora is None
    assert arch._optional_checkpoint_prefixes() == ()


def test_lora_targets_follow_experts_and_modules():
    action_only = _build(ACTION_LORA).lora.targets
    assert set(action_only) == _expected_targets("action_backbone")
    assert len(action_only) == len(set(action_only))

    both = _build(BOTH_LORA).lora.targets
    assert set(both) == _expected_targets("action_backbone") | _expected_targets("video_backbone.dit")


def test_lora_rejects_unknown_expert_or_module():
    with pytest.raises(ValueError, match="experts"):
        _build({**ACTION_LORA, "experts": ["vision"]})
    with pytest.raises(ValueError, match="modules"):
        _build({**ACTION_LORA, "modules": ["mlp"]})


def test_lora_rejects_expert_without_matching_layers():
    from tests.test_video_ttt_architecture import _build as build_mock  # mock video backbone has no Wan blocks

    with pytest.raises(ValueError, match="video"):
        build_mock(lora_cfg={**ACTION_LORA, "experts": ["video"]})


def test_ttt_and_lora_are_both_optional_in_checkpoints():
    assert _build(ACTION_LORA, ttt_cfg=TTT_CFG)._optional_checkpoint_prefixes() == ("video_ttt.", "lora.")


def test_pretrained_checkpoint_without_lora_loads_and_is_unchanged(tmp_path, caplog):
    base = _build()
    path = str(tmp_path / "base.safetensors")
    base.save_checkpoint(path)

    arch = _build(BOTH_LORA)
    with caplog.at_level(logging.WARNING):
        arch.load_checkpoint(path)  # strict: only lora.* may be missing
    assert "lora." in caplog.text
    _assert_same(_joint(arch, seed=3), _joint(base, seed=3))


def test_active_lora_changes_action_prediction():
    arch = _build(ACTION_LORA)
    before = _joint(arch, seed=3)
    _activate(arch)
    after = _joint(arch, seed=3)
    assert not torch.allclose(before[1], after[1])


def test_reset_parameters_returns_to_base_behaviour():
    arch = _build(BOTH_LORA)
    base = _joint(arch, seed=3)
    _activate(arch)
    assert not torch.allclose(base[1], _joint(arch, seed=3)[1])
    arch.lora.reset_parameters()
    _assert_same(_joint(arch, seed=3), base)


def test_lora_checkpoint_round_trips_trained_weights(tmp_path):
    trained = _build(BOTH_LORA)
    _activate(trained)
    path = str(tmp_path / "lora.safetensors")
    trained.save_checkpoint(path)

    fresh = _build(BOTH_LORA)
    fresh.load_checkpoint(path)
    _assert_same(_joint(fresh, seed=3), _joint(trained, seed=3))


def test_lora_only_training_reaches_every_adapter_through_checkpointed_blocks():
    arch = _build(BOTH_LORA)
    _activate(arch)
    arch.restrict_training_to_lora()
    arch.train()

    trainable = {n for n, p in arch.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("lora.") for n in trainable)
    assert set(arch.get_trainable_modules()) == {"lora"}

    video, action = _joint(arch, seed=3, use_gradient_checkpointing=True)
    (video.pow(2).mean() + action.pow(2).mean()).backward()

    for key, adapter in arch.lora.adapters.items():
        for pname in ("lora_A", "lora_B"):
            grad = getattr(adapter, pname).grad
            assert grad is not None and grad.abs().sum() > 0, f"{key}.{pname}"
    assert all(p.grad is None for n, p in arch.named_parameters() if not n.startswith("lora."))


def test_restrict_training_to_lora_requires_lora():
    with pytest.raises(RuntimeError, match="lora"):
        _build().restrict_training_to_lora()


def test_deploy_compile_is_skipped_when_lora_is_enabled():
    plain = _build()
    plain.apply_compile_optimizations({"enabled": True})
    assert plain._compiled_mot_run_joint_loop is not None  # the check below is meaningful

    arch = _build(ACTION_LORA)
    arch.apply_compile_optimizations({"enabled": True})
    assert arch._compiled_mot_run_joint_loop is None
    assert not arch.video_backbone._wan_blocks_compile_enabled
