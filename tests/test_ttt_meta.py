"""TTT meta-training step: write same-task context windows, then the usual loss on the target."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from tests.test_video_ttt_architecture import DIM, TTT_CFG, _build, _make_ttt_active

ACTION_DIM = 7
T_ACTION = 5
TEXT_DIM = 4096


def _inputs(seeds: list[int]) -> dict:
    """compute_loss inputs for a batch of fake samples identified by seed."""
    lat, ctx, act = [], [], []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        lat.append(torch.randn(1, DIM, 2, 2, 2, generator=g))
        ctx.append(torch.randn(1, 4, TEXT_DIM, generator=g))
        act.append(torch.randn(1, T_ACTION, ACTION_DIM, generator=g))
    B = len(seeds)
    return {
        "input_latents": torch.cat(lat),
        "latents": None,
        "context": torch.cat(ctx),
        "context_mask": torch.ones(B, 4, dtype=torch.bool),
        "seq_lens": torch.full((B,), 4),
        "actions": torch.cat(act),
    }


def _arch():
    arch = _build(TTT_CFG)
    arch.init_training_schedulers(1000)
    return arch


def _loss_fn(arch):
    return lambda seeds: arch.compute_loss(**_inputs(seeds))


def test_meta_step_gradient_reaches_ttt_slow_weights_through_context_writes():
    from openwam.train.utils.ttt_meta import ttt_meta_step

    arch = _arch()
    _make_ttt_active(arch)
    arch.restrict_training_to_ttt()
    arch.train()

    out = ttt_meta_step(arch.video_ttt, _loss_fn(arch), [[11], [12]], [3], inner_lr=1.0)
    out["loss"].backward()

    for layer in arch.video_ttt.layers:
        branch = arch.video_ttt.branch(layer)
        for name in ("k_proj", "v_proj"):  # only reachable through the context writes
            grad = getattr(branch, name).weight.grad
            assert grad is not None and grad.abs().sum() > 0, (layer, name)
    assert all(p.grad is None for n, p in arch.named_parameters() if not n.startswith("video_ttt."))


def test_meta_step_context_pass_keeps_no_backbone_graph_by_default():
    """Default: frozen backbone runs under no_grad during context writes (memory)."""
    from openwam.train.utils.ttt_meta import ttt_meta_step

    arch = _arch()
    _make_ttt_active(arch)
    seen = []

    def loss_fn(seeds):
        seen.append(torch.is_grad_enabled())
        return arch.compute_loss(**_inputs(seeds))

    ttt_meta_step(arch.video_ttt, loss_fn, [[11], [12]], [3], inner_lr=1.0)
    assert seen == [False, False, True]

    seen.clear()
    ttt_meta_step(arch.video_ttt, loss_fn, [[11]], [3], inner_lr=1.0, context_backbone_grad=True)
    assert seen == [True, True]


def test_zero_init_ttt_reports_exactly_zero_context_gain():
    """Gain compares the target with/without context under identical noise; a no-op memory must give 0."""
    from openwam.train.utils.ttt_meta import ttt_meta_step

    arch = _arch()
    out = ttt_meta_step(arch.video_ttt, _loss_fn(arch), [[11]], [3], inner_lr=1.0, measure_gain=True)
    assert out["context_gain"].item() == 0.0
    assert out["loss_no_context"].item() == out["loss"].item()


def test_active_ttt_reports_nonzero_context_gain():
    from openwam.train.utils.ttt_meta import ttt_meta_step

    arch = _arch()
    _make_ttt_active(arch)
    out = ttt_meta_step(arch.video_ttt, _loss_fn(arch), [[11]], [3], inner_lr=1.0, measure_gain=True)
    assert out["context_gain"].item() != 0.0


def test_meta_step_leaves_memory_empty():
    from openwam.train.utils.ttt_meta import ttt_meta_step

    arch = _arch()
    init = [w.detach().clone() for w in arch.video_ttt.fast_weights(0)]
    ttt_meta_step(arch.video_ttt, _loss_fn(arch), [[11]], [3], inner_lr=1.0)
    for now, ref in zip(arch.video_ttt.fast_weights(0), init):
        torch.testing.assert_close(now, ref)


def test_pair_batch_is_written_position_by_position_then_target():
    from openwam.train.utils.ttt_meta import TTTMetaConfig, compute_ttt_meta_losses

    arch = _arch()
    calls = []

    def prepare_inputs(samples):
        seeds = [s["seed"] for s in samples]
        calls.append(seeds)
        return _inputs(seeds)

    arch.prepare_inputs = prepare_inputs
    batch = [
        {"seed": 1, "ttt_context": [{"seed": 101}, {"seed": 102}]},
        {"seed": 2, "ttt_context": [{"seed": 201}, {"seed": 202}]},
    ]
    meta = TTTMetaConfig(inner_lr=1.0)
    losses = compute_ttt_meta_losses(arch, arch.video_ttt, batch, lambda_video=1.0, lambda_action=1.0, meta=meta)
    assert calls == [[101, 201], [102, 202], [1, 2]]
    assert set(losses) >= {"total", "video", "action"}
    assert "ttt_context_gain" not in losses  # gain_every=0


def test_gain_is_measured_every_n_steps():
    from openwam.train.utils.ttt_meta import TTTMetaConfig, compute_ttt_meta_losses

    arch = _arch()
    arch.prepare_inputs = lambda samples: _inputs([s["seed"] for s in samples])
    batch = [{"seed": 1, "ttt_context": [{"seed": 101}]}]
    meta = TTTMetaConfig(inner_lr=1.0, gain_every=2)
    measured = []
    for _ in range(4):
        losses = compute_ttt_meta_losses(arch, arch.video_ttt, batch, lambda_video=1.0, lambda_action=1.0, meta=meta)
        measured.append("ttt_context_gain" in losses)
    assert measured == [True, False, True, False]


def test_pair_batch_with_ragged_context_is_rejected():
    from openwam.train.utils.ttt_meta import TTTMetaConfig, compute_ttt_meta_losses

    arch = _arch()
    batch = [{"seed": 1, "ttt_context": [{"seed": 101}]}, {"seed": 2, "ttt_context": []}]
    with pytest.raises(ValueError, match="context"):
        compute_ttt_meta_losses(arch, arch.video_ttt, batch, lambda_video=1.0, lambda_action=1.0, meta=TTTMetaConfig())


def test_resolve_meta_config_from_training_cfg():
    from openwam.train.utils.ttt_meta import resolve_ttt_meta_config

    off = OmegaConf.create({"model": {"architecture": {"ttt": {"enabled": True}}}, "training": {}})
    assert resolve_ttt_meta_config(off) is None

    on = OmegaConf.create(
        {
            "model": {"architecture": {"ttt": {"enabled": True}}},
            "training": {"ttt_meta": {"enabled": True, "inner_lr": 0.05, "gain_every": 10}},
        }
    )
    meta = resolve_ttt_meta_config(on)
    assert meta.inner_lr == 0.05 and meta.gain_every == 10 and meta.context_backbone_grad is False

    no_memory = OmegaConf.create(
        {"model": {"architecture": {"ttt": {"enabled": False}}}, "training": {"ttt_meta": {"enabled": True}}}
    )
    with pytest.raises(ValueError, match="ttt.enabled"):
        resolve_ttt_meta_config(no_memory)
