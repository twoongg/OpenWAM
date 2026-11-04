"""Integration of the video-expert TTT memory into dual_system / joint_self_attn."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from openwam.model.ttt import VideoTTTMemory
from openwam.model.video_backbone import BlockLoopState
from tests.test_openwam_trainer import _MockVideoBackbone

DIM = 64
HEADS = 4
LAYERS = 4
ACTION_DIM = 7
T_ACTION = 5
TEXT_DIM = 4096
TTT_CFG = {"enabled": True, "layers": [0, 2], "num_heads": 2, "head_dim": 16, "hidden_dim": 32}


class _DeterministicVideoBackbone(_MockVideoBackbone):
    """Mock whose hidden states come from the latents and whose output is the final hidden state.

    Like the real Wan block, the post-attention residual add runs inside a
    ``GateModule`` (an ``nn.Module``), so freezing the backbone with
    ``freeze_modules`` (which wraps every submodule forward in ``no_grad``)
    would cut the residual stream — exactly what TTT meta-training must avoid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from openwam.model.video_backbone.wan.models.dit import GateModule

        self._gate = GateModule()

    def post_attn_at_layer(self, layer_id: int, state: BlockLoopState, attn_out, post_state: dict) -> BlockLoopState:
        state.hidden_states = self._gate(post_state["residual"], 1.0, attn_out)
        return state

    def prepare(self, **inputs) -> BlockLoopState:
        latents = inputs["latents"]
        B, C, f, h, w = latents.shape
        assert C == self._dim, "test latents carry the hidden dim as channels"
        x = latents.flatten(2).transpose(1, 2).contiguous()
        num_tokens = f * h * w
        freq_dim = self._dim // 2
        return BlockLoopState(
            hidden_states=x,
            time_mod=torch.zeros(B, num_tokens, 6, self._dim),
            rope_freqs=torch.polar(torch.ones(num_tokens, 1, freq_dim), torch.zeros(num_tokens, 1, freq_dim)),
            context=torch.zeros(B, 4, self._dim),
            context_mask=torch.ones(B, 4, dtype=torch.bool),
            grid_frames=f,
            grid_height=h,
            grid_width=w,
            extras={},
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        state.hidden_states = state.hidden_states + torch.tanh(self._dit(state.hidden_states))
        return state

    def finalize(self, state: BlockLoopState):
        # Final hidden state laid out as latents (B, C=dim, F, H, W) so compute_loss can use it.
        hs = state.hidden_states
        B, _, D = hs.shape
        return hs.transpose(1, 2).reshape(B, D, state.grid_frames, state.grid_height, state.grid_width)


def _build(ttt_cfg=None):
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

    class _Arch(DualSystemSelfAttnArchitecture):
        def _init_video_backbone(self, _cfg):
            self.video_backbone = _DeterministicVideoBackbone(dim=DIM, num_layers=LAYERS, num_heads=HEADS)

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_interval": 1,
        "dim": DIM,
        "ffn_dim": 4 * DIM,
    }
    if ttt_cfg is not None:
        cfg["ttt"] = ttt_cfg
    torch.manual_seed(0)
    arch = _Arch(cfg=cfg)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _inputs(seed: int, B: int = 1):
    g = torch.Generator().manual_seed(seed)
    return {
        "latents": torch.randn(B, DIM, 2, 2, 2, generator=g),
        "timestep": torch.full((B,), 500.0),
        "context": torch.randn(B, 4, TEXT_DIM, generator=g),
        "context_mask": torch.ones(B, 4, dtype=torch.bool),
    }


def _actions(seed: int, B: int = 1):
    g = torch.Generator().manual_seed(seed + 1000)
    return torch.randn(B, T_ACTION, ACTION_DIM, generator=g), torch.full((B,), 0.5)


def _joint_forward(arch, seed: int, **kw):
    noisy_actions, t_action = _actions(seed)
    return arch(noisy_actions, t_action, **_inputs(seed), **kw)


def _make_ttt_active(arch):
    """Stand-in for a meta-trained memory: give θ_O non-zero weights so reads matter."""
    torch.manual_seed(1)
    for layer in arch.video_ttt.layers:
        nn.init.normal_(arch.video_ttt.branch(layer).o_proj.weight, std=0.5)


def _outputs_equal(a, b) -> bool:
    return all(torch.allclose(x, y) for x, y in zip(a, b))


def test_ttt_disabled_by_default():
    arch = _build()
    assert arch.video_ttt is None
    assert arch.mot_driver.ttt is None


def test_ttt_cfg_builds_memory_on_requested_layers():
    arch = _build(TTT_CFG)
    assert isinstance(arch.video_ttt, VideoTTTMemory)
    assert arch.video_ttt.layers == (0, 2)
    assert arch.mot_driver.ttt is arch.video_ttt


def test_ttt_cfg_rejects_layer_beyond_backbone_depth():
    with pytest.raises(ValueError, match="layer"):
        _build({**TTT_CFG, "layers": [0, LAYERS]})


def test_zero_init_ttt_leaves_joint_forward_unchanged():
    arch = _build(TTT_CFG)
    with_ttt = _joint_forward(arch, seed=3)
    arch.mot_driver.ttt = None
    without_ttt = _joint_forward(arch, seed=3)
    assert _outputs_equal(with_ttt, without_ttt)


def test_active_ttt_changes_both_video_and_action_outputs():
    arch = _build(TTT_CFG)
    _make_ttt_active(arch)
    video_ttt, action_ttt = _joint_forward(arch, seed=3)
    arch.mot_driver.ttt = None
    video_base, action_base = _joint_forward(arch, seed=3)
    assert not torch.allclose(video_ttt, video_base)
    # The residual is added to the video stream only; the action expert sees it through joint attention.
    assert not torch.allclose(action_ttt, action_base)


def test_writing_a_context_video_changes_later_target_forward_until_reset():
    arch = _build(TTT_CFG)
    _make_ttt_active(arch)
    before = _joint_forward(arch, seed=3)

    with torch.no_grad(), arch.video_ttt.writing(lr=1.0):
        _joint_forward(arch, seed=7)  # context episode (e.g. an agent-assisted success)

    after_write = _joint_forward(arch, seed=3)
    assert not _outputs_equal(before, after_write)

    arch.video_ttt.reset()
    assert _outputs_equal(before, _joint_forward(arch, seed=3))


def test_video_only_forward_applies_ttt():
    arch = _build(TTT_CFG)
    _make_ttt_active(arch)
    video_ttt, _ = arch(None, None, **_inputs(3))
    arch.video_ttt.reset()
    for layer in arch.video_ttt.layers:
        nn.init.zeros_(arch.video_ttt.branch(layer).o_proj.weight)
    video_base, _ = arch(None, None, **_inputs(3))
    assert not torch.allclose(video_ttt, video_base)


def test_video_only_forward_can_write_memory():
    arch = _build(TTT_CFG)
    init = [w.detach().clone() for w in arch.video_ttt.fast_weights(0)]
    with torch.no_grad(), arch.video_ttt.writing(lr=1.0):
        arch(None, None, **_inputs(7))
    now = arch.video_ttt.fast_weights(0)
    assert any(not torch.allclose(n[0], i) for n, i in zip(now, init))


def test_pretrained_checkpoint_without_ttt_weights_loads_into_ttt_architecture(tmp_path):
    """A benchmark-finetuned checkpoint (no ``video_ttt.*`` keys) must load strictly;
    the fresh memory stays zero-initialised so the loaded policy is unchanged."""
    base = _build()
    path = str(tmp_path / "checkpoint_step_1.safetensors")
    base.save_checkpoint(path)

    arch = _build(TTT_CFG)
    arch.load_checkpoint(path)

    loaded = arch.state_dict()
    for key, value in base.state_dict().items():
        torch.testing.assert_close(loaded[key], value, msg=key)
    for layer in arch.video_ttt.layers:
        assert torch.count_nonzero(arch.video_ttt.branch(layer).o_proj.weight) == 0


def test_strict_load_still_rejects_missing_backbone_keys(tmp_path):
    from safetensors.torch import save_file

    base = _build()
    partial = {k: v.contiguous() for k, v in base.state_dict().items() if not k.startswith("action_backbone.")}
    path = str(tmp_path / "checkpoint_step_1.safetensors")
    save_file(partial, path)

    arch = _build(TTT_CFG)
    with pytest.raises(RuntimeError, match="action_backbone"):
        arch.load_checkpoint(path)


def test_ttt_checkpoint_round_trips_meta_trained_weights(tmp_path):
    arch = _build(TTT_CFG)
    _make_ttt_active(arch)
    path = str(tmp_path / "checkpoint_step_2.safetensors")
    arch.save_checkpoint(path)

    restored = _build(TTT_CFG)
    restored.load_checkpoint(path)
    for layer in arch.video_ttt.layers:
        torch.testing.assert_close(
            restored.video_ttt.branch(layer).o_proj.weight, arch.video_ttt.branch(layer).o_proj.weight
        )


def _ttt_grads(arch, *, use_gradient_checkpointing: bool):
    arch.zero_grad(set_to_none=True)
    arch.train()
    video, action = _joint_forward(arch, seed=3, use_gradient_checkpointing=use_gradient_checkpointing)
    (video.pow(2).mean() + action.pow(2).mean()).backward()
    return {n: p.grad.detach().clone() for n, p in arch.video_ttt.named_parameters() if p.grad is not None}


def test_gradient_checkpointing_gives_same_ttt_gradients():
    arch = _build(TTT_CFG)
    _make_ttt_active(arch)
    plain = _ttt_grads(arch, use_gradient_checkpointing=False)
    ckpt = _ttt_grads(arch, use_gradient_checkpointing=True)
    assert plain.keys() == ckpt.keys() and plain
    for name in plain:
        torch.testing.assert_close(plain[name], ckpt[name], msg=name)


def test_ttt_only_training_freezes_everything_else_but_keeps_gradient_path():
    """Meta-training updates only the TTT slow weights, and the target loss must still
    reach the earliest TTT layer through the frozen video/action experts."""
    arch = _build(TTT_CFG)
    _make_ttt_active(arch)
    arch.restrict_training_to_ttt()

    trainable = {n for n, p in arch.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("video_ttt.") for n in trainable)
    assert set(arch.get_trainable_modules()) == {"video_ttt"}

    with arch.video_ttt.writing(lr=1.0, create_graph=True):
        _joint_forward(arch, seed=7)
    video, action = _joint_forward(arch, seed=3)
    (video.pow(2).mean() + action.pow(2).mean()).backward()

    first = arch.video_ttt.branch(0)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        grad = getattr(first, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is None for n, p in arch.named_parameters() if not n.startswith("video_ttt."))
