"""Video TTT memory through the *real* Wan block code (tiny random weights, CPU).

The mock-backbone tests cover the plumbing; these run the actual Wan
``pre_attn_at_layer`` / ``post_attn_at_layer`` / ``run_block`` (AdaLN
modulation, GateModule residuals, cross-attention, FFN) so the TTT hook is
exercised against the code that runs with OpenWAM-α weights.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_LAYERS = 2
TEXT_DIM = 32
ACTION_DIM = 7
TTT_CFG = {"enabled": True, "layers": None, "num_heads": 2, "head_dim": 16, "hidden_dim": 32}


def _tiny_ti2v_backbone():
    """Two-block TI2V-style WanModel (mirrors tests/test_video_backbone_smoke._build_tiny_wan_backbone)."""
    from types import SimpleNamespace

    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan21

    # head_dim = 96 / 4 = 24 splits evenly into Wan's 3D RoPE (f, h, w) bands.
    model = WanModel(
        dim=96,
        in_dim=4,
        ffn_dim=192,
        out_dim=4,
        text_dim=TEXT_DIM,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=NUM_LAYERS,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
    )
    pipe = SimpleNamespace(dit=model, motion_controller=None, vace=None, use_unified_sequence_parallel=False)
    return Wan21(pipe)


def _build(ttt_cfg=TTT_CFG):
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
        "attention_mask_mode": "mutual",  # OpenWAM-α's setting
        "video_attention_mask_mode": "first_frame_causal",
    }
    if ttt_cfg is not None:
        cfg["ttt"] = ttt_cfg
    torch.manual_seed(0)
    arch = _Arch(cfg=cfg)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 4, 3, 4, 4, generator=g)
    return {
        "latents": latents,
        "first_frame_latents": latents[:, :, 0:1].clone(),
        "timestep": torch.tensor([500.0]),
        "context": torch.randn(1, 3, TEXT_DIM, generator=g),
        "context_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def _joint(arch, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    actions = torch.randn(1, 5, ACTION_DIM, generator=g)
    return arch(actions, torch.tensor([0.5]), **_inputs(seed))


def _activate(arch):
    torch.manual_seed(1)
    for layer in arch.video_ttt.layers:
        nn.init.normal_(arch.video_ttt.branch(layer).o_proj.weight, std=0.5)


def test_zero_init_ttt_is_identity_on_real_wan_blocks():
    arch = _build()
    with_ttt = _joint(arch, seed=3)
    arch.mot_driver.ttt = None
    without = _joint(arch, seed=3)
    for a, b in zip(with_ttt, without):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_written_memory_changes_real_wan_prediction_until_reset():
    arch = _build()
    _activate(arch)
    before = _joint(arch, seed=3)
    with torch.no_grad(), arch.video_ttt.writing(lr=1.0):
        _joint(arch, seed=7)
    after = _joint(arch, seed=3)
    assert not torch.allclose(before[0], after[0])
    assert not torch.allclose(before[1], after[1])
    arch.video_ttt.reset()
    for a, b in zip(before, _joint(arch, seed=3)):
        torch.testing.assert_close(a, b)


def test_video_only_real_wan_run_block_applies_ttt():
    arch = _build()
    _activate(arch)
    with_ttt, _ = arch(None, None, **_inputs(3))
    arch.video_ttt = None
    without, _ = arch(None, None, **_inputs(3))
    assert not torch.allclose(with_ttt, without)


def test_ttt_only_meta_gradient_crosses_real_frozen_wan_blocks():
    arch = _build()
    _activate(arch)
    arch.restrict_training_to_ttt()
    arch.train()
    with arch.video_ttt.writing(lr=1.0, create_graph=True):
        _joint(arch, seed=7)
    video, action = _joint(arch, seed=3)
    (video.pow(2).mean() + action.pow(2).mean()).backward()

    first = arch.video_ttt.branch(0)
    for name in ("q_proj", "k_proj", "v_proj"):
        grad = getattr(first, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is None for n, p in arch.named_parameters() if not n.startswith("video_ttt."))
