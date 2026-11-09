"""FlowDAgger-style action inversion through the real joint sampler (tiny random Wan + ActionDiT, CPU).

The sampler must reproduce ``BaseWAMArchitecture.generate`` exactly, and the
inversion must find the starting noise that makes that same loop emit a given
action chunk while keeping the pinned first frame and the inactive unified
action dims on their deploy paths.
"""

from __future__ import annotations

import pytest
import torch

from openwam.flowdagger.inversion import JointLatent, JointSampler, probe_action_target

ACTION_DIM = 7
T_ACTION = 5
ACTIVE = torch.tensor([True, True, True, True, False, False, False])


def _arch():
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from tests.test_video_ttt_real_wan import _build

    arch = _build(ttt_cfg=None)
    arch.video_backbone._scheduler = ActionScheduler()
    # Random init leaves output heads near zero; give every weight a small kick so the
    # velocity field actually depends on its inputs (otherwise inversion is trivial).
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in arch.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    arch.eval()
    return arch


def _fake_preprocess(video_backbone):
    """Stand-in for Wan's text/VAE preprocessing: seeded video noise + fixed conditioning."""

    def preprocess_input_for_inference(*, seed, **_kw):
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, 4, 3, 4, 4, generator=g)
        cond = torch.Generator().manual_seed(123)
        return {
            "noise": noise,
            "latents": noise,
            "first_frame_latents": torch.randn(1, 4, 1, 4, 4, generator=cond),
            "context": torch.randn(1, 3, 32, generator=cond),
            "context_mask": torch.ones(1, 3, dtype=torch.bool),
        }

    video_backbone.preprocess_input_for_inference = preprocess_input_for_inference


def _schedule(arch, steps=4):
    from openwam.deploy.denoise_schedule import make_schedule

    return make_schedule(
        "sync", video_scheduler=arch.video_scheduler, action_scheduler=arch.action_scheduler, num_steps=steps, shift=5.0
    )


def _sampler(arch, *, seed=42, active=ACTIVE, steps=4):
    _fake_preprocess(arch.video_backbone)
    return JointSampler.prepare(
        arch,
        schedule=_schedule(arch, steps),
        prompt="",
        first_frame_image=None,
        proprio=None,
        num_frames=9,
        action_num_frames=T_ACTION + 1,
        height=8,
        width=8,
        seed=seed,
        active_action_mask=active,
    )


@pytest.mark.parametrize("active", [None, ACTIVE], ids=["all_active", "unified_inactive_dims"])
def test_run_reproduces_generate_actions_exactly(active):
    arch = _arch()
    sampler, noise = _sampler(arch, active=active)
    ours = sampler.run(noise).action.squeeze(0)

    _fake_preprocess(arch.video_backbone)
    theirs = arch.generate(
        schedule=_schedule(arch),
        prompt="",
        num_frames=9,
        action_num_frames=T_ACTION + 1,
        height=8,
        width=8,
        seed=42,
        decode_video=False,
        active_action_mask=active,
    )["actions"]
    torch.testing.assert_close(ours, torch.from_numpy(theirs), rtol=0, atol=0)


def test_prepared_noise_pins_the_conditioning_frame():
    arch = _arch()
    sampler, noise = _sampler(arch)
    ref = sampler.inputs["first_frame_latents"]
    torch.testing.assert_close(noise.video[:, :, :1], ref)


def test_inverting_a_sample_of_the_sampler_recovers_its_noise():
    arch = _arch()
    sampler, noise = _sampler(arch)
    clean = sampler.run(noise, state_dtype=torch.float32)

    recovered, info = sampler.invert(clean, reference=noise, iters=60)

    free = slice(1, None)
    torch.testing.assert_close(recovered.action, noise.action, atol=1e-4, rtol=0)
    torch.testing.assert_close(recovered.video[:, :, free], noise.video[:, :, free], atol=1e-4, rtol=0)
    assert len(info["action_update"]) == len(sampler.steps)
    assert max(info["action_update"]) < 1e-4


def test_inversion_reaches_a_nearby_action_target_and_keeps_deploy_paths():
    arch = _arch()
    sampler, noise = _sampler(arch)
    base = sampler.run(noise, state_dtype=torch.float32)
    g = torch.Generator().manual_seed(9)
    shifted = base.action + 0.1 * torch.randn(base.action.shape, generator=g) * ACTIVE
    target = JointLatent(video=base.video, action=shifted)

    w, _ = sampler.invert(target, reference=noise, iters=60)
    out = sampler.run(w, state_dtype=torch.float32)

    base_err = (base.action - shifted).pow(2)[..., ACTIVE].mean()
    out_err = (out.action - shifted).pow(2)[..., ACTIVE].mean()
    assert out_err < 1e-3 * base_err
    # The conditioning frame stays pinned and inactive dims ride the analytic noise path.
    torch.testing.assert_close(w.video[:, :, :1], noise.video[:, :, :1], rtol=0, atol=0)
    torch.testing.assert_close(w.action[..., ~ACTIVE], noise.action[..., ~ACTIVE], rtol=0, atol=0)
    assert torch.count_nonzero(out.action[..., ~ACTIVE]) == 0


def test_probe_reports_reachability_against_the_deployed_sample():
    arch = _arch()
    sampler, noise = _sampler(arch)
    base = sampler.run(noise)
    g = torch.Generator().manual_seed(9)
    target = base.action + 0.1 * torch.randn(base.action.shape, generator=g)
    mask = ACTIVE.expand(1, T_ACTION, ACTION_DIM).clone()
    mask[:, -1] = False  # e.g. a padded tail step: never supervised

    extra = [_sampler(arch, seed=s)[1] for s in (1, 2)]
    report = probe_action_target(sampler, noise, target, mask, iters=60, extra_noises=extra, self_check=True)

    assert report["base_mse"] > 0
    assert report["recon_ratio"] < 1e-2
    assert report["self_recon_mse"] < 1e-8
    assert len(report["seed_mse"]) == 2
    assert report["active_dims"] == [0, 1, 2, 3]
    assert len(report["recon_per_dim"]) == 4
    for key in ("noise_action", "noise_video"):
        assert set(report[key]) == {"std", "mean", "absmax", "frac_abs_gt3"}
    assert len(report["fp_action_update"]) == len(sampler.steps)
    assert report["video_drift"] >= 0


def test_probe_ignores_unsupervised_entries_of_the_target():
    arch = _arch()
    sampler, noise = _sampler(arch)
    base = sampler.run(noise)
    target = base.action.clone()
    mask = ACTIVE.expand(1, T_ACTION, ACTION_DIM).clone()
    mask[:, -1] = False
    target[:, -1] = 100.0  # padded tail of a short window: garbage, must not count or be chased

    report = probe_action_target(sampler, noise, target, mask, iters=60)

    assert report["base_mse"] == 0.0
    assert report["recon_mse"] < 1e-8
