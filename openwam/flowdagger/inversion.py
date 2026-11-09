"""Run the deploy-time joint sampler forwards and backwards (FlowDAgger action inversion).

FlowDAgger (arXiv 2607.08877) adapts a frozen flow policy by learning which
starting noise to feed it. Its supervision comes from *action inversion*: map a
corrected action chunk back to the noise that would have produced it under the
frozen sampler, by undoing each Euler step with fixed-point iteration
(``z_k = z_{k+1} - Δ_k · v(z_k)``, Eq. 3; M=5 by default). For a world-action
model the whole joint latent is inverted, with the target taken from the base
policy's own clean sample and only the action replaced (their §4.2).

:class:`JointSampler` is ``BaseWAMArchitecture.generate``'s sync, no-CFG,
no-cache loop written as explicit steps, so the same steps can be run in
reverse. ``run`` reproduces ``generate`` bit-for-bit (pinned first frame,
inactive unified-action dims on their analytic noise path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor


@dataclass
class JointLatent:
    """One joint state of the sampler: video latents ``(1, C, F, H, W)`` and actions ``(1, T, D)``."""

    video: Tensor
    action: Tensor

    def to(self, dtype: torch.dtype) -> "JointLatent":
        return JointLatent(self.video.to(dtype), self.action.to(dtype))


@dataclass(frozen=True)
class SamplerStep:
    t_video: float
    t_action: float
    sigma_video: float
    sigma_video_next: float
    sigma_action: float
    sigma_action_next: float

    @property
    def video_stepping(self) -> bool:
        return self.sigma_video != self.sigma_video_next

    @property
    def action_stepping(self) -> bool:
        return self.sigma_action != self.sigma_action_next


class JointSampler:
    """The deploy joint Euler loop over a fixed conditioning, runnable in both directions.

    Args:
        arch: Loaded architecture (eval mode).
        inputs: Pipeline inputs as ``generate`` builds them (text context,
            first-frame latents, proprio, ...). ``latents`` is ignored: the
            video state is supplied per call.
        schedule: ``(t_video, t_action)`` pairs ending in ``(0, 0)``
            (``openwam.deploy.denoise_schedule.make_schedule``).
        active_action_mask: As in ``generate``; ``None`` infers it from the
            attached unify normalizer.
    """

    def __init__(self, arch, inputs: dict, schedule, *, active_action_mask: Optional[Tensor] = None):
        self.arch = arch
        self.inputs = {k: v for k, v in inputs.items() if k != "latents"}
        self.ref = inputs.get("first_frame_latents")
        self.inactive = arch._resolve_inactive_action_dims(active_action_mask, arch.device)
        n_v = float(arch.video_scheduler.num_train_timesteps)
        n_a = float(arch.action_scheduler.num_train_timesteps)
        steps = []
        for (t_v, t_a), (t_vn, t_an) in zip(schedule[:-1], schedule[1:]):
            step = SamplerStep(t_v, t_a, t_v / n_v, t_vn / n_v, t_a / n_a, t_an / n_a)
            if step.video_stepping or step.action_stepping:
                steps.append(step)
        self.steps: list[SamplerStep] = steps

    @classmethod
    def prepare(
        cls,
        arch,
        *,
        schedule,
        prompt: str,
        first_frame_image,
        proprio: Optional[Tensor],
        num_frames: int,
        action_num_frames: int,
        height: int,
        width: int,
        seed: int,
        shift: float = 5.0,
        tiled: bool = True,
        prompt_embed_cache: Optional[dict] = None,
        active_action_mask: Optional[Tensor] = None,
    ) -> tuple["JointSampler", JointLatent]:
        """Build the conditioning and the seeded starting noise exactly as ``generate`` does."""
        arch.eval()
        device, dtype = arch.device, arch.dtype
        inputs = arch.video_backbone.preprocess_input_for_inference(
            prompt=prompt,
            vace_video=None,
            first_frame_image=first_frame_image,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=len(schedule) - 1,
            shift=shift,
            tiled=tiled,
            vace_cache=None,
            prompt_embed_cache=prompt_embed_cache,
            cfg_scale=1.0,
            cfg_merge=False,
        )
        video = inputs["latents"].clone()
        ref = inputs.get("first_frame_latents")
        if ref is not None:
            video[:, :, : ref.shape[2]] = ref
        if arch.uses_proprioception:
            if proprio is None:
                raise ValueError("use_proprioception=True requires `proprio`.")
            inputs["proprio"] = proprio.to(device=device, dtype=dtype)
        action = torch.randn(
            1,
            action_num_frames - 1,
            arch.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        return cls(arch, inputs, schedule, active_action_mask=active_action_mask), JointLatent(video, action)

    # --- one step -----------------------------------------------------------

    def _pin(self, video: Tensor) -> Tensor:
        if self.ref is None:
            return video
        video = video.clone()
        video[:, :, : self.ref.shape[2]] = self.ref.to(video.dtype)
        return video

    @torch.no_grad()
    def velocity(self, z: JointLatent, step: SamplerStep) -> JointLatent:
        """Model velocities at ``z`` (in ``z``'s dtype; the model runs in its own dtype)."""
        arch = self.arch
        dtype, device = arch.dtype, arch.device
        v_ts = torch.tensor([step.t_video], dtype=dtype, device=device)
        a_ts = torch.tensor([step.t_action], dtype=dtype, device=device)
        v_video, v_action = arch.forward(
            z.action.to(dtype), a_ts, **{**self.inputs, "latents": z.video.to(dtype)}, timestep=v_ts
        )
        return JointLatent(v_video.to(z.video.dtype), v_action.to(z.action.dtype))

    def _inactive_noise(self, reference: JointLatent) -> Optional[Tensor]:
        """Unit noise of the inactive dims, as ``generate`` captures it at the first action step."""
        if self.inactive is None:
            return None
        first = next((s for s in self.steps if s.action_stepping), None)
        if first is None:
            return None
        return reference.action[..., self.inactive] / float(first.sigma_action)

    # --- forward --------------------------------------------------------------

    @torch.no_grad()
    def run(self, noise: JointLatent, *, state_dtype: Optional[torch.dtype] = None) -> JointLatent:
        """Integrate from ``noise`` to a clean sample, step for step like ``generate``.

        ``state_dtype`` defaults to the model dtype, which is what deploy does.
        """
        dtype = state_dtype or self.arch.dtype
        video = self._pin(noise.video.to(dtype))
        action = noise.action.to(dtype).clone()
        inactive_noise = None
        for step in self.steps:
            v = self.velocity(JointLatent(video, action), step)
            if step.video_stepping:
                video = self._pin(video + v.video * (step.sigma_video_next - step.sigma_video))
            if step.action_stepping:
                if self.inactive is not None and inactive_noise is None:
                    inactive_noise = action[..., self.inactive].detach().clone() / float(step.sigma_action)
                action = self.arch.action_scheduler.flow_step(
                    v.action, step.sigma_action, step.sigma_action_next, action
                )
                if self.inactive is not None:
                    action[..., self.inactive] = inactive_noise * float(step.sigma_action_next)
        return JointLatent(video, action)

    # --- backward -------------------------------------------------------------

    @torch.no_grad()
    def invert(self, target: JointLatent, *, reference: JointLatent, iters: int = 5) -> tuple[JointLatent, dict]:
        """Find the starting noise whose ``run`` ends at ``target`` (per-step fixed point).

        Args:
            target: Desired clean joint sample. Its pinned frames and inactive
                action dims are overridden by their deploy values.
            reference: A starting noise whose inactive action dims are kept
                (those dims never change the active outputs' sampler path
                other than through their fixed analytic schedule).
            iters: Fixed-point iterations per step (FlowDAgger's M).

        Returns:
            ``(noise, info)``; ``info`` holds, per step in sampler order, the RMS
            of the last fixed-point update for each stream — a convergence
            diagnostic (non-shrinking updates mean the step is not contracting).
        """
        f32 = torch.float32
        inactive_noise = self._inactive_noise(reference.to(f32))
        z = JointLatent(self._pin(target.video.to(f32)), target.action.to(f32).clone())
        video_update = [0.0] * len(self.steps)
        action_update = [0.0] * len(self.steps)

        def set_inactive(action: Tensor, sigma: float) -> Tensor:
            if inactive_noise is not None:
                action[..., self.inactive] = inactive_noise * sigma
            return action

        z.action = set_inactive(z.action, float(self.steps[-1].sigma_action_next)) if self.steps else z.action
        for idx in range(len(self.steps) - 1, -1, -1):
            step = self.steps[idx]
            nxt = z
            cur = JointLatent(nxt.video.clone(), set_inactive(nxt.action.clone(), float(step.sigma_action)))
            for _ in range(iters):
                v = self.velocity(cur, step)
                video, action = cur.video, cur.action
                if step.video_stepping:
                    video = self._pin(nxt.video - (step.sigma_video_next - step.sigma_video) * v.video)
                if step.action_stepping:
                    action = set_inactive(
                        nxt.action - (step.sigma_action_next - step.sigma_action) * v.action, float(step.sigma_action)
                    )
                video_update[idx] = _rms(video - cur.video)
                action_update[idx] = _rms(action - cur.action)
                cur = JointLatent(video, action)
            z = cur
        if inactive_noise is not None:
            z.action[..., self.inactive] = reference.action[..., self.inactive].to(f32)
        return z, {"video_update": video_update, "action_update": action_update}


def _rms(x: Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt()) if x.numel() else 0.0


def _noise_stats(x: Tensor) -> dict:
    x = x.float().flatten()
    return {
        "std": float(x.std()) if x.numel() > 1 else 0.0,
        "mean": float(x.mean()),
        "absmax": float(x.abs().max()),
        "frac_abs_gt3": float((x.abs() > 3).float().mean()),
    }


def _masked_mse(a: Tensor, b: Tensor, mask: Tensor) -> float:
    m = mask.to(a.device)
    return float(((a.float() - b.float()).pow(2) * m).sum() / m.sum().clamp_min(1))


def probe_action_target(
    sampler: JointSampler,
    noise: JointLatent,
    target_action: Tensor,
    mask: Tensor,
    *,
    iters: int = 5,
    extra_noises: Sequence[JointLatent] = (),
    self_check: bool = False,
) -> dict:
    """How reachable is ``target_action`` for the frozen sampler at this conditioning?

    The target follows FlowDAgger §4.2: the deployed sample from ``noise``
    with its action replaced by ``target_action`` where ``mask`` is set
    (``mask``: bool, same shape as the action). All errors are MSE in the
    model's normalized action space over the masked entries.

    Returns a JSON-able dict:
        base_mse / recon_mse / recon_ratio — deployed vs. demo, reconstruction
            from the inverted noise vs. demo, and their ratio (≪1 = reachable);
        base_per_dim / recon_per_dim — the same per active dim;
        seed_mse — deployed error under each extra noise (does plain resampling get there?);
        noise_action / noise_video — statistics of the inverted noise (a unit
            Gaussian has std≈1, frac_abs_gt3≈0.0027);
        noise_action_shift — RMS change of the active action noise vs. ``noise``;
        video_drift — MSE between reconstructed and deployed video latents,
            relative to the deployed latents' variance;
        fp_action_update / fp_video_update — per-step last fixed-point update;
        self_recon_mse / self_noise_err (``self_check``) — inverting the
            deployed sample itself: the numerical floor of the procedure.
    """
    f32 = torch.float32
    mask = mask.to(device=target_action.device, dtype=torch.bool)
    base = sampler.run(noise)
    target = JointLatent(
        base.video.to(f32), torch.where(mask, target_action.to(base.action.device, f32), base.action.to(f32))
    )
    w, info = sampler.invert(target, reference=noise, iters=iters)
    recon = sampler.run(w)

    active_dims = [int(d) for d in torch.nonzero(mask.reshape(-1, mask.shape[-1]).any(0)).flatten()]
    flat_mask = mask.reshape(-1, mask.shape[-1])

    def per_dim(pred: Tensor) -> list[float]:
        diff = (pred.float() - target.action).reshape(-1, mask.shape[-1]).pow(2)
        return [float(diff[flat_mask[:, d], d].mean()) for d in active_dims]

    base_mse = _masked_mse(base.action, target.action, mask)
    recon_mse = _masked_mse(recon.action, target.action, mask)
    n_ref = sampler.ref.shape[2] if sampler.ref is not None else 0
    action_sel = w.action[..., active_dims] if active_dims else w.action
    report = {
        "base_mse": base_mse,
        "recon_mse": recon_mse,
        "recon_ratio": recon_mse / base_mse if base_mse > 0 else float("nan"),
        "active_dims": active_dims,
        "base_per_dim": per_dim(base.action),
        "recon_per_dim": per_dim(recon.action),
        "seed_mse": [_masked_mse(sampler.run(n).action, target.action, mask) for n in extra_noises],
        "noise_action": _noise_stats(action_sel),
        "noise_video": _noise_stats(w.video[:, :, n_ref:]),
        "noise_action_shift": _rms(action_sel - noise.action.to(f32)[..., active_dims]),
        "video_drift": float((recon.video.float() - base.video.float()).pow(2).mean() / base.video.float().var()),
        "fp_action_update": info["action_update"],
        "fp_video_update": info["video_update"],
    }
    if self_check:
        w_self, _ = sampler.invert(base.to(f32), reference=noise, iters=iters)
        report["self_recon_mse"] = _masked_mse(sampler.run(w_self).action, base.action, mask)
        report["self_noise_err"] = _rms(w_self.action[..., active_dims] - noise.action.to(f32)[..., active_dims])
    return report


__all__ = ["JointLatent", "JointSampler", "SamplerStep", "probe_action_target"]
