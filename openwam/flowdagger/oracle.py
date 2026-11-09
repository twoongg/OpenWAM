"""Oracle noise server: how far can noise alone take the frozen model on a perturbed camera?

FlowDAgger (arXiv 2607.08877) never changes the base policy; a small network
learns which starting noise to feed it. Its ceiling is therefore what a
*perfect* noise choice achieves. This server measures that ceiling in closed
loop, with no training:

  teacher  the same frozen model, shown the clean (training) camera view of
           the current state. It solves the stock task ~97-100% of the time.
  student  the same frozen model, shown the perturbed camera view.

At every replan the teacher's action chunk is computed, the student's sampler
is inverted to the starting noise that makes the student emit that chunk
(joint video+action fixed-point inversion, FlowDAgger §4.1-4.2; see
``openwam.flowdagger.inversion``), and the student is run from that noise. The
action sent to the robot is the student's own output, so whatever the student
cannot reproduce stays in the rollout.

Modes (one per server process; run all three on the same start states):
  oracle   execute the student steered by inverted noise  -> FlowDAgger's ceiling
  teacher  execute the teacher's action                   -> pipeline sanity check
  student  execute the student from the seeded noise      -> same-pipeline baseline

Protocol: the stock obs message plus ``teacher_images`` (same keys as
``images``; cameras left out or null fall back to the obs's own ``images``, so a
driver whose perturbation only moves the third-person camera sends just
``{"head_camera": <clean view>}``). Each replan appends one JSON line to the log.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from openwam.deploy.denoise_schedule import make_schedule
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.obs_preprocess import ObsValidationError
from openwam.deploy.server import PolicyServer
from openwam.flowdagger.inversion import JointLatent, JointSampler, _masked_mse, _noise_stats

logger = logging.getLogger(__name__)

MODES = ("oracle", "teacher", "student")
TEACHER_IMAGES = "teacher_images"


class OracleInversionEngine(JointInferenceEngine):
    """Generates each chunk from the student (perturbed view), the teacher (clean view), or the inverted student.

    Runs the sync, no-CFG, no-velocity-cache joint sampler
    (:class:`~openwam.flowdagger.inversion.JointSampler`), which is
    ``BaseWAMArchitecture.generate`` step for step, so inversion and execution
    use exactly the same map from noise to actions.

    Args:
        cfg: Merged deploy config (``cfg.inference``).
        architecture: Loaded frozen checkpoint.
        mode: ``"oracle"``, ``"teacher"`` or ``"student"``.
        fp_iters: Fixed-point iterations per sampler step (FlowDAgger M).
        seed: Starting-noise seed (the stock server's is 42).
        log_path: JSONL file; one record per generated chunk.
    """

    def __init__(
        self,
        cfg,
        architecture,
        *,
        mode: str = "oracle",
        fp_iters: int = 5,
        seed: int = 42,
        log_path: Optional[str] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
        super().__init__(cfg, architecture=architecture)
        if str(OmegaConf.select(cfg, "inference.denoise_mode", default="sync")) != "sync":
            raise ValueError("the oracle server inverts the sync joint sampler; set inference.denoise_mode=sync")
        if self._cfg_scale != 1.0:
            raise ValueError("the oracle server does not model CFG; set inference.cfg_scale=1.0")
        self.mode = mode
        self.fp_iters = int(fp_iters)
        self.seed = int(seed)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode = 0
        self.chunk = 0

    def new_episode(self) -> None:
        """Called on client reset: later log records belong to the next episode."""
        self.episode += 1
        self.chunk = 0

    # --- one chunk ------------------------------------------------------------

    def _sampler_kwargs(self, conditions: dict) -> dict:
        """Schedule and conditioning arguments exactly as ``JointInferenceEngine.generate`` resolves them."""
        arch = self.architecture
        inf = self.cfg.inference
        steps = conditions.get("denoise_steps", inf.denoise_steps)
        shift = conditions.get("shift", getattr(arch.action_backbone, "shift_action", None))
        shift = 5.0 if shift is None else shift
        schedule = make_schedule(
            "sync",
            video_scheduler=arch.video_scheduler,
            action_scheduler=arch.action_scheduler,
            num_steps=steps,
            shift=shift,
            shift_video=conditions.get("shift_video", getattr(arch.video_backbone, "shift_video", None)),
        )
        proprio = conditions.get("proprio")
        if proprio is None:
            observation = conditions.get("observation") or {}
            proprio = observation.get("state") if isinstance(observation, dict) else None
        action_num_frames = int(conditions.get("num_frames", getattr(inf, "num_frames", 49)))
        return dict(
            schedule=schedule,
            prompt=conditions.get("prompt", ""),
            proprio=arch.normalize_deploy_proprio(proprio),
            num_frames=int(conditions.get("video_num_frames", getattr(inf, "video_num_frames", action_num_frames))),
            action_num_frames=action_num_frames,
            height=conditions.get("height", getattr(inf, "height", 384)),
            width=conditions.get("width", getattr(inf, "width", 320)),
            seed=conditions.get("seed", self.seed),
            shift=shift,
            tiled=conditions.get("tiled", True),
            prompt_embed_cache=self._prompt_embed_cache,
        )

    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        t0 = time.time()
        arch = self.architecture
        kwargs = self._sampler_kwargs(conditions)
        record: dict = {"episode": self.episode, "chunk": self.chunk, "mode": self.mode}

        taught = None
        if self.mode != "student":
            observation = conditions.get("observation") or {}
            teacher_image = observation.get("teacher_image") if isinstance(observation, dict) else None
            if teacher_image is None:
                raise ValueError(f"mode {self.mode!r} needs the clean view: send obs['{TEACHER_IMAGES}']")
            teacher, teacher_noise = JointSampler.prepare(arch, first_frame_image=[teacher_image], **kwargs)
            taught = teacher.run(teacher_noise)

        if self.mode == "teacher":
            out = taught
        else:
            student, noise = JointSampler.prepare(arch, first_frame_image=conditions.get("first_frame_image"), **kwargs)
            out = student.run(noise) if self.mode == "student" else self._steer(student, noise, taught, record)

        actions = out.action.squeeze(0).float().cpu().numpy()
        normalizer = getattr(arch, "normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)

        record["seconds"] = round(time.time() - t0, 3)
        self._log(record)
        self.chunk += 1
        return {"video": None, "actions": actions}

    def _steer(self, student: JointSampler, noise: JointLatent, taught: JointLatent, record: dict) -> JointLatent:
        """Run the student from the noise that makes it emit the teacher's actions (FlowDAgger §4.2 target)."""
        f32 = torch.float32
        base = student.run(noise)
        target = JointLatent(base.video.to(f32), taught.action.to(f32))
        w, info = student.invert(target, reference=noise, iters=self.fp_iters)
        out = student.run(w)

        action_dim = target.action.shape[-1]
        active = torch.ones(action_dim, dtype=torch.bool, device=target.action.device)
        if student.inactive is not None:
            active &= ~student.inactive.to(active.device)
        mask = active.expand_as(target.action)
        dims = [int(d) for d in torch.nonzero(active).flatten()]

        def per_dim(pred: JointLatent) -> list[float]:
            diff = (pred.action.float() - target.action).pow(2)
            return [float(diff[..., d].mean()) for d in dims]

        record.update(
            active_dims=dims,
            base_mse=_masked_mse(base.action, target.action, mask),
            recon_mse=_masked_mse(out.action, target.action, mask),
            base_per_dim=per_dim(base),
            recon_per_dim=per_dim(out),
            noise_action=_noise_stats(w.action[..., active]),
            fp_action_update_last=info["action_update"][-1] if info["action_update"] else 0.0,
            video_drift=float((out.video.float() - base.video.float()).pow(2).mean() / base.video.float().var()),
        )
        return out

    def _log(self, record: dict) -> None:
        if self.mode == "oracle":
            logger.info(
                "[oracle] ep=%d chunk=%d student->teacher mse %.4g -> %.4g, noise std %.3g |w|>3 %.3g (%.1fs)",
                record["episode"],
                record["chunk"],
                record["base_mse"],
                record["recon_mse"],
                record["noise_action"]["std"],
                record["noise_action"]["frac_abs_gt3"],
                record["seconds"],
            )
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")


class OracleInversionServer(PolicyServer):
    """Stock policy server that also takes the clean camera view (``obs['teacher_images']``)."""

    def predict(self, obs: dict) -> dict:
        self._init_policy()
        clean = obs.get(TEACHER_IMAGES)
        if clean is None and self.engine.mode != "student":
            raise ObsValidationError(
                f"oracle mode {self.engine.mode!r} needs obs['{TEACHER_IMAGES}'] (the clean camera view, "
                "same keys as 'images')"
            )
        if clean is not None:
            if not isinstance(clean, dict):
                raise ObsValidationError(f"obs['{TEACHER_IMAGES}'] must be a dict of camera name -> image")
            own = obs.get("images") if isinstance(obs.get("images"), dict) else {}
            cameras = {**own, **{k: v for k, v in clean.items() if v is not None}}
            teacher = self._obs_preprocessor.preprocess(
                {"images": cameras, "prompt": obs.get("prompt"), "state": obs.get("state")}
            )
            # The engine reads it from conditions["observation"] (WAMPolicy passes the obs through).
            obs = {**obs, "teacher_image": teacher["image"]}
        return super().predict(obs)

    def reset(self):
        super().reset()
        self.engine.new_episode()

    def _pong_payload(self) -> dict:
        return {**super()._pong_payload(), "oracle_mode": self.engine.mode}


def build_oracle_server(
    cfg,
    ckpt_dir: str,
    *,
    device: str = "cuda",
    ckpt_name: Optional[str] = None,
    mode: str = "oracle",
    fp_iters: int = 5,
    seed: int = 42,
    log_path: Optional[str] = None,
) -> OracleInversionServer:
    """Load the frozen checkpoint and wrap it in the oracle server.

    Always eager and without the DiT velocity cache: the inversion needs the
    exact uncached sampler, and every mode must run the same pipeline so their
    success rates compare.
    """
    from openwam.deploy import model_loader
    from openwam.deploy.executors import resolve_execution_config
    from openwam.deploy.server import _normalize_compile_enabled_in_cfg, merge_deploy_cfg

    cfg = copy.deepcopy(cfg) if cfg is not None else OmegaConf.create({})
    if resolve_execution_config(cfg).enabled:
        raise ValueError("the oracle server needs inference_mode=sync (one clean view per replan)")
    OmegaConf.update(cfg, "optimization.compile.enabled", False, merge=False)
    OmegaConf.update(cfg, "optimization.dit_cache.enabled", False, merge=False)
    _normalize_compile_enabled_in_cfg(cfg)
    training_cfg, architecture = model_loader.load_from_checkpoint_dir(ckpt_dir, device=device, ckpt_name=ckpt_name)
    merged = merge_deploy_cfg(training_cfg, cfg)
    engine = OracleInversionEngine(merged, architecture, mode=mode, fp_iters=fp_iters, seed=seed, log_path=log_path)
    logger.info("Oracle server: mode=%s fp_iters=%d seed=%d (eager, no velocity cache)", mode, fp_iters, seed)
    return OracleInversionServer(engine=engine, cfg=merged)


__all__ = ["MODES", "OracleInversionEngine", "OracleInversionServer", "build_oracle_server"]
