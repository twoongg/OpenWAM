"""Can the frozen OpenWAM checkpoint produce the demo actions? (FlowDAgger action-inversion probe)

For each sampled demo window this runs the deploy sampler (same schedule,
seed and conditioning as the policy server), then inverts the joint sampler to
find the starting noise that makes the frozen model emit the demo's action
chunk (FlowDAgger, arXiv 2607.08877, §4.1-4.2 / App. A), runs the model from
that noise, and reports how close it gets. No weights are changed.

Reading the result (normalized action space; see openwam/flowdagger/inversion.py):
  recon_ratio ≪ 1 with action-noise std ≈ 1 and |w|>3 fraction ≈ 0.003
      → the demo behavior is inside what the frozen model can produce:
        noise steering (FlowDAgger) can in principle reach it.
  recon_ratio near 1, or inverted noise far from unit Gaussian
      → the demo behavior is outside the model's reach: weight updates (LoRA) needed.
Run once on the new demos (e.g. LIBERO-plus camera perturbation) and once on
the original LIBERO demos as the reference.

Usage:
    python scripts/flowdagger/inversion_probe.py \
        --ckpt-dir /path/OpenWAM-Alpha-Sim-LIBERO \
        --dataset-dir /path/libero_plus_lerobot \
        --num-samples 64 --out results/inversion_libero_plus.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

logger = logging.getLogger("inversion_probe")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt-dir", required=True, help="Checkpoint dir (config.yaml + checkpoint_step_*.safetensors).")
    p.add_argument("--ckpt-name", default=None, help="Checkpoint file (default: latest step).")
    p.add_argument("--dataset-dir", required=True, help="Demos in the checkpoint's reader format (LeRobot v3 LIBERO).")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=64, help="Demo windows to probe (full windows only).")
    p.add_argument("--seed", type=int, default=0, help="Window-sampling seed.")
    p.add_argument("--deploy-config", default=None, help="Deploy yaml (default: configs/deploy.yaml).")
    p.add_argument("--denoise-steps", type=int, default=None, help="Override inference.denoise_steps.")
    p.add_argument("--deploy-seed", type=int, default=42, help="Noise seed the policy server uses (engine default 42).")
    p.add_argument("--fp-iters", type=int, default=5, help="Fixed-point iterations per step (FlowDAgger M=5).")
    p.add_argument("--extra-seeds", type=int, default=4, help="Extra noise seeds tried as a resampling baseline.")
    p.add_argument("--self-check", type=int, default=4, help="Also invert the deployed sample for the first N windows.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label", default="", help="Name printed with the summary.")
    p.add_argument("--out", required=True, help="Output JSON (rewritten after every window).")
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import torch
    from omegaconf import OmegaConf

    from openwam.dataloader.registry import build_dataset
    from openwam.deploy.denoise_schedule import make_schedule
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import _load_deploy_yaml, merge_deploy_cfg
    from openwam.flowdagger.inversion import JointSampler, probe_action_target
    from openwam.flowdagger.probe_data import format_summary, pick_windows, probe_dataset_config, summarize

    train_cfg, arch = load_from_checkpoint_dir(args.ckpt_dir, device=args.device, ckpt_name=args.ckpt_name)
    cfg = merge_deploy_cfg(train_cfg, _load_deploy_yaml(args.deploy_config))
    inf = cfg.inference
    denoise_steps = int(args.denoise_steps or inf.denoise_steps)

    # Same schedule as JointInferenceEngine.generate (shifts come from the checkpoint's backbones).
    shift = getattr(arch.action_backbone, "shift_action", None)
    shift = 5.0 if shift is None else shift
    schedule = make_schedule(
        str(OmegaConf.select(inf, "denoise_mode", default="sync")),
        video_scheduler=arch.video_scheduler,
        action_scheduler=arch.action_scheduler,
        num_steps=denoise_steps,
        shift=shift,
        shift_video=getattr(arch.video_backbone, "shift_video", None),
        lead=str(OmegaConf.select(inf, "lead_modality", default="video")),
        alpha=float(OmegaConf.select(inf, "variance_shift_alpha", default=1.0)),
        offset=float(OmegaConf.select(inf, "linear_offset", default=0.0)),
    )

    dataset = build_dataset(
        probe_dataset_config(train_cfg, ckpt_dir=args.ckpt_dir, dataset_dir=args.dataset_dir, split=args.split),
        split=args.split,
    )
    indices = pick_windows(dataset, args.num_samples, seed=args.seed)
    logger.info("Probing %d full windows of %d in %s", len(indices), len(dataset), args.dataset_dir)

    run_info = {
        "ckpt_dir": args.ckpt_dir,
        "dataset_dir": args.dataset_dir,
        "label": args.label,
        "denoise_steps": denoise_steps,
        "schedule": [list(map(float, s)) for s in schedule],
        "fp_iters": args.fp_iters,
        "deploy_seed": args.deploy_seed,
        "extra_seeds": args.extra_seeds,
    }
    frame_kwargs = dict(
        num_frames=int(inf.video_num_frames),
        action_num_frames=int(inf.num_frames),
        height=int(inf.height),
        width=int(inf.width),
        shift=shift,
    )
    prompt_cache: dict = {}
    reports = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        t0 = time.time()
        sample = dataset[idx]
        target = sample["action"].unsqueeze(0)
        mask = sample["action_mask"].unsqueeze(0)
        if target.shape[1] != int(inf.num_frames) - 1:
            raise ValueError(f"demo chunk has {target.shape[1]} steps, model emits {int(inf.num_frames) - 1}")
        cond = dict(
            schedule=schedule,
            prompt=sample["prompt"],
            first_frame_image=sample["first_frame_image"],
            proprio=sample["proprio"],
            prompt_embed_cache=prompt_cache,
            **frame_kwargs,
        )
        with torch.no_grad():
            sampler, noise = JointSampler.prepare(arch, seed=args.deploy_seed, **cond)
            extra = [JointSampler.prepare(arch, seed=1000 + s, **cond)[1] for s in range(args.extra_seeds)]
            report = probe_action_target(
                sampler,
                noise,
                target.to(arch.device),
                mask.to(arch.device),
                iters=args.fp_iters,
                extra_noises=extra,
                self_check=i < args.self_check,
            )
        report.update(index=int(idx), prompt=sample["prompt"], seconds=round(time.time() - t0, 2))
        reports.append(report)
        logger.info(
            "[%d/%d] idx=%d base=%.4g recon=%.4g ratio=%.3g noise_std=%.3g (%.1fs)",
            i + 1,
            len(indices),
            idx,
            report["base_mse"],
            report["recon_mse"],
            report["recon_ratio"],
            report["noise_action"]["std"],
            report["seconds"],
        )
        payload = {"run": run_info, "summary": summarize(reports), "reports": reports}
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, out_path)

    print(format_summary(summarize(reports), args.label))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
