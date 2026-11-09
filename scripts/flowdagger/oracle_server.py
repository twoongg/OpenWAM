"""Oracle noise server: FlowDAgger's closed-loop ceiling on a perturbed camera, with no training.

The frozen checkpoint is shown both the perturbed view (student) and the clean
view (teacher) of each state. In ``oracle`` mode the student's starting noise
is inverted so it emits the teacher's action chunk, and the student's output
is executed (see openwam/flowdagger/oracle.py). Run the same 30 start states
in all three modes:

  --mode teacher   clean view drives the robot   -> pipeline check, expect ~stock success
  --mode student   perturbed view, seeded noise  -> baseline on this same (eager, uncached) pipeline
  --mode oracle    perturbed view, inverted noise -> best any noise policy (FlowDAgger) can do

Same flags as the stock server (``openwam.deploy.server``); compile and the DiT
velocity cache are always off. The driver adds ``teacher_images`` to each obs,
e.g. ``{"head_camera": <clean third-person view>}`` (unlisted cameras reuse the
obs's own ``images``).

Usage:
    python scripts/flowdagger/oracle_server.py --ckpt-dir /path/OpenWAM-Alpha-Sim-LIBERO \
        --mode oracle --port 8877 --log results/oracle/oracle.jsonl
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


def main(argv: Optional[list[str]] = None) -> None:
    from omegaconf import OmegaConf

    from openwam.deploy.server import _build_argparser, resolve_deploy_args
    from openwam.flowdagger import oracle

    parser = _build_argparser()
    parser.description = __doc__.split("\n\n")[0]
    parser.add_argument("--mode", choices=oracle.MODES, required=True, help="What drives the robot (see above).")
    parser.add_argument("--fp-iters", type=int, default=5, help="Fixed-point iterations per step (FlowDAgger M=5).")
    parser.add_argument("--seed", type=int, default=42, help="Starting-noise seed (stock server: 42).")
    parser.add_argument("--log", default=None, help="JSONL file, one record per generated chunk.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        cfg, ckpt_dir = resolve_deploy_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    device = args.device or str(OmegaConf.select(cfg, "device", default="cuda"))
    server_cfg = getattr(cfg, "server", None)
    host = args.host or getattr(server_cfg, "host", "0.0.0.0")
    port = args.port or getattr(server_cfg, "port", 8848)
    server = oracle.build_oracle_server(
        cfg,
        ckpt_dir,
        device=device,
        ckpt_name=args.ckpt_name,
        mode=args.mode,
        fp_iters=args.fp_iters,
        seed=args.seed,
        log_path=args.log,
    )
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
