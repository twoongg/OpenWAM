"""Meta-training step for the video-expert TTT memory.

One step on a batch of :class:`~openwam.dataloader.ttt_pairs.RoboDojoTTTPairDataset`
items::

    memory.reset()
    for each context position k:  write batch[:].ttt_context[k] into memory   (create_graph)
    loss = usual compute_loss on the targets (reads the written memory)
    loss.backward()   →  reaches θ_{Q,K,V,O}, W_init through the writes

By default the context passes run the frozen backbone under ``no_grad`` so no
backbone activations are kept for them; the TTT writes still build their graph
(``VideoTTTMemory.update`` runs under ``enable_grad``). Cross-layer effects of
the context stream are then not differentiated — set
``context_backbone_grad=True`` for the exact (much more memory-hungry) version.

``measure_gain`` also evaluates the target with an empty memory under the same
noise / timestep draw: ``context_gain = loss_no_context - loss``. A memory that
learned to *use* what was written shows a positive gain; a branch that only
became a static adapter shows ~0.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch

from openwam.model.ttt import VideoTTTMemory


@dataclass
class TTTMetaConfig:
    inner_lr: float = 0.1
    gain_every: int = 0  # 0 = never measure the context gain
    context_backbone_grad: bool = False
    step: int = 0  # incremented by compute_ttt_meta_losses


def _rng_state():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda


def _restore_rng(state) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def ttt_meta_step(
    memory: VideoTTTMemory,
    loss_fn: Callable[[Any], dict],
    contexts: Sequence[Any],
    target: Any,
    *,
    inner_lr: float,
    context_backbone_grad: bool = False,
    measure_gain: bool = False,
) -> dict:
    """Write ``contexts`` in order, then return ``loss_fn(target)`` (+ optional context gain)."""
    memory.reset()
    backbone_grad = contextlib.nullcontext() if context_backbone_grad else torch.no_grad()
    with backbone_grad, memory.writing(lr=inner_lr, create_graph=True):
        for context in contexts:
            loss_fn(context)

    rng = _rng_state() if measure_gain else None
    out = loss_fn(target)
    result = {
        "loss": out["loss"],
        "loss_video": out.get("loss_video", torch.tensor(0.0)),
        "loss_action": out.get("loss_action", torch.tensor(0.0)),
    }
    if measure_gain:
        memory.reset()  # the target graph already holds the written weights it needs
        _restore_rng(rng)
        with torch.no_grad():
            no_context = loss_fn(target)["loss"].detach()
        result["loss_no_context"] = no_context
        result["context_gain"] = no_context - out["loss"].detach()
    memory.reset()
    return result


def compute_ttt_meta_losses(
    architecture,
    memory: VideoTTTMemory,
    batch: list[dict],
    *,
    lambda_video: float,
    lambda_action: float,
    meta: TTTMetaConfig,
) -> dict:
    """Trainer-facing wrapper: split a pair batch, run :func:`ttt_meta_step`, return trainer loss keys."""
    num_context = len(batch[0]["ttt_context"])
    if num_context == 0 or any(len(item["ttt_context"]) != num_context for item in batch):
        raise ValueError("every item in a TTT pair batch needs the same, non-zero number of context windows")

    targets = [{k: v for k, v in item.items() if k != "ttt_context"} for item in batch]
    contexts = [[item["ttt_context"][k] for item in batch] for k in range(num_context)]

    def loss_fn(samples):
        inputs = architecture.prepare_inputs(samples)
        return architecture.compute_loss(**inputs, lambda_video=lambda_video, lambda_action=lambda_action)

    measure = meta.gain_every > 0 and meta.step % meta.gain_every == 0
    meta.step += 1
    out = ttt_meta_step(
        memory,
        loss_fn,
        contexts,
        targets,
        inner_lr=meta.inner_lr,
        context_backbone_grad=meta.context_backbone_grad,
        measure_gain=measure,
    )
    losses = {"total": out["loss"], "video": out["loss_video"], "action": out["loss_action"]}
    if measure:
        losses["ttt_context_gain"] = out["context_gain"]
        losses["ttt_loss_no_context"] = out["loss_no_context"]
    return losses


def resolve_ttt_meta_config(cfg) -> Optional[TTTMetaConfig]:
    """``training.ttt_meta`` → :class:`TTTMetaConfig` (None when absent or disabled)."""
    from omegaconf import OmegaConf

    meta_cfg = OmegaConf.select(cfg, "training.ttt_meta", default=None)
    if meta_cfg is None or not bool(meta_cfg.get("enabled", False)):
        return None
    if not bool(OmegaConf.select(cfg, "model.architecture.ttt.enabled", default=False)):
        raise ValueError("training.ttt_meta.enabled requires model.architecture.ttt.enabled=true")
    return TTTMetaConfig(
        inner_lr=float(meta_cfg.get("inner_lr", 0.1)),
        gain_every=int(meta_cfg.get("gain_every", 0)),
        context_backbone_grad=bool(meta_cfg.get("context_backbone_grad", False)),
    )


TTT_METRIC_KEYS = ("ttt_context_gain", "ttt_loss_no_context")


def reduce_ttt_metrics(accelerator, losses: dict) -> dict:
    """Mean of the TTT diagnostics across ranks (present on all ranks at the same step)."""
    present = [k for k in TTT_METRIC_KEYS if k in losses]
    if not present:
        return {}
    values = torch.stack([losses[k].detach().float().reshape(()) for k in present])
    if accelerator is not None and accelerator.num_processes > 1:
        values = accelerator.gather(values.to(accelerator.device).reshape(1, -1)).mean(dim=0)
    return {k: float(v) for k, v in zip(present, values)}


__all__ = [
    "TTTMetaConfig",
    "TTT_METRIC_KEYS",
    "compute_ttt_meta_losses",
    "reduce_ttt_metrics",
    "resolve_ttt_meta_config",
    "ttt_meta_step",
]
