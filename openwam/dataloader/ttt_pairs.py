"""Same-task context → target episode pairs for video-TTT meta-training.

Each item is an ordinary training window (the *target*) plus ``ttt_context``:
windows from ``context_episodes`` *other* episodes of the same task, each
covered by ``context_windows`` evenly spaced windows in time order. The trainer
writes the context windows into the TTT memory, then computes the usual loss on
the target — the meta-training analogue of writing a successful episode before
the next test case.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset


def _config_get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class RoboDojoTTTPairDataset(BaseDataset):
    """Wrap a :class:`MultiTaskRoboDojoDataset` with same-task context episodes."""

    @classmethod
    def from_config(cls, config, split: str = "train"):
        base = MultiTaskRoboDojoDataset.from_config(config, split=split)
        return cls(
            base,
            context_episodes=int(_config_get(config, "ttt_context_episodes", 1)),
            context_windows=int(_config_get(config, "ttt_context_windows", 4)),
            seed=int(_config_get(config, "seed", 0) or 0),
        )

    def __init__(
        self,
        base: MultiTaskRoboDojoDataset,
        *,
        context_episodes: int = 1,
        context_windows: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        if context_episodes < 1 or context_windows < 1:
            raise ValueError("context_episodes and context_windows must be >= 1")
        for task in base.tasks:
            available = base.sub_dataset(task).num_episodes - 1
            if available < context_episodes:
                raise ValueError(
                    f"task {task!r} has {available + 1} episodes; context_episodes={context_episodes} "
                    "needs that many episodes besides the target"
                )
        self.base = base
        self.context_episodes = int(context_episodes)
        self.context_windows = int(context_windows)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # BaseDataset contract (normalisation stats, action width) follows the wrapped corpus.
    @property
    def action_dim(self) -> int:
        return self.base.action_dim

    @property
    def normalization_stats(self):
        return self.base.normalization_stats

    @property
    def normalization_stats_path(self):
        return self.base.normalization_stats_path

    def denormalize_action(self, action):
        return self.base.denormalize_action(action)

    def __len__(self) -> int:
        return len(self.base)

    def _context_starts(self, episode_length: int, num_frames: int) -> list[int]:
        last = max(episode_length - num_frames, 0)
        return sorted({int(round(s)) for s in np.linspace(0, last, self.context_windows)})

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = self.base[index]
        sub = self.base.sub_dataset(target["task_name"])
        rng = np.random.default_rng([self.seed, self.epoch, int(index) % len(self)])
        others = [e for e in range(sub.num_episodes) if e != target["episode_index"]]
        chosen = rng.choice(others, size=self.context_episodes, replace=False)

        context = []
        for episode_index in chosen:
            length = sub.episode_length(int(episode_index))
            for start in self._context_starts(length, sub.num_frames):
                context.append(sub.build_window(int(episode_index), start))
        return {**target, "ttt_context": context}


__all__ = ["RoboDojoTTTPairDataset"]
