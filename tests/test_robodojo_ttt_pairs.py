"""Same-task context/target episode pairs for TTT meta-training on RoboDojo."""

from __future__ import annotations

from pathlib import Path

import pytest

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset
from tests.test_robodojo_dataloader import write_episode, write_stats

T = 12
NUM_FRAMES = 3


def _corpus(root: Path, tasks=("task_a", "task_b", "task_c"), episodes=3, T=T) -> Path:
    for task in tasks:
        for episode in range(episodes):
            write_episode(root, task=task, episode=episode, T=T, instruction=f"Do {task}.")
    return write_stats(root / "stats.npy")


def _common(root: Path, stats: Path) -> dict:
    return {
        "dataset_dir": root,
        "normalization_stats_path": stats,
        "normalize_mode": "min-max",
        "num_frames": NUM_FRAMES,
        "height": 48,
        "width": 64,
        "video_stride": 1,
        "unify_action": True,
        "unify_action_map": ["0-9", "34-43"],
    }


def _pairs(root: Path, stats: Path, **kw):
    from openwam.dataloader.ttt_pairs import RoboDojoTTTPairDataset

    base = MultiTaskRoboDojoDataset(**_common(root, stats), tasks=kw.pop("tasks", None))
    return RoboDojoTTTPairDataset(base, **kw)


# --- task filter ------------------------------------------------------------------------------


def test_multitask_tasks_filter_keeps_only_requested_tasks(tmp_path):
    stats = _corpus(tmp_path)
    dataset = MultiTaskRoboDojoDataset(**_common(tmp_path, stats), tasks=["task_c", "task_a"])
    assert dataset.tasks == ["task_a", "task_c"]
    assert {dataset[i]["task_name"] for i in range(len(dataset))} == {"task_a", "task_c"}


def test_multitask_tasks_filter_rejects_missing_task(tmp_path):
    stats = _corpus(tmp_path)
    with pytest.raises(FileNotFoundError, match="task_zzz"):
        MultiTaskRoboDojoDataset(**_common(tmp_path, stats), tasks=["task_a", "task_zzz"])


def test_from_config_threads_tasks_filter(tmp_path):
    stats = _corpus(tmp_path)
    config = {**_common(tmp_path, stats), "type": "robodojo", "tasks": ["task_b"]}
    config["dataset_dir"] = str(tmp_path)
    config["normalization_stats_path"] = str(stats)
    dataset = build_dataset(config, split="train")
    assert dataset.tasks == ["task_b"]


# --- pair dataset -----------------------------------------------------------------------------


def test_pair_item_is_target_window_plus_same_task_context_from_other_episodes(tmp_path):
    stats = _corpus(tmp_path)
    pairs = _pairs(tmp_path, stats, context_episodes=2, context_windows=3, seed=0)
    assert len(pairs) == len(pairs.base)
    for index in range(0, len(pairs), 7):
        item = pairs[index]
        target_episode = item["episode_index"]
        context = item["ttt_context"]
        assert len(context) == 2 * 3
        assert all(c["task_name"] == item["task_name"] for c in context)
        assert all(c["prompt"] == item["prompt"] for c in context)
        assert all(c["episode_index"] != target_episode for c in context)
        assert len({c["episode_index"] for c in context}) == 2


def test_context_windows_cover_each_episode_in_time_order(tmp_path):
    stats = _corpus(tmp_path)
    pairs = _pairs(tmp_path, stats, context_episodes=1, context_windows=3, seed=0)
    context = pairs[0]["ttt_context"]
    starts = [c["start_frame"] for c in context]
    assert starts == sorted(starts)
    assert starts[0] == 0
    assert starts[-1] == T - NUM_FRAMES  # last window ends at the episode end


def test_pairs_are_reproducible_per_epoch_and_change_across_epochs(tmp_path):
    stats = _corpus(tmp_path, episodes=6)
    a = _pairs(tmp_path, stats, context_episodes=1, context_windows=1, seed=3)
    b = _pairs(tmp_path, stats, context_episodes=1, context_windows=1, seed=3)

    def picks(ds):
        return [ds[i]["ttt_context"][0]["episode_index"] for i in range(len(ds))]

    assert picks(a) == picks(b)
    first_epoch = picks(a)
    a.set_epoch(1)
    assert picks(a) != first_epoch


def test_pair_dataset_requires_enough_episodes_per_task(tmp_path):
    stats = _corpus(tmp_path, episodes=2)
    with pytest.raises(ValueError, match="context_episodes"):
        _pairs(tmp_path, stats, context_episodes=2, context_windows=1, seed=0)


def test_registry_builds_pair_dataset_on_selected_tasks(tmp_path):
    from openwam.dataloader.ttt_pairs import RoboDojoTTTPairDataset

    stats = _corpus(tmp_path)
    config = {
        **_common(tmp_path, stats),
        "type": "robodojo_ttt_pairs",
        "dataset_dir": str(tmp_path),
        "normalization_stats_path": str(stats),
        "tasks": ["task_a", "task_b"],
        "ttt_context_episodes": 1,
        "ttt_context_windows": 2,
        "seed": 0,
    }
    dataset = build_dataset(config, split="train")
    assert isinstance(dataset, RoboDojoTTTPairDataset)
    assert dataset.base.tasks == ["task_a", "task_b"]
    assert len(dataset[0]["ttt_context"]) == 2
