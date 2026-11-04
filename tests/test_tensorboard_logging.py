"""TensorBoard logging as a drop-in for the trainer's wandb run (.log(dict, step=) / .finish())."""

from __future__ import annotations

import os

from omegaconf import OmegaConf


def _scalars(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(log_dir)
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def test_tensorboard_run_writes_logged_scalars(tmp_path):
    from openwam.train.utils.training_utils import TensorBoardRun

    run = TensorBoardRun(str(tmp_path))
    run.log({"train/loss": 1.5, "train/ttt_context_gain": 0.25}, step=3)
    run.log({"train/loss": 1.0}, step=4)
    run.finish()

    scalars = _scalars(str(tmp_path))
    assert scalars["train/loss"] == [(3, 1.5), (4, 1.0)]
    assert scalars["train/ttt_context_gain"] == [(3, 0.25)]


def test_init_tensorboard_is_off_by_default(tmp_path):
    from openwam.train.utils.training_utils import init_tensorboard

    cfg = OmegaConf.create({"project": {"wandb": {"project": None}}})
    assert init_tensorboard(cfg, str(tmp_path)) is None


def test_init_tensorboard_defaults_to_output_dir(tmp_path):
    from openwam.train.utils.training_utils import init_tensorboard

    cfg = OmegaConf.create({"project": {"tensorboard": {"enabled": True, "log_dir": None}}})
    run = init_tensorboard(cfg, str(tmp_path))
    run.log({"train/loss": 2.0}, step=1)
    run.finish()
    assert _scalars(os.path.join(str(tmp_path), "tensorboard"))["train/loss"] == [(1, 2.0)]


def test_combined_runs_fan_out_and_skip_missing(tmp_path):
    from openwam.train.utils.training_utils import TensorBoardRun, combine_runs

    class _Rec:
        def __init__(self):
            self.logged, self.finished = [], False

        def log(self, data, step=None):
            self.logged.append((step, dict(data)))

        def finish(self):
            self.finished = True

    assert combine_runs(None, None) is None
    rec = _Rec()
    assert combine_runs(None, rec) is rec

    tb = TensorBoardRun(str(tmp_path))
    both = combine_runs(rec, tb)
    both.log({"train/loss": 3.0}, step=7)
    both.finish()
    assert rec.logged == [(7, {"train/loss": 3.0})] and rec.finished
    assert _scalars(str(tmp_path))["train/loss"] == [(7, 3.0)]
