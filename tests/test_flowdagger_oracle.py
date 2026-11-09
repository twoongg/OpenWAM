"""Oracle noise server: the frozen student, steered by inversion toward what the same model does on the clean view.

Tiny random Wan + ActionDiT on CPU. The fake preprocessing makes the
conditioning depend on the first-frame image, so the "teacher" (clean view)
and the "student" (perturbed view) are genuinely different conditionings.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError
from openwam.flowdagger.inversion import JointSampler
from openwam.flowdagger.oracle import OracleInversionEngine, OracleInversionServer, build_oracle_server
from tests.test_flowdagger_inversion import ACTION_DIM, T_ACTION, _arch, _schedule

ACTIVE_IDX = [0, 1, 2, 3]
STEPS = 4


class _FakeNormalizer:
    """Unified-action normalizer stand-in: dims 0-3 active, affine unnormalize so it is observable."""

    _dst_index = ACTIVE_IDX
    _unify_dim = ACTION_DIM

    def unnormalize(self, x):
        return np.asarray(x) * 2.0 + 1.0

    def normalize(self, x):
        return (np.asarray(x) - 1.0) / 2.0


def _image_conditioned_preprocess(video_backbone):
    """Seeded video noise; first-frame latents depend on the image, text context is fixed."""

    def preprocess_input_for_inference(*, seed, first_frame_image=None, **_kw):
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, 4, 3, 4, 4, generator=g)
        tag = first_frame_image[0].getpixel((0, 0))[0] if first_frame_image else 0
        ref = torch.Generator().manual_seed(1000 + int(tag))
        ctx = torch.Generator().manual_seed(123)
        return {
            "noise": noise,
            "latents": noise,
            "first_frame_latents": torch.randn(1, 4, 1, 4, 4, generator=ref),
            "context": torch.randn(1, 3, 32, generator=ctx),
            "context_mask": torch.ones(1, 3, dtype=torch.bool),
        }

    video_backbone.preprocess_input_for_inference = preprocess_input_for_inference


def _cfg(**inference):
    inf = {
        "denoise_steps": STEPS,
        "denoise_mode": "sync",
        "num_frames": T_ACTION + 1,
        "video_num_frames": 9,
        "height": 8,
        "width": 8,
        **inference,
    }
    return OmegaConf.create({"inference": inf})


def _model():
    arch = _arch()
    arch.normalizer = _FakeNormalizer()
    _image_conditioned_preprocess(arch.video_backbone)
    return arch


STUDENT = Image.new("RGB", (8, 8), (40, 0, 0))
TEACHER = Image.new("RGB", (8, 8), (200, 0, 0))


def _conditions(teacher=TEACHER):
    obs = {"teacher_image": teacher} if teacher is not None else {}
    return {"observation": obs, "first_frame_image": [STUDENT], "prompt": "put the bowl on the plate"}


def _stock(arch, image):
    return arch.generate(
        schedule=_schedule(arch, STEPS),
        prompt="put the bowl on the plate",
        first_frame_image=[image],
        num_frames=9,
        action_num_frames=T_ACTION + 1,
        height=8,
        width=8,
        seed=42,
        decode_video=False,
    )["actions"]


def test_student_mode_is_the_stock_sampler_on_the_perturbed_view():
    arch = _model()
    out = OracleInversionEngine(_cfg(), arch, mode="student").generate(_conditions(teacher=None))
    np.testing.assert_array_equal(out["actions"], _stock(arch, STUDENT))


def test_teacher_mode_is_the_stock_sampler_on_the_clean_view():
    arch = _model()
    out = OracleInversionEngine(_cfg(), arch, mode="teacher").generate(_conditions())
    np.testing.assert_array_equal(out["actions"], _stock(arch, TEACHER))
    assert not np.allclose(out["actions"], _stock(arch, STUDENT))


def test_oracle_mode_steers_the_student_onto_the_teacher_action(tmp_path):
    arch = _model()
    log = tmp_path / "oracle.jsonl"
    engine = OracleInversionEngine(_cfg(), arch, mode="oracle", fp_iters=60, log_path=str(log))
    out = engine.generate(_conditions())

    teacher, student = _stock(arch, TEACHER), _stock(arch, STUDENT)
    active = np.zeros(ACTION_DIM, dtype=bool)
    active[ACTIVE_IDX] = True
    err = np.mean((out["actions"][:, active] - teacher[:, active]) ** 2)
    base = np.mean((student[:, active] - teacher[:, active]) ** 2)
    assert err < 1e-3 * base

    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["mode"] == "oracle" and record["episode"] == 0 and record["chunk"] == 0
    assert record["active_dims"] == ACTIVE_IDX
    # Errors are logged in the model's normalized space (unnormalize scales by 2 -> MSE by 4).
    assert record["base_mse"] == pytest.approx(base / 4, rel=1e-4)
    assert record["recon_mse"] < 1e-3 * record["base_mse"]
    assert len(record["base_per_dim"]) == len(ACTIVE_IDX) == len(record["recon_per_dim"])
    for key in ("std", "absmax", "frac_abs_gt3"):
        assert key in record["noise_action"]
    assert record["fp_action_update_last"] >= 0 and record["seconds"] >= 0
    # FlowDAgger §4.2 target: the student's own imagined video, only the action swapped in,
    # so the steered student keeps (nearly) its own video instead of imagining the teacher's camera.
    kwargs = engine._sampler_kwargs(_conditions())
    s, s_noise = JointSampler.prepare(arch, first_frame_image=[STUDENT], **kwargs)
    t, t_noise = JointSampler.prepare(arch, first_frame_image=[TEACHER], **kwargs)
    own, other = s.run(s_noise).video.float(), t.run(t_noise).video.float()
    free = slice(s.ref.shape[2], None)  # the conditioning frame is pinned in every run
    camera_gap = float((other[:, :, free] - own[:, :, free]).pow(2).mean() / own.var())
    assert record["video_drift"] < 1e-3 * camera_gap


def test_log_counts_chunks_within_an_episode_and_restarts_on_new_episode(tmp_path):
    arch = _model()
    log = tmp_path / "oracle.jsonl"
    engine = OracleInversionEngine(_cfg(), arch, mode="teacher", log_path=str(log))
    engine.generate(_conditions())
    engine.generate(_conditions())
    engine.new_episode()
    engine.generate(_conditions())
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["episode"], r["chunk"]) for r in records] == [(0, 0), (0, 1), (1, 0)]
    assert all(r["mode"] == "teacher" for r in records)


@pytest.mark.parametrize("mode", ["oracle", "teacher"])
def test_teacher_modes_require_the_clean_view(mode):
    engine = OracleInversionEngine(_cfg(), _model(), mode=mode)
    with pytest.raises(ValueError, match="teacher"):
        engine.generate(_conditions(teacher=None))


def test_engine_seed_sets_the_sampler_noise():
    arch = _model()
    a = OracleInversionEngine(_cfg(), arch, mode="student", seed=7).generate(_conditions(teacher=None))
    b = OracleInversionEngine(_cfg(), arch, mode="student").generate(_conditions(teacher=None))
    assert not np.allclose(a["actions"], b["actions"])


@pytest.mark.parametrize(
    "kwargs, inference, match",
    [
        ({"mode": "dagger"}, {}, "mode"),
        ({}, {"denoise_mode": "async", "variance_shift_alpha": 2.0}, "sync"),
        ({}, {"cfg_scale": 2.0}, "cfg_scale"),
    ],
)
def test_engine_rejects_settings_the_inversion_does_not_model(kwargs, inference, match):
    with pytest.raises(ValueError, match=match):
        OracleInversionEngine(_cfg(**inference), _model(), **kwargs)


# --- server ---------------------------------------------------------------------------------


class _RecordingEngine(BaseInferenceEngine):
    def __init__(self, mode="oracle"):
        super().__init__(OmegaConf.create({}), architecture=None)
        self.mode = mode
        self.episode = 0
        self.seen = []

    def generate(self, conditions):
        self.seen.append(conditions)
        return {"actions": np.zeros((T_ACTION, 3), dtype=np.float32)}

    def new_episode(self):
        self.episode += 1


def _server(mode="oracle"):
    cfg = OmegaConf.create(
        {
            "dataloader": {"multiview": True, "camera_layout": ["head", "left", "right"]},
            "inference": {"height": 16, "width": 16},
        }
    )
    return OracleInversionServer(engine=_RecordingEngine(mode), cfg=cfg)


def _png(color):
    return Image.new("RGB", (16, 16), color)


def test_server_builds_the_clean_view_with_the_robots_own_wrist_camera():
    server = _server()
    head, wrist, clean_head = _png((200, 0, 0)), _png((0, 200, 0)), _png((0, 0, 200))
    server.predict(
        {
            "images": {"head_camera": head, "left_wrist_camera": wrist},
            "teacher_images": {"head_camera": clean_head},
            "prompt": "p",
        }
    )
    (conditions,) = server.engine.seen
    obs = conditions["observation"]
    pre = ObsPreprocessor(multiview=True, camera_layout=["head", "left", "right"], img_height=16, img_width=16)
    want_teacher = pre.preprocess({"images": {"head_camera": clean_head, "left_wrist_camera": wrist}})["image"]
    want_student = pre.preprocess({"images": {"head_camera": head, "left_wrist_camera": wrist}})["image"]
    assert np.array_equal(np.asarray(obs["teacher_image"]), np.asarray(want_teacher))
    assert np.array_equal(np.asarray(conditions["first_frame_image"][0]), np.asarray(want_student))


def test_server_rejects_obs_without_the_clean_view_in_teacher_modes():
    with pytest.raises(ObsValidationError, match="teacher_images"):
        _server("oracle").predict({"images": {"head_camera": _png((1, 1, 1))}, "prompt": "p"})
    _server("student").predict({"images": {"head_camera": _png((1, 1, 1))}, "prompt": "p"})


def test_server_reset_starts_a_new_engine_episode_and_pong_names_the_mode():
    server = _server("teacher")
    server.reset()
    assert server.engine.episode == 1
    assert server._pong_payload()["oracle_mode"] == "teacher"


# --- builder ----------------------------------------------------------------------------------


def test_builder_serves_eager_without_velocity_cache(monkeypatch):
    import openwam.deploy.model_loader as model_loader

    arch = _model()
    train_cfg = OmegaConf.create(
        {"dataloader": {"num_frames": T_ACTION + 1, "video_stride": 1, "height": 8, "width": 8}}
    )
    monkeypatch.setattr(model_loader, "load_from_checkpoint_dir", lambda *a, **k: (train_cfg, arch))
    deploy = OmegaConf.create(
        {
            "inference": {"denoise_steps": STEPS, "denoise_mode": "sync", "inference_mode": "sync"},
            "optimization": {"compile": {"enabled": True}, "dit_cache": {"enabled": True}},
        }
    )
    server = build_oracle_server(deploy, "/ckpt", device="cpu", mode="oracle", fp_iters=7, seed=3)
    engine = server.engine
    assert isinstance(engine, OracleInversionEngine)
    assert engine.mode == "oracle" and engine.fp_iters == 7 and engine.seed == 3
    assert engine.cfg.optimization.compile.enabled is False
    assert engine._dit_cache is None


def test_builder_refuses_async_execution(monkeypatch):
    import openwam.deploy.model_loader as model_loader

    monkeypatch.setattr(
        model_loader, "load_from_checkpoint_dir", lambda *a, **k: (OmegaConf.create({}), SimpleNamespace())
    )
    deploy = OmegaConf.create({"inference": {"denoise_steps": STEPS, "inference_mode": "async"}})
    with pytest.raises(ValueError, match="inference_mode"):
        build_oracle_server(deploy, "/ckpt", device="cpu", mode="oracle")


# --- CLI --------------------------------------------------------------------------------------


def test_cli_forwards_oracle_flags_and_serves(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    import openwam.flowdagger.oracle as oracle

    seen = {}

    class _Server:
        def run(self, host, port):
            seen["run"] = (host, port)

    def fake_build(cfg, ckpt_dir, **kwargs):
        seen["build"] = (ckpt_dir, kwargs, cfg)
        return _Server()

    monkeypatch.setattr(oracle, "build_oracle_server", fake_build)
    script = Path(__file__).resolve().parent.parent / "scripts" / "flowdagger" / "oracle_server.py"
    spec = importlib.util.spec_from_file_location("oracle_server", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "o.jsonl"
    module.main(
        ["--ckpt-dir", "/ckpt", "--mode", "teacher", "--fp-iters", "9", "--seed", "44", "--log", str(log)]
        + ["--port", "8877", "--device", "cpu", "--denoise-steps", "6"]
    )
    ckpt_dir, kwargs, cfg = seen["build"]
    assert ckpt_dir == "/ckpt"
    assert kwargs == dict(device="cpu", ckpt_name=None, mode="teacher", fp_iters=9, seed=44, log_path=str(log))
    assert cfg.inference.denoise_steps == 6
    assert seen["run"] == ("0.0.0.0", 8877)
