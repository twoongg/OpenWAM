"""DualSystem joint self-attention architecture.

True joint attention (MMDiT / FastWAM MoT style): at every transformer
layer, the video and action backbones each compute Q/K/V independently
through ``pre_attn_at_layer``; :class:`DualSystemMoTDriver` concatenates the
two modalities, runs a single mixed self-attention, splits the result,
and feeds each slice back through ``post_attn_at_layer``.

The driver lives on ``self._mot_driver`` as a plain Python object with
no parameters; ``forward`` delegates the per-layer loop to it.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO
from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    self_attn_compile_cfg,
    torch_compile_kwargs,
)
from openwam.model.lora import LoRAAdapters, select_lora_targets
from openwam.model.ttt import VideoTTTMemory, add_ttt_residual

logger = logging.getLogger(__name__)


@register_architecture(
    "dual_system_self_attn",
    status="supported",
    note="DualSystem joint self-attention: MoT-style mixed attention at every layer.",
    framework="dual_system",
    variant="joint_self_attn",
)
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with true joint self-attention (FastWAM MoT pattern)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._mot_driver: DualSystemMoTDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_mot_run_joint_loop: Optional[Callable[..., Tuple[object, object]]] = None
        self.video_ttt: VideoTTTMemory | None = None
        self.lora: LoRAAdapters | None = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)

        # FastWAM-Joint compat: action residual hidden_dim may differ from
        # video_dim. The MoT driver only requires num_heads / attn_head_dim
        # parity (validated at DualSystemMoTDriver.__init__).
        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))

        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )
        self.video_ttt = self._build_video_ttt(cfg, video_dim=video_dim, num_layers=len(bl))
        self.lora = self._build_lora(cfg)

        # MoT driver is built once both backbones are available. The video
        # backbone is normally constructed in ``BaseWAMArchitecture._init_video_backbone``
        # (already done by ``super().__init__``), so we can build the driver
        # right here. Tests that swap in a mock video backbone after init call
        # :meth:`build_mot_driver` directly.
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(cfg.get("mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(cfg.get("attention_mask_mode", ACTION_SEES_VIDEO)),
            "video_attention_mask_mode": str(cfg.get("video_attention_mask_mode", "first_frame_causal")),
        }
        if self.video_backbone is not None:
            self.build_mot_driver()

    def build_mot_driver(self) -> DualSystemMoTDriver:
        """Construct the :class:`DualSystemMoTDriver` from the current backbones.

        Re-callable; raises if either backbone is missing. Tests that swap in
        a mock video backbone after ``__init__`` should call this method to
        wire up the driver afterwards.
        """
        if self.video_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: video_backbone is not "
                "set. Construct the architecture with a video_backbone config or attach "
                "one before calling this method."
            )
        if self.action_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: action_backbone is not "
                "set. Architecture must be built from a non-None cfg."
            )

        self._mot_driver = DualSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            ttt=self.video_ttt,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    def _build_video_ttt(self, cfg, *, video_dim: int, num_layers: int) -> VideoTTTMemory | None:
        """Build the video-expert TTT memory from ``cfg.ttt`` (None when absent / disabled)."""
        ttt_cfg = self._cfg_get(cfg, "ttt", None)
        if not ttt_cfg or not bool(self._cfg_get(ttt_cfg, "enabled", False)):
            return None
        layers = self._cfg_get(ttt_cfg, "layers", None)
        layers = list(range(num_layers)) if layers is None else [int(layer) for layer in layers]
        bad = [layer for layer in layers if not 0 <= layer < num_layers]
        if bad:
            raise ValueError(f"ttt.layers {bad} out of range for a {num_layers}-layer video backbone")
        memory = VideoTTTMemory(
            dim=video_dim,
            layers=layers,
            num_heads=int(self._cfg_get(ttt_cfg, "num_heads", 8)),
            head_dim=int(self._cfg_get(ttt_cfg, "head_dim", 64)),
            hidden_dim=int(self._cfg_get(ttt_cfg, "hidden_dim", 128)),
            weight_norm=bool(self._cfg_get(ttt_cfg, "weight_norm", True)),
        )
        logger.info("Video TTT memory on layers %s", list(memory.layers))
        return memory

    def _build_lora(self, cfg) -> LoRAAdapters | None:
        """Attach LoRA adapters from ``cfg.lora`` (None when absent / disabled)."""
        lora_cfg = self._cfg_get(cfg, "lora", None)
        if not lora_cfg or not bool(self._cfg_get(lora_cfg, "enabled", False)):
            return None
        targets = select_lora_targets(
            self,
            experts=self._cfg_get(lora_cfg, "experts", ["action"]),
            modules=self._cfg_get(lora_cfg, "modules", ["self_attn", "ffn"]),
        )
        rank = int(self._cfg_get(lora_cfg, "rank", 16))
        adapters = LoRAAdapters(
            self,
            targets,
            rank=rank,
            alpha=float(self._cfg_get(lora_cfg, "alpha", rank)),
            dropout=float(self._cfg_get(lora_cfg, "dropout", 0.0)),
        )
        logger.info("LoRA adapters (rank %d) on %d Linear layers", rank, len(targets))
        return adapters

    def _optional_checkpoint_prefixes(self) -> tuple[str, ...]:
        # Benchmark-finetuned checkpoints predate the TTT memory and the LoRA adapters;
        # both start as an exact no-op (zero-initialised output projections).
        added = (("video_ttt.", self.video_ttt), ("lora.", self.lora))
        return tuple(prefix for prefix, module in added if module is not None)

    def restrict_training_to(self, prefixes: tuple[str, ...]) -> None:
        """Make parameters under ``prefixes`` the only trainable ones.

        Unlike ``freeze_modules`` this only flips ``requires_grad``: it must not wrap
        the backbones in ``no_grad``, because the loss reaches the trainable
        parameters by back-propagating through the frozen video / action experts.
        """
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(prefixes))

    def restrict_training_to_ttt(self) -> None:
        """Make the TTT slow weights the only trainable parameters (TTT meta-training)."""
        if self.video_ttt is None:
            raise RuntimeError("restrict_training_to_ttt: ttt is not enabled for this architecture")
        self.restrict_training_to(("video_ttt.",))

    def restrict_training_to_lora(self) -> None:
        """Make the LoRA adapters the only trainable parameters (adapter finetuning)."""
        if self.lora is None:
            raise RuntimeError("restrict_training_to_lora: lora is not enabled for this architecture")
        self.restrict_training_to(("lora.",))

    @property
    def mot_driver(self) -> DualSystemMoTDriver | None:
        """The MoT joint-attention driver (None if the architecture wasn't fully built)."""
        return self._mot_driver

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the eval-time MoT joint loop when requested."""

        self._compiled_mot_run_joint_loop = None
        if self.lora is not None:
            # The adapters run as forward hooks inside the compiled regions; stay eager
            # rather than rely on the compiler tracing them.
            if compile_enabled(compile_cfg, default=False, strict=True):
                logger.warning("LoRA adapters are attached: deploy compile disabled, running eager.")
            return
        super().apply_compile_optimizations(compile_cfg)
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = self_attn_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("dual self-attn MoT compile disabled by config; running eager.")
            return
        if self.video_backbone is None or self.action_backbone is None:
            logger.warning("dual self-attn MoT compile requested before backbones are ready; running eager.")
            return

        driver = self._mot_driver or self.build_mot_driver()

        def _run_joint_loop(vstate, astate):
            return driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        try:
            self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_mot_run_joint_loop = None
            logger.warning("dual self-attn MoT torch.compile setup failed; running eager: %s", exc)
            return
        logger.info("Enabled dual self-attn MoT compile with torch.compile kwargs=%s", kwargs)

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Opt every Wan backbone into 4D + clean-prefix-aligned t_mod (mirrors
        # TI2V's native ``seperated_timestep + fuse_vae_embedding_in_latents``
        # path; TI2V itself fires that path first so these kwargs are inert for
        # it). VACE and I2V do NOT emit ``first_frame_latents`` (VACE routes
        # its first-frame condition through ``vace_context``; I2V uses the
        # ``y`` channel), so for them ``zero_clean_prefix_t_mod`` is
        # structurally inert — kept on only so the joint MoT driver gets the
        # 4D ``t_mod`` it needs. ``setdefault`` so explicit callers can still
        # pass False.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                z_in = vstate.hidden_states
                vstate = vb.run_block(block_id, vstate)
                vstate = add_ttt_residual(self.video_ttt, block_id, z_in, vstate)
            return vb.finalize(vstate), None

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()

        astate = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        compiled_loop = getattr(self, "_compiled_mot_run_joint_loop", None)
        if self.video_ttt is not None:
            # Fast-weight state changes between calls (and writes run autograd);
            # keep the TTT path eager rather than re-tracing on every update.
            compiled_loop = None
        if compiled_loop is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            try:
                vstate, astate = compiled_loop(vstate, astate)
            except Exception as exc:
                self._compiled_mot_run_joint_loop = None
                logger.warning("dual self-attn compiled MoT loop failed; falling back to eager: %s", exc)
                vstate, astate = driver.run_joint_loop(
                    vstate,
                    astate,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
        else:
            vstate, astate = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vb.finalize(vstate), ab.extract_prediction(astate)


__all__ = ["DualSystemSelfAttnArchitecture"]
