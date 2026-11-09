"""LoRA adapters for the video / action experts of a World-Action Model.

Each targeted ``nn.Linear`` keeps its pretrained weight and gains a low-rank
update through a forward hook::

    y = W x + b + (alpha / rank) · B (A x),      B zero-initialised

so a freshly attached adapter leaves the pretrained model's outputs unchanged.

All adapter weights live in one module (``architecture.lora``), so the
state-dict prefix ``lora.`` is the whole adapter: it is what a checkpoint may
lack (pretrained weights), what "train only the adapter" selects, and what a
per-task reset re-initialises. The wrapped modules' own state-dict keys are
untouched, so released checkpoints load as before.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Expert name → regex for its transformer blocks (dotted names from the architecture root).
LORA_EXPERTS = {
    "action": r"action_backbone\.blocks\.\d+",
    "video": r"video_backbone\.dit\.blocks\.\d+",
}
# Module group → regex for the Linear layers inside one block (Wan DiTBlock / ActionDiT block layout).
LORA_MODULES = {
    "self_attn": r"self_attn\.(q|k|v|o)",
    "cross_attn": r"cross_attn\.(q|k|v|o)",
    "ffn": r"ffn\.(0|2)",
}


def select_lora_targets(root: nn.Module, *, experts: Iterable[str], modules: Iterable[str]) -> list[str]:
    """Dotted names of the ``nn.Linear`` layers in ``root`` that the adapters wrap."""
    experts, modules = list(experts), list(modules)
    bad = [e for e in experts if e not in LORA_EXPERTS]
    if bad or not experts:
        raise ValueError(f"lora.experts must be a non-empty subset of {sorted(LORA_EXPERTS)}, got {experts}")
    bad = [m for m in modules if m not in LORA_MODULES]
    if bad or not modules:
        raise ValueError(f"lora.modules must be a non-empty subset of {sorted(LORA_MODULES)}, got {modules}")

    module_re = "|".join(LORA_MODULES[m] for m in modules)
    linears = [name for name, mod in root.named_modules() if isinstance(mod, nn.Linear)]
    targets: list[str] = []
    for expert in experts:
        pattern = re.compile(rf"{LORA_EXPERTS[expert]}\.(?:{module_re})")
        found = [name for name in linears if pattern.fullmatch(name)]
        if not found:
            raise ValueError(f"lora.experts includes {expert!r} but no Linear layer matches {modules} in it")
        targets.extend(found)
    return targets


class LoRAAdapter(nn.Module):
    """Low-rank update ``(alpha / rank) · B A x`` for one Linear layer (B starts at zero)."""

    def __init__(self, in_features: int, out_features: int, *, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"lora.rank must be positive, got {rank}")
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        # Always real CPU tensors: the backbone may still be an uninitialised (meta) shell
        # that a checkpoint load fills later, and adapters absent from it keep this init.
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, device="cpu"))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.rank, device="cpu"))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        h = F.linear(self.dropout(x), self.lora_A.to(x.dtype))
        return F.linear(h, self.lora_B.to(x.dtype)) * self.scale


class LoRAAdapters(nn.Module):
    """The adapters of every targeted layer, attached to ``root`` with forward hooks."""

    def __init__(self, root: nn.Module, targets: Sequence[str], *, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.targets = tuple(targets)
        self.adapters = nn.ModuleDict()
        self._hooks = []
        for name in self.targets:
            linear = root.get_submodule(name)
            adapter = LoRAAdapter(linear.in_features, linear.out_features, rank=rank, alpha=alpha, dropout=dropout)
            self.adapters[self.key(name)] = adapter
            self._hooks.append(linear.register_forward_hook(_lora_hook(adapter)))

    @staticmethod
    def key(target: str) -> str:
        """ModuleDict key for a dotted target name (dots are not allowed in keys)."""
        return target.replace(".", "__")

    def reset_parameters(self) -> None:
        """Forget everything learned: every adapter returns to a zero update."""
        for adapter in self.adapters.values():
            adapter.reset_parameters()


def _lora_hook(adapter: LoRAAdapter):
    def hook(_module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        return output + adapter(args[0])

    return hook


__all__ = ["LORA_EXPERTS", "LORA_MODULES", "LoRAAdapter", "LoRAAdapters", "select_lora_targets"]
