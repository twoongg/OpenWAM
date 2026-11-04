"""Test-time-training memory for the video expert of a World-Action Model.

Each TTT layer ``l`` adds a residual branch to the video stream of one DiT
block (WAM-TTT, Eq. 1-2)::

    z^(l+1) = block_l(z^(l)) + Δz^(l),     Δz^(l) = θ_O · f_W(θ_Q(z^(l)))

``f_W`` is a per-head SwiGLU fast-weight network (LaCT / Spatial-TTT style)::

    f_W(x) = W1 · (silu(W0 · x) ⊙ (W2 · x))

Writing a video into memory runs one SGD step on the key-value
reconstruction loss (WAM-TTT, Eq. 3/5)::

    W ← W − η ∇_W mean ‖f_W(θ_K(h)) − θ_V(h)‖²

Slow parameters (θ_{Q,K,V,O}, W_init) are meta-trained; at test time only the
fast weights change. ``θ_O`` is zero-initialised so a freshly inserted memory
leaves the pretrained WAM's outputs unchanged.

Fast-weight state lives on :class:`VideoTTTMemory` (not in ``state_dict``) and
persists across forward passes until :meth:`VideoTTTMemory.reset`, which is
what lets the memory accumulate over sequential test cases.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FastWeights = Tuple[Tensor, Tensor, Tensor]  # (w0, w1, w2), each (H, r, c) or (B, H, r, c)

_EPS = 1e-6


def _l2_normalize(x: Tensor) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + _EPS)


def _apply_fast_weights(weights: FastWeights, x: Tensor) -> Tensor:
    """SwiGLU fast-weight forward. ``x``: (B, H, L, d) → (B, H, L, d)."""
    w0, w1, w2 = weights
    gate = x @ w0.transpose(-1, -2)
    up = x @ w2.transpose(-1, -2)
    return (F.silu(gate) * up) @ w1.transpose(-1, -2)


def _batched(w: Tensor, batch_size: int) -> Tensor:
    """Give every sample its own copy of a fast weight (keeps autograd history)."""
    if w.dim() == 3:
        return w.unsqueeze(0).expand(batch_size, *w.shape).clone()
    if w.shape[0] == batch_size:
        return w
    if w.shape[0] == 1:
        return w.expand(batch_size, *w.shape[1:]).clone()
    raise ValueError(f"fast-weight batch {w.shape[0]} does not match input batch {batch_size}")


class VideoTTTBranch(nn.Module):
    """Slow parameters of one TTT layer plus the fast-weight read / write rules."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        inner = num_heads * head_dim

        self.in_norm = nn.RMSNorm(dim, eps=_EPS, elementwise_affine=False)
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, dim, bias=False)
        nn.init.zeros_(self.o_proj.weight)
        self.out_norm_weight = nn.Parameter(torch.ones(head_dim))

        # W_init: learnable initial fast weights.
        self.w0 = nn.Parameter(torch.randn(num_heads, hidden_dim, head_dim) / math.sqrt(head_dim))
        self.w1 = nn.Parameter(torch.randn(num_heads, head_dim, hidden_dim) / math.sqrt(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_heads, hidden_dim, head_dim) / math.sqrt(head_dim))

    def init_fast_weights(self) -> FastWeights:
        return (self.w0, self.w1, self.w2)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2).float()

    def _keys_values(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        hn = self.in_norm(h)
        k = _l2_normalize(self._split_heads(self.k_proj(hn)))
        v = self._split_heads(self.v_proj(hn))
        return k, v

    def read(self, z: Tensor, weights: FastWeights) -> Tensor:
        q = _l2_normalize(self._split_heads(self.q_proj(self.in_norm(z))))
        weights = tuple(w.float() for w in weights)
        o = _apply_fast_weights(weights, q)
        o = F.rms_norm(o, (self.head_dim,), self.out_norm_weight.float(), _EPS)
        B, _, L, _ = o.shape
        o = o.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        return self.o_proj(o.to(z.dtype))

    def kv_loss(self, h: Tensor, weights: FastWeights) -> Tensor:
        """Per-sample key-value reconstruction loss, shape (B,)."""
        k, v = self._keys_values(h)
        weights = tuple(w.float() for w in weights)
        return (_apply_fast_weights(weights, k) - v).pow(2).mean(dim=(1, 2, 3))

    def update(
        self,
        h: Tensor,
        weights: FastWeights,
        *,
        lr: float,
        create_graph: bool,
        weight_norm: bool,
    ) -> FastWeights:
        """One inner SGD step on the reconstruction loss of ``h``; returns new (B, H, r, c) weights."""
        B = h.shape[0]
        # Everything stays under enable_grad: meta-training runs the (frozen) backbone
        # of the context pass under no_grad, and the new weights must still carry the
        # graph back to θ_K / θ_V / W_init when create_graph=True.
        with torch.enable_grad():
            k, v = self._keys_values(h)
            ws = [_batched(w.float(), B) for w in weights]
            if not create_graph:
                k, v = k.detach(), v.detach()
                ws = [w.detach().requires_grad_(True) for w in ws]
            else:
                ws = [w if w.requires_grad else w.detach().requires_grad_(True) for w in ws]
            loss = (_apply_fast_weights(tuple(ws), k) - v).pow(2).mean(dim=(1, 2, 3)).sum()
            grads = torch.autograd.grad(loss, ws, create_graph=create_graph)

            new = [w - lr * g for w, g in zip(ws, grads)]
            if weight_norm:
                ref_norms = [w.float().norm(dim=-1, keepdim=True) for w in self.init_fast_weights()]
                new = [n / (n.norm(dim=-1, keepdim=True) + _EPS) * r for n, r in zip(new, ref_norms)]
        if not create_graph:
            new = [n.detach() for n in new]
        return tuple(new)


class VideoTTTMemory(nn.Module):
    """TTT residual branches for a chosen set of video-expert DiT layers."""

    def __init__(
        self,
        dim: int,
        layers: Sequence[int],
        *,
        num_heads: int = 4,
        head_dim: int = 64,
        hidden_dim: int = 128,
        weight_norm: bool = True,
    ):
        super().__init__()
        self.layers = tuple(sorted({int(l) for l in layers}))
        if not self.layers:
            raise ValueError("VideoTTTMemory needs at least one layer")
        self.weight_norm = bool(weight_norm)
        self.branches = nn.ModuleDict(
            {str(l): VideoTTTBranch(dim, num_heads, head_dim, hidden_dim) for l in self.layers}
        )
        self._state: dict[int, FastWeights] = {}
        self._write_cfg: Optional[Tuple[float, bool]] = None

    def has_layer(self, layer_id: int) -> bool:
        return str(layer_id) in self.branches

    def branch(self, layer_id: int) -> VideoTTTBranch:
        return self.branches[str(layer_id)]

    def fast_weights(self, layer_id: int) -> FastWeights:
        state = self._state.get(layer_id)
        if state is not None:
            return state
        return self.branch(layer_id).init_fast_weights()

    def reset(self) -> None:
        """Forget everything written; fast weights fall back to W_init."""
        self._state.clear()

    @property
    def is_writing(self) -> bool:
        return self._write_cfg is not None

    @contextmanager
    def writing(self, *, lr: float, create_graph: bool = False) -> Iterator["VideoTTTMemory"]:
        """Within this context every :meth:`residual` call also writes its input into memory."""
        prev = self._write_cfg
        self._write_cfg = (float(lr), bool(create_graph))
        try:
            yield self
        finally:
            self._write_cfg = prev

    def residual(self, layer_id: int, z: Tensor) -> Tensor:
        """TTT residual Δz for the video hidden states ``z`` entering ``layer_id``."""
        branch = self.branch(layer_id)
        weights = self.fast_weights(layer_id)
        out = branch.read(z, weights)
        if self._write_cfg is not None:
            lr, create_graph = self._write_cfg
            self._state[layer_id] = branch.update(
                z, weights, lr=lr, create_graph=create_graph, weight_norm=self.weight_norm
            )
        return out

    def reconstruction_loss(self, layer_id: int, h: Tensor) -> Tensor:
        return self.branch(layer_id).kv_loss(h, self.fast_weights(layer_id))


def add_ttt_residual(memory: Optional[VideoTTTMemory], layer_id: int, z_in: Tensor, state):
    """Add ``memory``'s residual for ``layer_id`` to ``state.hidden_states`` (no-op if absent).

    ``z_in`` is the video hidden state that entered the block (WAM-TTT Eq. 1-2).
    """
    if memory is None or not memory.has_layer(layer_id):
        return state
    if z_in.dim() != 3:
        raise NotImplementedError(
            f"VideoTTTMemory expects (B, L, D) video hidden states (Wan backbones); got shape {tuple(z_in.shape)}"
        )
    state.hidden_states = state.hidden_states + memory.residual(layer_id, z_in)
    return state
