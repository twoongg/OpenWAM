"""Unit tests for the video-expert TTT memory (fast-weight residual branches).

The memory follows WAM-TTT Eq. 2-5: per TTT layer ``l``

    read : Δz = θ_O · f_W(θ_Q(z))
    write: W ← W − η ∇_W ‖f_W(θ_K(h)) − θ_V(h)‖²   (key-value reconstruction)

with a SwiGLU fast-weight network ``f_W`` (LaCT / Spatial-TTT style).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from openwam.model.ttt import VideoTTTMemory

DIM = 64
HEADS = 2
HEAD_DIM = 16
HIDDEN = 32


def _memory(layers=(0,), weight_norm=False, seed=0) -> VideoTTTMemory:
    torch.manual_seed(seed)
    return VideoTTTMemory(
        dim=DIM,
        layers=list(layers),
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        hidden_dim=HIDDEN,
        weight_norm=weight_norm,
    )


def _write(mem: VideoTTTMemory, layer: int, h: torch.Tensor, lr: float = 1.0, create_graph: bool = False):
    with mem.writing(lr=lr, create_graph=create_graph):
        mem.residual(layer, h)


def _unbatched(weights, sample: int = 0):
    return [w[sample] if w.dim() == 4 else w for w in weights]


def test_residual_is_zero_before_training():
    """θ_O is zero-initialised, so inserting TTT leaves the pretrained WAM unchanged."""
    mem = _memory()
    z = torch.randn(2, 10, DIM)
    out = mem.residual(0, z)
    assert out.shape == z.shape
    assert torch.count_nonzero(out) == 0


def test_has_layer_reports_configured_layers_only():
    mem = _memory(layers=(1, 3))
    assert mem.has_layer(1) and mem.has_layer(3)
    assert not mem.has_layer(0) and not mem.has_layer(2)


def test_write_reduces_reconstruction_loss_on_written_video():
    mem = _memory()
    h = torch.randn(1, 20, DIM)
    before = mem.reconstruction_loss(0, h)
    _write(mem, 0, h)
    after = mem.reconstruction_loss(0, h)
    assert before.shape == (1,)
    assert (after < before).all()


def test_read_without_write_does_not_change_fast_weights():
    mem = _memory()
    init = [w.detach().clone() for w in mem.fast_weights(0)]
    mem.residual(0, torch.randn(1, 20, DIM))
    for now, ref in zip(_unbatched(mem.fast_weights(0)), init):
        torch.testing.assert_close(now, ref)


def test_reset_restores_initial_fast_weights():
    mem = _memory()
    init = [w.detach().clone() for w in mem.fast_weights(0)]
    _write(mem, 0, torch.randn(1, 20, DIM))
    changed = _unbatched(mem.fast_weights(0))
    assert any(not torch.allclose(c, i) for c, i in zip(changed, init))
    mem.reset()
    for now, ref in zip(_unbatched(mem.fast_weights(0)), init):
        torch.testing.assert_close(now, ref)


def test_successive_writes_accumulate():
    """Memory persists across writes (evolving across test cases): A then B ≠ B alone."""
    h_a = torch.randn(1, 20, DIM)
    h_b = torch.randn(1, 20, DIM)

    mem_ab = _memory()
    _write(mem_ab, 0, h_a)
    _write(mem_ab, 0, h_b)

    mem_b = _memory()
    _write(mem_b, 0, h_b)

    diffs = [not torch.allclose(x, y) for x, y in zip(mem_ab.fast_weights(0), mem_b.fast_weights(0))]
    assert any(diffs)


def test_batch_samples_keep_independent_memories():
    h_a = torch.randn(1, 20, DIM)
    h_b = torch.randn(1, 20, DIM)

    mem_batch = _memory()
    _write(mem_batch, 0, torch.cat([h_a, h_b], dim=0))

    mem_single = _memory()
    _write(mem_single, 0, h_a)

    for batched, single in zip(mem_batch.fast_weights(0), mem_single.fast_weights(0)):
        torch.testing.assert_close(batched[0], single[0])


def test_write_only_touches_its_own_layer():
    mem = _memory(layers=(0, 1))
    init_1 = [w.detach().clone() for w in mem.fast_weights(1)]
    _write(mem, 0, torch.randn(1, 20, DIM))
    for now, ref in zip(_unbatched(mem.fast_weights(1)), init_1):
        torch.testing.assert_close(now, ref)


def test_weight_norm_keeps_row_norms_across_many_writes():
    """50 sequential writes (one per test case) must not let the fast weights blow up."""
    mem = _memory(weight_norm=True)
    init = [w.detach().clone() for w in mem.fast_weights(0)]
    for _ in range(50):
        _write(mem, 0, torch.randn(1, 20, DIM))
    for now, ref in zip(_unbatched(mem.fast_weights(0)), init):
        torch.testing.assert_close(now.norm(dim=-1), ref.norm(dim=-1), rtol=1e-4, atol=1e-4)


def test_write_works_under_no_grad_and_leaves_no_graph():
    """Test-time writes run inside the deploy loop's ``torch.no_grad()``."""
    mem = _memory()
    init = [w.detach().clone() for w in mem.fast_weights(0)]
    with torch.no_grad():
        _write(mem, 0, torch.randn(1, 20, DIM))
    weights = mem.fast_weights(0)
    assert all(not w.requires_grad for w in weights)
    assert any(not torch.allclose(w, i) for w, i in zip(_unbatched(weights), init))


def test_target_loss_gradient_flows_through_write_to_slow_weights():
    """Meta-training: the target-episode loss must reach θ_K / θ_V / W_init through the write."""
    mem = _memory()
    branch = mem.branch(0)
    nn.init.normal_(branch.o_proj.weight, std=0.1)

    _write(mem, 0, torch.randn(1, 20, DIM), create_graph=True)
    out = mem.residual(0, torch.randn(1, 20, DIM))
    out.pow(2).mean().backward()

    for name in ("k_proj", "v_proj"):
        grad = getattr(branch, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.init_fast_weights())


def test_create_graph_write_inside_no_grad_still_reaches_slow_weights():
    """Meta-training runs the context pass with the frozen backbone under no_grad
    (no backbone activations kept); the TTT write itself must still build the graph."""
    mem = _memory(weight_norm=True)
    branch = mem.branch(0)
    nn.init.normal_(branch.o_proj.weight, std=0.1)

    with torch.no_grad():
        _write(mem, 0, torch.randn(1, 20, DIM), create_graph=True)
    assert all(w.requires_grad for w in mem.fast_weights(0))

    out = mem.residual(0, torch.randn(1, 20, DIM))
    out.pow(2).mean().backward()
    for name in ("k_proj", "v_proj"):
        grad = getattr(branch, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.init_fast_weights())


def test_write_without_create_graph_blocks_meta_gradient():
    mem = _memory()
    branch = mem.branch(0)
    nn.init.normal_(branch.o_proj.weight, std=0.1)

    _write(mem, 0, torch.randn(1, 20, DIM), create_graph=False)
    out = mem.residual(0, torch.randn(1, 20, DIM))
    out.pow(2).mean().backward()

    assert branch.k_proj.weight.grad is None or branch.k_proj.weight.grad.abs().sum() == 0


def test_residual_preserves_input_dtype():
    mem = _memory().to(torch.bfloat16)
    z = torch.randn(1, 10, DIM, dtype=torch.bfloat16)
    assert mem.residual(0, z).dtype == torch.bfloat16


def test_residual_rejects_unknown_layer():
    mem = _memory(layers=(0,))
    with pytest.raises(KeyError):
        mem.residual(5, torch.randn(1, 10, DIM))
