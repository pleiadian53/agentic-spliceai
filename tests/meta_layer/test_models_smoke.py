"""Smoke tests for sequence-level meta-splice model architectures.

For each architecture registered in ``ARCH_REGISTRY``, verify that the
model:

  - instantiates from the factory with a tiny config
  - produces the expected output shape on synthetic tensors
  - propagates gradients on backward
  - roundtrips through ``state_dict`` save/load
  - has ``cfg.effective_context_padding`` consistent with ``cfg.receptive_field``

These tests run in seconds with synthetic tensors — no FASTA, no
predictions, no gene cache. The goal is to catch the "would fail on pod"
class of errors (import failures, shape bugs, gradient detach bugs,
state_dict key drift, RF/context-padding wiring drift) before paying
for pod time.

Run::

    pytest tests/meta_layer/test_models_smoke.py -v
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from agentic_spliceai.splice_engine.meta_layer.models.factory import (
    ARCH_REGISTRY,
    build_model,
)


# Per-architecture tiny config. Adding a new arch?  Add an entry here.
#   - Keep hidden_dim small (~8) to make ckpt small and forward fast.
#   - Keep window_size small enough that input tensors fit in RAM in seconds.
#   - Use ``num_blocks=2`` or arch-equivalent for the same reason.
TINY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "v3": dict(
        seq_n_blocks=2,
        seq_dilations=[1, 1],
        mm_n_blocks=2,
        mm_dilations=[1, 1],
        fusion_n_blocks=2,
        fusion_dilations=[1, 1],
        window_size=512,
    ),
    "v4_xattn": dict(
        seq_n_blocks=2,
        seq_dilations=[1, 1],
        mm_n_blocks=2,
        mm_dilations=[1, 1],
        fusion_n_blocks=2,
        fusion_dilations=[1, 1],
        window_size=512,
        n_heads=2,  # hidden_dim=8 must be divisible by n_heads
    ),
    "v5_transformer": dict(
        stem_layers=2,
        seq_n_layers=2,
        signal_n_layers=1,
        fusion_n_layers=1,
        n_heads=2,  # hidden_dim=8 must be divisible by n_heads (and even for PE)
        ff_mult=2,
        window_size=512,
    ),
}


def _make_tiny_model(arch: str):
    """Instantiate ``arch`` from the factory with the tiny config."""
    return build_model(
        arch=arch,
        mode="m1",
        hidden_dim=8,
        mm_channels=9,
        activation="gelu",
        **TINY_CONFIGS[arch],
    )


def _synthetic_batch(cfg, batch_size: int = 2):
    """Synthetic forward-input tensors matching the model's contract."""
    seq = torch.randn(batch_size, 4, cfg.total_input_length)
    base = torch.softmax(torch.randn(batch_size, cfg.window_size, 3), dim=-1)
    mm = torch.randn(batch_size, cfg.mm_channels, cfg.window_size)
    return seq, base, mm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=ARCH_REGISTRY)
def arch_name(request) -> str:
    """Parametrize over every registered architecture."""
    return request.param


@pytest.fixture
def model_and_cfg(arch_name):
    """Tiny model + its config for the parametrized architecture."""
    model, cfg = _make_tiny_model(arch_name)
    return model, cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_registered(arch_name: str) -> None:
    """Every name in ``ARCH_REGISTRY`` must build successfully."""
    model, cfg = _make_tiny_model(arch_name)
    assert model is not None
    assert cfg is not None


def test_forward_shape(arch_name: str, model_and_cfg) -> None:
    """Forward pass must yield ``[B, W, num_classes]``."""
    model, cfg = model_and_cfg
    model.eval()
    seq, base, mm = _synthetic_batch(cfg)
    with torch.no_grad():
        out = model(seq, base, mm)
    assert out.shape == (2, cfg.window_size, cfg.num_classes), (
        f"[{arch_name}] output shape {tuple(out.shape)} != "
        f"({2}, {cfg.window_size}, {cfg.num_classes})"
    )


def test_gradient_flow(arch_name: str, model_and_cfg) -> None:
    """Backward must propagate non-zero gradients to trainable params."""
    model, cfg = model_and_cfg
    model.train()
    seq, base, mm = _synthetic_batch(cfg)
    out = model(seq, base, mm)
    loss = out.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad, f"[{arch_name}] no gradient reached any trainable parameter"


def test_checkpoint_roundtrip(arch_name: str, model_and_cfg) -> None:
    """``state_dict`` save/load must reproduce forward outputs exactly."""
    model, cfg = model_and_cfg
    model.eval()
    seq, base, mm = _synthetic_batch(cfg)
    with torch.no_grad():
        before = model(seq, base, mm)

    sd = model.state_dict()
    model2, _ = _make_tiny_model(arch_name)
    model2.load_state_dict(sd)
    model2.eval()
    with torch.no_grad():
        after = model2(seq, base, mm)

    max_diff = (before - after).abs().max().item()
    assert torch.allclose(before, after, atol=1e-6), (
        f"[{arch_name}] checkpoint roundtrip mismatch: max diff {max_diff:.2e}"
    )


def test_effective_context_padding_matches_rf(arch_name: str, model_and_cfg) -> None:
    """When ``context_padding`` isn't overridden, must equal ``receptive_field``."""
    model, cfg = model_and_cfg
    if cfg.context_padding is None:
        assert cfg.effective_context_padding == cfg.receptive_field, (
            f"[{arch_name}] effective_ctx={cfg.effective_context_padding} "
            f"!= receptive_field={cfg.receptive_field} (no override set)"
        )
    if hasattr(model, "receptive_field"):
        assert model.receptive_field == cfg.receptive_field, (
            f"[{arch_name}] model.receptive_field={model.receptive_field} "
            f"!= cfg.receptive_field={cfg.receptive_field}"
        )


def test_explicit_context_padding_override(arch_name: str) -> None:
    """An explicit ``context_padding`` must win over the derived RF."""
    extra = {**TINY_CONFIGS[arch_name], "context_padding": 64}
    _, cfg = build_model(
        arch=arch_name, mode="m1",
        hidden_dim=8, mm_channels=9, activation="gelu",
        **extra,
    )
    assert cfg.effective_context_padding == 64, (
        f"[{arch_name}] explicit context_padding override not honored"
    )


# ---------------------------------------------------------------------------
# Stage 1: wider-RF override (v3-specific — extend to future archs as added)
# ---------------------------------------------------------------------------


def test_v3_wider_dilations_yield_larger_rf() -> None:
    """Wider ``seq_dilations`` → larger RF → ``effective_context_padding``
    grows in lockstep → model forward still works at the new input length.
    """
    _, cfg_base = _make_tiny_model("v3")
    base_rf = cfg_base.receptive_field

    wider = {
        **TINY_CONFIGS["v3"],
        "seq_dilations": [1, 2, 4, 8],
        "seq_n_blocks": 4,
    }
    model_wide, cfg_wide = build_model(
        arch="v3", mode="m1",
        hidden_dim=8, mm_channels=9, activation="gelu",
        **wider,
    )

    assert cfg_wide.receptive_field > base_rf, (
        f"wider dilations should increase RF (base={base_rf}, "
        f"wide={cfg_wide.receptive_field})"
    )
    assert cfg_wide.effective_context_padding == cfg_wide.receptive_field, (
        "effective_context_padding must track receptive_field when no override"
    )

    # Forward still works with the new (larger) input length
    model_wide.eval()
    seq = torch.randn(2, 4, cfg_wide.total_input_length)
    base = torch.softmax(torch.randn(2, cfg_wide.window_size, 3), dim=-1)
    mm = torch.randn(2, cfg_wide.mm_channels, cfg_wide.window_size)
    with torch.no_grad():
        out = model_wide(seq, base, mm)
    assert out.shape == (2, cfg_wide.window_size, cfg_wide.num_classes)
