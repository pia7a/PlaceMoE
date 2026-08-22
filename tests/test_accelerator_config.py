# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import pytest

from veomni.arguments.arguments_types import AcceleratorConfig


def test_accelerator_config_adds_expert_parallel_dimension() -> None:
    config = AcceleratorConfig(ep_size=16, ep_outside=False)

    assert config.extra_parallel_names == ["ep"]
    assert config.extra_parallel_sizes == [16]
    assert config.extra_parallel_placement_innermost == [False]


def test_accelerator_config_accepts_serialized_expert_parallel_dimension() -> None:
    config = AcceleratorConfig(
        ep_size=16,
        ep_outside=False,
        extra_parallel_names=["ep"],
        extra_parallel_sizes=[16],
        extra_parallel_placement_innermost=[False],
    )

    assert config.extra_parallel_names == ["ep"]
    assert config.extra_parallel_sizes == [16]
    assert config.extra_parallel_placement_innermost == [False]


def test_accelerator_config_canonicalizes_legacy_duplicate_ep_dimensions() -> None:
    config = AcceleratorConfig(
        ep_size=16,
        ep_outside=False,
        extra_parallel_names=["ep", "ep"],
        extra_parallel_sizes=[16, 16],
        extra_parallel_placement_innermost=[False, False],
    )

    assert config.extra_parallel_names == ["ep"]
    assert config.extra_parallel_sizes == [16]
    assert config.extra_parallel_placement_innermost == [False]


def test_accelerator_config_rejects_conflicting_serialized_ep() -> None:
    with pytest.raises(ValueError, match="must match ep_size"):
        AcceleratorConfig(
            ep_size=16,
            extra_parallel_names=["ep"],
            extra_parallel_sizes=[8],
            extra_parallel_placement_innermost=[False],
        )
