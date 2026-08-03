#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Deprecated compatibility entry point for the canonical PlaceMoE planner."""

from __future__ import annotations

import warnings

from scripts.profile.placemoe_planner import main


if __name__ == "__main__":
    warnings.warn(
        "build_hiermoe_recursive_classifier_layout.py is deprecated; use plan_placemoe.py.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
