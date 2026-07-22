# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional fused NPU kernels for exact HierMoE current-route planning."""

from __future__ import annotations

import importlib
import os
from functools import cache
from pathlib import Path
from types import ModuleType

from veomni.utils.import_utils import is_torch_npu_available


@cache
def get_hiermoe_planner_npu_ops() -> ModuleType | None:
    """Load the package-local NPU extension when it has been built."""

    if not is_torch_npu_available():
        return None
    package_dir = Path(__file__).resolve().parent
    custom_opp = package_dir / "build" / "opp_install" / "packages" / "vendors" / "customize"
    if not custom_opp.is_dir():
        return None
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    custom_opp_text = str(custom_opp)
    if custom_opp_text not in entries:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join((custom_opp_text, *entries))
    try:
        return importlib.import_module(f"{__name__}._hiermoe_npu_ops")
    except ImportError:
        return None


__all__ = ["get_hiermoe_planner_npu_ops"]
