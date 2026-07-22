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

"""Production-path parity tests for the fused CoRe-MoE NPU planner."""

import pytest
import torch

from veomni.distributed.moe.hiermoe.core_planner import CoReMoEPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops
from veomni.utils.import_utils import is_torch_npu_available


pytestmark = pytest.mark.skipif(not is_torch_npu_available(), reason="Ascend NPU is required")


def _require_production_ops():
    extension = get_hiermoe_planner_npu_ops()
    required = ("swap_select_with_stats", "replica_project", "replica_match", "quota_policy", "quota_map")
    if extension is None or not all(hasattr(extension, name) for name in required):
        pytest.skip("The complete CoRe-MoE production planner extension is not built")
    return extension


def _planner(*, fused: bool) -> CoReMoEPlanner:
    ep_size = 4

    def gather_fixed(payload: torch.Tensor) -> torch.Tensor:
        return payload.unsqueeze(0).expand(ep_size, -1).clone()

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=(2, 4), source="test", local_world_size=2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=64,
        bytes_per_element=2,
        slots_per_rank=2,
        communication_scale=1.0,
        forward_compute_per_assignment=0.25,
        reducer=lambda value: value * ep_size,
        gather_fixed=gather_fixed,
        route_sample_size=256,
    )
    if not fused:
        planner._fused_planner_extension = lambda _device: None
    return planner


def _placement() -> tuple[torch.Tensor, torch.Tensor]:
    owners = torch.arange(8, dtype=torch.long, device="npu")
    return owners.clone(), owners


@pytest.mark.parametrize("max_swaps", (0, 1, 4))
def test_fused_production_plan_matches_exact_eager_for_swap_prefix(max_swaps: int):
    _require_production_ops()
    selected = torch.tensor(
        [[0, 1]] * 96 + [[0, 2]] * 64 + [[1, 3]] * 48 + [[4, 5]] * 8 + [[6, 7]] * 8,
        dtype=torch.long,
        device="npu",
    )
    layout, owners = _placement()

    eager = _planner(fused=False).plan(
        selected,
        layout,
        owners,
        source_ranks=0,
        max_swaps=max_swaps,
        max_replicas=0,
        step=3,
        layer_seed=11,
    )
    fused = _planner(fused=True).plan(
        selected,
        layout,
        owners,
        source_ranks=0,
        max_swaps=max_swaps,
        max_replicas=0,
        step=3,
        layer_seed=11,
    )
    torch.npu.synchronize()

    assert fused.actions == eager.actions
    assert fused.final_layout == eager.final_layout
    assert fused.final_owner_slots == eager.final_owner_slots
    assert fused.quota_policy == eager.quota_policy
    assert fused.baseline_cost == eager.baseline_cost
    assert fused.final_cost == eager.final_cost
    torch.testing.assert_close(fused.local_physical_routes.cpu(), eager.local_physical_routes.cpu(), rtol=0, atol=0)
    if max_swaps:
        assert [(action.src_logical, action.dst_logical) for action in fused.actions] == [(0, 3)]


def test_fused_production_p0s1_matches_exact_eager_with_nonzero_bottleneck_rank():
    _require_production_ops()
    selected = torch.full((256, 1), 3, dtype=torch.long, device="npu")
    owners = torch.tensor((0, 2, 4, 6), dtype=torch.long, device="npu")
    layout = torch.tensor((0, -1, 1, -1, 2, -1, 3, -1), dtype=torch.long, device="npu")

    eager = _planner(fused=False).plan(
        selected,
        layout,
        owners,
        source_ranks=0,
        max_swaps=0,
        max_replicas=1,
        step=3,
        layer_seed=11,
    )
    fused = _planner(fused=True).plan(
        selected,
        layout,
        owners,
        source_ranks=0,
        max_swaps=0,
        max_replicas=1,
        step=3,
        layer_seed=11,
    )
    torch.npu.synchronize()

    assert eager.baseline_cost.peak_communication_rank != 0
    assert eager.baseline_cost.peak_compute_rank != 0
    assert eager.replica_rounds == fused.replica_rounds
    assert fused.actions == eager.actions
    assert fused.final_layout == eager.final_layout
    assert fused.final_owner_slots == eager.final_owner_slots
    assert fused.quota_policy == eager.quota_policy
    assert fused.baseline_cost == eager.baseline_cost
    assert fused.final_cost == eager.final_cost
    torch.testing.assert_close(fused.local_physical_routes.cpu(), eager.local_physical_routes.cpu(), rtol=0, atol=0)
