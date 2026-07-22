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

"""NPU parity tests for exact greedy swap/cover candidate scoring."""

import pytest
import torch

from veomni.distributed.moe.hiermoe.greedy_planner import (
    GreedyCommunicationPlanner,
    _route_hash,
    assign_tokens_to_copies_greedy,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops
from veomni.utils.import_utils import is_torch_npu_available


pytestmark = pytest.mark.skipif(not is_torch_npu_available(), reason="Ascend NPU is required")


def _require_cover_score():
    extension = get_hiermoe_planner_npu_ops()
    if extension is None or not hasattr(extension, "cover_score"):
        pytest.skip("The exact greedy cover-score extension is not built")
    return extension


def _planner(*, fused: bool = True) -> GreedyCommunicationPlanner:
    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test", local_world_size=2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=2,
        candidate_chunk_size=4,
    )
    if not fused:
        planner._fused_candidate_local_deltas = lambda *_args, **_kwargs: None
    return planner


def _case():
    selected = torch.tensor(
        [
            [0, 0, 1],
            [2, 3, 2],
            [1, 3, 0],
            [0, 2, 3],
            [3, 3, 1],
            [2, 0, 1],
        ],
        dtype=torch.long,
        device="npu",
    )
    selected = selected.repeat((5, 1))
    layout = torch.tensor([0, 3, 1, 2, 2, 3, 3, 1], dtype=torch.long, device="npu")
    owners = torch.tensor([0, 2, 4, 6], dtype=torch.long, device="npu")
    sources = torch.zeros((selected.shape[0],), dtype=torch.long, device="npu")
    ordinals = torch.arange(selected.shape[0], dtype=torch.long, device="npu")
    return selected, layout, owners, sources, ordinals


def test_fused_candidate_deltas_match_exact_pytorch_for_every_swap_and_cover():
    _require_cover_score()
    planner = _planner()
    selected, layout, owners, sources, ordinals = _case()
    cover_slots = torch.tensor([1, 3, 5, 7], dtype=torch.long, device="npu")
    rows = torch.cat((planner._swap_rows(layout, owners), planner._cover_rows(layout, owners, cover_slots)))
    physical = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
    )
    occupancies = planner._token_level_occupancies(physical)
    copy_slots = planner._copy_table(layout, 4)
    route_hashes = _route_hash(selected, token_ordinals=ordinals, step=7, layer_seed=11)
    route_hash_by_expert = torch.zeros((selected.shape[0], 4), dtype=torch.long, device="npu")
    route_hash_by_expert.scatter_(1, selected, route_hashes)
    multiplicity_by_expert = torch.zeros((selected.shape[0], 4), dtype=torch.int32, device="npu")
    multiplicity_by_expert.scatter_add_(1, selected, torch.ones_like(selected, dtype=torch.int32))
    route_ranks = torch.div(physical, 2, rounding_mode="floor")
    rank_by_expert = torch.zeros_like(route_hash_by_expert)
    rank_by_expert.scatter_(1, selected, route_ranks)

    eager = planner._candidate_local_deltas(
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        occupancies=occupancies,
        source_ranks=sources,
        route_hash_by_expert=route_hash_by_expert,
        multiplicity_by_expert=multiplicity_by_expert,
        rank_by_expert=rank_by_expert,
    )
    fused = planner._fused_candidate_local_deltas(
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        physical=physical,
        occupancies=occupancies,
        source_ranks=sources,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        num_experts=4,
    )
    torch.npu.synchronize()

    assert fused is not None
    torch.testing.assert_close(fused.cpu(), eager.cpu(), rtol=0, atol=0)


def test_fused_plan_matches_exact_pytorch_plan():
    _require_cover_score()
    selected, layout, owners, sources, ordinals = _case()
    kwargs = dict(
        source_ranks=sources,
        token_ordinals=ordinals,
        max_swaps=1,
        max_replicas=1,
        step=7,
        layer_seed=11,
    )
    eager = _planner(fused=False).plan(selected, layout, owners, **kwargs)
    fused = _planner(fused=True).plan(selected, layout, owners, **kwargs)
    torch.npu.synchronize()

    assert fused.actions == eager.actions
    assert fused.final_layout == eager.final_layout
    assert fused.final_owner_slots == eager.final_owner_slots
    assert fused.baseline_cost == eager.baseline_cost
    assert fused.final_cost == eager.final_cost
    torch.testing.assert_close(fused.local_physical_routes.cpu(), eager.local_physical_routes.cpu(), rtol=0, atol=0)
