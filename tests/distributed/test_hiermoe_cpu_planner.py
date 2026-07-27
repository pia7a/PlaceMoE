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

import os
import time

import torch
import torch.distributed as dist

from tests.tools.launch_utils import torchrun
from veomni.distributed.moe.hiermoe.cpu_planner import (
    AsyncCPULayerOwnerPlanner,
    CPUExactPlanner,
    CPUHCCLBatchedPlanner,
    CPULayerOwnerPlanner,
    SharedMemoryCPUPlannerProcess,
    assert_exact_plan_match,
    balanced_layer_owner_ranks,
    resolve_cpu_planner_resources,
)
from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.npu_layer_owner_planner import NPULayerOwnerPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _planner(ep_size: int, *, slots_per_rank: int = 3) -> GreedyCommunicationPlanner:
    group_sizes = (ep_size,) if ep_size <= 2 else (2, ep_size)
    return GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=group_sizes, source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        communication_scale=0.75,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=0.5,
        candidate_chunk_size=8,
        max_copies=4,
        assume_unique_routes=True,
    )


def _assert_balanced(values: tuple[int, ...], ep_size: int) -> None:
    counts = torch.bincount(torch.tensor(values), minlength=ep_size)
    assert int(counts.max() - counts.min()) <= 1


def test_layer_owner_mapping_supports_ep16_ep32_ep64():
    for ep_size in (16, 32, 64):
        owners = balanced_layer_owner_ranks(48, ep_size)
        assert len(owners) == 48
        assert all(0 <= value < ep_size for value in owners)
        assert len(set(owners)) == min(48, ep_size)
        _assert_balanced(owners, ep_size)


def test_cpu_resource_resolution_avoids_local_rank_oversubscription():
    unbound = resolve_cpu_planner_resources(
        layer_count=48,
        local_process_count=8,
        visible_cpu_cores=192,
        reserve_cpu_cores=2,
    )
    assert unbound.cpu_cores_per_rank == 24
    assert unbound.usable_cpu_cores == 22
    assert unbound.layer_workers == 22
    assert unbound.intraop_threads == 1

    pinned = resolve_cpu_planner_resources(
        layer_count=48,
        local_process_count=8,
        visible_cpu_cores=24,
        reserve_cpu_cores=2,
    )
    assert pinned.cpu_cores_per_rank == 24
    assert pinned.layer_workers == 22


def test_single_process_cpu_facade_is_bit_exact():
    planner = _planner(2)
    layout = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    owners = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    routes = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 3]] * 4, dtype=torch.long)
    kwargs = {
        "source_ranks": 0,
        "max_swaps": 1,
        "max_replicas": 1,
        "layer_seeds": [11],
        "step": 7,
        "communication_scales": [0.75],
        "forward_compute_per_assignment": [0.25],
        "forward_compute_constant": [0.5],
        "skip_final_route_update": True,
    }
    reference = planner.plan_layers([routes], [layout], [owners], **kwargs)[0]
    actual = CPUExactPlanner(planner).plan_layers([routes], [layout], [owners], **kwargs)[0]
    assert_exact_plan_match(reference, actual)


def test_shared_memory_cpu_process_is_bit_exact():
    planner = _planner(1, slots_per_rank=3)
    layout = torch.tensor([0, 1, 0], dtype=torch.long)
    owners = torch.tensor([0, 1], dtype=torch.long)
    routes = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
    reference = CPUExactPlanner(planner).plan_layers(
        [routes],
        [layout],
        [owners],
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=[11],
        step=7,
        communication_scales=[0.75],
        forward_compute_per_assignment=[0.25],
        forward_compute_constant=[0.5],
        skip_final_route_update=True,
    )[0]
    cpu_id = min(os.sched_getaffinity(0))
    process = SharedMemoryCPUPlannerProcess(planner_cpu_ids=[cpu_id])
    try:
        process.submit(
            slot=1,
            source_step=7,
            planner=planner,
            selected_experts=[process.share_cpu_tensor(routes)],
            slot_to_logical=[layout],
            owner_slots=[owners],
            source_rank=0,
            max_swaps=1,
            max_replicas=1,
            layer_seeds=[11],
            communication_scales=[0.75],
            compute_slopes=[0.25],
            compute_constants=[0.5],
        )
        source_step, shared_statistics = process.wait_collective(1)
        assert source_step == 7
        assert shared_statistics.is_shared()
        process.complete_collective(1)
        completion = process.wait_result(1)
        assert completion.source_step == 7
        assert completion.result is not None
        assert_exact_plan_match(reference, completion.result.plans[0])
    finally:
        process.close()


def _layer_owner_exact_parity_worker():
    rank = dist.get_rank()
    torch.set_num_threads(1)
    layout = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    owners = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    alternate_layout = torch.tensor([0, 1, 0, 2, 3, 2], dtype=torch.long)
    alternate_owners = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    base_routes = (
        torch.tensor([[0, 1], [0, 2], [1, 3], [0, 3], [2, 3], [1, 2]], dtype=torch.long)
        if rank == 0
        else torch.tensor([[2, 3], [0, 3], [1, 2], [0, 1], [0, 2]], dtype=torch.long)
    )
    routes = [torch.roll(base_routes, shifts=layer, dims=0) for layer in range(3)]
    layouts = [layout, alternate_layout, layout]
    owner_slots = [owners, alternate_owners, owners]
    seeds = [11, 17, 23]

    def reducer(tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    reference_planner = _planner(2)
    reference_planner.reducer = reducer
    reference = reference_planner.plan_layers(
        routes,
        layouts,
        owner_slots,
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=seeds,
        step=7,
        skip_final_route_update=True,
    )

    owner_planner = CPULayerOwnerPlanner(
        _planner(2),
        process_group=dist.group.WORLD,
        local_process_count=1,
        reserve_cpu_cores=0,
        layer_workers=2,
        intraop_threads=1,
    )
    result = owner_planner.plan_layers(
        routes,
        layouts,
        owner_slots,
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=seeds,
        step=7,
    )
    assert result.owner_ranks == (0, 0, 1)
    assert result.timing.owned_layer_count == (2 if rank == 0 else 1)
    assert result.timing.local_payload_bytes > 0
    for expected, actual in zip(reference, result.plans, strict=True):
        assert_exact_plan_match(expected, actual)

    accelerator_owner = NPULayerOwnerPlanner(
        _planner(2),
        process_group=dist.group.WORLD,
    )
    accelerator_result = accelerator_owner.plan_layers(
        routes,
        layouts,
        owner_slots,
        source_rank=rank,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=seeds,
        step=7,
    )
    assert accelerator_result.owner_ranks == (0, 1, 0)
    assert accelerator_result.timing.owned_layer_count == (2 if rank == 0 else 1)
    assert accelerator_result.timing.sent_statistic_bytes > 0
    for expected, actual in zip(reference, accelerator_result.plans, strict=True):
        assert_exact_plan_match(expected, actual)

    batched = CPUHCCLBatchedPlanner(
        _planner(2),
        reducer=reducer,
        local_process_count=1,
        reserve_cpu_cores=0,
        layer_workers=2,
        intraop_threads=1,
    ).plan_layers(
        routes,
        layouts,
        owner_slots,
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=seeds,
        step=7,
    )
    assert batched.timing.local_payload_bytes == batched.timing.received_payload_bytes
    for expected, actual in zip(reference, batched.plans, strict=True):
        assert_exact_plan_match(expected, actual)

    # EP can exceed the number of layers (for example EP64 with 48 layers).
    # Ranks with no owned layer must participate with a zero-length receive.
    single_layer = owner_planner.plan_layers(
        routes[:1],
        layouts[:1],
        owner_slots[:1],
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=seeds[:1],
        step=7,
    )
    assert single_layer.owner_ranks == (0,)
    assert single_layer.timing.owned_layer_count == (1 if rank == 0 else 0)
    assert single_layer.timing.received_payload_bytes == (
        2 * single_layer.timing.local_payload_bytes if rank == 0 else 0
    )
    assert_exact_plan_match(reference[0], single_layer.plans[0])


def test_distributed_layer_owner_matches_full_exact_bit_for_bit():
    torchrun(_layer_owner_exact_parity_worker, world_size=2, backend="gloo")


def test_async_cpu_planner_uses_two_buffers_and_rejects_stale_layout():
    layout = torch.tensor([0, 1, 0], dtype=torch.long)
    owners = torch.tensor([0, 1], dtype=torch.long)
    routes = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
    backend = CPULayerOwnerPlanner(
        _planner(1),
        local_process_count=1,
        reserve_cpu_cores=0,
        layer_workers=1,
        intraop_threads=1,
    )
    asynchronous = AsyncCPULayerOwnerPlanner(backend)
    kwargs = {
        "source_ranks": 0,
        "max_swaps": 1,
        "max_replicas": 1,
        "layer_seeds": [11],
    }
    try:
        assert asynchronous.submit(
            7,
            [routes],
            [layout],
            [owners],
            placement_versions=[3],
            **kwargs,
        )
        assert asynchronous.submit(
            8,
            [routes],
            [layout],
            [owners],
            placement_versions=[3],
            **kwargs,
        )
        assert not asynchronous.submit(
            9,
            [routes],
            [layout],
            [owners],
            placement_versions=[3],
            **kwargs,
        )

        deadline = time.monotonic() + 5.0
        completion = None
        while completion is None and time.monotonic() < deadline:
            completion = asynchronous.poll(
                current_placement_versions=[4],
                current_layouts=[layout],
                current_owner_slots=[owners],
            )
            if completion is None:
                time.sleep(0.01)
        assert completion is not None
        assert not completion.valid
        assert completion.stale_reason == "placement_version_changed"
        assert asynchronous.wait_next(timeout=5.0) is not None
    finally:
        asynchronous.close()
