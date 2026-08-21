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

from collections import defaultdict

import pytest
import torch

from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops
from veomni.utils.import_utils import is_torch_npu_available


pytestmark = pytest.mark.skipif(not is_torch_npu_available(), reason="Ascend NPU is required")


def _extension():
    extension = get_hiermoe_planner_npu_ops()
    if extension is None or not hasattr(extension, "quota_policy"):
        pytest.skip("The HierMoE quota-policy NPU operator is not built")
    return extension


def _multiplicity(routes: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(routes)
    for sample, row in enumerate(routes.tolist()):
        seen: set[int] = set()
        for position, logical in enumerate(row):
            if logical not in seen:
                result[sample, position] = row.count(logical)
                seen.add(logical)
    return result


def _copy_slots(layout: list[int], num_experts: int) -> list[list[int]]:
    copies = [[] for _ in range(num_experts)]
    for slot, logical in enumerate(layout):
        if logical >= 0:
            copies[logical].append(slot)
    return copies


def _destinations(
    copies: list[list[int]],
    logical: int,
    mask: int,
    slots_per_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pairs = sorted((slot // slots_per_rank, copy) for copy, slot in enumerate(copies[logical]) if mask & (1 << copy))
    return tuple(rank for rank, _ in pairs), tuple(copy for _, copy in pairs)


def _tie_mask(
    *,
    route: list[int],
    source: int,
    logical: int,
    copies: list[list[int]],
    owner_slots: list[int],
    slots_per_rank: int,
    level_sizes: tuple[int, ...],
) -> int:
    visited = (source, *(owner_slots[other] // slots_per_rank for other in route if other != logical))

    def score(destination: int) -> tuple[int, ...]:
        values = [int(all(destination // size != rank // size for rank in visited)) for size in reversed(level_sizes)]
        values.append(int(destination not in visited))
        return tuple(values)

    canonical: list[int] = []
    for copy, slot in enumerate(copies[logical]):
        rank = slot // slots_per_rank
        if all(later_slot // slots_per_rank != rank for later_slot in copies[logical][copy + 1 :]):
            canonical.append(copy)
    scores = {copy: score(copies[logical][copy] // slots_per_rank) for copy in canonical}
    best = min(scores.values())
    return sum(1 << copy for copy, value in scores.items() if value == best)


def _waterfill(loads: list[int], destinations: tuple[int, ...], total: int) -> tuple[int, ...]:
    quotas = dict.fromkeys(destinations, 0)
    ordered = sorted((loads[rank], rank) for rank in destinations)
    remaining = max(0, total)
    active = 1
    while active < len(ordered):
        required = max(0, ordered[active][0] - ordered[active - 1][0]) * active
        if required > remaining:
            break
        if required:
            increment, extra = divmod(required, active)
            for index in range(active):
                quotas[ordered[index][1]] += increment + int(index < extra)
            remaining -= required
        active += 1
    increment, extra = divmod(remaining, active)
    for index in range(active):
        quotas[ordered[index][1]] += increment + int(index < extra)
    return tuple(quotas[rank] for rank in destinations)


def _digest_prefix(
    *,
    sample_routes: torch.Tensor,
    sample_multiplicity: torch.Tensor,
    sample_sources: torch.Tensor,
    sample_ordinals: torch.Tensor,
    assignment_counts: torch.Tensor,
    layout: torch.Tensor,
    owner_slots: torch.Tensor,
    slots_per_rank: int,
    max_copies: int,
    samples_per_source: int,
    level_sizes: tuple[int, ...],
) -> tuple[int, int]:
    first = 17
    second = 29

    def feed(value: int) -> None:
        nonlocal first, second
        first = (first * 131 + value % 1048573 + 1) % 1048573
        second = (second * 257 + value % 1000003 + 1) % 1000003

    values = (
        607543,
        sample_routes.shape[0],
        sample_routes.shape[1],
        assignment_counts.shape[0],
        assignment_counts.shape[1],
        layout.numel(),
        max_copies,
        samples_per_source,
        len(level_sizes),
        level_sizes[0] if level_sizes else 1,
        level_sizes[1] if len(level_sizes) > 1 else 1,
    )
    for value in values:
        feed(int(value))
    for sample, row in enumerate(sample_routes.tolist()):
        feed(int(sample_sources[sample]))
        feed(int(sample_ordinals[sample]))
        for position, logical in enumerate(row):
            feed(logical)
            feed(int(sample_multiplicity[sample, position]))
    for value in assignment_counts.flatten().tolist():
        feed(value)
    for value in layout.tolist():
        feed(value)
    for value in owner_slots.tolist():
        feed(value)
    return first, second


def _quota_policy_reference(
    *,
    sample_routes: torch.Tensor,
    sample_multiplicity: torch.Tensor,
    sample_sources: torch.Tensor,
    sample_ordinals: torch.Tensor,
    assignment_counts: torch.Tensor,
    layouts: torch.Tensor,
    owner_slots: torch.Tensor,
    slots_per_rank: int,
    source_rank: int,
    max_copies: int,
    samples_per_source: int,
    level_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ep_size, num_experts = assignment_counts.shape
    top_k = sample_routes.shape[1]
    mask_count = 1 << max_copies
    row_capacity = samples_per_source * top_k
    row_width = 3 + 2 * max_copies
    weights = torch.zeros((2, num_experts, mask_count, max_copies), dtype=torch.long)
    configured = torch.zeros((2, num_experts, mask_count), dtype=torch.long)
    rows = torch.zeros((2, row_capacity, row_width), dtype=torch.long)
    row_counts = torch.zeros((2,), dtype=torch.long)
    digest = torch.empty((2, 2), dtype=torch.long)

    for layout_index in range(2):
        layout = layouts[layout_index].tolist()
        owners = owner_slots[layout_index].tolist()
        copies = _copy_slots(layout, num_experts)
        sample_buckets: dict[tuple[int, int, int], int] = defaultdict(int)
        for sample, route in enumerate(sample_routes.tolist()):
            source = int(sample_sources[sample])
            for position, logical in enumerate(route):
                multiplicity = int(sample_multiplicity[sample, position])
                if multiplicity == 0:
                    continue
                mask = _tie_mask(
                    route=route,
                    source=source,
                    logical=logical,
                    copies=copies,
                    owner_slots=owners,
                    slots_per_rank=slots_per_rank,
                    level_sizes=level_sizes,
                )
                sample_buckets[(source, logical, mask)] += multiplicity

        projected: dict[tuple[int, int, int], int] = {}
        for source in range(ep_size):
            for logical in range(num_experts):
                observed = {
                    mask: count
                    for (bucket_source, bucket_logical, mask), count in sample_buckets.items()
                    if bucket_source == source and bucket_logical == logical
                }
                sample_total = sum(observed.values())
                if sample_total == 0:
                    continue
                exact = int(assignment_counts[source, logical])
                base = {mask: exact * count // sample_total for mask, count in observed.items()}
                fractions = {mask: exact * count % sample_total for mask, count in observed.items()}
                remainder = exact - sum(base.values())
                order = sorted(
                    observed,
                    key=lambda mask: (
                        -fractions[mask],
                        _destinations(copies, logical, mask, slots_per_rank)[0],
                        mask,
                    ),
                )
                for mask in order[:remainder]:
                    base[mask] += 1
                for mask, total in base.items():
                    projected[(source, logical, mask)] = total

        loads = [0 for _ in range(ep_size)]
        ambiguous: list[tuple[int, int, int, tuple[int, ...], int]] = []
        for (source, logical, mask), total in projected.items():
            if total <= 0:
                continue
            destinations, _ = _destinations(copies, logical, mask, slots_per_rank)
            if len(destinations) == 1:
                loads[destinations[0]] += total
            elif len(destinations) > 1:
                ambiguous.append((source, logical, mask, destinations, total))
        ambiguous.sort(key=lambda item: (-item[4], item[0], item[1], item[3], item[2]))

        first, second = _digest_prefix(
            sample_routes=sample_routes,
            sample_multiplicity=sample_multiplicity,
            sample_sources=sample_sources,
            sample_ordinals=sample_ordinals,
            assignment_counts=assignment_counts,
            layout=layouts[layout_index],
            owner_slots=owner_slots[layout_index],
            slots_per_rank=slots_per_rank,
            max_copies=max_copies,
            samples_per_source=samples_per_source,
            level_sizes=level_sizes,
        )

        def feed(value: int) -> None:
            nonlocal first, second
            first = (first * 131 + value % 1048573 + 1) % 1048573
            second = (second * 257 + value % 1000003 + 1) % 1000003

        local_rows: list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]] = []
        for source, logical, mask, destinations, total in ambiguous:
            quotas = _waterfill(loads, destinations, total)
            for rank, quota in zip(destinations, quotas, strict=True):
                loads[rank] += quota
            for value in (21073, source, logical, len(destinations), *destinations, *quotas):
                feed(value)
            if source != source_rank:
                continue
            _, copy_indices = _destinations(copies, logical, mask, slots_per_rank)
            configured[layout_index, logical, mask] = 1
            for copy, quota in zip(copy_indices, quotas, strict=True):
                weights[layout_index, logical, mask, copy] = quota
            local_rows.append((source, logical, len(destinations), destinations, quotas))

        row_counts[layout_index] = len(local_rows)
        for row_index, (source, logical, count, destinations, quotas) in enumerate(local_rows):
            rows[layout_index, row_index, :3] = torch.tensor((source, logical, count))
            rows[layout_index, row_index, 3 : 3 + count] = torch.tensor(destinations)
            rows[
                layout_index,
                row_index,
                3 + max_copies : 3 + max_copies + count,
            ] = torch.tensor(quotas)
        digest[layout_index] = torch.tensor((first, second))
    return weights, configured, rows, row_counts, digest


def _run_quota_policy(
    *,
    sample_routes: torch.Tensor,
    sample_multiplicity: torch.Tensor,
    sample_sources: torch.Tensor,
    sample_ordinals: torch.Tensor,
    assignment_counts: torch.Tensor,
    layouts: torch.Tensor,
    owner_slots: torch.Tensor,
    slots_per_rank: int,
    source_rank: int,
    max_copies: int,
    samples_per_source: int,
    level_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    padded_levels = (*level_sizes, 1, 1)[:2]
    result = _extension().quota_policy(
        sample_routes.npu(),
        sample_multiplicity.npu(),
        sample_sources.npu(),
        sample_ordinals.npu(),
        assignment_counts.npu(),
        layouts.npu(),
        owner_slots.npu(),
        slots_per_rank,
        source_rank,
        assignment_counts.shape[0],
        max_copies,
        samples_per_source,
        len(level_sizes),
        padded_levels[0],
        padded_levels[1],
    )
    return tuple(value.cpu() for value in result)


def _assert_reference_parity(**kwargs) -> tuple[torch.Tensor, ...]:
    actual = _run_quota_policy(**kwargs)
    expected = _quota_policy_reference(**kwargs)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    return actual


def test_quota_policy_projects_sample_classes_to_exact_assignments():
    layouts = torch.tensor(
        [
            [0, 2, 1, -1, 2, 0, 1, -1],
            [0, 1, 1, 2, 2, -1, 0, -1],
        ],
        dtype=torch.long,
    )
    owners = torch.tensor([[0, 2, 4], [0, 2, 4]], dtype=torch.long)
    routes = torch.tensor([[0, 1], [0, 1], [0, 2], [1, 2], [0, 1]], dtype=torch.long)
    sources = torch.tensor([1, 1, 1, 0, 3], dtype=torch.long)
    ordinals = torch.tensor([7, 9, 11, 2, 5], dtype=torch.long)
    assignments = torch.zeros((4, 3), dtype=torch.long)
    assignments[0, 1:] = torch.tensor([3, 5])
    assignments[1] = torch.tensor([7, 4, 2])
    assignments[3, :2] = torch.tensor([2, 6])
    arguments = dict(
        sample_routes=routes,
        sample_multiplicity=_multiplicity(routes),
        sample_sources=sources,
        sample_ordinals=ordinals,
        assignment_counts=assignments,
        layouts=layouts,
        owner_slots=owners,
        slots_per_rank=2,
        source_rank=1,
        max_copies=2,
        samples_per_source=3,
        level_sizes=(),
    )
    _, _, rows, row_counts, _ = _assert_reference_parity(**arguments)
    local_rows = rows[0, : int(row_counts[0])]
    matching = local_rows[(local_rows[:, 0] == 1) & (local_rows[:, 1] == 0)]
    assert matching.shape[0] == 1
    assert matching[0, 2].item() == 2
    assert matching[0, 3:5].tolist() == [0, 2]
    assert sum(matching[0, 5:7].tolist()) == 5


def test_quota_policy_fast_path_matches_single_and_same_rank_copies():
    layouts = torch.tensor(
        [
            [0, 0, 1, -1, 2, -1, -1, -1],
            [0, -1, 1, -1, 2, 0, -1, -1],
        ],
        dtype=torch.long,
    )
    owners = torch.tensor([[0, 2, 4], [0, 2, 4]], dtype=torch.long)
    routes = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 1]], dtype=torch.long)
    assignments = torch.zeros((4, 3), dtype=torch.long)
    assignments[0] = torch.tensor([9, 5, 4])
    assignments[1] = torch.tensor([0, 7, 6])
    assignments[3] = torch.tensor([3, 2, 8])
    _assert_reference_parity(
        sample_routes=routes,
        sample_multiplicity=_multiplicity(routes),
        sample_sources=torch.tensor([0, 0, 1, 3], dtype=torch.long),
        sample_ordinals=torch.tensor([2, 4, 6, 8], dtype=torch.long),
        assignment_counts=assignments,
        layouts=layouts,
        owner_slots=owners,
        slots_per_rank=2,
        source_rank=0,
        max_copies=2,
        samples_per_source=2,
        level_sizes=(2,),
    )


@pytest.mark.parametrize("ep_size", (16, 32, 64))
def test_quota_policy_supports_dynamic_ep_shapes(ep_size: int):
    layout = torch.full((ep_size,), -1, dtype=torch.long)
    layout[0] = 0
    layout[1] = 1
    layout[-2] = 0
    layout[-1] = 1
    layouts = torch.stack((layout, torch.roll(layout, shifts=2)))
    owners = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    routes = torch.tensor([[0, 1], [0, 0], [1, 0]], dtype=torch.long)
    sources = torch.tensor([0, ep_size // 2, ep_size - 1], dtype=torch.long)
    assignments = torch.zeros((ep_size, 2), dtype=torch.long)
    assignments[0] = torch.tensor([5, 3])
    assignments[ep_size // 2, 0] = 7
    assignments[ep_size - 1] = torch.tensor([4, 6])
    _assert_reference_parity(
        sample_routes=routes,
        sample_multiplicity=_multiplicity(routes),
        sample_sources=sources,
        sample_ordinals=torch.tensor([1, 3, 5], dtype=torch.long),
        assignment_counts=assignments,
        layouts=layouts,
        owner_slots=owners,
        slots_per_rank=1,
        source_rank=ep_size // 2,
        max_copies=2,
        samples_per_source=1,
        level_sizes=(ep_size // 2, 2),
    )


def test_quota_policy_handles_empty_samples_without_host_reconstruction():
    layouts = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long)
    owners = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    routes = torch.empty((0, 2), dtype=torch.long)
    _assert_reference_parity(
        sample_routes=routes,
        sample_multiplicity=torch.empty_like(routes),
        sample_sources=torch.empty((0,), dtype=torch.long),
        sample_ordinals=torch.empty((0,), dtype=torch.long),
        assignment_counts=torch.zeros((4, 2), dtype=torch.long),
        layouts=layouts,
        owner_slots=owners,
        slots_per_rank=1,
        source_rank=0,
        max_copies=2,
        samples_per_source=2,
        level_sizes=(2,),
    )


def test_quota_policy_marks_invalid_multiplicity_in_digest():
    layouts = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long)
    owners = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    routes = torch.tensor([[0, 0]], dtype=torch.long)
    _, configured, rows, row_counts, digest = _run_quota_policy(
        sample_routes=routes,
        sample_multiplicity=torch.tensor([[1, 1]], dtype=torch.long),
        sample_sources=torch.tensor([0], dtype=torch.long),
        sample_ordinals=torch.tensor([0], dtype=torch.long),
        assignment_counts=torch.tensor([[2, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.long),
        layouts=layouts,
        owner_slots=owners,
        slots_per_rank=1,
        source_rank=0,
        max_copies=2,
        samples_per_source=1,
        level_sizes=(),
    )
    assert configured.count_nonzero().item() == 0
    assert rows.count_nonzero().item() == 0
    assert row_counts.tolist() == [0, 0]
    assert digest.tolist() == [[-1, -1], [-1, -1]]
