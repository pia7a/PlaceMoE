# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

from pathlib import Path

import pytest

from veomni.distributed.moe.hiermoe.placemoe.runtime.config import PlaceMoEPlannerResources
from veomni.distributed.moe.hiermoe.placemoe.runtime.cpu_affinity import (
    CPUCore,
    discover_cpu_cores,
    format_cpu_ids,
    parse_cpu_ids,
    resolve_cpu_affinity,
)


def _dual_numa_smt_cores(physical_cores_per_node: int) -> tuple[CPUCore, ...]:
    physical_cores = physical_cores_per_node * 2
    return tuple(
        CPUCore(
            numa_node=core_id // physical_cores_per_node,
            package_id=core_id // physical_cores_per_node,
            core_id=core_id,
            cpu_ids=(core_id, core_id + physical_cores),
        )
        for core_id in range(physical_cores)
    )


def _visible_cpu_ids(cores: tuple[CPUCore, ...]) -> tuple[int, ...]:
    return tuple(cpu_id for core in cores for cpu_id in core.cpu_ids)


def test_cpu_id_format_round_trip() -> None:
    cpu_ids = (0, 1, 2, 4, 8, 9, 10)

    assert format_cpu_ids(cpu_ids) == "0-2,4,8-10"
    assert parse_cpu_ids(format_cpu_ids(cpu_ids)) == cpu_ids


def test_auto_affinity_balances_numa_and_keeps_smt_siblings_together() -> None:
    cores = _dual_numa_smt_cores(4)
    resources = PlaceMoEPlannerResources(workers=8, candidate_workers=4, worker_threads=1)

    plan = resolve_cpu_affinity(resources, visible_cpu_ids=_visible_cpu_ids(cores), cores=cores)

    assert plan.automatic
    assert plan.planner_physical_cores == 2
    assert plan.planner_cpu_ids == (3, 7, 11, 15)
    assert set(plan.training_cpu_ids).isdisjoint(plan.planner_cpu_ids)
    assert set(plan.training_cpu_ids) | set(plan.planner_cpu_ids) == set(_visible_cpu_ids(cores))
    assert plan.workers == 2
    assert plan.candidate_workers == 1
    assert plan.worker_threads == 1
    for core in cores:
        assert set(core.cpu_ids).issubset(plan.training_cpu_ids) or set(core.cpu_ids).issubset(plan.planner_cpu_ids)


def test_auto_affinity_matches_current_gpu_topology_policy() -> None:
    cores = _dual_numa_smt_cores(32)
    resources = PlaceMoEPlannerResources(workers=48, candidate_workers=4, worker_threads=1)

    plan = resolve_cpu_affinity(resources, visible_cpu_ids=_visible_cpu_ids(cores), cores=cores)

    assert plan.planner_physical_cores == 16
    assert len(plan.planner_cpu_ids) == 32
    assert len(plan.training_cpu_ids) == 96
    assert plan.workers == 16
    assert plan.candidate_workers == 1
    planner_nodes = [core.numa_node for core in cores if set(core.cpu_ids) & set(plan.planner_cpu_ids)]
    assert planner_nodes.count(0) == planner_nodes.count(1) == 8


def test_explicit_affinity_is_preserved_and_parallelism_is_bounded() -> None:
    cores = _dual_numa_smt_cores(4)
    resources = PlaceMoEPlannerResources(
        workers=8,
        candidate_workers=4,
        worker_threads=1,
        training_cpu_ids="0-5,8-13",
        planner_cpu_ids="6-7,14-15",
    )

    plan = resolve_cpu_affinity(resources, visible_cpu_ids=_visible_cpu_ids(cores), cores=cores)

    assert not plan.automatic
    assert plan.training_cpu_ids == (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)
    assert plan.planner_cpu_ids == (6, 7, 14, 15)
    assert plan.planner_physical_cores == 2
    assert plan.workers == 2
    assert plan.planner_resources().planner_cpu_ids == "6-7,14-15"


def test_explicit_affinity_rejects_split_smt_core() -> None:
    cores = _dual_numa_smt_cores(4)
    resources = PlaceMoEPlannerResources(
        workers=8,
        candidate_workers=4,
        worker_threads=1,
        training_cpu_ids="0-6,8-15",
        planner_cpu_ids="7",
    )

    with pytest.raises(RuntimeError, match="split physical CPU cores"):
        resolve_cpu_affinity(resources, visible_cpu_ids=_visible_cpu_ids(cores), cores=cores)


def test_discover_cpu_cores_uses_visible_sysfs_topology(tmp_path: Path) -> None:
    for cpu_id, core_id, node_id in ((0, 0, 0), (4, 0, 0), (1, 1, 1), (5, 1, 1)):
        cpu_root = tmp_path / f"cpu{cpu_id}"
        topology_root = cpu_root / "topology"
        topology_root.mkdir(parents=True)
        (topology_root / "physical_package_id").write_text(str(node_id), encoding="utf-8")
        (topology_root / "core_id").write_text(str(core_id), encoding="utf-8")
        (cpu_root / f"node{node_id}").mkdir()

    cores = discover_cpu_cores((0, 1, 4, 5), sysfs_root=tmp_path)

    assert cores == (
        CPUCore(numa_node=0, package_id=0, core_id=0, cpu_ids=(0, 4)),
        CPUCore(numa_node=1, package_id=1, core_id=1, cpu_ids=(1, 5)),
    )
