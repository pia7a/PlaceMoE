"""Hardware-neutral CPU isolation for PlaceMoE hot-update planning."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import PlaceMoEPlannerResources


_DEFAULT_PLANNER_CORE_FRACTION = 0.25
_CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")


@dataclass(frozen=True, order=True)
class CPUCore:
    """One visible physical core and its visible hardware threads."""

    numa_node: int
    package_id: int
    core_id: int
    cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class CPUAffinityPlan:
    """Disjoint training and planner CPU masks with bounded parallelism."""

    automatic: bool
    training_cpu_ids: tuple[int, ...]
    planner_cpu_ids: tuple[int, ...]
    planner_physical_cores: int
    workers: int
    candidate_workers: int
    worker_threads: int

    def planner_resources(self) -> PlaceMoEPlannerResources:
        return PlaceMoEPlannerResources(
            workers=self.workers,
            candidate_workers=self.candidate_workers,
            worker_threads=self.worker_threads,
            planner_cpu_ids=format_cpu_ids(self.planner_cpu_ids),
            training_cpu_ids=format_cpu_ids(self.training_cpu_ids),
        )


def parse_cpu_ids(value: str) -> tuple[int, ...]:
    """Parse a Linux CPU list such as ``0-3,8,10-11``."""

    cpu_ids: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            cpu_id = int(part)
            if cpu_id < 0:
                raise ValueError(f"Invalid CPU id {cpu_id}.")
            cpu_ids.add(cpu_id)
            continue
        raw_start, raw_end = part.split("-", maxsplit=1)
        start = int(raw_start)
        end = int(raw_end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid CPU range {part!r}.")
        cpu_ids.update(range(start, end + 1))
    return tuple(sorted(cpu_ids))


def format_cpu_ids(cpu_ids: Iterable[int]) -> str:
    """Format CPU ids using the compact Linux CPU-list syntax."""

    ordered = sorted({int(cpu_id) for cpu_id in cpu_ids})
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu_id in ordered[1:]:
        if cpu_id == previous + 1:
            previous = cpu_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu_id
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _read_int(path: Path, fallback: int) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return int(fallback)


def _numa_node(cpu_root: Path, fallback: int) -> int:
    try:
        nodes = sorted(
            int(path.name.removeprefix("node"))
            for path in cpu_root.glob("node[0-9]*")
            if path.name.removeprefix("node").isdigit()
        )
    except OSError:
        nodes = []
    return nodes[0] if nodes else int(fallback)


def discover_cpu_cores(
    visible_cpu_ids: Iterable[int],
    *,
    sysfs_root: Path = _CPU_SYSFS_ROOT,
) -> tuple[CPUCore, ...]:
    """Group visible logical CPUs by physical core and NUMA node."""

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    nodes: dict[tuple[int, int], int] = {}
    for cpu_id in sorted({int(cpu_id) for cpu_id in visible_cpu_ids}):
        topology_root = sysfs_root / f"cpu{cpu_id}" / "topology"
        package_id = _read_int(topology_root / "physical_package_id", 0)
        core_id = _read_int(topology_root / "core_id", cpu_id)
        key = (package_id, core_id)
        grouped[key].append(cpu_id)
        nodes[key] = _numa_node(sysfs_root / f"cpu{cpu_id}", package_id)
    return tuple(
        sorted(
            CPUCore(
                numa_node=nodes[key],
                package_id=key[0],
                core_id=key[1],
                cpu_ids=tuple(sorted(cpu_ids)),
            )
            for key, cpu_ids in grouped.items()
        )
    )


def _select_balanced_planner_cores(cores: Sequence[CPUCore], count: int) -> tuple[CPUCore, ...]:
    by_node: dict[int, list[CPUCore]] = defaultdict(list)
    for core in sorted(cores):
        by_node[core.numa_node].append(core)
    selected: list[CPUCore] = []
    node_ids = sorted(by_node)
    while len(selected) < count:
        made_progress = False
        for node_id in node_ids:
            if not by_node[node_id]:
                continue
            selected.append(by_node[node_id].pop())
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            break
    return tuple(sorted(selected))


def _effective_parallelism(
    resources: PlaceMoEPlannerResources,
    planner_physical_cores: int,
) -> tuple[int, int, int]:
    worker_threads = min(resources.worker_threads, planner_physical_cores)
    workers = min(resources.workers, max(1, planner_physical_cores // worker_threads))
    candidate_capacity = max(1, planner_physical_cores // (workers * worker_threads))
    candidate_workers = min(resources.candidate_workers, candidate_capacity)
    return workers, candidate_workers, worker_threads


def _build_plan(
    *,
    automatic: bool,
    training_cpu_ids: Iterable[int],
    planner_cpu_ids: Iterable[int],
    cores: Sequence[CPUCore],
    resources: PlaceMoEPlannerResources,
) -> CPUAffinityPlan:
    training = tuple(sorted({int(cpu_id) for cpu_id in training_cpu_ids}))
    planner = tuple(sorted({int(cpu_id) for cpu_id in planner_cpu_ids}))
    if not training or not planner:
        raise RuntimeError("PlaceMoE CPU isolation requires non-empty training and planner CPU masks.")
    overlap = sorted(set(training) & set(planner))
    if overlap:
        raise RuntimeError(f"PlaceMoE training and planner CPU masks overlap: {overlap}.")
    split_cores = [core for core in cores if set(core.cpu_ids) & set(training) and set(core.cpu_ids) & set(planner)]
    if split_cores:
        identities = [(core.package_id, core.core_id) for core in split_cores]
        raise RuntimeError(f"PlaceMoE training and planner masks split physical CPU cores: {identities}.")
    planner_core_count = sum(bool(set(core.cpu_ids) & set(planner)) for core in cores)
    if planner_core_count <= 0:
        raise RuntimeError("PlaceMoE planner CPU mask contains no visible physical core.")
    workers, candidate_workers, worker_threads = _effective_parallelism(resources, planner_core_count)
    return CPUAffinityPlan(
        automatic=automatic,
        training_cpu_ids=training,
        planner_cpu_ids=planner,
        planner_physical_cores=planner_core_count,
        workers=workers,
        candidate_workers=candidate_workers,
        worker_threads=worker_threads,
    )


def resolve_cpu_affinity(
    resources: PlaceMoEPlannerResources,
    *,
    visible_cpu_ids: Iterable[int] | None = None,
    cores: Sequence[CPUCore] | None = None,
) -> CPUAffinityPlan:
    """Resolve explicit masks or build a topology-aware automatic plan."""

    visible = tuple(
        sorted({int(cpu_id) for cpu_id in (os.sched_getaffinity(0) if visible_cpu_ids is None else visible_cpu_ids)})
    )
    if len(visible) < 2:
        raise RuntimeError("PlaceMoE CPU isolation requires at least two visible CPUs.")
    topology = tuple(discover_cpu_cores(visible) if cores is None else cores)
    if len(topology) < 2:
        raise RuntimeError("PlaceMoE CPU isolation requires at least two visible physical cores.")

    if resources.planner_cpu_ids or resources.training_cpu_ids:
        if not resources.planner_cpu_ids or not resources.training_cpu_ids:
            raise RuntimeError("planner_cpu_ids and training_cpu_ids must be configured together.")
        planner = parse_cpu_ids(resources.planner_cpu_ids)
        training = parse_cpu_ids(resources.training_cpu_ids)
        unavailable = sorted((set(planner) | set(training)) - set(visible))
        if unavailable:
            raise RuntimeError(f"PlaceMoE configured CPUs are not visible: {unavailable}.")
        return _build_plan(
            automatic=False,
            training_cpu_ids=training,
            planner_cpu_ids=planner,
            cores=topology,
            resources=resources,
        )

    requested_parallelism = resources.workers * resources.candidate_workers * resources.worker_threads
    planner_core_count = min(
        len(topology) - 1,
        max(1, int(len(topology) * _DEFAULT_PLANNER_CORE_FRACTION)),
        requested_parallelism,
    )
    planner_cores = _select_balanced_planner_cores(topology, planner_core_count)
    planner_core_keys = {(core.package_id, core.core_id) for core in planner_cores}
    training_cores = [core for core in topology if (core.package_id, core.core_id) not in planner_core_keys]
    return _build_plan(
        automatic=True,
        training_cpu_ids=(cpu_id for core in training_cores for cpu_id in core.cpu_ids),
        planner_cpu_ids=(cpu_id for core in planner_cores for cpu_id in core.cpu_ids),
        cores=topology,
        resources=resources,
    )


__all__ = [
    "CPUAffinityPlan",
    "CPUCore",
    "discover_cpu_cores",
    "format_cpu_ids",
    "parse_cpu_ids",
    "resolve_cpu_affinity",
]
