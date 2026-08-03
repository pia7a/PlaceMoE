"""Lifecycle controller for asynchronous PlaceMoE hot updates."""

from __future__ import annotations

from dataclasses import dataclass, field

from .planner_process import HotUpdateJob
from .scheduler import HotUpdateScheduler, UpdateKind


@dataclass
class HotUpdateController:
    """Own scheduling and single-job lifecycle independently of training code."""

    layout_interval_steps: int
    mapping_interval_steps: int
    last_update_step: int
    failure_policy: str = "continue"
    active_job: HotUpdateJob | None = None
    scheduler: HotUpdateScheduler = field(init=False)

    def __post_init__(self) -> None:
        if self.failure_policy not in {"continue", "raise"}:
            raise ValueError("failure_policy must be 'continue' or 'raise'.")
        self.scheduler = HotUpdateScheduler(
            layout_interval_steps=self.layout_interval_steps,
            mapping_interval_steps=self.mapping_interval_steps,
            last_update_step=self.last_update_step,
        )

    def observe_step(self, training_step: int) -> None:
        self.scheduler.observe_step(training_step)

    def next_update(self) -> UpdateKind | None:
        if self.active_job is not None:
            return None
        return self.scheduler.pop_next()

    def start(self, job: HotUpdateJob) -> None:
        if self.active_job is not None:
            raise RuntimeError("a PlaceMoE planner job is already active.")
        self.active_job = job

    def finish(self) -> HotUpdateJob | None:
        job = self.active_job
        self.active_job = None
        return job


__all__ = ["HotUpdateController"]
