"""Pure scheduling state machine for independent PlaceMoE refreshes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UpdateKind(str, Enum):
    FULL = "full"
    MAPPING_ONLY = "mapping"


@dataclass
class HotUpdateScheduler:
    """Coalesce hot-update events while allowing at most one planner job."""

    layout_interval_steps: int
    mapping_interval_steps: int
    last_update_step: int
    pending_full: bool = False
    pending_mapping: bool = False

    def observe_step(self, training_step: int) -> None:
        within_window = 0 < training_step <= self.last_update_step
        layout_due = bool(
            within_window and self.layout_interval_steps > 0 and training_step % self.layout_interval_steps == 0
        )
        mapping_due = bool(
            within_window and self.mapping_interval_steps > 0 and training_step % self.mapping_interval_steps == 0
        )
        if layout_due:
            self.pending_full = True
            self.pending_mapping = False
        elif mapping_due:
            self.pending_mapping = True

    def pop_next(self) -> UpdateKind | None:
        if self.pending_full:
            self.pending_full = False
            self.pending_mapping = False
            return UpdateKind.FULL
        if self.pending_mapping:
            self.pending_mapping = False
            return UpdateKind.MAPPING_ONLY
        return None


__all__ = ["HotUpdateScheduler", "UpdateKind"]
