"""Accelerator timing helpers for CUDA and Ascend NPU."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .device import get_device_type, synchronize


@dataclass
class AcceleratorEvent:
    device_type: str
    event: Any | None
    wall_time: float

    def elapsed_time(self, end_event: "AcceleratorEvent") -> float:
        if self.event is not None and end_event.event is not None:
            try:
                return float(self.event.elapsed_time(end_event.event))
            except Exception:
                pass
        return float((end_event.wall_time - self.wall_time) * 1000.0)


def _event_namespace() -> Any | None:
    device_type = get_device_type()
    if device_type == "cpu":
        return None
    namespace = getattr(torch, device_type, None)
    if namespace is None:
        return None
    is_available = getattr(namespace, "is_available", None)
    if callable(is_available):
        try:
            if not is_available():
                return None
        except Exception:
            return None
    return namespace


def accelerator_timing_available() -> bool:
    return _event_namespace() is not None


def record_accelerator_event() -> AcceleratorEvent | None:
    namespace = _event_namespace()
    if namespace is None:
        return None

    event = None
    event_ctor = getattr(namespace, "Event", None)
    if event_ctor is not None:
        try:
            event = event_ctor(enable_timing=True)
        except TypeError:
            event = event_ctor()
        except Exception:
            event = None
        if event is not None:
            try:
                event.record()
                return AcceleratorEvent(get_device_type(), event, time.perf_counter())
            except Exception:
                event = None

    try:
        synchronize()
    except Exception:
        pass
    return AcceleratorEvent(get_device_type(), event, time.perf_counter())


def synchronize_accelerator() -> None:
    if accelerator_timing_available():
        synchronize()


def cuda_nvtx_available() -> bool:
    return bool(torch.cuda.is_available())
