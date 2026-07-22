# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from typing import Any


_HIERMOE_METRIC_RECORDS: list[dict[str, Any]] = []


def record_hiermoe_metrics(record: dict[str, Any]) -> None:
    _HIERMOE_METRIC_RECORDS.append(record)


def _aggregate_hiermoe_metrics(records: list[dict[str, Any]]) -> dict[str, float | int | str]:
    if not records:
        return {}

    totals: dict[str, float] = defaultdict(float)
    averages: dict[str, list[float]] = defaultdict(list)
    latest: dict[str, Any] = {}

    for record in records:
        for key in ("dispatch_wall_ms", "combine_wall_ms", "local_expert_compute_wall_ms"):
            totals[key] += float(record.get(key, 0.0))
        if "baseline_original_all_to_all_ms" in record:
            totals["baseline_original_all_to_all_ms"] += float(record["baseline_original_all_to_all_ms"])
        for key in ("dedup_ratio_dispatch", "dedup_ratio_combine", "selected_dim"):
            if key in record:
                averages[key].append(float(record[key]))
        for key in (
            "enable",
            "expert_swap_pair",
            "expert_swap_interval",
            "expert_swap_max_pairs_per_layer",
            "perf_model_source",
        ):
            if key in record:
                latest[key] = record[key]

    payload: dict[str, float | int | str] = {
        "hiermoe/enable": int(bool(latest.get("enable", True))),
        "hiermoe/dispatch_wall_ms": totals["dispatch_wall_ms"],
        "hiermoe/combine_wall_ms": totals["combine_wall_ms"],
        "hiermoe/local_expert_compute_wall_ms": totals["local_expert_compute_wall_ms"],
        "hiermoe/expert_swap_interval": int(latest.get("expert_swap_interval", 0)),
        "hiermoe/expert_swap_max_pairs_per_layer": int(latest.get("expert_swap_max_pairs_per_layer", 1)),
        "hiermoe/expert_swap_pair": str(latest.get("expert_swap_pair", "not_implemented")),
        "hiermoe/perf_model_source": str(latest.get("perf_model_source", "unknown")),
    }
    if "baseline_original_all_to_all_ms" in totals:
        payload["baseline/original_all_to_all_ms"] = totals["baseline_original_all_to_all_ms"]
    if averages["selected_dim"]:
        payload["hiermoe/selected_dim"] = int(round(sum(averages["selected_dim"]) / len(averages["selected_dim"])))
    if averages["dedup_ratio_dispatch"]:
        payload["hiermoe/dedup_ratio_dispatch"] = sum(averages["dedup_ratio_dispatch"]) / len(
            averages["dedup_ratio_dispatch"]
        )
    if averages["dedup_ratio_combine"]:
        payload["hiermoe/dedup_ratio_combine"] = sum(averages["dedup_ratio_combine"]) / len(
            averages["dedup_ratio_combine"]
        )
    return payload


def peek_hiermoe_metrics() -> dict[str, float | int | str]:
    return _aggregate_hiermoe_metrics(_HIERMOE_METRIC_RECORDS)


def flush_hiermoe_metrics() -> dict[str, float | int | str]:
    global _HIERMOE_METRIC_RECORDS
    records = _HIERMOE_METRIC_RECORDS
    _HIERMOE_METRIC_RECORDS = []
    return _aggregate_hiermoe_metrics(records)
