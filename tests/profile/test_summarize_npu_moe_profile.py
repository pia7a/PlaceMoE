import importlib.util
import json
from pathlib import Path

import pytest


def _load_summarizer_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "profile" / "scripts" / "summarize_npu_moe_profile.py"
    spec = importlib.util.spec_from_file_location("summarize_npu_moe_profile", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row) + "\n")


def test_npu_moe_summarizer_normalizes_baseline_hiermoe_sections(tmp_path):
    module = _load_summarizer_module()
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "full_timing" / "step_timing_rank0.jsonl",
        [
            {
                "record_type": "step",
                "section": "train_step_total",
                "step": 3,
                "rank": 0,
                "cuda_ms": 100.0,
                "wall_ms": 101.0,
            },
            {
                "record_type": "step",
                "section": "hiermoe_expert_swap",
                "step": 3,
                "rank": 0,
                "cuda_ms": 7.0,
                "wall_ms": 8.0,
            },
        ],
    )
    _write_jsonl(
        run_dir / "moe_timing" / "moe_timing_rank0.jsonl",
        [
            {
                "step": 3,
                "rank": 0,
                "span_layers_by_phase": [
                    {
                        "phase": "forward",
                        "direction": "forward",
                        "component": "all_to_all",
                        "section": "pre_all_to_all",
                        "layer": 0,
                        "cuda_ms_sum": 11.0,
                        "calls": 1,
                        "tokens": 16,
                        "token_expert_assignments": 32,
                    },
                    {
                        "phase": "forward",
                        "direction": "forward",
                        "component": "all_to_all",
                        "section": "hiermoe_post_all_to_all",
                        "layer": 0,
                        "cuda_ms_sum": 13.0,
                        "calls": 1,
                        "tokens": 16,
                        "token_expert_assignments": 32,
                    },
                    {
                        "phase": "forward",
                        "direction": "forward",
                        "component": "expert_compute",
                        "section": "hiermoe_expert_compute",
                        "layer": 0,
                        "cuda_ms_sum": 17.0,
                        "calls": 1,
                        "tokens": 16,
                        "token_expert_assignments": 32,
                    },
                ],
            }
        ],
    )

    profile = module.load_profile(run_dir)
    rows = {(row["backend_path"], row["logical_section"]): row for row in profile["logical_section"]}

    assert rows[("baseline", "dispatch")]["cuda_ms_sum"] == 11.0
    assert rows[("hiermoe", "combine")]["cuda_ms_sum"] == 13.0
    assert rows[("hiermoe", "expert_compute")]["cuda_ms_sum"] == 17.0
    assert rows[("hiermoe", "expert_swap")]["cuda_ms_sum"] == 7.0


def test_npu_moe_summarizer_filters_historical_rank_and_steps(tmp_path):
    module = _load_summarizer_module()
    run_dir = tmp_path / "run"
    rows = []
    moe_rows = []
    for rank in (0, 8):
        for step in range(1, 7):
            rows.append(
                {
                    "record_type": "step",
                    "section": "train_step_total",
                    "step": step,
                    "rank": rank,
                    "cuda_ms": float(rank + step),
                    "wall_ms": float(rank + step + 1),
                }
            )
            moe_rows.append(
                {
                    "step": step,
                    "rank": rank,
                    "span_layers_by_phase": [
                        {
                            "phase": "forward",
                            "direction": "forward",
                            "component": "all_to_all",
                            "section": "hiermoe_pre_all_to_all",
                            "layer": 0,
                            "cuda_ms_sum": float(rank + step),
                            "calls": 1,
                        }
                    ],
                }
            )
    _write_jsonl(run_dir / "full_timing" / "step_timing_rank0.jsonl", rows)
    _write_jsonl(run_dir / "moe_timing" / "moe_timing_rank0.jsonl", moe_rows)

    profile = module.load_profile(
        run_dir,
        measurement_rank=0,
        start_step=3,
        end_step=6,
        require_train_steps=4,
    )

    assert profile["train_step_total_ms"] == 18.0
    assert profile["sampled_component_totals"]["all_to_all"] == 18.0
    assert profile["sampled_pairs"] == {("3", "0"), ("4", "0"), ("5", "0"), ("6", "0")}
    assert profile["measurement"] == {
        "rank": 0,
        "start_step": 3,
        "end_step": 6,
        "train_step_records": 4,
    }


def test_npu_moe_summarizer_rejects_incomplete_or_duplicate_historical_steps(tmp_path):
    module = _load_summarizer_module()
    run_dir = tmp_path / "run"
    rows = [
        {
            "record_type": "step",
            "section": "train_step_total",
            "step": step,
            "rank": 0,
            "cuda_ms": 1.0,
        }
        for step in (3, 4, 5)
    ]
    _write_jsonl(run_dir / "full_timing" / "step_timing_rank0.jsonl", rows)

    with pytest.raises(ValueError, match="Expected 4"):
        module.load_profile(
            run_dir,
            measurement_rank=0,
            start_step=3,
            end_step=6,
            require_train_steps=4,
        )

    rows.append(dict(rows[-1]))
    _write_jsonl(run_dir / "full_timing" / "step_timing_rank0.jsonl", rows)
    with pytest.raises(ValueError, match="Duplicate train_step_total"):
        module.load_profile(
            run_dir,
            measurement_rank=0,
            start_step=3,
            end_step=6,
        )
