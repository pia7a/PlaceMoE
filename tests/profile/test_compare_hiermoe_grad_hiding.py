import importlib.util
from pathlib import Path

import pytest


def _load_comparator_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "profile" / "compare_hiermoe_grad_hiding.py"
    spec = importlib.util.spec_from_file_location("compare_hiermoe_grad_hiding", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary(mode: str, layout_sha256: str = "layout") -> dict:
    def metric(mean: float) -> dict:
        return {"count": 10, "mean": mean}

    return {
        "run_name": f"ours_{mode}",
        "steady_steps": [11, 20],
        "observed_moe_ranks": 32,
        "e2e_source": "full_timing",
        "hiermoe_ablation_grad_mode": mode,
        "layout_sha256": layout_sha256,
        "e2e_step_ms": metric(10.0 if mode == "hidden" else 12.0),
        "forward_a2a_ms": metric(2.0),
        "backward_a2a_ms": metric(1.0),
        "moe_communication_region_ms": metric(4.0),
        "expert_compute_ms": metric(3.0),
        "tokens_per_second_millions": metric(5.0 if mode == "hidden" else 4.0),
    }


def test_compare_grad_hiding_validates_pair_and_reports_total_a2a():
    module = _load_comparator_module()
    report = module.compare(
        _summary("hidden"),
        _summary("blocking"),
        Path("hidden.json"),
        Path("blocking.json"),
    )

    assert report["pair_validated"] is True
    assert report["time_metrics"][0]["blocking_vs_hidden_speedup"] == pytest.approx(10 / 12)
    total_a2a = next(row for row in report["time_metrics"] if row["metric"] == "total_a2a_ms")
    assert total_a2a["hidden_mean"] == 3.0
    assert total_a2a["blocking_mean"] == 3.0
    assert report["throughput"]["blocking_over_hidden"] == pytest.approx(0.8)


def test_compare_grad_hiding_rejects_different_layouts():
    module = _load_comparator_module()

    with pytest.raises(ValueError, match="different static layouts"):
        module.compare(
            _summary("hidden", "layout-a"),
            _summary("blocking", "layout-b"),
            Path("hidden.json"),
            Path("blocking.json"),
        )
