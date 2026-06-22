import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class SimpleLinearProj(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.mark.parametrize("profile_name", ["Small"])
def test_refinements_telemetry(otr_bench: Any, profile_name: str) -> None:
    """
    Verifies that the framework automatically detects outputs requiring gradients,
    runs the backward pass, and populates backward latencies and VRAM metrics.
    """

    def inputs_factory(shapes: Dict[str, Any]) -> Tuple[nn.Module, torch.Tensor]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleLinearProj(shapes["C"], shapes["C"]).to(device)
        x = torch.randn(
            shapes["B"], shapes["T"], shapes["C"], device=device, requires_grad=True
        )
        return (model, x)

    # Run with auto-detection of backward pass
    metrics = otr_bench(
        component_id="Refinements Unified Telemetry",
        profile_input=profile_name,
        func=lambda model, x: model(x),
        inputs_factory=inputs_factory,
    )

    assert metrics["status"] in [
        "GEN_0_INITIALIZED",
        "PROMOTED",
        "REJECTED",
        "IMPROVED_INTRA",
    ]
    assert metrics["requires_backward"] is True
    assert metrics["forward_latency_ms"] > 0.0
    assert metrics["latency_ms"] >= metrics["forward_latency_ms"]


@pytest.mark.parametrize("profile_name", ["Small"])
def test_crash_containment(otr_bench: Any, profile_name: str) -> None:
    """
    Verifies that exceptions raised during execution are caught and returned
    as a FAILED status in metrics.
    """

    def inputs_factory(shapes: Dict[str, Any]) -> Tuple[None, None]:
        return (None, None)

    def failing_func(a: Any, b: Any) -> None:
        raise ValueError("Simulated OOM or Shape Mismatch")

    metrics = otr_bench(
        component_id="Refinements Unified Telemetry",
        profile_input=profile_name,
        func=failing_func,
        inputs_factory=inputs_factory,
    )

    assert metrics["status"] == "FAILED"
    assert metrics["latency_ms"] == -1.0
    assert "ValueError" in metrics["error"]


def test_test_only_and_purging() -> None:
    from storage import OTRStateManager
    import json

    state_mgr = OTRStateManager()
    # Reset pending promotions
    OTRStateManager.pending_promotions = []

    metrics = {
        "latency_ms": 0.05,
        "forward_latency_ms": 0.05,
        "backward_latency_ms": 0.0,
        "requires_backward": False,
        "vram_base_mb": 1.0,
        "vram_transient_mb": 2.0,
        "compute_ratio_pct": 90.0,
        "memory_transfer_pct": 10.0,
        "status": "SUCCESS",
    }

    status = state_mgr.evaluate_and_log(
        component="Test Component Purging",
        profile_size="Small",
        run_name="test_only_run",
        notes="test run only",
        metrics=metrics,
        test_only=True,
    )

    # Verify that in test_only mode, it is NOT added to pending promotions
    assert len(OTRStateManager.pending_promotions) == 0

    # Verify purging: load workspace and ensure all profiles of that component are purged
    workspace_data = state_mgr._load(state_mgr.workspace_path, list)
    # Add a stale entry for "Small" with generation_context "GEN_0"
    workspace_data.append(
        {
            "component": "Test Component Purging",
            "profile_size": "Small",
            "run_name": "stale_run_small",
            "notes": "stale notes",
            "generation_context": "GEN_0",
            "metrics": metrics,
        }
    )
    # Add another entry for "Medium" with generation_context "GEN_0" (which belongs to same component)
    workspace_data.append(
        {
            "component": "Test Component Purging",
            "profile_size": "Medium",
            "run_name": "run_medium",
            "notes": "medium notes",
            "generation_context": "GEN_0",
            "metrics": metrics,
        }
    )
    state_mgr._save(state_mgr.workspace_path, workspace_data)

    # Now set the g_best to have GEN_1 for "Small" profile only
    g_best = state_mgr._load(state_mgr.best_path, dict)
    g_best["Test Component Purging::Small"] = {
        "latency_ms": 0.04,
        "generation": "GEN_1",
    }
    g_best["Test Component Purging::Medium"] = {
        "latency_ms": 0.04,
        "generation": "GEN_0",
    }
    state_mgr._save(state_mgr.best_path, g_best)

    # Evaluate a new run. This should trigger purging of all profiles for "Test Component Purging".
    state_mgr.evaluate_and_log(
        component="Test Component Purging",
        profile_size="Small",
        run_name="new_run",
        notes="new notes",
        metrics=metrics,
        test_only=True,
    )

    # Load workspace and verify that BOTH the stale small run and the medium run are gone
    final_workspace = state_mgr._load(state_mgr.workspace_path, list)
    stale_small_found = any(
        t.get("run_name") == "stale_run_small" for t in final_workspace
    )
    medium_found = any(t.get("run_name") == "run_medium" for t in final_workspace)
    assert not stale_small_found
    assert not medium_found

    # Clean up test component from workspace and g_best
    g_best = state_mgr._load(state_mgr.best_path, dict)
    g_best.pop("Test Component Purging::Small", None)
    g_best.pop("Test Component Purging::Medium", None)
    state_mgr._save(state_mgr.best_path, g_best)

    workspace_data = state_mgr._load(state_mgr.workspace_path, list)
    workspace_data = [
        t for t in workspace_data if t.get("component") != "Test Component Purging"
    ]
    state_mgr._save(state_mgr.workspace_path, workspace_data)
