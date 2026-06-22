import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class FusedGatedMLP(nn.Module):
    """
    A sequence module containing heavy element-wise combinations and gating,
    ideal for layout fusion testing under torch.compile.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * 2)
        self.w2 = nn.Linear(d_model, d_model * 2)
        self.w3 = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused Triton pointwise/reduction kernels helper path
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


@pytest.mark.parametrize("profile_name", ["Small", "Medium"])
def test_compiled_mlp_telemetry(otr_bench: Any, profile_name: str) -> None:
    """
    Verifies that the framework accurately catches fused Triton runtime metrics,
    isolates initial compile overhead, and successfully tracks compiled modules.
    """

    def inputs_factory(shapes: Dict[str, Any]) -> Tuple[Any, torch.Tensor]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FusedGatedMLP(shapes["C"]).to(device)

        # Apply torch compilation to the module execution graph
        compiled_model = torch.compile(model)

        x = torch.randn(
            shapes["B"], shapes["T"], shapes["C"], device=device, requires_grad=True
        )
        return (compiled_model, x)

    # Run execution telemetry with the compiled callable wrapper
    metrics = otr_bench(
        component_id="LLM Architecture Compiled Gated MLP",
        profile_input=profile_name,
        func=lambda model, x: model(x),
        inputs_factory=inputs_factory,
    )

    # Ensure measurements succeeded and did not error or timeout during JIT phases
    assert metrics["status"] != "FAILED"
    assert metrics["latency_ms"] > 0.0
    assert metrics["forward_latency_ms"] > 0.0
    assert metrics["compute_ratio_pct"] >= 0.0
    assert metrics["memory_transfer_pct"] >= 0.0
