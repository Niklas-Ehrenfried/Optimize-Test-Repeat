import pytest
import torch
from typing import Tuple, Dict, Any

class AsynchronousDataBatcher:
    """Simulates pipeline worker sequences preparing incoming multi-dimensional tensors."""
    def __init__(self, shape_tuple: Tuple[int, ...]) -> None:
        # Allocate tensors in host memory using pinned allocations for fast PCI-E transfer paths if CUDA is available
        self.host_tensor: torch.Tensor = torch.randn(shape_tuple, pin_memory=torch.cuda.is_available())
        
    def stage_to_device(self) -> torch.Tensor:
        # Transfer the payload to the GPU target device asynchronously if available
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        device_tensor: torch.Tensor = self.host_tensor.to(device, non_blocking=True)
        
        # Apply standard min-max range normalizations to confirm device synchronization
        min_val: torch.Tensor = device_tensor.min()
        max_val: torch.Tensor = device_tensor.max()
        normalized: torch.Tensor = (device_tensor - min_val) / max(max_val - min_val, 1e-5)
        return normalized

@pytest.mark.parametrize("profile_name", ["Medium", "Large", "Ultra"])
def test_host_to_device_streaming_throughput(otr_bench: Any, profile_name: str) -> None:
    """
    Profiles processing tasks outside standard network layers to isolate 
    data loading transfer boundaries and monitor host allocation overhead.
    """
    def inputs_factory(shapes: Dict[str, Any]) -> Tuple[AsynchronousDataBatcher, Tuple[()]]:
        # Construct dense multidimensional shapes for computer vision or audio inputs
        # e.g., mapping Batch Size x Sequence Channels x Frame Features
        large_feature_dimensions: Tuple[int, ...] = (shapes["B"] * 16, shapes["C"], shapes["T"])
        batcher = AsynchronousDataBatcher(large_feature_dimensions)
        return (batcher, ())

    metrics: Dict[str, Any] = otr_bench(
        component_id="Pipeline Asynchronous H2D Streaming",
        profile_input=profile_name,
        func=lambda batcher, _: batcher.stage_to_device(),
        inputs_factory=inputs_factory
    )
    
    if torch.cuda.is_available():
        # This utility highlights memory transfer bottlenecks directly on your Rich console panel
        assert metrics["memory_transfer_pct"] > 0.0
    else:
        assert metrics["latency_ms"] > 0.0
