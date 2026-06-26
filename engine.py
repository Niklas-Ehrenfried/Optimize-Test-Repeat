"""
OTR Profiler Engine Module.

Provides deep execution profiling capabilities, including latency measurement,
transient VRAM tracking, and low-level hardware operation bounds profiles. Supports
both native Python execution paths and PyTorch/CUDA-driven deep learning models.
"""

import time
import gc
import sys
from typing import List, Tuple, Union, Callable, Any, Set, Optional, Dict

# Lazy / deferred import of torch to prevent immediate import errors on CPU/non-ML machines
torch = sys.modules.get("torch", None)


class OTRProfilerEngine:
    """
    Core engine responsible for running benchmarks, capturing execution times, VRAM usage,
    and analyzing PyTorch profile activity to estimate hardware limitations.
    """

    @staticmethod
    def _get_device_time(op: Any) -> float:
        """
        Extracts device execution time from a PyTorch profiler key average operation object.
        Supports multiple fallback attribute names representing device/CUDA execution time.

        Args:
            op: The key average operation object from PyTorch's profile.

        Returns:
            float: Device/CUDA execution time in milliseconds.
        """
        if torch is None:
            return 0.0
        for attr in [
            "device_time_total",
            "cuda_time_total",
            "device_time",
            "cuda_time",
        ]:
            val = getattr(op, attr, None)
            if val is not None:
                return float(val)
        return 0.0

    @staticmethod
    def _find_tensors_requiring_grad(x: Any) -> List[Any]:
        """
        Recursively searches an arbitrary nested data structure (lists, tuples, dicts)
        for any PyTorch tensors that require gradients.

        Args:
            x: Any python object/container to search.

        Returns:
            list: A list of PyTorch.Tensor objects that have `requires_grad=True`.
        """
        if torch is None:
            return []
        tensors = []
        if isinstance(x, torch.Tensor):
            if x.requires_grad:
                tensors.append(x)
        elif isinstance(x, (list, tuple)):
            for item in x:
                tensors.extend(OTRProfilerEngine._find_tensors_requiring_grad(item))
        elif isinstance(x, dict):
            for item in x.values():
                tensors.extend(OTRProfilerEngine._find_tensors_requiring_grad(item))
        return tensors

    @staticmethod
    def _fwd_bwd_wrapper(
        f: Callable[..., Any], inp: Union[Tuple[Any, ...], List[Any]]
    ) -> None:
        """
        Executes both the forward and backward passes of a function. Computes a dummy
        scalar loss by summing all gradient-requiring outputs and backpropagating.

        Args:
            f (callable): The function to execute.
            inp (tuple/list): The inputs to pass to the function.
        """
        out = f(*inp)
        grad_tensors = OTRProfilerEngine._find_tensors_requiring_grad(out)
        if grad_tensors:
            loss = sum(t.sum() for t in grad_tensors)
            loss.backward()

    @staticmethod
    def _has_tensors(x: Any, visited: Optional[Set[int]] = None) -> bool:
        """
        Recursively scans an object structure for PyTorch tensors. Employs a cycle detection
        guard (visited set) to safely scan complex custom dataset wrappers.

        Args:
            x: Object to scan.
            visited (set, optional): Set of object IDs visited to prevent infinite loops.

        Returns:
            bool: True if at least one PyTorch tensor is found, False otherwise.
        """
        if torch is None:
            return False
        if visited is None:
            visited = set()

        obj_id = id(x)
        if obj_id in visited:
            return False
        visited.add(obj_id)

        if isinstance(x, torch.Tensor):
            return True
        if isinstance(x, (list, tuple)):
            return any(OTRProfilerEngine._has_tensors(item, visited) for item in x)
        if isinstance(x, dict):
            return any(
                OTRProfilerEngine._has_tensors(item, visited) for item in x.values()
            )
        if hasattr(x, "__dict__"):
            return any(
                OTRProfilerEngine._has_tensors(val, visited)
                for val in x.__dict__.values()
            )
        return False

    @staticmethod
    def run_telemetry(
        func: Callable[..., Any],
        inputs: Union[Tuple[Any, ...], List[Any]],
        num_iter: int = 10,
        require_backward: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Profiles the target function over multiple runs to collect high-fidelity telemetry.

        Correctly distinguishes between ML (PyTorch/CUDA) tasks and standard software algorithms:
        - For standard algorithms: profiles cleanly using time.perf_counter without PyTorch overhead.
        - For ML models: profiles forward/backward passes, records GPU transient VRAM usage, and
          uses PyTorch's execution profiler to estimate compute ratio vs memory transfer bottlenecks.
        """
        global torch
        if torch is None:
            try:
                import torch  # Lazy fallback load if torch exists in python path
            except ImportError:
                torch = None

        try:
            metrics = {}

            # --- DEFENSE: Auto-detect Non-ML/Standard Python Functions ---
            has_tensors = OTRProfilerEngine._has_tensors(inputs) if torch else False

            if not has_tensors:
                # Optimized time profiling loop without PyTorch/CUDA overhead
                gc.collect()
                t0 = time.perf_counter()
                for _ in range(num_iter * 2):
                    _ = func(*inputs)
                latency = ((time.perf_counter() - t0) / (num_iter * 2)) * 1000.0

                return {
                    "latency_ms": latency,
                    "forward_latency_ms": latency,
                    "backward_latency_ms": 0.0,
                    "vram_base_mb": 0.0,
                    "vram_transient_mb": 0.0,
                    "vram_transient_forward_mb": 0.0,
                    "vram_transient_backward_mb": 0.0,
                    "compute_ratio_pct": 100.0,
                    "memory_transfer_pct": 0.0,
                    "gpu_idle_overhead_pct": 0.0,
                    "requires_backward": False,
                    "status": "SUCCESS",
                }

            # --- Deep Learning Architecture Telemetry Flow ---
            from torch.profiler import profile, ProfilerActivity, record_function
            import torch.utils.benchmark as benchmark

            # Initial lightweight pass to safely determine backward graph requirement
            first_out = func(*inputs)
            requires_bwd = (
                require_backward
                if require_backward is not None
                else len(OTRProfilerEngine._find_tensors_requiring_grad(first_out)) > 0
            )

            # ENHANCED WARMUP: Complete a sequence of passes to guarantee full JIT / Triton kernel compilation
            is_compiled = (
                hasattr(func, "_compiled_callable")
                or "OptimizedModule" in type(func).__name__
            )
            warmup_iters = 3 if is_compiled else 1

            for _ in range(warmup_iters):
                if requires_bwd:
                    OTRProfilerEngine._fwd_bwd_wrapper(func, inputs)
                else:
                    _ = func(*inputs)

            # Clean cache and sync before profiling base memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                vram_base = torch.cuda.memory_allocated() / (1024**2)
            else:
                vram_base = 0.0

            # Forward pass timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iter):
                _ = func(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_latency_ms = ((time.perf_counter() - t0) / num_iter) * 1000.0

            if not requires_bwd:
                latency_ms = forward_latency_ms
                backward_latency_ms = 0.0
            else:
                # Combined forward and backward pass timer
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_iter):
                    OTRProfilerEngine._fwd_bwd_wrapper(func, inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_ms = ((time.perf_counter() - t0) / num_iter) * 1000.0
                backward_latency_ms = max(latency_ms - forward_latency_ms, 0.0)

            metrics.update(
                {
                    "latency_ms": latency_ms,
                    "forward_latency_ms": forward_latency_ms,
                    "backward_latency_ms": backward_latency_ms,
                    "requires_backward": requires_bwd,
                }
            )

            # Peak VRAM usage profiling
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                out = func(*inputs)
                torch.cuda.synchronize()
                vram_transient_forward_mb = max(
                    (torch.cuda.max_memory_allocated() / (1024**2)) - vram_base, 0.0
                )

                if requires_bwd:
                    torch.cuda.reset_peak_memory_stats()
                    grad_tensors = OTRProfilerEngine._find_tensors_requiring_grad(out)
                    if grad_tensors:
                        sum(t.sum() for t in grad_tensors).backward()
                    torch.cuda.synchronize()
                    vram_transient_backward_mb = max(
                        (torch.cuda.max_memory_allocated() / (1024**2)) - vram_base, 0.0
                    )
                else:
                    vram_transient_backward_mb = 0.0

                metrics.update(
                    {
                        "vram_base_mb": vram_base,
                        "vram_transient_forward_mb": vram_transient_forward_mb,
                        "vram_transient_backward_mb": vram_transient_backward_mb,
                        "vram_transient_mb": max(
                            vram_transient_forward_mb, vram_transient_backward_mb
                        ),
                    }
                )
            else:
                metrics.update(
                    {
                        "vram_base_mb": 0.0,
                        "vram_transient_forward_mb": 0.0,
                        "vram_transient_backward_mb": 0.0,
                        "vram_transient_mb": 0.0,
                    }
                )

            # Low-Level PyTorch Operations Profiling (CPU / GPU bound ratio checks)
            # Detect ROCm / HIP to avoid ProfilerActivity.CUDA hangs in key_averages()
            is_hip = getattr(torch, "version", None) is not None and getattr(torch.version, "hip", None) is not None
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available() and not is_hip:
                activities.append(ProfilerActivity.CUDA)

            with profile(activities=activities, record_shapes=False) as prof:
                with record_function("otr_target"):
                    profiler_iters = min(2, num_iter)
                    for _ in range(profiler_iters):
                        if requires_bwd:
                            OTRProfilerEngine._fwd_bwd_wrapper(func, inputs)
                        else:
                            func(*inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            averages = prof.key_averages()
            if torch.cuda.is_available() and not is_hip:
                total_device_time = sum(
                    OTRProfilerEngine._get_device_time(op) for op in averages
                )
                total_cpu_time = sum(op.cpu_time_total for op in averages)
                mem_sigs = [
                    "copy_",
                    "memcpy",
                    "contiguous",
                    "slice",
                    "gather",
                    "index",
                    "scatter",
                    "triton_poi",
                    "triton_red",
                    "extern_copy",
                ]
                transfer_time = sum(
                    OTRProfilerEngine._get_device_time(op)
                    for op in averages
                    if any(m in op.key.lower() for m in mem_sigs)
                )
                runtime_stalls = sum(
                    op.cpu_time_total
                    for op in averages
                    if "cudaLaunchKernel" in op.key or "cudaSync" in op.key
                )

                metrics["compute_ratio_pct"] = (
                    max(total_device_time - transfer_time, 0.0)
                    / max(total_device_time, 1e-5)
                ) * 100.0
                metrics["memory_transfer_pct"] = (
                    transfer_time / max(total_device_time, 1e-5)
                ) * 100.0
                metrics["gpu_idle_overhead_pct"] = (
                    runtime_stalls / max(total_cpu_time, 1e-5)
                ) * 100.0
            else:
                total_cpu_time = sum(op.cpu_time_total for op in averages)
                mem_sigs = [
                    "copy_",
                    "contiguous",
                    "slice",
                    "gather",
                    "index",
                    "scatter",
                    "triton_poi",
                    "triton_red",
                    "extern_copy",
                ]
                transfer_time = sum(
                    op.cpu_time_total
                    for op in averages
                    if any(m in op.key.lower() for m in mem_sigs)
                )
                metrics["compute_ratio_pct"] = (
                    max(total_cpu_time - transfer_time, 0.0) / max(total_cpu_time, 1e-5)
                ) * 100.0
                metrics["memory_transfer_pct"] = (
                    transfer_time / max(total_cpu_time, 1e-5)
                ) * 100.0
                metrics["gpu_idle_overhead_pct"] = 0.0

            metrics["status"] = "SUCCESS"

            # Aggressive cleanup of local references to prevent memory accumulation and OOM
            try:
                del first_out
            except NameError:
                pass
            try:
                del out
            except NameError:
                pass
            try:
                del grad_tensors
            except NameError:
                pass
            try:
                del prof
            except NameError:
                pass
            try:
                del averages
            except NameError:
                pass

            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            return metrics

        except Exception as e:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return {
                "latency_ms": -1.0,
                "forward_latency_ms": -1.0,
                "backward_latency_ms": -1.0,
                "vram_base_mb": 0.0,
                "vram_transient_mb": 0.0,
                "vram_transient_forward_mb": 0.0,
                "vram_transient_backward_mb": 0.0,
                "compute_ratio_pct": 0.0,
                "memory_transfer_pct": 0.0,
                "gpu_idle_overhead_pct": 0.0,
                "requires_backward": False,
                "status": "FAILED",
                "error": f"{type(e).__name__}: {str(e)}",
            }
