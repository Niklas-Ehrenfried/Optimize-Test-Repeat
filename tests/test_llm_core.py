import pytest
import torch
from typing import Tuple, Dict, Any, Optional

class NaiveKVCache:
    """Manages appending new key/value tokens to an active sequence context."""
    def __init__(self, num_heads: int = 8, head_dim: int = 64) -> None:
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # new_k/new_v shapes: (B, 1, num_heads, head_dim)
        if self.k_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            # Re-allocates memory and concatenates along the sequence length (dim=1)
            self.k_cache = torch.cat([self.k_cache, new_k], dim=1)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=1)
        return self.k_cache, self.v_cache

@pytest.mark.parametrize("profile_name", ["Small", "Medium", "Large"])
def test_kv_cache_append_efficiency(otr_bench: Any, profile_name: str) -> None:
    """
    Profiles the memory overhead and transfer characteristics of appending 
    new token states to an existing autoregressive KV cache history.
    """
    def inputs_factory(shapes: Dict[str, Any]) -> Tuple[NaiveKVCache, Tuple[torch.Tensor, torch.Tensor]]:
        # Resolve dimensions based on global profile configurations
        B: int = shapes["B"]
        num_heads: int = shapes["num_exp"] # Borrowing scaling indices for hyperparameter layout
        head_dim: int = shapes["C"]
        
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
        cache_manager = NaiveKVCache(num_heads=num_heads, head_dim=head_dim)
        
        # Pre-populate the cache with a history sequence length of 128 tokens
        init_k: torch.Tensor = torch.randn(B, 128, num_heads, head_dim, device=device)
        init_v: torch.Tensor = torch.randn(B, 128, num_heads, head_dim, device=device)
        cache_manager.update(init_k, init_v)
        
        # New incoming query projection token payload to append
        new_k: torch.Tensor = torch.randn(B, 1, num_heads, head_dim, device=device)
        new_v: torch.Tensor = torch.randn(B, 1, num_heads, head_dim, device=device)
        
        return (cache_manager, (new_k, new_v))

    metrics: Dict[str, Any] = otr_bench(
        component_id="LLM Inference KV-Cache Splicing",
        profile_input=profile_name,
        func=lambda cache, tokens: cache.update(*tokens),
        inputs_factory=inputs_factory
    )
    
    assert metrics["latency_ms"] > 0.0
