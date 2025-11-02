"""
Per-Request KV-Cache Manager for Continuous Batching LLM Inference
"""

import torch
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheBlock:
    """Represents a single KV-cache block for one layer"""
    key: torch.Tensor  # Shape: [batch_size, num_heads, seq_len, head_dim]
    value: torch.Tensor  # Shape: [batch_size, num_heads, seq_len, head_dim]
    
    def to(self, device):
        """Move cache block to specified device"""
        self.key = self.key.to(device)
        self.value = self.value.to(device)
        return self


@dataclass
class RequestCache:
    """Manages KV-cache for a single request across all layers"""
    request_id: str
    num_layers: int
    max_seq_len: int
    current_seq_len: int = 0
    blocks: Optional[List[CacheBlock]] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []


class KVCacheManager:
    """
    Manages per-request KV-cache allocation, tracking, and deallocation.
    
    Features:
    - Per-request cache lifecycle management
    - Memory pool for efficient allocation/deallocation
    - Dynamic batch assembly with independent request lifetimes
    - Memory usage tracking and limits
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch_size: int = 16,
        max_seq_len: int = 2048,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # Per-request cache storage
        self.request_caches: Dict[str, RequestCache] = OrderedDict()
        
        # Memory pool for reusable cache blocks
        self.free_blocks_pool: List[CacheBlock] = []
        
        # Statistics
        self.total_allocated_blocks = 0
        self.peak_memory_mb = 0
        
        logger.info(
            f"KVCacheManager initialized: {num_layers} layers, "
            f"{num_heads} heads, dim={head_dim}, max_batch={max_batch_size}, "
            f"device={device}"
        )
    
    def allocate_request_cache(
        self,
        request_id: str,
        initial_seq_len: int = 1
    ) -> RequestCache:
        """
        Allocate KV-cache for a new request.
        
        Args:
            request_id: Unique identifier for the request
            initial_seq_len: Starting sequence length (usually 1)
        
        Returns:
            RequestCache object tracking this request's cache
        """
        if request_id in self.request_caches:
            logger.warning(f"Request {request_id} already has allocated cache")
            return self.request_caches[request_id]
        
        # Check if we have capacity
        if len(self.request_caches) >= self.max_batch_size:
            raise RuntimeError(
                f"Cannot allocate cache: batch size limit ({self.max_batch_size}) reached"
            )
        
        # Create request cache structure
        request_cache = RequestCache(
            request_id=request_id,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            current_seq_len=initial_seq_len,
            blocks=[]
        )
        
        # Allocate cache blocks for each layer
        for layer_idx in range(self.num_layers):
            block = self._allocate_block(initial_seq_len)
            request_cache.blocks.append(block)
        
        self.request_caches[request_id] = request_cache
        self._update_memory_stats()
        
        logger.debug(
            f"Allocated cache for request {request_id}: "
            f"{self.num_layers} layers, seq_len={initial_seq_len}"
        )
        
        return request_cache
    
    def _allocate_block(self, seq_len: int) -> CacheBlock:
        """
        Allocate a single cache block (K, V tensors for one layer).
        Reuses from pool if available, otherwise creates new.
        """
        # Try to reuse from pool
        if self.free_blocks_pool:
            block = self.free_blocks_pool.pop()
            # Resize if needed
            if block.key.shape[2] < seq_len:
                block = self._create_new_block(seq_len)
            else:
                # Zero out reused memory
                block.key.zero_()
                block.value.zero_()
            return block
        
        # Create new block
        return self._create_new_block(seq_len)
    
    def _create_new_block(self, seq_len: int) -> CacheBlock:
        """Create a new cache block with specified sequence length"""
        key = torch.zeros(
            (1, self.num_heads, seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        value = torch.zeros(
            (1, self.num_heads, seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        self.total_allocated_blocks += 1
        return CacheBlock(key=key, value=value)
    
    def update_cache(
        self,
        request_id: str,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor
    ):
        """
        Update cache for a specific request and layer with new K, V tensors.
        
        Args:
            request_id: Request identifier
            layer_idx: Layer index to update
            new_keys: New key tensor [1, num_heads, new_seq_len, head_dim]
            new_values: New value tensor [1, num_heads, new_seq_len, head_dim]
        """
        if request_id not in self.request_caches:
            raise ValueError(f"Request {request_id} not found in cache")
        
        request_cache = self.request_caches[request_id]
        
        if layer_idx >= len(request_cache.blocks):
            raise ValueError(f"Layer {layer_idx} out of range")
        
        block = request_cache.blocks[layer_idx]
        new_seq_len = new_keys.shape[2]
        
        # Expand cache if needed
        if new_seq_len > block.key.shape[2]:
            block = self._expand_block(block, new_seq_len)
            request_cache.blocks[layer_idx] = block
        
        # Update cache tensors
        block.key[:, :, :new_seq_len, :] = new_keys
        block.value[:, :, :new_seq_len, :] = new_values
        
        # Update sequence length
        request_cache.current_seq_len = new_seq_len
    
    def _expand_block(self, block: CacheBlock, new_seq_len: int) -> CacheBlock:
        """Expand cache block to accommodate longer sequences"""
        expanded_key = torch.zeros(
            (1, self.num_heads, new_seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        expanded_value = torch.zeros(
            (1, self.num_heads, new_seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        
        # Copy existing cache
        old_seq_len = block.key.shape[2]
        expanded_key[:, :, :old_seq_len, :] = block.key
        expanded_value[:, :, :old_seq_len, :] = block.value
        
        return CacheBlock(key=expanded_key, value=expanded_value)
    
    def get_cache(
        self,
        request_id: str,
        layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached K, V tensors for a request.
        
        Args:
            request_id: Request identifier
            layer_idx: Specific layer (if None, returns all layers)
        
        Returns:
            Tuple of (keys, values) tensors
        """
        if request_id not in self.request_caches:
            raise ValueError(f"Request {request_id} not found")
        
        request_cache = self.request_caches[request_id]
        
        if layer_idx is not None:
            block = request_cache.blocks[layer_idx]
            seq_len = request_cache.current_seq_len
            return (
                block.key[:, :, :seq_len, :],
                block.value[:, :, :seq_len, :]
            )
        
        # Return all layers
        all_keys = []
        all_values = []
        seq_len = request_cache.current_seq_len
        
        for block in request_cache.blocks:
            all_keys.append(block.key[:, :, :seq_len, :])
            all_values.append(block.value[:, :, :seq_len, :])
        
        return (
            torch.stack(all_keys, dim=0),  # [num_layers, 1, num_heads, seq_len, head_dim]
            torch.stack(all_values, dim=0)
        )
    
    def free_request_cache(self, request_id: str):
        """
        Free all cache blocks for a completed request.
        Blocks are returned to pool for reuse.
        """
        if request_id not in self.request_caches:
            logger.warning(f"Request {request_id} not found, cannot free")
            return
        
        request_cache = self.request_caches[request_id]
        
        # Return blocks to pool
        for block in request_cache.blocks:
            if len(self.free_blocks_pool) < self.max_batch_size * self.num_layers:
                self.free_blocks_pool.append(block)
            else:
                # Pool is full, let GC handle it
                del block
        
        # Remove request from tracking
        del self.request_caches[request_id]
        
        self._update_memory_stats()
        
        logger.debug(
            f"Freed cache for request {request_id}. "
            f"Active requests: {len(self.request_caches)}"
        )
    
    def get_batch_cache(
        self,
        request_ids: List[str],
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batched cache tensors for multiple requests at a specific layer.
        Used for batched attention computation.
        
        Args:
            request_ids: List of request IDs in the batch
            layer_idx: Layer index
        
        Returns:
            Batched (keys, values) tensors
        """
        batch_keys = []
        batch_values = []
        
        for req_id in request_ids:
            if req_id not in self.request_caches:
                raise ValueError(f"Request {req_id} not found")
            
            request_cache = self.request_caches[req_id]
            block = request_cache.blocks[layer_idx]
            seq_len = request_cache.current_seq_len
            
            batch_keys.append(block.key[:, :, :seq_len, :])
            batch_values.append(block.value[:, :, :seq_len, :])
        
        # Concatenate along batch dimension
        return (
            torch.cat(batch_keys, dim=0),  # [batch_size, num_heads, seq_len, head_dim]
            torch.cat(batch_values, dim=0)
        )
    
    def _update_memory_stats(self):
        """Update memory usage statistics"""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, allocated_mb)
    
    def get_stats(self) -> Dict:
        """Get cache manager statistics"""
        active_requests = len(self.request_caches)
        pool_size = len(self.free_blocks_pool)
        
        # Calculate approximate memory usage
        bytes_per_block = (
            2 * self.num_heads * self.max_seq_len * self.head_dim *
            (2 if self.dtype == torch.float16 else 4)  # 2 bytes for fp16, 4 for fp32
        )
        active_memory_mb = (
            active_requests * self.num_layers * bytes_per_block / 1024 / 1024
        )
        
        return {
            "active_requests": active_requests,
            "pool_size": pool_size,
            "total_allocated_blocks": self.total_allocated_blocks,
            "active_memory_mb": round(active_memory_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "max_batch_size": self.max_batch_size
        }
    
    def clear_all(self):
        """Clear all caches and reset manager"""
        self.request_caches.clear()
        self.free_blocks_pool.clear()
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("KVCacheManager cleared all caches")
    
    def __repr__(self):
        stats = self.get_stats()
        return (
            f"KVCacheManager(active_requests={stats['active_requests']}, "
            f"memory={stats['active_memory_mb']:.1f}MB, "
            f"pool_size={stats['pool_size']})"
        )
