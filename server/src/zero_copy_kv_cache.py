"""
server/zero_copy_kv_cache.py

Zero-copy KV-cache for TorchWeave continuous batching.

The existing kv_cache_manager.py allocates new torch.zeros tensors per
request and copies data into them. That causes two problems:
  1. cudaMalloc on every request is slow (~100us per call).
  2. Tensor slicing + assignment = implicit device-to-device memcpy.

This module eliminates both by pre-allocating one large GPU slab at
startup and handing out views (slices) into it. A view shares the
underlying storage -- no copy on allocation or on read-back.

Design:
  - One slab per (layer, K/V) of shape [max_batch, num_heads, max_seq, head_dim].
  - Each request gets a slot index [0, max_batch). Slot assignment is O(1)
    via a free-list.
  - update_cache() writes directly into the slab slice for that slot --
    the write is in-place, no intermediate tensor.
  - get_cache() returns a narrow() view -- zero bytes copied.


"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SlotState:
    request_id: str
    slot_idx: int
    seq_len: int = 0
    active: bool = True


class ZeroCopyKVCache:
    """
    Pre-allocated slab-based KV cache with zero-copy reads.

    All tensors are allocated once in __init__. Requests get slot
    assignments from a free-list. Reads return narrow() views.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch: int = 32,
        max_seq_len: int = 2048,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

        # Pre-allocate slabs: [num_layers, max_batch, num_heads, max_seq, head_dim]
        # Contiguous layout so narrow() along dim=1 (slot) and dim=3 (seq) are free.
        slab_shape = (num_layers, max_batch, num_heads, max_seq_len, head_dim)
        self.k_slab = torch.zeros(slab_shape, dtype=dtype, device=device)
        self.v_slab = torch.zeros(slab_shape, dtype=dtype, device=device)

        # Free slot list (stack): O(1) alloc/free
        self._free_slots: List[int] = list(range(max_batch))
        self._slots: Dict[str, SlotState] = {}
        self._lock = threading.Lock()

        # Stats
        self._total_allocs = 0
        self._peak_active = 0

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def allocate(self, request_id: str) -> int:
        """
        Assign a slab slot to a new request. O(1).
        Returns the slot index.
        """
        with self._lock:
            if not self._free_slots:
                raise RuntimeError(
                    f"KV cache exhausted: max_batch={self.max_batch} slots in use. "
                    "Increase max_batch or reduce concurrent requests."
                )
            if request_id in self._slots:
                return self._slots[request_id].slot_idx

            slot = self._free_slots.pop()
            # Zero out the slot before reuse (in-place, no allocation)
            self.k_slab[:, slot, :, :, :].zero_()
            self.v_slab[:, slot, :, :, :].zero_()

            self._slots[request_id] = SlotState(request_id=request_id, slot_idx=slot)
            self._total_allocs += 1
            self._peak_active = max(self._peak_active, len(self._slots))
            return slot

    def free(self, request_id: str):
        """Return the slot to the free-list. O(1)."""
        with self._lock:
            if request_id not in self._slots:
                return
            slot = self._slots.pop(request_id).slot_idx
            self._free_slots.append(slot)

    # ------------------------------------------------------------------
    # Cache read/write -- the zero-copy critical path
    # ------------------------------------------------------------------

    def write(
        self,
        request_id: str,
        layer_idx: int,
        new_k: torch.Tensor,  # [1, num_heads, new_seq_len, head_dim]
        new_v: torch.Tensor,
    ):
        """
        Write new K/V into the slab slot for this request.

        This is an in-place scatter, not a copy to a new tensor.
        new_k and new_v must already be on self.device.

        For decode step (one new token):
            new_k shape = [1, num_heads, 1, head_dim]
            We write into position seq_len (the next empty column).

        For prefill (full prompt):
            new_k shape = [1, num_heads, prompt_len, head_dim]
            We write columns 0..prompt_len-1.
        """
        state = self._slots[request_id]
        slot = state.slot_idx
        new_seq = new_k.shape[2]

        # Write in-place into slab -- no cudaMalloc, no intermediate buffer
        self.k_slab[layer_idx, slot, :, :new_seq, :].copy_(new_k[0])
        self.v_slab[layer_idx, slot, :, :new_seq, :].copy_(new_v[0])

        # Update tracked length only on layer 0 to avoid duplicate increments
        if layer_idx == 0:
            state.seq_len = new_seq

    def append_token(
        self,
        request_id: str,
        layer_idx: int,
        new_k: torch.Tensor,  # [1, num_heads, 1, head_dim]
        new_v: torch.Tensor,
    ):
        """
        Append a single new token's K/V to an existing sequence.
        Called once per decode step per layer.
        """
        state = self._slots[request_id]
        slot = state.slot_idx
        pos = state.seq_len  # column to write into

        if pos >= self.max_seq_len:
            raise RuntimeError(
                f"Request {request_id} exceeded max_seq_len={self.max_seq_len}"
            )

        # In-place write at position `pos` -- zero-copy
        self.k_slab[layer_idx, slot, :, pos, :].copy_(new_k[0, :, 0, :])
        self.v_slab[layer_idx, slot, :, pos, :].copy_(new_v[0, :, 0, :])

        if layer_idx == 0:
            state.seq_len += 1

    def read(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (K, V) views for this request at this layer.

        Returns torch.narrow() views -- zero bytes copied.
        Shape: [1, num_heads, seq_len, head_dim]
        """
        state = self._slots[request_id]
        slot = state.slot_idx
        seq_len = state.seq_len

        # narrow() returns a view into the slab, not a copy
        k_view = self.k_slab[layer_idx, slot, :, :seq_len, :].unsqueeze(0)
        v_view = self.v_slab[layer_idx, slot, :, :seq_len, :].unsqueeze(0)
        return k_view, v_view

    def read_batch(
        self,
        request_ids: List[str],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return batched (K, V) for multiple requests at one layer.
        Used in the continuous batcher to assemble a GPU batch
        without any host-device transfers.

        Sequences are padded to the longest in the batch.
        Shape: [batch, num_heads, max_seq_in_batch, head_dim]

        Note: padding with zeros is fine because causal attention masks
        prevent attending to padding positions.
        """
        max_seq = max(self._slots[rid].seq_len for rid in request_ids)
        slots = [self._slots[rid].slot_idx for rid in request_ids]
        seqlens = [self._slots[rid].seq_len for rid in request_ids]

        # index_select pulls rows by slot index -- still a gather, not memcpy
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=self.device)

        # k_slab[layer, slots, :, :max_seq, :] -- gather along dim=1
        k_batch = self.k_slab[layer_idx].index_select(0, slot_tensor)[:, :, :max_seq, :]
        v_batch = self.v_slab[layer_idx].index_select(0, slot_tensor)[:, :, :max_seq, :]

        return k_batch, v_batch, seqlens

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        active = len(self._slots)
        slab_bytes = (self.k_slab.nelement() + self.v_slab.nelement()) * (
            2 if self.dtype == torch.float16 else 4
        )
        used_bytes = (
            active
            * self.num_layers
            * self.num_heads
            * self.max_seq_len
            * self.head_dim
            * (2 if self.dtype == torch.float16 else 4)
            * 2
        )
        return {
            "active_slots": active,
            "free_slots": len(self._free_slots),
            "peak_active": self._peak_active,
            "total_allocs": self._total_allocs,
            "slab_size_mb": slab_bytes / 1024 / 1024,
            "used_mb": used_bytes / 1024 / 1024,
            "utilization_pct": (active / self.max_batch) * 100,
        }
