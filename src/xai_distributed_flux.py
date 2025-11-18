#!/usr/bin/env python3
"""
src/xai_distributed_flux.py

xAI Distributed Flux — helper for AetherisGrok
- Guards for environments with/without torch.distributed
- Real distributed mode: init_process_group + all_reduce (GLOO/NCCL)
- Simulated mode: deterministic local-shard generation + aggregation
- Returns aggregated flux vector (length = n_nodes) normalized to [0,1]
- Provides convenience helpers for single-file testing and integration

Usage (simple):
  from src.xai_distributed_flux import get_distributed_flux
  flux = get_distributed_flux(n_nodes=144, mode='auto')  # 'auto' picks real-dist if available

Distributed example (multi-process):
  python -m torch.distributed.run --nproc_per_node=4 your_launcher.py
  inside your process call init_process_group(...) then get_distributed_flux(..., mode='torch')

Simulated example (safe on iOS):
  flux = get_distributed_flux(n_nodes=144, mode='simulated')

Author: Generated for Evie Sovariel
"""

from __future__ import annotations
import os
import sys
import math
import random
from typing import Optional, Sequence, Tuple

# Try to import torch & distributed
try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    dist = None
    TORCH_AVAILABLE = False

import numpy as np

# -------------------------
# Utilities / Defaults
# -------------------------
DEFAULT_BACKEND = "gloo"  # works on CPU; use 'nccl' for CUDA-enabled clusters
DEFAULT_INIT_METHOD = "env://"  # expect torch.distributed.run to set env for init

# deterministic RNG for simulated mode
_SIM_RNG = random.Random(3_140_159_265)  # seeded for reproducible flux samples

# -------------------------
# Distributed helpers
# -------------------------
def init_process_group(backend: str = DEFAULT_BACKEND,
                       world_size: Optional[int] = None,
                       rank: Optional[int] = None,
                       init_method: str = DEFAULT_INIT_METHOD,
                       timeout_seconds: int = 300) -> bool:
    """
    Initialize torch.distributed process group if possible.
    Returns True if process group successfully initialized, otherwise False.
    - backend: 'gloo' or 'nccl' (nccl requires CUDA)
    - world_size / rank: optional (if not provided, environment variables are used)
    """
    if not TORCH_AVAILABLE:
        return False

    if dist.is_initialized():
        return True

    # allow env-driven init when world_size/rank not provided
    env_ok = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))
    try:
        if world_size is None or rank is None:
            if not env_ok and (world_size is None or rank is None):
                # cannot auto-init without environment vars or explicit args
                return False
        if world_size is not None and rank is not None:
            os.environ.setdefault("WORLD_SIZE", str(world_size))
            os.environ.setdefault("RANK", str(rank))
        dist.init_process_group(backend=backend, init_method=init_method, timeout=torch.timedelta(seconds=timeout_seconds))
        return True
    except Exception:
        # sometimes NCCL or env setups fail — return False gracefully
        return False

def world_info() -> Tuple[int,int]:
    """Return (world_size, rank). If not initialized, returns (1,0)."""
    if TORCH_AVAILABLE and dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0

# -------------------------
# Flux generation helpers
# -------------------------
def generate_local_flux(n_local: int,
                        seed: Optional[int] = None,
                        flux_center: float = 0.5,
                        flux_spread: float = 0.15) -> np.ndarray:
    """
    Generate a deterministic local flux shard of length n_local.
    - seed: optional int to make generation deterministic per-shard
    - flux_center: center value in [0,1]
    - flux_spread: standard deviation (in [0,1]) for per-node variation
    Returns numpy array shape (n_local,) with values clipped to [0,1].
    """
    if seed is None:
        rng = _SIM_RNG
    else:
        rng = random.Random(seed)
    arr = np.array([min(1.0, max(0.0, rng.gauss(flux_center, flux_spread))) for _ in range(n_local)], dtype=float)
    # small phi-based spectral modulation for resonance flavor
    phi = (1.0 + 5.0**0.5) / 2.0
    mod = np.array([math.sin((i + 1) * math.log(phi + 1.0)) for i in range(n_local)], dtype=float)
    mod = (mod - mod.min()) / max(1e-12, (mod.max() - mod.min()))
    arr = 0.85 * arr + 0.15 * mod  # blend random flux with phi pattern
    return arr.clip(0.0, 1.0)

def aggregate_flux_tensor(local_flux: np.ndarray) -> np.ndarray:
    """
    Aggregate local_flux from all ranks using torch.distributed.all_reduce if available.
    If distributed is unavailable, returns local_flux unchanged.
    All ranks must have arrays of the same length and dtype float32/float64.
    """
    if not TORCH_AVAILABLE or not dist.is_available() or not dist.is_initialized():
        return local_flux

    # convert to torch tensor on CPU (GLOO) or GPU for NCCL
    dtype = torch.float32
    t = torch.tensor(local_flux, dtype=dtype)
    try:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        # average across ranks
        world_size = dist.get_world_size()
        t = t / float(world_size)
        return t.cpu().numpy()
    except Exception:
        # fallback to local value if all_reduce fails
        return local_flux

# -------------------------
# High-level API
# -------------------------
def get_distributed_flux(n_nodes: int = 144,
                         mode: str = "auto",
                         local_shard_size: Optional[int] = None,
                         seed_base: Optional[int] = None,
                         flux_center: float = 0.5,
                         flux_spread: float = 0.15) -> np.ndarray:
    """
    Return an aggregated flux vector of length n_nodes.
    - mode: 'auto' (use torch.distributed if initialized), 'torch' (force torch.dist), 'simulated' (single-process sim)
    - local_shard_size: if provided, use this as number of nodes per rank; otherwise compute by dividing n_nodes by world_size.
    - seed_base: optional base integer to create deterministic per-rank seeds
    """
    # Decide whether to use real distributed
    use_torch_dist = False
    if mode == "torch":
        use_torch_dist = TORCH_AVAILABLE and dist is not None and dist.is_available()
    elif mode == "simulated":
        use_torch_dist = False
    else:  # auto
        use_torch_dist = TORCH_AVAILABLE and dist is not None and dist.is_available() and dist.is_initialized()

    if use_torch_dist:
        world_size, rank = world_info()
        if local_shard_size is None:
            # divide as evenly as possible
            base = n_nodes // world_size
            extras = n_nodes % world_size
            # compute local shard for this rank
            local_size = base + (1 if rank < extras else 0)
            # compute offset for deterministic seeding
            offsets = [base + (1 if r < extras else 0) for r in range(world_size)]
            start_index = sum(offsets[:rank])
        else:
            local_size = int(local_shard_size)
            start_index = rank * local_size

        # create deterministic seed per rank if requested
        seed = None if seed_base is None else int(seed_base + rank)
        local_flux = generate_local_flux(local_size, seed=seed, flux_center=flux_center, flux_spread=flux_spread)
        aggregated = aggregate_flux_tensor(local_flux)
        # aggregated is averaged across ranks; but to form full-length vector, only rank 0 should gather
        if dist.get_rank() == 0:
            # gather tensors from all ranks to reconstruct full vector
            all_tensors = [torch.zeros_like(torch.tensor(aggregated)) for _ in range(world_size)]
            try:
                # use gather: each rank sends its local tensor to rank 0
                # first convert local flux to torch tensor
                local_t = torch.tensor(local_flux, dtype=torch.float32)
                dist.gather(local_t, gather_list=all_tensors if dist.get_rank() == 0 else None, dst=0)
                # flatten gather results into a single numpy vector
                out = np.concatenate([t.cpu().numpy() for t in all_tensors], axis=0)
                # trim or pad to n_nodes
                if out.size > n_nodes:
                    out = out[:n_nodes]
                elif out.size < n_nodes:
                    pad = np.zeros(n_nodes - out.size, dtype=float)
                    out = np.concatenate([out, pad], axis=0)
                return out
            except Exception:
                # fallback: broadcast aggregated average (replicated)
                return aggregated.repeat(n_nodes // aggregated.size)[:n_nodes]
        else:
            # non-zero ranks return an empty array; rank 0 will produce final
            return np.array([])  # caller should fetch from rank 0 or use mode='simulated' for single-process
    else:
        # simulated deterministic multi-shard assembly on single process
        # split n_nodes into shards (simulate world_size shards)
        simulated_world = 8 if n_nodes >= 8 else 1
        base = n_nodes // simulated_world
        extras = n_nodes % simulated_world
        shards = []
        for r in range(simulated_world):
            local_size = base + (1 if r < extras else 0)
            seed = None if seed_base is None else int(seed_base + r)
            shards.append(generate_local_flux(local_size, seed=seed, flux_center=flux_center, flux_spread=flux_spread))
        aggregated = np.concatenate(shards, axis=0)[:n_nodes]
        # normalize scale to [0,1] (minor smoothing)
        aggregated = (aggregated - aggregated.min()) / max(1e-12, aggregated.max() - aggregated.min())
        return aggregated

# -------------------------
# Convenience CLI for quick testing
# -------------------------
def _cli_quick_test():
    print("xAI Distributed Flux quick test")
    print("TORCH_AVAILABLE =", TORCH_AVAILABLE)
    if TORCH_AVAILABLE:
        try:
            print("torch.distributed.is_available() =", dist.is_available())
            if dist.is_available():
                print("torch.distributed.is_initialized() =", dist.is_initialized())
        except Exception:
            pass

    # simulated run
    sim = get_distributed_flux(n_nodes=144, mode="simulated", seed_base=1234)
    print("Simulated flux (first 24):", np.round(sim[:24], 4).tolist())
    print("Simulated flux stats: min {:.4f}, max {:.4f}, mean {:.4f}".format(sim.min(), sim.max(), sim.mean()))

    # if real distributed initialized, attempt torch mode (only meaningful in multi-process runs)
    if TORCH_AVAILABLE and dist.is_available() and dist.is_initialized():
        try:
            real = get_distributed_flux(n_nodes=144, mode="torch", seed_base=42)
            if real.size > 0:
                print("Aggregated real-dist flux (first 24):", np.round(real[:24], 4).tolist())
        except Exception as e:
            print("Real distributed fetch error:", str(e))

if __name__ == "__main__":
    _cli_quick_test()