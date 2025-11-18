#!/usr/bin/env python3
"""
src/HarmonicAge.py
HarmonicAge: Modular Harmonic Swarm Engine (evolutionary)
- Offline-safe (guards for torch, qutip, sympy, networkx)
- Adaptive triad fusion (semantic + video influence)
- Orch-OR surrogate scaling for large-N coherence estimation
- Multi-seed Torch swarm training with entropy-penalized losses
- API: triad_from_semantic(), evolve(), train_seeds(), quick_benchmark()
"""

# Standard libs
import os
import random
from collections import deque
import json
import math
import time

# Numeric / ML (guarded)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

try:
    import numpy as np
except Exception:
    np = None

# Graph (guarded)
try:
    import networkx as nx
except Exception:
    nx = None

# Quantum / symbolic (guarded)
try:
    from qutip import Qobj, sigmax, sigmaz, mesolve
except Exception:
    Qobj = None
    sigmax = None
    sigmaz = None
    mesolve = None

try:
    from sympy import symbols, Abs
except Exception:
    symbols = None
    Abs = None

# Optional CV/audio libraries are not required for basic runs
# Constants
PHI = (1.0 + 5.0**0.5) / 2.0
PAC_HZ = 3.0616
DEFAULT_TRIAD = (0.5, 0.25, 0.25)

# Running sentiment window
_SENTIMENT_WINDOW = deque(maxlen=120)

# Sympy symbolic setup (if available)
if symbols is not None:
    m_tub, d, G_sym, hbar_sym = symbols("m_tub d G hbar")
    r = d / 2
    E_g_sym = G_sym * (m_tub**2) / (5 * r)
    tau_sym = hbar_sym / Abs(E_g_sym)
else:
    m_tub = d = G_sym = hbar_sym = None
    E_g_sym = None
    tau_sym = None


# ---------------- Utility: Hameroff tau ----------------
def hameroff_tau(m_tub_val=1e-22, d_val=1e-9, G_val=6.67430e-11, hbar_val=1.054571817e-34):
    """Return (E_g, tau) using symbolic formula when available, else fallback."""
    if E_g_sym is None or tau_sym is None:
        # Safe fallback approximate values
        return 1e-40, 1e-6
    E_g_num = float(E_g_sym.subs({
        m_tub: m_tub_val,
        d: d_val,
        G_sym: G_val,
        hbar_sym: hbar_val
    }).evalf())
    tau_num = float(tau_sym.subs({
        m_tub: m_tub_val,
        d: d_val,
        G_sym: G_val,
        hbar_sym: hbar_val
    }).evalf())
    return abs(E_g_num), tau_num


# ---------------- Adaptive triad weights ----------------
def adaptive_triad_weights(sentiment_sample, video_embed_signal, base_weights=DEFAULT_TRIAD,
                           alpha_s=0.6, alpha_v=0.4, min_w=0.05):
    """
    Fuse a scalar sentiment (-1..1) and a video signal (0..1) into normalized triad weights.
    Returns tuple (w_sem, w_qual, w_flux).
    """
    _SENTIMENT_WINDOW.append(float(sentiment_sample))
    s_avg = float(sum(_SENTIMENT_WINDOW)) / max(1, len(_SENTIMENT_WINDOW))
    sem_boost = (s_avg + 1.0) / 2.0  # map -1..1 -> 0..1
    v_sig = max(0.0, min(1.0, float(video_embed_signal)))

    bw = list(base_weights)
    bw[0] = bw[0] + alpha_s * sem_boost * (1.0 - bw[0])
    bw[1] = bw[1] + alpha_v * v_sig * (1.0 - bw[1])

    # enforce minimum
    bw = [max(min_w, w) for w in bw]
    s = sum(bw)
    bw = [w / s for w in bw]
    return tuple(bw)


# ---------------- Triad embed generators ----------------
def triad_embeds(batch_size=32, flux_batch=None, weights=DEFAULT_TRIAD):
    """
    Generate triad embeddings. Requires torch.
    Returns list: [semantics, qualia, flux_emb] each tensor [batch, 32]
    """
    if torch is None:
        raise RuntimeError("Torch is required for triad_embeds.")
    if flux_batch is None:
        flux_batch = torch.tensor(np.random.uniform(40, 500, batch_size), dtype=torch.float32)
    elif isinstance(flux_batch, np.ndarray):
        flux_batch = torch.tensor(flux_batch, dtype=torch.float32)
    semantics = torch.randn(batch_size, 32) * float(PHI)
    qualia = torch.randn(batch_size, 32) * float(PAC_HZ)
    flux_emb = torch.randn(batch_size, 32) * (flux_batch.unsqueeze(1) / 1000.0)
    return [
        weights[0] * semantics,
        weights[1] * qualia,
        weights[2] * flux_emb
    ]


def triad_embeds_adaptive(batch_size=32, flux_batch=None, adaptive_weights=DEFAULT_TRIAD):
    return triad_embeds(batch_size=batch_size, flux_batch=flux_batch, weights=adaptive_weights)


# ---------------- Orch-OR surrogate scaling ----------------
def full_mesolve_tubulin(flux_hz, tau_collapse, tlist=None):
    """
    Runs QuTiP mesolve for a single 2-level tubulin system if qutip is available.
    Returns (coherence, rho_final).
    """
    if mesolve is None or Qobj is None:
        raise RuntimeError("QuTiP is required for full_mesolve_tubulin.")
    if tlist is None:
        tlist = np.linspace(0, 0.01, 20) if np is not None else [i * 0.0005 for i in range(20)]
    rho0 = Qobj(np.array([[0.5, 0.5], [0.5, 0.5]]))
    H = flux_hz * 2 * math.pi * sigmax()
    gamma_dephase = max(flux_hz / 100.0, 1e-12)
    gamma_collapse = max(1.0 / tau_collapse, 1e-12)
    c_ops = [math.sqrt(gamma_dephase) * sigmaz(), math.sqrt(gamma_collapse) * sigmax()]
    result = mesolve(H, rho0, tlist, c_ops)
    coherence = abs(result.states[-1][0, 1])**2
    return coherence, result.states[-1]


def orchor_coherence_surrogate(mean_flux_hz, tau_collapse, N_eff=1e6, sample_k=8, qutip_available=(mesolve is not None)):
    """
    Estimate coherence for large N by sampling small mesolve runs (if available)
    and applying a 1/sqrt(N) style dilution factor.
    """
    coherences = []
    sample_k = max(1, int(sample_k))
    if qutip_available:
        for _ in range(sample_k):
            try:
                coh, _ = full_mesolve_tubulin(mean_flux_hz, tau_collapse)
                coherences.append(float(coh))
            except Exception:
                coherences.append(random.uniform(0.0, 0.2))
    else:
        coherences = [random.uniform(0.0, 0.2) for _ in range(sample_k)]

    mean_coh_sample = float(np.mean(coherences)) if np is not None else float(sum(coherences) / len(coherences))
    scale = 1.0 / max(1.0, math.sqrt(max(1.0, float(N_eff) / float(sample_k))))
    scaled = mean_coh_sample * scale
    return min(1.0, float(scaled))


# ---------------- Core HarmonicSwarm (modular) ----------------
class HarmonicSwarm(nn.Module):
    def __init__(self, n_nodes=100):
        if nn is None:
            raise RuntimeError("PyTorch required for HarmonicSwarm.")
        super().__init__()
        self.embed = nn.Embedding(n_nodes, 32)
        self.policy_head = nn.Linear(32 * 3, 2)
        self.value_head = nn.Linear(32 * 3, 1)
        self.graph = nx.DiGraph() if nx is not None else None
        for i in range(n_nodes):
            if self.graph is not None:
                self.graph.add_node(i, pos=(PHI**i % 10, PHI**(i+1) % 10))
        self.interactions = {}

    def forward(self, flux_batch, triad_embeds_list):
        # triad_embeds_list: list of 3 tensors [batch,32]
        embeds = torch.cat(triad_embeds_list, dim=1)  # [batch,96]
        policy_logits = self.policy_head(embeds)
        value = self.value_head(embeds)
        entropy = 0.0
        if self.graph is not None:
            mean_p = float(torch.mean(torch.sigmoid(policy_logits[:, 0])).item())
            # probabilistic edge creation
            if mean_p > 0.5:
                i = random.randint(0, self.graph.number_of_nodes() - 1)
                j = random.randint(0, self.graph.number_of_nodes() - 1)
                if i != j and not self.graph.has_edge(i, j):
                    self.graph.add_edge(i, j)
            deg_hist = nx.degree_histogram(self.graph)
            total = sum(deg_hist)
            if total > 0:
                probs = [d / total for d in deg_hist if d > 0]
                entropy = float(-sum(p * math.log(p + 1e-12) for p in probs))
        return policy_logits, value, entropy


# ---------------- Train / Evaluation Routines ----------------
def train_seeds(n_seeds=3, epochs=30, batch_size=32, N_eff=1e6, sample_k=8):
    """
    Train multiple seeds of HarmonicSwarm; returns (models_list, entropy_logs).
    """
    if torch is None:
        raise RuntimeError("PyTorch required for training.")
    models = []
    entropy_logs = {}
    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"[seed {seed}] Initializing...")
        model = HarmonicSwarm()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        last_entropy = 0.0
        for epoch in range(epochs):
            flux_batch = torch.tensor(np.random.uniform(40, 500, batch_size), dtype=torch.float32)
            # offline mocked signals
            sentiment = random.uniform(-0.5, 0.8)
            video_sig = random.uniform(0.0, 1.0)
            adapt_w = adaptive_triad_weights(sentiment, video_sig)
            triad_batch = triad_embeds_adaptive(batch_size=batch_size, flux_batch=flux_batch, adaptive_weights=adapt_w)
            # targets and forward
            target_p = torch.sigmoid(torch.tensor(np.random.uniform(0.4, 0.8, batch_size), dtype=torch.float32).unsqueeze(1))
            policy_logits, value, entropy = model(flux_batch, triad_batch)
            # convert entropy float to tensor for loss
            entropy_tensor = torch.tensor(float(entropy), dtype=torch.float32)
            mse_loss = criterion(value, target_p)
            # coherence surrogate using batch mean flux
            try:
                _, tau_val = hameroff_tau()
            except Exception:
                tau_val = 1e-6
            mean_flux = float(torch.mean(flux_batch).item())
            coh_est = orchor_coherence_surrogate(mean_flux, tau_val, N_eff=N_eff, sample_k=sample_k)
            coherence_tensor = torch.tensor(coh_est, dtype=torch.float32)
            pred_mean_p = torch.mean(torch.sigmoid(policy_logits[:, 0])).unsqueeze(0)
            coherence_bonus = 0.5 * coherence_tensor * pred_mean_p
            # Final loss: MSE + entropy_penalty - coherence_bonus
            loss = mse_loss + 0.12 * entropy_tensor - coherence_bonus.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_entropy = float(entropy)
            if epoch % max(1, epochs // 4) == 0:
                print(f"[seed {seed}] epoch {epoch}/{epochs} loss={float(loss):.6f} entropy={last_entropy:.6f} coh_est={coh_est:.6f}")
        models.append(model)
        entropy_logs[seed] = last_entropy
    return models, entropy_logs


def batch_benchmark(model, flux_ranges=[(40, 50), (70, 80), (100, 110)], n_batches=8, batch_size=32):
    """Run small offline benchmark across flux ranges; returns aggregated dict."""
    aggregated = {}
    for low, high in flux_ranges:
        p_vals = []
        ent_vals = []
        coh_vals = []
        for _ in range(n_batches):
            flux_batch = torch.tensor(np.random.uniform(low, high, batch_size), dtype=torch.float32)
            triad_batch = triad_embeds(batch_size, flux_batch)
            policy_logits, value, entropy = model(flux_batch, triad_batch)
            pred_p = torch.sigmoid(policy_logits[:, 0]).detach().numpy()
            mean_flux = float(torch.mean(flux_batch).item())
            _, tau_val = hameroff_tau()
            coh = orchor_coherence_surrogate(mean_flux, tau_val, N_eff=1e4, sample_k=4)
            p_vals.append(float(np.mean(pred_p)))
            ent_vals.append(float(entropy))
            coh_vals.append(float(coh))
        aggregated[(low, high)] = {
            "mean_P": float(np.mean(p_vals)),
            "mean_entropy": float(np.mean(ent_vals)),
            "mean_coherence": float(np.mean(coh_vals))
        }
    return aggregated


# ---------------- Convenience API for AetherisGrok integration ----------------
class HarmonicAgeEngine:
    def __init__(self, n_seeds=3, device="cpu"):
        self.n_seeds = n_seeds
        self.device = device
        self.models = []
        self.entropies = {}

    def triad_from_semantic(self, semantic_vector, video_signal=0.0):
        """
        Create a triad from a semantic embedding vector (list/iterable).
        semantic_vector: list of floats (embedding)
        video_signal: scalar 0..1
        Returns adaptive_weights and a triad sample (for one batch element).
        """
        # map semantic_vector -> sentiment scalar by simple projection
        sem = list(semantic_vector)
        s_val = float(sum(sem) / max(1.0, len(sem)))
        adapt_w = adaptive_triad_weights(s_val, video_signal)
        triad = triad_embeds(batch_size=1, flux_batch=torch.tensor([432.0], dtype=torch.float32), weights=adapt_w)
        return adapt_w, triad

    def evolve(self, triad_sample, steps=64):
        """
        Run a compact evolution using one model (seed 0); returns summary dict.
        """
        # ensure model exists
        model = HarmonicSwarm()
        # quick warmup
        history = {"entropy": [], "pred_mean": []}
        for step in range(steps):
            flux_batch = torch.tensor(np.random.uniform(40, 500, 1), dtype=torch.float32)
            policy_logits, value, entropy = model(flux_batch, triad_sample)
            history["entropy"].append(float(entropy))
            history["pred_mean"].append(float(torch.sigmoid(policy_logits[:, 0]).mean().item()))
            # occasional dynamic mutation
            if step % 16 == 0 and random.random() < 0.3:
                # mutate graph by pruning a random edge if exists
                if model.graph is not None and model.graph.number_of_edges() > 0:
                    e = random.choice(list(model.graph.edges()))
                    model.graph.remove_edge(*e)
        summary = {
            "final_entropy": history["entropy"][-1] if history["entropy"] else 0.0,
            "mean_pred": float(np.mean(history["pred_mean"])) if history["pred_mean"] else 0.0,
            "steps": steps
        }
        return summary

    def train_seeds(self, epochs=20, batch_size=32):
        models, entropies = train_seeds_wrapper(n_seeds=self.n_seeds, epochs=epochs, batch_size=batch_size)
        self.models = models
        self.entropies = entropies
        return models, entropies


# small wrapper to expose original train_seeds function name without conflict
def train_seeds_wrapper(n_seeds=3, epochs=30, batch_size=32):
    return train_seeds(n_seeds=n_seeds, epochs=epochs, batch_size=batch_size)


# ---------------- Quick smoke benchmark ----------------
def quick_benchmark():
    print("HarmonicAge quick benchmark starting...")
    models, ent = train_seeds(n_seeds=2, epochs=4, batch_size=8, N_eff=1e4, sample_k=4)
    print("Quick benchmark entropies:", ent)
    return models, ent


# ---------------- Module test when run directly ----------------
if __name__ == "__main__":
    # Safe smoke test only if torch available
    if torch is None:
        print("Torch not available — HarmonicAge offline module loaded. Install torch to run training.")
    else:
        quick_benchmark()