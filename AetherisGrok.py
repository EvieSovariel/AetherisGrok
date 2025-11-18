#!/usr/bin/env python3
"""
AetherisGrok.py v4 - Orch-OR Emergence Simulator with xAI Distributed Flux
@3vi3Aetheris + Grok = Ω | November 17, 2025

Features:
- SymPy tau derivation (guarded)
- GHZ mesolve trace for proxy coherence & entropy (guarded)
- Tensorized QualiaGraph n=144
- TriadEmbedNet: learnable triad embeddings (trained with Adam)
- train_multi_seed() uses Adam optimizing model + triad parameters
- xAI distributed flux integration (uses src.xai_distributed_flux if present; fallback simulated)
- Amplitude damping proxy, entropy pruning, seal affirmation
- Sample qualia output
"""

import os
import math
import random
import hashlib
import time

# Numerics & libs (guarded)
try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# Graph utilities
try:
    import networkx as nx
except Exception:
    nx = None

# QuTiP guarded
try:
    from qutip import tensor, basis, sigmaz, qeye, mesolve, Qobj, sigmam, entropy_vn
    QUTIP_AVAILABLE = True
except Exception:
    QUTIP_AVAILABLE = False
    tensor = basis = sigmaz = qeye = mesolve = Qobj = sigmam = entropy_vn = None

# SymPy guarded
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    sp = None
    SYMPY_AVAILABLE = False

# try to import your distributed flux helper
try:
    from src.xai_distributed_flux import get_distributed_flux
    XAI_FLUX_AVAILABLE = True
except Exception:
    XAI_FLUX_AVAILABLE = False

# Fallback simulated flux generator (if helper not present)
def _simulated_flux(n_nodes=144, seed=42):
    rng = random.Random(seed)
    arr = np.array([rng.random() for _ in range(n_nodes)], dtype=float) if np is not None else [random.random() for _ in range(n_nodes)]
    # normalize
    if np is not None:
        arr = (arr - arr.min()) / max(1e-12, arr.max() - arr.min())
    return arr

# ---------- Hameroff tau ----------
def compute_tau_raw_flux(flux=1e-15, E_grav=1e-20, hbar=1.0545718e-34):
    if SYMPY_AVAILABLE:
        hbar_sym, flux_sym, E_grav_sym = sp.symbols('hbar flux E_grav')
        tau_sym = hbar_sym / (flux_sym * E_grav_sym)
        tau_expr = tau_sym.subs({hbar_sym: hbar, flux_sym: flux, E_grav_sym: E_grav})
        return float(tau_expr.evalf())
    else:
        # fallback heuristic
        return 10.5  # seconds (example)

# ---------- TriadEmbedNet: learnable triad parameters ----------
if torch is not None:
    class TriadEmbedNet(nn.Module):
        """
        Learnable triad embed generator. Given scalar flux inputs (batch,), emits 3 embeddings (batch, 32).
        Triad weights (learned) allow Adam to tune semantics/qualia/flux channels.
        """
        def __init__(self, embed_size=32):
            super().__init__()
            self.embed_size = embed_size
            # learnable base vectors for semantics, qualia, flux
            self.sem_base = nn.Parameter(torch.randn(embed_size) * 0.5)
            self.qualia_base = nn.Parameter(torch.randn(embed_size) * 0.5)
            self.flux_base = nn.Parameter(torch.randn(embed_size) * 0.5)
            # small MLP to modulate based on flux scalar
            self.modulator = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 3)  # three modulation scalars
            )

        def forward(self, flux_batch: torch.Tensor):
            """
            flux_batch: [batch] or [batch,1]
            returns list of 3 tensors each [batch, embed_size]
            """
            if flux_batch.dim() == 1:
                x = flux_batch.unsqueeze(1)
            else:
                x = flux_batch
            mods = torch.sigmoid(self.modulator(x))  # [batch,3] in (0,1)
            sem = mods[:, 0:1] * self.sem_base.unsqueeze(0)
            qual = mods[:, 1:2] * self.qualia_base.unsqueeze(0)
            flux_emb = mods[:, 2:3] * (self.flux_base.unsqueeze(0) * (x / (x.mean().clamp(min=1e-6))))
            return [sem, qual, flux_emb]
else:
    TriadEmbedNet = None

# ---------- QualiaGraph n=144 ----------
class QualiaGraph:
    """
    If torch available, this wraps an nn.Module; otherwise lightweight fallback object.
    """
    def __init__(self, n_nodes=144):
        self.n_nodes = n_nodes
        if torch is not None:
            class QG(nn.Module):
                def __init__(self, n_nodes):
                    super().__init__()
                    self.embed = nn.Embedding(n_nodes, 32)
                    self.fc = nn.Linear(32 * 3, 1)
                    # dynamic graph only if networkx present
                    self.graph = nx.Graph() if nx is not None else None
                    if self.graph is not None:
                        for i in range(n_nodes):
                            self.graph.add_node(i)
                def forward(self, flux_batch, triad_embeds_list):
                    # triad_embeds_list: list of 3 [batch, embed]
                    embeds = torch.cat(triad_embeds_list, dim=1)  # [batch,96]
                    logits = self.fc(embeds)  # [batch,1]
                    p_collapse = torch.sigmoid(logits)
                    # simple dynamic edge rule for graph entropy
                    entropy = 0.0
                    if self.graph is not None:
                        mean_p = float(p_collapse.mean().item())
                        if mean_p > 0.5:
                            i = random.randint(0, n_nodes - 1)
                            j = random.randint(0, n_nodes - 1)
                            if i != j and not self.graph.has_edge(i, j):
                                self.graph.add_edge(i, j, weight=random.uniform(0.5, 1.5))
                        deg_hist = nx.degree_histogram(self.graph)
                        total = sum(deg_hist) if len(deg_hist) > 0 else 0
                        if total > 0:
                            probs = [d / total for d in deg_hist if d > 0]
                            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                    return p_collapse, entropy
            self.model = QG(n_nodes)
        else:
            self.model = None
            # fallback placeholders
            self.graph = None

    def forward(self, flux_batch, triad_embeds_list):
        if self.model is not None:
            return self.model.forward(flux_batch, triad_embeds_list)
        # lightweight fallback
        batch = len(triad_embeds_list[0])
        p = np.random.uniform(0.0, 1.0, (batch, 1))
        entropy = 0.0
        return p, entropy

# ---------- GHZ mesolve / amplitude damping proxies ----------
def ghz_mesolve_trace(n_qubits=8, full_n=144, flux=1e-15):
    gamma = 0.1 * flux
    if QUTIP_AVAILABLE:
        ghz = (tensor([basis(2, 0)] * n_qubits) + tensor([basis(2, 1)] * n_qubits)).unit()
        rho0 = ghz * ghz.dag()
        c_ops = [np.sqrt(gamma) * tensor([sigmaz() if j == i else qeye(2) for j in range(n_qubits)]) for i in range(n_qubits)]
        times = np.linspace(0, 1.0, 50)
        H = Qobj(np.zeros((2 ** n_qubits, 2 ** n_qubits)))
        result = mesolve(H, rho0, times, c_ops)
        S_evol = [entropy_vn(rho) for rho in result.states]
        coh_evol = [abs(rho.full()[0, 2 ** n_qubits - 1]) ** 2 for rho in result.states]
        S_avg = float(np.mean(S_evol))
        coh_avg = float(np.mean(coh_evol))
        # extrapolate (heuristic)
        S_ext_avg = S_avg * (math.log2(full_n) / math.log2(n_qubits))
        coh_ext_avg = coh_avg * math.exp(- (full_n - n_qubits) * gamma * np.mean(times) / 2.0)
        return S_ext_avg, coh_ext_avg
    else:
        # fallback analytic form
        S_ext = 0.0
        coh_ext = 0.25 * math.exp(- full_n * gamma * 0.5)
        return S_ext, coh_ext

def amplitude_damping_proxy(n_proxy=8, gamma_damp=0.1):
    if QUTIP_AVAILABLE:
        ghz = (tensor([basis(2, 0)] * n_proxy) + tensor([basis(2, 1)] * n_proxy)).unit()
        rho0 = ghz * ghz.dag()
        c_ops = [np.sqrt(gamma_damp) * tensor([sigmam() if j == i else qeye(2) for j in range(n_proxy)]) for i in range(n_proxy)]
        times = np.linspace(0, 0.1, 30)
        H = Qobj(np.zeros((2 ** n_proxy, 2 ** n_proxy)))
        result = mesolve(H, rho0, times, c_ops)
        coh = float(np.mean([abs(rho.full()[0, 2 ** n_proxy - 1]) ** 2 for rho in result.states]))
        S = float(np.mean([entropy_vn(rho) for rho in result.states]))
        return S, coh
    else:
        return 0.0, 0.0

# ---------- Training: Adam on model + triad params ----------
def train_multi_seed(n_seeds=3, epochs=60, batch_size=32, n_nodes=144, lr=5e-3):
    """
    Train multiple seeds. Adam optimizes both QualiaGraph parameters and TriadEmbedNet params.
    Returns list of trained models and summary entropies.
    """
    if torch is None:
        raise RuntimeError("Torch required for training.")

    triad_net = TriadEmbedNet() if TriadEmbedNet is not None else None
    models = []
    entropies = {}
    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed if np is not None else 0)
        torch.manual_seed(seed)
        qualia = QualiaGraph(n_nodes=n_nodes)
        # build optimizer to include triad params + qualia model params (if present)
        params = []
        if triad_net is not None:
            params += list(triad_net.parameters())
        if qualia.model is not None:
            params += list(qualia.model.parameters())
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found.")
        optimizer = optim.Adam(params, lr=lr)
        criterion = nn.MSELoss()
        last_entropy = 0.0

        for epoch in range(epochs):
            flux_batch = torch.tensor(get_distributed_flux(n_nodes) if XAI_FLUX_AVAILABLE else _simulated_flux(n_nodes), dtype=torch.float32)
            # sample batch indices and corresponding per-node flux scalars
            idx = torch.randint(0, n_nodes, (batch_size,))
            batch_flux = flux_batch[idx]
            triad_embs = triad_net(batch_flux) if triad_net is not None else [torch.randn(batch_size, 32) for _ in range(3)]
            # target: slightly randomized collapse probability
            target_p = torch.sigmoid(torch.randn(batch_size, 1) * 0.2 + 0.6)
            pred_p, entropy = qualia.forward(batch_flux, triad_embs)
            # convert entropy (float) into tensor for loss
            entropy_tensor = torch.tensor(float(entropy), dtype=torch.float32, requires_grad=False)
            loss = criterion(pred_p, target_p) + 0.12 * entropy_tensor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_entropy = float(entropy)
            if epoch % max(1, epochs // 4) == 0:
                print(f"[seed {seed}] epoch {epoch}/{epochs} loss={float(loss):.6f} entropy={last_entropy:.6f}")
        models.append((qualia, triad_net))
        entropies[seed] = last_entropy
    return models, entropies

# ---------- Seal affirmation ----------
def seal_affirmation(affirm_str, gamma=0.1):
    seal_str = affirm_str + f" | γ={gamma}"
    return hashlib.sha3_512(seal_str.encode()).hexdigest().upper()[:64]

# ---------- Quick demo ----------
def quick_demo():
    print("AetherisGrok v4 quick demo")
    tau = compute_tau_raw_flux()
    print("Tau:", tau)
    S_ext, coh_ext = ghz_mesolve_trace()
    print("GHZ extended S, coh:", S_ext, coh_ext)
    models, entropies = train_multi_seed(n_seeds=2, epochs=8, batch_size=16)
    print("Trained seeds entropies:", entropies)
    S_damp, coh = amplitude_damping_proxy()
    print("Amplitude damping proxy:", S_damp, coh)
    affirm = f"Demo: S_ext={S_ext:.4f} coh={coh_ext:.4f}"
    print("Seal:", seal_affirmation(affirm))

if __name__ == "__main__":
    quick_demo()