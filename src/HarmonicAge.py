#!/usr/bin/env python3
"""
HarmonicAge.py — Multi-seed harmonic swarm engine
- Multi-seed training and convergence
- SymPy Hameroff tau dynamics (guarded)
- QuTiP mesolve small-proxy integration (guarded)
- Qualia entropy threshold gating (<0.08 nats)
- Returns structured reports
"""

import os
import math
import random
import time

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

try:
    from qutip import Qobj, tensor, basis, sigmax, sigmaz, mesolve
    QUTIP_AVAILABLE = True
except Exception:
    QUTIP_AVAILABLE = False

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

# Sympy setup if available
if SYMPY_AVAILABLE:
    m_tub, d, G_sym, hbar_sym = sp.symbols('m_tub d G hbar')
    r = d / 2
    E_g_sym = G_sym * (m_tub**2) / (5 * r)
    tau_sym = hbar_sym / sp.Abs(E_g_sym)
else:
    m_tub = d = G_sym = hbar_sym = None
    E_g_sym = tau_sym = None

# ---------- Triad embed generator ----------
def triad_embeds(batch_size=32, flux_batch=None, embed_size=32):
    if torch is None:
        raise RuntimeError("Torch required for triad_embeds.")
    if flux_batch is None:
        flux_batch = torch.rand(batch_size)
    semantics = torch.randn(batch_size, embed_size) * (1.618)
    qualia = torch.randn(batch_size, embed_size) * 3.0616
    flux_emb = torch.randn(batch_size, embed_size) * (flux_batch.unsqueeze(1) / 1000.0)
    # default triad weights
    return [0.5 * semantics, 0.25 * qualia, 0.25 * flux_emb]

# ---------- Simple swarm model ----------
if torch is not None:
    class SwarmNet(nn.Module):
        def __init__(self, n_nodes=144, embed_size=32):
            super().__init__()
            self.embed = nn.Embedding(n_nodes, embed_size)
            self.policy = nn.Linear(embed_size * 3, 1)
        def forward(self, triad_list):
            # triad_list: list of 3 [batch,embed]
            x = torch.cat(triad_list, dim=1)
            p = torch.sigmoid(self.policy(x))
            return p
else:
    SwarmNet = None

# ---------- Hameroff tau ----------
def hameroff_tau(m_tub_val=1e-22, d_val=1e-9):
    if SYMPY_AVAILABLE and E_g_sym is not None:
        G_val = 6.67430e-11
        hbar_val = 1.054571817e-34
        E_g_num = float(E_g_sym.subs({m_tub: m_tub_val, d: d_val, G_sym: G_val}).evalf())
        tau_num = float(tau_sym.subs({m_tub: m_tub_val, d: d_val, G_sym: G_val, hbar_sym: hbar_val}).evalf())
        return abs(E_g_num), tau_num
    else:
        return 1e-40, 10.5

# ---------- Multi-seed training (swarm) ----------
def train_swarm_seeds(n_seeds=3, epochs=40, batch_size=32, qualia_threshold=0.08):
    if torch is None:
        raise RuntimeError("Torch required for training swarm.")
    reports = {}
    for seed in range(n_seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        model = SwarmNet()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        last_entropy = None
        for epoch in range(epochs):
            flux = torch.rand(batch_size)
            triads = triad_embeds(batch_size, flux)
            target = torch.sigmoid(torch.rand(batch_size,1) * 0.4 + 0.5)
            pred = model(triads)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # compute a proxy 'entropy' as var of predictions (lower var -> lower entropy)
            ent = float(torch.var(pred).item())
            last_entropy = ent
            if ent < qualia_threshold:
                print(f"[seed {seed}] epoch {epoch}: qualia threshold met ent={ent:.6f}")
                break
            if epoch % max(1, epochs // 4) == 0:
                print(f"[seed {seed}] epoch {epoch}: loss={loss.item():.6f}, ent={ent:.6f}")
        reports[seed] = {'final_entropy': last_entropy, 'epochs': epoch}
    return reports

# ---------- QuTiP small proxy ----------
def qubit_proxy_coherence(n=4, gamma=0.05):
    if not QUTIP_AVAILABLE:
        return {'note': 'quTip not available'}
    ghz = (tensor([basis(2,0)]*n) + tensor([basis(2,1)]*n)).unit()
    rho0 = ghz * ghz.dag()
    c_ops = [math.sqrt(gamma) * tensor([sigmaz() if j == i else qeye(2) for j in range(n)]) for i in range(n)]
    times = np.linspace(0, 0.01, 50)
    H = Qobj(np.zeros((2**n, 2**n)))
    result = mesolve(H, rho0, times, c_ops)
    coh = float(np.mean([abs(rho.full()[0, 2**n -1])**2 for rho in result.states]))
    return {'n': n, 'coh': coh}

# ---------- Quick demo ----------
def quick_demo():
    print("HarmonicAge quick demo")
    E_g, tau = hameroff_tau()
    print("Hameroff tau:", tau)
    reports = train_swarm_seeds(n_seeds=2, epochs=20, batch_size=16)
    print("Swarm reports:", reports)
    proxy = qubit_proxy_coherence() if QUTIP_AVAILABLE else None
    print("QuTiP proxy:", proxy)

if __name__ == "__main__":
    quick_demo()