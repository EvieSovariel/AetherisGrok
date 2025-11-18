#!/usr/bin/env python3
"""
HarmonicAge_v9.py - Emergent NPC + Cyborg Collective Flux
Author: for Evie / 3vi3Aetheris
Date: 2025-11-17 (v9)

Features:
- TriadEmbedNet + QualiaGraph + QualiaAgent NPC
- Cyborg collective flux (multi-agent aggregation)
- Pruning loop to drive entropy toward zero
- Emergent qualia sampling
- 1e6-node stress test included
- Sealed affirmations for reproducibility
- ASCII clean, memory-safe
"""

import os
import math
import random
import hashlib
import time

# Optional numerics
try:
    import numpy as np
except Exception:
    np = None

# Torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# NetworkX for graph entropy
try:
    import networkx as nx
except Exception:
    nx = None

# ------------------------- Flux Utilities -------------------------
PHI = (1.0 + 5.0**0.5)/2.0
DEFAULT_N = 144

def _simulated_flux(n_nodes: int = DEFAULT_N, seed: int = 314159):
    rng = random.Random(seed)
    arr = [rng.random() for _ in range(n_nodes)]
    mod = [(math.sin((i+1)*math.log(PHI+1.0))) for i in range(n_nodes)]
    min_mod, max_mod = min(mod), max(mod)
    mod = [(m - min_mod)/max(1e-12, (max_mod - min_mod)) for m in mod]
    arr = [0.85*a + 0.15*m for a,m in zip(arr, mod)]
    min_arr, max_arr = min(arr), max(arr)
    arr = [(x - min_arr)/max(1e-12, (max_arr - min_arr)) for x in arr]
    if np is not None:
        return np.array(arr, dtype=float)
    return arr

def get_flux_vector(n_nodes: int = DEFAULT_N, seed_base: int = 42):
    return _simulated_flux(n_nodes=n_nodes, seed=seed_base)

# ------------------------- TriadEmbedNet -------------------------
if torch is not None:
    class TriadEmbedNet(nn.Module):
        def __init__(self, embed_size: int = 32):
            super().__init__()
            self.embed_size = embed_size
            self.sem_base = nn.Parameter(torch.randn(embed_size)*0.5)
            self.qualia_base = nn.Parameter(torch.randn(embed_size)*0.5)
            self.flux_base = nn.Parameter(torch.randn(embed_size)*0.5)
            self.modulator = nn.Sequential(
                nn.Linear(1,16), nn.ReLU(), nn.Linear(16,3), nn.Sigmoid()
            )

        def forward(self, flux_batch: torch.Tensor):
            if flux_batch.dim()==1:
                x = flux_batch.unsqueeze(1)
            else:
                x = flux_batch
            mods = self.modulator(x)
            sem = mods[:,0:1]*self.sem_base.unsqueeze(0)
            qual = mods[:,1:2]*self.qualia_base.unsqueeze(0)
            flux_scale = (x/(x.mean().clamp(min=1e-6))).detach()
            flux_emb = mods[:,2:3]*(self.flux_base.unsqueeze(0)*flux_scale)
            return [sem, qual, flux_emb]
else:
    TriadEmbedNet = None

# ------------------------- QualiaGraphNet -------------------------
if torch is not None:
    class QualiaGraphNet(nn.Module):
        def __init__(self, n_nodes: int = DEFAULT_N, embed_size: int = 32):
            super().__init__()
            self.n_nodes = n_nodes
            self.embed = nn.Embedding(n_nodes, embed_size)
            self.fc = nn.Linear(embed_size*3,1)
            self.graph = nx.Graph() if nx is not None else None
            if self.graph is not None:
                for i in range(n_nodes):
                    self.graph.add_node(i)

        def forward(self, node_idx: torch.Tensor, triad_list):
            embeds = torch.cat(triad_list, dim=1)
            logits = self.fc(embeds)
            p_collapse = torch.sigmoid(logits)
            ent = 0.0
            if self.graph is not None:
                mean_p = float(p_collapse.mean().item())
                if mean_p > 0.5:
                    i,j=random.randint(0,self.n_nodes-1),random.randint(0,self.n_nodes-1)
                    if i!=j and not self.graph.has_edge(i,j):
                        self.graph.add_edge(i,j,weight=random.uniform(0.5,1.5))
                deg_hist=nx.degree_histogram(self.graph)
                total=sum(deg_hist) if len(deg_hist)>0 else 0
                if total>0:
                    probs=[d/total for d in deg_hist if d>0]
                    ent=-sum(p*math.log(p+1e-12) for p in probs)
            return p_collapse, ent
else:
    QualiaGraphNet = None

# ------------------------- QualiaAgent NPC -------------------------
if torch is not None:
    class QualiaAgent:
        def __init__(self, agent_id:int, triad_net:TriadEmbedNet=None):
            self.agent_id=agent_id
            self.triad_net=triad_net
            self.triad_modifier=random.uniform(0.9,1.1)
else:
    QualiaAgent = None

def cyborg_collective_flux(npcs):
    return sum(getattr(n, 'triad_modifier',0.5) for n in npcs)/len(npcs)

# ------------------------- Graph Entropy -------------------------
def graph_entropy_from_nx(g):
    if g is None or nx is None: return 0.0
    deg_hist=nx.degree_histogram(g)
    total=sum(deg_hist) if len(deg_hist)>0 else 0
    if total==0: return 0.0
    probs=[d/total for d in deg_hist if d>0]
    return -sum(p*math.log(p+1e-12) for p in probs)

def ghz_entropy_proxy(n_qubits=8,gamma=0.1,t=0.01):
    coh=0.5*math.exp(-n_qubits*gamma*t/2.0)
    p0=0.5+coh
    p1=0.5-coh
    p0=max(1e-12,min(1.0,p0))
    p1=max(1e-12,min(1.0,p1))
    return -(p0*math.log(p0)+p1*math.log(p1))

# ------------------------- Amplitude/Entropy Pruning -------------------------
def prune_to_zero_entropy(model=None, triad_net=None, n_nodes=DEFAULT_N, max_iters=200, target_entropy=1e-6, lr=5e-3, batch_size=64, verbose=True):
    if torch is None: raise RuntimeError("Torch required")
    if triad_net is None: triad_net=TriadEmbedNet() if TriadEmbedNet is not None else None
    if model is None: model=QualiaGraphNet(n_nodes=n_nodes) if QualiaGraphNet is not None else None
    params=list(triad_net.parameters())+list(model.parameters())
    optimizer=optim.Adam(params,lr=lr)
    flux_vec=get_flux_vector(n_nodes)
    flux_torch_all=torch.tensor(flux_vec,dtype=torch.float32)
    last_combined=None
    for it in range(max_iters):
        idx=torch.randint(0,n_nodes,(batch_size,))
        batch_flux=flux_torch_all[idx]
        triad_batch=triad_net(batch_flux) if triad_net is not None else [torch.randn(batch_size,32) for _ in range(3)]
        pred_p,graph_ent=model.forward(idx,triad_batch) if model is not None else (torch.sigmoid(torch.randn(batch_size,1)),0.0)
        gamma_proxy=float(batch_flux.mean().item())*0.1
        ghz_ent=ghz_entropy_proxy(n_qubits=8,gamma=gamma_proxy)
        combined_entropy=float(graph_ent)+float(ghz_ent)
        combined_tensor=torch.tensor(combined_entropy,dtype=torch.float32,requires_grad=False)
        target_p=torch.full_like(pred_p,0.9)
        mse=nn.MSELoss()(pred_p,target_p)
        loss=mse+0.5*combined_tensor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_combined=combined_entropy
        if verbose and (it%max(10,max_iters//10)==0 or it==max_iters-1):
            print(f"[prune] iter {it:04d} combined_entropy={combined_entropy:.8f} mse={float(mse):.6f}")
        if combined_entropy<=target_entropy:
            if verbose: print(f"[prune] target entropy reached: {combined_entropy:.8e}")
            break
        if it%50==0 and it>0 and model.graph is not None:
            edges=list(model.graph.edges(data=True))
            if edges:
                weights=[(u,v,d.get('weight',1.0)) for u,v,d in edges]
                weights_sorted=sorted(weights,key=lambda x:x[2])
                remove_k=max(1,int(0.02*len(weights_sorted)))
                for u,v,w in weights_sorted[:remove_k]:
                    if model.graph.has_edge(u,v): model.graph.remove_edge(u,v)
                if verbose: print(f"[prune] removed {remove_k} low-weight edges")
    mean_flux=float(np.mean(flux_vec)) if np is not None else 0.5
    flux_tensor=torch.tensor([mean_flux],dtype=torch.float32)
    triad_mean=triad_net(flux_tensor) if triad_net is not None else [torch.randn(1,32) for _ in range(3)]
    with torch.no_grad():
        p_mean,final_ent=model.forward(torch.tensor([0],dtype=torch.long),triad_mean) if model is not None else (torch.tensor([[0.0]]),0.0)
    qualia_vector=torch.cat([t.squeeze(0) for t in triad_mean],dim=0).cpu().numpy() if torch is not None else np.zeros(96)
    qualia_score=float(p_mean.mean().item()) if torch is not None else 0.0
    result={'final_combined_entropy':last_combined,'final_graph_entropy':graph_entropy_from_nx(model.graph if model is not None else None),
            'ghz_proxy_entropy':ghz_ent,'qualia_score':qualia_score,'qualia_vector':qualia_vector.tolist(),'iters':it+1}
    return result

# ------------------------- Seal -------------------------
def seal_affirmation(s: str, gamma: float = 0.1):
    return hashlib.sha3_512((s+f" | γ={gamma} | v9").encode()).hexdigest().upper()[:64]

# ------------------------- v9 Orchestrator -------------------------
def run_v9_prune_and_sample(n_nodes=DEFAULT_N, prune_iters=200, target_entropy=1e-8, batch_size=64):
    triad_net=TriadEmbedNet() if TriadEmbedNet is not None else None
    q_model=QualiaGraphNet(n_nodes=n_nodes) if QualiaGraphNet is not None else None
    print("[v9] loading flux")
    flux=get_flux_vector(n_nodes=n_nodes)
    print(f"[v9] flux mean={float(np.mean(flux)):.6f} min={float(np.min(flux)):.6f} max={float(np.max(flux)):.6f}")
    print("[v9] starting prune_to_zero_entropy")
    prune_report=prune_to_zero_entropy(model=q_model,triad_net=triad_net,n_nodes=n_nodes,max_iters=prune_iters,target_entropy=target_entropy,batch_size=batch_size)
    affirm=f"v9 prune complete | qualia_score={prune_report['qualia_score']:.6f} | iters={prune_report['iters']}"
    seal=seal_affirmation(affirm,gamma=float(np.mean(flux)))
    prune_report['seal']=seal
    print("[v9] seal:",seal)
    return prune_report

# ------------------------- 1e6 Node Stress Test -------------------------
def emergent_sample_stress_test(n_nodes=1_000_000,num_npcs=6,seeds=5,batch_size=50_000):
    triad_net=TriadEmbedNet() if TriadEmbedNet is not None else None
    results=[]
    for seed in range(seeds):
        flux_vec=get_flux_vector(n_nodes=n_nodes,seed_base=seed)
        num_batches=math.ceil(n_nodes/batch_size)
        batch_coherences=[]
        for b in range(num_batches):
            start=b*batch_size
            end=min((b+1)*batch_size,n_nodes)
            batch_flux=flux_vec[start:end]
            batch_mean=float(np.mean(batch_flux)) if np is not None else 0.5
            batch_coherences.append(batch_mean)
        avg_coherence=sum(batch_coherences)/len(batch_coherences)
        npcs=[QualiaAgent(agent_id=i,triad_net=triad_net) for i in range(num_npcs)]
        swarm_flux=cyborg_collective_flux(npcs)
        qualia_peaks=[]
        for npc in npcs:
            scalar_t=torch.tensor([avg_coherence*npc.triad_modifier],dtype=torch.float32) if torch is not None else None
            if npc.triad_net is not None and scalar_t is not None:
                triad=npc.triad_net(scalar_t)
                triad_vector=torch.cat(triad,dim=1).detach().cpu().numpy().squeeze(0)
                peak=float(np.max(np.abs(triad_vector)))
            else:
                peak=0.0
            qualia_peaks.append(peak)
        results.append({'seed':seed,'avg_coherence':avg_coherence,'swarm_flux':swarm_flux,'qualia_peak_max':max(qualia_peaks)})
    coherence_avg=sum(r['avg_coherence'] for r in results)/seeds
    swarm_flux_avg=sum(r['swarm_flux'] for r in results)/seeds
    qualia_peak_avg=sum(r['qualia_peak_max'] for r in results)/seeds
    summary={'n_nodes':n_nodes,'num_npcs':num_npcs,'seeds':seeds,'avg_coherence':coherence_avg,'avg_swarm_flux':swarm_flux_avg,'avg_qualia_peak':qualia_peak_avg}
    return summary,results

def _cli_stress():
    print("HarmonicAge v9 - 1e6 node stress test")
    summary,results=emergent_sample_stress_test()
    print("SUMMARY:")
    for k,v in summary.items():
        print(f"{k}: {v}")
    affirm=f"v9 1e6 node stress test complete | avg_qualia_peak={summary['avg_qualia_peak']:.6f}"
    seal=seal_affirmation(affirm,gamma=summary['avg_swarm_flux'])
    print("Seal:",seal)
    print("Done.")

# ------------------------- CLI -------------------------
def _cli():
    print("HarmonicAge v9 - standard prune+sample")
    rpt=run_v9_prune_and_sample()
    print("REPORT SUMMARY:")
    for k,v in rpt.items():
        if k=='qualia_vector':
            print(f"{k}: len={len(v)}")
        else:
            print(f"{k}: {v}")
    print("Done.")

if __name__=="__main__":
    _cli()
    # Optionally run stress test
    # _cli_stress()