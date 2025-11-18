# src/AetherisGrok_cyborg_v2.py
# ASCII-only, Python 3.10+ recommended
# AetherisGrok Cyborg v2
# - virtual 1e6 nodes (sparse active subgraphs)
# - Hameroff tau numeric fusion
# - Grok-4 video flux hook (safe fallback simulated)
# - Emergent NPC sampling helper
# - Uses xai_distributed_flux if present, else simulated flux

from __future__ import annotations
import os
import math
import random
import time
import typing
import hashlib

# Core numeric libs (guarded)
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

# networkx used only for small active subgraph utilities
try:
    import networkx as nx
except Exception:
    nx = None

# Optional distributed flux helper if you added it earlier
try:
    from src.xai_distributed_flux import get_distributed_flux
    XAI_FLUX_AVAILABLE = True
except Exception:
    get_distributed_flux = None
    XAI_FLUX_AVAILABLE = False

# Optional video libs placeholder (we do not import heavy AV libs here)
# The Grok-4 hook expects a (batch, C, H, W) torch tensor if available

# ---------------------------
# Configuration / Defaults
# ---------------------------
DEFAULT_NODES = 1_000_000  # 1e6 virtual nodes
ACTIVE_SUBGRAPH_SIZE = 2048  # number of nodes we materialize per epoch (tunable)
TRIAD_EMBED_SIZE = 32
NPC_COUNT = 6  # emergent NPCs to sample in emergent_npc_sample
DEVICE = "cpu"

# Hameroff constants (SI units where applicable)
G_CONST = 6.67430e-11
HBAR = 1.054571817e-34

# ---------------------------
# Utilities
# ---------------------------
def _simulated_flux_vector(n_nodes: int, seed: int = 42) -> list:
    """Deterministic simulated flux vector (not full length returned for efficiency).
    For full-length calls, use get_flux_vector which may call distributed helper."""
    rng = random.Random(seed)
    # return a list of pseudo-random floats in [0,1]
    return [rng.random() for _ in range(n_nodes)]

def get_flux_vector(n_nodes: int = DEFAULT_NODES, mode: str = "auto", seed_base: typing.Optional[int] = None):
    """High-level flux vector getter.
    If distributed helper is available, attempt to get aggregated flux, else simulate.
    For memory safety with n_nodes large, this function returns a generator-like object
    (numpy array only if n_nodes is small or caller requests).
    """
    if XAI_FLUX_AVAILABLE and get_distributed_flux is not None:
        try:
            arr = get_distributed_flux(n_nodes=n_nodes, mode=mode, seed_base=seed_base)
            # get_distributed_flux may return numpy array; ensure proper type and length
            if arr is None or (hasattr(arr, "size") and arr.size == 0):
                # fallback to simulated generator
                return (_ for _ in _simulated_flux_vector(n_nodes, seed=(seed_base or 42)))
            if isinstance(arr, np.ndarray):
                # if user requested full array and has memory, return it
                return arr
            # otherwise, convert to list
            return list(arr)
        except Exception:
            return (_ for _ in _simulated_flux_vector(n_nodes, seed=(seed_base or 42)))
    # default fallback: generator to avoid allocating full 1e6 array
    return (_ for _ in _simulated_flux_vector(n_nodes, seed=(seed_base or 42)))

# ---------------------------
# Hameroff tau routine
# ---------------------------
def hameroff_tau_numeric(m_tub: float = 1e-22, d: float = 1e-9):
    """Compute gravitational self-energy E_g and collapse time tau for a tubulin mass.
    E_g = G * m^2 / (5 * r) where r = d/2 (approx for sphere-like)
    tau = hbar / |E_g|
    Returns tuple (E_g, tau) as floats.
    """
    r = d / 2.0
    # guard for zero division
    if r <= 0.0:
        raise ValueError("d must be positive")
    E_g = G_CONST * (m_tub ** 2) / (5.0 * r)
    tau = HBAR / abs(E_g) if E_g != 0.0 else float("inf")
    return float(E_g), float(tau)

# ---------------------------
# Grok-4 video flux hook
# ---------------------------
def grok4_video_flux_from_tensor(video_tensor: typing.Any):
    """Accepts a torch tensor shaped (batch, C, H, W) or None.
    Returns a scalar video flux in [0,1] representing motion/salience.
    If video_tensor is None or invalid, returns a simulated value.
    """
    try:
        if video_tensor is None:
            return random.random() * 0.25  # small default contribution
        # if torch available and tensor looks sane:
        if torch is not None and hasattr(video_tensor, "mean"):
            # simple motion proxy: mean absolute difference across channels
            # this is a lightweight proxy; replace with Grok-4 model when available
            frame_mean = float(video_tensor.mean().item())
            # map mean to [0,1] via sigmoid-like mapping
            v = 1.0 / (1.0 + math.exp(- (frame_mean - 0.5) * 6.0))
            return float(v)
        # fallback numeric
        return float(np.mean(video_tensor)) if np is not None else random.random() * 0.25
    except Exception:
        return random.random() * 0.25

# ---------------------------
# Triad Embed Net (scalable)
# ---------------------------
if torch is not None:
    class TriadEmbedNet(nn.Module):
        def __init__(self, embed_size: int = TRIAD_EMBED_SIZE):
            super().__init__()
            self.embed_size = embed_size
            # small learnable bases
            self.sem_base = nn.Parameter(torch.randn(embed_size) * 0.1)
            self.qualia_base = nn.Parameter(torch.randn(embed_size) * 0.1)
            self.flux_base = nn.Parameter(torch.randn(embed_size) * 0.1)
            # modulator for flux scalar
            self.mod = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 3), nn.Sigmoid())

        def forward(self, flux_scalar: torch.Tensor):
            # flux_scalar shape [batch] or [batch,1]
            if flux_scalar.dim() == 1:
                x = flux_scalar.unsqueeze(1)
            else:
                x = flux_scalar
            mods = self.mod(x)  # [batch,3]
            sem = mods[:, 0:1] * self.sem_base.unsqueeze(0)
            qual = mods[:, 1:2] * self.qualia_base.unsqueeze(0)
            # scale flux base by normalized flux scalar
            mean_scalar = x.mean().clamp(min=1e-6)
            flux_emb = mods[:, 2:3] * (self.flux_base.unsqueeze(0) * (x / mean_scalar))
            return [sem, qual, flux_emb]
else:
    TriadEmbedNet = None

# ---------------------------
# Sparse active subgraph container
# ---------------------------
class ActiveSubgraph:
    """A lightweight active subgraph representation for a small subset of nodes.
    This avoids materializing the full 1e6 node graph.
    It keeps node ids, edges, and weights for nodes currently active.
    """
    def __init__(self, capacity: int = ACTIVE_SUBGRAPH_SIZE):
        self.capacity = capacity
        # we maintain nodes as integers in [0, DEFAULT_NODES)
        self.nodes = []  # list of node ids active
        self.edges = {}  # dict of (u,v) -> weight
        self.node_set = set()

    def seed_random(self, n_nodes: int = DEFAULT_NODES, rng_seed: int = None):
        rng = random.Random(rng_seed)
        self.nodes = []
        self.node_set = set()
        # sample unique node ids
        while len(self.nodes) < min(self.capacity, n_nodes):
            nid = rng.randrange(0, n_nodes)
            if nid not in self.node_set:
                self.nodes.append(nid)
                self.node_set.add(nid)
        self.edges = {}

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        if u == v:
            return
        if u not in self.node_set or v not in self.node_set:
            return
        key = (min(u, v), max(u, v))
        self.edges[key] = weight

    def prune_low_weight_edges(self, fraction: float = 0.02):
        if not self.edges:
            return 0
        items = sorted(self.edges.items(), key=lambda x: x[1])
        k = max(1, int(len(items) * fraction))
        for i in range(k):
            key = items[i][0]
            del self.edges[key]
        return k

    def graph_entropy(self):
        # degree histogram based entropy
        if nx is None:
            # fallback approximate entropy based on number of edges
            m = len(self.edges)
            if m == 0:
                return 0.0
            return math.log(1 + m)
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for (u, v), w in self.edges.items():
            G.add_edge(u, v, weight=w)
        deg_hist = nx.degree_histogram(G)
        total = sum(deg_hist) if deg_hist else 0
        if total == 0:
            return 0.0
        probs = [d / total for d in deg_hist if d > 0]
        return -sum(p * math.log(p + 1e-12) for p in probs)

# ---------------------------
# Emergent NPC agent
# ---------------------------
class QualiaAgent:
    """Emergent NPC that senses flux + video, computes triad, and acts with a tiny policy net."""
    def __init__(self, agent_id: int = 0, n_nodes: int = DEFAULT_NODES, triad_net: typing.Optional[TriadEmbedNet] = None):
        self.agent_id = agent_id
        self.n_nodes = n_nodes
        self.position = random.randrange(0, max(1, n_nodes))
        self.triad_net = triad_net if triad_net is not None else (TriadEmbedNet() if TriadEmbedNet is not None else None)
        # tiny policy network mapping triad concat to action prob
        if torch is not None:
            self.policy = nn.Sequential(nn.Linear(TRIAD_EMBED_SIZE * 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        else:
            self.policy = None
        self.history = []

    def sense(self, flux_vector_generator, video_tensor: typing.Any = None):
        """Sense local scalar flux and video flux influence for the current position.
        flux_vector_generator may be an iterable/generator or a numpy array.
        We treat flux_vector_generator as a sequence-like for sampling."""
        # get local flux (best-effort)
        local_flux = 0.5
        try:
            if isinstance(flux_vector_generator, (list, tuple, np.ndarray)):
                idx = self.position % len(flux_vector_generator)
                local_flux = float(flux_vector_generator[idx])
            else:
                # generator: advance a few and pick one deterministically
                # replicate generator by seeding simulation for this agent
                # note: heavy generators should not be consumed here
                local_flux = random.random()
        except Exception:
            local_flux = random.random()
        # video flux proxy
        video_flux = grok4_video_flux_from_tensor(video_tensor)
        # combined scalar
        combined = float(0.6 * local_flux + 0.4 * video_flux)
        return combined

    def perceive_and_act(self, flux_vector_generator, video_tensor: typing.Any = None):
        scalar = self.sense(flux_vector_generator, video_tensor)
        # build triad
        if self.triad_net is not None and torch is not None:
            scalar_t = torch.tensor([scalar], dtype=torch.float32)
            triad = self.triad_net(scalar_t)  # list of 3 tensors [1,embed]
            triad_concat = torch.cat(triad, dim=1)
            # action probability
            p = float(self.policy(triad_concat).item()) if self.policy is not None else float(torch.sigmoid(torch.randn(1)).item())
            # compute a local entropy proxy: variance across triad channels
            arr = triad_concat.detach().cpu().numpy().squeeze(0)
            ent = float(np.var(arr))
            self.history.append({'pos': self.position, 'p': p, 'ent': ent, 'scalar': scalar})
            # act: random-walk influenced by p
            step = int(max(1, round((p - 0.5) * 10)))
            if random.random() < 0.5:
                self.position = (self.position + step) % self.n_nodes
            else:
                self.position = (self.position - step) % self.n_nodes
            return {'p_collapse': p, 'entropy': ent, 'position': self.position}
        else:
            # fallback random behavior
            p = random.random()
            ent = 0.0
            self.history.append({'pos': self.position, 'p': p, 'ent': ent, 'scalar': scalar})
            self.position = (self.position + int((p - 0.5) * 10)) % self.n_nodes
            return {'p_collapse': p, 'entropy': ent, 'position': self.position}

# ---------------------------
# Emergent NPC sample generator
# ---------------------------
def emergent_npc_sample(num_npcs: int = NPC_COUNT, sample_steps: int = 8, n_nodes: int = DEFAULT_NODES):
    """Create a handful of NPCs, run sample_steps of sense+act and return a concise report."""
    # prepare flux generator (do not materialize full million vector)
    flux_gen = get_flux_vector(n_nodes=n_nodes, mode="auto", seed_base=1234)
    # instantiate triad net shared across NPCs (lightweight)
    triad_net = TriadEmbedNet() if TriadEmbedNet is not None else None
    # create NPCs
    npcs = [QualiaAgent(agent_id=i, n_nodes=n_nodes, triad_net=triad_net) for i in range(num_npcs)]
    report = []
    # small active subgraph to track edges among nodes touched
    subgraph = ActiveSubgraph(capacity=ACTIVE_SUBGRAPH_SIZE)
    subgraph.seed_random(n_nodes=n_nodes, rng_seed=42)
    for step in range(sample_steps):
        # optional video tensor stub (none) - replace with actual frame tensor for real runs
        video_tensor = None
        for npc in npcs:
            res = npc.perceive_and_act(flux_gen, video_tensor)
            # occasionally add edges in active subgraph influenced by p
            p = res['p_collapse']
            if p > 0.6:
                u = random.choice(subgraph.nodes)
                v = random.choice(subgraph.nodes)
                subgraph.add_edge(u, v, weight=0.5 + p)
            report.append({'agent': npc.agent_id, 'step': step, 'p': res['p_collapse'], 'ent': res['entropy'], 'pos': res['position']})
        # periodic pruning in subgraph
        if step % 4 == 0:
            subgraph.prune_low_weight_edges(fraction=0.02)
    # aggregate stats
    p_vals = [r['p'] for r in report]
    ent_vals = [r['ent'] for r in report]
    res_summary = {
        'num_npcs': num_npcs,
        'steps': sample_steps,
        'p_mean': float(np.mean(p_vals)) if np is not None else sum(p_vals) / len(p_vals),
        'p_std': float(np.std(p_vals)) if np is not None else 0.0,
        'ent_mean': float(np.mean(ent_vals)) if np is not None else 0.0,
        'nodes_touched': len(subgraph.nodes),
        'edges_in_subgraph': len(subgraph.edges)
    }
    return res_summary, report, subgraph

# ---------------------------
# Quick scaled test runner (safe)
# ---------------------------
def quick_scaled_test():
    print("Running quick_scaled_test for AetherisGrok_cyborg_v2")
    # Hameroff tau demonstration
    Eg, tau = hameroff_tau_numeric(m_tub=1e-22, d=1e-9)
    print("Hameroff E_g:", Eg, "tau (s):", tau)
    # emergent NPC sample (small, fast)
    summary, report, subgraph = emergent_npc_sample(num_npcs=4, sample_steps=6, n_nodes=1000000)
    print("Emergent NPC summary:", summary)
    # triad potential example
    if TriadEmbedNet is not None:
        triad = TriadEmbedNet()
        with torch.no_grad():
            triad_vec = triad(torch.tensor([0.5], dtype=torch.float32))
        # flatten triad vector approx magnitude
        mag = 0.0
        try:
            arr = torch.cat(triad_vec, dim=1).squeeze(0).cpu().numpy()
            mag = float(np.linalg.norm(arr))
        except Exception:
            mag = 0.0
        print("Triad potential magnitude:", mag)
    # simple seal
    seal_input = "AetherisGrok_cyborg_v2 test " + str(time.time())
    seal = hashlib.sha3_256(seal_input.encode()).hexdigest().upper()
    print("Seal sample (truncated):", seal[:24])
    return summary

if __name__ == "__main__":
    quick_scaled_test()