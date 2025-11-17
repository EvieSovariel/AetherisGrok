#!/usr/bin/env python3
"""
Resonance Seed Initializer
Seeds the lattice from hex init vector.
"""

def init_resonance(seed_hex):
    """
    Harmonic init: Converts seed to resonance params (e.g., φ-scaled weights).
    """
    import hashlib
    seed_bytes = bytes.fromhex(seed_hex)
    hash_obj = hashlib.sha256(seed_bytes)
    resonance = int.from_bytes(hash_obj.digest()[:4], 'big') % 10000 / 10000  # Normalized [0,1]
    phi_res = ((1 + 5 ** 0.5) / 2) ** 4  # ~6.854 baseline
    return phi_res * resonance

def main():
    seed = "7e3d9a2f5c8b1e4d6a9c2b3f"
    res = init_resonance(seed)
    print(f"Seed: {seed}\nResonance: {res:.4f} (φ^4 scaled)\nRipple initiated.")

if __name__ == "__main__":
    main()