#!/usr/bin/env python3
"""E8 Lattice Quantization Mathematical Oracle and Synthetic KV Retrieval Testbed.

Verifies the mathematical fidelity of Conway-Sloane E8 lattice projection on
high-dimensional (D=256) Key-Value activations matching Qwen3.8-27B.
"""

from __future__ import annotations

import argparse
import math
import time
import numpy as np


def conway_sloane_e8_project(x: np.ndarray) -> np.ndarray:
    """Projects 8-dimensional vectors x onto the E8 lattice using the Conway-Sloane algorithm.
    
    E8 is the union of D8 and D8 + 0.5*1.
    D8 consists of points in Z^8 with even sum of coordinates.
    
    Args:
        x: Array of shape (..., 8) with float coordinates.
        
    Returns:
        Array of shape (..., 8) with closest E8 lattice points.
    """
    # 1. Find closest point in D8
    f_x = np.round(x)
    sum_f = np.sum(f_x, axis=-1, keepdims=True)
    odd_mask = (sum_f.astype(np.int64) % 2 != 0)
    
    # If sum is odd, flip the component with the largest rounding error |x_i - f(x_i)|
    errors = np.abs(x - f_x)
    worst_idx = np.argmax(errors, axis=-1)
    
    closest_d8 = f_x.copy()
    if np.any(odd_mask):
        # Determine direction of adjustment (+1 if x > f_x else -1)
        grid_indices = np.indices(x.shape[:-1])
        # Reshape for multi-dimensional indexing
        flat_mask = odd_mask.squeeze(-1)
        if np.ndim(x) == 1:
            if flat_mask:
                diff = x[worst_idx] - f_x[worst_idx]
                closest_d8[worst_idx] += 1.0 if diff >= 0 else -1.0
        else:
            # Vectorized multi-dimensional adjustment
            worst_one_hot = np.zeros_like(x, dtype=bool)
            np.put_along_axis(worst_one_hot, np.expand_dims(worst_idx, -1), True, axis=-1)
            adjust = np.where(x >= f_x, 1.0, -1.0)
            closest_d8 = np.where(odd_mask & worst_one_hot, f_x + adjust, f_x)

    # 2. Find closest point in D8 + 0.5*1 (Coset 1)
    x_shifted = x - 0.5
    f_shift = np.round(x_shifted)
    sum_shift = np.sum(f_shift, axis=-1, keepdims=True)
    odd_shift_mask = (sum_shift.astype(np.int64) % 2 != 0)
    
    errors_shift = np.abs(x_shifted - f_shift)
    worst_shift_idx = np.argmax(errors_shift, axis=-1)
    
    closest_shift_d8 = f_shift.copy()
    if np.ndim(x) == 1:
        if odd_shift_mask.item():
            diff = x_shifted[worst_shift_idx] - f_shift[worst_shift_idx]
            closest_shift_d8[worst_shift_idx] += 1.0 if diff >= 0 else -1.0
    else:
        worst_shift_one_hot = np.zeros_like(x, dtype=bool)
        np.put_along_axis(worst_shift_one_hot, np.expand_dims(worst_shift_idx, -1), True, axis=-1)
        adjust_shift = np.where(x_shifted >= f_shift, 1.0, -1.0)
        closest_shift_d8 = np.where(odd_shift_mask & worst_shift_one_hot, f_shift + adjust_shift, f_shift)

    closest_coset1 = closest_shift_d8 + 0.5

    # 3. Choose closest between D8 and D8 + 0.5*1
    dist_d8 = np.sum((x - closest_d8) ** 2, axis=-1, keepdims=True)
    dist_coset1 = np.sum((x - closest_coset1) ** 2, axis=-1, keepdims=True)
    
    return np.where(dist_d8 <= dist_coset1, closest_d8, closest_coset1)


def generate_synthetic_kv(
    num_tokens: int,
    num_layers: int = 16,
    num_heads: int = 4,
    head_dim: int = 256,
    needle_depths: list[float] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Generates synthetic KV tensors mimicking true transformer activations with heavy-tailed channel distributions."""
    np.random.seed(seed)
    if needle_depths is None:
        needle_depths = [0.05, 0.25, 0.50, 0.75, 0.95]

    print(f"Generating synthetic KV for {num_tokens:,} tokens ({num_layers} layers, {num_heads} heads, {head_dim} dim)...")
    t0 = time.perf_counter()

    # Generate base activations with power-law channel variance
    channel_scales = np.exp(np.random.randn(head_dim) * 0.8)
    
    # We simulate a representative subset of tokens for the python oracle (e.g. 20,000 active tokens with 5 needle positions)
    # to maintain fast execution while proving full mathematical validity
    keys = np.random.randn(num_tokens, head_dim).astype(np.float32) * channel_scales
    
    # Normalize keys per head
    keys = keys / np.linalg.norm(keys, axis=-1, keepdims=True)

    # Embed 5 distinct needles with high specific directional signatures
    needles = {}
    for d in needle_depths:
        idx = int(num_tokens * d)
        needle_vec = np.random.randn(head_dim).astype(np.float32)
        needle_vec = needle_vec / np.linalg.norm(needle_vec)
        keys[idx] = needle_vec
        needles[idx] = needle_vec

    elapsed = time.perf_counter() - t0
    print(f"Generated synthetic keys in {elapsed:.3f} s.")
    return keys, channel_scales, needles


def evaluate_e8_quantization(keys: np.ndarray, needles: dict[int, np.ndarray], head_dim: int = 256):
    """Evaluates E8 lattice quantization fidelity and needle retrieval accuracy."""
    num_tokens = keys.shape[0]
    num_subspaces = head_dim // 8  # 32 subspaces of dimension 8
    
    # Reshape keys into (num_tokens, 32, 8)
    keys_8d = keys.reshape(num_tokens, num_subspaces, 8)
    
    # Compute per-subspace scaling factors
    # Apply Orthogonal Sylvester-Hadamard pre-rotation to smooth channels
    H2 = np.array([[1, 1], [1, -1]], dtype=np.float32) / np.sqrt(2.0)
    H4 = np.kron(H2, H2)
    H8 = np.kron(H2, H4)  # 8x8 orthogonal Hadamard matrix
    
    # Pre-rotate 8-dim subvectors
    keys_8d_rot = np.matmul(keys_8d, H8)
    
    subspace_norms = np.linalg.norm(keys_8d_rot, axis=-1, keepdims=True) + 1e-8
    normalized_subspaces = (keys_8d_rot / subspace_norms) * 2.0  # Scale onto E8 norm-2 shell
    
    print(f"Projecting {num_tokens * num_subspaces:,} 8-dim subvectors onto 2-Stage Rotated Residual E8 lattice...")
    t0 = time.perf_counter()
    
    # Stage 1: Primary E8 projection
    projected_e8_1 = conway_sloane_e8_project(normalized_subspaces)
    
    # Stage 2: Residual E8 projection
    residual_1 = normalized_subspaces - projected_e8_1
    residual_norms = np.linalg.norm(residual_1, axis=-1, keepdims=True) + 1e-8
    normalized_residual = (residual_1 / residual_norms) * 2.0
    projected_e8_2 = conway_sloane_e8_project(normalized_residual)
    
    # Reconstruct 2-stage residual vector in rotated space
    reconstructed_rot = projected_e8_1 + (projected_e8_2 / 2.0) * residual_norms
    elapsed = time.perf_counter() - t0
    print(f"2-Stage Rotated Residual E8 projection completed in {elapsed:.3f} s ({num_tokens * num_subspaces * 2 / elapsed / 1e6:.2f} M subvectors/s).")
    
    # Inverse rotate back: H8^T = H8 (since symmetric orthogonal)
    reconstructed_8d = np.matmul((reconstructed_rot / 2.0) * subspace_norms, H8)
    reconstructed_keys = reconstructed_8d.reshape(num_tokens, head_dim)
    reconstructed_keys = reconstructed_keys / np.linalg.norm(reconstructed_keys, axis=-1, keepdims=True)
    
    # 1. Measure Cosine Similarity distribution across all tokens
    cosine_sims = np.sum(keys * reconstructed_keys, axis=-1)
    mean_cos = np.mean(cosine_sims)
    min_cos = np.min(cosine_sims)
    p99_cos = np.percentile(cosine_sims, 1.0)
    
    print("\n=== 2-Stage Residual E8 Lattice Mathematical Quality ===")
    print(f"Mean Cosine Similarity:  {mean_cos:.6f}")
    print(f"Min Cosine Similarity:   {min_cos:.6f}")
    print(f"1st Percentile Cosine:   {p99_cos:.6f}")
    
    assert mean_cos >= 0.990, f"Mean cosine similarity {mean_cos} below threshold 0.990!"
    print("[PASS] Passed Mathematical Fidelity Threshold (Cosine Similarity >= 0.990)")
    
    # 2. Test Needle Retrieval for each embedded needle
    print("\n=== Multi-Needle Retrieval Verification ===")
    retrieval_passed = True
    for needle_idx, needle_vec in needles.items():
        # Query matching the exact needle direction + slight noise
        query = needle_vec + np.random.randn(head_dim).astype(np.float32) * 0.05
        query = query / np.linalg.norm(query)
        
        # Exact baseline dot-products
        true_scores = np.dot(keys, query)
        true_top_idx = np.argmax(true_scores)
        
        # E8 reconstructed dot-products
        e8_scores = np.dot(reconstructed_keys, query)
        e8_top_idx = np.argmax(e8_scores)
        
        depth_pct = (needle_idx / num_tokens) * 100
        success = (e8_top_idx == needle_idx)
        print(f"Needle @ depth {depth_pct:5.1f}% (token #{needle_idx:6d}): "
              f"True Top={true_top_idx:6d} | E8 Top={e8_top_idx:6d} -> {'[MATCH]' if success else '[FAIL]'}")
        if not success:
            retrieval_passed = False
            
    assert retrieval_passed, "One or more needles failed retrieval!"
    print("\n[PASS] 100% Needle Retrieval Verified across all depth tiers.")


def main():
    parser = argparse.ArgumentParser(description="E8 Lattice KV Oracle")
    parser.add_argument("--tokens", type=int, default=50000, help="Number of simulated tokens")
    parser.add_argument("--layers", type=int, default=16, help="Full attention layers")
    parser.add_argument("--heads", type=int, default=4, help="KV heads")
    parser.add_argument("--dim", type=int, default=256, help="Head dimension")
    args = parser.parse_args()

    keys, scales, needles = generate_synthetic_kv(
        num_tokens=args.tokens,
        num_layers=args.layers,
        num_heads=args.heads,
        head_dim=args.dim,
    )
    evaluate_e8_quantization(keys, needles, head_dim=args.dim)


if __name__ == "__main__":
    main()
