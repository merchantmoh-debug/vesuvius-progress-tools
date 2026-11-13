"""Inspect Hecate depth logits, normalized projection weights and attainable bounds.

These calculations do not change model parameters or create recovery images.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def sigmoid(value):
    x = np.asarray(value, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    result[~positive] = exp / (1 + exp)
    return result


def projection(logits, attention, axis=0):
    """Hecate projects logits first, then applies sigmoid; never average probabilities here."""
    l, w = np.asarray(logits, dtype=np.float64), np.asarray(attention, dtype=np.float64)
    if l.shape != w.shape or l.size == 0 or not np.isfinite(l).all() or not np.isfinite(w).all():
        raise ValueError('Matching nonempty finite logits and attention are required')
    if np.any(w < 0) or not np.allclose(w.sum(axis=axis), 1, rtol=0, atol=2e-6):
        raise ValueError('Attention must be nonnegative and sum to one over depth')
    return sigmoid(np.sum(w * l, axis=axis))


def depth_bounds(logits, axis=0):
    """Tight bounds over every nonnegative, normalized depth projection at fixed logits."""
    l = np.asarray(logits, dtype=np.float64)
    if l.size == 0 or not np.isfinite(l).all():
        raise ValueError('Finite nonempty logits are required')
    return sigmoid(l.min(axis=axis)), sigmoid(l.max(axis=axis))


def blended_profile(tiles):
    """Apply the same convex bound to positive spatial blending of multiple tiles."""
    if not tiles:
        raise ValueError('At least one tile is required')
    weights = np.array([t['blend_weight'] for t in tiles], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-8):
        raise ValueError('Spatial blend weights must form a convex combination')
    rows = []
    for tile in tiles:
        low, high = depth_bounds(tile['logits'])
        value = projection(tile['logits'], tile['attention'])
        if np.ndim(value):
            raise ValueError('Point profiles must have one depth dimension')
        rows.append((float(low), float(value), float(high)))
    values = weights @ np.asarray(rows)
    return dict(zip(('lower_bound', 'projected_probability', 'upper_bound'), map(float, values)))


def factor_swap(logits_a, attention_a, depth_a, logits_b, attention_b, depth_b):
    """Synthetic factor comparison only when both arrays name the same physical samples."""
    if not np.array_equal(np.asarray(depth_a), np.asarray(depth_b)):
        raise ValueError('Align physical depths before swapping factors')
    return {'baseline': float(projection(logits_a, attention_a)),
            'weights_only': float(projection(logits_a, attention_b)),
            'logits_only': float(projection(logits_b, attention_a)),
            'both': float(projection(logits_b, attention_b)),
            'recovery_eligible': False}


def sample_offsets(depth_count, centre_index, source_pitch_um, selected_indices, reverse=False):
    """Return signed physical depths for the actual selected source indices."""
    indices = np.asarray(selected_indices)
    if (depth_count < 1 or indices.ndim != 1 or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= depth_count)
            or not np.isfinite([centre_index, source_pitch_um]).all() or source_pitch_um <= 0):
        raise ValueError('Invalid source depth contract')
    actual = depth_count - 1 - indices if reverse else indices
    return (actual - centre_index) * source_pitch_um


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path, help='JSON with profiles, each containing tiles/logits/attention/blend_weight')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = json.loads(a.input.read_text(encoding='utf-8'))
    profiles = source['profiles']
    if isinstance(profiles, dict):
        profiles = [dict(name=k, **v) for k, v in profiles.items()]
    result = [{'name': q.get('name', q.get('mark', 'profile')), 'phase': q.get('phase'),
               **blended_profile(q['tiles'])} for q in profiles]
    a.output.write_text(json.dumps({'profiles': result, 'scope': 'Fixed-logit projection bounds, not ink authentication'}, indent=2), encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
