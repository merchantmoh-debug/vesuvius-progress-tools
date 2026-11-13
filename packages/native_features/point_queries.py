"""Bound the point-head query batch while retaining the same native feature context."""
import numpy as np


def query_points(head, tokens, embed_skips, points_zyx, spatial_shape, *, output_key='ink', chunk_size=4096):
    """Call the released forward_points interface without refitting or feature normalization."""
    import torch
    if type(chunk_size) is not int or chunk_size < 1 or head.training:
        raise ValueError('Positive query chunk size and evaluation-mode head required')
    points = torch.as_tensor(points_zyx, dtype=tokens.dtype, device=tokens.device)
    shape = torch.as_tensor(spatial_shape, dtype=points.dtype, device=points.device)
    if shape.shape != (3,) or (shape <= 0).any() or not torch.isfinite(shape).all():
        raise ValueError('Positive spatial ZYX shape required')
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not torch.isfinite(points).all():
        raise ValueError('Finite nonempty N,3 ZYX points required')
    if ((points < -.5) | (points >= shape - .5)).any():
        raise ValueError('Query leaves its native feature context')
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(points), chunk_size):
            selected = points[start:start + chunk_size]
            result = head.forward_points(tokens, embed_skips, selected[None], tuple(spatial_shape))[output_key]
            if result.ndim != 3 or result.shape[:2] != (1, len(selected)) or not torch.isfinite(result).all():
                raise ValueError('Point head returned an incompatible result')
            pieces.append(result[0].float().cpu().numpy())
    return np.concatenate(pieces)
