"""Measure decoder phase texture and sample fixed block means in native coordinates."""
import argparse
import json
from pathlib import Path
import numpy as np


def periodicity(logits, period=8):
    """Compare equal within-block phases; input must be one complete ZYX logit cube."""
    a = np.asarray(logits)
    if a.ndim != 3 or not np.isfinite(a).all() or type(period) is not int or period < 1:
        raise ValueError('Finite ZYX logits and a positive integer period are required')
    if any(n % period for n in a.shape):
        raise ValueError('Each axis must contain complete periods')
    nz, ny, nx = (n // period for n in a.shape)
    blocks = a.reshape(nz, period, ny, period, nx, period)
    template = blocks[0, :, 0, :, 0, :]
    error = np.abs(blocks - template[None, :, None, :, None, :])
    return {'period_zyx': [period] * 3, 'shape_zyx': list(a.shape),
            'constant_token_repeat_max_error': float(error.max()),
            'phase_standard_deviation': float(template.std()),
            'phase_range': [float(template.min()), float(template.max())],
            'scope': 'Use logits from spatially constant feature tokens to isolate decoder-generated texture.'}


def constant_token_control(decoder, features, period=8):
    """Run a supplied evaluation-mode decoder on spatially constant feature channels."""
    import torch
    if features.ndim != 5 or not torch.isfinite(features).all() or decoder.training:
        raise ValueError('Finite B,C,Z,Y,X features and an evaluation-mode decoder are required')
    with torch.inference_mode():
        constant = features.mean((2, 3, 4), keepdim=True).expand_as(features).contiguous()
        logits = decoder(constant)
    if logits.ndim != 5 or logits.shape[:2] != (1, 1):
        raise ValueError('This diagnostic requires one scalar B=1,C=1 output cube')
    return periodicity(logits[0, 0].float().cpu().numpy(), period)


def sample_block_means(logits, points_zyx, period=8):
    """Average LOGITS by the fixed block period; return scores and common support.

    Points use centres of native voxels. Coarse samples sit at block centres.
    The common mask excludes coordinates requiring edge padding after pooling.
    """
    import torch
    from torch.nn import functional as F
    if logits.ndim != 5 or tuple(logits.shape[:2]) != (1, 1) or not torch.isfinite(logits).all():
        raise ValueError('Finite scalar 1,1,Z,Y,X logits required')
    if type(period) is not int or period < 1 or any(n % period for n in logits.shape[-3:]):
        raise ValueError('Period must divide every spatial output dimension')
    points = torch.as_tensor(points_zyx, dtype=logits.dtype, device=logits.device)
    if points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all():
        raise ValueError('Finite N,3 native ZYX query points required')
    shape = torch.tensor(logits.shape[-3:], dtype=points.dtype, device=points.device)
    if ((points < -.5) | (points >= shape - .5)).any():
        raise ValueError('Query leaves the native cube')
    grid = ((points + .5) / shape * 2 - 1)[:, [2, 1, 0]][None, None, None]
    pooled = F.avg_pool3d(logits, period, period)
    native = F.grid_sample(logits, grid, align_corners=False, mode='bilinear')[0, 0, 0, 0]
    coarse = F.grid_sample(pooled, grid, align_corners=False, mode='bilinear')[0, 0, 0, 0]
    half = (period - 1) / 2
    common = ((points >= half) & (points <= shape - 1 - half)).all(-1)
    return {'unpooled': native.sigmoid(), 'pooled': coarse.sigmoid(), 'common': common}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--constant-logits', type=Path, required=True, help='ZYX .npy from a spatially constant-token forward')
    p.add_argument('--period', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = periodicity(np.load(a.constant_logits, allow_pickle=False), a.period)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
