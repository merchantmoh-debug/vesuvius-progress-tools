"""Export unchanged Hecate 9.6um depth tensors through its released Python model API."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inspect(source, checkpoint, raw):
    import torch
    if raw.dtype != np.uint8 or raw.shape != (16, 64, 64):
        raise ValueError('Supply one uint8 tile with shape (16,64,64)')
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('released_hecate_depth', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_model(checkpoint, 'cpu')
    if tuple(model.patch_size) != (16, 64, 64) or model.divisor != 255:
        raise ValueError('This adapter is for the released 9.6um Hecate contract')
    model.eval()
    with torch.inference_mode():
        image = torch.from_numpy((raw.astype(np.float32) / 255)[None, None])
        features, logits = model.features_logits(image)
        scores = model.canonical.decoder.depth_collapse.attn_conv(features).float()
        half = scores.shape[2] // 2
        z = (torch.arange(scores.shape[2], dtype=torch.float32) - half) / half
        scores = scores + model.depth_coordinate_scale * z.view(1, 1, -1, 1, 1)
        weights = scores.softmax(2)
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
        reconstructed = (weights * logits).sum(2).sigmoid()
        original = model(image).sigmoid()
        error = float((original - reconstructed).abs().max())
        if error > 2e-7:
            raise ValueError(f'Original projection replay differs by {error}; incompatible source contract')
    return {'logits_3d': logits[0, 0].numpy(), 'attention': weights[0, 0].numpy(),
            'probability_2d': original[0, 0].numpy()}, error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Trusted released hecate.py')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if a.output.suffix != '.npz' or a.output.exists() or a.output.with_suffix('.json').exists():
        raise ValueError('Choose a new .npz output and matching .json path')
    import torch
    torch.set_num_threads(a.threads)
    inputs = {str(f): digest(f) for f in (a.source, a.checkpoint, a.input)}
    arrays, error = inspect(a.source, a.checkpoint, np.load(a.input, allow_pickle=False))
    np.savez_compressed(a.output, **arrays)
    a.output.with_suffix('.json').write_text(json.dumps({'source_inputs': inputs, 'projection_max_error': error,
        'scope': 'Unchanged model tensors. No trained weights or scan samples are modified.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
