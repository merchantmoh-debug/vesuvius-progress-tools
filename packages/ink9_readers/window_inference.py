"""Rectangular native-reader inference with explicit depth, orientation and raw/display outputs."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import numpy as np
from oriented_context import orient_forward


def sliding_positions(length, patch_size, stride):
    if min(length, patch_size, stride) < 1 or stride > patch_size:
        raise ValueError('Positive dimensions and stride <= patch size are required')
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def infer_array(model, normalize, raw, crop_size, *, reverse=False, depth_start=None, quarter_turns=0):
    import torch
    depth, size, width = map(int, crop_size)
    if raw.dtype != np.uint8 or raw.ndim != 3 or depth != 17 or size != width or min(raw.shape[1:]) < size:
        raise ValueError('Expected uint8 ZYX with a 17-plane square-window reader and enough XY context')
    z0 = raw.shape[0] // 2 - 8 if depth_start is None else int(depth_start)
    if z0 < 0 or z0 + depth > raw.shape[0] or quarter_turns not in range(4):
        raise ValueError('Invalid depth window or orientation')
    stack = (raw[::-1] if reverse else raw)[z0:z0 + depth]
    supported = raw.any(axis=0)
    if not supported.any():
        raise ValueError('No supported source pixels')
    axis = torch.hann_window(size, periodic=False)
    window = torch.outer(axis, axis)
    window = (window / window.max()).clamp_min(.001).numpy()
    numerator = np.zeros(raw.shape[1:], np.float32)
    denominator = np.zeros_like(numerator)
    forward = orient_forward(model.forward, quarter_turns, torch)
    count = 0
    model.eval()
    with torch.inference_mode():
        for y in sliding_positions(raw.shape[1], size, size // 2):
            for x in sliding_positions(raw.shape[2], size, size // 2):
                region = np.s_[y:y + size, x:x + size]
                if not supported[region].any():
                    continue
                patch = np.ascontiguousarray(normalize(stack[:, y:y + size, x:x + size]))
                logits = forward(torch.from_numpy(patch)[None, None])['ink']
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred = logits.float().sigmoid().squeeze().cpu().numpy()
                if pred.shape != (size, size) or not np.isfinite(pred).all():
                    raise ValueError('Model output violates the square-window contract')
                numerator[region] += pred * window
                denominator[region] += window
                count += 1
    if not (denominator[supported] > 0).all():
        raise ValueError('Supported output has an uncovered window gap')
    np.divide(numerator, denominator, out=numerator, where=denominator > 0)
    return numerator, supported, {'tiles': count, 'depth_halfopen_after_optional_reversal': [z0, z0 + depth],
                                 'reverse_depth': reverse, 'inplane_quarter_turns': quarter_turns}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--villa-src', type=Path, required=True, help='Trusted villa/vesuvius/src directory')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='Unnormalized uint8 ZYX .npy')
    p.add_argument('--output', type=Path, required=True, help='New raw probability .tif')
    p.add_argument('--config', type=Path, help='Saved JSON config when loading a safetensors release')
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--depth-start', type=int)
    p.add_argument('--quarter-turns', type=int, choices=range(4), default=0)
    p.add_argument('--experimental-patch-size', type=int, help='Explicit XY context intervention; not a new checkpoint or validated accuracy setting')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-output-pixels', type=int, default=16000000)
    a = p.parse_args()
    if a.output.suffix.lower() not in ('.tif', '.tiff') or any(a.output.with_suffix(s).exists() for s in ('.tif','.tiff','.png','.json')):
        raise ValueError('Choose unused output paths')
    import torch
    import tifffile
    from PIL import Image
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(a.villa_src.resolve()))
    from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, load_model_state, select_inference_weights
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.ink_detection.data.normalization import normalize_image
    if a.checkpoint.suffix == '.safetensors':
        if a.config is None:
            raise ValueError('A safetensors release requires its saved config')
        from safetensors.torch import load_file
        payload = {'model': load_file(str(a.checkpoint)), 'config': json.loads(a.config.read_text(encoding='utf-8'))}
    else:
        payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    cfg = config_from_checkpoint(payload, source=a.checkpoint)
    key, weights = select_inference_weights(payload, source=a.checkpoint)
    model = make_model(cfg)
    load_model_state(model, weights)
    del payload, weights
    gc.collect()
    raw = np.load(a.input, mmap_mode='r', allow_pickle=False)
    if np.prod(raw.shape[-2:]) > a.max_output_pixels:
        raise ValueError('Output exceeds the declared pixel budget')
    crop=list(cfg.model.crop_size)
    if a.experimental_patch_size is not None:
        if a.experimental_patch_size<128:raise ValueError('Experimental context must be at least 128 pixels')
        crop[1:]=[a.experimental_patch_size]*2
    result, support, metadata = infer_array(model, lambda x: normalize_image(x, cfg.data.normalization), raw,
        crop, reverse=a.reverse, depth_start=a.depth_start, quarter_turns=a.quarter_turns)
    tifffile.imwrite(a.output, result)
    display = np.clip((result - .25) / .5, 0, 1)
    display[~support] = 0
    Image.fromarray(np.rint(display * 255).astype(np.uint8)).save(a.output.with_suffix('.png'))
    with a.checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata.update(checkpoint_sha256=checkpoint_hash, state_key=key, checkpoint_crop_size=list(cfg.model.crop_size), actual_crop_size=crop,
                    display_window=[.25, .75], raw_output='float32 sigmoid scores; no display scaling')
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
