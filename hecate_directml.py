"""Optional Windows DirectML inference for Hecate's 9.6 um 2D ink map.

Keep this file beside the released hecate.py. Export once, then reuse the graph.
CPU/CUDA inference and the released weights are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np


@contextmanager
def mapped_array(path: Path, dtype: str, shape: tuple):
    """Close the Windows file mapping even when inference or writing raises."""
    array = np.memmap(path, mode='w+', dtype=dtype, shape=shape)
    try:
        yield array
    finally:
        array._mmap.close()


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_model(checkpoint: Path, output: Path) -> None:
    import onnx
    import torch
    from hecate import load_model

    sidecar = output.with_suffix('.onnx.json')
    if output.suffix != '.onnx' or output.exists() or sidecar.exists():
        raise ValueError('Choose new .onnx and .onnx.json output paths')
    model = load_model(checkpoint, 'cpu')
    if model.sampling_um != 9.6 or tuple(model.patch_size) != (16, 64, 64):
        raise ValueError('This backend supports only Hecate 9.6 um / 16x64x64')
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(checkpoint)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        graph_path = Path(tmp) / 'model.onnx'
        torch.onnx.export(
            model, (torch.zeros(1, 1, 16, 64, 64),), str(graph_path),
            input_names=['volume'], output_names=['logits'], opset_version=18,
            dynamo=False, external_data=False, do_constant_folding=True,
        )
        # A small sidecar avoids loading and serializing another 513 MB protobuf
        # while the traced PyTorch model is still resident.
        onnx.checker.check_model(str(graph_path))
        metadata = {
            'hecate_backend_contract': '9.6um-2d-fp32-v1',
            'checkpoint_sha256': checkpoint_hash,
            'graph_sha256': sha256(graph_path),
        }
        metadata_path = Path(tmp) / 'model.onnx.json'
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        graph_path.replace(output)
        metadata_path.replace(sidecar)


def load_session(graph: Path, device_id: int = 0, profile_prefix: str | None = None):
    import onnxruntime as ort

    if device_id < 0:
        raise ValueError('device_id must be nonnegative')
    if 'DmlExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('Install onnxruntime-directml in a Windows environment')
    metadata = json.loads(graph.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    if metadata.get('hecate_backend_contract') != '9.6um-2d-fp32-v1' or metadata.get('graph_sha256') != sha256(graph):
        raise ValueError('Use the unchanged graph and sidecar exported by this helper')
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    session = ort.InferenceSession(str(graph), sess_options=options,
        providers=[('DmlExecutionProvider', {'device_id': device_id})])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != 'volume'
            or inputs[0].shape != [1, 1, 16, 64, 64] or inputs[0].type != 'tensor(float)'
            or len(outputs) != 1 or outputs[0].name != 'logits'
            or outputs[0].shape != [1, 1, 64, 64] or outputs[0].type != 'tensor(float)'):
        raise ValueError('Unexpected model input/output contract')
    return session


def predict_patch(session, raw: np.ndarray) -> np.ndarray:
    if raw.shape != (16, 64, 64) or raw.dtype != np.uint8:
        raise ValueError('Expected uint8 16x64x64 CT patch')
    image = (raw.astype(np.float32) / 255)[None, None]
    logits = session.run(['logits'], {'volume': image})[0]
    if logits.shape != (1, 1, 64, 64) or not np.isfinite(logits).all():
        raise FloatingPointError('Invalid model output')
    return (1 / (1 + np.exp(-logits)))[0, 0]


def predict(session, volume, output, reverse: bool = False, stride: int = 32, tile_cache=None) -> dict:
    """Stream uint8 ZYX input into an HxW uint8 probability array.

    At 9.6 um, attention and output use the same XY grid. Its validity mask is
    constant through depth for each column. Nonempty columns therefore match
    unmasked attention exactly; empty columns are zeroed before blending.
    This equivalence does not justify using this helper for the 2.4 um model.
    """
    import torch
    from hecate import axis_starts

    if len(volume.shape) != 3 or np.dtype(volume.dtype) != np.dtype('uint8'):
        raise ValueError('Input must be an unnormalized uint8 ZYX surface volume')
    depth, height, width = volume.shape
    if depth < 16 or height < 1 or width < 1 or not 0 < stride <= 64:
        raise ValueError('Need depth >= 16, positive XY dimensions and stride 1..64')
    if output.shape != (height, width) or output.dtype != np.uint8:
        raise ValueError('Output must be an HxW uint8 array')
    z0 = depth // 2 - 8
    zs = slice(depth - z0 - 16, depth - z0) if reverse else slice(z0, z0 + 16)
    axis = torch.hann_window(64, periodic=False)
    window = axis[:, None] * axis[None, :]
    window = (window / window.max()).clamp_min(.001).numpy()
    ys, xs = axis_starts(height, 64, stride), axis_starts(width, 64, stride)
    forward_seconds = []
    with (tempfile.TemporaryDirectory(prefix='hecate-dml-') as tmp,
          mapped_array(Path(tmp) / 'sum', 'float32', (height, width)) as numerator,
          mapped_array(Path(tmp) / 'weight', 'float32', (height, width)) as denominator):
        for y in ys:
            for x in xs:
                ph, pw = min(64, height - y), min(64, width - x)
                region = np.s_[y:y+ph, x:x+pw]
                denominator[region] += window[:ph, :pw]
                raw = np.asarray(volume[zs, y:y+ph, x:x+pw])
                if reverse:
                    raw = raw[::-1]
                if not raw.any():
                    continue
                patch = np.zeros((16, 64, 64), dtype=np.uint8)
                patch[:, :ph, :pw] = raw
                tick = time.perf_counter()
                if tile_cache is None:
                    probability = predict_patch(session, patch)
                    hit = False
                else:
                    probability, hit = tile_cache.get_or_compute(
                        patch, lambda raw: predict_patch(session, raw))
                if not hit:
                    forward_seconds.append(time.perf_counter() - tick)
                weighted = (probability * patch.any(0)) * window
                numerator[region] += weighted[:ph, :pw]
        for y in range(0, height, 128):
            for x in range(0, width, 256):
                region = np.s_[y:y+128, x:x+256]
                probability = numerator[region] / denominator[region]
                if not np.isfinite(probability).all():
                    raise FloatingPointError('Non-finite blended output')
                output[region] = np.rint(probability.clip(0, 1) * 255).astype(np.uint8)
    return {'tiles': len(ys) * len(xs), 'model_forwards': len(forward_seconds),
            'forward_seconds': forward_seconds,
            **({} if tile_cache is None else tile_cache.stats())}


def finish_profile(session) -> dict:
    path = Path(session.end_profiling())
    providers = Counter(event.get('args', {}).get('provider')
        for event in json.loads(path.read_text(encoding='utf-8'))
        if event.get('args', {}).get('provider'))
    if providers and set(providers) != {'DmlExecutionProvider'}:
        raise RuntimeError(f'Unexpected executed providers: {dict(providers)}')
    return {'path': str(path), 'executed_providers': dict(providers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export', help='Export the unchanged released checkpoint once')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('predict', help='Run the exported model on a Windows GPU')
    run.add_argument('--model', type=Path, required=True)
    run.add_argument('--input', required=True, help='uint8 ZYX surface .npy or local Zarr')
    run.add_argument('--array', default='0')
    run.add_argument('--output', type=Path, required=True, help='New probability .png')
    run.add_argument('--spacing-um', type=float, required=True)
    run.add_argument('--reverse', action='store_true')
    run.add_argument('--stride', type=int, default=32)
    run.add_argument('--device-id', type=int, default=0)
    run.add_argument('--profile', action='store_true')
    run.add_argument('--tile-cache', type=Path, help='Persistent prediction cache; one process per directory')
    run.add_argument('--cache-max-mib', type=int, default=128)
    run.add_argument('--cache-epoch', default='', help='Change after a GPU driver or execution-environment change')
    args = parser.parse_args()
    if args.command == 'export':
        export_model(args.checkpoint, args.output)
        return
    if not np.isfinite(args.spacing_um) or abs(args.spacing_um / 9.6 - 1) > .02:
        parser.error('Resample the surface render to 9.6 um first; this script does not resample')
    if args.output.suffix.lower() != '.png' or args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Choose new .png and matching .json output paths')
    from hecate import open_volume, write_png
    volume = open_volume(args.input, args.array)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    session = load_session(args.model, args.device_id,
        str(args.output.with_suffix('.profile')) if args.profile else None)
    tile_cache = None
    if args.tile_cache:
        import platform
        import onnxruntime as ort
        from hecate_tile_cache import TileCache
        if args.cache_max_mib < 1:
            parser.error('--cache-max-mib must be positive')
        tile_cache = TileCache(args.tile_cache, {
            'contract': 'hecate-9p6-fp32-tile-v1',
            'graph_sha256': sha256(args.model),
            'helper_sha256': sha256(Path(__file__)),
            'runtime': ort.__version__, 'provider': 'DmlExecutionProvider',
            'device_id': args.device_id, 'host': platform.node(),
            'machine': platform.machine(), 'cache_epoch': args.cache_epoch,
            'normalization': 'uint8/255', 'input_shape': [16, 64, 64],
        }, max_bytes=args.cache_max_mib * 2**20)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='hecate-dml-') as tmp:
        partial = Path(tmp) / 'prediction.png'
        with mapped_array(Path(tmp) / 'prediction', 'uint8', volume.shape[-2:]) as result:
            stats = predict(session, volume, result, args.reverse, args.stride, tile_cache)
            write_png(result, partial)
        partial.replace(args.output)
    metadata = {
        'sampling_um': args.spacing_um, 'shape_yx': list(volume.shape[-2:]),
        'reverse_z': args.reverse, 'central_depth': 16, 'stride': args.stride,
        'precision': 'fp32', 'provider': 'DmlExecutionProvider', 'device_id': args.device_id,
        'cpu_fallback_disabled': True,
        'checkpoint_sha256': json.loads(args.model.with_suffix('.onnx.json').read_text(encoding='utf-8'))['checkpoint_sha256'],
        'probabilities': 'uint8 = round(255 * probability); no min-max scaling',
        'elapsed_seconds': time.perf_counter() - start, **stats,
    }
    if args.profile:
        metadata['profile'] = finish_profile(session)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
