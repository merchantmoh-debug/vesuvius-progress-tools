# ink9 and Reader-family context tools

Rectangular inference, depth-order control, fixed-window rotations, physical-context mapping and per-crop normalization tracing.

## Dependencies and supported interface

Use the ink-detection Python dependencies from [villa](https://github.com/ScrollPrize/villa), plus `requirements.txt`. The tested source snapshot is commit `e0bbb8b40a2db58b1d71864f286eb85717e59e64`. `--villa-src` names its `vesuvius/src` directory; no private project package is required.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s . -p "test_*.py"
python window_inference.py --villa-src path/to/villa/vesuvius/src --checkpoint reader.pth --input surface.npy --output reading.tif
```

The checkpoint supplies the architecture and normalization. The supported family uses 17 depth planes and a square window of at least 128 pixels. Input is unnormalized uint8 ZYX `.npy`, with enough physical context for complete windows. Safetensors releases require their saved JSON configuration through `--config`.

The result includes raw float32 scores in TIFF, a separate 0.25–0.75 display PNG, and JSON describing the actual window/depth settings. Raw scores are never silently replaced with display intensities. The default output budget is 16 million pixels and can be changed explicitly.

`--reverse` reverses the source depth axis before the selected window is taken. `--depth-start` is the start index after that optional reversal. `--quarter-turns` rotates each normalized input window and inversely rotates its output before the unchanged Hann blend. It does not shift window membership.

`--experimental-patch-size` explicitly changes XY context for fully convolutional diagnostics. It is not a new checkpoint or an accepted accuracy setting. The historical 384-pixel comparison reduced tile-phase disagreement by 42.8%, but failed the separate frozen w060 reading comparison.

## Map a reader's physical context through a changed chart

`map_contexts.py` maps known original window centres through XYZ geometry into the new chart. It never chooses coordinates by ink score. It accepts TIFXYZ directories or H×W×3 XYZ arrays. The bundled `surface_geometry.py` supplies the numerical mapping.

The mapping JSON has:

```json
{
  "centres_xy": [[119.5, 127.5]],
  "original_scale_xy": [0.25, 0.25],
  "new_scale_xy": [0.25, 0.25],
  "window_size": 128,
  "max_distance": 0.2
}
```

These numbers illustrate the schema; supply your actual chart scales and original centres. Pixel centre `(x,y)` maps to grid coordinate `(pixel + 0.5) * scale`. The maximum distance is in the XYZ grid's units. A local inverse requires the intended sheet; it does not resolve competing sheets.

```text
python map_contexts.py --original-mesh old_surface --new-mesh new_surface --mapping mapping.json --volume new_render.npy --output contexts.json
```

With `--volume`, each complete context is saved as a separate `.npy` beside the mapping. Out-of-volume windows fail instead of receiving artificial padding. Run the fixed reader on those inputs with the required physical orientation and unchanged normalization.

## Trace per-crop normalization

`normalization_trace.point_trace(model, conv_name, norm_name, tensor, point_zyx, output=...)` captures the point's first-convolution values, crop means/variances and InstanceNorm output. It clones features before in-place nonlinearities and removes its hooks after use. It requires evaluation mode, batch one and InstanceNorm3d without running statistics.

This separates changed input normalization from changed internal feature normalization. The package test constructs identical local convolution input under different surrounding context and verifies the internal normalization effect analytically.

## Evidence

The historical PHerc0343 comparison in `evidence/context_restoration.json` restored four known component patterns after flattening, with correlations 0.9881–0.9971. The right-leg peak was 0.777218 before flattening, 0.505366 on the changed lattice and 0.792078 with mapped context. `context_restoration.png` shows the actual outputs.

The portable mapper reproduces the four recorded crop rectangles exactly. This is a numerical port check, separate from the historical reader execution. The original [PHerc343 phase-preserving merge](https://github.com/Nieuwlaar/pherc343-first-letters/blob/main/ink/scripts/merge_tile_phase.py) is relevant prior work; this package's mapped case uses the changed 3D geometry rather than a universal padding offset.

Code MIT. Scientific comparison images CC BY-NC 4.0 with Vesuvius Challenge attribution. Original model and training code remain the upstream authors' work.
