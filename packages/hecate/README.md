# Hecate inference and depth diagnostics

This package groups the Windows DirectML backend and persistent prediction reuse with tools for inspecting the reader's depth evidence and physical input placement.

## Install and run

Use Python 3.11. Install the base dependencies:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s . -p "test_*.py"
```

For Windows GPU inference, install `requirements-directml.txt` in an environment separate from other ONNX Runtime variants. Follow the [released Hecate model](https://huggingface.co/scrollprize/hecate) for its checkpoint and model dependencies. Put the tested upstream `hecate.py` beside the helper:

https://huggingface.co/scrollprize/hecate/resolve/8bbd28e6c5065c6d1cddd79da5c7f5a77a9105f6/hecate.py

Its SHA-256 is `c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59`.

```text
python hecate_directml.py export --checkpoint hecate_9.6um.pth --output model.onnx
python hecate_directml.py predict --model model.onnx --input surface.npy --output ink.png --spacing-um 9.6 --tile-cache tile_cache --cache-max-mib 128
```

Input is an unnormalized uint8 ZYX surface volume, already sampled at the declared physical spacing. The spacing flag does not resample it. The supported model has a 16×64×64 window, FP32, batch one and 2D output. CPU fallback is disabled.

Repeat the command after interruption, or use the same cache for a larger overlapping field and a new output filename. Reuse requires identical model-input bytes and execution binding. The default limit covers 128 MiB of prediction entries; small manifests are additional. Use one process per cache directory. Change `--cache-epoch` after a driver/environment change not represented in the binding. Corruption raises an error.

## Expose and inspect depth evidence

With the trusted released source and checkpoint, export one tile's original depth logits, attention and 2D prediction:

```text
python inspect_depth_tile.py --source hecate.py --checkpoint hecate_9.6um.pth --input tile.npy --output depth.npz
```

The input must be uint8 `(16,64,64)`. This calls the source model API and checks that reconstructing its projection matches its original forward. It does not optimize the weights or the input.

`depth_diagnostics.py` provides `projection`, `depth_bounds`, `blended_profile`, `factor_swap` and `sample_offsets`. Hecate averages depth logits before sigmoid. Nonnegative normalized depth weights cannot produce a probability above the largest depth probability at that fixed pixel. Positive spatial blending preserves the corresponding blended bound.

Recompute the saved real PHerc0813 profile bounds without a model:

```text
python depth_diagnostics.py evidence/depth_profiles.json --output depth_bounds.json
```

Factor swapping is a synthetic diagnostic and requires matching physical depth vectors. It does not produce a recovery image. `sample_offsets` keeps odd/even centres and reversal explicit.

## CT-only depth alignment

`ct_depth_alignment.py` compares profiles in physical micrometres. Its JSON input contains `reference_z_um`, `reference`, `moving_z_um`, `moving`, `shifts_um`, and `bin_width_um`.

```text
python ct_depth_alignment.py profiles.json --output alignment.json
```

The moving profile is sampled at `reference_z_um + shift_um`. The output records the entire supplied search grid, common support and whether the best value lies on its boundary. Fit and check regions should be separate when evaluating transfer. No ink map is used by this calculation.

## Evidence

`reuse_validation.json` and `reuse_comparison.png` contain the compact real-data cache checks. Expansion reuses two of four predictions with all 9,216 output pixels identical; complete replay uses zero model calls; interruption and input mutation are exercised. `fetch_example.py` reconstructs that example from official, hash-checked chunks rather than redistributing CT.

```text
python fetch_example.py --output example.npy
python verify_reuse.py --model model.onnx --input example.npy --output reuse_proof
```

`evidence/depth_alignment.json` and its image preserve a historical known-character comparison: native Hecate correspondence with a fine-reference prediction increased from 0.567740 to 0.801831 after physical-depth correction. A broken connection remained. This is model-to-model correspondence on an exposed case, not an independent accuracy percentage.

The October DirectML backend and first November cache implementation are prior contributions. The depth-inspection, projection and CT-profile tools are grouped with them here. Current package tests and historical full-model results remain separately labelled.

Code MIT. Evidence images and model-output profiles CC BY-NC 4.0; Vesuvius Challenge data attribution applies. No model weights are bundled.
