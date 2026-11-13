# Hecate tile reuse and fibre endpoint preservation

Two small improvements developed while working on Vesuvius Challenge scroll data.
Code is MIT-licensed. The included traced-curve examples have their separate data attribution below.

## What is new beyond the October DirectML contribution

The October contribution added a Windows DirectML backend for Hecate 9.6 µm.
It received a $250 Progress Prize. That backend and its original speed result are prior work, not new claims here.

This update adds optional, persistent tile predictions to that helper.
An interrupted run can resume without repeating finished model calls.
An expanded field can reuse matching tiles from an earlier smaller field.
Every reuse requires identical model-input bytes and an identical execution binding.
The default inference path is unchanged when `--tile-cache` is absent.

The second contribution is a small patch to villa's fibre point decimator.
It retains the actual final point when that point is closer than the spacing threshold to its predecessor.
Previously retained interior points stay identical.

Original backend: https://huggingface.co/scrollprize/hecate/discussions/2

## Hecate: run and resume

Place these Python files beside the tested upstream `hecate.py` and install its dependencies.
The checked source is available at this pinned public revision:
https://huggingface.co/scrollprize/hecate/resolve/8bbd28e6c5065c6d1cddd79da5c7f5a77a9105f6/hecate.py
Its SHA-256 is `c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59`.
Install `requirements-directml.txt` in a Windows environment separate from other ONNX Runtime variants.
The supported model is Hecate 9.6 µm, FP32, static 16 × 64 × 64 input, batch one.
The runtime disables CPU fallback.

Export the released checkpoint once:

```powershell
python hecate_directml.py export --checkpoint hecate_9.6um.pth --output model.onnx
```

Predict using a dedicated cache directory:

```powershell
python hecate_directml.py predict --model model.onnx --input surface.npy --output ink.png --spacing-um 9.6 --tile-cache .cache --cache-max-mib 128
```

After an interrupted prediction, run the same command again.
For a larger overlapping input, use the same cache and a new output filename.
Matching model-input tiles are reused even when their canvas positions change.
The source's depth selection, masks, padding, tile placement and Hann blending remain unchanged.

The 128 MiB ceiling covers retained prediction entries; small binding manifests are additional.
The cache contains predictions and hashes, not model weights or raw CT.
It evicts older prediction entries when required to remain within that ceiling.
Use one inference process per cache directory.
Use a new `--cache-epoch` value after changing a GPU driver or execution environment beyond the recorded runtime binding.
A corrupted cache entry raises an error instead of silently returning it.

## Demonstrated behaviour

The new integration was tested on the real published PHerc0139 example used for the original backend validation.
The input has 16 × 96 × 96 uint8 values. All 9,216 output pixels match the unchanged awarded helper exactly.

| Operation | Model calls | Cached tiles | Output |
|---|---:|---:|---|
| Fresh 96 × 96 field | 4 | 0 | Baseline |
| Expand an already-read 64 × 96 field to 96 × 96 | 2 | 2 | Exact baseline match |
| Read the completed 96 × 96 field again | 0 | 4 | Exact baseline match |
| Resume after one completed tile and an injected interruption | 3 | 1 | Exact baseline match |
| Change one source pixel after caching | 1 | 3 | Exact fresh changed-input match |

Only DirectML executed model operators. See `reuse_validation.json` for the complete measurements and scope.
The claim is avoided model work with preserved output, not improved ink detection accuracy.
Single-run timing is provided as an observation, not a stable speed benchmark.

An earlier internal implementation was also used on a 1,517-tile PHerc0813 field.
It reused 285 predictions, computed 1,232 new forwards, and preserved 258,048 shared interior pixels exactly.
That historical execution motivates the portable helper; it is not a measurement of this packaged version.

Reproduce the compact public test after exporting the model:

```powershell
python fetch_example.py --output pherc0139_crop.npy
python verify_reuse.py --model model.onnx --input pherc0139_crop.npy --output proof
```

Storage and invalidation regressions run without a neural model:

```powershell
python -m unittest -v test_hecate_tile_cache
```

## Fibre endpoint fix

See `fiber_endpoint_fix/README.md` for the upstream patch and direct reproduction.
On two retained PHerc0813 curves, the current upstream spacing function drops the final point.
The patch changes retained counts from 78/79 to 79/80 and preserves all existing retained points.
These curves are examples for a data-loader defect, not certified paper or ink ground truth.

## Attribution and development

Hecate's model, weights and original predictor belong to the Vesuvius Challenge team.
The helper reuses its volume loader, tile origins and PNG writer.
The endpoint patch targets ScrollPrize/villa, copyright 2024 Vesuvius Challenge, MIT.
These additions were developed and tested by Mohamad Al-Zawahreh with Codex.
No model weights, private conversations, private research-control documents or personal data are included.

The rendered example and generated curve coordinates derive from the newer Vesuvius Challenge scans.
Dataset citation: Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, et al. *Vesuvius Challenge - CT Scans of Herculaneum Papyri*. Vesuvius Challenge. 2026.
The official open-data catalogue lists CC BY-NC 4.0: https://github.com/ScrollPrize/open-data#license.
No raw CT chunks are redistributed. The comparison image demonstrates unchanged prediction output and makes no new textual revelation.
