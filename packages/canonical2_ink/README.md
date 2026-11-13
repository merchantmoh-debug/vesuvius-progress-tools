# Canonical fine-ink runtime and diagnostics

Read-only weight storage, controlled GroupNorm interventions, a targeted ONNX export repair, and the calibrated DirectML prediction contract for the canonical fine-resolution model.

## Install and test

```text
python -m pip install -r requirements.txt
python -m unittest discover -s . -p "test_*.py"
```

The tested model is the released [canonical 2 µm ink model](https://huggingface.co/scrollprize/ink_canonical_2um). Its architecture, checkpoint and preparation remain upstream inputs. These utilities do not convert arbitrary ink models into that contract.

## Stream weights and load read-only

Convert an explicit tensor dictionary from a checkpoint without materializing one concatenated weight payload:

```text
python readonly_weights.py model.ckpt --state-key state_dict --output weights.safetensors
```

Use `--state-key ""` for a root tensor dictionary. Prefix stripping is explicit and must apply to every name. The verified dtype contract is float32/int64. Conversion is per tensor; peak conversion storage still includes the largest tensor.

Construct the released model on the meta device, then assign the mapped state:

```python
import torch
from readonly_weights import ReadOnlyState
from model_resnet3d_3d_decoder import RegressionModel

with torch.device("meta"):
    model = RegressionModel(with_norm=True)
backing = ReadOnlyState("weights.safetensors")
model = backing.load_into(model)
# Use torch.inference_mode() for inference. Never train or mutate mapped weights.
```

The backing object remains attached to the model. PyTorch warns that its tensor interface cannot enforce read-only use; the operating-system mapping is read-only. Do not perform in-place parameter writes or reuse this loader for training.

## Controlled GroupNorm probe

```python
from groupnorm_probe import compare
predictions, report = compare(
    model, baseline_tensor, candidate_tensor,
    output=lambda logits: logits.sigmoid(),
    mask=output_region_mask,
    replay_tolerance=1e-6,
)
```

The tensors must have identical shape and the model must be in evaluation mode. The function runs baseline/candidate under ordinary statistics and under baseline-frozen GroupNorm statistics. It checks baseline replay before returning an interpretation-bearing report. The tolerance belongs to your actual dtype/output path. Hooks are removed even on exceptions.

The intervention changes diagnostic computation, not stored weights. A reduction in sensitivity is conditional evidence about these statistics; it does not make frozen statistics a better production reader.

## Repair a specific ONNX exporter defect

The demonstrated export inserted 168 malformed all-zero Conv biases of shape `[out_channels,2]` where the source model had no bias. The repair removes only those optional inputs. It requires exact source tensor names, checks zero values, keeps valid biases and leaves parameter initializers unchanged.

```text
python repair_zero_bias.py --graph ink_shapes_folded.onnx --source-weights weights.safetensors --output ink_bias_repaired.onnx --expected-repairs 168
```

The output must share the source graph directory, preserving external-data references. Other malformed shapes, nonzero biases, genuine source biases and mismatched counts fail. This is not a general optimizer or an instruction to remove arbitrary Conv biases.

## DirectML prediction

On Windows, install `requirements-directml.txt` in an environment separate from other ONNX Runtime variants. Supply a verified graph with input `volume: [1,1,62,256,256]` and output `logits: [1,1,64,64]`.

```text
python directml_runtime.py --graph ink_bias_repaired.onnx --input tile.npy --output probability.npy
```

The tile is uint8 `(62,256,256)`. Preparation clips to 0–200 and divides by 200. Output logits receive sigmoid and bilinear upsampling with `align_corners=False`. CPU fallback is disabled. This command consumes a compatible exported graph; it is not an arbitrary-checkpoint exporter.

## Evidence

`evidence/directml_calibration.json` records the historical complete-model result: all strong reference ink retained in the two exposed controls, exact GPU repeats, and DirectML-only execution. CPU/GPU values were close but not byte-identical.

The historical GroupNorm intervention reduced the untouched-region response change by 62.5328%. The read-only runtime preserved 978 tensors and reproduced its retained prediction exactly. Current package tests exercise hook cleanup, replay, read-only loading, inference and selective bias repair separately from those historical full-model runs.

Code MIT. Models and original architecture belong to the Vesuvius Challenge authors. No checkpoint or raw CT is included.
