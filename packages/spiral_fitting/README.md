# Spiral fitting, winding memory and fibre guidance

The package combines exact active-state CPU optimization, memory-bounded winding-network operations, endpoint-preserving fibre sampling and a reproducible between-node consistency comparison.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s . -p "test_*.py"
python replay_spacing.py --output fibre_comparison.json
```

## Active-state AdamW

Use this for **fresh** CPU float32 optimizer history with **zero weight decay**. It is not an arbitrary dense-history checkpoint loader, CUDA-fused trajectory substitute or sparse-gradient approximation.

```python
from active_adamw import ActiveAdamW

optimizer = ActiveAdamW(lr=6e-6, weight_decay=0)
# parameter and gradient are matching contiguous CPU float32 tensors.
optimizer.step(parameter, gradient)
saved = optimizer.snapshot(parameter)
```

State is retained for every coordinate that has ever received a nonzero gradient. Previously active coordinates keep evolving through zero-gradient steps. The parameter-wide step counter advances even on empty steps. Fixed aligned CPU blocks preserve vector/tail arithmetic; only active values are written back to mapped parameters.

The historical local update had 288,960,000 coordinates and 23,781 active coordinates. Its moments used 190,248 bytes, with a 190,248-byte index, avoiding 2,311,489,752 bytes of dense moments. This measures optimizer-state storage, not the whole fitting process.

The historical implementation passed 28 controlled steps. The current code uses a more selective block/write path, so its current tests are reported separately. Both preserve the dense-update contract under the specified conditions. Current villa's lazy-moment option retains full arrays and has different zero-gradient semantics.

## Winding inference memory

`cpu_attention.partitioned_attention` partitions only independent batch members. Every token and head stays together. It requires CPU execution and zero dropout. The `FunctionalProxy` can replace only a loaded model module's attention dispatch without modifying PyTorch globally.

`cpu_winding_memory.tiled_detail_head` includes the interpolation halo and supports pointwise Conv3d/ChannelLayerNorm/GELU heads. `memory_bounded_forward` follows the demonstrated three-stage winding model with no extra decoder or auxiliary heads. Unsupported architectures fail its explicit checks.

The historical complete 128×128×384 output matched the original CPU output bit-for-bit. Current tests separately compare attention masks and detail-head seams, including a final short tile. No model weights are bundled.

## Fibre endpoints and between-node consistency

```text
python fiber_guidance.py --points fiber_endpoint_fix/guide_001.npy --spacing 16 --output retained_points.json
```

`retained_indices` keeps the actual first/last endpoints and explicit forced indices. The source patch is in `fiber_endpoint_fix/preserve_fiber_endpoints.patch`. It applies to villa's `_decimate_ordered_points_min_spacing` at the source version recorded in the result JSON. From the villa repository root:

```text
git apply path/to/preserve_fiber_endpoints.patch
python path/to/fiber_endpoint_fix/verify.py spiral-fitting/spiral_helpers.py
```

Run the same verifier with `--expect-baseline-defect` before applying the patch to reproduce the two observed omissions.

`compare_phase` excludes the union of both arms' training memberships. By default it measures deviations from each arm's median over those excluded evaluation points; callers can instead provide a fixed target phase. Input arithmetic dtype is retained to reproduce the original float32 calculation. This distinguishes consistency along a guide from closeness at only the supplied training nodes.

The included `evidence/fibre_phases.npz` contains model phase outputs and training indices, not CT or model weights. `replay_spacing.py` recomputes the historical comparison. The only changed setting was spacing 40→16, with 200 updates from the same parent in each arm. P90 deviation fell approximately 77% on both guides over 2,321 excluded evaluation points. Broad sheet geometry remained a separate unsuccessful acceptance test.

## Attribution

Original spiral and winding models are Vesuvius Challenge work. These are local execution and constraint-processing extensions. Code MIT. Derived PHerc0813 curve/phase examples CC BY-NC 4.0 with Vesuvius Challenge data attribution. The endpoint patch and tile-reuse component were already included in the first November bundle; they are not claimed as new again here.
