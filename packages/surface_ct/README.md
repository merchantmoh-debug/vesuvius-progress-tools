# Surface geometry and CT utilities

Numerical tools for preserving physical sampling, comparing saved surfaces, reading only required CT chunks and constructing bounded direction fields.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s . -p "test_*.py"
python surface_geometry.py --mesh surface_directory --voxel-um 9.362 --points query_xyz.npy --output geometry_report.json
```

## Surface geometry

`load_grid` reads a TIFXYZ directory (`x.tif`, `y.tif`, `z.tif`) or an H×W×3 `.npy`. XYZ units must be explicit. The CLI uses the positive-coordinate TIFXYZ validity convention; array callers can supply their own validity mask.

`triangles` uses the native ABF diagonal p10–p01. `exact_surface_distances` searches every triangle that can improve its initial distance bound. It does not assume a fixed number of nearest centroids is sufficient. Work is bounded by batches and a candidate-pair budget; reducing the batch size changes memory, not the distance definition.

`grid_pitch` measures row, column and area-equivalent pitch from actual coordinates. `parent_motion` applies the child's immediate XY grid offset, rather than subtracting successive offsets. `trim_invalid_border` removes only a completely empty outer border and crops every supplied channel alike; it refuses interior holes.

`bilinear` and `world_to_grid` support physical context mapping. The inverse is local and requires the intended sheet. Its residual is returned and checked against an explicit distance. Numerical proximity is not a same-sheet classifier.

## Fractional CT sampling

`fractional_sampling.affine_sample` accepts any positive physical ratio, including 19.2/8.64. Axes are rows for output X/Y/Z; `origin_xyz` is the first sample centre in native CT coordinates. Intensities are trilinearly sampled without normalization. Requests outside the actual volume fail rather than receiving fabricated padding.

```text
python fractional_sampling.py ct.npy --frame frame.json --output crop.npy
```

The frame JSON contains `origin_xyz`, `axes_xyz`, `shape_zyx`, `trace_pitch_um`, and `ct_pitch_um`. This array-level interface does not depend on villa's retired CTVolume class.

`fractional_ct_volume_legacy.py` and `bounded_remote_ct_legacy.py` preserve the historical integrations with the CTVolume/RemoteChunkedArray interface from villa's `ccb4318` snapshot (abbreviated revision). They are not drop-in patches for the newer vc_reader API. The historical fractional control records the exact source-file hash. The transport accepts only its declared uncompressed uint8 128-cubed volume, checks response size, deduplicates concurrent requests and caps total requested bytes. It preserves the upstream interpolation and normalization.

## Read planes and points without allocating a cube

`native_ct_planes.gather(origin_zyx, shape_zyx, xy_z, yz_x, nearest_zyx, acquire, chunk_size=128, workers=1)` asks `acquire(key_zyx)` for each required chunk once. It returns exact XY planes, YZ planes, selected values and the required keys. Source bounds and complete coverage are checked. One or four request workers are supported.

The historical PHerc0800 inspection used 248,672,256 peak private bytes while avoiding a 1,868,562,432-byte bounding-cube allocation. It needed 516 of that cube's 891 chunks and reproduced the previous planes and vertex samples exactly.

## Streamed directions and native normal blocks

```text
python normal_tools.py stream --input ct.npy --output coefficient_fields --gradient-sigma 0.8 --tensor-sigma 2 --slab-depth 32
python normal_tools.py block --chunks chunk_manifest.json --key-zyx 60 40 35 --reference-xyz 1 0 0 --output block.npy
python normal_tools.py align --input complete_field.npy --seed-zyx 12 12 12 --reference-xyz 1 0 0 --output aligned.npy
```

`streamed_directions` retains only Jxy, Jyy and Jyz for the demonstrated brightness-slope calculation. Gaussian halos preserve the complete calculation at slab boundaries. Output mappings are flushed and closed after each slab write; the returned arrays are read-only memory maps.

`ct_normal_blocks` computes one 32-cubed XYZ-encoded normal block from native CT, sampled every four voxels. The chunk manifest lists `key_zyx`, file `path` and SHA-256. Every input is checked before use. Its fixed gradient/tensor settings match the original tested block implementation.

`normal_alignment` uses deterministic six-neighbour sign transport across the complete produced field. Inconsistent adjacent cells are left at the neutral value 128. Neutral cells represent unknown direction, not absent paper. These array utilities do not by themselves claim global orientation or a writing face.

## Existing volcomp codec adapter

```text
node decode_volcomp_chunk.mjs volcomp_viewer.wasm chunk.volc chunk.u8 decode.json
```

Supply the original trusted WASM module built from [volume-compressor revision 81f6519](https://github.com/SuperOptimizer/volume-compressor/tree/81f651964a93c059561775adb4178ba8a2a48248). The wrapper checks its import contract, repeats the decode exactly, and compares a 16-cubed sub-block against the full 128-cubed decode. It does not replace the codec or redistribute its binary. Node.js with WebAssembly support is required.

## Evidence

The historical flattening repair reduced more-than-one-voxel departures from 17,141 to one on the same 463,625-point population. The parent geometry and UV solution were preserved. The renderer-support example restored 126,400 previously blank border pixels in each of three slices by retaining the full parent geometry and cropping the output; extra CT alone did not repair it.

These are representation and data-access outcomes. Source physical geometry and reading quality remain separate claims. Current tests include analytic distances, the exact quad diagonal, affine fractional values, resume offsets, border-only trimming, serial/parallel reads, Gaussian slab seams and sign-conflict handling.

Code MIT. Scientific evidence and derived examples carry Vesuvius Challenge attribution and CC BY-NC 4.0. No raw CT or model weights are included.
