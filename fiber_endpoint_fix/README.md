# Preserve the final fibre constraint

The spiral fitter's `_decimate_ordered_points_min_spacing` keeps the first point and then selects points by distance.
When the final point is too close to the last retained point, it disappears.
The fitter can therefore receive a shortened guide even though its supplied trace includes a definite endpoint.

The patch force-keeps the final index using the function's existing exception mechanism.
It leaves every previous spacing decision unchanged.
Already-retained endpoints are not duplicated, and explicit branch/contact indices remain preserved.

The current upstream object-store fibre loader calls this function after trimming to the control-point span.
Its existing forced indices cover contacts and branches, not every unlinked terminal control.
The observed defect therefore remains relevant to the current input path.

## Real-data reproduction

The examples are two traced curves from PHerc0813, scan 20250821151723, native 9.362 µm.
Arrays use native XYZ float32 coordinates. The tested spacing is 16 native voxels.

| Curve | Original points | Before retained | After retained | Old last index | True last index |
|---|---:|---:|---:|---:|---:|
| guide001 | 1265 | 78 | 79 | 1249 | 1264 |
| guide002 | 1271 | 79 | 80 | 1269 | 1270 |

The omitted terminal arc lengths are 14.868882 and 0.544222 native voxels.
All earlier retained indices and coordinate values are unchanged after the patch.
This is an input-retention result; it does not claim a newly readable surface or character.

Before applying the patch, reproduce the baseline defect:

```powershell
python verify.py path/to/villa/spiral-fitting/spiral_helpers.py --expect-baseline-defect
```

From the villa repository root, apply the patch:

```powershell
git apply path/to/preserve_fiber_endpoints.patch
```

Then run the fixed-path and boundary checks:

```powershell
python verify.py path/to/villa/spiral-fitting/spiral_helpers.py
```

The verifier extracts and runs the actual function from the supplied source file.
It requires NumPy; it does not import or start the GPU fitter.
`real_curve_results.json` records the source blob and exact example identities.
The patch was prepared against upstream blob 6019b954437414fde83f4d554941462bded024a5, observed November 12, 2026.

## Data attribution

The two traced-curve examples are derived from Vesuvius Challenge CT data:
https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr/

They were generated using the team's released fibre follower and retained for this loader reproduction.
They are not manually verified ground-truth annotations.
Example data: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/.
Official data-catalogue license: https://github.com/ScrollPrize/open-data#license.
Dataset citation: Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, et al. *Vesuvius Challenge - CT Scans of Herculaneum Papyri*. Vesuvius Challenge. 2026.
Code and patch: MIT, with upstream attribution retained.
