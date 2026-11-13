# Vesuvius engineering contribution packages

Software for preserving reader context, inspecting sampling accuracy, reducing fitting memory, and running existing ink models.

| Package | Included capabilities | Download |
|---|---|---|
| [Hecate](packages/hecate) | DirectML inference, persistent tile reuse, depth tensors, projection bounds, CT-profile depth alignment | [hecate.zip](archives/hecate.zip) |
| [ink9 and Reader-family tools](packages/ink9_readers) | Rectangular inference, explicit depth/orientation, mapped physical contexts, InstanceNorm tracing | [ink9_readers.zip](archives/ink9_readers.zip) |
| [Canonical fine-ink tools](packages/canonical2_ink) | Read-only weights, controlled GroupNorm probes, zero-bias ONNX repair, DirectML runtime | [canonical2_ink.zip](archives/canonical2_ink.zip) |
| [Spiral fitting and fibres](packages/spiral_fitting) | Active-state AdamW, memory-bounded winding inference, fibre endpoint repair, between-node phase comparison | [spiral_fitting.zip](archives/spiral_fitting.zip) |
| [Surface geometry and CT](packages/surface_ct) | Exact surface-distance checks, sampling metrics, resume-grid handling, sparse CT reads, fractional sampling, streamed directions, normal alignment, volcomp adapter | [surface_ct.zip](archives/surface_ct.zip) |
| [Native features and decoders](packages/native_features) | Constant-token phase diagnostics, physical block-mean sampling, bounded point queries | [native_features.zip](archives/native_features.zip) |

Download the relevant archive and open its README. Each archive contains its code, dependencies, tests and evidence. Model weights and raw CT are not redistributed.

## Evidence and compatibility

The packages include historical real-data results and current portable-code verification as separate records. The historical experiments demonstrate specific outcomes on PHerc0139, PHerc0343, PHerc0800, PHerc0813 and PHerc1218. They are not universal accuracy estimates.

Examples include restoration of four known component patterns through physical-context mapping, reduction of greater-than-one-voxel sampling departures from 17,141 to one on a fixed population, and approximately 77% lower between-node phase error in a matched fibre-spacing test. The fitting package also preserves the implementation and controls behind a 2.31 GB reduction in dense optimizer-moment storage for one local update.

The current code checks exercise numerical contracts, model-input/output handling, memory-preserving computations and exact/restricted replays against retained real outputs. Read each package's `evidence/` directory for its scope. A successful diagnostic does not certify a newly recovered character.

The tested ink-reader Python interfaces come from ScrollPrize/villa commit `e0bbb8b40a2db58b1d71864f286eb85717e59e64`. Legacy CT adapters are explicitly version-bound. Current array-level tools do not require that legacy API.

## Earlier contributions

The October Hecate DirectML contribution is [upstream discussion 2](https://huggingface.co/scrollprize/hecate/discussions/2). The small November tile-reuse and fibre-endpoint bundle remains available in this repository's original root paths and [original release commit](https://github.com/merchantmoh-debug/vesuvius-progress-tools/tree/f54f708b36d2f7361f5aed32e14dca13a2a6579d). Those earlier capabilities are identified as prior work inside the larger packages.

## Attribution

Contributions developed by Mohamad Al-Zawahreh. Original models, training code, VC3D, spiral fitting and scan data remain credited to their respective authors. These tools extend or inspect those systems; they do not claim authorship of the upstream models.

Code: MIT, with attribution in each package. Derived example coordinates, model-output arrays and scientific comparison images: CC BY-NC 4.0. Dataset: Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, et al., *Vesuvius Challenge - CT Scans of Herculaneum Papyri*, Vesuvius Challenge, 2026. [Official data catalogue](https://github.com/ScrollPrize/open-data).

[Current package verification](verification/README.md) · [Source attribution and compatibility](PROVENANCE.md) · [Download checksums](SHA256SUMS)
