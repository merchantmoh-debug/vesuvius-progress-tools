# Portable package verification — 13 November 2026

The six packages passed 35 local numerical and integration tests. Fourteen command-line help paths and two numerical examples also ran successfully. See `package_checks.json` for exact results.

The portable ink9 calculation matched the original source calculation bit-for-bit on all 16,384 pixels of one retained 17×128×128 CT window, using the actual released seed42 checkpoint. No new reading assessment was performed.

Four geometry-mapped contexts reproduced their saved crop rectangles exactly. A 513-point subset, including the recorded maximum, reproduced the archived surface-distance map within 7.29e-8 native voxels. Four Hecate depth-profile bounds differed by at most 3.41e-8. Both fibre comparisons reproduced their excluded populations and medians exactly; interpolation rounding changes the reported p90 ratios by less than 1e-6. The retained volcomp block reproduced its decoded SHA-256 exactly and passed the independent sub-block address check.

These checks validate the portable implementations. The earlier full-model GroupNorm, DirectML, winding-memory, CT-streaming and accuracy-related results are separately labelled historical in each package. Models and scans needed for full model execution are upstream dependencies, not included files.

The release check took 13.297 seconds and peaked at 932,823,040 private process-tree bytes on the development machine. This is validation cost, not a benchmark of a full scroll run.
