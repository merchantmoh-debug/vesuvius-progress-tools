# Provenance and compatibility

Software contributions by Mohamad Al-Zawahreh, developed and tested with Codex. Upstream models and algorithms retain their original attribution.

The Hecate DirectML backend was the October contribution. Persistent prediction reuse and the fibre endpoint patch were submitted in the first November bundle. They are grouped here with later work and are not presented as new a second time.

The ink-reader interface was tested with ScrollPrize/villa revision e0bbb8b40a2db58b1d71864f286eb85717e59e64. Hecate's original Python source has SHA-256 c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59. Native-feature diagnostics derive from the DinoVol interface in villa revision 020f2754d3c32225e6a66fca36cab9dc027f01e4. The legacy CTVolume integrations retain their older ccb4318 interface and are explicitly named legacy; the independent array tools avoid that dependency.

Historical evidence records retain their original scope and population. Present implementation checks are in verification/. Synthetic test fixtures are labelled and are not scroll-recovery measurements. Small guide coordinates, model-output phase arrays and comparison images are derived data under the attribution in each DATA_LICENSE.md. No raw CT or model weights are redistributed.

The original model source, checkpoints and data remain available through the linked upstream projects. Packages expose file, array or model-object interfaces; none requires the private research workspace.
