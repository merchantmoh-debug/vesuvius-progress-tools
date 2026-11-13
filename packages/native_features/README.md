# Native feature and decoder diagnostics

These tools inspect texture introduced by a spatial decoder and bound native point-head queries. They accept existing model objects and arrays. They do not train a model or redistribute checkpoints.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s . -p "test_*.py"
python phase_artifacts.py --constant-logits constant_forward.npy --period 8 --output phase_report.json
```

`constant_token_control(decoder, features, period=8)` replaces each feature channel with its spatial mean. It then runs the evaluation-mode decoder and measures repeated within-block phases. Input features are B,C,Z,Y,X; output is one scalar 1,1,Z,Y,X cube. Nonzero phase variation with zero repeat error demonstrates a periodic decoder response despite spatially constant features.

`sample_block_means(logits, points_zyx, period=8)` averages logits over complete blocks, then samples native voxel-centre coordinates with `align_corners=False`. It returns unpooled and pooled sigmoid scores plus their common-support mask. For a 128-cubed output and period eight, the common domain is 3.5 through 123.5 in each axis. Keep that mask when comparing the two outputs. Pool logits before sigmoid, not probabilities afterwards.

`point_queries.query_points(head, tokens, embed_skips, points_zyx, spatial_shape, output_key="ink", chunk_size=4096)` calls the released DinoVol-style `forward_points` interface in bounded batches. All batches receive the same complete context, tokens and skip features. The return preserves query order and the final short batch. Model construction and normalization stay with the upstream implementation. Supply an evaluation-mode head whose queries are independent; test a new head against an unsplit forward before deployment.

## Retained results

The original eight-voxel diagnostic reproduced constant-token decoder texture exactly in three cases. On exposed PHerc1447 data, fixed block averaging changed AUROC from 0.542388 to 0.678861. A subsequent phase-trained decoder scored 0.369708 on a new reference cube, below the earlier linear readout's 0.381560. Those are separate experiments. The diagnostic exposed a real output artifact; the repaired decoder did not establish a better general reader.

The included JSON records preserve those numerical results. The public implementation generalizes the original fixed-128 sampling expression to a declared spatial shape. Current tests check physical block centres, common support, periodic versus constant responses, point ordering and incomplete batches. They use controlled synthetic arrays; they do not repeat model training.

The earlier local sheet-identity check used the released object head on 478 non-anchor points. Its sign agreed with all inspected band labels, versus 474 for the geometric-distance baseline. Both achieved AUROC 1.0. These were local, physically inspected labels, not independent physical ground truth. That historical result is retained as a usage finding, not a new object classifier supplied by this package.

Original model and head: [ScrollPrize/villa DinoVol at the tested revision](https://github.com/ScrollPrize/villa/tree/020f2754d3c32225e6a66fca36cab9dc027f01e4/dinovol/dinovol_2). The head interface is source-version-specific. The array diagnostic is independent of villa.

Code MIT. Derived numerical results: CC BY-NC 4.0 with the attribution in DATA_LICENSE.md. No model weights, raw CT, AI prompts or agent operating procedures are included.
