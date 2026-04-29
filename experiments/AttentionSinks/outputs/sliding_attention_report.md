# Sliding Attention Sink Report

Generated: 2026-04-25T23:30:36

## Setup

- Model: `google/gemma-4-E2B-it`
- Source: `towerblocks` from `experiments/AttentionSinks/data/TowerBlocks-v0.1`
- Device: `mps`
- Dtype: `bfloat16`
- Limit: `5` qualifying prompts
- Max length: `640`
- Late-query regime: `q >= 512`
- Filters: `lang=en`, `task=named_entity_recognition`, `split=None`, `dataset=None`

## Sample

- Scanned rows: `661`
- Kept prompts: `5`
- Skipped invalid: `0`
- Skipped short: `656`
- Skipped without BOS eviction: `0`

| source_idx | lang | task | split | dataset | seq_len |
| --- | --- | --- | --- | --- | --- |
| 3382 | en | named_entity_recognition | dev | multiconer2023 | 640 |
| 3383 | en | named_entity_recognition | dev | multiconer2023 | 640 |
| 3385 | en | named_entity_recognition | dev | multiconer2023 | 640 |
| 3386 | en | named_entity_recognition | dev | multiconer2023 | 640 |
| 3389 | en | named_entity_recognition | dev | multiconer2023 | 595 |

## Layer-Type Summary

| type | rows | bos_mass | edge_mass | recent_mass | middle_mass | self_mass | argmax_edge_frac | argmax_self_frac | argmax_edge_offset | norm_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 33320 | 0.268 | 0.268 | 0.218 | 0.514 | 0.038 | 0.528 | 0.081 | 206.324 | 0.589 |
| sliding | 133280 | 0.000 | 0.002 | 0.416 | 0.582 | 0.043 | 0.000 | 0.065 | 375.854 | 0.518 |

Interpretation:
- Full layers place `0.268` mass on tokens `[0..3]` for late queries, while sliding layers place `0.000` there once BOS is evicted.
- Sliding layers place `0.002` mass on the moving window edge and `0.416` on the recent tail.
- The top attention destination lands on the edge `0.000` of the time in sliding layers versus `0.528` in full layers.
- This sample does not support a strong moving-edge sink. Sliding attention mostly shifts into recent and interior tokens instead.

## Per-Layer Summary

| layer | type | bos_mass | edge_mass | recent_mass | middle_mass | self_mass | argmax_edge_frac | argmax_self_frac | argmax_edge_offset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 0.000 | 0.002 | 0.642 | 0.356 | 0.024 | 0.002 | 0.041 | 490.514 |
| 1 | S | 0.000 | 0.004 | 0.546 | 0.450 | 0.023 | 0.003 | 0.015 | 465.474 |
| 2 | S | 0.000 | 0.003 | 0.627 | 0.369 | 0.029 | 0.000 | 0.033 | 468.346 |
| 3 | S | 0.000 | 0.002 | 0.609 | 0.388 | 0.026 | 0.001 | 0.020 | 461.015 |
| 4 | F | 0.239 | 0.239 | 0.217 | 0.545 | 0.018 | 0.388 | 0.011 | 219.736 |
| 5 | S | 0.001 | 0.004 | 0.422 | 0.574 | 0.030 | 0.002 | 0.032 | 443.110 |
| 6 | S | 0.000 | 0.002 | 0.717 | 0.282 | 0.107 | 0.000 | 0.238 | 493.775 |
| 7 | S | 0.000 | 0.003 | 0.498 | 0.499 | 0.040 | 0.001 | 0.079 | 463.803 |
| 8 | S | 0.000 | 0.003 | 0.509 | 0.489 | 0.055 | 0.000 | 0.159 | 472.587 |
| 9 | F | 0.151 | 0.151 | 0.231 | 0.618 | 0.076 | 0.375 | 0.155 | 258.204 |
| 10 | S | 0.000 | 0.002 | 0.507 | 0.491 | 0.060 | 0.001 | 0.102 | 431.525 |
| 11 | S | 0.000 | 0.002 | 0.381 | 0.617 | 0.025 | 0.000 | 0.033 | 426.163 |
| 12 | S | 0.000 | 0.001 | 0.453 | 0.546 | 0.063 | 0.000 | 0.097 | 390.993 |
| 13 | S | 0.000 | 0.001 | 0.520 | 0.479 | 0.066 | 0.000 | 0.111 | 413.457 |
| 14 | F | 0.098 | 0.098 | 0.386 | 0.516 | 0.070 | 0.261 | 0.221 | 393.944 |
| 15 | S | 0.000 | 0.002 | 0.378 | 0.620 | 0.039 | 0.000 | 0.053 | 346.596 |
| 16 | S | 0.000 | 0.002 | 0.407 | 0.591 | 0.044 | 0.000 | 0.079 | 376.317 |
| 17 | S | 0.000 | 0.002 | 0.356 | 0.642 | 0.036 | 0.000 | 0.049 | 346.188 |
| 18 | S | 0.000 | 0.002 | 0.468 | 0.530 | 0.056 | 0.000 | 0.095 | 390.946 |
| 19 | F | 0.243 | 0.243 | 0.282 | 0.475 | 0.055 | 0.478 | 0.114 | 259.707 |
| 20 | S | 0.000 | 0.001 | 0.491 | 0.508 | 0.067 | 0.000 | 0.086 | 370.421 |
| 21 | S | 0.000 | 0.002 | 0.408 | 0.590 | 0.083 | 0.000 | 0.119 | 376.182 |
| 22 | S | 0.000 | 0.001 | 0.455 | 0.544 | 0.049 | 0.000 | 0.070 | 354.179 |
| 23 | S | 0.000 | 0.001 | 0.503 | 0.496 | 0.081 | 0.000 | 0.123 | 378.151 |
| 24 | F | 0.289 | 0.289 | 0.224 | 0.487 | 0.034 | 0.605 | 0.059 | 185.653 |
| 25 | S | 0.000 | 0.003 | 0.239 | 0.758 | 0.025 | 0.000 | 0.029 | 293.445 |
| 26 | S | 0.000 | 0.002 | 0.322 | 0.676 | 0.063 | 0.000 | 0.086 | 298.578 |
| 27 | S | 0.000 | 0.002 | 0.216 | 0.782 | 0.014 | 0.000 | 0.003 | 264.326 |
| 28 | S | 0.000 | 0.001 | 0.245 | 0.754 | 0.032 | 0.000 | 0.026 | 274.389 |
| 29 | F | 0.374 | 0.374 | 0.110 | 0.516 | 0.008 | 0.681 | 0.003 | 97.961 |
| 30 | S | 0.000 | 0.002 | 0.191 | 0.807 | 0.018 | 0.000 | 0.015 | 272.935 |
| 31 | S | 0.000 | 0.002 | 0.162 | 0.836 | 0.009 | 0.000 | 0.002 | 267.166 |
| 32 | S | 0.000 | 0.002 | 0.148 | 0.850 | 0.015 | 0.000 | 0.007 | 243.658 |
| 33 | S | 0.000 | 0.002 | 0.218 | 0.780 | 0.025 | 0.000 | 0.016 | 249.674 |
| 34 | F | 0.484 | 0.484 | 0.078 | 0.438 | 0.006 | 0.906 | 0.000 | 29.067 |

## Takeaways

- The classical BOS sink is present in full-attention layers and disappears in sliding-attention layers once BOS is masked out.
- On this run, sliding layers do not form a strong moving-edge sink. Their mass concentrates much more on recent/interior tokens than on the window edge.
- `argmax_edge_frac` and `argmax_edge_offset` are the clearest checks for whether the sink truly tracks the sliding-window boundary.
