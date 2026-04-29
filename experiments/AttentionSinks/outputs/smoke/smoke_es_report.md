# Sliding Residual Contribution Report

Generated: 2026-04-25T19:25:27+00:00

## Research Context

This study treats attention-sink papers as hypothesis context, not as a direct answer for Gemma-4 on TowerBlocks. The motivating references were:

- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453): Initial sink tokens can stabilize streaming/windowed attention.
- [Why do LLMs attend to the first token?](https://arxiv.org/abs/2504.02732): Attention sinks can help prevent over-mixing and preserve information flow.
- [Attention Sinks Are Functionally Essential in Softmax Transformers: Theoretical Evidence](https://openreview.net/forum?id=TvG8VtbDLB): Softmax normalization can force stable sink-like behavior for default routing.
- [On the Existence and Behavior of Secondary Attention Sinks](https://openreview.net/forum?id=2DmhKvGLSC): Middle-layer activation dynamics can create secondary sinks beyond BOS.

None of those papers directly settle Gemma-4 sliding-layer behavior on TowerBlocks; the results below are measured from this repo's own instrumented runs.

## Method

- Model path: `google/gemma-4-E2B-it` via Hugging Face eager attention.
- Data: TowerBlocks local mirror, stratified over five fixed task/lang buckets.
- Eligibility: chat template applied, truncated to `max_length=640`, then keep prompts with `seq_len >= 513`.
- Global distribution metric: additive pre-norm contribution score `score_pre(q, k) = ||c_pre(q, k)||_2` over all late queries.
- Exact residual-effect metric: leave-one-out post-attention RMSNorm score on anchor queries only (`q=512`, midpoint late query, final query).
- Why the split: post-attention RMSNorm is nonlinear, so exact per-position leave-one-out for every late query would be prohibitively expensive at the 100-prompt target.
- Interpretation rule: high attention mass with low residual-effect score behaves like a probability reservoir; high exact residual-effect score means the token materially changes the residual update.

## Sample

| stratum | eligible | selected | shortfall |
| --- | --- | --- | --- |
| named_entity_recognition/es | 1 | 1 | 0 |

Scanned rows: `325`; matched rows: `325`; invalid: `0`; short after templating: `324`; without BOS eviction: `0`.

Sampling stopped early because `sampling_mode=first` filled every bucket before the end of the dataset.

## Validation

- Reconstruction error `||sum_k c_pre(q,k) - C(q)||_2`: `0.088655` (relative `0.001812`).
- Sliding-layer BOS visible rate for late queries: `0.000000`.
- Negative exact-score count: `0`.
- Capture shapes: `attn=[1, 8, 517, 517]`, `value=[1, 1, 517, 256]`, `per_head=[1, 517, 8, 256]`, `pre_norm=[1, 517, 1536]`.

## Global Findings

| type | rows | bos_share_pre | edge_share_pre | recent_share_pre | self_share_pre | middle_share_pre | top_bos_frac | top_edge_frac | top_recent_frac | top_self_frac | top_middle_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sliding | 140 | 0.000 | 0.005 | 0.283 | 0.041 | 0.671 | 0.000 | 0.000 | 0.371 | 0.036 | 0.593 |
| full | 35 | 0.096 | 0.133 | 0.108 | 0.025 | 0.637 | 0.286 | 0.114 | 0.086 | 0.029 | 0.486 |

| type | attn_bos | attn_edge | attn_recent | attn_self | attn_middle |
| --- | --- | --- | --- | --- | --- |
| sliding | 0.000 | 0.005 | 0.298 | 0.040 | 0.697 |
| full | 0.062 | 0.327 | 0.118 | 0.024 | 0.555 |

| type | anchor_rows | exact_bos | exact_edge | exact_recent | exact_self | exact_middle |
| --- | --- | --- | --- | --- | --- | --- |
| sliding | 84 | 0.000 | 0.007 | 0.318 | 0.056 | 0.619 |
| full | 21 | 0.124 | 0.143 | 0.149 | 0.043 | 0.540 |

Top-contributor histograms use the position with the largest `score_pre(q, k)` for each late query.

### Sliding Top Positions

| abs_pos | count | frac |
| --- | --- | --- |
| 226 | 76 | 0.543 |
| 511 | 12 | 0.086 |
| 515 | 12 | 0.086 |
| 510 | 11 | 0.079 |
| 514 | 8 | 0.057 |
| 512 | 4 | 0.029 |
| 286 | 4 | 0.029 |
| 513 | 4 | 0.029 |
| 509 | 4 | 0.029 |
| 288 | 2 | 0.014 |
| 501 | 1 | 0.007 |
| 506 | 1 | 0.007 |

| query_offset | count | frac |
| --- | --- | --- |
| 1 | 27 | 0.193 |
| 290 | 17 | 0.121 |
| 288 | 16 | 0.114 |
| 289 | 16 | 0.114 |
| 287 | 15 | 0.107 |
| 2 | 12 | 0.086 |
| 286 | 12 | 0.086 |
| 3 | 7 | 0.050 |
| 0 | 5 | 0.036 |
| 4 | 3 | 0.021 |
| 229 | 3 | 0.021 |
| 226 | 2 | 0.014 |

| edge_offset | count | frac |
| --- | --- | --- |
| 510 | 27 | 0.193 |
| 221 | 17 | 0.121 |
| 223 | 16 | 0.114 |
| 222 | 16 | 0.114 |
| 224 | 15 | 0.107 |
| 509 | 12 | 0.086 |
| 225 | 12 | 0.086 |
| 508 | 7 | 0.050 |
| 511 | 5 | 0.036 |
| 507 | 3 | 0.021 |
| 282 | 3 | 0.021 |
| 285 | 2 | 0.014 |

### Full Top Positions

| abs_pos | count | frac |
| --- | --- | --- |
| 0 | 10 | 0.286 |
| 286 | 4 | 0.114 |
| 1 | 3 | 0.086 |
| 278 | 2 | 0.057 |
| 136 | 2 | 0.057 |
| 515 | 2 | 0.057 |
| 293 | 2 | 0.057 |
| 288 | 2 | 0.057 |
| 282 | 2 | 0.057 |
| 289 | 2 | 0.057 |
| 509 | 1 | 0.029 |
| 2 | 1 | 0.029 |

| query_offset | count | frac |
| --- | --- | --- |
| 514 | 5 | 0.143 |
| 515 | 4 | 0.114 |
| 228 | 3 | 0.086 |
| 3 | 2 | 0.057 |
| 513 | 2 | 0.057 |
| 219 | 2 | 0.057 |
| 234 | 2 | 0.057 |
| 224 | 2 | 0.057 |
| 254 | 1 | 0.029 |
| 235 | 1 | 0.029 |
| 378 | 1 | 0.029 |
| 0 | 1 | 0.029 |

| edge_offset | count | frac |
| --- | --- | --- |
| 0 | 10 | 0.286 |
| 286 | 4 | 0.114 |
| 1 | 3 | 0.086 |
| 278 | 2 | 0.057 |
| 136 | 2 | 0.057 |
| 515 | 2 | 0.057 |
| 293 | 2 | 0.057 |
| 288 | 2 | 0.057 |
| 282 | 2 | 0.057 |
| 289 | 2 | 0.057 |
| 509 | 1 | 0.029 |
| 2 | 1 | 0.029 |

## Layer-Depth Findings

| bucket | rows | pre_edge | pre_recent | pre_middle | top_edge | top_recent | top_middle | exact_edge | exact_recent | exact_middle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_sliding | 40 | 0.008 | 0.411 | 0.528 | 0.000 | 0.775 | 0.125 | 0.012 | 0.447 | 0.467 |
| mid_sliding | 60 | 0.003 | 0.299 | 0.654 | 0.000 | 0.350 | 0.633 | 0.004 | 0.362 | 0.578 |
| late_sliding | 40 | 0.004 | 0.132 | 0.838 | 0.000 | 0.000 | 1.000 | 0.005 | 0.123 | 0.833 |
| full | 35 | 0.133 | 0.108 | 0.637 | 0.114 | 0.086 | 0.486 | 0.143 | 0.149 | 0.540 |

Candidate secondary sinks are prompt/layer pairs where one middle position dominates at least the configured threshold of late queries:

| stratum | layer | top_frac | pos | token | type |
| --- | --- | --- | --- | --- | --- |
| named_entity_recognition/es | 1 | 0.400 | 286 | model | content |
| named_entity_recognition/es | 2 | 0.200 | 288 | [ | punctuation/newline |
| named_entity_recognition/es | 5 | 0.200 | 288 | [ | punctuation/newline |
| named_entity_recognition/es | 9 | 0.200 | 258 |  ones | content |
| named_entity_recognition/es | 11 | 0.200 | 282 | : | punctuation/newline |
| named_entity_recognition/es | 13 | 0.200 | 286 | model | content |
| named_entity_recognition/es | 14 | 0.400 | 288 | [ | punctuation/newline |
| named_entity_recognition/es | 15 | 0.800 | 226 |  as | content |
| named_entity_recognition/es | 16 | 0.800 | 226 |  as | content |
| named_entity_recognition/es | 17 | 0.800 | 226 |  as | content |
| named_entity_recognition/es | 18 | 0.600 | 226 |  as | content |
| named_entity_recognition/es | 19 | 0.400 | 282 | : | punctuation/newline |

## Token Case Studies

These tables show representative prompts only: the first sampled prompt in each stratum, final anchor query (`q = S-1`), and representative layers `0, 15, 30, 34`. The full anchor-query dump is in `sliding_residual_top_positions.parquet`.

### named_entity_recognition/es

Representative prompt: `sample_id=0` `source_idx=324` `seq_len=517` `dataset=multiconer2023`.

| layer | type | key_pos | pos_group | token | tok_group | attn | score_pre | score_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 515 | recent_nonself | <turn|> | special/chat-marker | 0.229 | 24.867 | 372.686 |
| 0 | S | 286 | middle | model | content | 0.050 | 9.513 | 144.170 |
| 0 | S | 514 | recent_nonself | ")] | punctuation/newline | 0.057 | 7.538 | 109.797 |
| 0 | S | 510 | recent_nonself |  (". | punctuation/newline | 0.027 | 3.632 | 63.993 |
| 0 | S | 512 | recent_nonself |  " | punctuation/newline | 0.026 | 2.673 | 49.412 |
| 0 | S | 283 | middle | <turn|> | special/chat-marker | 0.025 | 3.009 | 35.930 |
| 15 | S | 226 | middle |  as | content | 0.248 | 6.395 | 5.329 |
| 15 | S | 515 | recent_nonself | <turn|> | special/chat-marker | 0.065 | 3.329 | 3.180 |
| 15 | S | 286 | middle | model | content | 0.050 | 2.768 | 3.055 |
| 15 | S | 312 | middle |  " | punctuation/newline | 0.061 | 1.304 | 1.481 |
| 15 | S | 514 | recent_nonself | ")] | punctuation/newline | 0.021 | 1.228 | 1.411 |
| 15 | S | 509 | recent_nonself | "), | punctuation/newline | 0.025 | 1.245 | 1.379 |
| 30 | S | 226 | middle |  as | content | 0.590 | 26.591 | 11.282 |
| 30 | S | 286 | middle | model | content | 0.062 | 2.706 | 1.117 |
| 30 | S | 289 | middle | (" | punctuation/newline | 0.018 | 1.853 | 0.796 |
| 30 | S | 515 | recent_nonself | <turn|> | special/chat-marker | 0.030 | 1.625 | 0.654 |
| 30 | S | 288 | middle | [ | punctuation/newline | 0.017 | 1.210 | 0.518 |
| 30 | S | 312 | middle |  " | punctuation/newline | 0.080 | 3.627 | 0.450 |
| 34 | F | 0 | bos | <bos> | special/chat-marker | 0.069 | 6.660 | 8.016 |
| 34 | F | 241 | middle |  B | content | 0.023 | 1.645 | 2.709 |
| 34 | F | 281 | middle | Answer | content | 0.041 | 1.886 | 2.659 |
| 34 | F | 1 | edge | <|turn> | special/chat-marker | 0.261 | 1.943 | 1.847 |
| 34 | F | 237 | middle | Besides | content | 0.013 | 1.022 | 1.548 |
| 34 | F | 286 | middle | model | content | 0.020 | 1.085 | 1.327 |

layer 0 (sliding) leans hardest on pos 515 [recent_nonself] token `<turn|>`; layer 15 (sliding) leans hardest on pos 226 [middle] token ` as`; layer 30 (sliding) leans hardest on pos 226 [middle] token ` as`; layer 34 (full) leans hardest on pos 0 [bos] token `<bos>`.

## Per-Layer Summary

| layer | type | rows | pre_bos | pre_edge | pre_recent | pre_self | pre_middle | exact_bos | exact_edge | exact_recent | exact_self | exact_middle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 5 | 0.000 | 0.004 | 0.604 | 0.020 | 0.372 | 0.000 | 0.005 | 0.538 | 0.047 | 0.410 |
| 1 | S | 5 | 0.000 | 0.012 | 0.374 | 0.027 | 0.587 | 0.000 | 0.021 | 0.345 | 0.054 | 0.580 |
| 2 | S | 5 | 0.000 | 0.008 | 0.424 | 0.044 | 0.523 | 0.000 | 0.009 | 0.549 | 0.075 | 0.368 |
| 3 | S | 5 | 0.000 | 0.012 | 0.383 | 0.044 | 0.561 | 0.000 | 0.014 | 0.382 | 0.044 | 0.560 |
| 4 | F | 5 | 0.081 | 0.141 | 0.088 | 0.017 | 0.674 | 0.057 | 0.196 | 0.126 | 0.038 | 0.583 |
| 5 | S | 5 | 0.000 | 0.010 | 0.301 | 0.053 | 0.636 | 0.000 | 0.016 | 0.358 | 0.066 | 0.560 |
| 6 | S | 5 | 0.000 | 0.009 | 0.515 | 0.111 | 0.366 | 0.000 | 0.017 | 0.553 | 0.128 | 0.302 |
| 7 | S | 5 | 0.000 | 0.006 | 0.322 | 0.063 | 0.609 | 0.000 | 0.010 | 0.413 | 0.093 | 0.484 |
| 8 | S | 5 | 0.000 | 0.004 | 0.367 | 0.056 | 0.572 | 0.000 | 0.008 | 0.437 | 0.084 | 0.470 |
| 9 | F | 5 | 0.025 | 0.087 | 0.101 | 0.052 | 0.735 | 0.041 | 0.129 | 0.170 | 0.083 | 0.577 |
| 10 | S | 5 | 0.000 | 0.001 | 0.438 | 0.086 | 0.475 | 0.000 | 0.002 | 0.530 | 0.098 | 0.369 |
| 11 | S | 5 | 0.000 | 0.002 | 0.270 | 0.050 | 0.679 | 0.000 | 0.003 | 0.353 | 0.086 | 0.559 |
| 12 | S | 5 | 0.000 | 0.002 | 0.419 | 0.060 | 0.519 | 0.000 | 0.003 | 0.552 | 0.086 | 0.359 |
| 13 | S | 5 | 0.000 | 0.001 | 0.393 | 0.043 | 0.562 | 0.000 | 0.003 | 0.398 | 0.060 | 0.539 |
| 14 | F | 5 | 0.034 | 0.101 | 0.202 | 0.046 | 0.617 | 0.062 | 0.074 | 0.261 | 0.081 | 0.522 |
| 15 | S | 5 | 0.000 | 0.004 | 0.234 | 0.032 | 0.730 | 0.000 | 0.005 | 0.319 | 0.048 | 0.627 |
| 16 | S | 5 | 0.000 | 0.004 | 0.220 | 0.024 | 0.752 | 0.000 | 0.007 | 0.296 | 0.028 | 0.669 |
| 17 | S | 5 | 0.000 | 0.004 | 0.247 | 0.020 | 0.729 | 0.000 | 0.005 | 0.375 | 0.016 | 0.604 |
| 18 | S | 5 | 0.000 | 0.002 | 0.357 | 0.021 | 0.620 | 0.000 | 0.003 | 0.434 | 0.028 | 0.535 |
| 19 | F | 5 | 0.045 | 0.119 | 0.151 | 0.031 | 0.654 | 0.078 | 0.168 | 0.219 | 0.045 | 0.490 |
| 20 | S | 5 | 0.000 | 0.004 | 0.282 | 0.038 | 0.676 | 0.000 | 0.004 | 0.320 | 0.047 | 0.630 |
| 21 | S | 5 | 0.000 | 0.003 | 0.162 | 0.052 | 0.783 | 0.000 | 0.004 | 0.180 | 0.059 | 0.756 |
| 22 | S | 5 | 0.000 | 0.002 | 0.266 | 0.041 | 0.691 | 0.000 | 0.002 | 0.307 | 0.045 | 0.646 |
| 23 | S | 5 | 0.000 | 0.002 | 0.300 | 0.068 | 0.630 | 0.000 | 0.002 | 0.277 | 0.073 | 0.647 |
| 24 | F | 5 | 0.077 | 0.132 | 0.116 | 0.016 | 0.658 | 0.106 | 0.109 | 0.137 | 0.028 | 0.621 |
| 25 | S | 5 | 0.000 | 0.008 | 0.113 | 0.012 | 0.868 | 0.000 | 0.011 | 0.141 | 0.016 | 0.832 |
| 26 | S | 5 | 0.000 | 0.004 | 0.190 | 0.077 | 0.730 | 0.000 | 0.003 | 0.138 | 0.099 | 0.760 |
| 27 | S | 5 | 0.000 | 0.003 | 0.122 | 0.008 | 0.867 | 0.000 | 0.003 | 0.109 | 0.009 | 0.880 |
| 28 | S | 5 | 0.000 | 0.003 | 0.134 | 0.040 | 0.823 | 0.000 | 0.003 | 0.139 | 0.053 | 0.806 |
| 29 | F | 5 | 0.116 | 0.224 | 0.028 | 0.008 | 0.624 | 0.147 | 0.198 | 0.033 | 0.008 | 0.614 |
| 30 | S | 5 | 0.000 | 0.003 | 0.164 | 0.033 | 0.800 | 0.000 | 0.003 | 0.133 | 0.071 | 0.792 |
| 31 | S | 5 | 0.000 | 0.003 | 0.080 | 0.006 | 0.911 | 0.000 | 0.005 | 0.085 | 0.008 | 0.902 |
| 32 | S | 5 | 0.000 | 0.004 | 0.132 | 0.011 | 0.853 | 0.000 | 0.006 | 0.124 | 0.015 | 0.855 |
| 33 | S | 5 | 0.000 | 0.006 | 0.121 | 0.019 | 0.854 | 0.000 | 0.009 | 0.117 | 0.036 | 0.838 |
| 34 | F | 5 | 0.296 | 0.127 | 0.070 | 0.007 | 0.499 | 0.379 | 0.127 | 0.099 | 0.021 | 0.375 |

## Conclusion

Full-attention layers still provide the control case: BOS remains visible and usually captures a sizeable share of both attention mass and residual-effect score.

In this run, sliding layers do not recreate a strong moving-edge BOS substitute. The leftover routing is better described as a distributed interior/recent pattern than as a single backup sink.

Middle-layer secondary-sink candidates did appear in some prompt/layer pairs, which is consistent with the possibility that activation dynamics can create temporary non-BOS anchors.

The practical answer to the no-op question is therefore layer-dependent: full layers can still offload onto BOS, but sliding layers mostly have to distribute the branch over visible content/special tokens that survive the mask, with exact anchor scores revealing which of those positions materially change the residual update.
