# Sliding Residual Contribution Report

Generated: 2026-04-25T19:28:50+00:00

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
| chat/en | 152212 | 1 | 0 |
| named_entity_recognition/en | 299 | 1 | 0 |
| named_entity_recognition/es | 301 | 1 | 0 |
| machine_translation/en-de | 29 | 1 | 0 |
| machine_translation_evaluation/zh_en | 1 | 1 | 0 |

Scanned rows: `320140`; matched rows: `173167`; invalid: `0`; short after templating: `20325`; without BOS eviction: `0`.

Sampling stopped early because `sampling_mode=first` filled every bucket before the end of the dataset.

## Validation

- Reconstruction error `||sum_k c_pre(q,k) - C(q)||_2`: `0.184332` (relative `0.002284`).
- Sliding-layer BOS visible rate for late queries: `0.000000`.
- Negative exact-score count: `0`.
- Capture shapes: `attn=[1, 8, 640, 640]`, `value=[1, 1, 640, 256]`, `per_head=[1, 640, 8, 256]`, `pre_norm=[1, 640, 1536]`.

## Global Findings

| type | rows | bos_share_pre | edge_share_pre | recent_share_pre | self_share_pre | middle_share_pre | top_bos_frac | top_edge_frac | top_recent_frac | top_self_frac | top_middle_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sliding | 12320 | 0.000 | 0.002 | 0.364 | 0.049 | 0.585 | 0.000 | 0.000 | 0.443 | 0.101 | 0.456 |
| full | 3080 | 0.103 | 0.097 | 0.155 | 0.037 | 0.607 | 0.308 | 0.064 | 0.130 | 0.122 | 0.376 |

| type | attn_bos | attn_edge | attn_recent | attn_self | attn_middle |
| --- | --- | --- | --- | --- | --- |
| sliding | 0.000 | 0.002 | 0.379 | 0.041 | 0.619 |
| full | 0.066 | 0.250 | 0.171 | 0.030 | 0.579 |

| type | anchor_rows | exact_bos | exact_edge | exact_recent | exact_self | exact_middle |
| --- | --- | --- | --- | --- | --- | --- |
| sliding | 420 | 0.000 | 0.004 | 0.345 | 0.074 | 0.577 |
| full | 105 | 0.128 | 0.118 | 0.162 | 0.062 | 0.529 |

Top-contributor histograms use the position with the largest `score_pre(q, k)` for each late query.

### Sliding Top Positions

| abs_pos | count | frac |
| --- | --- | --- |
| 325 | 1390 | 0.113 |
| 259 | 900 | 0.073 |
| 243 | 895 | 0.073 |
| 359 | 582 | 0.047 |
| 163 | 518 | 0.042 |
| 375 | 280 | 0.023 |
| 398 | 277 | 0.022 |
| 539 | 130 | 0.011 |
| 550 | 130 | 0.011 |
| 566 | 118 | 0.010 |
| 514 | 112 | 0.009 |
| 528 | 111 | 0.009 |

| query_offset | count | frac |
| --- | --- | --- |
| 1 | 2670 | 0.217 |
| 0 | 1239 | 0.101 |
| 2 | 884 | 0.072 |
| 3 | 516 | 0.042 |
| 4 | 274 | 0.022 |
| 5 | 230 | 0.019 |
| 6 | 130 | 0.011 |
| 7 | 106 | 0.009 |
| 8 | 98 | 0.008 |
| 260 | 76 | 0.006 |
| 9 | 66 | 0.005 |
| 12 | 55 | 0.004 |

| edge_offset | count | frac |
| --- | --- | --- |
| 510 | 2670 | 0.217 |
| 511 | 1239 | 0.101 |
| 509 | 884 | 0.072 |
| 508 | 516 | 0.042 |
| 507 | 274 | 0.022 |
| 506 | 230 | 0.019 |
| 505 | 130 | 0.011 |
| 504 | 106 | 0.009 |
| 503 | 98 | 0.008 |
| 251 | 76 | 0.006 |
| 502 | 66 | 0.005 |
| 499 | 55 | 0.004 |

### Full Top Positions

| abs_pos | count | frac |
| --- | --- | --- |
| 0 | 949 | 0.308 |
| 1 | 168 | 0.055 |
| 544 | 29 | 0.009 |
| 606 | 24 | 0.008 |
| 573 | 24 | 0.008 |
| 577 | 24 | 0.008 |
| 2 | 23 | 0.007 |
| 567 | 23 | 0.007 |
| 224 | 22 | 0.007 |
| 551 | 22 | 0.007 |
| 555 | 22 | 0.007 |
| 563 | 21 | 0.007 |

| query_offset | count | frac |
| --- | --- | --- |
| 0 | 377 | 0.122 |
| 259 | 176 | 0.057 |
| 1 | 135 | 0.044 |
| 260 | 69 | 0.022 |
| 2 | 67 | 0.022 |
| 49 | 33 | 0.011 |
| 307 | 28 | 0.009 |
| 258 | 27 | 0.009 |
| 4 | 25 | 0.008 |
| 3 | 25 | 0.008 |
| 535 | 25 | 0.008 |
| 48 | 21 | 0.007 |

| edge_offset | count | frac |
| --- | --- | --- |
| 0 | 949 | 0.308 |
| 1 | 168 | 0.055 |
| 544 | 29 | 0.009 |
| 606 | 24 | 0.008 |
| 573 | 24 | 0.008 |
| 577 | 24 | 0.008 |
| 2 | 23 | 0.007 |
| 567 | 23 | 0.007 |
| 224 | 22 | 0.007 |
| 551 | 22 | 0.007 |
| 555 | 22 | 0.007 |
| 563 | 21 | 0.007 |

## Layer-Depth Findings

| bucket | rows | pre_edge | pre_recent | pre_middle | top_edge | top_recent | top_middle | exact_edge | exact_recent | exact_middle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_sliding | 3520 | 0.003 | 0.516 | 0.439 | 0.000 | 0.840 | 0.093 | 0.006 | 0.510 | 0.417 |
| mid_sliding | 5280 | 0.001 | 0.381 | 0.553 | 0.000 | 0.456 | 0.363 | 0.003 | 0.367 | 0.531 |
| late_sliding | 3520 | 0.002 | 0.186 | 0.780 | 0.000 | 0.028 | 0.959 | 0.003 | 0.148 | 0.805 |
| full | 3080 | 0.097 | 0.155 | 0.607 | 0.064 | 0.130 | 0.376 | 0.118 | 0.162 | 0.529 |

Candidate secondary sinks are prompt/layer pairs where one middle position dominates at least the configured threshold of late queries:

| stratum | layer | top_frac | pos | token | type |
| --- | --- | --- | --- | --- | --- |
| chat/en | 16 | 0.219 | 243 | - | punctuation/newline |
| chat/en | 17 | 0.258 | 243 | - | punctuation/newline |
| chat/en | 18 | 0.422 | 243 | - | punctuation/newline |
| chat/en | 22 | 0.531 | 243 | - | punctuation/newline |
| chat/en | 23 | 0.211 | 243 | - | punctuation/newline |
| chat/en | 25 | 0.508 | 243 | - | punctuation/newline |
| chat/en | 26 | 0.602 | 243 | - | punctuation/newline |
| chat/en | 27 | 0.617 | 243 | - | punctuation/newline |
| chat/en | 28 | 0.578 | 243 | - | punctuation/newline |
| chat/en | 30 | 0.609 | 243 | - | punctuation/newline |
| chat/en | 31 | 0.586 | 243 | - | punctuation/newline |
| chat/en | 32 | 0.570 | 243 | - | punctuation/newline |

## Token Case Studies

These tables show representative prompts only: the first sampled prompt in each stratum, final anchor query (`q = S-1`), and representative layers `0, 15, 30, 34`. The full anchor-query dump is in `sliding_residual_top_positions.parquet`.

### chat/en

Representative prompt: `sample_id=0` `source_idx=5312` `seq_len=640` `dataset=ultrachat_filtered`.

| layer | type | key_pos | pos_group | token | tok_group | attn | score_pre | score_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 638 | recent_nonself |  ' | punctuation/newline | 0.241 | 24.714 | 246.197 |
| 0 | S | 637 | recent_nonself |  select | content | 0.070 | 7.319 | 82.740 |
| 0 | S | 636 | recent_nonself | , | punctuation/newline | 0.038 | 3.655 | 46.771 |
| 0 | S | 634 | recent_nonself |  If | content | 0.027 | 3.695 | 34.391 |
| 0 | S | 633 | recent_nonself | . | punctuation/newline | 0.026 | 2.313 | 17.688 |
| 0 | S | 628 | recent_nonself | Image | content | 0.011 | 2.617 | 16.488 |
| 15 | S | 638 | recent_nonself |  ' | punctuation/newline | 0.051 | 3.792 | 5.507 |
| 15 | S | 637 | recent_nonself |  select | content | 0.041 | 2.724 | 3.691 |
| 15 | S | 359 | middle | ' | punctuation/newline | 0.132 | 3.014 | 3.362 |
| 15 | S | 630 | recent_nonself | '. | punctuation/newline | 0.038 | 1.560 | 1.922 |
| 15 | S | 634 | recent_nonself |  If | content | 0.025 | 1.358 | 1.830 |
| 15 | S | 588 | middle | 1 | content | 0.010 | 0.974 | 1.419 |
| 30 | S | 359 | middle | ' | punctuation/newline | 0.337 | 16.067 | 3.212 |
| 30 | S | 638 | recent_nonself |  ' | punctuation/newline | 0.047 | 5.197 | 1.972 |
| 30 | S | 602 | middle |  the | content | 0.163 | 8.318 | 1.204 |
| 30 | S | 525 | middle |  image | content | 0.035 | 2.324 | 0.807 |
| 30 | S | 639 | self | Show | content | 0.017 | 1.385 | 0.496 |
| 30 | S | 622 | recent_nonself |  ' | punctuation/newline | 0.010 | 1.243 | 0.453 |
| 34 | F | 0 | bos | <bos> | special/chat-marker | 0.032 | 3.699 | 6.339 |
| 34 | F | 332 | middle |  secondary | content | 0.061 | 2.391 | 3.632 |
| 34 | F | 1 | edge | <|turn> | special/chat-marker | 0.233 | 3.072 | 3.188 |
| 34 | F | 100 | middle |  secondary | content | 0.028 | 1.246 | 1.752 |
| 34 | F | 68 | middle |  secondary | content | 0.015 | 0.930 | 1.369 |
| 34 | F | 2 | edge | user | content | 0.156 | 1.154 | 1.279 |

layer 0 (sliding) leans hardest on pos 638 [recent_nonself] token ` '`; layer 15 (sliding) leans hardest on pos 638 [recent_nonself] token ` '`; layer 30 (sliding) leans hardest on pos 359 [middle] token `'`; layer 34 (full) leans hardest on pos 0 [bos] token `<bos>`.

### named_entity_recognition/en

Representative prompt: `sample_id=1` `source_idx=3382` `seq_len=640` `dataset=multiconer2023`.

| layer | type | key_pos | pos_group | token | tok_group | attn | score_pre | score_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 638 | recent_nonself | split | content | 0.091 | 14.790 | 271.764 |
| 0 | S | 637 | recent_nonself |  (" | punctuation/newline | 0.195 | 19.407 | 233.316 |
| 0 | S | 636 | recent_nonself | "), | punctuation/newline | 0.090 | 7.244 | 101.900 |
| 0 | S | 629 | recent_nonself |  (" | punctuation/newline | 0.046 | 5.314 | 58.429 |
| 0 | S | 639 | self | ", | punctuation/newline | 0.032 | 2.712 | 31.166 |
| 0 | S | 628 | recent_nonself | "), | punctuation/newline | 0.027 | 2.541 | 28.378 |
| 15 | S | 639 | self | ", | punctuation/newline | 0.093 | 6.865 | 7.906 |
| 15 | S | 633 | recent_nonself | I | content | 0.052 | 3.292 | 3.719 |
| 15 | S | 375 | middle | - | punctuation/newline | 0.108 | 2.447 | 2.306 |
| 15 | S | 474 | middle |  " | punctuation/newline | 0.016 | 1.973 | 2.240 |
| 15 | S | 636 | recent_nonself | "), | punctuation/newline | 0.036 | 1.950 | 2.177 |
| 15 | S | 618 | recent_nonself | Target | content | 0.024 | 1.924 | 2.110 |
| 30 | S | 259 | middle |  " | punctuation/newline | 0.270 | 11.725 | 2.876 |
| 30 | S | 621 | recent_nonself | (" | punctuation/newline | 0.046 | 5.150 | 2.313 |
| 30 | S | 375 | middle | - | punctuation/newline | 0.184 | 8.227 | 1.766 |
| 30 | S | 265 | middle |  " | punctuation/newline | 0.180 | 7.746 | 1.603 |
| 30 | S | 639 | self | ", | punctuation/newline | 0.038 | 3.037 | 1.243 |
| 30 | S | 637 | recent_nonself |  (" | punctuation/newline | 0.017 | 1.708 | 0.730 |
| 34 | F | 0 | bos | <bos> | special/chat-marker | 0.032 | 2.946 | 5.083 |
| 34 | F | 93 | middle |  (" | punctuation/newline | 0.017 | 1.916 | 3.456 |
| 34 | F | 1 | edge | <|turn> | special/chat-marker | 0.260 | 1.986 | 3.170 |
| 34 | F | 85 | middle | (" | punctuation/newline | 0.011 | 1.154 | 2.020 |
| 34 | F | 2 | edge | user | content | 0.188 | 1.123 | 1.466 |
| 34 | F | 98 | middle |  " | punctuation/newline | 0.012 | 0.665 | 1.157 |

layer 0 (sliding) leans hardest on pos 638 [recent_nonself] token `split`; layer 15 (sliding) leans hardest on pos 639 [self] token `",`; layer 30 (sliding) leans hardest on pos 259 [middle] token ` "`; layer 34 (full) leans hardest on pos 0 [bos] token `<bos>`.

### named_entity_recognition/es

Representative prompt: `sample_id=2` `source_idx=324` `seq_len=517` `dataset=multiconer2023`.

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

### machine_translation/en-de

Representative prompt: `sample_id=3` `source_idx=178040` `seq_len=627` `dataset=gender-eval-sentences`.

| layer | type | key_pos | pos_group | token | tok_group | attn | score_pre | score_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 625 | recent_nonself | <turn|> | special/chat-marker | 0.248 | 26.076 | 418.064 |
| 0 | S | 580 | middle | model | content | 0.061 | 9.058 | 163.058 |
| 0 | S | 577 | middle | <turn|> | special/chat-marker | 0.114 | 11.769 | 159.719 |
| 0 | S | 624 | recent_nonself | . | punctuation/newline | 0.042 | 4.178 | 67.577 |
| 0 | S | 529 | middle | \n | punctuation/newline | 0.019 | 2.171 | 43.728 |
| 0 | S | 581 | middle | \n | punctuation/newline | 0.033 | 2.808 | 36.659 |
| 15 | S | 580 | middle | model | content | 0.155 | 7.531 | 6.502 |
| 15 | S | 625 | recent_nonself | <turn|> | special/chat-marker | 0.133 | 7.123 | 6.336 |
| 15 | S | 325 | middle |   | punctuation/newline | 0.241 | 6.044 | 4.601 |
| 15 | S | 577 | middle | <turn|> | special/chat-marker | 0.067 | 3.418 | 3.271 |
| 15 | S | 530 | middle | Source | content | 0.025 | 1.594 | 1.553 |
| 15 | S | 575 | middle | Target | content | 0.032 | 1.755 | 1.521 |
| 30 | S | 325 | middle |   | punctuation/newline | 0.490 | 20.804 | 9.272 |
| 30 | S | 580 | middle | model | content | 0.186 | 7.046 | 3.424 |
| 30 | S | 625 | recent_nonself | <turn|> | special/chat-marker | 0.058 | 3.929 | 1.904 |
| 30 | S | 576 | middle | : | punctuation/newline | 0.019 | 1.654 | 0.795 |
| 30 | S | 575 | middle | Target | content | 0.022 | 1.532 | 0.745 |
| 30 | S | 463 | middle |   | punctuation/newline | 0.076 | 3.406 | 0.706 |
| 34 | F | 0 | bos | <bos> | special/chat-marker | 0.103 | 10.873 | 9.337 |
| 34 | F | 95 | middle | Target | content | 0.028 | 1.797 | 1.855 |
| 34 | F | 580 | middle | model | content | 0.029 | 1.813 | 1.622 |
| 34 | F | 1 | edge | <|turn> | special/chat-marker | 0.224 | 1.948 | 1.175 |
| 34 | F | 575 | middle | Target | content | 0.017 | 0.935 | 1.009 |
| 34 | F | 2 | edge | user | content | 0.203 | 1.299 | 0.818 |

layer 0 (sliding) leans hardest on pos 625 [recent_nonself] token `<turn|>`; layer 15 (sliding) leans hardest on pos 580 [middle] token `model`; layer 30 (sliding) leans hardest on pos 325 [middle] token `▁`; layer 34 (full) leans hardest on pos 0 [bos] token `<bos>`.

### machine_translation_evaluation/zh_en

Representative prompt: `sample_id=4` `source_idx=320139` `seq_len=576` `dataset=google_mqm`.

| layer | type | key_pos | pos_group | token | tok_group | attn | score_pre | score_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 574 | recent_nonself | <turn|> | special/chat-marker | 0.281 | 27.069 | 442.813 |
| 0 | S | 411 | middle | model | content | 0.067 | 11.195 | 177.682 |
| 0 | S | 408 | middle | <turn|> | special/chat-marker | 0.050 | 6.208 | 73.897 |
| 0 | S | 573 | recent_nonself | . | punctuation/newline | 0.043 | 4.196 | 66.531 |
| 0 | S | 412 | middle | \n | punctuation/newline | 0.020 | 2.379 | 45.808 |
| 0 | S | 565 | recent_nonself |  Beijing | content | 0.017 | 2.079 | 25.893 |
| 15 | S | 163 | middle | ' | punctuation/newline | 0.191 | 4.764 | 3.780 |
| 15 | S | 500 | middle |  the | content | 0.057 | 3.275 | 3.403 |
| 15 | S | 411 | middle | model | content | 0.056 | 3.056 | 3.156 |
| 15 | S | 549 | recent_nonself |  the | content | 0.047 | 3.094 | 3.008 |
| 15 | S | 574 | recent_nonself | <turn|> | special/chat-marker | 0.061 | 3.223 | 2.906 |
| 15 | S | 407 | middle | . | punctuation/newline | 0.024 | 1.531 | 1.843 |
| 30 | S | 163 | middle | ' | punctuation/newline | 0.445 | 20.222 | 6.591 |
| 30 | S | 411 | middle | model | content | 0.086 | 3.543 | 1.569 |
| 30 | S | 398 | middle | - | punctuation/newline | 0.177 | 8.298 | 1.404 |
| 30 | S | 469 | middle |  < | content | 0.023 | 1.926 | 0.869 |
| 30 | S | 412 | middle | \n | punctuation/newline | 0.015 | 1.114 | 0.498 |
| 30 | S | 574 | recent_nonself | <turn|> | special/chat-marker | 0.023 | 1.107 | 0.484 |
| 34 | F | 0 | bos | <bos> | special/chat-marker | 0.064 | 6.840 | 9.072 |
| 34 | F | 315 | middle | Errors | content | 0.071 | 2.680 | 3.762 |
| 34 | F | 351 | middle | major | content | 0.037 | 2.337 | 3.168 |
| 34 | F | 157 | middle | Translation | content | 0.027 | 2.228 | 3.070 |
| 34 | F | 1 | edge | <|turn> | special/chat-marker | 0.212 | 2.797 | 2.130 |
| 34 | F | 327 | middle | minor | content | 0.018 | 1.205 | 1.721 |

layer 0 (sliding) leans hardest on pos 574 [recent_nonself] token `<turn|>`; layer 15 (sliding) leans hardest on pos 163 [middle] token `'`; layer 30 (sliding) leans hardest on pos 163 [middle] token `'`; layer 34 (full) leans hardest on pos 0 [bos] token `<bos>`.

## Per-Layer Summary

| layer | type | rows | pre_bos | pre_edge | pre_recent | pre_self | pre_middle | exact_bos | exact_edge | exact_recent | exact_self | exact_middle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | S | 440 | 0.000 | 0.002 | 0.570 | 0.018 | 0.409 | 0.000 | 0.002 | 0.528 | 0.037 | 0.432 |
| 1 | S | 440 | 0.000 | 0.003 | 0.501 | 0.018 | 0.478 | 0.000 | 0.009 | 0.378 | 0.030 | 0.583 |
| 2 | S | 440 | 0.000 | 0.002 | 0.653 | 0.040 | 0.305 | 0.000 | 0.004 | 0.708 | 0.057 | 0.231 |
| 3 | S | 440 | 0.000 | 0.002 | 0.542 | 0.022 | 0.433 | 0.000 | 0.008 | 0.482 | 0.032 | 0.478 |
| 4 | F | 440 | 0.095 | 0.102 | 0.160 | 0.013 | 0.630 | 0.079 | 0.164 | 0.192 | 0.017 | 0.549 |
| 5 | S | 440 | 0.000 | 0.004 | 0.403 | 0.037 | 0.557 | 0.000 | 0.009 | 0.426 | 0.043 | 0.522 |
| 6 | S | 440 | 0.000 | 0.001 | 0.583 | 0.107 | 0.308 | 0.000 | 0.005 | 0.616 | 0.143 | 0.236 |
| 7 | S | 440 | 0.000 | 0.003 | 0.446 | 0.050 | 0.501 | 0.000 | 0.005 | 0.480 | 0.093 | 0.422 |
| 8 | S | 440 | 0.000 | 0.003 | 0.434 | 0.046 | 0.517 | 0.000 | 0.005 | 0.460 | 0.100 | 0.435 |
| 9 | F | 440 | 0.036 | 0.062 | 0.120 | 0.073 | 0.708 | 0.058 | 0.096 | 0.128 | 0.126 | 0.592 |
| 10 | S | 440 | 0.000 | 0.002 | 0.453 | 0.074 | 0.472 | 0.000 | 0.003 | 0.495 | 0.121 | 0.382 |
| 11 | S | 440 | 0.000 | 0.001 | 0.343 | 0.029 | 0.626 | 0.000 | 0.006 | 0.385 | 0.054 | 0.555 |
| 12 | S | 440 | 0.000 | 0.001 | 0.443 | 0.058 | 0.498 | 0.000 | 0.003 | 0.499 | 0.092 | 0.406 |
| 13 | S | 440 | 0.000 | 0.001 | 0.444 | 0.062 | 0.493 | 0.000 | 0.002 | 0.451 | 0.088 | 0.460 |
| 14 | F | 440 | 0.020 | 0.060 | 0.298 | 0.055 | 0.567 | 0.041 | 0.059 | 0.313 | 0.113 | 0.474 |
| 15 | S | 440 | 0.000 | 0.001 | 0.364 | 0.043 | 0.592 | 0.000 | 0.003 | 0.344 | 0.074 | 0.580 |
| 16 | S | 440 | 0.000 | 0.001 | 0.339 | 0.043 | 0.616 | 0.000 | 0.003 | 0.302 | 0.076 | 0.619 |
| 17 | S | 440 | 0.000 | 0.002 | 0.306 | 0.039 | 0.653 | 0.000 | 0.004 | 0.301 | 0.077 | 0.618 |
| 18 | S | 440 | 0.000 | 0.001 | 0.388 | 0.063 | 0.549 | 0.000 | 0.002 | 0.377 | 0.113 | 0.508 |
| 19 | F | 440 | 0.046 | 0.102 | 0.207 | 0.060 | 0.585 | 0.068 | 0.112 | 0.209 | 0.096 | 0.514 |
| 20 | S | 440 | 0.000 | 0.001 | 0.411 | 0.088 | 0.500 | 0.000 | 0.002 | 0.334 | 0.122 | 0.542 |
| 21 | S | 440 | 0.000 | 0.001 | 0.305 | 0.131 | 0.562 | 0.000 | 0.002 | 0.252 | 0.164 | 0.582 |
| 22 | S | 440 | 0.000 | 0.001 | 0.351 | 0.053 | 0.595 | 0.000 | 0.001 | 0.310 | 0.072 | 0.616 |
| 23 | S | 440 | 0.000 | 0.001 | 0.421 | 0.103 | 0.476 | 0.000 | 0.002 | 0.355 | 0.143 | 0.500 |
| 24 | F | 440 | 0.088 | 0.090 | 0.149 | 0.039 | 0.634 | 0.116 | 0.105 | 0.142 | 0.062 | 0.575 |
| 25 | S | 440 | 0.000 | 0.003 | 0.194 | 0.027 | 0.777 | 0.000 | 0.007 | 0.166 | 0.042 | 0.785 |
| 26 | S | 440 | 0.000 | 0.002 | 0.242 | 0.079 | 0.677 | 0.000 | 0.002 | 0.166 | 0.117 | 0.715 |
| 27 | S | 440 | 0.000 | 0.002 | 0.178 | 0.020 | 0.801 | 0.000 | 0.002 | 0.148 | 0.021 | 0.830 |
| 28 | S | 440 | 0.000 | 0.001 | 0.221 | 0.050 | 0.728 | 0.000 | 0.002 | 0.178 | 0.058 | 0.762 |
| 29 | F | 440 | 0.163 | 0.163 | 0.063 | 0.006 | 0.605 | 0.160 | 0.180 | 0.069 | 0.006 | 0.585 |
| 30 | S | 440 | 0.000 | 0.002 | 0.166 | 0.019 | 0.813 | 0.000 | 0.002 | 0.137 | 0.033 | 0.828 |
| 31 | S | 440 | 0.000 | 0.001 | 0.110 | 0.007 | 0.882 | 0.000 | 0.002 | 0.119 | 0.007 | 0.871 |
| 32 | S | 440 | 0.000 | 0.002 | 0.180 | 0.029 | 0.790 | 0.000 | 0.002 | 0.128 | 0.037 | 0.833 |
| 33 | S | 440 | 0.000 | 0.002 | 0.195 | 0.030 | 0.773 | 0.000 | 0.004 | 0.139 | 0.039 | 0.819 |
| 34 | F | 440 | 0.275 | 0.103 | 0.090 | 0.012 | 0.519 | 0.377 | 0.113 | 0.080 | 0.018 | 0.412 |

## Conclusion

Full-attention layers still provide the control case: BOS remains visible and usually captures a sizeable share of both attention mass and residual-effect score.

In this run, sliding layers do not recreate a strong moving-edge BOS substitute. The leftover routing is better described as a distributed interior/recent pattern than as a single backup sink.

Middle-layer secondary-sink candidates did appear in some prompt/layer pairs, which is consistent with the possibility that activation dynamics can create temporary non-BOS anchors.

The practical answer to the no-op question is therefore layer-dependent: full layers can still offload onto BOS, but sliding layers mostly have to distribute the branch over visible content/special tokens that survive the mask, with exact anchor scores revealing which of those positions materially change the residual update.
