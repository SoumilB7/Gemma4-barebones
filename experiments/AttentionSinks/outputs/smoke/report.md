# Attention Sinks — Results

Generated: 2026-05-03T11:14:09+00:00 · model: `google/gemma-4-E2B-it` · device: `mps`
Prompts: 5 · strata: 5

## Validation

| check | value | pass |
|---|---|---|
| reconstruction_relative_error | 0.00228 | ✓ |
| sliding_bos_visible_rate | 0.00000 | ✓ |
| anchor_exact_negative_count | 0 | ✓ |
| exact_bos > attn_bos (full) | 8.9092 > 0.5317 | ✓ |

## Score-pre share by layer type

| group | sliding_pre | full_pre |
|---|---|---|
| bos | 0.0000 | 0.1034 |
| edge | 0.0018 | 0.0974 |
| self | 0.0494 | 0.0370 |
| recent | 0.3637 | 0.1554 |
| middle | 0.5851 | 0.6068 |

## Cancedda test (full layers — BOS)

- BOS mean exact score : **8.9092**
- BOS mean attn mass   : **0.5317**
- Ratio                : **16.76×**

BOS delivers **more** residual-stream impact per unit attention mass than average — **falsifies Cancedda 2024's low-V no-op claim.**