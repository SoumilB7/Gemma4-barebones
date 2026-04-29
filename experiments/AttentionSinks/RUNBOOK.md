# Attention Sinks — Experiment Runbook

**Research question:** In Gemma 4 E2B's sliding-window layers (28 of 35), BOS is evicted from the context window for late queries. Where does the "leftover" attention mass go, and does it materially affect the residual stream?

**What this disproves:** Cancedda 2024's "low-V no-op" hypothesis — that BOS is a cost-free attention dump because its value vector has low norm. Full-layer measurements show `exact_bos / attn_bos ≈ 1.9`, meaning BOS delivers nearly twice the residual-stream impact per unit attention mass compared to average positions.

---

## Files

| File | Role |
|---|---|
| `run_sliding_sink_experiment.py` | Main experiment: sample → forward pass → score → report |
| `capture.py` | Optional: dump raw attention tensors to `.pt` bundles for offline inspection |
| `analyze.ipynb` | Notebook for visual exploration of capture bundles |

## Output folders

```
outputs/
├── smoke/     ← smoke-test runs (1 prompt per stratum)
├── full/      ← canonical run (20 prompts per stratum)  default
└── (*.pt)     ← raw attention bundles from capture.py
```

Files written per run:

| File | Contents |
|---|---|
| `report.md` | Human-readable summary with tables and conclusions |
| `summary.json` | Full aggregates + validation metadata |
| `cases.parquet` | Per-anchor-query, per-key-position top-k rows |
| `prompt_index.jsonl` | One row per sampled prompt |

---

## Prerequisites (one-time)

```bash
pip install torch transformers pyarrow nbformat
```

Model weights (`google/gemma-4-E2B-it`) download automatically on first run.  
TowerBlocks data must be at `data/TowerBlocks-v0.1/data/train-*.parquet`.

---

## Stage 1 — Smoke test  *(~5 min, 1 prompt per stratum)*

Validates the full pipeline. Check that `reconstruction_relative_error < 0.01` in the report.

```bash
cd experiments/AttentionSinks

python3 run_sliding_sink_experiment.py \
  --target-per-stratum 1 \
  --sampling-mode first \
  --device mps \
  --report-path       outputs/smoke/report.md \
  --summary-path      outputs/smoke/summary.json \
  --case-path         outputs/smoke/cases.parquet \
  --prompt-index-path outputs/smoke/prompt_index.jsonl
```

To smoke a single stratum only (faster):

```bash
python3 run_sliding_sink_experiment.py \
  --target-per-stratum 1 \
  --sampling-mode first \
  --strata "named_entity_recognition/en" \
  --device mps \
  --report-path       outputs/smoke/report.md \
  --summary-path      outputs/smoke/summary.json \
  --case-path         outputs/smoke/cases.parquet \
  --prompt-index-path outputs/smoke/prompt_index.jsonl
```

---

## Stage 2 — Full run  *(~2–4 h on MPS, 20 prompts × 5 strata)*

Writes to `outputs/full/` by default.

```bash
cd experiments/AttentionSinks

python3 run_sliding_sink_experiment.py \
  --target-per-stratum 20 \
  --sampling-mode random \
  --sample-seed 0 \
  --device mps
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--target-per-stratum` | 20 | Prompts per task/lang bucket |
| `--sampling-mode` | random | `first` stops early; `random` uses reservoir |
| `--sample-seed` | 0 | Change to get a different draw |
| `--max-length` | 640 | Truncation; must exceed 512 for BOS eviction |
| `--device` | auto | `mps` \| `cuda` \| `cpu` |
| `--strata` | all five | Comma-separated subset |

Available strata: `chat/en`, `named_entity_recognition/en`, `named_entity_recognition/es`, `machine_translation/en-de`, `machine_translation_evaluation/zh_en`

---

## Stage 3 — Notebook

Open `analyze.ipynb` directly in Jupyter or VS Code. It loads `.pt` bundles from `outputs/`.

```bash
cd experiments/AttentionSinks
jupyter notebook analyze.ipynb
```

---

## Stage 4 — Optional: raw activation capture

Saves full per-layer attention tensors to disk for custom analysis. Not required by the main script.

```bash
cd experiments/AttentionSinks

python3 capture.py \
  --data data/TowerBlocks-v0.1 \
  --limit 5 \
  --require-bos-evicted \
  --lang en \
  --task named_entity_recognition \
  --device mps
```

Output: `outputs/conv_00000.pt` … one bundle per prompt with full attention tensors in fp16.

---

## Validation checks

After any run, open `report.md` and confirm:

- `reconstruction_relative_error` < 0.01
- `sliding_bos_visible_for_late_query` ≈ 0.000
- `anchor_exact_negative_count` = 0
- Full-layer rows: `exact_bos > attn_bos` (the Cancedda signal)
