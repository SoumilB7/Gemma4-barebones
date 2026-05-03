# Attention Sinks — Experiment Runbook

**Research question:** In Gemma 4 E2B's sliding-window layers (28 of 35), BOS is evicted from
the context window for late queries. Where does the leftover attention mass go, and does it
materially affect the residual stream?

**What this disproves:** Cancedda 2024's "low-V no-op" hypothesis — that BOS is a cost-free
attention dump because its value vector has low norm. Full-layer measurements show
`exact_bos / attn_bos ≈ 1.9×`, meaning BOS delivers nearly twice the residual-stream impact
per unit attention mass compared to average positions.

---

## Notebooks

| Notebook | What it does |
|---|---|
| `experiment.ipynb` | Samples prompts → runs forward passes → scores contributions → saves all outputs |
| `analyze.ipynb` | Loads saved outputs and produces plots and tables |

Everything else:

| File | Role |
|---|---|
| `capture.py` | Optional: dump raw fp16 attention tensors to `.pt` bundles for custom analysis |

---

## Output folders

```
outputs/
├── smoke/    ← smoke run  (TARGET_PER_STRATUM = 1)
│   ├── report.md
│   ├── summary.json
│   ├── cases.parquet
│   └── prompt_index.jsonl
└── full/     ← canonical run  (TARGET_PER_STRATUM = 20)
    ├── report.md
    ├── summary.json
    ├── cases.parquet
    └── prompt_index.jsonl
```

---

## Prerequisites (one-time)

```bash
pip install torch transformers pyarrow pandas nbformat jupyter
```

Model weights (`google/gemma-4-E2B-it`) download automatically on first run.
TowerBlocks data must be at `data/TowerBlocks-v0.1/data/train-*.parquet`.

---

## Stage 1 — Smoke test  *(~5 min on MPS)*

Open `experiment.ipynb`. The **config cell** (second cell) has two lines to set:

```python
TARGET_PER_STRATUM = 1        # ← smoke
SAMPLING_MODE      = "first"  # ← smoke
```

Run all cells. Outputs go to `outputs/smoke/`.

Check in the printed output:
- `reconstruction error: X.XXXXX  ✓` (expect < 0.01)
- `sliding BOS visible rate  : 0.00000  ✓`
- `negative exact scores     : 0  ✓`

---

## Stage 2 — Full run  *(~2–4 h on MPS, 20 prompts × 5 strata)*

In the config cell, change to:

```python
TARGET_PER_STRATUM = 20       # ← full run
SAMPLING_MODE      = "random" # ← full run
```

Run all cells. Outputs go to `outputs/full/`.

---

## Stage 3 — Analyze

Open `analyze.ipynb` and run all cells.
It auto-detects `outputs/full/` first, falls back to `outputs/smoke/`.

---

## Validation checks

After any run, confirm in the notebook output or `report.md`:

| Check | Target |
|---|---|
| `reconstruction_relative_error` | < 0.01 |
| `sliding_bos_visible_rate` | ≈ 0.000 |
| `anchor_exact_negative_count` | = 0 |
| `exact_bos > attn_bos` (full layers) | ratio > 1× |
