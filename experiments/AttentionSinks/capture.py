"""
capture.py — dump per-layer attention activations from Gemma 4 E2B.

Supported sources:

1. `test_data/conversations.json`
      [{"user": "...", "assistant": "..."}]

2. Local mirror of Hugging Face TowerBlocks:
      experiments/AttentionSinks/data/TowerBlocks-v0.1/
      └── data/train-*.parquet

For each conversation we apply Gemma's chat template, run a single forward
pass with `output_attentions=True`, and save the attention tensors plus
the decoded tokens and source metadata.

Output file (one per captured conversation), in
`experiments/AttentionSinks/outputs/conv_XXXXX.pt`:

    {
        "conv_idx":       int,        # sequential index in this run
        "source_idx":     int,        # original row index in the source dataset
        "source_format":  str,        # "json" | "towerblocks"
        "messages":       list[dict], # chat-template messages
        "user_text":      str,        # concatenated user turns (compat)
        "asst_text":      str,        # concatenated assistant turns (compat)
        "source_meta":    dict,       # lang/task/split/... when available
        "input_ids":      LongTensor (1, S),
        "tokens":         list[str],
        "layer_types":    list[str],
        "sliding_window": int,
        "attentions":     list[Tensor],  # one per layer, each (H, S, S) in fp16
        "model_id":       str,
    }

Attentions are saved in fp16 (downcast from bf16) to keep the bundle small
without losing visible signal — sink-mass differences are well above 0.001.

Usage:
    python experiments/AttentionSinks/capture.py
    python experiments/AttentionSinks/capture.py --data experiments/AttentionSinks/data/TowerBlocks-v0.1 --limit 32 --require-bos-evicted
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pyarrow.dataset as ds
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO              = Path(__file__).resolve().parents[2]
JSON_DATA         = REPO / "test_data" / "conversations.json"
TOWERBLOCKS_DATA  = REPO / "experiments" / "AttentionSinks" / "data" / "TowerBlocks-v0.1"
OUT_DIR           = REPO / "experiments" / "AttentionSinks" / "outputs"
MODEL_ID          = "google/gemma-4-E2B-it"

ROLE_MAP = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
}


def _default_data_path():
    if JSON_DATA.exists():
        return JSON_DATA
    if TOWERBLOCKS_DATA.exists():
        return TOWERBLOCKS_DATA
    return JSON_DATA


def _display_path(path):
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def _infer_source_format(path):
    if path.is_file() and path.suffix.lower() == ".json":
        return "json"
    if path.is_dir() and any((path / "data").glob("*.parquet")):
        return "towerblocks"
    raise SystemExit(
        f"could not infer data format for {path}\n"
        f"supported inputs:\n"
        f"  - JSON list file like {JSON_DATA}\n"
        f"  - TowerBlocks dataset dir with data/*.parquet"
    )


def _messages_to_text(messages, role):
    return "\n\n".join(msg["content"] for msg in messages if msg["role"] == role)


def _normalize_messages(turns):
    messages = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        raw_role = turn.get("from", turn.get("role"))
        role = ROLE_MAP.get(raw_role)
        content = turn.get("value", turn.get("content"))
        if role is None or not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": role, "content": content})
    return messages


def _normalize_json_row(source_idx, row):
    if not isinstance(row, dict):
        return None
    if not ("user" in row and "assistant" in row):
        return None

    messages = [
        {"role": "user", "content": row["user"]},
        {"role": "assistant", "content": row["assistant"]},
    ]
    return {
        "source_idx": source_idx,
        "messages": messages,
        "user_text": str(row["user"]),
        "asst_text": str(row["assistant"]),
        "source_meta": {},
    }


def _normalize_towerblocks_row(source_idx, row):
    messages = _normalize_messages(row.get("conversations", []))
    if not messages:
        return None
    if not any(msg["role"] == "user" for msg in messages):
        return None
    if not any(msg["role"] == "assistant" for msg in messages):
        return None

    return {
        "source_idx": source_idx,
        "messages": messages,
        "user_text": _messages_to_text(messages, "user"),
        "asst_text": _messages_to_text(messages, "assistant"),
        "source_meta": {
            "lang": row.get("lang"),
            "split": row.get("split"),
            "dataset": row.get("dataset"),
            "task": row.get("task"),
            "num_turns": len(messages),
        },
    }


def _iter_json_records(path):
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            f"drop a JSON file there with shape:\n"
            f'  [{{"user": "...", "assistant": "..."}}, ...]'
        )

    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} must be a non-empty JSON list.")

    for idx, row in enumerate(rows):
        yield idx, row


def _iter_towerblocks_records(path, lang=None, task=None, source_split=None, dataset_name=None):
    data_dir = path / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet shards found under {data_dir}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    columns = ["conversations", "lang", "split", "dataset", "task"]

    source_idx = 0
    for batch in dataset.scanner(columns=columns, batch_size=128).to_batches():
        for row in batch.to_pylist():
            if lang is not None and row.get("lang") != lang:
                source_idx += 1
                continue
            if task is not None and row.get("task") != task:
                source_idx += 1
                continue
            if source_split is not None and row.get("split") != source_split:
                source_idx += 1
                continue
            if dataset_name is not None and row.get("dataset") != dataset_name:
                source_idx += 1
                continue
            yield source_idx, row
            source_idx += 1


def _iter_records(path, source_format, lang=None, task=None, source_split=None, dataset_name=None):
    if source_format == "json":
        yield from _iter_json_records(path)
        return
    if source_format == "towerblocks":
        yield from _iter_towerblocks_records(
            path,
            lang=lang,
            task=task,
            source_split=source_split,
            dataset_name=dataset_name,
        )
        return
    raise ValueError(f"unsupported source_format={source_format!r}")


def _normalize_row(source_format, source_idx, row):
    if source_format == "json":
        return _normalize_json_row(source_idx, row)
    if source_format == "towerblocks":
        return _normalize_towerblocks_row(source_idx, row)
    raise ValueError(f"unsupported source_format={source_format!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=Path,
        default=_default_data_path(),
        help="JSON list file or TowerBlocks dataset directory.",
    )
    p.add_argument(
        "--format",
        choices=["auto", "json", "towerblocks"],
        default="auto",
        help="Input format. Default: auto-detect from --data.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Capture at most this many conversations. TowerBlocks defaults to 32 if omitted.",
    )
    p.add_argument("--max-length", type=int, default=1024,
                   help="Truncate sequences to this many tokens (default 1024).")
    p.add_argument("--min-length", type=int, default=0,
                   help="Skip prompts shorter than this many tokens after templating/truncation.")
    p.add_argument("--require-bos-evicted", action="store_true",
                   help="Only keep prompts whose length exceeds the sliding window.")
    p.add_argument("--lang", default=None,
                   help="TowerBlocks only: exact match on row['lang'].")
    p.add_argument("--task", default=None,
                   help="TowerBlocks only: exact match on row['task'].")
    p.add_argument("--source-split", default=None,
                   help="TowerBlocks only: exact match on row['split'].")
    p.add_argument("--dataset-name", default=None,
                   help="TowerBlocks only: exact match on row['dataset'].")
    p.add_argument("--dry-run", action="store_true",
                   help="Load/tokenize/filter only; do not run the model or write bundles.")
    p.add_argument("--device", default="cpu",
                   help="cpu | mps | cuda  (default cpu — safest, slowest)")
    args = p.parse_args()

    data_path = args.data.resolve()
    if not data_path.exists():
        sys.exit(f"missing data source: {data_path}")

    source_format = _infer_source_format(data_path) if args.format == "auto" else args.format
    limit = args.limit
    if source_format == "towerblocks" and limit is None:
        limit = 32
        print("[capture] TowerBlocks source detected — defaulting to --limit 32 for safety")
    if args.require_bos_evicted and args.max_length and args.max_length <= 512:
        print("[capture] warning: --max-length <= 512 makes --require-bos-evicted impossible")

    print(f"[capture] source={source_format}  path={_display_path(data_path)}")
    if source_format == "towerblocks":
        print(
            f"[capture] filters: lang={args.lang!r} task={args.task!r} "
            f"split={args.source_split!r} dataset={args.dataset_name!r}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[capture] loading {MODEL_ID} processor/config ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    text_cfg = AutoConfig.from_pretrained(MODEL_ID).text_config
    layer_types  = list(text_cfg.layer_types)
    swa_window   = int(text_cfg.sliding_window)
    n_sliding    = layer_types.count("sliding_attention")
    n_full       = layer_types.count("full_attention")
    print(f"[capture]   {len(layer_types)} layers — {n_sliding} sliding "
          f"(window={swa_window}) + {n_full} full")
    print(f"[capture]   processor/config loaded in {time.time() - t0:.1f}s")

    model = None
    if not args.dry_run:
        print(f"[capture] loading model weights (eager attention, bf16) ...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, attn_implementation="eager", dtype=torch.bfloat16
        )
        model.eval()
        if args.device != "cpu":
            model = model.to(args.device)
        print(f"[capture]   model loaded in {time.time() - t0:.1f}s on device={args.device}")

    scanned = 0
    saved = 0
    skipped_invalid = 0
    skipped_short = 0
    skipped_window = 0

    for source_idx, row in _iter_records(
        data_path,
        source_format,
        lang=args.lang,
        task=args.task,
        source_split=args.source_split,
        dataset_name=args.dataset_name,
    ):
        scanned += 1
        conv = _normalize_row(source_format, source_idx, row)
        if conv is None:
            skipped_invalid += 1
            continue

        templated = processor.tokenizer.apply_chat_template(
            conv["messages"],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        ids = templated["input_ids"]
        if args.max_length and ids.shape[1] > args.max_length:
            ids = ids[:, : args.max_length]
        S = ids.shape[1]
        if S < args.min_length:
            skipped_short += 1
            continue
        if args.require_bos_evicted and S <= swa_window:
            skipped_window += 1
            continue

        label = f"[capture] [src={source_idx}]"
        if source_format == "towerblocks":
            meta = conv["source_meta"]
            label += (
                f" lang={meta.get('lang')} task={meta.get('task')}"
                f" split={meta.get('split')} dataset={meta.get('dataset')}"
            )

        if args.dry_run:
            print(f"{label} S={S} tokens  turns={len(conv['messages'])}  dry-run keep")
            saved += 1
            if limit is not None and saved >= limit:
                break
            continue

        ids_run = ids.to(args.device) if args.device != "cpu" else ids

        print(f"{label} S={S} tokens, running forward ...")
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids_run, output_attentions=True)
        dt = time.time() - t0

        attentions = [a[0].to(torch.float16).cpu() for a in out.attentions]
        ids_cpu    = ids.cpu()
        tokens     = [processor.tokenizer.decode([t]) for t in ids_cpu[0].tolist()]

        bundle = {
            "conv_idx":       saved,
            "source_idx":     source_idx,
            "source_format":  source_format,
            "messages":       conv["messages"],
            "user_text":      conv["user_text"],
            "asst_text":      conv["asst_text"],
            "source_meta":    conv["source_meta"],
            "input_ids":      ids_cpu,
            "tokens":         tokens,
            "layer_types":    layer_types,
            "sliding_window": swa_window,
            "attentions":     attentions,
            "model_id":       MODEL_ID,
        }
        path = OUT_DIR / f"conv_{saved:05d}.pt"
        torch.save(bundle, path)
        sz = path.stat().st_size / 1e6
        print(f"{label}   saved {path.name}  ({sz:.0f} MB, fwd={dt:.1f}s)")
        saved += 1

        if limit is not None and saved >= limit:
            break

    print(
        f"\n[capture] done. kept={saved} scanned={scanned} "
        f"skipped_invalid={skipped_invalid} skipped_short={skipped_short} "
        f"skipped_window={skipped_window}"
    )
    if not args.dry_run:
        print(f"[capture] outputs in {_display_path(OUT_DIR)}/")


if __name__ == "__main__":
    main()
