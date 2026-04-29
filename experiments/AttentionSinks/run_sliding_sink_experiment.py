#!/usr/bin/env python3
"""
Residual-contribution study for Gemma-4 sliding attention on TowerBlocks.

This runner extends the earlier sink-mass experiment by instrumenting the
Hugging Face Gemma eager-attention path at the activation level. The main
question is not only where attention probability goes, but which positions
actually contribute to the attention branch that is added back into the
residual stream.

Default outputs:
    - outputs/full/report.md
    - outputs/full/summary.json
    - outputs/full/cases.parquet
    - outputs/full/prompt_index.jsonl

The implementation makes one practical concession explicit in both code and
reporting:

    * All late queries (`q >= 512`) are analyzed with additive pre-norm
      contribution scores:
          score_pre(q, k) = ||c_pre(q, k)||_2
    * Exact post-RMSNorm leave-one-out scores are computed for anchor queries
      and selected candidate positions/groups. This keeps the study tractable
      for 100 long prompts without modifying the model code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import string
import time
import types
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.models.gemma4 import modeling_gemma4

from capture import (
    MODEL_ID,
    TOWERBLOCKS_DATA,
    _display_path,
    _infer_source_format,
    _iter_records,
    _normalize_row,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[2]
_OUT = REPO / "experiments" / "AttentionSinks" / "outputs" / "full"
DEFAULT_REPORT_PATH = _OUT / "report.md"
DEFAULT_SUMMARY_PATH = _OUT / "summary.json"
DEFAULT_CASE_PATH = _OUT / "cases.parquet"
DEFAULT_PROMPT_INDEX_PATH = _OUT / "prompt_index.jsonl"

RESEARCH_SOURCES = [
    {
        "title": "Efficient Streaming Language Models with Attention Sinks",
        "url": "https://arxiv.org/abs/2309.17453",
        "claim": "Initial sink tokens can stabilize streaming/windowed attention.",
    },
    {
        "title": "Why do LLMs attend to the first token?",
        "url": "https://arxiv.org/abs/2504.02732",
        "claim": "Attention sinks can help prevent over-mixing and preserve information flow.",
    },
    {
        "title": "Attention Sinks Are Functionally Essential in Softmax Transformers: Theoretical Evidence",
        "url": "https://openreview.net/forum?id=TvG8VtbDLB",
        "claim": "Softmax normalization can force stable sink-like behavior for default routing.",
    },
    {
        "title": "On the Existence and Behavior of Secondary Attention Sinks",
        "url": "https://openreview.net/forum?id=2DmhKvGLSC",
        "claim": "Middle-layer activation dynamics can create secondary sinks beyond BOS.",
    },
]

POSITION_OVERLAP_GROUPS = ("bos", "edge", "recent", "self", "middle")
TOKEN_GROUPS = ("special/chat-marker", "punctuation/newline", "content")
POSITION_PARTITION_GROUPS = ("bos", "edge", "self", "recent_nonself", "middle")
REPRESENTATIVE_LAYERS = (0, 15, 30, 34)
FULL_LAYER_SET = {4, 9, 14, 19, 24, 29, 34}


@dataclass(frozen=True)
class StratumSpec:
    name: str
    task: str
    lang: str


@dataclass
class SamplePrompt:
    sample_id: int
    stratum: str
    task: str
    lang: str
    source_idx: int
    source_format: str
    messages: list[dict[str, str]]
    source_meta: dict[str, Any]
    input_ids: list[int]
    seq_len: int

    def to_prompt_index_row(self) -> dict[str, Any]:
        meta = dict(self.source_meta)
        return {
            "sample_id": self.sample_id,
            "stratum": self.stratum,
            "task": self.task,
            "lang": self.lang,
            "source_idx": self.source_idx,
            "source_format": self.source_format,
            "seq_len": self.seq_len,
            "num_turns": meta.get("num_turns"),
            "split": meta.get("split"),
            "dataset": meta.get("dataset"),
        }


STRATA = [
    StratumSpec(name="chat/en", task="chat", lang="en"),
    StratumSpec(name="named_entity_recognition/en", task="named_entity_recognition", lang="en"),
    StratumSpec(name="named_entity_recognition/es", task="named_entity_recognition", lang="es"),
    StratumSpec(name="machine_translation/en-de", task="machine_translation", lang="en-de"),
    StratumSpec(name="machine_translation_evaluation/zh_en", task="machine_translation_evaluation", lang="zh_en"),
]
STRATUM_BY_KEY = {(spec.task, spec.lang): spec for spec in STRATA}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    return str(dtype)


def _init_bucket() -> dict[str, Any]:
    return {
        "query_count": 0,
        "anchor_count": 0,
        "pre_overlap_score_sum": defaultdict(float),
        "pre_position_partition_sum": defaultdict(float),
        "pre_position_partition_share_sum": defaultdict(float),
        "pre_token_score_sum": defaultdict(float),
        "pre_token_share_sum": defaultdict(float),
        "attn_overlap_mass_sum": defaultdict(float),
        "top_position_group_counts": Counter(),
        "top_token_group_counts": Counter(),
        "top_abs_position_counts": Counter(),
        "top_query_offset_counts": Counter(),
        "top_edge_offset_counts": Counter(),
        "anchor_exact_overlap_sum": defaultdict(float),
        "anchor_exact_position_partition_sum": defaultdict(float),
        "anchor_exact_position_partition_share_sum": defaultdict(float),
        "anchor_exact_token_sum": defaultdict(float),
        "anchor_exact_token_share_sum": defaultdict(float),
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _top_items(counter: Counter, total: int, top_n: int = 10) -> list[list[Any]]:
    rows = []
    for key, count in counter.most_common(top_n):
        frac = count / total if total else 0.0
        rows.append([key, count, _format_float(frac)])
    return rows


def _layer_scope_name(layer_type: str) -> str:
    return "full" if layer_type == "full_attention" else "sliding"


def _layer_depth_bucket(layer_idx: int, layer_type: str) -> str:
    if layer_type == "full_attention":
        return "full"
    if layer_idx <= 8:
        return "early_sliding"
    if 10 <= layer_idx <= 23:
        return "mid_sliding"
    if 25 <= layer_idx <= 33:
        return "late_sliding"
    return "other_sliding"


def _token_type_for_piece(tokenizer, token_id: int, piece: str, decoded: str) -> str:
    probe = f"{piece} {decoded}"
    special_markers = (
        "<start_of_turn>",
        "<end_of_turn>",
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<image",
        "<audio",
        "<video",
    )
    if token_id in tokenizer.all_special_ids:
        return "special/chat-marker"
    if any(marker in probe for marker in special_markers):
        return "special/chat-marker"

    text = decoded if decoded else piece.replace("▁", " ")
    if not text.strip():
        return "punctuation/newline"

    only_punct_or_space = True
    for char in text:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if not category.startswith("P"):
            only_punct_or_space = False
            break
    if only_punct_or_space:
        return "punctuation/newline"
    return "content"


def _primary_position_group(index: int, masks: dict[str, torch.Tensor]) -> str:
    if masks["bos"][index].item():
        return "bos"
    if masks["edge_partition"][index].item():
        return "edge"
    if masks["self"][index].item():
        return "self"
    if masks["recent_nonself"][index].item():
        return "recent_nonself"
    return "middle"


def _match_stratum(source_meta: dict[str, Any]) -> StratumSpec | None:
    key = (source_meta.get("task"), source_meta.get("lang"))
    return STRATUM_BY_KEY.get(key)


def _tokenize_messages(processor, messages: list[dict[str, str]], max_length: int) -> torch.Tensor:
    templated = processor.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    ids = templated["input_ids"]
    if max_length and ids.shape[1] > max_length:
        ids = ids[:, :max_length]
    return ids


def _reservoir_insert(bucket: list[SamplePrompt], item: SamplePrompt, seen_count: int, target_size: int, rng: random.Random) -> None:
    if len(bucket) < target_size:
        bucket.append(item)
        return
    replace_at = rng.randrange(seen_count)
    if replace_at < target_size:
        bucket[replace_at] = item


def _sample_prompts(
    processor,
    data_path: Path,
    source_format: str,
    max_length: int,
    min_length: int,
    target_per_stratum: int,
    sampling_mode: str,
    sample_seed: int,
    active_strata: list[StratumSpec],
    progress_every: int,
) -> tuple[list[SamplePrompt], dict[str, Any]]:
    rng = random.Random(sample_seed)
    reservoirs: dict[str, list[SamplePrompt]] = {spec.name: [] for spec in active_strata}
    eligible_counts = Counter()
    kept_counts = Counter()
    scanned_rows = 0
    matched_rows = 0
    invalid_rows = 0
    skipped_short = 0
    skipped_window = 0
    stopped_early = False

    for source_idx, row in _iter_records(data_path, source_format):
        scanned_rows += 1
        if progress_every and scanned_rows % progress_every == 0:
            print(
                f"[residual-study] sampling progress: scanned={scanned_rows} "
                f"matched={matched_rows} eligible={dict(eligible_counts)}"
            )
        source_meta = {
            "lang": row.get("lang"),
            "task": row.get("task"),
            "split": row.get("split"),
            "dataset": row.get("dataset"),
        }
        spec = _match_stratum(source_meta)
        if spec is None:
            continue
        if spec.name not in reservoirs:
            continue
        matched_rows += 1

        conv = _normalize_row(source_format, source_idx, row)
        if conv is None:
            invalid_rows += 1
            continue

        ids = _tokenize_messages(processor, conv["messages"], max_length)
        seq_len = ids.shape[1]
        if seq_len < min_length:
            skipped_short += 1
            continue
        if seq_len <= 512:
            skipped_window += 1
            continue

        eligible_counts[spec.name] += 1
        prompt = SamplePrompt(
            sample_id=-1,
            stratum=spec.name,
            task=spec.task,
            lang=spec.lang,
            source_idx=source_idx,
            source_format=source_format,
            messages=conv["messages"],
            source_meta=conv["source_meta"],
            input_ids=ids[0].tolist(),
            seq_len=seq_len,
        )

        if sampling_mode == "first":
            if len(reservoirs[spec.name]) < target_per_stratum:
                reservoirs[spec.name].append(prompt)
                kept_counts[spec.name] = len(reservoirs[spec.name])
            if all(len(reservoirs[s.name]) >= target_per_stratum for s in active_strata):
                stopped_early = True
                break
            continue

        _reservoir_insert(
            bucket=reservoirs[spec.name],
            item=prompt,
            seen_count=eligible_counts[spec.name],
            target_size=target_per_stratum,
            rng=rng,
        )
        kept_counts[spec.name] = min(target_per_stratum, len(reservoirs[spec.name]))

    selected: list[SamplePrompt] = []
    next_sample_id = 0
    for spec in active_strata:
        chosen = sorted(reservoirs[spec.name], key=lambda item: item.source_idx)
        for prompt in chosen:
            prompt.sample_id = next_sample_id
            next_sample_id += 1
            selected.append(prompt)

    stats = {
        "scanned_rows": scanned_rows,
        "matched_rows": matched_rows,
        "invalid_rows": invalid_rows,
        "skipped_short": skipped_short,
        "skipped_without_bos_eviction": skipped_window,
        "stopped_early": stopped_early,
        "sampling_mode": sampling_mode,
        "sample_seed": sample_seed,
        "target_per_stratum": target_per_stratum,
        "eligible_by_stratum": {spec.name: int(eligible_counts[spec.name]) for spec in active_strata},
        "selected_by_stratum": {spec.name: int(len(reservoirs[spec.name])) for spec in active_strata},
        "shortfall_by_stratum": {
            spec.name: max(0, target_per_stratum - len(reservoirs[spec.name])) for spec in active_strata
        },
    }
    return selected, stats


class ResidualContributionCollector:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        layer_types: list[str],
        sliding_window: int,
        max_case_positions: int,
        secondary_sink_threshold: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_types = layer_types
        self.sliding_window = sliding_window
        self.max_case_positions = max_case_positions
        self.secondary_sink_threshold = secondary_sink_threshold

        self.layer_type_stats = {name: _init_bucket() for name in ("sliding", "full")}
        self.depth_bucket_stats = {
            name: _init_bucket()
            for name in ("early_sliding", "mid_sliding", "late_sliding", "other_sliding", "full")
        }
        self.per_layer_stats = {layer_idx: _init_bucket() for layer_idx in range(len(layer_types))}

        self.case_rows: list[dict[str, Any]] = []
        self.secondary_sink_rows: list[dict[str, Any]] = []
        self.validation: dict[str, Any] = {
            "capture_shapes": {},
            "layer_types_match_config": True,
            "sliding_bos_visible_for_late_query": [],
            "reconstruction_error_l2": None,
            "reconstruction_relative_error": None,
            "anchor_exact_negative_count": 0,
        }

        self.current_prompt: SamplePrompt | None = None
        self.current_ids: torch.Tensor | None = None
        self.current_pieces: list[str] = []
        self.current_decoded: list[str] = []
        self.current_token_types: list[str] = []
        self.current_anchor_queries: set[int] = set()
        self.current_prompt_top_positions: dict[int, Counter] = defaultdict(Counter)
        self.current_prompt_middle_top_positions: dict[int, Counter] = defaultdict(Counter)
        self.current_prompt_late_counts: Counter = Counter()
        self._did_reconstruction_check = False

    def start_prompt(self, prompt: SamplePrompt) -> torch.Tensor:
        input_ids = torch.tensor([prompt.input_ids], dtype=torch.long)
        self.current_prompt = prompt
        self.current_ids = input_ids
        ids_list = input_ids[0].tolist()
        self.current_pieces = self.tokenizer.convert_ids_to_tokens(ids_list)
        self.current_decoded = [
            self.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in ids_list
        ]
        self.current_token_types = [
            _token_type_for_piece(self.tokenizer, token_id, piece, decoded)
            for token_id, piece, decoded in zip(ids_list, self.current_pieces, self.current_decoded)
        ]

        seq_len = len(ids_list)
        anchors = {512, (512 + seq_len - 1) // 2, seq_len - 1}
        self.current_anchor_queries = {query for query in anchors if 0 <= query < seq_len}
        self.current_prompt_top_positions.clear()
        self.current_prompt_middle_top_positions.clear()
        self.current_prompt_late_counts.clear()
        return input_ids

    def finish_prompt(self) -> None:
        if self.current_prompt is None:
            return

        for layer_idx, counter in self.current_prompt_middle_top_positions.items():
            total = self.current_prompt_late_counts[layer_idx]
            if not total or not counter:
                continue
            position, count = counter.most_common(1)[0]
            frac = count / total
            if frac < self.secondary_sink_threshold:
                continue
            self.secondary_sink_rows.append(
                {
                    "sample_id": self.current_prompt.sample_id,
                    "source_idx": self.current_prompt.source_idx,
                    "stratum": self.current_prompt.stratum,
                    "layer": layer_idx,
                    "layer_type": self.layer_types[layer_idx],
                    "depth_bucket": _layer_depth_bucket(layer_idx, self.layer_types[layer_idx]),
                    "position": position,
                    "token_piece": self.current_pieces[position],
                    "decoded_token": self.current_decoded[position].replace("\n", "\\n"),
                    "token_type": self.current_token_types[position],
                    "top_fraction": frac,
                    "late_queries": total,
                }
            )

        self.current_prompt = None
        self.current_ids = None
        self.current_pieces = []
        self.current_decoded = []
        self.current_token_types = []
        self.current_anchor_queries = set()
        self.current_prompt_top_positions.clear()
        self.current_prompt_middle_top_positions.clear()
        self.current_prompt_late_counts.clear()

    def capture_layer(
        self,
        *,
        module,
        layer_idx: int,
        layer_type: str,
        attention_mask: torch.Tensor | None,
        attn_weights: torch.Tensor,
        value_states: torch.Tensor,
        per_head_output: torch.Tensor,
        attn_output_pre_norm: torch.Tensor,
        parent_layer,
    ) -> None:
        if self.current_prompt is None:
            return

        seq_len = self.current_prompt.seq_len
        late_queries = list(range(512, seq_len))
        if not late_queries:
            return

        layer_scope = _layer_scope_name(layer_type)
        depth_bucket = _layer_depth_bucket(layer_idx, layer_type)
        scope_buckets = (
            self.layer_type_stats[layer_scope],
            self.depth_bucket_stats[depth_bucket],
            self.per_layer_stats[layer_idx],
        )

        attn = attn_weights[0].to(dtype=torch.float32)
        value_rep = modeling_gemma4.repeat_kv(value_states, module.num_key_value_groups)[0].to(dtype=torch.float32)
        pre_output = attn_output_pre_norm[0].to(dtype=torch.float32)

        if not self.validation["capture_shapes"]:
            self.validation["capture_shapes"] = {
                "attn_weights": list(attn_weights.shape),
                "value_states": list(value_states.shape),
                "per_head_output": list(per_head_output.shape),
                "attn_output_pre_norm": list(attn_output_pre_norm.shape),
            }

        hidden_size = module.o_proj.weight.shape[0]
        num_attention_heads = int(module.config.num_attention_heads)
        head_weights = module.o_proj.weight.to(dtype=torch.float32).view(
            hidden_size, num_attention_heads, module.head_dim
        )
        head_weights = head_weights.permute(1, 0, 2).contiguous()
        projected_values = torch.einsum("hkd,hod->hko", value_rep, head_weights)
        gram = torch.einsum("hko,jko->khj", projected_values, projected_values)
        attn_qkh = attn.permute(1, 2, 0).contiguous()
        late_qkh = attn_qkh[late_queries]
        score_sq = torch.einsum("qkh,khj,qkj->qk", late_qkh, gram, late_qkh)
        score_pre = score_sq.clamp_min(0.0).sqrt()

        if not self._did_reconstruction_check:
            q = late_queries[0]
            contrib_vectors = torch.einsum("hk,hko->ko", attn[:, q, :], projected_values)
            reconstructed = contrib_vectors.sum(dim=0)
            reference = pre_output[q]
            diff = reconstructed - reference
            self.validation["reconstruction_error_l2"] = float(torch.linalg.norm(diff).item())
            denom = float(torch.linalg.norm(reference).item()) or 1.0
            self.validation["reconstruction_relative_error"] = self.validation["reconstruction_error_l2"] / denom
            self._did_reconstruction_check = True

        for offset, query_idx in enumerate(late_queries):
            score_row = score_pre[offset]
            attn_row = attn[:, query_idx, :]
            visible_mask = attn_row.sum(dim=0) > 0
            visible_indices = torch.nonzero(visible_mask, as_tuple=False).flatten().tolist()
            if not visible_indices:
                continue

            if layer_type == "sliding_attention":
                self.validation["sliding_bos_visible_for_late_query"].append(bool(visible_mask[0].item()))

            masks = self._build_position_masks(visible_indices, query_idx, seq_len, score_row.device)
            overlap_scores = {
                group: float(score_row[masks[group]].sum().item()) for group in POSITION_OVERLAP_GROUPS
            }
            partition_scores = {
                "bos": float(score_row[masks["bos"]].sum().item()),
                "edge": float(score_row[masks["edge_partition"]].sum().item()),
                "self": float(score_row[masks["self"]].sum().item()),
                "recent_nonself": float(score_row[masks["recent_nonself"]].sum().item()),
                "middle": float(score_row[masks["middle"]].sum().item()),
            }
            token_scores = {
                group: float(score_row[masks[group]].sum().item()) for group in TOKEN_GROUPS
            }
            attn_masses = {
                group: float(attn_row[:, masks[group]].sum(dim=-1).mean().item()) for group in POSITION_OVERLAP_GROUPS
            }

            top_visible_scores = score_row[visible_mask]
            top_visible_offset = int(torch.argmax(top_visible_scores).item())
            top_position = visible_indices[top_visible_offset]
            top_position_group = _primary_position_group(top_position, masks)
            top_token_group = self.current_token_types[top_position]
            self.current_prompt_top_positions[layer_idx][top_position] += 1
            self.current_prompt_late_counts[layer_idx] += 1
            if top_position_group == "middle":
                self.current_prompt_middle_top_positions[layer_idx][top_position] += 1

            query_offset = query_idx - top_position
            edge_offset = top_position - visible_indices[0]
            self._update_pre_buckets(
                scope_buckets=scope_buckets,
                overlap_scores=overlap_scores,
                partition_scores=partition_scores,
                token_scores=token_scores,
                attn_masses=attn_masses,
                top_position_group=top_position_group,
                top_token_group=top_token_group,
                top_position=top_position,
                query_offset=query_offset,
                edge_offset=edge_offset,
            )

            if query_idx in self.current_anchor_queries:
                self._record_anchor_case_rows(
                    scope_buckets=scope_buckets,
                    layer_idx=layer_idx,
                    layer_type=layer_type,
                    query_idx=query_idx,
                    score_row=score_row,
                    attn_row=attn_row,
                    projected_values=projected_values,
                    pre_output=pre_output[query_idx],
                    post_attention_layernorm=parent_layer.post_attention_layernorm,
                    masks=masks,
                    visible_indices=visible_indices,
                )

    def _build_position_masks(
        self,
        visible_indices: list[int],
        query_idx: int,
        seq_len: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        visible = torch.zeros(seq_len, dtype=torch.bool, device=device)
        visible[visible_indices] = True
        edge_indices = visible_indices[: min(4, len(visible_indices))]
        recent_indices = visible_indices[max(0, len(visible_indices) - 32) :]
        bos_indices = [0] if seq_len > 0 and bool(visible[0].item()) else []

        bos = torch.zeros(seq_len, dtype=torch.bool, device=device)
        bos[bos_indices] = True
        edge = torch.zeros(seq_len, dtype=torch.bool, device=device)
        edge[edge_indices] = True
        recent = torch.zeros(seq_len, dtype=torch.bool, device=device)
        recent[recent_indices] = True
        self_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        self_mask[query_idx] = True

        edge_partition = edge & ~bos
        recent_nonself = recent & ~self_mask
        middle = visible & ~bos & ~edge_partition & ~recent & ~self_mask

        special_mask = torch.tensor(
            [group == "special/chat-marker" for group in self.current_token_types], dtype=torch.bool, device=device
        )
        punct_mask = torch.tensor(
            [group == "punctuation/newline" for group in self.current_token_types], dtype=torch.bool, device=device
        )
        content_mask = ~(special_mask | punct_mask)

        return {
            "visible": visible,
            "bos": bos,
            "edge": edge,
            "recent": recent,
            "self": self_mask,
            "edge_partition": edge_partition,
            "recent_nonself": recent_nonself,
            "middle": middle,
            "special/chat-marker": visible & special_mask,
            "punctuation/newline": visible & punct_mask,
            "content": visible & content_mask,
        }

    def _update_pre_buckets(
        self,
        *,
        scope_buckets: tuple[dict[str, Any], ...],
        overlap_scores: dict[str, float],
        partition_scores: dict[str, float],
        token_scores: dict[str, float],
        attn_masses: dict[str, float],
        top_position_group: str,
        top_token_group: str,
        top_position: int,
        query_offset: int,
        edge_offset: int,
    ) -> None:
        position_total = sum(partition_scores.values()) or 1.0
        token_total = sum(token_scores.values()) or 1.0
        for bucket in scope_buckets:
            bucket["query_count"] += 1
            for group, value in overlap_scores.items():
                bucket["pre_overlap_score_sum"][group] += value
            for group, value in partition_scores.items():
                bucket["pre_position_partition_sum"][group] += value
                bucket["pre_position_partition_share_sum"][group] += value / position_total
            for group, value in token_scores.items():
                bucket["pre_token_score_sum"][group] += value
                bucket["pre_token_share_sum"][group] += value / token_total
            for group, value in attn_masses.items():
                bucket["attn_overlap_mass_sum"][group] += value
            bucket["top_position_group_counts"][top_position_group] += 1
            bucket["top_token_group_counts"][top_token_group] += 1
            bucket["top_abs_position_counts"][top_position] += 1
            bucket["top_query_offset_counts"][query_offset] += 1
            bucket["top_edge_offset_counts"][edge_offset] += 1

    def _record_anchor_case_rows(
        self,
        *,
        scope_buckets: tuple[dict[str, Any], ...],
        layer_idx: int,
        layer_type: str,
        query_idx: int,
        score_row: torch.Tensor,
        attn_row: torch.Tensor,
        projected_values: torch.Tensor,
        pre_output: torch.Tensor,
        post_attention_layernorm,
        masks: dict[str, torch.Tensor],
        visible_indices: list[int],
    ) -> None:
        partition_vectors = {}
        token_vectors = {}

        for group in POSITION_PARTITION_GROUPS:
            mask_name = group
            if group == "edge":
                mask_name = "edge_partition"
            mask = masks[mask_name]
            positions = torch.nonzero(mask, as_tuple=False).flatten()
            if positions.numel() == 0:
                partition_vectors[group] = pre_output.new_zeros(pre_output.shape)
                continue
            partition_vectors[group] = torch.einsum(
                "hk,hko->o", attn_row[:, positions], projected_values[:, positions, :]
            )

        for group in TOKEN_GROUPS:
            positions = torch.nonzero(masks[group], as_tuple=False).flatten()
            if positions.numel() == 0:
                token_vectors[group] = pre_output.new_zeros(pre_output.shape)
                continue
            token_vectors[group] = torch.einsum(
                "hk,hko->o", attn_row[:, positions], projected_values[:, positions, :]
            )

        base_norm = self._apply_post_attention_layernorm(post_attention_layernorm, pre_output)
        exact_partition_scores = {
            group: self._exact_score(base_norm, pre_output, vector, post_attention_layernorm)
            for group, vector in partition_vectors.items()
        }
        exact_token_scores = {
            group: self._exact_score(base_norm, pre_output, vector, post_attention_layernorm)
            for group, vector in token_vectors.items()
        }
        exact_overlap_scores = {
            "bos": exact_partition_scores["bos"],
            "edge": exact_partition_scores["edge"],
            "self": exact_partition_scores["self"],
            "recent": self._exact_score(
                base_norm,
                pre_output,
                partition_vectors["recent_nonself"] + partition_vectors["self"],
                post_attention_layernorm,
            ),
            "middle": exact_partition_scores["middle"],
        }

        if any(score < -1e-8 for score in exact_partition_scores.values()) or any(
            score < -1e-8 for score in exact_token_scores.values()
        ):
            self.validation["anchor_exact_negative_count"] += 1

        position_total = sum(exact_partition_scores.values()) or 1.0
        token_total = sum(exact_token_scores.values()) or 1.0
        for bucket in scope_buckets:
            bucket["anchor_count"] += 1
            for group, value in exact_overlap_scores.items():
                bucket["anchor_exact_overlap_sum"][group] += value
            for group, value in exact_partition_scores.items():
                bucket["anchor_exact_position_partition_sum"][group] += value
                bucket["anchor_exact_position_partition_share_sum"][group] += value / position_total
            for group, value in exact_token_scores.items():
                bucket["anchor_exact_token_sum"][group] += value
                bucket["anchor_exact_token_share_sum"][group] += value / token_total

        visible_scores = score_row[visible_indices]
        top_k = min(self.max_case_positions, len(visible_indices))
        top_positions = []
        if top_k:
            values, indices = torch.topk(visible_scores, k=top_k)
            top_positions.extend(visible_indices[int(idx)] for idx in indices.tolist())
        candidate_positions = set(top_positions)
        candidate_positions.update(visible_indices[: min(4, len(visible_indices))])
        candidate_positions.add(query_idx)
        if layer_type == "full_attention" and 0 in visible_indices:
            candidate_positions.add(0)

        for key_idx in sorted(candidate_positions):
            contrib_vec = torch.einsum("h,ho->o", attn_row[:, key_idx], projected_values[:, key_idx, :])
            exact_score = self._exact_score(base_norm, pre_output, contrib_vec, post_attention_layernorm)
            position_group = _primary_position_group(key_idx, masks)
            token_group = self.current_token_types[key_idx]
            self.case_rows.append(
                {
                    "sample_id": self.current_prompt.sample_id,
                    "source_idx": self.current_prompt.source_idx,
                    "stratum": self.current_prompt.stratum,
                    "layer": layer_idx,
                    "layer_type": layer_type,
                    "depth_bucket": _layer_depth_bucket(layer_idx, layer_type),
                    "query_pos": query_idx,
                    "query_anchor": self._anchor_label(query_idx, self.current_prompt.seq_len),
                    "key_pos": key_idx,
                    "visible_start": visible_indices[0],
                    "visible_end": visible_indices[-1],
                    "edge_offset": key_idx - visible_indices[0],
                    "query_offset": query_idx - key_idx,
                    "token_id": self.current_prompt.input_ids[key_idx],
                    "token_piece": self.current_pieces[key_idx],
                    "decoded_token": self.current_decoded[key_idx].replace("\n", "\\n"),
                    "position_group": position_group,
                    "token_type": token_group,
                    "attention_mass_mean": float(attn_row[:, key_idx].mean().item()),
                    "score_pre": float(score_row[key_idx].item()),
                    "score_resid_exact": exact_score,
                    "is_topk_pre": key_idx in top_positions,
                    "is_self": key_idx == query_idx,
                    "is_bos": key_idx == 0,
                    "dataset": self.current_prompt.source_meta.get("dataset"),
                    "split": self.current_prompt.source_meta.get("split"),
                }
            )

    @staticmethod
    def _apply_post_attention_layernorm(layernorm, vector: torch.Tensor) -> torch.Tensor:
        return layernorm(vector.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).to(dtype=torch.float32)

    def _exact_score(
        self,
        base_norm: torch.Tensor,
        base_pre_output: torch.Tensor,
        contribution_vector: torch.Tensor,
        post_attention_layernorm,
    ) -> float:
        variant = base_pre_output - contribution_vector
        variant_norm = self._apply_post_attention_layernorm(post_attention_layernorm, variant)
        return float(torch.linalg.norm(base_norm - variant_norm).item())

    @staticmethod
    def _anchor_label(query_idx: int, seq_len: int) -> str:
        if query_idx == 512:
            return "q512"
        if query_idx == seq_len - 1:
            return "last"
        return "mid"


def _install_attention_wrappers(model, collector: ResidualContributionCollector, layer_types: list[str]) -> list[tuple[Any, Any]]:
    patched: list[tuple[Any, Any]] = []
    layers = model.model.language_model.layers
    for layer_idx, layer in enumerate(layers):
        attn_module = layer.self_attn
        original_forward = attn_module.forward
        layer_type = layer_types[layer_idx]

        def _make_wrapped_forward(bound_layer_idx: int, bound_layer_type: str, bound_parent_layer):
            def wrapped_forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: torch.Tensor,
                attention_mask: torch.Tensor | None,
                shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
                past_key_values=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                cos, sin = position_embeddings

                query_states = self.q_proj(hidden_states).view(hidden_shape)
                query_states = self.q_norm(query_states)
                query_states = modeling_gemma4.apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
                query_states = query_states.transpose(1, 2)

                if self.is_kv_shared_layer:
                    key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
                    key_states = key_states.to(query_states.device)
                    value_states = value_states.to(query_states.device)
                else:
                    key_states = self.k_proj(hidden_states).view(hidden_shape)
                    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

                    key_states = self.k_norm(key_states)
                    key_states = modeling_gemma4.apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                    key_states = key_states.transpose(1, 2)

                    value_states = self.v_norm(value_states)
                    value_states = value_states.transpose(1, 2)

                if past_key_values is not None and not self.is_kv_shared_layer:
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                if self.store_full_length_kv:
                    shared_kv_states[self.layer_idx] = key_states, value_states

                attn_output_heads, attn_weights = modeling_gemma4.eager_attention_forward(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=self.attention_dropout if self.training else 0.0,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )

                attn_output_pre_o_proj = attn_output_heads.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output_pre_o_proj)

                collector.capture_layer(
                    module=self,
                    layer_idx=bound_layer_idx,
                    layer_type=bound_layer_type,
                    attention_mask=attention_mask,
                    attn_weights=attn_weights,
                    value_states=value_states,
                    per_head_output=attn_output_heads,
                    attn_output_pre_norm=attn_output,
                    parent_layer=bound_parent_layer,
                )
                return attn_output, attn_weights

            return wrapped_forward

        attn_module.forward = types.MethodType(
            _make_wrapped_forward(layer_idx, layer_type, layer),
            attn_module,
        )
        patched.append((attn_module, original_forward))
    return patched


def _restore_attention_wrappers(patched: list[tuple[Any, Any]]) -> None:
    for module, original_forward in patched:
        module.forward = original_forward


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    query_count = bucket["query_count"] or 1
    anchor_count = bucket["anchor_count"] or 1
    result = {
        "query_count": bucket["query_count"],
        "anchor_count": bucket["anchor_count"],
        "pre_overlap_score_mean": {
            group: bucket["pre_overlap_score_sum"][group] / query_count for group in POSITION_OVERLAP_GROUPS
        },
        "pre_position_partition_mean": {
            group: bucket["pre_position_partition_sum"][group] / query_count for group in POSITION_PARTITION_GROUPS
        },
        "pre_position_partition_share_mean": {
            group: bucket["pre_position_partition_share_sum"][group] / query_count for group in POSITION_PARTITION_GROUPS
        },
        "pre_token_score_mean": {
            group: bucket["pre_token_score_sum"][group] / query_count for group in TOKEN_GROUPS
        },
        "pre_token_share_mean": {
            group: bucket["pre_token_share_sum"][group] / query_count for group in TOKEN_GROUPS
        },
        "attention_overlap_mass_mean": {
            group: bucket["attn_overlap_mass_sum"][group] / query_count for group in POSITION_OVERLAP_GROUPS
        },
        "top_position_group_frac": {
            group: bucket["top_position_group_counts"][group] / query_count
            for group in ("bos", "edge", "self", "recent_nonself", "middle")
        },
        "top_token_group_frac": {
            group: bucket["top_token_group_counts"][group] / query_count for group in TOKEN_GROUPS
        },
        "top_abs_position_hist": dict(bucket["top_abs_position_counts"]),
        "top_query_offset_hist": dict(bucket["top_query_offset_counts"]),
        "top_edge_offset_hist": dict(bucket["top_edge_offset_counts"]),
        "anchor_exact_overlap_mean": {
            group: bucket["anchor_exact_overlap_sum"][group] / anchor_count for group in POSITION_OVERLAP_GROUPS
        },
        "anchor_exact_position_partition_mean": {
            group: bucket["anchor_exact_position_partition_sum"][group] / anchor_count
            for group in POSITION_PARTITION_GROUPS
        },
        "anchor_exact_position_partition_share_mean": {
            group: bucket["anchor_exact_position_partition_share_sum"][group] / anchor_count
            for group in POSITION_PARTITION_GROUPS
        },
        "anchor_exact_token_mean": {
            group: bucket["anchor_exact_token_sum"][group] / anchor_count for group in TOKEN_GROUPS
        },
        "anchor_exact_token_share_mean": {
            group: bucket["anchor_exact_token_share_sum"][group] / anchor_count for group in TOKEN_GROUPS
        },
    }
    return result


def _write_prompt_index(path: Path, prompts: list[SamplePrompt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.to_prompt_index_row(), ensure_ascii=True) + "\n")


def _write_case_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _prompt_rows_by_stratum(prompts: list[SamplePrompt]) -> dict[str, list[SamplePrompt]]:
    out: dict[str, list[SamplePrompt]] = defaultdict(list)
    for prompt in prompts:
        out[prompt.stratum].append(prompt)
    return out


def _representative_case_rows(prompts: list[SamplePrompt], case_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    first_prompt_by_stratum = {}
    for prompt in prompts:
        first_prompt_by_stratum.setdefault(prompt.stratum, prompt)

    grouped = defaultdict(list)
    for row in case_rows:
        prompt = first_prompt_by_stratum.get(row["stratum"])
        if prompt is None:
            continue
        if row["sample_id"] != prompt.sample_id:
            continue
        if row["query_anchor"] != "last":
            continue
        if row["layer"] not in REPRESENTATIVE_LAYERS:
            continue
        grouped[row["stratum"]].append(row)

    filtered = {}
    for stratum, rows in grouped.items():
        rows = sorted(rows, key=lambda item: (item["layer"], -item["score_resid_exact"], item["key_pos"]))
        compact = []
        counts = Counter()
        for row in rows:
            if counts[row["layer"]] >= 6:
                continue
            compact.append(row)
            counts[row["layer"]] += 1
        filtered[stratum] = compact
    return filtered


def _interpret_case_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No representative anchor rows were available for this stratum."

    by_layer = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)

    notes = []
    for layer in sorted(by_layer):
        top = max(by_layer[layer], key=lambda item: item["score_resid_exact"])
        token = top["decoded_token"]
        if token.strip() == "":
            token = top["token_piece"]
        notes.append(
            f"layer {layer} ({'full' if top['layer_type'] == 'full_attention' else 'sliding'}) "
            f"leans hardest on pos {top['key_pos']} [{top['position_group']}] token `{token}`"
        )
    return "; ".join(notes) + "."


def _render_report(
    *,
    generated_at: str,
    prompts: list[SamplePrompt],
    sampling_stats: dict[str, Any],
    summary: dict[str, Any],
    case_rows: list[dict[str, Any]],
    report_path: Path,
    active_strata: list[StratumSpec],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    representative = _representative_case_rows(prompts, case_rows)
    prompt_by_stratum = _prompt_rows_by_stratum(prompts)
    lines: list[str] = []

    layer_type = summary["aggregates"]["layer_type"]
    depth_bucket = summary["aggregates"]["depth_bucket"]
    per_layer = summary["aggregates"]["per_layer"]

    lines.append("# Sliding Residual Contribution Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Research Context")
    lines.append("")
    lines.append(
        "This study treats attention-sink papers as hypothesis context, not as a direct answer for Gemma-4 on "
        "TowerBlocks. The motivating references were:"
    )
    lines.append("")
    for source in RESEARCH_SOURCES:
        lines.append(f"- [{source['title']}]({source['url']}): {source['claim']}")
    lines.append("")
    lines.append(
        "None of those papers directly settle Gemma-4 sliding-layer behavior on TowerBlocks; the results below are "
        "measured from this repo's own instrumented runs."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Model path: `google/gemma-4-E2B-it` via Hugging Face eager attention.")
    lines.append("- Data: TowerBlocks local mirror, stratified over five fixed task/lang buckets.")
    lines.append("- Eligibility: chat template applied, truncated to `max_length=640`, then keep prompts with `seq_len >= 513`.")
    lines.append("- Global distribution metric: additive pre-norm contribution score `score_pre(q, k) = ||c_pre(q, k)||_2` over all late queries.")
    lines.append("- Exact residual-effect metric: leave-one-out post-attention RMSNorm score on anchor queries only (`q=512`, midpoint late query, final query).")
    lines.append("- Why the split: post-attention RMSNorm is nonlinear, so exact per-position leave-one-out for every late query would be prohibitively expensive at the 100-prompt target.")
    lines.append("- Interpretation rule: high attention mass with low residual-effect score behaves like a probability reservoir; high exact residual-effect score means the token materially changes the residual update.")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    sample_rows = []
    for spec in active_strata:
        sample_rows.append(
            [
                spec.name,
                sampling_stats["eligible_by_stratum"][spec.name],
                sampling_stats["selected_by_stratum"][spec.name],
                sampling_stats["shortfall_by_stratum"][spec.name],
            ]
        )
    lines.append(_markdown_table(["stratum", "eligible", "selected", "shortfall"], sample_rows))
    lines.append("")
    lines.append(
        f"Scanned rows: `{sampling_stats['scanned_rows']}`; matched rows: `{sampling_stats['matched_rows']}`; "
        f"invalid: `{sampling_stats['invalid_rows']}`; short after templating: `{sampling_stats['skipped_short']}`; "
        f"without BOS eviction: `{sampling_stats['skipped_without_bos_eviction']}`."
    )
    if sampling_stats["stopped_early"]:
        lines.append("")
        lines.append("Sampling stopped early because `sampling_mode=first` filled every bucket before the end of the dataset.")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    validation = summary["validation"]
    bos_visible = validation["sliding_bos_visible_for_late_query"]
    bos_visible_rate = sum(1 for flag in bos_visible if flag) / len(bos_visible) if bos_visible else 0.0
    lines.append(
        f"- Reconstruction error `||sum_k c_pre(q,k) - C(q)||_2`: `{_format_float(validation['reconstruction_error_l2'], 6)}` "
        f"(relative `{_format_float(validation['reconstruction_relative_error'], 6)}`)."
    )
    lines.append(f"- Sliding-layer BOS visible rate for late queries: `{_format_float(bos_visible_rate, 6)}`.")
    lines.append(f"- Negative exact-score count: `{validation['anchor_exact_negative_count']}`.")
    lines.append(
        f"- Capture shapes: `attn={validation['capture_shapes'].get('attn_weights')}`, "
        f"`value={validation['capture_shapes'].get('value_states')}`, "
        f"`per_head={validation['capture_shapes'].get('per_head_output')}`, "
        f"`pre_norm={validation['capture_shapes'].get('attn_output_pre_norm')}`."
    )
    lines.append("")
    lines.append("## Global Findings")
    lines.append("")
    global_rows = []
    for scope in ("sliding", "full"):
        bucket = layer_type[scope]
        global_rows.append(
            [
                scope,
                bucket["query_count"],
                _format_float(bucket["pre_position_partition_share_mean"]["bos"]),
                _format_float(bucket["pre_position_partition_share_mean"]["edge"]),
                _format_float(bucket["pre_position_partition_share_mean"]["recent_nonself"]),
                _format_float(bucket["pre_position_partition_share_mean"]["self"]),
                _format_float(bucket["pre_position_partition_share_mean"]["middle"]),
                _format_float(bucket["top_position_group_frac"]["bos"]),
                _format_float(bucket["top_position_group_frac"]["edge"]),
                _format_float(bucket["top_position_group_frac"]["recent_nonself"]),
                _format_float(bucket["top_position_group_frac"]["self"]),
                _format_float(bucket["top_position_group_frac"]["middle"]),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "type",
                "rows",
                "bos_share_pre",
                "edge_share_pre",
                "recent_share_pre",
                "self_share_pre",
                "middle_share_pre",
                "top_bos_frac",
                "top_edge_frac",
                "top_recent_frac",
                "top_self_frac",
                "top_middle_frac",
            ],
            global_rows,
        )
    )
    lines.append("")
    control_rows = []
    for scope in ("sliding", "full"):
        bucket = layer_type[scope]
        control_rows.append(
            [
                scope,
                _format_float(bucket["attention_overlap_mass_mean"]["bos"]),
                _format_float(bucket["attention_overlap_mass_mean"]["edge"]),
                _format_float(bucket["attention_overlap_mass_mean"]["recent"]),
                _format_float(bucket["attention_overlap_mass_mean"]["self"]),
                _format_float(bucket["attention_overlap_mass_mean"]["middle"]),
            ]
        )
    lines.append(
        _markdown_table(
            ["type", "attn_bos", "attn_edge", "attn_recent", "attn_self", "attn_middle"],
            control_rows,
        )
    )
    lines.append("")
    exact_rows = []
    for scope in ("sliding", "full"):
        bucket = layer_type[scope]
        exact_rows.append(
            [
                scope,
                bucket["anchor_count"],
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["bos"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["edge"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["recent_nonself"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["self"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["middle"]),
            ]
        )
    lines.append(
        _markdown_table(
            ["type", "anchor_rows", "exact_bos", "exact_edge", "exact_recent", "exact_self", "exact_middle"],
            exact_rows,
        )
    )
    lines.append("")
    lines.append("Top-contributor histograms use the position with the largest `score_pre(q, k)` for each late query.")
    lines.append("")
    for scope in ("sliding", "full"):
        bucket = layer_type[scope]
        lines.append(f"### {scope.title()} Top Positions")
        lines.append("")
        lines.append(_markdown_table(["abs_pos", "count", "frac"], _top_items(Counter(bucket["top_abs_position_hist"]), bucket["query_count"], top_n=12)))
        lines.append("")
        lines.append(_markdown_table(["query_offset", "count", "frac"], _top_items(Counter(bucket["top_query_offset_hist"]), bucket["query_count"], top_n=12)))
        lines.append("")
        lines.append(_markdown_table(["edge_offset", "count", "frac"], _top_items(Counter(bucket["top_edge_offset_hist"]), bucket["query_count"], top_n=12)))
        lines.append("")

    lines.append("## Layer-Depth Findings")
    lines.append("")
    depth_rows = []
    for bucket_name in ("early_sliding", "mid_sliding", "late_sliding", "full"):
        bucket = depth_bucket[bucket_name]
        depth_rows.append(
            [
                bucket_name,
                bucket["query_count"],
                _format_float(bucket["pre_position_partition_share_mean"]["edge"]),
                _format_float(bucket["pre_position_partition_share_mean"]["recent_nonself"]),
                _format_float(bucket["pre_position_partition_share_mean"]["middle"]),
                _format_float(bucket["top_position_group_frac"]["edge"]),
                _format_float(bucket["top_position_group_frac"]["recent_nonself"]),
                _format_float(bucket["top_position_group_frac"]["middle"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["edge"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["recent_nonself"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["middle"]),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "bucket",
                "rows",
                "pre_edge",
                "pre_recent",
                "pre_middle",
                "top_edge",
                "top_recent",
                "top_middle",
                "exact_edge",
                "exact_recent",
                "exact_middle",
            ],
            depth_rows,
        )
    )
    lines.append("")
    if summary["aggregates"]["secondary_sinks"]:
        secondary_rows = []
        for row in summary["aggregates"]["secondary_sinks"][:12]:
            secondary_rows.append(
                [
                    row["stratum"],
                    row["layer"],
                    _format_float(row["top_fraction"]),
                    row["position"],
                    row["decoded_token"],
                    row["token_type"],
                ]
            )
        lines.append("Candidate secondary sinks are prompt/layer pairs where one middle position dominates at least the configured threshold of late queries:")
        lines.append("")
        lines.append(_markdown_table(["stratum", "layer", "top_frac", "pos", "token", "type"], secondary_rows))
        lines.append("")
    else:
        lines.append("No prompt/layer pair crossed the configured secondary-sink threshold.")
        lines.append("")

    lines.append("## Token Case Studies")
    lines.append("")
    lines.append(
        "These tables show representative prompts only: the first sampled prompt in each stratum, final anchor query (`q = S-1`), "
        f"and representative layers `{', '.join(str(layer) for layer in REPRESENTATIVE_LAYERS)}`. "
        "The full anchor-query dump is in `cases.parquet`."
    )
    lines.append("")
    for spec in active_strata:
        rows = representative.get(spec.name, [])
        prompt = prompt_by_stratum.get(spec.name, [None])[0]
        lines.append(f"### {spec.name}")
        lines.append("")
        if prompt is not None:
            lines.append(
                f"Representative prompt: `sample_id={prompt.sample_id}` `source_idx={prompt.source_idx}` "
                f"`seq_len={prompt.seq_len}` `dataset={prompt.source_meta.get('dataset')}`."
            )
            lines.append("")
        if not rows:
            lines.append("No case rows were captured for this stratum.")
            lines.append("")
            continue
        table_rows = []
        for row in rows:
            table_rows.append(
                [
                    row["layer"],
                    "F" if row["layer_type"] == "full_attention" else "S",
                    row["key_pos"],
                    row["position_group"],
                    row["decoded_token"],
                    row["token_type"],
                    _format_float(row["attention_mass_mean"]),
                    _format_float(row["score_pre"]),
                    _format_float(row["score_resid_exact"]),
                ]
            )
        lines.append(
            _markdown_table(
                ["layer", "type", "key_pos", "pos_group", "token", "tok_group", "attn", "score_pre", "score_exact"],
                table_rows,
            )
        )
        lines.append("")
        lines.append(_interpret_case_rows(rows))
        lines.append("")

    lines.append("## Per-Layer Summary")
    lines.append("")
    layer_rows = []
    for layer_idx in range(len(per_layer)):
        bucket = per_layer[str(layer_idx)]
        layer_rows.append(
            [
                layer_idx,
                "F" if layer_idx in FULL_LAYER_SET else "S",
                bucket["query_count"],
                _format_float(bucket["pre_position_partition_share_mean"]["bos"]),
                _format_float(bucket["pre_position_partition_share_mean"]["edge"]),
                _format_float(bucket["pre_position_partition_share_mean"]["recent_nonself"]),
                _format_float(bucket["pre_position_partition_share_mean"]["self"]),
                _format_float(bucket["pre_position_partition_share_mean"]["middle"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["bos"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["edge"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["recent_nonself"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["self"]),
                _format_float(bucket["anchor_exact_position_partition_share_mean"]["middle"]),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "layer",
                "type",
                "rows",
                "pre_bos",
                "pre_edge",
                "pre_recent",
                "pre_self",
                "pre_middle",
                "exact_bos",
                "exact_edge",
                "exact_recent",
                "exact_self",
                "exact_middle",
            ],
            layer_rows,
        )
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")

    sliding = layer_type["sliding"]
    full = layer_type["full"]
    edge_pre = sliding["pre_position_partition_share_mean"]["edge"]
    middle_pre = sliding["pre_position_partition_share_mean"]["middle"]
    recent_pre = sliding["pre_position_partition_share_mean"]["recent_nonself"]
    edge_exact = sliding["anchor_exact_position_partition_share_mean"]["edge"]
    middle_exact = sliding["anchor_exact_position_partition_share_mean"]["middle"]
    recent_exact = sliding["anchor_exact_position_partition_share_mean"]["recent_nonself"]

    lines.append(
        "Full-attention layers still provide the control case: BOS remains visible and usually captures a sizeable "
        "share of both attention mass and residual-effect score."
    )
    lines.append("")
    if edge_pre > max(middle_pre, recent_pre):
        lines.append(
            "In this run, sliding layers behave like a moving edge sink: once BOS is removed, the left boundary takes over as the dominant backup position."
        )
    elif recent_pre > middle_pre and recent_exact >= edge_exact:
        lines.append(
            "In this run, sliding layers do not recreate BOS at the window edge. They lean more on recent visible tokens, and the anchor-query exact scores say those recent tokens are not just probability reservoirs."
        )
    else:
        lines.append(
            "In this run, sliding layers do not recreate a strong moving-edge BOS substitute. The leftover routing is better described as a distributed interior/recent pattern than as a single backup sink."
        )
    lines.append("")
    if summary["aggregates"]["secondary_sinks"]:
        lines.append(
            "Middle-layer secondary-sink candidates did appear in some prompt/layer pairs, which is consistent with the possibility that activation dynamics can create temporary non-BOS anchors."
        )
    else:
        lines.append(
            "No strong middle-layer secondary sink crossed the configured threshold, so the data favors distributed routing over a single activation-born backup anchor in this sample."
        )
    lines.append("")
    lines.append(
        "The practical answer to the no-op question is therefore layer-dependent: full layers can still offload onto BOS, but sliding layers mostly have to distribute the branch over visible content/special tokens that survive the mask, with exact anchor scores revealing which of those positions materially change the residual update."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _make_summary(
    *,
    generated_at: str,
    data_path: Path,
    source_format: str,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    target_per_stratum: int,
    sampling_mode: str,
    sample_seed: int,
    collector: ResidualContributionCollector,
    prompts: list[SamplePrompt],
    sampling_stats: dict[str, Any],
    active_strata: list[StratumSpec],
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "model_id": MODEL_ID,
        "source_format": source_format,
        "data_path": str(_display_path(data_path)),
        "device": device,
        "dtype": _dtype_name(dtype),
        "max_length": max_length,
        "target_per_stratum": target_per_stratum,
        "sampling_mode": sampling_mode,
        "sample_seed": sample_seed,
        "num_prompts": len(prompts),
        "active_strata": [spec.name for spec in active_strata],
        "sampling": sampling_stats,
        "validation": collector.validation,
        "research_sources": RESEARCH_SOURCES,
        "aggregates": {
            "layer_type": {
                name: _finalize_bucket(bucket) for name, bucket in collector.layer_type_stats.items()
            },
            "depth_bucket": {
                name: _finalize_bucket(bucket) for name, bucket in collector.depth_bucket_stats.items()
            },
            "per_layer": {
                str(layer_idx): _finalize_bucket(bucket)
                for layer_idx, bucket in collector.per_layer_stats.items()
            },
            "secondary_sinks": collector.secondary_sink_rows,
        },
    }


def _run_analysis(
    *,
    prompts: list[SamplePrompt],
    processor,
    model,
    collector: ResidualContributionCollector,
    device: str,
) -> None:
    patched = _install_attention_wrappers(model, collector, collector.layer_types)
    try:
        for prompt in prompts:
            input_ids = collector.start_prompt(prompt)
            input_ids = input_ids.to(device) if device != "cpu" else input_ids
            with torch.no_grad():
                model(input_ids=input_ids)
            collector.finish_prompt()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
    finally:
        _restore_attention_wrappers(patched)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=TOWERBLOCKS_DATA, help="TowerBlocks dataset directory.")
    parser.add_argument("--format", choices=["auto", "towerblocks", "json"], default="auto")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--min-length", type=int, default=513)
    parser.add_argument("--target-per-stratum", type=int, default=20)
    parser.add_argument("--sampling-mode", choices=["random", "first"], default="random")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--strata",
        default=",".join(spec.name for spec in STRATA),
        help="Comma-separated subset of strata to run. Defaults to all planned strata.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print sampling progress every N scanned rows. Use 0 to disable.",
    )
    parser.add_argument("--max-case-positions", type=int, default=8)
    parser.add_argument("--secondary-sink-threshold", type=float, default=0.20)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--case-path", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--prompt-index-path", type=Path, default=DEFAULT_PROMPT_INDEX_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_path = args.data.resolve()
    if not data_path.exists():
        raise SystemExit(f"missing data source: {data_path}")

    source_format = _infer_source_format(data_path) if args.format == "auto" else args.format
    if source_format != "towerblocks":
        raise SystemExit("this runner is designed for TowerBlocks-backed experiments.")

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype, device)
    generated_at = _utc_now_iso()
    requested_strata = [item.strip() for item in args.strata.split(",") if item.strip()]
    active_strata = [spec for spec in STRATA if spec.name in requested_strata]
    if not active_strata:
        raise SystemExit(f"no valid strata selected from --strata={args.strata!r}")

    print(f"[residual-study] source={source_format} path={_display_path(data_path)}")
    print(f"[residual-study] device={device} dtype={_dtype_name(dtype)} max_length={args.max_length}")
    print(f"[residual-study] active_strata={', '.join(spec.name for spec in active_strata)}")
    print(f"[residual-study] loading processor/config for {MODEL_ID} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    text_cfg = AutoConfig.from_pretrained(MODEL_ID).text_config
    layer_types = list(text_cfg.layer_types)
    print(
        f"[residual-study] config loaded in {time.time() - t0:.1f}s "
        f"({len(layer_types)} layers, sliding_window={text_cfg.sliding_window})"
    )

    print("[residual-study] selecting stratified prompts ...")
    t0 = time.time()
    prompts, sampling_stats = _sample_prompts(
        processor=processor,
        data_path=data_path,
        source_format=source_format,
        max_length=args.max_length,
        min_length=args.min_length,
        target_per_stratum=args.target_per_stratum,
        sampling_mode=args.sampling_mode,
        sample_seed=args.sample_seed,
        active_strata=active_strata,
        progress_every=args.progress_every,
    )
    print(
        f"[residual-study] selected {len(prompts)} prompts in {time.time() - t0:.1f}s "
        f"(sampling_mode={args.sampling_mode}, seed={args.sample_seed})"
    )
    for spec in active_strata:
        print(
            f"[residual-study]   {spec.name}: eligible={sampling_stats['eligible_by_stratum'][spec.name]} "
            f"selected={sampling_stats['selected_by_stratum'][spec.name]} "
            f"shortfall={sampling_stats['shortfall_by_stratum'][spec.name]}"
        )

    if not prompts:
        raise SystemExit("no prompts selected; nothing to analyze")

    print("[residual-study] loading model weights ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        dtype=dtype,
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    print(f"[residual-study] model loaded in {time.time() - t0:.1f}s")

    collector = ResidualContributionCollector(
        model=model,
        tokenizer=processor.tokenizer,
        layer_types=layer_types,
        sliding_window=int(text_cfg.sliding_window),
        max_case_positions=args.max_case_positions,
        secondary_sink_threshold=args.secondary_sink_threshold,
    )

    print("[residual-study] running prompt analysis ...")
    t0 = time.time()
    _run_analysis(
        prompts=prompts,
        processor=processor,
        model=model,
        collector=collector,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"[residual-study] analysis finished in {elapsed:.1f}s")

    summary = _make_summary(
        generated_at=generated_at,
        data_path=data_path,
        source_format=source_format,
        device=device,
        dtype=dtype,
        max_length=args.max_length,
        target_per_stratum=args.target_per_stratum,
        sampling_mode=args.sampling_mode,
        sample_seed=args.sample_seed,
        collector=collector,
        prompts=prompts,
        sampling_stats=sampling_stats,
        active_strata=active_strata,
    )

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_prompt_index(args.prompt_index_path, prompts)
    _write_case_parquet(args.case_path, collector.case_rows)
    _render_report(
        generated_at=generated_at,
        prompts=prompts,
        sampling_stats=sampling_stats,
        summary=summary,
        case_rows=collector.case_rows,
        report_path=args.report_path,
        active_strata=active_strata,
    )

    print(f"[residual-study] report: {_display_path(args.report_path)}")
    print(f"[residual-study] summary: {_display_path(args.summary_path)}")
    print(f"[residual-study] cases: {_display_path(args.case_path)}")
    print(f"[residual-study] prompt index: {_display_path(args.prompt_index_path)}")


if __name__ == "__main__":
    main()
