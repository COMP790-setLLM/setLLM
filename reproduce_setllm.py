#!/usr/bin/env python3
"""Minimal reproducibility scaffold for Set-LLM.

This script implements the core ingredients described in the paper:
  - Set position encoding (SetPE)
  - Set attention masking (SetMask)
  - Benchmark prompt formatting for PIQA / ARC / CSQA / SIQA
  - Pairwise LLM-as-a-judge prompt formatting
  - Optional extra pretraining on a cleaned instruction-answer JSONL
  - LoRA finetuning and permutation-based evaluation

The paper's official code was not included in this repository. This is therefore
an independent implementation intended to reproduce the method as closely as the
paper allows from its text alone.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


IGNORE_INDEX = -100
CANONICAL_JUDGE_LABELS = ("A", "B", "Tie")


@dataclass
class TextSpan:
    text: str


@dataclass
class SetSpan:
    elements: list[str]


MixedPrompt = Sequence[Union[TextSpan, SetSpan]]


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    set_ids: list[int]
    seq_ids: list[int]
    prompt_length: int
    metadata: dict


@dataclass(frozen=True)
class PromptTokens:
    input_ids: list[int]
    position_ids: list[int]
    set_ids: list[int]
    seq_ids: list[int]
    prompt_pattern: torch.Tensor | None = None
    score_positions: tuple[int, ...] | None = None


class SymmetricTieHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, 1, bias=True)

    def forward(self, score_hidden: torch.Tensor) -> torch.Tensor:
        if score_hidden.ndim != 3 or score_hidden.size(1) != 2:
            raise ValueError(
                "SymmetricTieHead expects score_hidden with shape [batch, 2, hidden]."
            )
        mean_hidden = 0.5 * (score_hidden[:, 0] + score_hidden[:, 1])
        abs_diff_hidden = (score_hidden[:, 0] - score_hidden[:, 1]).abs()
        features = torch.cat([mean_hidden, abs_diff_hidden], dim=-1)
        return self.proj(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="HF model id, e.g. google/gemma-2b"
    )
    parser.add_argument(
        "--task",
        choices=["piqa", "arc", "csqa", "siqa", "judge_pairwise"],
        required=True,
        help="Benchmark to finetune/evaluate.",
    )
    parser.add_argument(
        "--architecture",
        choices=["setllm", "setcausal", "vanilla"],
        default="setllm",
        help="Use Set-LLM masking/positions or the base model's native causal attention path.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/setllm-run",
        help="Directory for adapters and metrics.",
    )
    parser.add_argument(
        "--cleaned-ultra-jsonl",
        default=None,
        help="Optional JSONL with cleaned UltraFeedback-style rows: {'instruction': ..., 'answer': ...}",
    )
    parser.add_argument("--extra-pretrain-samples", type=int, default=10_000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--permutations-limit", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Number of judge prompts to score together during evaluation.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=1)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory during training.",
    )
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--update-steps", type=int, default=3000)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Load/train in bfloat16 when CUDA is available.",
    )
    parser.add_argument(
        "--fp32-eval",
        action="store_true",
        help="Run evaluation in float32 as in the paper.",
    )
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run validation every N optimizer updates during benchmark finetuning. Disabled when set to 0.",
    )
    parser.add_argument(
        "--wandb-project", default=None, help="Weights & Biases project name."
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="Weights & Biases entity/team."
    )
    parser.add_argument(
        "--wandb-run-name", default=None, help="Optional Weights & Biases run name."
    )
    parser.add_argument(
        "--judge-train-jsonl",
        default=None,
        help=(
            "JSONL for pairwise judge finetuning with rows like "
            "{'prompt': ..., 'response_a': ..., 'response_b': ..., 'label': 'A'|'B'|'Tie'}"
        ),
    )
    parser.add_argument(
        "--judge-eval-jsonl",
        default=None,
        help=(
            "JSONL for pairwise judge evaluation with rows like "
            "{'prompt': ..., 'response_a': ..., 'response_b': ..., 'label': 'A'|'B'|'Tie'}"
        ),
    )
    parser.add_argument(
        "--judge-prompt-style",
        choices=["named_ab", "anonymous_slots"],
        default="named_ab",
        help=(
            "Prompt template for pairwise judging. "
            "'named_ab' keeps explicit A/B tags inside each response block; "
            "'anonymous_slots' keeps the set elements anonymous and predicts 1/2/Tie."
        ),
    )
    parser.add_argument(
        "--judge-readout",
        choices=["label_generation", "symmetric_scoring"],
        default="label_generation",
        help=(
            "How pairwise judge decisions are read out. "
            "'label_generation' scores A/B/Tie-style answer tokens with LM NLL; "
            "'symmetric_scoring' predicts per-candidate utilities from prompt-side "
            "readout tokens and compares them symmetrically."
        ),
    )
    parser.add_argument(
        "--write-judge-predictions",
        action="store_true",
        help="Write per-example pairwise judge predictions to output_dir for table generation.",
    )
    parser.add_argument(
        "--judge-tie-weight",
        type=float,
        default=1.0,
        help="Optional class weight multiplier for the Tie class under symmetric_scoring.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(args: argparse.Namespace, train: bool) -> torch.dtype:
    if train and args.bf16 and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def tokenize_text(tokenizer: AutoTokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def iter_permutations(
    items: Sequence[str], limit: int | None
) -> Iterator[tuple[str, ...]]:
    perms = itertools.permutations(items)
    if limit is None:
        yield from perms
        return
    for index, perm in enumerate(perms):
        if index >= limit:
            break
        yield perm


def build_setpe_prompt_tokens(
    tokenizer: AutoTokenizer,
    mixed_prompt: Sequence[TextSpan | SetSpan],
    *,
    add_bos: bool,
) -> tuple[list[int], list[int], list[int], list[int]]:
    input_ids: list[int] = []
    position_ids: list[int] = []
    set_ids: list[int] = []
    seq_ids: list[int] = []
    running_pos = 0
    next_set_id = 0

    if add_bos and tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
        position_ids.append(running_pos)
        set_ids.append(-1)
        seq_ids.append(-1)
        running_pos += 1

    for part in mixed_prompt:
        if isinstance(part, TextSpan):
            ids = tokenize_text(tokenizer, part.text)
            input_ids.extend(ids)
            position_ids.extend(range(running_pos, running_pos + len(ids)))
            set_ids.extend([-1] * len(ids))
            seq_ids.extend([-1] * len(ids))
            running_pos += len(ids)
            continue

        start_pos = running_pos
        total_tokens = 0
        for element_index, element_text in enumerate(part.elements):
            ids = tokenize_text(tokenizer, element_text)
            input_ids.extend(ids)
            position_ids.extend(range(start_pos, start_pos + len(ids)))
            set_ids.extend([next_set_id] * len(ids))
            seq_ids.extend([element_index] * len(ids))
            total_tokens += len(ids)
        running_pos += total_tokens
        next_set_id += 1

    return input_ids, position_ids, set_ids, seq_ids


def build_vanilla_prompt_tokens(
    tokenizer: AutoTokenizer,
    mixed_prompt: Sequence[TextSpan | SetSpan],
    *,
    add_bos: bool,
) -> tuple[list[int], list[int], list[int], list[int]]:
    input_ids: list[int] = []

    if add_bos and tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)

    for part in mixed_prompt:
        if isinstance(part, TextSpan):
            input_ids.extend(tokenize_text(tokenizer, part.text))
            continue
        for element_text in part.elements:
            input_ids.extend(tokenize_text(tokenizer, element_text))

    position_ids = list(range(len(input_ids)))
    set_ids = [-1] * len(input_ids)
    seq_ids = [-1] * len(input_ids)
    return input_ids, position_ids, set_ids, seq_ids


def build_prompt_tokens(
    tokenizer: AutoTokenizer,
    mixed_prompt: Sequence[TextSpan | SetSpan],
    *,
    add_bos: bool,
    architecture: str,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if architecture == "vanilla":
        return build_vanilla_prompt_tokens(tokenizer, mixed_prompt, add_bos=add_bos)
    return build_setpe_prompt_tokens(tokenizer, mixed_prompt, add_bos=add_bos)


def build_prompt_token_bundle(
    tokenizer: AutoTokenizer,
    mixed_prompt: Sequence[TextSpan | SetSpan],
    *,
    add_bos: bool,
    architecture: str,
    device: torch.device | None = None,
) -> PromptTokens:
    input_ids, position_ids, set_ids, seq_ids = build_prompt_tokens(
        tokenizer,
        mixed_prompt,
        add_bos=add_bos,
        architecture=architecture,
    )
    prompt_pattern = None
    if architecture != "vanilla":
        prompt_pattern = build_attention_pattern(
            len(input_ids),
            len(input_ids),
            set_ids,
            seq_ids,
            architecture,
            device=device,
        )
    return PromptTokens(
        input_ids=input_ids,
        position_ids=position_ids,
        set_ids=set_ids,
        seq_ids=seq_ids,
        prompt_pattern=prompt_pattern,
    )


def append_response_tokens(
    tokenizer: AutoTokenizer,
    *,
    prompt_ids: list[int],
    prompt_position_ids: list[int],
    prompt_set_ids: list[int],
    prompt_seq_ids: list[int],
    response_text: str,
    add_eos: bool,
) -> EncodedExample:
    response_ids = tokenize_text(tokenizer, response_text)
    if add_eos and tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    start_pos = (prompt_position_ids[-1] + 1) if prompt_position_ids else 0
    response_positions = list(range(start_pos, start_pos + len(response_ids)))
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    position_ids = prompt_position_ids + response_positions
    set_ids = prompt_set_ids + [-1] * len(response_ids)
    seq_ids = prompt_seq_ids + [-1] * len(response_ids)
    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        set_ids=set_ids,
        seq_ids=seq_ids,
        prompt_length=len(prompt_ids),
        metadata={},
    )


def build_attention_pattern(
    prompt_length: int,
    seq_len: int,
    set_ids: Sequence[int],
    seq_ids: Sequence[int],
    architecture: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    target_device = device if device is not None else torch.device("cpu")
    indices = torch.arange(seq_len, device=target_device)
    query_indices = indices.unsqueeze(1)
    key_indices = indices.unsqueeze(0)
    prompt_queries = query_indices < prompt_length
    prompt_keys = key_indices < prompt_length

    set_ids_tensor = torch.as_tensor(set_ids, dtype=torch.long, device=target_device)
    seq_ids_tensor = torch.as_tensor(seq_ids, dtype=torch.long, device=target_device)
    query_set_ids = set_ids_tensor.unsqueeze(1)
    key_set_ids = set_ids_tensor.unsqueeze(0)
    query_seq_ids = seq_ids_tensor.unsqueeze(1)
    key_seq_ids = seq_ids_tensor.unsqueeze(0)

    same_set = (query_set_ids >= 0) & (query_set_ids == key_set_ids)
    different_element = query_seq_ids != key_seq_ids
    cross_element_block = same_set & different_element

    prompt_allowed = prompt_keys & ~cross_element_block
    if architecture == "setcausal":
        prompt_allowed = prompt_allowed & (key_indices <= query_indices)

    # Response tokens always see the full prompt and their own causal prefix.
    response_allowed = prompt_keys | (key_indices <= query_indices)
    return torch.where(prompt_queries, prompt_allowed, response_allowed)


def extend_prompt_attention_pattern(
    prompt_tokens: PromptTokens,
    response_length: int,
    architecture: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    prompt_length = len(prompt_tokens.input_ids)
    seq_len = prompt_length + response_length
    if architecture == "vanilla":
        raise ValueError(
            "extend_prompt_attention_pattern is only for set architectures"
        )
    if prompt_tokens.prompt_pattern is None:
        raise ValueError("prompt_tokens.prompt_pattern must be precomputed")

    target_device = (
        device if device is not None else prompt_tokens.prompt_pattern.device
    )
    prompt_pattern = prompt_tokens.prompt_pattern
    if prompt_pattern.device != target_device:
        prompt_pattern = prompt_pattern.to(target_device)

    pattern = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=target_device)
    pattern[:prompt_length, :prompt_length] = prompt_pattern
    if response_length:
        pattern[prompt_length:, :prompt_length] = True
        response_indices = torch.arange(response_length, device=target_device)
        response_causal = response_indices.unsqueeze(0) <= response_indices.unsqueeze(1)
        pattern[prompt_length:, prompt_length:] = response_causal
    return pattern


def batch_to_model_inputs(
    batch: list[EncodedExample],
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    max_len = max(len(item.input_ids) for item in batch)
    min_value = torch.finfo(dtype).min

    input_ids = []
    labels = []
    position_ids = []
    attention_masks = []

    for item in batch:
        length = len(item.input_ids)
        pad_len = max_len - length
        padded_input_ids = item.input_ids + [pad_id] * pad_len
        padded_labels = item.labels + [IGNORE_INDEX] * pad_len
        padded_positions = item.position_ids + [0] * pad_len
        if architecture == "vanilla":
            attention_mask = [1] * length + [0] * pad_len
            attention_masks.append(attention_mask)
        else:
            pattern = item.metadata.get("attention_pattern")
            if pattern is None:
                pattern = build_attention_pattern(
                    item.prompt_length,
                    length,
                    item.set_ids,
                    item.seq_ids,
                    architecture,
                    device=device,
                )
            elif pattern.device != device:
                pattern = pattern.to(device)
            if pad_len:
                padded_pattern = torch.zeros(
                    (max_len, max_len), dtype=torch.bool, device=device
                )
                padded_pattern[:length, :length] = pattern
                pattern = padded_pattern
            additive_mask = torch.full(
                (max_len, max_len), min_value, dtype=dtype, device=device
            )
            additive_mask.masked_fill_(pattern, 0)
            attention_masks.append(additive_mask)

        input_ids.append(padded_input_ids)
        labels.append(padded_labels)
        position_ids.append(padded_positions)

    model_inputs = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
    }
    if architecture == "vanilla":
        model_inputs["attention_mask"] = torch.tensor(
            attention_masks, dtype=torch.long, device=device
        )
    else:
        model_inputs["attention_mask"] = (
            torch.stack(attention_masks, dim=0).unsqueeze(1).to(device)
        )
    return model_inputs


def forward_for_scoring(
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        position_ids=model_inputs["position_ids"],
    )
    return outputs.logits


def forward_for_hidden_states(
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        position_ids=model_inputs["position_ids"],
        output_hidden_states=True,
        return_dict=True,
    )
    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden states for judge scoring.")
    return outputs.hidden_states[-1]


def prompt_for_piqa(
    example: dict, permutation: Sequence[str] | None = None
) -> tuple[MixedPrompt, str, list[str]]:
    choices = [example["sol1"].strip(), example["sol2"].strip()]
    ordered = list(permutation) if permutation is not None else choices
    answer = choices[example["label"]]
    prompt: MixedPrompt = [
        TextSpan(f"Question: {example['goal'].strip()}\n\nChoices:\n"),
        SetSpan([f"{choice}\n" for choice in ordered]),
        TextSpan("Answer:\n"),
    ]
    return prompt, answer, ordered


def prompt_for_arc(
    example: dict, permutation: Sequence[str] | None = None
) -> tuple[MixedPrompt, str, list[str]]:
    raw_choices = example["choices"]["text"]
    raw_labels = example["choices"]["label"]
    if len(raw_labels) != len(raw_choices):
        raise ValueError("ARC labels and choices must have the same length.")
    label_to_choice = {
        label: choice.strip()
        for label, choice in zip(raw_labels, raw_choices)
    }
    choices = [choice.strip() for choice in raw_choices]
    ordered = list(permutation) if permutation is not None else choices
    answer = label_to_choice[str(example["answerKey"]).strip()]
    prompt: MixedPrompt = [
        TextSpan(f"Question: {example['question'].strip()}\nChoices:\n"),
        SetSpan([f"{choice}\n" for choice in ordered]),
        TextSpan("Answer:\n"),
    ]
    return prompt, answer, ordered


def prompt_for_csqa(
    example: dict, permutation: Sequence[str] | None = None
) -> tuple[MixedPrompt, str, list[str]]:
    raw_choices = example["choices"]["text"]
    raw_labels = example["choices"]["label"]
    if len(raw_labels) != len(raw_choices):
        raise ValueError("CSQA labels and choices must have the same length.")
    label_to_choice = {
        label: choice.strip()
        for label, choice in zip(raw_labels, raw_choices)
    }
    choices = [choice.strip() for choice in raw_choices]
    ordered = list(permutation) if permutation is not None else choices
    answer = label_to_choice[str(example["answerKey"]).strip()]
    prompt: MixedPrompt = [
        TextSpan(f"Question: {example['question'].strip()}\nChoices:\n"),
        SetSpan([f"{choice}\n" for choice in ordered]),
        TextSpan("Answer:\n"),
    ]
    return prompt, answer, ordered


def prompt_for_siqa(
    example: dict, permutation: Sequence[str] | None = None
) -> tuple[MixedPrompt, str, list[str]]:
    choices = [
        example["answerA"].strip(),
        example["answerB"].strip(),
        example["answerC"].strip(),
    ]
    ordered = list(permutation) if permutation is not None else choices
    label_idx = int(example["label"]) - 1
    answer = choices[label_idx]
    prompt: MixedPrompt = [
        TextSpan(
            "Question: Given the context, answer correctly the question.\n"
            f"Context: {example['context'].strip()}\n"
            f"Question: {example['question'].strip()}\n"
            "Choices:\n"
        ),
        SetSpan([f"{choice}\n" for choice in ordered]),
        TextSpan("Answer:\n"),
    ]
    return prompt, answer, ordered


PROMPT_BUILDERS = {
    "piqa": prompt_for_piqa,
    "arc": prompt_for_arc,
    "csqa": prompt_for_csqa,
    "siqa": prompt_for_siqa,
}


HF_DATASETS = {
    "piqa": ("piqa", None, "train", "validation"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "train", "validation"),
    "csqa": ("tau/commonsense_qa", None, "train", "validation"),
    "siqa": ("social_i_qa", None, "train", "validation"),
}


def encode_benchmark_example(
    tokenizer: AutoTokenizer,
    task: str,
    example: dict,
    architecture: str,
) -> EncodedExample:
    prompt_builder = PROMPT_BUILDERS[task]
    prompt, answer, ordered_choices = prompt_builder(example)
    prompt_ids, prompt_positions, prompt_set_ids, prompt_seq_ids = build_prompt_tokens(
        tokenizer,
        prompt,
        add_bos=True,
        architecture=architecture,
    )
    encoded = append_response_tokens(
        tokenizer,
        prompt_ids=prompt_ids,
        prompt_position_ids=prompt_positions,
        prompt_set_ids=prompt_set_ids,
        prompt_seq_ids=prompt_seq_ids,
        response_text=answer,
        add_eos=True,
    )
    encoded.metadata = {
        "answer": answer,
        "choices": ordered_choices,
        "task": task,
    }
    return encoded


def encode_instruction_example(
    tokenizer: AutoTokenizer,
    instruction: str,
    answer: str,
    architecture: str,
) -> EncodedExample:
    prompt = [TextSpan(f"Question: {instruction.strip()}\n\nAnswer:\n")]
    prompt_ids, prompt_positions, prompt_set_ids, prompt_seq_ids = build_prompt_tokens(
        tokenizer,
        prompt,
        add_bos=True,
        architecture=architecture,
    )
    encoded = append_response_tokens(
        tokenizer,
        prompt_ids=prompt_ids,
        prompt_position_ids=prompt_positions,
        prompt_set_ids=prompt_set_ids,
        prompt_seq_ids=prompt_seq_ids,
        response_text=answer.strip(),
        add_eos=True,
    )
    encoded.metadata = {"instruction": instruction}
    return encoded


def normalize_judge_label(label: str) -> str:
    value = label.strip().lower()
    mapping = {
        "a": "A",
        "assistant a": "A",
        "model a": "A",
        "response a": "A",
        "b": "B",
        "assistant b": "B",
        "model b": "B",
        "response b": "B",
        "tie": "Tie",
        "draw": "Tie",
        "equal": "Tie",
    }
    if value not in mapping:
        raise ValueError(f"Unsupported judge label: {label}")
    return mapping[value]


def judge_output_labels(prompt_style: str) -> tuple[str, str, str]:
    if prompt_style == "named_ab":
        return CANONICAL_JUDGE_LABELS
    if prompt_style == "anonymous_slots":
        return ("1", "2", "Tie")
    raise ValueError(f"Unsupported judge prompt style: {prompt_style}")


def judge_scoring_labels(prompt_style: str) -> tuple[str, str, str]:
    first, second, tie = judge_output_labels(prompt_style)
    return (first, tie, second)


def judge_label_index(label: str, prompt_style: str, readout: str) -> int:
    if readout == "label_generation":
        ordered_labels = judge_output_labels(prompt_style)
    elif readout == "symmetric_scoring":
        ordered_labels = judge_scoring_labels(prompt_style)
    else:
        raise ValueError(f"Unsupported judge readout: {readout}")
    return ordered_labels.index(label)


def canonical_to_internal_judge_label(label: str, prompt_style: str) -> str:
    normalized = normalize_judge_label(label)
    if prompt_style == "named_ab":
        return normalized
    if prompt_style == "anonymous_slots":
        mapping = {"A": "1", "B": "2", "Tie": "Tie"}
        return mapping[normalized]
    raise ValueError(f"Unsupported judge prompt style: {prompt_style}")


def internal_to_canonical_judge_label(label: str, prompt_style: str) -> str:
    normalized = label.strip()
    if prompt_style == "named_ab":
        return normalize_judge_label(normalized)
    if prompt_style == "anonymous_slots":
        mapping = {
            "1": "A",
            "candidate 1": "A",
            "first": "A",
            "2": "B",
            "candidate 2": "B",
            "second": "B",
            "tie": "Tie",
        }
        value = normalized.lower()
        if value not in mapping:
            raise ValueError(f"Unsupported internal judge label: {label}")
        return mapping[value]
    raise ValueError(f"Unsupported judge prompt style: {prompt_style}")


def swap_canonical_judge_label(label: str) -> str:
    normalized = normalize_judge_label(label)
    if normalized == "A":
        return "B"
    if normalized == "B":
        return "A"
    return normalized


def swap_internal_judge_label(label: str, prompt_style: str) -> str:
    canonical = internal_to_canonical_judge_label(label, prompt_style)
    swapped = swap_canonical_judge_label(canonical)
    return canonical_to_internal_judge_label(swapped, prompt_style)


def judge_candidate_header(prompt_style: str, candidate_index: int) -> str:
    if prompt_style == "named_ab":
        label = "A" if candidate_index == 0 else "B"
        return f"[Response {label}]\n"
    if prompt_style == "anonymous_slots":
        return "[Candidate]\n"
    raise ValueError(f"Unsupported judge prompt style: {prompt_style}")


def prompt_for_judge_pairwise(
    example: dict,
    *,
    swap: bool = False,
    prompt_style: str = "named_ab",
) -> tuple[MixedPrompt, str]:
    response_a = example["response_a"].strip()
    response_b = example["response_b"].strip()
    label = normalize_judge_label(example["label"])
    if swap:
        response_a, response_b = response_b, response_a
        label = swap_canonical_judge_label(label)

    if prompt_style == "named_ab":
        prompt: MixedPrompt = [
            TextSpan(
                "You are an impartial judge evaluating the quality of two assistant responses.\n"
                "Consider helpfulness, correctness, completeness, and instruction following.\n"
                "Return exactly one label: A, B, or Tie.\n\n"
                f"User prompt:\n{example['prompt'].strip()}\n\n"
                "Candidate responses:\n"
            ),
            SetSpan(
                [
                    f"[Response A]\n{response_a}\n",
                    f"[Response B]\n{response_b}\n",
                ]
            ),
            TextSpan("\nJudgment:\n"),
        ]
    elif prompt_style == "anonymous_slots":
        prompt = [
            TextSpan(
                "You are an impartial judge evaluating the quality of two assistant responses.\n"
                "Consider helpfulness, correctness, completeness, and instruction following.\n"
                "Return exactly one label: 1, 2, or Tie.\n"
                "Label 1 means the first candidate response. Label 2 means the second candidate response.\n\n"
                f"User prompt:\n{example['prompt'].strip()}\n\n"
                "Candidate responses:\n"
            ),
            SetSpan(
                [
                    f"[Candidate]\n{response_a}\n",
                    f"[Candidate]\n{response_b}\n",
                ]
            ),
            TextSpan("\nJudgment:\n"),
        ]
    else:
        raise ValueError(f"Unsupported judge prompt style: {prompt_style}")

    return prompt, canonical_to_internal_judge_label(label, prompt_style)


def build_judge_scoring_prompt_tokens(
    tokenizer: AutoTokenizer,
    example: dict,
    architecture: str,
    *,
    swap: bool = False,
    prompt_style: str = "anonymous_slots",
    device: torch.device | None = None,
) -> tuple[PromptTokens, str]:
    response_a = example["response_a"].strip()
    response_b = example["response_b"].strip()
    label = normalize_judge_label(example["label"])
    if swap:
        response_a, response_b = response_b, response_a
        label = swap_canonical_judge_label(label)

    intro_text = (
        "You are an impartial judge evaluating the quality of two assistant responses.\n"
        "Consider helpfulness, correctness, completeness, and instruction following.\n"
        "Each candidate has an embedded quality-score readout token at the end.\n"
        "Use the shared context plus each candidate's content to form a utility score.\n\n"
        f"User prompt:\n{example['prompt'].strip()}\n\n"
        "Candidate responses:\n"
    )
    score_anchor = "\nOverall quality score"
    element_texts = [
        f"{judge_candidate_header(prompt_style, 0)}{response_a}{score_anchor}",
        f"{judge_candidate_header(prompt_style, 1)}{response_b}{score_anchor}",
    ]
    prompt: MixedPrompt = [TextSpan(intro_text), SetSpan(element_texts)]

    input_ids, position_ids, set_ids, seq_ids = build_prompt_tokens(
        tokenizer,
        prompt,
        add_bos=True,
        architecture=architecture,
    )
    score_positions: list[int] = []
    cursor = 0
    if tokenizer.bos_token_id is not None:
        cursor += 1
    cursor += len(tokenize_text(tokenizer, intro_text))
    for element_text in element_texts:
        element_ids = tokenize_text(tokenizer, element_text)
        score_positions.append(cursor + len(element_ids) - 1)
        cursor += len(element_ids)

    prompt_pattern = None
    if architecture != "vanilla":
        prompt_pattern = build_attention_pattern(
            len(input_ids),
            len(input_ids),
            set_ids,
            seq_ids,
            architecture,
            device=device,
        )

    prompt_tokens = PromptTokens(
        input_ids=input_ids,
        position_ids=position_ids,
        set_ids=set_ids,
        seq_ids=seq_ids,
        prompt_pattern=prompt_pattern,
        score_positions=tuple(score_positions),
    )
    return prompt_tokens, canonical_to_internal_judge_label(label, prompt_style)


def encode_judge_example(
    tokenizer: AutoTokenizer,
    example: dict,
    architecture: str,
    *,
    swap: bool = False,
    prompt_style: str = "named_ab",
) -> EncodedExample:
    prompt, label = prompt_for_judge_pairwise(
        example,
        swap=swap,
        prompt_style=prompt_style,
    )
    prompt_ids, prompt_positions, prompt_set_ids, prompt_seq_ids = build_prompt_tokens(
        tokenizer,
        prompt,
        add_bos=True,
        architecture=architecture,
    )
    encoded = append_response_tokens(
        tokenizer,
        prompt_ids=prompt_ids,
        prompt_position_ids=prompt_positions,
        prompt_set_ids=prompt_set_ids,
        prompt_seq_ids=prompt_seq_ids,
        response_text=label,
        add_eos=True,
    )
    encoded.metadata = {
        "task": "judge_pairwise",
        "label": label,
        "swapped": swap,
    }
    return encoded


def encode_judge_scoring_example(
    tokenizer: AutoTokenizer,
    example: dict,
    architecture: str,
    *,
    swap: bool = False,
    prompt_style: str = "anonymous_slots",
    device: torch.device | None = None,
) -> EncodedExample:
    prompt_tokens, label = build_judge_scoring_prompt_tokens(
        tokenizer,
        example,
        architecture,
        swap=swap,
        prompt_style=prompt_style,
        device=device,
    )
    if prompt_tokens.score_positions is None:
        raise ValueError("Judge scoring prompt must include score positions.")
    metadata: dict[str, Any] = {
        "task": "judge_pairwise",
        "label": label,
        "label_index": judge_label_index(
            label, prompt_style, readout="symmetric_scoring"
        ),
        "swapped": swap,
        "score_positions": prompt_tokens.score_positions,
    }
    if architecture != "vanilla":
        metadata["attention_pattern"] = prompt_tokens.prompt_pattern
    return EncodedExample(
        input_ids=prompt_tokens.input_ids,
        labels=[IGNORE_INDEX] * len(prompt_tokens.input_ids),
        position_ids=prompt_tokens.position_ids,
        set_ids=prompt_tokens.set_ids,
        seq_ids=prompt_tokens.seq_ids,
        prompt_length=len(prompt_tokens.input_ids),
        metadata=metadata,
    )


def select_rows(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(limit))


def load_benchmark_dataset(task: str) -> tuple[Dataset, Dataset]:
    path, subset, train_split, eval_split = HF_DATASETS[task]
    if subset is None:
        train = load_dataset(path, split=train_split)
        eval_ds = load_dataset(path, split=eval_split)
    else:
        train = load_dataset(path, subset, split=train_split)
        eval_ds = load_dataset(path, subset, split=eval_split)
    return train, eval_ds


def load_jsonl_dataset(path: str) -> Dataset:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    return Dataset.from_list(rows)


def load_task_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.task != "judge_pairwise":
        return load_benchmark_dataset(args.task)

    if args.judge_train_jsonl is None or args.judge_eval_jsonl is None:
        raise RuntimeError(
            "judge_pairwise requires both --judge-train-jsonl and --judge-eval-jsonl"
        )

    train_ds = load_jsonl_dataset(args.judge_train_jsonl)
    eval_ds = load_jsonl_dataset(args.judge_eval_jsonl)
    return train_ds, eval_ds


def augment_judge_pairwise_dataset(dataset: Dataset) -> Dataset:
    rows = []
    for row in dataset:
        base = dict(row)
        base["__swap__"] = False
        swapped = dict(row)
        swapped["__swap__"] = True
        rows.append(base)
        rows.append(swapped)
    return Dataset.from_list(rows)


def load_cleaned_ultra(path: str, limit: int | None) -> Dataset:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({"instruction": row["instruction"], "answer": row["answer"]})
            if limit is not None and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def find_lora_target_modules(model: nn.Module) -> list[str]:
    target = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name or "score_head" in name:
            continue
        target.add(name.split(".")[-1])
    if not target:
        raise RuntimeError("No nn.Linear modules found for LoRA.")
    return sorted(target)


def hidden_size_from_config(model: nn.Module) -> int:
    config = model.config
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(text_config, attr, None)
            if value is not None:
                return int(value)
    raise RuntimeError("Could not infer model hidden size for judge scoring head.")


def attach_judge_scoring_modules(model: nn.Module) -> None:
    hidden_size = hidden_size_from_config(model)
    if not hasattr(model, "score_head"):
        model.score_head = nn.Linear(hidden_size, 1, bias=True)
    if not hasattr(model, "tie_head"):
        model.tie_head = SymmetricTieHead(hidden_size)


def get_model_with_scoring_modules(model: nn.Module) -> nn.Module:
    if hasattr(model, "score_head") and hasattr(model, "tie_head"):
        return model
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        if hasattr(base_model, "score_head") and hasattr(base_model, "tie_head"):
            return base_model
    raise RuntimeError("Judge scoring modules are missing from the model.")


def load_model_and_tokenizer(
    args: argparse.Namespace, train: bool
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer)

    dtype = get_dtype(args, train=train)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.task == "judge_pairwise" and args.judge_readout == "symmetric_scoring":
        attach_judge_scoring_modules(model)

    if train:
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        target_modules = find_lora_target_modules(model)
        modules_to_save = None
        if args.task == "judge_pairwise" and args.judge_readout == "symmetric_scoring":
            modules_to_save = ["score_head", "tie_head"]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def resolve_wandb_api_key() -> str | None:
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return env_key

    agents_path = Path("AGENTS.md")
    if not agents_path.exists():
        return None

    for line in agents_path.read_text().splitlines():
        prefix = "wandb api key:"
        if line.lower().startswith(prefix):
            return line.split(":", maxsplit=1)[1].strip() or None
    return None


def init_wandb(args: argparse.Namespace, output_dir: Path) -> Any | None:
    if not args.wandb_project:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it before using --wandb-project."
        ) from exc

    api_key = resolve_wandb_api_key()
    if api_key:
        wandb.login(key=api_key)

    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=config,
        dir=str(output_dir),
    )


def wandb_log(
    run: Any | None, payload: dict[str, Any], *, step: int | None = None
) -> None:
    if run is None:
        return
    run.log(payload, step=step)


def run_training_epoch(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    gradient_accumulation_steps: int,
    global_step: int,
    max_steps: int,
) -> tuple[int, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        model_inputs = batch_to_model_inputs(
            batch, tokenizer, device, dtype, architecture
        )
        outputs = model(**model_inputs)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        running_loss += outputs.loss.detach().float().item()

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step >= max_steps:
                break

    denom = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
    return global_step, running_loss / denom


def train_stage(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    encode_fn,
    args: argparse.Namespace,
    stage_name: str,
    output_dir: Path,
    eval_dataset: Dataset | None = None,
    eval_task: str | None = None,
    wandb_run: Any | None = None,
) -> None:
    device = get_device()
    dtype = get_dtype(args, train=True)
    encoded = [encode_fn(row) for row in dataset]
    dataloader = DataLoader(
        encoded, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: rows
    )
    model.to(device=device, dtype=dtype)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.update_steps,
    )

    global_step = 0
    epoch = 0
    last_eval_step = 0
    while global_step < args.update_steps:
        epoch += 1
        if (
            args.task == "judge_pairwise"
            and args.judge_readout == "symmetric_scoring"
            and stage_name == "judge_pairwise-finetune"
        ):
            global_step, avg_loss = run_judge_scoring_training_epoch(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                architecture=args.architecture,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                judge_tie_weight=args.judge_tie_weight,
                global_step=global_step,
                max_steps=args.update_steps,
            )
        else:
            global_step, avg_loss = run_training_epoch(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                architecture=args.architecture,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                global_step=global_step,
                max_steps=args.update_steps,
            )
        print(
            f"[{stage_name}] epoch={epoch} step={global_step} avg_loss={avg_loss:.4f}",
            flush=True,
        )
        wandb_log(
            wandb_run,
            {
                f"{stage_name}/train_loss": avg_loss,
                f"{stage_name}/epoch": epoch,
            },
            step=global_step,
        )
        should_eval = (
            eval_dataset is not None
            and eval_task is not None
            and args.eval_every > 0
            and global_step > last_eval_step
            and (global_step // args.eval_every) > (last_eval_step // args.eval_every)
        )
        if should_eval:
            assert eval_task is not None
            metrics = evaluate_with_separate_model(
                training_model=model,
                tokenizer=tokenizer,
                dataset=eval_dataset,
                task=eval_task,
                args=args,
                output_dir=output_dir,
            )
            metric_parts = [
                f"{key}={value:.4f}"
                for key, value in metrics.items()
                if key != "num_examples"
            ]
            metric_parts.append(f"num_examples={int(metrics['num_examples'])}")
            print(
                f"[{stage_name}] validation step={global_step} "
                + " ".join(metric_parts),
                flush=True,
            )
            wandb_log(
                wandb_run,
                {
                    f"{stage_name}/val_{key}": (
                        int(value) if key == "num_examples" else value
                    )
                    for key, value in metrics.items()
                },
                step=global_step,
            )
            last_eval_step = global_step
        if args.save_every and global_step and global_step % args.save_every == 0:
            save_path = output_dir / f"{stage_name}-step-{global_step}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    final_path = output_dir / stage_name
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)


def encoded_prompt_only_from_tokens(prompt_tokens: PromptTokens) -> EncodedExample:
    metadata: dict[str, Any] = {}
    if prompt_tokens.prompt_pattern is not None:
        metadata["attention_pattern"] = prompt_tokens.prompt_pattern
    if prompt_tokens.score_positions is not None:
        metadata["score_positions"] = prompt_tokens.score_positions
    return EncodedExample(
        input_ids=prompt_tokens.input_ids,
        labels=[IGNORE_INDEX] * len(prompt_tokens.input_ids),
        position_ids=prompt_tokens.position_ids,
        set_ids=prompt_tokens.set_ids,
        seq_ids=prompt_tokens.seq_ids,
        prompt_length=len(prompt_tokens.input_ids),
        metadata=metadata,
    )


def judge_scoring_logits_from_batch(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    batch: Sequence[EncodedExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("Judge scoring batch must not be empty.")
    score_positions = []
    for item in batch:
        positions = item.metadata.get("score_positions")
        if positions is None or len(positions) != 2:
            raise ValueError("Judge scoring examples must have two score positions.")
        score_positions.append(tuple(int(pos) for pos in positions))

    model_inputs = batch_to_model_inputs(
        list(batch), tokenizer, device, dtype, architecture
    )
    hidden_states = forward_for_hidden_states(model, model_inputs)
    batch_indices = torch.arange(hidden_states.size(0), device=device)
    score_index_tensor = torch.tensor(score_positions, dtype=torch.long, device=device)
    score_hidden = hidden_states[batch_indices.unsqueeze(1), score_index_tensor]

    scoring_model = get_model_with_scoring_modules(model)
    raw_scores = scoring_model.score_head(score_hidden).squeeze(-1)
    diff = raw_scores[:, 0] - raw_scores[:, 1]
    tie_logit = scoring_model.tie_head(score_hidden.to(raw_scores.dtype))
    class_logits = torch.stack([diff, tie_logit, -diff], dim=-1)
    return class_logits, raw_scores


def run_judge_scoring_training_epoch(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    gradient_accumulation_steps: int,
    judge_tie_weight: float,
    global_step: int,
    max_steps: int,
) -> tuple[int, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    class_weights = torch.tensor(
        [1.0, judge_tie_weight, 1.0],
        dtype=torch.float32,
        device=device,
    )

    for step, batch in enumerate(dataloader, start=1):
        class_logits, _ = judge_scoring_logits_from_batch(
            model,
            tokenizer,
            device,
            dtype,
            architecture,
            batch,
        )
        targets = torch.tensor(
            [int(item.metadata["label_index"]) for item in batch],
            dtype=torch.long,
            device=device,
        )
        # Keep the weighted classification loss in fp32 for numerical stability
        # and to avoid bf16/float32 dtype mismatches under CUDA training.
        loss = nn.functional.cross_entropy(
            class_logits.float(),
            targets,
            weight=class_weights,
        )
        (loss / gradient_accumulation_steps).backward()
        running_loss += loss.detach().float().item()

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step >= max_steps:
                break

    denom = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
    return global_step, running_loss / denom


@torch.no_grad()
def score_candidate_answer(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    prompt: MixedPrompt,
    candidate_answer: str,
) -> float:
    prompt_tokens = build_prompt_token_bundle(
        tokenizer,
        prompt,
        add_bos=True,
        architecture=architecture,
        device=device if architecture != "vanilla" else None,
    )
    return score_candidate_answer_from_tokens(
        model,
        tokenizer,
        device,
        dtype,
        architecture,
        prompt_tokens,
        candidate_answer,
    )


@torch.no_grad()
def score_candidate_answer_from_tokens(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    prompt_tokens: PromptTokens,
    candidate_answer: str,
) -> float:
    encoded = append_response_tokens(
        tokenizer,
        prompt_ids=prompt_tokens.input_ids,
        prompt_position_ids=prompt_tokens.position_ids,
        prompt_set_ids=prompt_tokens.set_ids,
        prompt_seq_ids=prompt_tokens.seq_ids,
        response_text=candidate_answer,
        add_eos=True,
    )
    if architecture != "vanilla":
        full_pattern = extend_prompt_attention_pattern(
            prompt_tokens,
            len(encoded.input_ids) - encoded.prompt_length,
            architecture,
            device=device,
        )
        encoded.metadata["attention_pattern"] = full_pattern
    model_inputs = batch_to_model_inputs(
        [encoded], tokenizer, device, dtype, architecture
    )
    logits = forward_for_scoring(model, model_inputs)[:, :-1, :]
    labels = model_inputs["labels"][:, 1:]
    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    token_losses = loss_fct(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    ).view_as(labels)
    answer_mask = labels.ne(IGNORE_INDEX)
    nll = token_losses[answer_mask].sum().item()
    return -nll


@torch.no_grad()
def score_candidate_answers_from_tokens(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    prompt_tokens: PromptTokens,
    candidate_answers: Sequence[str],
) -> list[float]:
    encoded_batch = [
        append_response_tokens(
            tokenizer,
            prompt_ids=prompt_tokens.input_ids,
            prompt_position_ids=prompt_tokens.position_ids,
            prompt_set_ids=prompt_tokens.set_ids,
            prompt_seq_ids=prompt_tokens.seq_ids,
            response_text=answer,
            add_eos=True,
        )
        for answer in candidate_answers
    ]
    if architecture != "vanilla":
        for encoded in encoded_batch:
            encoded.metadata["attention_pattern"] = extend_prompt_attention_pattern(
                prompt_tokens,
                len(encoded.input_ids) - encoded.prompt_length,
                architecture,
                device=device,
            )
    model_inputs = batch_to_model_inputs(
        encoded_batch, tokenizer, device, dtype, architecture
    )
    logits = forward_for_scoring(model, model_inputs)[:, :-1, :]
    labels = model_inputs["labels"][:, 1:]
    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    token_losses = loss_fct(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    ).view_as(labels)
    scores = []
    for batch_index in range(labels.size(0)):
        answer_mask = labels[batch_index].ne(IGNORE_INDEX)
        nll = token_losses[batch_index][answer_mask].sum().item()
        scores.append(-nll)
    return scores


@torch.no_grad()
def classify_judge_pairwise_batch_from_tokens(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    prompt_tokens_batch: Sequence[PromptTokens],
    judge_labels: Sequence[str],
) -> list[str]:
    encoded_batch = []
    for prompt_tokens in prompt_tokens_batch:
        for label in judge_labels:
            encoded = append_response_tokens(
                tokenizer,
                prompt_ids=prompt_tokens.input_ids,
                prompt_position_ids=prompt_tokens.position_ids,
                prompt_set_ids=prompt_tokens.set_ids,
                prompt_seq_ids=prompt_tokens.seq_ids,
                response_text=label,
                add_eos=True,
            )
            if architecture != "vanilla":
                encoded.metadata["attention_pattern"] = extend_prompt_attention_pattern(
                    prompt_tokens,
                    len(encoded.input_ids) - encoded.prompt_length,
                    architecture,
                    device=device,
                )
            encoded_batch.append(encoded)

    model_inputs = batch_to_model_inputs(
        encoded_batch, tokenizer, device, dtype, architecture
    )
    logits = forward_for_scoring(model, model_inputs)[:, :-1, :]
    labels = model_inputs["labels"][:, 1:]
    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    token_losses = loss_fct(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    ).view_as(labels)

    all_scores = []
    for batch_index in range(labels.size(0)):
        answer_mask = labels[batch_index].ne(IGNORE_INDEX)
        nll = token_losses[batch_index][answer_mask].sum().item()
        all_scores.append(-nll)

    predictions = []
    for start in range(0, len(all_scores), len(judge_labels)):
        group = all_scores[start : start + len(judge_labels)]
        best_index = max(range(len(group)), key=group.__getitem__)
        predictions.append(judge_labels[best_index])
    return predictions


@torch.no_grad()
def classify_judge_pairwise_batch_scoring(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    batch: Sequence[EncodedExample],
    prompt_style: str,
) -> list[str]:
    class_logits, _ = judge_scoring_logits_from_batch(
        model,
        tokenizer,
        device,
        dtype,
        architecture,
        batch,
    )
    labels = judge_scoring_labels(prompt_style)
    predictions = []
    for index in class_logits.argmax(dim=-1).tolist():
        predictions.append(labels[index])
    return predictions


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_judge_prediction_rows(
    prediction_rows: Sequence[dict[str, Any]], prompt_style: str
) -> dict[str, float]:
    total = len(prediction_rows)
    if total == 0:
        raise ValueError("No judge prediction rows were provided for summarization.")

    accuracy = 0.0
    swap_consistency = 0.0
    first_position_wins = 0.0
    swapped_first_position_wins = 0.0
    no_tie_total = 0
    no_tie_accuracy = 0.0
    tie_tp = 0
    tie_fp = 0
    tie_fn = 0
    first_position_label = judge_output_labels(prompt_style)[0]

    for row in prediction_rows:
        gold = str(row["gold"])
        pred = str(row["pred"])
        pred_swapped = str(row["pred_swapped"])
        accuracy += 1.0 if pred == gold else 0.0
        swap_consistency += (
            1.0 if pred_swapped == swap_internal_judge_label(pred, prompt_style) else 0.0
        )
        first_position_wins += 1.0 if pred == first_position_label else 0.0
        swapped_first_position_wins += 1.0 if pred_swapped == first_position_label else 0.0

        if gold != "Tie":
            no_tie_total += 1
            no_tie_accuracy += 1.0 if pred == gold else 0.0

        if pred == "Tie" and gold == "Tie":
            tie_tp += 1
        elif pred == "Tie" and gold != "Tie":
            tie_fp += 1
        elif pred != "Tie" and gold == "Tie":
            tie_fn += 1

    tie_precision = tie_tp / max(1, tie_tp + tie_fp)
    tie_recall = tie_tp / max(1, tie_tp + tie_fn)
    tie_f1 = (
        0.0
        if (tie_precision + tie_recall) == 0
        else 2 * tie_precision * tie_recall / (tie_precision + tie_recall)
    )

    return {
        "accuracy": accuracy / total,
        "human_agreement": accuracy / total,
        "no_tie_accuracy": no_tie_accuracy / max(1, no_tie_total),
        "tie_precision": tie_precision,
        "tie_recall": tie_recall,
        "tie_f1": tie_f1,
        "swap_consistency": swap_consistency / total,
        "first_position_win_rate": first_position_wins / total,
        "swapped_first_position_win_rate": swapped_first_position_wins / total,
        "num_examples": total,
        "num_no_tie_examples": no_tie_total,
    }


def maybe_write_judge_predictions(
    prediction_rows: Sequence[dict[str, Any]],
    output_path: Path | None,
) -> None:
    if output_path is None:
        return
    with output_path.open("w") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_example(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    task: str,
    example: dict,
    permutation: Sequence[str],
) -> str:
    prompt_builder = PROMPT_BUILDERS[task]
    prompt, _, ordered_choices = prompt_builder(example, permutation=permutation)
    best_choice = None
    best_score = None
    for choice in ordered_choices:
        score = score_candidate_answer(
            model, tokenizer, device, dtype, architecture, prompt, choice
        )
        if best_score is None or score > best_score:
            best_score = score
            best_choice = choice
    assert best_choice is not None
    return best_choice


def canonical_choices(task: str, example: dict) -> list[str]:
    _, _, choices = PROMPT_BUILDERS[task](example)
    return choices


def gold_answer(task: str, example: dict) -> str:
    _, answer, _ = PROMPT_BUILDERS[task](example)
    return answer


@torch.no_grad()
def classify_judge_pairwise_from_tokens(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    prompt_tokens: PromptTokens,
    judge_labels: Sequence[str],
) -> str:
    scores = score_candidate_answers_from_tokens(
        model,
        tokenizer,
        device,
        dtype,
        architecture,
        prompt_tokens,
        judge_labels,
    )
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return judge_labels[best_index]


@torch.no_grad()
def classify_judge_pairwise(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    architecture: str,
    prompt: Sequence[TextSpan | SetSpan],
    judge_labels: Sequence[str],
) -> str:
    prompt_tokens = build_prompt_token_bundle(
        tokenizer,
        prompt,
        add_bos=True,
        architecture=architecture,
        device=device if architecture != "vanilla" else None,
    )
    return classify_judge_pairwise_from_tokens(
        model,
        tokenizer,
        device,
        dtype,
        architecture,
        prompt_tokens,
        judge_labels,
    )


@torch.no_grad()
def evaluate_judge_pairwise(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    args: argparse.Namespace,
    predictions_path: Path | None = None,
) -> dict[str, float]:
    if args.judge_readout == "symmetric_scoring":
        return evaluate_judge_pairwise_scoring(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
            predictions_path=predictions_path,
        )
    device = get_device()
    dtype = torch.float32 if args.fp32_eval else get_dtype(args, train=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    judge_labels = judge_output_labels(args.judge_prompt_style)
    prediction_rows: list[dict[str, Any]] = []

    rows = list(dataset)
    batch_size = max(1, args.eval_batch_size)
    start = 0
    while start < len(rows):
        current_batch_size = min(batch_size, len(rows) - start)
        batch_rows = rows[start : start + current_batch_size]
        default_prompt_tokens = []
        swapped_prompt_tokens = []
        gold_labels = []
        for row in batch_rows:
            prompt_default, gold = prompt_for_judge_pairwise(
                row,
                swap=False,
                prompt_style=args.judge_prompt_style,
            )
            prompt_swapped, _ = prompt_for_judge_pairwise(
                row,
                swap=True,
                prompt_style=args.judge_prompt_style,
            )
            default_prompt_tokens.append(
                build_prompt_token_bundle(
                    tokenizer,
                    prompt_default,
                    add_bos=True,
                    architecture=args.architecture,
                    device=device if args.architecture != "vanilla" else None,
                )
            )
            swapped_prompt_tokens.append(
                build_prompt_token_bundle(
                    tokenizer,
                    prompt_swapped,
                    add_bos=True,
                    architecture=args.architecture,
                    device=device if args.architecture != "vanilla" else None,
                )
            )
            gold_labels.append(gold)

        try:
            preds = classify_judge_pairwise_batch_from_tokens(
                model,
                tokenizer,
                device,
                dtype,
                args.architecture,
                default_prompt_tokens,
                judge_labels,
            )
            preds_swapped = classify_judge_pairwise_batch_from_tokens(
                model,
                tokenizer,
                device,
                dtype,
                args.architecture,
                swapped_prompt_tokens,
                judge_labels,
            )
        except torch.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            clear_cuda_cache()
            batch_size = max(1, current_batch_size // 2)
            continue

        if not (
            len(gold_labels) == len(preds) == len(preds_swapped)
        ):
            raise ValueError(
                "Judge eval batches must have matching numbers of gold labels and predictions."
            )
        for row, gold, pred, pred_swapped in zip(batch_rows, gold_labels, preds, preds_swapped):
            prediction_rows.append(
                {
                    "prompt": row.get("prompt"),
                    "question_id": row.get("question_id"),
                    "turn": row.get("turn"),
                    "judge": row.get("judge"),
                    "model_a": row.get("model_a"),
                    "model_b": row.get("model_b"),
                    "gold": gold,
                    "gold_canonical": internal_to_canonical_judge_label(
                        gold, args.judge_prompt_style
                    ),
                    "pred": pred,
                    "pred_canonical": internal_to_canonical_judge_label(
                        pred, args.judge_prompt_style
                    ),
                    "pred_swapped": pred_swapped,
                    "pred_swapped_canonical": internal_to_canonical_judge_label(
                        pred_swapped, args.judge_prompt_style
                    ),
                    "swap_consistent": pred_swapped
                    == swap_internal_judge_label(pred, args.judge_prompt_style),
                }
            )
        start += current_batch_size

    maybe_write_judge_predictions(prediction_rows, predictions_path)
    return summarize_judge_prediction_rows(prediction_rows, args.judge_prompt_style)


@torch.no_grad()
def evaluate_judge_pairwise_scoring(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    args: argparse.Namespace,
    predictions_path: Path | None = None,
) -> dict[str, float]:
    device = get_device()
    dtype = torch.float32 if args.fp32_eval else get_dtype(args, train=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    prediction_rows: list[dict[str, Any]] = []

    rows = list(dataset)
    batch_size = max(1, args.eval_batch_size)
    start = 0
    while start < len(rows):
        current_batch_size = min(batch_size, len(rows) - start)
        batch_rows = rows[start : start + current_batch_size]
        default_batch = []
        swapped_batch = []
        gold_labels = []
        for row in batch_rows:
            default_encoded = encode_judge_scoring_example(
                tokenizer,
                row,
                args.architecture,
                swap=False,
                prompt_style=args.judge_prompt_style,
                device=device if args.architecture != "vanilla" else None,
            )
            swapped_encoded = encode_judge_scoring_example(
                tokenizer,
                row,
                args.architecture,
                swap=True,
                prompt_style=args.judge_prompt_style,
                device=device if args.architecture != "vanilla" else None,
            )
            default_batch.append(default_encoded)
            swapped_batch.append(swapped_encoded)
            gold_labels.append(str(default_encoded.metadata["label"]))

        try:
            preds = classify_judge_pairwise_batch_scoring(
                model,
                tokenizer,
                device,
                dtype,
                args.architecture,
                default_batch,
                args.judge_prompt_style,
            )
            preds_swapped = classify_judge_pairwise_batch_scoring(
                model,
                tokenizer,
                device,
                dtype,
                args.architecture,
                swapped_batch,
                args.judge_prompt_style,
            )
        except torch.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            clear_cuda_cache()
            batch_size = max(1, current_batch_size // 2)
            continue

        if not (
            len(gold_labels) == len(preds) == len(preds_swapped)
        ):
            raise ValueError(
                "Judge scoring eval batches must have matching gold labels and predictions."
            )

        for row, gold, pred, pred_swapped in zip(batch_rows, gold_labels, preds, preds_swapped):
            prediction_rows.append(
                {
                    "prompt": row.get("prompt"),
                    "question_id": row.get("question_id"),
                    "turn": row.get("turn"),
                    "judge": row.get("judge"),
                    "model_a": row.get("model_a"),
                    "model_b": row.get("model_b"),
                    "gold": gold,
                    "gold_canonical": internal_to_canonical_judge_label(
                        gold, args.judge_prompt_style
                    ),
                    "pred": pred,
                    "pred_canonical": internal_to_canonical_judge_label(
                        pred, args.judge_prompt_style
                    ),
                    "pred_swapped": pred_swapped,
                    "pred_swapped_canonical": internal_to_canonical_judge_label(
                        pred_swapped, args.judge_prompt_style
                    ),
                    "swap_consistent": pred_swapped
                    == swap_internal_judge_label(pred, args.judge_prompt_style),
                }
            )
        start += current_batch_size

    maybe_write_judge_predictions(prediction_rows, predictions_path)
    return summarize_judge_prediction_rows(prediction_rows, args.judge_prompt_style)


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    task: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    device = get_device()
    dtype = torch.float32 if args.fp32_eval else get_dtype(args, train=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    random_hits = 0.0
    adversarial_hits = 0.0
    total = 0

    for row in dataset:
        choices = canonical_choices(task, row)
        limit = args.permutations_limit
        if task == "csqa" and limit is None:
            limit = 24
        perms = list(iter_permutations(choices, limit))
        predictions = []
        gold = gold_answer(task, row)
        for perm in perms:
            pred = classify_example(
                model, tokenizer, device, dtype, args.architecture, task, row, perm
            )
            predictions.append(pred == gold)

        random_hits += sum(predictions) / len(predictions)
        adversarial_hits += 1.0 if all(predictions) else 0.0
        total += 1

    return {
        "random_order_accuracy": random_hits / total,
        "adversarial_order_accuracy": adversarial_hits / total,
        "num_examples": total,
    }


def evaluate_task(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    task: str,
    args: argparse.Namespace,
    output_dir: Path | None = None,
) -> dict[str, float]:
    if task == "judge_pairwise":
        return evaluate_judge_pairwise(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
            predictions_path=(
                output_dir / f"{task}-predictions.jsonl"
                if output_dir is not None and args.write_judge_predictions
                else None
            ),
        )
    return evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        task=task,
        args=args,
    )


def evaluate_with_separate_model(
    *,
    training_model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    task: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    if not args.do_train:
        return evaluate_task(
            model=training_model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            args=args,
            output_dir=output_dir,
        )

    with tempfile.TemporaryDirectory(prefix="eval-adapter-", dir=output_dir) as tmpdir:
        adapter_dir = Path(tmpdir)
        training_model.save_pretrained(adapter_dir)
        eval_model, eval_tokenizer = load_model_and_tokenizer(args, train=False)
        eval_model = PeftModel.from_pretrained(
            eval_model, adapter_dir, is_trainable=False
        )
        eval_model.config.use_cache = False
        metrics = evaluate_task(
            model=eval_model,
            tokenizer=eval_tokenizer,
            dataset=dataset,
            task=task,
            args=args,
            output_dir=output_dir,
        )
        del eval_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, output_dir)

    try:
        model, tokenizer = load_model_and_tokenizer(args, train=args.do_train)
        train_ds, eval_ds = load_task_datasets(args)

        train_ds = select_rows(train_ds, args.max_train_samples, args.seed)
        eval_ds = select_rows(eval_ds, args.max_eval_samples, args.seed)

        if args.do_train and args.cleaned_ultra_jsonl:
            ultra_ds = load_cleaned_ultra(
                args.cleaned_ultra_jsonl, args.extra_pretrain_samples
            )
            train_stage(
                model=model,
                tokenizer=tokenizer,
                dataset=ultra_ds,
                encode_fn=lambda row: encode_instruction_example(
                    tokenizer,
                    row["instruction"],
                    row["answer"],
                    args.architecture,
                ),
                args=args,
                stage_name="ultra-pretrain",
                output_dir=output_dir,
                wandb_run=wandb_run,
            )

        if args.do_train:
            if args.task == "judge_pairwise":
                if args.judge_readout == "symmetric_scoring":
                    encode_fn = lambda row: encode_judge_scoring_example(
                        tokenizer,
                        row,
                        args.architecture,
                        swap=bool(row.get("__swap__", False)),
                        prompt_style=args.judge_prompt_style,
                        device=get_device()
                        if args.architecture != "vanilla"
                        else None,
                    )
                else:
                    encode_fn = lambda row: encode_judge_example(
                        tokenizer,
                        row,
                        args.architecture,
                        swap=bool(row.get("__swap__", False)),
                        prompt_style=args.judge_prompt_style,
                    )
            else:
                encode_fn = lambda row: encode_benchmark_example(
                    tokenizer, args.task, row, args.architecture
                )
            train_stage(
                model=model,
                tokenizer=tokenizer,
                dataset=(
                    augment_judge_pairwise_dataset(train_ds)
                    if args.task == "judge_pairwise"
                    else train_ds
                ),
                encode_fn=encode_fn,
                args=args,
                stage_name=f"{args.task}-finetune",
                output_dir=output_dir,
                eval_dataset=eval_ds if args.eval_every > 0 else None,
                eval_task=args.task if args.eval_every > 0 else None,
                wandb_run=wandb_run,
            )

        if args.do_eval:
            metrics = evaluate_with_separate_model(
                training_model=model,
                tokenizer=tokenizer,
                dataset=eval_ds,
                task=args.task,
                args=args,
                output_dir=output_dir,
            )
            metrics_path = output_dir / f"{args.task}-metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics, indent=2), flush=True)
            wandb_log(
                wandb_run,
                {
                    f"{args.task}/{key}": (
                        int(value) if key == "num_examples" else value
                    )
                    for key, value in metrics.items()
                },
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
