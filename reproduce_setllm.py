#!/usr/bin/env python3
"""Minimal reproducibility scaffold for Set-LLM.

This script implements the core ingredients described in the paper:
  - Set position encoding (SetPE)
  - Set attention masking (SetMask)
  - Benchmark prompt formatting for PIQA / ARC / CSQA / SIQA
  - Optional extra pretraining on a cleaned instruction-answer JSONL
  - LoRA finetuning and permutation-based evaluation

The paper's official code was not included in this repository. This is therefore
an independent implementation intended to reproduce the method as closely as the
paper allows from its text alone.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


IGNORE_INDEX = -100


@dataclass
class TextSpan:
    text: str


@dataclass
class SetSpan:
    elements: list[str]


MixedPrompt = list[TextSpan | SetSpan]


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    set_ids: list[int]
    seq_ids: list[int]
    prompt_length: int
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id, e.g. google/gemma-2b")
    parser.add_argument(
        "--task",
        choices=["piqa", "arc", "csqa", "siqa"],
        required=True,
        help="Benchmark to finetune/evaluate.",
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--update-steps", type=int, default=3000)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Load/train in bfloat16 when CUDA is available.")
    parser.add_argument("--fp32-eval", action="store_true", help="Run evaluation in float32 as in the paper.")
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run validation every N optimizer updates during benchmark finetuning. Disabled when set to 0.",
    )
    parser.add_argument("--wandb-project", default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity/team.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional Weights & Biases run name.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(args: argparse.Namespace, train: bool) -> torch.dtype:
    if train and args.bf16 and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def tokenize_text(tokenizer: AutoTokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def iter_permutations(items: Sequence[str], limit: int | None) -> Iterator[tuple[str, ...]]:
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
    mixed_prompt: MixedPrompt,
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
) -> torch.Tensor:
    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool)

    for query_idx in range(seq_len):
        for key_idx in range(seq_len):
            if query_idx < prompt_length:
                if key_idx >= prompt_length:
                    continue
                same_set = set_ids[query_idx] >= 0 and set_ids[query_idx] == set_ids[key_idx]
                different_element = seq_ids[query_idx] != seq_ids[key_idx]
                allowed[query_idx, key_idx] = not (same_set and different_element)
                continue

            # Prefix-style prompt attention + causal attention over response.
            allowed[query_idx, key_idx] = key_idx < prompt_length or key_idx <= query_idx

    return allowed


def batch_to_model_inputs(
    batch: list[EncodedExample],
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
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
        pattern = build_attention_pattern(item.prompt_length, length, item.set_ids, item.seq_ids)
        if pad_len:
            padded_pattern = torch.zeros((max_len, max_len), dtype=torch.bool)
            padded_pattern[:length, :length] = pattern
            pattern = padded_pattern
        additive_mask = torch.full((max_len, max_len), min_value, dtype=dtype)
        additive_mask[pattern] = 0

        input_ids.append(padded_input_ids)
        labels.append(padded_labels)
        position_ids.append(padded_positions)
        attention_masks.append(additive_mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
        "attention_mask": torch.stack(attention_masks, dim=0).unsqueeze(1).to(device),
    }


def prompt_for_piqa(example: dict, permutation: Sequence[str] | None = None) -> tuple[MixedPrompt, str, list[str]]:
    choices = [example["sol1"].strip(), example["sol2"].strip()]
    ordered = list(permutation) if permutation is not None else choices
    answer = choices[example["label"]]
    prompt: MixedPrompt = [
        TextSpan(f"Question: {example['goal'].strip()}\n\nChoices:\n"),
        SetSpan([f"{choice}\n" for choice in ordered]),
        TextSpan("Answer:\n"),
    ]
    return prompt, answer, ordered


def prompt_for_arc(example: dict, permutation: Sequence[str] | None = None) -> tuple[MixedPrompt, str, list[str]]:
    raw_choices = example["choices"]["text"]
    raw_labels = example["choices"]["label"]
    label_to_choice = {label: choice.strip() for label, choice in zip(raw_labels, raw_choices, strict=True)}
    choices = [choice.strip() for choice in raw_choices]
    ordered = list(permutation) if permutation is not None else choices
    answer = label_to_choice[str(example["answerKey"]).strip()]
    prompt: MixedPrompt = [
        TextSpan(f"Question: {example['question'].strip()}\nChoices:\n"),
        SetSpan([f"{choice}\n" for choice in ordered]),
        TextSpan("Answer:\n"),
    ]
    return prompt, answer, ordered


def prompt_for_csqa(example: dict, permutation: Sequence[str] | None = None) -> tuple[MixedPrompt, str, list[str]]:
    raw_choices = example["choices"]["text"]
    raw_labels = example["choices"]["label"]
    label_to_choice = {label: choice.strip() for label, choice in zip(raw_labels, raw_choices, strict=True)}
    choices = [choice.strip() for choice in raw_choices]
    ordered = list(permutation) if permutation is not None else choices
    answer = label_to_choice[str(example["answerKey"]).strip()]
    prompt: MixedPrompt = [
        TextSpan(f"Question: {example['question'].strip()}\nChoices:\n"),
        SetSpan([f"{choice}\n" for choice in ordered]),
        TextSpan("Answer:\n"),
    ]
    return prompt, answer, ordered


def prompt_for_siqa(example: dict, permutation: Sequence[str] | None = None) -> tuple[MixedPrompt, str, list[str]]:
    choices = [example["answerA"].strip(), example["answerB"].strip(), example["answerC"].strip()]
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
) -> EncodedExample:
    prompt_builder = PROMPT_BUILDERS[task]
    prompt, answer, ordered_choices = prompt_builder(example)
    prompt_ids, prompt_positions, prompt_set_ids, prompt_seq_ids = build_setpe_prompt_tokens(
        tokenizer,
        prompt,
        add_bos=True,
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
) -> EncodedExample:
    prompt = [TextSpan(f"Question: {instruction.strip()}\n\nAnswer:\n")]
    prompt_ids, prompt_positions, prompt_set_ids, prompt_seq_ids = build_setpe_prompt_tokens(
        tokenizer,
        prompt,
        add_bos=True,
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
        if "lm_head" in name:
            continue
        target.add(name.split(".")[-1])
    if not target:
        raise RuntimeError("No nn.Linear modules found for LoRA.")
    return sorted(target)


def load_model_and_tokenizer(args: argparse.Namespace, train: bool) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer)

    dtype = get_dtype(args, train=train)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    if train:
        target_modules = find_lora_target_modules(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=1,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
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


def wandb_log(run: Any | None, payload: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    run.log(payload, step=step)


def run_training_epoch(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
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
        model_inputs = batch_to_model_inputs(batch, tokenizer, device, dtype)
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
    dataloader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: rows)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
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
        global_step, avg_loss = run_training_epoch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            global_step=global_step,
            max_steps=args.update_steps,
        )
        print(f"[{stage_name}] epoch={epoch} step={global_step} avg_loss={avg_loss:.4f}")
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
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=eval_dataset,
                task=eval_task,
                args=args,
            )
            print(
                f"[{stage_name}] validation step={global_step} "
                f"random_order_accuracy={metrics['random_order_accuracy']:.4f} "
                f"adversarial_order_accuracy={metrics['adversarial_order_accuracy']:.4f} "
                f"num_examples={int(metrics['num_examples'])}"
            )
            wandb_log(
                wandb_run,
                {
                    f"{stage_name}/val_random_order_accuracy": metrics["random_order_accuracy"],
                    f"{stage_name}/val_adversarial_order_accuracy": metrics["adversarial_order_accuracy"],
                    f"{stage_name}/val_num_examples": int(metrics["num_examples"]),
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


@torch.no_grad()
def score_candidate_answer(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    prompt: MixedPrompt,
    candidate_answer: str,
) -> float:
    prompt_ids, prompt_positions, prompt_set_ids, prompt_seq_ids = build_setpe_prompt_tokens(
        tokenizer,
        prompt,
        add_bos=True,
    )
    encoded = append_response_tokens(
        tokenizer,
        prompt_ids=prompt_ids,
        prompt_position_ids=prompt_positions,
        prompt_set_ids=prompt_set_ids,
        prompt_seq_ids=prompt_seq_ids,
        response_text=candidate_answer,
        add_eos=True,
    )
    model_inputs = batch_to_model_inputs([encoded], tokenizer, device, dtype)
    outputs = model(**model_inputs)
    logits = outputs.logits[:, :-1, :]
    labels = model_inputs["labels"][:, 1:]
    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    token_losses = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).view_as(labels)
    answer_mask = labels.ne(IGNORE_INDEX)
    nll = token_losses[answer_mask].sum().item()
    return -nll


def classify_example(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    task: str,
    example: dict,
    permutation: Sequence[str],
) -> str:
    prompt_builder = PROMPT_BUILDERS[task]
    prompt, _, ordered_choices = prompt_builder(example, permutation=permutation)
    best_choice = None
    best_score = None
    for choice in ordered_choices:
        score = score_candidate_answer(model, tokenizer, device, dtype, prompt, choice)
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
            pred = classify_example(model, tokenizer, device, dtype, task, row, perm)
            predictions.append(pred == gold)

        random_hits += sum(predictions) / len(predictions)
        adversarial_hits += 1.0 if all(predictions) else 0.0
        total += 1

    return {
        "random_order_accuracy": random_hits / total,
        "adversarial_order_accuracy": adversarial_hits / total,
        "num_examples": total,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, output_dir)

    try:
        model, tokenizer = load_model_and_tokenizer(args, train=args.do_train)
        train_ds, eval_ds = load_benchmark_dataset(args.task)

        train_ds = select_rows(train_ds, args.max_train_samples, args.seed)
        eval_ds = select_rows(eval_ds, args.max_eval_samples, args.seed)

        if args.do_train and args.cleaned_ultra_jsonl:
            ultra_ds = load_cleaned_ultra(args.cleaned_ultra_jsonl, args.extra_pretrain_samples)
            train_stage(
                model=model,
                tokenizer=tokenizer,
                dataset=ultra_ds,
                encode_fn=lambda row: encode_instruction_example(tokenizer, row["instruction"], row["answer"]),
                args=args,
                stage_name="ultra-pretrain",
                output_dir=output_dir,
                wandb_run=wandb_run,
            )

        if args.do_train:
            train_stage(
                model=model,
                tokenizer=tokenizer,
                dataset=train_ds,
                encode_fn=lambda row: encode_benchmark_example(tokenizer, args.task, row),
                args=args,
                stage_name=f"{args.task}-finetune",
                output_dir=output_dir,
                eval_dataset=eval_ds if args.eval_every > 0 else None,
                eval_task=args.task if args.eval_every > 0 else None,
                wandb_run=wandb_run,
            )

        if args.do_eval:
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=eval_ds,
                task=args.task,
                args=args,
            )
            metrics_path = output_dir / f"{args.task}-metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics, indent=2))
            wandb_log(
                wandb_run,
                {
                    f"{args.task}/random_order_accuracy": metrics["random_order_accuracy"],
                    f"{args.task}/adversarial_order_accuracy": metrics["adversarial_order_accuracy"],
                    f"{args.task}/num_examples": int(metrics["num_examples"]),
                },
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
