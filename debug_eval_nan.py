import argparse
import json
import os

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

QWEN_CHAT_TEMPLATE_WITH_GENERATION = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'system' %}\n"
    "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{% elif message['role'] == 'user' %}\n"
    "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|im_start|>assistant\\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\\n' }}\n"
    "{% endif %}\n"
    "{% endfor %}\n"
    "{% if add_generation_prompt %}\n"
    "{{ '<|im_start|>assistant\\n' }}\n"
    "{% endif %}\n"
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assistant_tokens_survive(messages, tokenizer, max_len):
    if not messages or messages[-1].get("role") != "assistant":
        return 0
    prompt_ids = tokenizer(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        ),
        truncation=False,
    )["input_ids"]
    full_ids = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        truncation=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)
    full_len = len(full_ids)
    if tokenizer.truncation_side == "left":
        tail_start = max(0, full_len - max_len)
        kept = max(0, full_len - max(prompt_len, tail_start))
    else:
        kept = max(0, min(full_len, max_len) - prompt_len)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    sft_cfg = cfg["sft"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    tokenizer.padding_side = tokenizer_cfg.get("padding_side", "right")
    truncation_side = tokenizer_cfg.get("truncation_side")
    if truncation_side:
        tokenizer.truncation_side = truncation_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assistant_only_loss = sft_cfg.get("assistant_only_loss", False)
    template_has_generation = "{% generation %}" in (tokenizer.chat_template or "")
    patched_template = False
    if assistant_only_loss and not template_has_generation and "Qwen" in model_cfg["name_or_path"]:
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE_WITH_GENERATION
        patched_template = True
        template_has_generation = True

    split_name = args.split or data_cfg.get("eval_split", "validation")
    dataset = load_dataset(data_cfg["dataset_name"], split=split_name)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    max_len = tokenizer_cfg.get("max_seq_length", 1024)
    counts = {
        "examples": 0,
        "assistant_last": 0,
        "empty_assistant_content": 0,
        "mask_missing": 0,
        "mask_zero": 0,
        "kept_zero": 0,
    }

    for ex in dataset:
        counts["examples"] += 1
        messages = ex.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            continue
        counts["assistant_last"] += 1
        content = (messages[-1].get("content") or "").strip()
        if not content:
            counts["empty_assistant_content"] += 1

        mask_len = None
        try:
            out = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=max_len,
            )
            mask = out.get("assistant_tokens_mask")
            if mask is not None:
                mask_len = int(sum(mask))
        except Exception:
            mask_len = None

        if mask_len is None:
            counts["mask_missing"] += 1
        elif mask_len == 0:
            counts["mask_zero"] += 1

        if assistant_tokens_survive(messages, tokenizer, max_len) == 0:
            counts["kept_zero"] += 1

    report = {
        "assistant_only_loss": assistant_only_loss,
        "template_has_generation": template_has_generation,
        "patched_template": patched_template,
        "tokenizer_truncation_side": tokenizer.truncation_side,
        "max_seq_length": max_len,
        "split": split_name,
        "counts": counts,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
