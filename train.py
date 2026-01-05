import argparse
import inspect
import json
import math
import os
import time

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_latest_checkpoint(run_dir):
    if not os.path.isdir(run_dir):
        return None
    candidates = []
    for name in os.listdir(run_dir):
        if not name.startswith("checkpoint-"):
            continue
        parts = name.split("-")
        if len(parts) != 2:
            continue
        try:
            step = int(parts[1])
        except ValueError:
            continue
        path = os.path.join(run_dir, name)
        if os.path.isdir(path):
            candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def get_split(dataset_dict, split_name, fallback, label):
    name = split_name or fallback
    if name not in dataset_dict:
        available = list(dataset_dict.keys())
        raise ValueError(f"{label} split '{name}' not found. Available splits: {available}")
    return dataset_dict[name]


def ensure_messages_column(split, label):
    if "messages" not in split.column_names:
        raise ValueError(
            f"{label} split missing 'messages' column. Columns: {split.column_names}"
        )


def keep_messages_only(split):
    remove_columns = [col for col in split.column_names if col != "messages"]
    if remove_columns:
        return split.remove_columns(remove_columns)
    return split


def count_trainable_parameters(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


def format_ultrachat_batch(batch, tokenizer):
    texts = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def count_target_modules(model, target_modules):
    matches = 0
    for name, module in model.named_modules():
        if name.split(".")[-1] in target_modules and isinstance(module, torch.nn.Linear):
            matches += 1
    return matches


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (percent / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def sanitize_value(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, jsonl_path, metrics_cfg, tokens_per_step):
        self.jsonl_path = jsonl_path
        self.metrics_cfg = metrics_cfg
        self.tokens_per_step = tokens_per_step
        self._step_start_time = None
        self._last_step_time = None
        self._step_times = []
        self._file = None
        self.latency_stats = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._file = open(self.jsonl_path, "a", encoding="utf-8")

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start_time is None:
            return
        self._last_step_time = time.perf_counter() - self._step_start_time
        self._step_times.append(self._last_step_time)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        if self.metrics_cfg.get("log_step_time", True) and self._last_step_time:
            logs["step_time_sec"] = self._last_step_time

        if (
            self.metrics_cfg.get("log_tokens_per_sec", True)
            and self._last_step_time
            and self._last_step_time > 0
        ):
            logs["tokens_per_sec"] = self.tokens_per_step / self._last_step_time

        if self.metrics_cfg.get("log_max_vram", True):
            if torch.cuda.is_available():
                logs["max_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                logs["max_vram_gb"] = 0.0

        record = {key: sanitize_value(value) for key, value in logs.items()}
        record["step"] = state.global_step
        if state.epoch is not None:
            record["epoch"] = sanitize_value(state.epoch)

        if self._file:
            self._file.write(json.dumps(record, ensure_ascii=True) + "\n")
            self._file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._step_times and self.metrics_cfg.get("log_latency_percentiles", True):
            p50 = percentile(self._step_times, 50)
            p90 = percentile(self._step_times, 90)
            p99 = percentile(self._step_times, 99)
            self.latency_stats = {
                "latency_p50_ms": (p50 * 1000.0) if p50 is not None else None,
                "latency_p90_ms": (p90 * 1000.0) if p90 is not None else None,
                "latency_p99_ms": (p99 * 1000.0) if p99 is not None else None,
                "latency_samples": len(self._step_times),
            }
            if self._file:
                record = {"event": "latency_percentiles", **self.latency_stats}
                self._file.write(json.dumps(record, ensure_ascii=True) + "\n")
                self._file.flush()
        if self._file:
            self._file.close()
            self._file = None


def filter_sft_kwargs(kwargs):
    sig = inspect.signature(SFTConfig.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    filtered = {}
    dropped = []
    for key, value in kwargs.items():
        if key in params:
            filtered[key] = value
        elif key == "evaluation_strategy" and "eval_strategy" in params:
            filtered["eval_strategy"] = value
        elif key == "max_seq_length" and "max_length" in params:
            filtered["max_length"] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"SFTConfig: dropped unsupported args: {dropped}")
    return filtered


def filter_trainer_kwargs(kwargs):
    sig = inspect.signature(SFTTrainer.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    filtered = {}
    dropped = []
    for key, value in kwargs.items():
        if key in params:
            filtered[key] = value
        elif key == "tokenizer" and "processing_class" in params:
            filtered["processing_class"] = value
        elif key == "processing_class" and "tokenizer" in params:
            filtered["tokenizer"] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"SFTTrainer: dropped unsupported args: {dropped}")
    return filtered


def get_effective_seq_len(sft_args, tokenizer_cfg):
    return (
        getattr(sft_args, "max_seq_length", None)
        or getattr(sft_args, "max_length", None)
        or tokenizer_cfg.get("max_seq_length", 1024)
    )


def ensure_assistant_template(tokenizer, model_name, assistant_only_loss):
    if not assistant_only_loss:
        return False
    template = getattr(tokenizer, "chat_template", None)
    if template and "{% generation %}" in template:
        return True
    if "Qwen" in model_name:
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE_WITH_GENERATION
        print("Tokenizer chat_template patched for assistant_only_loss.")
        return True
    print(
        "assistant_only_loss disabled: tokenizer chat_template lacks generation tag."
    )
    return False


def log_assistant_mask_check(trainer):
    dataloader = trainer.get_train_dataloader()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("Assistant-only mask check skipped: empty dataloader.")
        return

    labels = batch.get("labels")
    if labels is None:
        print("Assistant-only mask check skipped: labels not found in batch.")
        return

    sample_labels = labels[0].detach().cpu()
    total = sample_labels.numel()
    masked = int((sample_labels == -100).sum().item())
    unmasked = total - masked
    pct = (masked / total * 100.0) if total else 0.0
    print(
        "Assistant-only mask check (sample 0): "
        f"masked {masked}/{total} ({pct:.1f}%), unmasked {unmasked}"
    )


def build_run_name(base_name, r_value, default_r):
    if r_value == default_r:
        return base_name
    return f"{base_name}_r{r_value}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_cfg = cfg["project"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    sft_cfg = cfg["sft"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["train"]
    metrics_cfg = cfg.get("metrics", {})
    ablation_cfg = cfg.get("ablation", {})

    set_seed(project_cfg.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    tokenizer.padding_side = tokenizer_cfg.get("padding_side", "right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(data_cfg["dataset_name"])
    train_split = get_split(
        raw, data_cfg.get("train_split"), "train", label="Train"
    )
    eval_split = get_split(
        raw, data_cfg.get("eval_split"), "validation", label="Eval"
    )
    ensure_messages_column(train_split, "Train")
    ensure_messages_column(eval_split, "Eval")

    max_train = data_cfg.get("max_train_samples")
    if max_train:
        train_split = train_split.select(range(min(max_train, len(train_split))))
    max_eval = data_cfg.get("max_eval_samples")
    if max_eval:
        eval_split = eval_split.select(range(min(max_eval, len(eval_split))))

    assistant_only_loss = ensure_assistant_template(
        tokenizer, model_cfg["name_or_path"], sft_cfg.get("assistant_only_loss", False)
    )
    if assistant_only_loss:
        train_dataset = keep_messages_only(train_split)
        eval_dataset = keep_messages_only(eval_split)
    else:
        train_dataset = train_split.map(
            lambda batch: format_ultrachat_batch(batch, tokenizer),
            batched=True,
            remove_columns=train_split.column_names,
        )
        eval_dataset = eval_split.map(
            lambda batch: format_ultrachat_batch(batch, tokenizer),
            batched=True,
            remove_columns=eval_split.column_names,
        )

    r_values = ablation_cfg.get("r_values") or [lora_cfg["r"]]
    output_root = project_cfg.get("output_dir", "outputs")

    precision = train_cfg.get("precision", "fp16").lower()
    fp16 = precision == "fp16"
    bf16 = precision == "bf16"

    for r_value in r_values:
        run_name = build_run_name(project_cfg["run_name"], r_value, lora_cfg["r"])
        run_dir = os.path.join(output_root, run_name)
        ensure_dir(run_dir)
        resume_cfg = train_cfg.get("resume_from_checkpoint")
        resume_path = None
        if resume_cfg:
            if isinstance(resume_cfg, str) and resume_cfg.lower() == "auto":
                resume_path = find_latest_checkpoint(run_dir)
            else:
                resume_path = resume_cfg
        if resume_path:
            print(f"Resuming from checkpoint: {resume_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name_or_path"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            torch_dtype=torch.float16 if fp16 else (torch.bfloat16 if bf16 else None),
            low_cpu_mem_usage=True,
        )
        if model_cfg.get("use_cache") is False:
            model.config.use_cache = False

        target_modules = lora_cfg.get("target_modules", [])
        if not target_modules:
            raise ValueError("lora.target_modules is empty; LoRA will not be applied.")
        match_count = count_target_modules(model, target_modules)
        print(f"LoRA target_modules matches: {match_count}")
        if match_count == 0:
            raise ValueError(
                f"No LoRA target_modules matched: {target_modules}. Check model naming."
            )

        alpha_cfg = lora_cfg.get("alpha")
        lora_alpha = 2 * r_value if alpha_cfg is None else alpha_cfg

        peft_config = LoraConfig(
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            r=r_value,
            lora_alpha=lora_alpha,
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            target_modules=target_modules,
        )

        sft_kwargs = {
            "output_dir": run_dir,
            "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 1),
            "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 1),
            "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 1),
            "max_steps": train_cfg.get("max_steps", -1),
            "learning_rate": train_cfg.get("learning_rate", 2e-4),
            "weight_decay": train_cfg.get("weight_decay", 0.0),
            "warmup_ratio": train_cfg.get("warmup_ratio", 0.0),
            "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
            "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
            "logging_steps": train_cfg.get("logging_steps", 10),
            "evaluation_strategy": "steps",
            "eval_steps": train_cfg.get("eval_steps", 100),
            "save_strategy": "steps",
            "save_steps": train_cfg.get("save_steps", 100),
            "save_total_limit": train_cfg.get("save_total_limit", 2),
            "report_to": train_cfg.get("report_to", "none"),
            "fp16": fp16,
            "bf16": bf16,
            "gradient_checkpointing": train_cfg.get("gradient_checkpointing", False),
            "max_seq_length": tokenizer_cfg.get("max_seq_length", 1024),
            "packing": sft_cfg.get("packing", False),
            "assistant_only_loss": assistant_only_loss,
            "seed": project_cfg.get("seed", 42),
            "save_safetensors": True,
        }
        if not assistant_only_loss:
            sft_kwargs["dataset_text_field"] = "text"
        sft_args = SFTConfig(**filter_sft_kwargs(sft_kwargs))

        trainer_kwargs = {
            "model": model,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
            "args": sft_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "peft_config": peft_config,
        }
        trainer = SFTTrainer(**filter_trainer_kwargs(trainer_kwargs))

        trainable, total, pct = count_trainable_parameters(trainer.model)
        print(
            f"Trainable params: {trainable} | Total params: {total} | Trainable%: {pct:.2f}"
        )

        seq_len = get_effective_seq_len(sft_args, tokenizer_cfg)
        world_size = getattr(sft_args, "world_size", 1)
        tokens_per_step_estimate = (
            sft_args.per_device_train_batch_size
            * sft_args.gradient_accumulation_steps
            * world_size
            * seq_len
        )
        metrics_callback = None
        if metrics_cfg.get("log_jsonl", True):
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            metrics_callback = JsonlMetricsCallback(
                metrics_path, metrics_cfg, tokens_per_step_estimate
            )
            trainer.add_callback(metrics_callback)

        if assistant_only_loss and trainer.is_world_process_zero():
            log_assistant_mask_check(trainer)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if resume_path:
            train_result = trainer.train(resume_from_checkpoint=resume_path)
        else:
            train_result = trainer.train()
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        adapter_dir = os.path.join(run_dir, "adapter")
        ensure_dir(adapter_dir)
        trainer.model.save_pretrained(adapter_dir)

        metadata = {
            "run_name": run_name,
            "output_dir": run_dir,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": data_cfg["dataset_name"],
            "train_split": data_cfg.get("train_split"),
            "eval_split": data_cfg.get("eval_split"),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "model": model_cfg["name_or_path"],
            "precision": precision,
            "lora_r": r_value,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_cfg.get("dropout", 0.05),
            "lora_target_modules": target_modules,
            "adapter_dir": adapter_dir,
            "adapter_only": True,
            "tokens_per_step_estimate": tokens_per_step_estimate,
            "tokens_per_sec_is_estimate": True,
            "seq_length_estimate": seq_len,
            "seed": project_cfg.get("seed", 42),
            "config": cfg,
        }
        if metrics_callback and metrics_callback.latency_stats:
            metadata.update(metrics_callback.latency_stats)
        metadata_path = os.path.join(run_dir, "run_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=True)

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
