"""
Fine-tune ByT5 (medium/small) dengan LoRA untuk task text cleaning.

Dataset: hasil pretokenize.py (parquet dgn kolom input_ids/labels),
atau HF dataset (pretokenize on-the-fly).

Usage:
  python src/finetune.py --data-dir dataset/tok --model google/byt5-medium --out models/m1
  python src/finetune.py --hf-dataset DranyX/corpus-cleaning-v1 --model google/byt5-small --out models/s1
"""
import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class TokenParquetDataset(Dataset):
    """Parquet dgn kolom input_ids/labels."""

    def __init__(self, data_dir: str):
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
        if not files:
            raise SystemExit(f"tidak ada parquet train-* di {data_dir}")
        self.input_ids, self.labels = [], []
        for fn in files:
            t = pq.read_table(fn)
            self.input_ids.extend(t.column("input_ids").to_pylist())
            self.labels.extend(t.column("labels").to_pylist())
        print(f"loaded {len(self.input_ids)} rows dari {len(files)} file")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": np.asarray(self.input_ids[idx], dtype=np.int64),
            "labels": np.asarray(self.labels[idx], dtype=np.int64),
        }


class HFDataset(Dataset):
    """HF dataset (lang/raw/clean) — encode on-the-fly via byte+3."""

    BYTE_OFFSET = 3
    EOS = 1

    def __init__(self, hf_dataset, max_input: int, max_target: int):
        self.ds = hf_dataset
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.ds)

    def _enc(self, s: str, max_len: int):
        ids = [b + self.BYTE_OFFSET for b in s.encode("utf-8")[:max_len]]
        if len(ids) < max_len:
            ids.append(self.EOS)
        return ids

    def __getitem__(self, idx):
        ex = self.ds[idx]
        return {
            "input_ids": np.asarray(self._enc(ex["raw"], self.max_input), dtype=np.int64),
            "labels": np.asarray(self._enc(ex["clean"], self.max_target), dtype=np.int64),
        }


class PadCollator:
    """Dynamic padding; label pad = -100 (di-ignore dari loss)."""

    def __init__(self, pad_id: int = 0, label_pad: int = -100):
        self.pad_id = pad_id
        self.label_pad = label_pad

    def __call__(self, features):
        max_in = max(len(f["input_ids"]) for f in features)
        max_lab = max(len(f["labels"]) for f in features)
        batch_in, batch_lab, batch_attn = [], [], []
        for f in features:
            in_ids, lab_ids = list(f["input_ids"]), list(f["labels"])
            in_pad = max_in - len(in_ids)
            batch_in.append(in_ids + [self.pad_id] * in_pad)
            batch_lab.append(lab_ids + [self.label_pad] * (max_lab - len(lab_ids)))
            batch_attn.append([1] * len(in_ids) + [0] * in_pad)
        return {
            "input_ids": torch.tensor(batch_in, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_lab, dtype=torch.long),
        }


def compute_metrics(pred):
    labels, preds = pred.label_ids, pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    mask = labels != -100
    return {"token_accuracy": float((preds[mask] == labels[mask]).mean())}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--hf-dataset", default=None)
    ap.add_argument("--model", default="google/byt5-medium")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-input", type=int, default=1024)
    ap.add_argument("--max-target", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="GPU Ampere+ (30xx/A100)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = ap.parse_args()

    if not args.data_dir and not args.hf_dataset:
        raise SystemExit("isi --data-dir ATAU --hf-dataset")

    from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model

    # ---- dataset
    if args.data_dir:
        full = TokenParquetDataset(args.data_dir)
        n_val = int(len(full) * args.val_frac)
        train_ds, val_ds = torch.utils.data.random_split(
            full, [len(full) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split="train").train_test_split(
            test_size=args.val_frac, seed=args.seed
        )
        train_ds = HFDataset(ds["train"], args.max_input, args.max_target)
        val_ds = HFDataset(ds["test"], args.max_input, args.max_target)

    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    # ---- model + LoRA
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, attn_implementation=args.attn
    )
    model.config.tie_word_embeddings = False
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- training
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        fp16=not args.bf16,
        bf16=args.bf16,
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()

    # ---- save adapter
    save_dir = os.path.join(args.out, "final")
    model.save_pretrained(save_dir)
    import shutil
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in ["tokenizer.json", "tokenizer_config.json", "added_tokens.json"]:
        src = os.path.join(repo_root, f)
        if os.path.exists(src):
            shutil.copy(src, save_dir)
    print(f"saved adapter -> {save_dir}")


if __name__ == "__main__":
    main()
