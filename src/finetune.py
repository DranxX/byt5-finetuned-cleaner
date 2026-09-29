"""
finetune.py — LoRA fine-tune untuk text cleaning (raw -> clean).

Pakai config arsitektur dari config.py:
  --model-family mt5-small      # google/mt5-small (default, tercepat di 8 GB)
  --model-family t5gemma-270m   # google/t5gemma-2-270m-270m (gated: login dulu)
  --model-family byt5-medium    # legacy byte-level (baseline saja, OOM-prone)

Dataset:
  --data-dir  folder parquet dgn kolom lang/raw/clean ATAU input_ids/labels
              (tokenized on-the-fly, byte+3 hanya utk ByT5 legacy)
  --hf-dataset DranxX/corpus-cleaning-v1 (download dari HF)

WAJIB: python src/check_env.py harus exit 0 dulu.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config  # noqa: E402

BYTE_OFFSET = 3   # hanya utk family byt5 (legacy)
EOS_ID = 1


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
def encode_text(s: str, max_len: int, family: str):
    """string -> list ids. hf = AutoTokenizer caller-side; byte = byte+3 manual."""
    if family == "byt5":
        ids = [b + BYTE_OFFSET for b in s.encode("utf-8")[:max_len]]
        if len(ids) < max_len:
            ids.append(EOS_ID)
        return ids
    raise ValueError("family gak dikenal utk encode manual")


class RawDataset(Dataset):
    """Parquet lang/raw/clean, tokenize on-the-fly via AutoTokenizer (hf family)."""

    def __init__(self, data_dir, tokenizer, cfg, max_input, max_target):
        import glob
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not files:
            raise SystemExit(f"tidak ada parquet di {data_dir}")
        self.langs, self.raws, self.cleans = [], [], []
        for fn in files:
            t = pq.read_table(fn, columns=["lang", "raw", "clean"])
            self.langs.extend(t.column("lang").to_pylist())
            self.raws.extend(t.column("raw").to_pylist())
            self.cleans.extend(t.column("clean").to_pylist())
        self.tok = tokenizer
        self.cfg = cfg
        self.max_input = max_input
        self.max_target = max_target
        print(f"loaded {len(self.raws):,} rows dari {len(files)} file")

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx):
        # prefix bahasa membantu model multilingual tau target bahasa apa
        src = f"<{self.langs[idx]}> {self.raws[idx]}"
        enc = self.tok(src, truncation=True, max_length=self.max_input)
        lab = self.tok(self.cleans[idx], truncation=True, max_length=self.max_target)
        return {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
            "labels": np.asarray(lab["input_ids"], dtype=np.int64),
        }


class HFRawDataset(Dataset):
    """HF dataset (lang/raw/clean), tokenize on-the-fly."""

    def __init__(self, hf_ds, tokenizer, cfg, max_input, max_target):
        self.ds = hf_ds
        self.tok = tokenizer
        self.cfg = cfg
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        src = f"<{ex['lang']}> {ex['raw']}"
        enc = self.tok(src, truncation=True, max_length=self.max_input)
        lab = self.tok(ex["clean"], truncation=True, max_length=self.max_target)
        return {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
            "labels": np.asarray(lab["input_ids"], dtype=np.int64),
        }


class ByteDataset(Dataset):
    """Legacy ByT5: parquet dgn input_ids/labels byte-level (dari pretokenizer lama)."""

    def __init__(self, data_dir):
        import glob
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
        if not files:
            raise SystemExit(f"tidak ada parquet train-* di {data_dir}")
        in_flat, in_off = [], [0]
        lab_flat, lab_off = [], [0]
        for fn in files:
            t = pq.read_table(fn, columns=["input_ids", "labels"])
            for col, flat, off in (("input_ids", in_flat, in_off),
                                   ("labels", lab_flat, lab_off)):
                for arr in t.column(col).chunks:
                    vals = np.asarray(arr.values, dtype=np.int32)
                    offs = np.asarray(arr.offsets, dtype=np.int64)
                    start = int(offs[0])
                    flat.append(vals[start:int(offs[-1])])
                    off.extend((offs[1:] - start + off[-1]).tolist())
        self.input_ids = np.concatenate(in_flat) if in_flat else np.empty(0, np.int32)
        self.labels = np.concatenate(lab_flat) if lab_flat else np.empty(0, np.int32)
        self.input_offsets = np.asarray(in_off, dtype=np.int64)
        self.label_offsets = np.asarray(lab_off, dtype=np.int64)
        print(f"loaded {len(self.input_offsets) - 1:,} rows (byte-level flat, "
              f"{(self.input_ids.nbytes + self.labels.nbytes) / 2**30:.2f} GB)")

    def __len__(self):
        return len(self.input_offsets) - 1

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[self.input_offsets[idx]:self.input_offsets[idx + 1]].astype(np.int64),
            "labels": self.labels[self.label_offsets[idx]:self.label_offsets[idx + 1]].astype(np.int64),
        }


class PadCollator:
    """Dynamic padding; label pad = -100 (ignored di loss)."""

    def __init__(self, pad_id=0, label_pad=-100):
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
    ap.add_argument("--model-family", default="mt5-small",
                    help="mt5-small | t5gemma-270m | byt5-medium (legacy)")
    ap.add_argument("--data-dir", default=None, help="folder parquet lokal")
    ap.add_argument("--hf-dataset", default=None, help="atau HF dataset id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-input", type=int, default=None, help="override budget dari config")
    ap.add_argument("--max-target", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="GPU Ampere+ (30xx/A100) — WAJIB di 3070 Ti")
    ap.add_argument("--attn", default=None, choices=[None, "sdpa", "eager"])
    ap.add_argument("--optim", default="paged_adamw_8bit",
                    help="fallback aman: adamw_torch (kalau bitsandbytes bermasalah)")
    args = ap.parse_args()

    if not args.data_dir and not args.hf_dataset:
        raise SystemExit("isi --data-dir ATAU --hf-dataset")

    cfg = get_config(args.model_family)
    max_input = args.max_input or cfg.max_input
    max_target = args.max_target or cfg.max_target
    attn = args.attn or cfg.attn
    print(f"[config] family={cfg.key} repo={cfg.repo}")
    print(f"[config] max_input={max_input} max_target={max_target} attn={attn} lora_targets={cfg.lora_targets}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model

    # ---- tokenizer + dataset
    if cfg.tokenizer == "hf":
        tok = AutoTokenizer.from_pretrained(cfg.repo)
        if args.data_dir:
            full = RawDataset(args.data_dir, tok, cfg, max_input, max_target)
        else:
            from datasets import load_dataset
            ds = load_dataset(args.hf_dataset, split="train").train_test_split(
                test_size=args.val_frac, seed=args.seed)
            full = HFRawDataset(ds["train"], tok, cfg, max_input, max_target)
            val_ds = HFRawDataset(ds["test"], tok, cfg, max_input, max_target)
        if args.data_dir:
            n_val = int(len(full) * args.val_frac)
            train_ds, val_ds = torch.utils.data.random_split(
                full, [len(full) - n_val, n_val],
                generator=torch.Generator().manual_seed(args.seed))
    else:
        # legacy byt5: parquet sudah pretokenized byte-level
        full = ByteDataset(args.data_dir)
        tok = None
        n_val = int(len(full) * args.val_frac)
        train_ds, val_ds = torch.utils.data.random_split(
            full, [len(full) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed))

    print(f"train: {len(train_ds):,} | val: {len(val_ds):,}")

    # ---- model + LoRA
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.repo, attn_implementation=attn)
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=cfg.lora_targets,
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
        # WAJIB utk transformers < 4.49 (default di sana use_reentrant=True
        # yang crash "element 0 ... does not require grad" dgn LoRA frozen embed)
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_checkpointing=True,
        # PENTING: fp16=False utk mT5 — fp16 bikin forward overflow (loss nan).
        # Di GPU tanpa bf16 (T4): biarkan fp32 penuh (bf16 off => fp16 off dgn flag ini)
        fp16=False,   # dilarang; mt5 + fp16 = NaN (verified). T5Gemma aman bf16 di Ampere
        bf16=args.bf16,
        optim=args.optim,
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

    collator = PadCollator(pad_id=tok.pad_token_id if tok else 0)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()

    # ---- save adapter + tokenizer
    save_dir = os.path.join(args.out, "final")
    model.save_pretrained(save_dir)
    if tok is not None:
        tok.save_pretrained(save_dir)
    else:
        import shutil
        for f in ["tokenizer.json", "tokenizer_config.json", "added_tokens.json"]:
            src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f)
            if os.path.exists(src):
                shutil.copy(src, save_dir)
    print(f"saved adapter -> {save_dir}")


if __name__ == "__main__":
    main()
