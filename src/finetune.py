import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config

PREPROCESS_VERSION = 3
SUPPORTED_LANGUAGES = {"id", "en", "zh"}
LABEL_PAD_ID = -100
PADDING_MULTIPLE = 8

class ArrowRawDataset:

    def __init__(self, ds):
        self.ds = ds

    @classmethod
    def load(cls, data_dir, hf_dataset, tok, max_input, max_target,
             val_frac, seed, val_max, num_proc):
        from datasets import load_dataset
        cache_dir = os.path.join(data_dir or ".", ".hf_cache")
        os.makedirs(cache_dir, exist_ok=True)

        if data_dir:
            import glob
            files = sorted(os.path.realpath(f) for f in
                           glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
            if not files:
                raise SystemExit(f"tidak ada parquet di {data_dir} (scan rekursif)")
            ds = load_dataset("parquet", data_files={"train": files},
                              split="train", cache_dir=cache_dir)
        else:
            ds = load_dataset(hf_dataset, split="train", cache_dir=cache_dir)

        if not {"lang", "raw", "clean"}.issubset(ds.column_names):
            raise SystemExit("dataset harus memiliki kolom lang/raw/clean; gunakan parquet mentah")
        cache_identity = {
            "dataset": ds._fingerprint,
            "tokenizer": tok.name_or_path,
            "vocabulary": tok.get_vocab(),
            "special_tokens": tok.special_tokens_map,
            "max_input": max_input,
            "max_target": max_target,
            "preprocess": PREPROCESS_VERSION,
        }
        src_tag = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:16]

        def preprocess(ex):
            if any(not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES for lang in ex["lang"]):
                raise ValueError("dataset mengandung bahasa yang tidak didukung")
            if any(not isinstance(text, str) for key in ("raw", "clean") for text in ex[key]):
                raise ValueError("raw/clean harus string dan tidak boleh null")
            src = [f"<{lang}> {raw}" for lang, raw in zip(ex["lang"], ex["raw"])]
            enc = tok(src, truncation=False, verbose=False)
            lab = tok(ex["clean"], truncation=False, verbose=False)
            is_input_truncated = [len(ids) > max_input for ids in enc["input_ids"]]
            is_target_truncated = [len(ids) > max_target for ids in lab["input_ids"]]
            for encoded, texts, budget, flags in (
                    (enc, src, max_input, is_input_truncated),
                    (lab, ex["clean"], max_target, is_target_truncated)):
                indices = [index for index, flag in enumerate(flags) if flag]
                if not indices:
                    continue
                clipped = tok([texts[index] for index in indices], truncation=True,
                              max_length=budget, verbose=False)["input_ids"]
                for index, ids in zip(indices, clipped):
                    encoded["input_ids"][index] = ids
            return {"input_ids": enc["input_ids"], "labels": lab["input_ids"],
                    "length": [len(ids) for ids in enc["input_ids"]],
                    "is_input_truncated": is_input_truncated,
                    "is_target_truncated": is_target_truncated}

        tag = f"tok_{src_tag}_in{max_input}_out{max_target}"
        ds = ds.map(preprocess, batched=True, num_proc=num_proc if num_proc > 1 else None,
                    remove_columns=ds.column_names,
                    desc="tokenize", cache_file_name=os.path.join(cache_dir, f"{tag}.arrow"))

        print(f"[budget] input terpotong={sum(ds['is_input_truncated']):,}; "
              f"target terpotong={sum(ds['is_target_truncated']):,}; rows={len(ds):,}")
        ds = ds.remove_columns(["is_input_truncated", "is_target_truncated"])

        split = ds.train_test_split(test_size=val_frac, seed=seed)
        val = split["test"]
        if len(val) > val_max:
            val = val.shuffle(seed=seed).select(range(val_max))
        return cls(split["train"]), cls(val), len(split["train"]), len(val)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        return {
            "input_ids": np.asarray(ex["input_ids"], dtype=np.int64),
            "labels": np.asarray(ex["labels"], dtype=np.int64),
            "length": ex["length"],
        }

class PadCollator:

    def __init__(self, pad_id=0, label_pad=LABEL_PAD_ID):
        self.pad_id = pad_id
        self.label_pad = label_pad

    def __call__(self, features):


        features = [{k: v for k, v in f.items() if k != "length"} for f in features]
        if not features or any(len(f["input_ids"]) == 0 or len(f["labels"]) == 0 for f in features):
            raise ValueError("batch tidak boleh memiliki sekuens kosong")
        max_in = max(len(f["input_ids"]) for f in features)
        max_lab = max(len(f["labels"]) for f in features)
        max_in = math.ceil(max_in / PADDING_MULTIPLE) * PADDING_MULTIPLE
        max_lab = math.ceil(max_lab / PADDING_MULTIPLE) * PADDING_MULTIPLE
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
    mask = labels != LABEL_PAD_ID
    if not mask.any():
        return {"token_accuracy": 0.0}
    return {"token_accuracy": float((preds[mask] == labels[mask]).mean())}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-family", default="umt5-base",
                    help="umt5-base (default/utama) | t5gemma-270m (opsional)")
    ap.add_argument("--data-dir", default=None, help="folder parquet lokal")
    ap.add_argument("--hf-dataset", default=None, help="atau HF dataset id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-input", type=int, default=None, help="override budget dari config")
    ap.add_argument("--max-target", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=2,
                    help="micro-batch per GPU; ukur VRAM pada batch panjang sebelum menaikkan")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="learning rate LoRA; evaluasi loss dan gradient sebelum menaikkan")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=400,
                    help="WAJIB kelipatan bulat dari --eval-steps (load_best_model_at_end)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--val-max", type=int, default=4000,
                    help="cap jumlah row val (eval tiap N step jangan jadi bottleneck)")
    ap.add_argument("--num-proc", type=int, default=4,
                    help="proses paralel tokenize (.map); 1 kalau Windows rewel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="paksa BF16 native pada GPU Ampere+")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--attn", default=None, choices=[None, "sdpa", "eager"])
    ap.add_argument("--optim", default="adamw_torch",
                    help="adamw_torch (default, stabil — hasil pilot T4). "
                         "paged_adamw_8bit cuma kalau VRAM bener-bener sempit")
    args = ap.parse_args()

    positive_ints = (args.batch, args.accum, args.lora_r, args.lora_alpha,
                     args.eval_steps, args.save_steps, args.val_max, args.num_proc)
    if any(value <= 0 for value in positive_ints) or args.warmup < 0:
        raise SystemExit("batch, accum, LoRA, interval, val-max, num-proc harus positif; warmup >= 0")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.epochs, args.lr, args.max_grad_norm)):
        raise SystemExit("epochs, lr, max-grad-norm harus finite dan positif")
    if not math.isfinite(args.val_frac) or not 0 < args.val_frac < 1:
        raise SystemExit("val-frac harus di antara 0 dan 1")
    if any(value is not None and value <= 0 for value in (args.max_input, args.max_target)):
        raise SystemExit("budget sekuens harus positif")
    if bool(args.data_dir) == bool(args.hf_dataset):
        raise SystemExit("isi tepat satu dari --data-dir atau --hf-dataset")
    if args.bf16 and args.precision == "fp32":
        raise SystemExit("--bf16 tidak boleh dipakai bersama --precision fp32")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak tersedia; perbaiki environment lalu jalankan check_env.py")
    is_native_bf16 = all(torch.cuda.get_device_capability(index) >= (8, 0)
                         for index in range(torch.cuda.device_count()))
    should_use_bf16 = args.bf16 or args.precision == "bf16" or (
        args.precision == "auto" and is_native_bf16)
    if should_use_bf16 and not is_native_bf16:
        raise SystemExit("BF16 native memerlukan Ampere+; gunakan --precision fp32 di T4")
    model_dtype = torch.bfloat16 if should_use_bf16 else torch.float32


    import transformers
    major = int(transformers.__version__.split(".")[0])
    if major != 4:
        raise SystemExit(
            f"transformers {transformers.__version__} terdeteksi — project ini diuji di 4.55.4.\n"
            f"transformers 5.x ngapus group_by_length dll. Perbaiki environment:\n"
            f"  python -m pip install transformers==4.55.4\n"
            f"lalu jalankan python src/check_env.py (harus exit 0) sebelum training.")


    if args.save_steps % args.eval_steps != 0:
        raise SystemExit(
            f"--save-steps ({args.save_steps}) harus kelipatan bulat dari "
            f"--eval-steps ({args.eval_steps}) — load_best_model_at_end nolak kombinasi ini")

    cfg = get_config(args.model_family)
    if cfg.model_cls != "auto" and not hasattr(transformers, cfg.model_cls):
        raise SystemExit(
            f"{cfg.key} memerlukan {cfg.model_cls}, yang tidak tersedia di transformers "
            f"{transformers.__version__}. Pin dependency belum mendukung model ini.")
    max_input = args.max_input or cfg.max_input
    max_target = args.max_target or cfg.max_target
    attn = args.attn or cfg.attn
    if cfg.key == "umt5-base" and attn != "eager":
        raise SystemExit("UMT5 pada transformers 4.55.4 hanya mendukung --attn eager")
    print(f"[precision] {model_dtype}; fp16=False")
    print(f"[config] family={cfg.key} repo={cfg.repo}")
    print(f"[config] max_input={max_input} max_target={max_target} attn={attn} lora_targets={cfg.lora_targets}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model

    def load_model_cls(repo, attn):
        if cfg.model_cls == "auto":
            return AutoModelForSeq2SeqLM.from_pretrained(
                repo, torch_dtype=model_dtype, attn_implementation=attn)
        import transformers
        cls = getattr(transformers, cfg.model_cls)
        return cls.from_pretrained(repo, torch_dtype=model_dtype, attn_implementation=attn)


    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(cfg.repo)
    if tok.pad_token_id is None:
        raise SystemExit("tokenizer harus memiliki pad token")


    train_ds, val_ds, n_train, n_val = ArrowRawDataset.load(
        args.data_dir, args.hf_dataset, tok, max_input, max_target,
        args.val_frac, args.seed, args.val_max, args.num_proc)

    print(f"train: {n_train:,} | val: {n_val:,}")


    model = load_model_cls(cfg.repo, attn)
    model.config.use_cache = False
    model.config.encoder_max_length = max_input

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=cfg.lora_targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()


    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,


        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_checkpointing=True,


        fp16=False,
        bf16=should_use_bf16,
        optim=args.optim,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        logging_steps=50,
        logging_nan_inf_filter=False,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        seed=args.seed,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        targs._n_gpu = 1

    collator = PadCollator(pad_id=tok.pad_token_id)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds.ds,
        eval_dataset=val_ds.ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        processing_class=tok,
    )
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    trainer.train(resume_from_checkpoint=last_checkpoint)


    save_dir = os.path.join(args.out, "final")
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "cleaning_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"model_family": cfg.key, "base_model": cfg.repo,
                   "max_input": max_input, "max_target": max_target}, handle, indent=2)
    print(f"saved adapter -> {save_dir}")


if __name__ == "__main__":
    main()
