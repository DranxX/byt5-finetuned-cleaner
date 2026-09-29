# Notebook Colab — Pilot mT5-small (single T4)

Cara pakai: buka [colab.research.google.com](https://colab.research.google.com),
pilih runtime **T4 GPU**, lalu copy-paste cell di bawah satu per satu.

Tujuan pilot ini: **bukti pipeline jalan end-to-end** (env → data → LoRA train →
inference) di GPU gratisan sebelum diputer di 3070 Ti. Kualitas model belum
jadi fokus — kalau ini lancar, setup yang sama dipindah ke mesin lokal.

> **Catatan T4:** compute capability sm_75 (Turing) → **gak support bf16**.
> Notebook ini pakai fp16. 3070 Ti (sm_86) nanti pakai `--bf16` di mesin asli.
> FA2 juga gak relevan (mT5 gak support FA2 — SDPA otomatis).

---

## Cell 1 — GPU & environment checkup (WAJIB duluan)

```python
# --- Cell 1: cek GPU & torch build ----------------------------------------
import subprocess, sys

r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", r.stdout.strip() or "TIDAK ADA — aktifkan T4 di Runtime > Change runtime type")

import torch
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "GPU gak terdeteksi — stop, jangan lanjut"

props = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU 0: {props.name} | sm_{cap[0]}{cap[1]} | VRAM {props.total_memory/2**30:.1f} GB")
print(f"bf16 support: {cap >= (8,0)} (T4=False → pakai fp16)")
```

Kalau cell ini gagal → runtime belum pakai GPU. Jangan lanjut ke cell lain.

## Cell 2 — Install deps (pin yang sama dengan repo)

```python
# --- Cell 2: install pin yang diuji ----------------------------------------
# Colab sudah punya torch + transformers (build CUDA). Kita pin library sisanya
# supaya identik dengan repo (transformers 4.55.4 diperlukan utk t5gemma2 nanti;
# mt5 jalan di versi ini juga).
%pip install -q "transformers==4.55.4" "datasets==3.2.0" "accelerate==1.2.1" \
    "peft==0.14.0" "bitsandbytes>=0.45.0" "sentencepiece==0.2.0" \
    "tokenizers>=0.21,<0.22" "safetensors>=0.4.3"

# torchao pre-installed di beberapa image Colab bikin PEFT raise ImportError:
!pip uninstall -y torchao -q 2>/dev/null || true

import importlib, importlib.metadata as im
for p in ("transformers", "peft", "datasets", "accelerate", "bitsandbytes"):
    try:
        print(f"{p} {im.version(p)}")
    except Exception:
        print(f"{p} TIDAK TERINSTALL")
```

Setelah cell ini: **Runtime > Restart session** kalau transformers sebelumnya
sudah di-import (Colab kadang perlu restart setelah ganti versi).

## Cell 3 — Download dataset & cek shard

```python
# --- Cell 3: dataset (parquet dari HF) + verifikasi ------------------------
from huggingface_hub import snapshot_download
import glob, pyarrow.parquet as pq

path = snapshot_download("DranxX/corpus-cleaning-v1", repo_type="dataset",
                         allow_patterns=["data/train-00000-of-00008.parquet"])  # 1 shard hemat
# NOTE: file parquet ada di folder data/ di dalam HF repo — pattern wajib pakai prefix data/
files = sorted(glob.glob(f"{path}/train-*.parquet")) or sorted(glob.glob(f"{path}/data/train-*.parquet"))
print(f"{len(files)} shard")
assert files, "parquet gak ketemu — cek allow_patterns harus 'data/train-...'"

total, langs = 0, {}
for fn in files:
    md = pq.ParquetFile(fn).metadata
    total += md.num_rows
print(f"total rows: {total:,}  (harus 178,922 = shard 0 utuh)")
assert total == 178_922, "shard gak utuh!"
```

> 1 shard (shard 0) = 178,922 rows ≈ 55 MB, isinya id semua — **cukup buat pilot**.
> Kolom bahasa lain (en/zh) ada di shard 5–7; buat ngetes pipeline, satu shard id
> sudah mewakili. Full 8 shard nanti di training asli.

## Cell 4 — Training (mT5-small + LoRA)

```python
# --- Cell 4: finetune mt5-small --------------------------------------------
# DATASET_PATH di-set dari Cell 3 (path snapshot). Kalau jalankan cell ini
# terpisah, set manual: DATASET_PATH = path dari Cell 3
DATASET_PATH = path   # dari Cell 3; folder snapshot yang isinya data/train-00000-...

import sys, os
sys.path.insert(0, "/content/corpus-finetuned/src")  # kalau clone repo; ATAU inline di bawah

# Opsi A (disarankan): clone repo biar pakai src/config.py dst.
!git clone https://github.com/DranxX/byt5-finetuned-cleaner.git corpus-finetuned 2>/dev/null || true
# NOTE: repo lama nama-nya byt5-finetuned-cleaner; kalau sudah di-rename di GitHub,
# ganti ke DranxX/corpus-finetuned.

# Opsi B (inline, tanpa repo): seluruh logic penting ada di cell ini.
import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, TaskType, get_peft_model

MODEL = "google/mt5-small"
MAX_IN, MAX_TGT = 512, 256          # token budget (id median 92 tok, p99 225)
BATCH, ACCUM = 16, 2                # T4 16GB: batch 16-32 aman dgn ckpting
LR = 2e-4
EPOCHS = 1.0                       # PILOT 1 shard: 178K rows, 1 epoch penuh = ~11K steps
                                   # kualitas masih kasar, tapi lebih informatif dr 0.2

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, attn_implementation="sdpa")
model.config.use_cache = False

lora = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
                  lora_dropout=0.05,
                  target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # expect ~0.4% trainable

ds = load_dataset("parquet", data_files={"train": f"{DATASET_PATH}/data/train-*.parquet"},
                  split="train")
ds = ds.train_test_split(test_size=0.02, seed=42)   # pilot: val 2% aja
print(ds)

def preprocess(ex):
    src = f"<{ex['lang']}> {ex['raw']}"
    enc = tok(src, truncation=True, max_length=MAX_IN)
    lab = tok(ex["clean"], truncation=True, max_length=MAX_TGT)
    return {"input_ids": enc["input_ids"], "labels": lab["input_ids"]}

train_ds = ds["train"].map(preprocess, remove_columns=ds["train"].column_names,
                           desc="tokenize train")
val_ds   = ds["test"].map(preprocess,  remove_columns=ds["test"].column_names,
                          desc="tokenize val")

from torch.utils.data import Dataset
class PadCollator:
    def __call__(self, features):
        import torch
        max_in  = max(len(f["input_ids"]) for f in features)
        max_lab = max(len(f["labels"]) for f in features)
        return {
            "input_ids": torch.tensor([f["input_ids"] + [tok.pad_token_id]*(max_in-len(f["input_ids"])) for f in features]),
            "attention_mask": torch.tensor([[1]*len(f["input_ids"]) + [0]*(max_in-len(f["input_ids"])) for f in features]),
            # mT5: label pad = pad_token_id DAN diganti -100 (label shift dilakukan model)
            "labels": torch.tensor([
                [x if x != tok.pad_token_id else -100 for x in f["labels"] + [-100]*(max_lab-len(f["labels"]))]
                for f in features]),
        }

args = TrainingArguments(
    output_dir="/content/models/mt5s_pilot",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},   # WAJIB <4.49 & aman di semua
    fp16=True,                    # T4: fp16 (bukan bf16!)
    optim="paged_adamw_8bit",
    learning_rate=LR,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=25,
    eval_strategy="steps", eval_steps=500,
    save_strategy="steps", save_steps=500, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    group_by_length=True,
    dataloader_num_workers=2,
    report_to="none",
    seed=42,
)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=PadCollator())
trainer.train()

model.save_pretrained("/content/models/mt5s_pilot/final")
tok.save_pretrained("/content/models/mt5s_pilot/final")
print("adapter saved -> /content/models/mt5s_pilot/final")
```

Perkiraan: T4 + batch 16×accum 2, fp16, seq 512/256 → **~2.5–4 it/s**.
1 shard (178K rows, 95% train) ≈ 10.6K steps ≈ **1.5–2.5 jam**. VRAM ~6–8 GB (T4 16 GB aman).

## Cell 5 — Inference test

```python
# --- Cell 5: uji adapter ----------------------------------------------------
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", attn_implementation="sdpa")
model = PeftModel.from_pretrained(base, "/content/models/mt5s_pilot/final")
tok = AutoTokenizer.from_pretrained("/content/models/mt5s_pilot/final")
model.eval().cuda()

@torch.no_grad()
def clean(text, lang="id", max_new=256, beams=1):
    ids = tok(f"<{lang}> {text}", return_tensors="pt", truncation=True, max_length=512).to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, num_beams=beams, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

tests = [
    ("id", "No. 24; Diperbarui Maret 2011  \nKlik di sini untuk mengunduh dan mencetak versi PDF dokumen ini.  \nOrang tua biasanya yang pertama mengenali bahwa anak mereka mengalami masalah emosional."),
    ("id", "Dijual  Ce1paISSN9754-701  8~E,-ISSN:2898-20717S.~  R.PratiwiK--a,j  ia--n0x00P,sikolo--giSosial"),
    ("en", "Foxy Trend Brass Silver Contemporary Contemporary/Fashion None Necklace | April 2019 | Deal70\nDeals Of The Day\nPromotions\nHandpicked\nCoupons\nSearch\nExample: iPhone Xs, Redmi Note 7..."),
]
for lang, t in tests:
    print(f"\n[{lang}] RAW  : {t[:90]}...")
    print(f"[{lang}] CLEAN: {clean(t, lang)[:200]}")
```

Yang dicari di output: baris nav/menu/boilerplate hilang, konten inti utuh,
karakter rusak (`0x00`, `~`, digit terselip) keberesin. Kualitas penuh butuh
training 1 epoch penuh — ini cuma sanity check.

## Cell 6 — (opsional) tarik adapter ke lokal

```python
# --- Cell 6: zip adapter buat didownload ------------------------------------
!zip -r /content/mt5s_pilot_adapter.zip /content/models/mt5s_pilot/final -i "*"
from google.colab import files
files.download("/content/mt5s_pilot_adapter.zip")
```

Adapter mt5-small LoRA r=32 ≈ 5 MB — kecil, gak perlu upload ke HF dulu.

---

## Checklist kalau error

| Gejala | Fix |
|---|---|
| Cell 1: GPU kosong | Runtime > Change runtime type > T4 GPU, restart |
| Setelah Cell 2: import error transformers | Restart session (Runtime > Restart), ulangi dari Cell 2 |
| `torchao` ImportError saat inject LoRA | `!pip uninstall -y torchao` lalu restart session |
| OOM di Cell 4 | `BATCH=8, ACCUM=4` |
| Loss `nan` | turunkan LR ke 1e-4; T4 fp16 kadang sensitif |
| `element 0 ... does not require grad` | pastikan `gradient_checkpointing_kwargs={"use_reentrant": False}` ada (sudah di cell) |
| Tokenizer error `<id>` prefix | pastikan pakai format `f"<{lang}> {raw}"` yang sama di training & inference |

## Setelah pilot sukses

1. Simpan angka: it/s aktual, VRAM peak (`torch.cuda.max_memory_allocated()/2**30`), eval loss.
2. Bawa setup yang sama ke 3070 Ti: ganti `fp16=True` → `--bf16`, batch bisa naik.
3. Config `t5gemma-270m` tinggal ganti `MODEL` + LoRA targets (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`) — atau pakai `src/config.py` dari repo.
