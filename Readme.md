# Corpus Cleaning — ByT5 Fine-tuned (LoRA)

Fine-tune **ByT5** untuk task **text cleaning**: input teks kotor (korupsi karakter,
boilerplate, dump web) → output teks bersih.

Repo ini berisi script buat fine-tune `google/byt5-medium` dengan **LoRA** di
dataset [`DranxX/corpus-cleaning-v1`](https://huggingface.co/datasets/DranxX/corpus-cleaning-v1)
(1.431.369 pasangan raw→clean; id/en/zh, 8 shard parquet).

## Isi repo

```
byt5-finetuned/
├── Readme.md              ← file ini
├── requirements.txt       ← dependencies + versi (Kaggle-tested)
├── tokenizer.json         ← tokenizer byte-level ByT5 (format HF tokenizers)
├── tokenizer_config.json  ← config tokenizer ByT5 resmi
├── added_tokens.json      ← special tokens (<pad>, </s>, <extra_id_*>)
├── src/
│   ├── check_tokenizer.py   ← verifikasi tokenizer vs corpus (coverage byte)
│   ├── pretokenize.py       ← pre-tokenisasi dataset -> parquet siap-train (Cepat, byte+3)
│   ├── finetune.py          ← training loop (LoRA, fp16, T4/Ampere-compatible)
│   └── inference.py         ← load adapter + generate clean text
└── binary/
    └── (opsional) byt5-rust-tokenizer binary — encode byte-level ~440x lebih cepat
```

## Kenapa tokenizer-nya statis

ByT5 itu **byte-level**: vocab = 384 token tetap

- id 0-2   : `<pad>`, `</s>`, `<unk>`
- id 3-258 : byte 0-255 (id = byte + 3)
- id 259-383: `<extra_id_0..124>`

Karena semua 256 byte udah ke-cover, gak ada token yang "belum kenal"
— tokenizer **gak perlu di-train ulang** dari corpus. File `tokenizer.json`
di repo ini udah final & verifikasi-nya jalan di `src/check_tokenizer.py`
(benchmark corpus: 242/256 byte terpakai, sisanya tetap valid karena vocab-nya komplet).

## Cara pakai

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Verifikasi tokenizer (opsional tapi disarankan)

```bash
python src/check_tokenizer.py --data-dir /path/ke/parquet-folder
```

### 3. Pre-tokenize dataset (sekali aja, disarankan)

```bash
python src/pretokenize.py --data-dir /path/ke/parquet-folder --out data_tok
```

Output: `data_tok/train-*.parquet` (kolom: `input_ids`, `labels`) — siap dimakan Trainer.

### 4. Fine-tune

```bash
python src/finetune.py \
  --data-dir data_tok \
  --model google/byt5-medium \
  --out output/byt5-medium-cleaner
```

Per-GPU config:

```bash
# T4 x2 (16GB, fp16):
python src/finetune.py --data-dir data_tok --model google/byt5-medium --out output/m1 --batch 8 --accum 2

# RTX 3070 Ti (8GB, bf16):
python src/finetune.py --data-dir data_tok --model google/byt5-medium --out output/m1 --bf16 --batch 4 --accum 4
```

Flag penting:

| Flag | Default | Catatan |
|---|---|---|
| `--model` | `google/byt5-medium` | bisa juga `byt5-small` untuk eksperimen cepat |
| `--max-input` | 1024 | byte, panjang encoder input (3070 Ti: biarin 1024; naik ke 2048 cuma kalau VRAM lega) |
| `--max-target` | 512 | byte, panjang decoder target |
| `--lora-r` | 32 | LoRA rank |
| `--epochs` | 1 | |
| `--batch` | 8 | per-device batch (**3070 Ti 8GB: pakai 4**) |
| `--accum` | 2 | gradient accumulation (**3070 Ti: pakai 4** biar effective batch tetep 32) |
| `--bf16` | off | **wajib ON buat 3070 Ti** (Ampere) — lebih stabil dari fp16 di ByT5 |
| `--attn` | sdpa | memory-efficient attention; eager cuma fallback debug |

### 5. Inference

```python
from src.inference import load_model, clean

model, tok = load_model("output/byt5-medium-cleaner")
print(clean(model, tok, "Dijual  Ce1paISSN9754-701  8~E,-ISSN:2898-20717S.~"))
```

## Hardware

| Setup | Status |
|---|---|
| Kaggle T4 ×2 (16 GB) | ✅ tested (byt5-small, seq 1024/512) |
| RTX 3070 Ti 8GB | ✅ pakai: `--bf16 --batch 4 --accum 4` (bf16 = Ampere native, lebih stabil dari fp16) |
| A100 / H100 | ✅ bisa naikin batch & seq, tambah `--bf16` |

**Catatan FlashAttention:** T4 & RTX 30xx **gak didukung** FlashAttention-2 (butuh Ampere datacenter / Ada+). Script pakai **SDPA** (bawaan PyTorch) — memory-efficient & cepat, otomatis aktif via `--attn sdpa` (default). Kalau GPU lu support FA2 (A100/H100/40xx), install `flash-attn` lalu ganti manual ke `attn_implementation="flash_attention_2"`.

## Catatan desain

1. **fp16, bukan bf16** — T4 (CC 7.5) gak support bf16 native. Di Ampere+ bisa ganti.
2. **`tie_word_embeddings=False`** — eksplisit biar gak ada warning tied-weights.
3. **LoRA target modules**: `q, k, v, o, wi_0, wi_1, wo` (attention + FFN T5).
4. **Label padding = -100** — gak dihitung loss.
5. **Pre-tokenize pakai byte+3 langsung** (identik ByT5Tokenizer, verifikasi di
   `check_tokenizer.py`) — ±440× lebih cepat daripada slow tokenizer Python.
6. **torchao harus di-uninstall** di Kaggle (PEFT 0.19 bentrok dgn torchao < 0.16).
   `requirements.txt` sudah handle, atau `pip uninstall -y torchao` manual.

## Dataset

[`DranxX/corpus-cleaning-v1`](https://huggingface.co/datasets/DranxX/corpus-cleaning-v1):

| lang | rows | % |
|---|---|---|
| id | 1.054.198 | 73,6% |
| en | 293.904 | 20,5% |
| zh | 83.267 | 5,8% |

## Lisensi & kredit

- Dataset dikumpulkan dari `yuyijiong/pretrain-data-clean-delete-only`,
  `wira-pratama/t5-pypdf-cleaner`, `PJMixers-Dev/Nelathan_synthetic-sugar-quill-cleaner`,
  + subset Indonesia self-collected (validated Qwen3.7/3.8/Deepseek V4).
- Base model: [google/byt5-medium](https://huggingface.co/google/byt5-medium).
