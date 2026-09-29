# ByT5-Finetuned Cleaner

LoRA fine-tune toolkit for **ByT5** text cleaning: dirty text in, clean text out.

- **Dataset:** [DranxX/corpus-cleaning-v1](https://huggingface.co/datasets/DranxX/corpus-cleaning-v1) — 1.43M raw→clean pairs (id/en/zh)
- **Base model:** [google/byt5-medium](https://huggingface.co/google/byt5-medium) (or `byt5-small` for quick tests)

## Install

```bash
pip install -r requirements.txt
pip uninstall -y torchao   # Kaggle/older envs: PEFT 0.19 breaks on torchao < 0.16
```

> **Windows + GPU:** wheel `pip install torch` dari PyPI itu **CPU-only** —
> seberapa sering di-install ulang pun hasilnya sama. Makanya `requirements.txt`
> sengaja **skip torch di Windows**. Urutan yang bener:
>
> ```bash
> pip uninstall -y torch torchvision torchaudio
> pip install torch --index-url https://download.pytorch.org/whl/cu124   # ← dulu
> pip install -r requirements.txt                                        # torch di-skip otomatis
> python src/check_env.py                                                # harus exit 0
> ```
>
> Pakai `python -m pip install ...` dan pastikan `python`-nya sama dengan yang
> dipakai training — salah environment/venv adalah penyebab klasik
> "udah install CUDA kok tetep kebaca CPU".

## 0. Cek environment dulu

```bash
python src/check_env.py
```

Ngecek: Python/OS, build torch (CPU vs CUDA — penyebab paling umum training
nyasar ke CPU), GPU + VRAM + dukungan bf16, smoke test CUDA runtime,
flash-attn — gak berlaku utk T5/ByT5 (arch gak didukung transformers), versi library vs pin
di `requirements.txt`, dan sisa disk. Exit code `0` = siap,
`1` = ada yang harus dibenerin dulu — bisa juga dipakai buat gate agent/CI.

Output-nya sekalian ngasih rekomendasi flag (mis. `--batch 4 --accum 4 --bf16`
buat 8GB). Kalau ternyata jalan di CPU: STOP, jangan lanjutin — 1.43M rows
berhari-hari sampai berminggu-minggu. Perbaiki torch-nya, bukan script-nya.

## Usage

```bash
# optional: pretokenize once (byte-level, fast)
python src/pretokenize.py --data-dir <parquet-folder> --out dataset/tok

# cek environment (GPU, torch build, libs) — wajib sebelum training
python src/check_env.py

# fine-tune
python src/finetune.py --data-dir dataset/tok --model google/byt5-medium --out models/m1

# RTX 3070 Ti (8GB): bf16 + smaller batch
python src/finetune.py --data-dir dataset/tok --model google/byt5-medium --out models/m1 --bf16 --batch 4 --accum 4

# inference
python src/inference.py --adapter models/m1/final --text "dirty text here"
```

## Directory layout

Scripts create these automatically:

```
dataset/
└── tok/            ← pretokenized parquet (input_ids/labels)
models/
└── m1/
    ├── checkpoint-*/  ← intermediate checkpoints
    └── final/         ← best LoRA adapter + tokenizer files
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | `google/byt5-medium` | `byt5-small` for quick tests |
| `--max-input` / `--max-target` | 1024 / 512 | byte-level sequence budget |
| `--batch` / `--accum` | 8 / 2 | 3070 Ti: use `--batch 4 --accum 4` |
| `--bf16` | off | enable on Ampere+ (30xx/A100) |
| `--attn` | `sdpa` | `sdpa` (default, tercepat) / `eager` — FA2 **gak berlaku** |

## Model & sequence length

ByT5 gak punya tokenizer — setiap byte = 1 token. Teks 500 karakter = **500 token**,
vs ~125 token di SentencePiece. Makanya ByT5 berat soal sekuens:

| | byt5-small (300M) | byt5-medium (582M) |
|---|---|---|
| Weights fp16 | ~0.6 GB | ~1.2 GB |
| Attention di seq 1024 | ±3x biaya seq 512 | ±3x biaya seq 512 |

Default `--max-input/--max-target` di **512/256 byte**: dari sampling 537K baris
corpus, 99.1% input < 512 char dan 99.9% < 1024 — jadi budget 512/256 kehilangan
<1% data sambil memangkas biaya attention ±4x dan aktivasi ±2x vs 1024/512.

## VRAM (RTX 3070 Ti 8GB, byt5-medium, bf16 + LoRA + grad checkpointing)

weights ~1.2 GB + LoRA/grads/optimizer ~0.4 GB + CUDA ctx ~0.5 GB → sisa ~5.9 GB
untuk aktivasi. Dengan checkpointing, batch 8 @ 512/256 ≈ 1.6 GB — aman. Tanpa
checkpointing, batch 4 @ 1024/512 butuh ~10 GB → OOM. Kombinasi gak masuk akal
yang bikin OOM: `--max-input 1024` + lupa gradient checkpointing (default kita
selalu on).

Saran: mulai `--model google/byt5-small --batch 8` buat memvalidasi pipeline,
baru medium `--batch 4 --accum 4` (aktif ~1.3 GB, aman). Kalau mau maksain
1024/512 di 8 GB: batch 1–2 + checkpointing, tapi lemot — lebih baik jangan.

## Notes

- Verified identical encoding with HF `ByT5Tokenizer` — see `src/check_tokenizer.py`.
- **FlashAttention-2 gak berlaku buat model ini** — bukan soal GPU (T4 sm_75 memang gak support, tapi itu moot): arch T5/ByT5 di transformers gak punya integrasi FA2 → `ValueError: T5ForConditionalGeneration does not support Flash Attention 2 yet`. SDPA = implementasi tercepat yang tersedia, jadi default. `check_env.py` cek + catat ini.
- **Gradient checkpointing + LoRA**: `finetune.py` eksplisit set `gradient_checkpointing_kwargs={"use_reentrant": False}` — WAJIB di transformers < 4.49 (termasuk pin 4.46.3) karena default di sana `use_reentrant=True` yang bikin crash `element 0 of tensors does not require grad` (embedding dibekukan LoRA, tidak ada grad yang mengalir). transformers ≥ 4.49 sudah default `False`, kwargs ini jadi no-op yang aman. `check_env.py` membuktikan grad mengalir lewat smoke test backward di GPU.
- `TokenParquetDataset` nyimpen byte flat int32 + offsets — 1.43M rows ~5 GB RAM (bukan list-of-lists puluhan GB). Taruh dataset di disk internal (NVMe), jangan HDD/USB eksternal — loading dari USB bikin I/O jadi bottleneck.
