# ByT5-Finetuned Cleaner

LoRA fine-tune toolkit for **ByT5** text cleaning: dirty text in, clean text out.

- **Dataset:** [DranxX/corpus-cleaning-v1](https://huggingface.co/datasets/DranxX/corpus-cleaning-v1) — 1.43M raw→clean pairs (id/en/zh)
- **Base model:** [google/byt5-medium](https://huggingface.co/google/byt5-medium) (or `byt5-small` for quick tests)

## Install

```bash
pip install -r requirements.txt
pip uninstall -y torchao   # Kaggle/older envs: PEFT 0.19 breaks on torchao < 0.16
```

> **Windows + GPU:** wheel `pip install torch` di Windows itu **CPU-only** —
> `torch.cuda.is_available()` pasti `False` walau GPU ada. Install build CUDA:
>
> ```bash
> pip uninstall -y torch torchvision torchaudio
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

## 0. Cek environment dulu

```bash
python src/check_env.py
```

Ngecek: Python/OS, build torch (CPU vs CUDA — penyebab paling umum training
nyasar ke CPU), GPU + VRAM + dukungan bf16, smoke test CUDA runtime, versi
library vs pin di `requirements.txt`, dan sisa disk. Exit code `0` = siap,
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
| `--attn` | `sdpa` | memory-efficient attention |

## Notes

- Verified identical encoding with HF `ByT5Tokenizer` — see `src/check_tokenizer.py`.
- FlashAttention-2 is not supported on T4/RTX 30xx; SDPA (default) is the fastest available option there.
