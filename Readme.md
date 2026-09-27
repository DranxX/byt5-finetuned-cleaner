# ByT5-Finetuned Cleaner

LoRA fine-tune toolkit for **ByT5** text cleaning: dirty text in, clean text out.

- **Dataset:** [DranxX/corpus-cleaning-v1](https://huggingface.co/datasets/DranxX/corpus-cleaning-v1) — 1.43M raw→clean pairs (id/en/zh)
- **Base model:** [google/byt5-medium](https://huggingface.co/google/byt5-medium) (or `byt5-small` for quick tests)

## Install

```bash
pip install -r requirements.txt
pip uninstall -y torchao   # Kaggle/older envs: PEFT 0.19 breaks on torchao < 0.16
```

## Usage

```bash
# optional: pretokenize once (byte-level, fast)
python src/pretokenize.py --data-dir <parquet-folder> --out data_tok

# fine-tune
python src/finetune.py --data-dir data_tok --model google/byt5-medium --out output/m1

# RTX 3070 Ti (8GB): bf16 + smaller batch
python src/finetune.py --data-dir data_tok --model google/byt5-medium --out output/m1 --bf16 --batch 4 --accum 4

# inference
python src/inference.py --adapter output/m1/final --text "dirty text here"
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
