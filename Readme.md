# corpus-cleaner

**v0.1** — LoRA fine-tune toolkit untuk **text cleaning / denoising**: teks
kotor masuk, teks bersih keluar. Dua arsitektur subword encoder-decoder
(ByT5 byte-level di-drop — 1 byte = 1 token selalu OOM di RTX 3070 Ti 8 GB
utk korpus ini).

**Status v0.1: pipeline teruji end-to-end** (pilot T4 Colab: 178K rows,
val loss 2.25; validasi ulang 100K rows di WSL2/Kaggle oleh owner: loss 2.9 /
val 2.8). Full training 1.43M rows belum dijalankan — jadwal & hasil menyusul.

- **Dataset:** [DranxX/corpus-cleaning-v1](https://huggingface.co/datasets/DranxX/corpus-cleaning-v1) — 1,431,369 pair raw→clean (id 73.6% / en 20.5% / zh 5.8%)
- **Config arsitektur (lihat `src/config.py`):**
  1. `umt5-base` → `google/umt5-base` — **580M, pipeline utama**. Pretraining lebih baik dari mT5 (EMA/scalable attention), vocab 256K, tokenizer paling padat utk id
  2. `t5gemma-270m` → `google/t5gemma-2-270m-270m` — ~370M aktif, sliding window 4096, **gated repo** (dipakai lain waktu)
  3. `byt5-medium` → legacy baseline (byte-level, OOM-prone — jangan dipakai training serius)

## ATURAN #1: checkup dulu, no exceptions

```bash
python src/check_env.py
```

**Script apapun (finetune/inference) TIDAK BOLEH dijalankan sebelum
`check_env.py` exit 0.** Ini berlaku juga untuk agent AI. Script ini mengecek:

| # | Cek | Kenapa |
|---|---|---|
| 1 | Python + OS | versi minimal |
| 2 | torch build CPU vs CUDA | Windows PyPI torch = CPU-only; biang kerok "training nyasar CPU" & OOM palsu |
| 3 | GPU + VRAM + bf16 | 3070 Ti 8 GB target; bf16 wajib di Ampere |
| 4 | Library vs pin | versi lain sering rusak (PEFT/transformers mismatch) |
| 5 | flash-attn | harus **tidak** dipakai — kedua arch gak support FA2 (verified) |
| 6 | LoRA + grad checkpointing smoke test | forward+backward beneran di GPU; deteksi bug `use_reentrant` |
| 7 | Akses repo HF model | deteksi gated repo tanpa token (t5gemma) |
| 8 | Dataset shards | lengkap/berurutan/schema/total rows (+`--deep` decode) |

Tambahan: `--deep` buat decode semua shard, `--no-model` kalau offline, `--data-dir` folder parquet kustom.

## Install (urutan WAJIB, terutama Windows)

```bash
# 1. torch dulu — jangan dari PyPI kalau Windows (CPU-only!)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. requirements (torch di-skip otomatis di Windows via marker)
python -m pip install -r requirements.txt

# 3. t5gemma gated: login akun HF yang udah granted + accept Gemma terms
huggingface-cli login
# akun: DranxX (sudah granted)

# 4. WAJIB: cek environment
python src/check_env.py
```

Kalau `bitsandbytes` gagal di Windows: training masih bisa jalan pakai
`--optim adamw_torch` (sedikit lebih boros VRAM, lihat tabel VRAM).

## Dataset

```bash
# download dataset utuh (1.43M rows) — ambil folder parquet ke dataset/
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("DranxX/corpus-cleaning-v1", repo_type="dataset",
                  local_dir="dataset", allow_patterns=["*.parquet"])
PY

# cek shards (jumlah/urutan/schema/rows)
python src/check_env.py
```

## Fine-tune

```bash
# CONFIG utama — umt5-base (580M)
python src/finetune.py --data-dir dataset \
  --out models/umt5-full --bf16

# opsional — t5gemma-270m (butuh login HF, lain waktu)
python src/finetune.py --model-family t5gemma-270m --data-dir dataset \
  --out models/t5g-full --bf16

# dataset dari HF langsung (on-the-fly tokenize)
python src/finetune.py --model-family mt5-small \
  --hf-dataset DranxX/corpus-cleaning-v1 --out models/mt5s1 --bf16
```

## Inference

```bash
python src/inference.py --adapter models/umt5-full/final \
  --lang id --text "No. 24; Diperbarui Maret 2011  \nKlik di sini..."
```

Format input training: `<{lang}> {raw}` — prefix bahasa wajib ada di inference juga.

## Directory layout

```
corpus-cleaner/
├── AGENTS.md             ← PANDUAN AGENT (baca dulu sebelum ngapa-ngapain)
├── Readme.md             ← file ini
├── requirements.txt      ← pin yang diuji; urutan install di atas
├── src/
│   ├── config.py         ← 2 config arsitektur resmi (mt5-small, t5gemma-270m) + legacy
│   ├── check_env.py      ← WAJIB JALAN DULU (gate exit 0)
│   ├── finetune.py       ← LoRA training
│   └── inference.py      ← load base+adapter, generate
├── dataset/              ← parquet dari HF (di-gitignore)
├── models/               ← checkpoint + adapter (di-gitignore)
└── temp/runtime/         ← snapshot lokal model gated t5gemma (di-gitignore)
```

## VRAM (RTX 3070 Ti 8 GB, budget seq 4096/4096)

| model | weights bf16 | LoRA+grads | aktivasi (ckpt, batch 8, seq 4096 long-tail) | total |
|---|---|---|---|---|
| mt5-small | 0.6 GB | ~0.1 GB | ~4.5 GB worst-case batch | **~5.2 GB** ✅ |
| t5gemma-270m | 0.75 GB | ~0.25 GB | ~4.8 GB worst-case batch | **~5.8 GB** ✅ |
| byt5-medium (legacy) | 1.2 GB | ~0.3 GB | byte-level 4× panjang → OOM | ⚠️ |

group_by_length bikin mayoritas batch jalan di seq pendek (median target id
~50-90 token) → throughput harian jauh di atas worst-case. Batch 16 HANYA
kalau budget diturunin (mis. `--max-input 1024`).

## Notes penting

- **FA2 gak berlaku** untuk kedua arch (`_supports_flash_attn=False` verified).
  SDPA = default & tercepat yang tersedia. `check_env.py` yang menegaskan ini.
- **`gradient_checkpointing_kwargs={"use_reentrant": False}`** WAJIB di
  transformers < 4.49 — default di sana `use_reentrant=True` yang crash
  `element 0 of tensors does not require grad` saat LoRA. Sudah dipasang di
  `finetune.py`.
- **Gated t5gemma**: snapshot lokal tersimpan di `temp/runtime/` — bisa pakai
  `--base temp/runtime` buat offline.
- Korpus panjang-skew: budget 4096/4096 menangkap ~99% rows utuh (input:
  id p99 185 tok, en median 1056 tok; target: id p99.9 174 tok, en p95 3.4K
  tok). Sisanya (en ekstrem >4K tok) ke-truncate — chunking per-paragraf
  jadi pengembangan lanjutan kalau dibutuhin.
- Statistik lengkap dataset: lihat `experiments/merged/data.md`.
