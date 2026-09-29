# corpus-finetuned

LoRA fine-tune toolkit untuk **text cleaning / denoising**: teks kotor masuk,
teks bersih keluar. Meneruskan project sebelumnya `byt5-finetuned` — arsitektur
ByT5 di-drop karena byte-level (1 byte = 1 token) selalu OOM di RTX 3070 Ti 8 GB
untuk korpus ini, diganti dua arsitektur subword encoder-decoder.

- **Dataset:** [DranxX/corpus-cleaning-v1](https://huggingface.co/datasets/DranxX/corpus-cleaning-v1) — 1,431,369 pair raw→clean (id 73.6% / en 20.5% / zh 5.8%)
- **Config arsitektur (lihat `src/config.py`):**
  1. `mt5-small` → `google/mt5-small` — 300M, SDPA, zero-dependency, **default**
  2. `t5gemma-270m` → `google/t5gemma-2-270m-270m` — ~370M aktif, sliding window 4096, **gated repo** (perlu login HF + accept Gemma terms)
  3. `byt5-medium` → legacy baseline saja (byte-level, OOM-prone, jangan dipakai training serius)

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
# CONFIG 1 — mt5-small (default, tercepat di 8 GB)
python src/finetune.py --model-family mt5-small --data-dir dataset \
  --out models/mt5s1 --bf16 --batch 16 --accum 1

# CONFIG 2 — t5gemma-270m (butuh login HF)
python src/finetune.py --model-family t5gemma-270m --data-dir dataset \
  --out models/t5g1 --bf16 --batch 8 --accum 2

# dataset dari HF langsung (on-the-fly tokenize)
python src/finetune.py --model-family mt5-small \
  --hf-dataset DranxX/corpus-cleaning-v1 --out models/mt5s1 --bf16
```

## Inference

```bash
python src/inference.py --model-family mt5-small --adapter models/mt5s1/final \
  --lang id --text "No. 24; Diperbarui Maret 2011  \nKlik di sini..."
```

Format input training: `<{lang}> {raw}` — prefix bahasa wajib ada di inference juga.

## Directory layout

```
corpus-finetuned/
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

## VRAM (RTX 3070 Ti 8 GB)

| model | weights bf16 | LoRA+grads | aktivasi (ckpt, batch, seq 512/256) | total |
|---|---|---|---|---|
| mt5-small | 0.6 GB | ~0.1 GB | ~0.5 GB @ batch 16 | **~1.7 GB** ✅ |
| t5gemma-270m | 0.75 GB | ~0.25 GB | ~0.8 GB @ batch 8 | **~2.5 GB** ✅ |
| byt5-medium (legacy) | 1.2 GB | ~0.3 GB | seq byte-level 4× lebih panjang → OOM-prone | ⚠️ |

## Notes penting

- **FA2 gak berlaku** untuk kedua arch (`_supports_flash_attn=False` verified).
  SDPA = default & tercepat yang tersedia. `check_env.py` yang menegaskan ini.
- **`gradient_checkpointing_kwargs={"use_reentrant": False}`** WAJIB di
  transformers < 4.49 — default di sana `use_reentrant=True` yang crash
  `element 0 of tensors does not require grad` saat LoRA. Sudah dipasang di
  `finetune.py`.
- **Gated t5gemma**: snapshot lokal tersimpan di `temp/runtime/` — bisa pakai
  `--base temp/runtime` buat offline.
- Korpus panjang-skew: id median 234 char tapi en p95 17K char. Budget token
  512/256 menangkap mayoritas; teks lebih panjang ke-truncate (chunking bisa
  jadi pengembangan lanjutan).
- Statistik lengkap dataset: lihat `experiments/merged/data.md`.
