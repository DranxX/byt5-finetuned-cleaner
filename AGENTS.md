# AGENTS.md — panduan untuk AI agent

Halo agent. Ini project **corpus-finetuned**: LoRA fine-tune toolkit untuk
text cleaning (raw → clean) di dataset `DranxX/corpus-cleaning-v1` (1.43M rows,
id/en/zh). Hardware target: **RTX 3070 Ti 8 GB** milik pemilik akun, OS Windows
(bisa juga WSL2/Kaggle T4).

Baca dokumen ini SEBELUM melakukan apa pun. Jangan improvisasi.

## ATURAN MUTLAK

1. **WAJIB jalankan `python src/check_env.py` dulu** sebelum menjalankan atau
   menyarankan `finetune.py` / `inference.py`. Exit code **0 = boleh lanjut,
   1 = baca output, perbaiki, ulangi**. Ini bukan formalitas — script itu
   mendeteksi: torch CPU-only (penyebab "training nyasar CPU" / OOM palsu),
   bitsandbytes rusak, shard dataset bolong, gated repo tanpa token, dan bug
   LoRA/gradient-checkpointing. Percayai hasilnya.
2. **JANGAN edit kode buat "mematikan GPU features"** (8-bit optimizer, bf16,
   gradient checkpointing, dll) ketika ada error GPU. Error GPU hampir selalu
   karena environment (torch salah build / driver / OOM karena batch kegedean),
   bukan kode. Perbaiki environment atau kurangi `--batch`. Edit kode = reject.
3. **JANGAN downgrade/pindah versi** di `requirements.txt` tanpa ijazah owner.
   Pin itu sudah diuji (lihat file-nya ada catatan). Transformer versi tua
   (< 4.49) butuh `gradient_checkpointing_kwargs={"use_reentrant": False}` —
   sudah dipasang di `finetune.py`, jangan dihapus.
4. **JANGAN pakai `--attn flash_attention_2`.** Kedua arsitektur di config.py
   TIDAK mendukung FA2 (verified dari source transformers). SDPA = default.
5. **Training di CPU itu jalur mati.** 1.43M rows di CPU = berminggu-minggu.
   Kalau GPU gak terdeteksi: fix torch (README § Install), bukan paksa CPU.
6. Kalau butuh konteks dataset (statistik panjang, komposisi, duplikat): baca
   `../merged/data.md`. Jangan scan dataset sendiri tanpa perlu.

## Arsitektur model — pakai config.py, JANGAN hardcode

Dua config resmi (`src/config.py`):

| key | repo | note |
|---|---|---|
| `mt5-small` | `google/mt5-small` | default. 300M, SDPA, zero-deps, tercepat di 8 GB |
| `t5gemma-270m` | `google/t5gemma-2-270m-270m` | ~370M aktif, Gemma3-based enc-dec. **GATED** — wajib `huggingface-cli login` (akun DranxX sudah granted) |
| `byt5-medium` | `google/byt5-medium` | **LEGACY** — byte-level, selalu OOM di 8 GB utk korpus ini. Hanya buat pembanding. JANGAN sarankan utk training baru |

LoraConfig target modules beda per arch — diambil otomatis dari config.py.

## Struktur file

```
corpus-finetuned/
├── AGENTS.md            ← file ini
├── Readme.md            ← install, usage, tabel VRAM
├── requirements.txt     ← pin teruji + urutan install Windows (torch dulu!)
├── src/
│   ├── config.py        ← 2 config arsitektur (repo, LoRA targets, seq budget)
│   ├── check_env.py     ← GATE: env/GPU/library/dataset/model-access (exit 0/1)
│   ├── finetune.py      ← LoRA training (--model-family mt5-small | t5gemma-270m)
│   └── inference.py     ← load base+adapter, generate (prefix <lang> wajib)
├── dataset/             ← parquet dataset (download dari HF) [gitignored]
├── models/              ← checkpoint + adapter final [gitignored]
└── temp/runtime/        ← snapshot lokal t5gemma-270m (sudah didownload) [gitignored]
```

## Cara menjalankan (urutan)

```bash
# 0. (sekali) install — Windows WAJIB torch CUDA dulu dari index pytorch, lihat Readme
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
huggingface-cli login        # utk t5gemma gated

# 1. WAJIB: checkup — harus exit 0
python src/check_env.py

# 2. dataset (kalau belum ada): snapshot_download ke dataset/ (lihat Readme)

# 3. training
python src/finetune.py --model-family mt5-small --data-dir dataset \
  --out models/mt5s1 --bf16 --batch 16 --accum 1
python src/finetune.py --model-family t5gemma-270m --data-dir dataset \
  --out models/t5g1 --bf16 --batch 8 --accum 2

# 4. inference
python src/inference.py --model-family mt5-small --adapter models/mt5s1/final \
  --lang id --text "teks kotor di sini"
```

Offline t5gemma: `--base temp/runtime` (snapshot lokal lengkap).

## Error umum → penanganan

| Gejala | Sebab | Fix |
|---|---|---|
| `torch.cuda.is_available() = False` | torch CPU build (Windows PyPI) | reinstall dari index cu124, lihat check_env output |
| `bitsandbytes` gagal import | torch CPU / versi lama | fix torch dulu; darurat: `--optim adamw_torch` |
| OOM pas training | batch kegedean / seq kepanjangan | `--batch 8 --accum 2` (mt5) atau 4/4 (t5gemma); checkpointing jangan dimatikan |
| `element 0 of tensors does not require grad` | transformers tua default `use_reentrant=True` | sudah di-fix di finetune.py; jangan hapus kwargs-nya |
| t5gemma 401/gated | belum login / belum accept terms | `huggingface-cli login` + buka repo di browser, klik acknowledge |
| shard bolong / tmp-*.parquet | download/pretokenize ke-interupted | hapus folder, download ulang |

## Hal yang SUDAH diuji (jangan diragukan lagi)

- `t5gemma-270m`: load dari `temp/runtime` OK, LoRA inject (252 params dapat
  grad), forward-backward OK, generate OK (transformers 4.55.x, CPU bf16).
- `mt5-small`: load OK, LoRA inject OK, forward-backward OK.
- check_env.py mendeteksi & memberi solusi utk semua kasus di tabel atas.
