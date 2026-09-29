# AGENTS.md — panduan untuk AI agent

Ini project **corpus-cleaner**: LoRA fine-tune toolkit untuk
text cleaning (raw → clean) di dataset `DranxX/corpus-cleaning-v1` (1.43M rows,
id/en/zh). Hardware target: **RTX 3070 Ti 8 GB** milik pemilik akun, OS Windows
(bisa juga WSL2/Kaggle T4 — sudah teruji sendiri oleh owner di 100K rows: loss 2.9 / val 2.8).

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
6. Statistik dataset penting (ringkas): 1,431,369 rows — id 73.6% (median
   input 234 char), en 20.5% (median 4,129 char, p95 17K!), zh 5.8%. TARGET
   clean: id p99=149 tok / en p95=3407 tok / zh p95=639 tok (mt5) — makanya
   max_target default 2048 (96% rows utuh; > itu truncate, chunking nanti).
   Duplikat 0.03%. FA2 gak didukung arch mana pun di config. Detail lengkap ada di repo `experiments/merged/data.md`
   (bukan bagian repo ini — owner yang punya).

## LINGKUNGAN MESIN TRAINING (pemilik: Windows, drive D)

Pemilik pengen SEMUA artefak besar di **drive D**, bukan drive sistem (C:).
1. **Buat & pakai venv di drive D** (atau di dalam repo ini), contoh:
   ```powershell
   D:\Python\python -m venv D:\venvs\corpus-cleaner
   D:\venvs\corpus-cleaner\Scripts\python -m pip install ...
   ```
   Semua command python/pip nanti WAJIB lewat venv itu — JANGAN python global,
   JANGAN bikin venv baru di tempat lain (penyebab klasik "udah install CUDA
   kok tetep kebaca CPU" = beda python). Kalau repo di-clone ke D, venv di
   `<repo>\.venv` juga boleh.
2. **Bersihin cache lama SEBELUM training baru** (bekas eksperimen byt5 lama
   & dataset lama):
   ```powershell
   # cache HF (model & dataset lama) — bisa puluhan GB
   D:\venvs\corpus-cleaner\Scripts\huggingface-cli delete-cache
   # atau langsung:
   rmdir /s /q %USERPROFILE%\.cache\huggingface
   # cache pip
   pip cache purge
   # artefak lama di repo (kalau ada): models/, dataset/, temp/, data_tok/
   ```
   setelah bersih, verifikasi sisa disk >= 30 GB (check_env.py juga ngecek).
3. Dataset & model snapshot taruh di `dataset/` dan `temp/` dalam repo (di
   drive D), BUKAN `%USERPROFILE%\.cache` — supaya gak dobel di C:.
   `check_env.py --data-dir dataset` untuk validasi.

## Arsitektur model — pakai config.py, JANGAN hardcode

Dua config resmi (`src/config.py`):

| key | repo | note |
|---|---|---|
| `umt5-base` | `google/umt5-base` | **DEFAULT & PIPELINE UTAMA**. 580M, pretraining lebih baik (EMA/scalable attention, max_distance 128), vocab 256K +300 sentinel, tokenizer paling padat utk id (5.37 B/tok terukur). Weights bf16 1.2 GB — batch 8 aman di 8 GB. fp16 DILARANG (T5-family overflow) |
| `t5gemma-270m` | `google/t5gemma-2-270m-270m` | ~370M aktif, Gemma3-based enc-dec — **DIPAKAI LAIN WAKTU** (setelah umt5 selesai). **GATED** — wajib `huggingface-cli login` (akun DranxX sudah granted) |
mt5-small & byt5-medium SUDAH DIHAPUS dari config (get_config nolak dgn pesan
yang ngarahin ke jalur yang bener). Pilot sebelumnya (T4 + WSL2 validasi owner
100K rows: loss 2.9/val 2.8) jalan di mt5-small — semua hasilnya valid utk
umt5-base karena arch & hyperparam identik (T5-family, LoRA targets sama).

LoraConfig target modules beda per arch — diambil otomatis dari config.py.

## Struktur file

```
corpus-cleaner/
├── AGENTS.md            ← file ini
├── Readme.md            ← install, usage, tabel VRAM
├── requirements.txt     ← pin teruji + urutan install Windows (torch dulu!)
├── src/
│   ├── config.py        ← 2 config arsitektur (repo, LoRA targets, seq budget)
│   ├── check_env.py     ← GATE: env/GPU/library/dataset/model-access (exit 0/1)
│   ├── finetune.py      ← LoRA training (default umt5-base; t5gemma-270m opsional)
│   └── inference.py     ← load base+adapter, generate (prefix <lang> wajib)
├── dataset/             ← parquet dataset (download dari HF) [gitignored]
├── models/              ← checkpoint + adapter final [gitignored]
└── (temp/ optional)      ← snapshot model gated kalau mau offline [gitignored]

File YANG TIDAK ADA di repo dan gak perlu dibuat ulang: notebook pilot,
ipynb, screenshot — semuanya lokal-only, di-gitignore. Jangan commit.
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

# 3. dataset utuh (8 shard, 1.43M rows) — taruh di dataset/ dalam repo (drive D)
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("DranxX/corpus-cleaning-v1", repo_type="dataset",
                  local_dir="dataset", allow_patterns=["data/*.parquet"])
PY

# 4. FULL TRAINING di 3070 Ti — SEMUA DEFAULT udah hasil pilot, jangan diubah:
python src/finetune.py --data-dir dataset \
  --out models/umt5-full --bf16
#   default: lr 3e-5 | r16/alpha32 | warmup 500 | clip 1.0 | adamw_torch |
#            fp16 OFF | eval 1000 / save 2000 | val 1% capped 4000 rows |
#            max_input & max_target 4096 | batch 2 x accum 8 (eff 16)
#   KENAPA BATCH 2: attention T5-family gak pake fused kernel (relative position
#   bias dijumlah manual ke scores sebelum softmax) -> memory O(L^2) beneran.
#   batch 4 @ 4096 = 3GB utk SATU softmax output = OOM di T4 16GB (terbukti).
#   batch 2 @ 4096 bf16 di 8GB ~4GB peak = aman.
#   durasi est: 12-18 jam/epoch (long-tail lambat; group_by_length bikin
#   mayoritas batch pendek cepet).
#   Kalau loss nan di 1000 step pertama: jangan panik, baca output (nan
#   ditampilkan jujur) -> turunin --lora-r 8 --lora-alpha 16 & --lr 1e-5.
#   Lanjut epoch ekstra: python src/finetune.py ... (resume otomatis dari
#   checkpoint terakhir di folder --out yang sama).

python src/finetune.py --model-family t5gemma-270m --data-dir dataset \
  --out models/t5g-full --bf16 --batch 8 --accum 2

# 5. inference (default udah bawa mitigation pilot: rep-penalty 1.2, no-repeat 4,
#    beams 4, min-new 40, sentinel-strip)
python src/inference.py --adapter models/umt5-full/final \
  --lang id --text "teks kotor di sini"

# 6. WAJIB setelah training: jalankan smoke test 7 kasus (5 out-of-corpus
#    + 2 in-domain) — teks dijamin gak ada di corpus (verified substring+hash):
python src/eval_cases.py --adapter models/umt5-full/final
```

Urutan kerja agent: **check_env exit 0 -> dataset -> umt5-base full ->
`eval_cases.py` (5 ood + 2 in-domain; target: konten utuh, tanpa repetition,
boilerplate terbuang)**. t5gemma-270m = eksperimen LAIN WAKTU, bukan bagian
run ini.
Ekspektasi val loss: < 1.0 itu bagus; 2.25 = hasil pilot 1-shard (baseline).

Offline t5gemma: `--base temp/runtime` (snapshot lokal lengkap).

## Error umum → penanganan

| Gejala | Sebab | Fix |
|---|---|---|
| `torch.cuda.is_available() = False` | torch CPU build (Windows PyPI) | reinstall dari index cu124, lihat check_env output |
| `bitsandbytes` gagal import | torch CPU / versi lama | fix torch dulu; darurat: `--optim adamw_torch` |
| OOM pas training | batch kegedean / seq kepanjangan | `--batch 8 --accum 2` (mt5) atau 4/4 (t5gemma); checkpointing jangan dimatikan |
| loss `0.0` / `nan` di mT5 | **mT5 + fp16 = overflow (known issue, verified)** | jangan pakai fp16 utk mt5; GPU tanpa bf16 (T4) → fp32 penuh; Ampere → `--bf16` |
| `element 0 of tensors does not require grad` | transformers tua default `use_reentrant=True` | sudah di-fix di finetune.py; jangan hapus kwargs-nya |
| t5gemma 401/gated | belum login / belum accept terms | `huggingface-cli login` + buka repo di browser, klik acknowledge |
| shard bolong / tmp-*.parquet | download/pretokenize ke-interupted | hapus folder, download ulang |

## Hal yang SUDAH diuji (jangan diragukan lagi)

- `t5gemma-270m`: load dari `temp/runtime` OK, LoRA inject (252 params dapat
  grad), forward-backward OK, generate OK (transformers 4.55.x, CPU bf16).
- `mt5-small`: load OK, LoRA inject OK, forward-backward OK.
- `mt5-small`: pilot end-to-end di Colab T4 (1 shard, LoRA train, generate) —
  berhasil; catatan pilot ada di `notebook.md` (lokal, gak di-commit).
- check_env.py mendeteksi & memberi solusi utk semua kasus di tabel atas.
