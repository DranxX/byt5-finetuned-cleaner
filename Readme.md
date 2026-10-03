# corpus-cleaner

LoRA fine-tune untuk pasangan raw → clean. Model utama yang aktif adalah `google/umt5-base`; pilot lama memakai `mt5-small`. Hasil pilot mT5 tidak membuktikan kualitas, kebutuhan VRAM, atau throughput UMT5.

## Precision dan attention

| Model/GPU | Pilihan yang tersedia | Catatan |
|---|---|---|
| UMT5 pada T4 | FP32 | T4 tidak memiliki BF16 native; FP16 tidak dipaksakan karena risiko overflow forward |
| UMT5 pada RTX 30xx | BF16 | `--precision auto` memilih BF16 native dan memuat base weights BF16 |
| T5Gemma 2 pada T4 | FP32 sebagai baseline numerik | Pin dependency saat ini belum mendukung arsitektur T5Gemma 2 |
| T5Gemma 2 pada RTX 30xx | BF16 setelah environment kompatibel disetujui | Bukan model T5Gemma generasi pertama yang tersedia pada 4.55.4 |

UMT5 pada Transformers 4.55.4 memakai attention eager. Meminta SDPA pada versi tersebut tidak menghasilkan kernel fused yang lebih hemat: model tidak mendukungnya. Gradient checkpointing non-reentrant tetap aktif.

T5Gemma 2 tersedia pada Transformers 5.0.0, sedangkan project ini masih mematok 4.55.4. `config.py` sekarang menyebut class yang benar dan script menolak model yang tidak tersedia sebelum tokenisasi. Dependency tidak dinaikkan otomatis; migrasi Trainer/PEFT/Accelerate perlu disetujui dan diverifikasi sebagai satu environment.

Referensi: [source UMT5 4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/umt5/modeling_umt5.py), [T5Gemma 2 pada Transformers 5.0.0](https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/t5gemma2), [NVIDIA Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

## Environment gate

Gunakan environment Python yang sama untuk instalasi, check, dan training. Di Windows, simpan venv, model, dataset, dan cache pada drive D sesuai panduan pemilik. Jangan menimpa torch runtime yang sudah bekerja.

```bash
python src/check_env.py --model-family umt5-base --data-dir dataset
```

Training/inference hanya dijalankan setelah check exit 0. Check memeriksa build CUDA, versi pin, gradient checkpointing, akses model yang dipilih, shard rekursif, schema, dan disk. `--deep` memvalidasi dataset mentah maupun dataset tokenized; training script menerima schema mentah `lang/raw/clean`.

`--no-model` melewati pemeriksaan akses Hub untuk environment offline. Ini tidak melewati gate CUDA atau versi library.

## Training UMT5

Setelah environment gate lulus, gunakan precision sesuai GPU:

```bash
python src/finetune.py --data-dir dataset --out models/umt5-full --precision auto
```

Pada T4, mulai pengukuran dengan batch kecil:

```bash
python src/finetune.py --data-dir dataset --out models/umt5-t4 --precision fp32 --batch 1 --accum 16
```

Default script: input 4.096, target 512, LoRA r32/alpha64, learning rate 3e-5, batch 2, accumulation 8, dua epoch. `--bf16` tetap tersedia; jangan gabungkan dengan `--precision fp32`.

- Tokenisasi dilakukan batched dan disimpan pada Arrow cache. Identitas cache mencakup dataset, vocabulary/tokenizer, special tokens, budget, dan versi preprocessing.
- Trainer menerima dataset Arrow asli beserta kolom panjang, sehingga grouping tidak memindai ulang wrapper per row.
- Padding disejajarkan ke kelipatan delapan; padding labels memakai -100 dan tidak masuk loss.
- Satu GPU menjadi default untuk proses biasa. Checkpoint terakhir dipilih secara numerik melalui utilitas Trainer.
- Base weights dimuat dengan dtype yang dipilih. LoRA adapters dapat tetap FP32; itu tidak berarti compute diam-diam memakai FP32.
- `cleaning_config.json` pada adapter final menyimpan model-family dan budget encoder untuk inference.

Sequence budget adalah batas resource, bukan jaminan seluruh row muat. Target yang dipotong dapat mengajari cleaner membuang akhir dokumen; input yang dipotong dapat kehilangan bukti yang dibutuhkan target. Kebijakan skip versus truncation perlu dipilih sebelum menafsirkan fidelity hasil training. Jangan membuat chunk raw/clean dengan offset token yang sama tanpa memastikan alignment.

Gunakan folder run baru jika model, dataset, budget, rank, atau konfigurasi optimizer berubah. Resume checkpoint lama tidak otomatis memvalidasi semua perubahan tersebut.

## Inference

Setelah environment gate lulus dan adapter tersedia:

```bash
python src/inference.py --adapter models/umt5-full/final --lang id --text "teks sumber"
```

Default generation menggunakan min-new 0, satu beam, repetition penalty 1, dan no-repeat-ngram 0. Ini membolehkan output pendek serta repetisi sumber yang bermakna. Input di luar budget encoder ditolak; script tidak memotong dokumen diam-diam. Bahasa yang didukung tetap id/en/zh dan prefix bahasa mengikuti training.

Jika kualitas cleaner masih buruk, evaluasi kehilangan angka, hedge, negasi, struktur list/tabel, dan bagian akhir dokumen. Jangan menutupinya dengan minimum output panjang atau larangan repetisi global.

## Pengukuran resource

Jangan menganggap LoRA membuat attention atau vocabulary logits murah. UMT5 eager tetap memiliki biaya attention kuadratik; decoder logits tumbuh dengan batch × target length × vocabulary. Naikkan batch hanya setelah mengukur peak VRAM pada batch panjang yang benar-benar muncul dalam dataset.

Catat precision, jumlah row, truncation, it/s, peak VRAM, eval loss, dan pemeriksaan preservasi konten. Notebook dua shard adalah pilot Indonesian; hasilnya tidak mewakili seluruh distribusi en/zh.
