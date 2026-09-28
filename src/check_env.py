"""
Environment doctor — cek kesiapan mesin sebelum training.

Dicek:
  1. Python + OS
  2. PyTorch: build CPU vs CUDA, cuDNN, CUDA runtime smoke test
  3. GPU: nama, compute capability (bf16?), VRAM, driver NVIDIA
  4. Library: versi terinstall vs pin di requirements.txt
  5. Sisa disk (model + checkpoint butuh ~25 GB)

Usage:
  python src/check_env.py                     # cek env + dataset (kalau ada)
  python src/check_env.py --data-dir <dir>    # paksa cek folder parquet tertentu
  python src/check_env.py --deep              # decode semua shard (lambat, tapi teliti)

Exit code 0 = siap training, 1 = ada masalah (baca output).
Pakai exit code ini buat gate agent/CI.
"""
import glob
import os
import platform
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIBS = [
    "transformers", "peft", "accelerate", "datasets",
    "bitsandbytes", "sentencepiece", "tokenizers", "pyarrow", "numpy",
]

# total baris corpus-cleaning-v1 utuh (schema lang/raw/clean)
EXPECTED_TOTAL = 1_431_369

SHARD_RE = re.compile(r"^train-(\d+)-of-(\d+)\.parquet$")

problems = []
warnings = []
cuda_ok = False
suggest = ""
gpu_caps = []
flash_ok = False


def head(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def ok(msg):
    print(f"  [OK  ] {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def pip_version(dist):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def nvidia_smi(cli_args):
    try:
        r = subprocess.run(
            ["nvidia-smi", *cli_args], capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0:
            return r.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def read_pins():
    """Parse pin `pkg==ver` dari requirements.txt."""
    pins = {}
    try:
        with open(os.path.join(ROOT, "requirements.txt")) as f:
            for line in f:
                m = re.match(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)", line.strip())
                if m:
                    pins[m.group(1).lower()] = m.group(2)
    except OSError:
        pass
    return pins


def check_python():
    head("Python")
    v = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python {v} | {platform.system()} {platform.release()} | {platform.machine()}")
    if sys.version_info < (3, 9):
        bad(f"Python {v} terlalu tua -- butuh >= 3.9")
        problems.append("python terlalu tua")
    else:
        ok(f"Python {v}")


def check_torch_and_gpu():
    global cuda_ok, suggest
    head("PyTorch & GPU")
    try:
        import torch
    except Exception as e:
        bad(f"torch gak bisa diimport: {e}")
        problems.append("torch tidak terinstall")
        return

    cuda_build = torch.version.cuda
    if cuda_build is None:
        bad(f"torch {torch.__version__} = build CPU-ONLY <-- ini biang keroknya")
        problems.append("torch build CPU-only (CUDA gak ada)")
        print("""       torch.cuda.is_available() pasti False walaupun GPU ada.
       Fix -- install build CUDA (Windows: wheel PyPI itu CPU-only):
         pip uninstall -y torch torchvision torchaudio
         pip install torch --index-url https://download.pytorch.org/whl/cu124
       Lalu jalankan lagi script ini.""")
        return

    cudnn = torch.backends.cudnn.version()
    ok(f"torch {torch.__version__} | CUDA build {cuda_build} | cuDNN {cudnn}")

    if not torch.cuda.is_available():
        bad("torch build CUDA tapi torch.cuda.is_available() = False")
        problems.append("CUDA tidak tersedia (driver?)")
        print("       cek: jalankan `nvidia-smi` -- kalau gagal, driver NVIDIA belum terpasang")
        return

    n_gpu = torch.cuda.device_count()
    ok(f"CUDA available -- {n_gpu} GPU")

    # smoke test: beneran bisa eksekusi di CUDA (nangkep driver mismatch dll.)
    try:
        x = torch.ones(64, device="cuda")
        (x @ x).sum().item()
        ok("CUDA runtime smoke test lulus")
    except Exception as e:
        bad(f"CUDA runtime error: {str(e).splitlines()[0]}")
        problems.append("CUDA runtime error")

    # driver via nvidia-smi (kalau ada)
    driver_cuda = None
    plain = nvidia_smi([])
    if plain:
        m = re.search(r"CUDA Version\s*:\s*([\d.]+)", plain)
        if m:
            driver_cuda = m.group(1)
    drivers = nvidia_smi(["--query-gpu=driver_version", "--format=csv,noheader"])
    drivers = [d.strip() for d in drivers.splitlines()] if drivers else []

    for i in range(n_gpu):
        props = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        gpu_caps.append(cap)
        bf16 = cap >= (8, 0)
        total = props.total_memory / 2**30
        try:
            free = torch.cuda.mem_get_info(i)[0] / 2**30
            vram = f"{total:.1f} GB ({free:.1f} GB free)"
        except Exception:
            vram = f"{total:.1f} GB"

        ok(f"GPU {i}: {props.name} | sm_{cap[0]}{cap[1]} | VRAM {vram}")
        ok(f"GPU {i}: bf16 {'OK (Ampere+)' if bf16 else 'TIDAK -- pakai fp16'}")
        if i < len(drivers):
            drv = f", driver CUDA {driver_cuda}" if driver_cuda else ""
            print(f"         driver: {drivers[i]}{drv}")

        if total >= 10:
            suggest = "--batch 8 --accum 2" + (" --bf16" if bf16 else "")
        elif total >= 6:
            suggest = "--batch 4 --accum 4" + (" --bf16" if bf16 else "")
        else:
            suggest = "--batch 2 --accum 8" + (" --bf16" if bf16 else "")

    cuda_ok = True


def check_libs():
    global cuda_ok
    head("Library vs requirements.txt")
    pins = read_pins()
    for name in LIBS:
        v = pip_version(name)
        pin = pins.get(name)
        if v is None:
            bad(f"{name}: tidak terinstall" + (f" (pin {pin})" if pin else ""))
            problems.append(f"{name} tidak terinstall")
            continue
        note = f" (pin {pin})" if pin else ""
        if pin and v != pin:
            warn(f"{name} {v}{note} -- versi beda dgn pin")
            warnings.append(f"{name} {v} != pin {pin}")
        else:
            ok(f"{name} {v}{note}".rstrip())

    # bitsandbytes: wajib jalan buat paged_adamw_8bit (GPU only)
    try:
        import bitsandbytes.optim  # noqa: F401
        if cuda_ok and hasattr(bitsandbytes.optim, "PagedAdamW8bit"):
            ok("bitsandbytes: PagedAdamW8bit tersedia")
    except Exception as e:
        msg = str(e).splitlines()[0]
        if cuda_ok:
            bad(f"bitsandbytes gagal: {msg} -- butuh build CUDA torch + driver NVIDIA")
            problems.append("bitsandbytes rusak")
        # kalau torch-nya aja belum CUDA, error bnb udah tercakup di atas


def check_flash_attn():
    """flash-attn itu gak relevan buat model ini: T5/ByT5 gak support FA2.

    (transformers raise: 'T5ForConditionalGeneration does not support Flash
    Attention 2 yet'). Yang tersedia: SDPA (default, tercepat) dan eager.
    """
    head("FlashAttention (gak berlaku utk T5/ByT5)")
    try:
        import flash_attn

        flash_ok = True
        warn(f"flash-attn {getattr(flash_attn, '__version__', '?')} terinstall tapi GAK kepakai:")
        print("         arch T5/ByT5 gak support FA2 di transformers (ValueError kalau dipaksa).")
    except Exception:
        if not cuda_ok:
            print("  [INFO] flash-attn gak terinstall. Gak masalah -- T5/ByT5 emang gak support FA2.")
            return
        print("  [INFO] flash-attn gak terinstall. Gak masalah -- T5/ByT5 emang gak support FA2.")
    print("         Pilihan attention buat model ini: SDPA (default, tercepat) / eager.")
    suggest = suggest  # gak ada rekomendasi FA2


def check_dataset(data_dir=None, deep=False):
    head("Dataset shards")
    if data_dir:
        cands = [data_dir]
    else:
        cands = [os.path.join(ROOT, c) for c in ("dataset/tok", "dataset")]
        cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        print("  [INFO] belum ada dataset/ -- skip cek shard (pretokenize dulu)")
        return

    for d in cands:
        _check_one_dir(d, deep)


def _check_one_dir(d, deep):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(d, "train-*.parquet")))
    tmps = glob.glob(os.path.join(d, "tmp-*.parquet"))
    if tmps:
        bad(f"{d}: ada {len(tmps)} file tmp-*.parquet")
        problems.append("pretokenize ke-interupted (tmp belum rename)")
        print("         hapus folder tsb dan jalankan pretokenize.py ulang dari awal.")
        return
    if not files:
        other = [f for f in glob.glob(os.path.join(d, "*.parquet"))
                 if not SHARD_RE.match(os.path.basename(f))]
        if other:
            warn(f"{d}: {len(other)} parquet gak match pola train-*-of-*.parquet --")
            print("         finetune.py cuma baca train-*.parquet, sisanya DIABAIKAN diam-diam:")
            for f in other[:5]:
                print(f"           - {os.path.basename(f)}")
            problems.append("nama shard gak sesuai pola train-N-of-M")
        else:
            print(f"  [INFO] {d}: kosong")
        return

    # --- konsistensi penamaan: urutan 0..N-1 lengkap, N == angka 'of-M'
    idxs, of_vals = [], set()
    for f in files:
        m = SHARD_RE.match(os.path.basename(f))
        idxs.append(int(m.group(1)))
        of_vals.add(int(m.group(2)))
    n = len(files)
    dup = len(idxs) != len(set(idxs))
    missing = sorted(set(range(max(idxs) + 1)) - set(idxs))
    if dup or missing or of_vals != {n}:
        bad(f"{d}: penamaan shard gak konsisten")
        problems.append("shard bolong / penamaan kacau")
        if dup:
            print("         ada nomor duplikat")
        if missing:
            print(f"         shard hilang: {missing}")
        if of_vals != {n}:
            print(f"         'of-M' di nama file: {sorted(of_vals)}, faktanya {n} file")
        return
    ok(f"{d}: {n} shard, urutan 0..{n - 1} lengkap")

    # --- metadata + schema per shard (murah, gak decode isi)
    kinds, rows_per_file = set(), []
    for f in files:
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:
            bad(f"{os.path.basename(f)}: parquet rusak -- {e}")
            problems.append("parquet corrupt")
            return
        md = pf.metadata
        cols = set(pf.schema_arrow.names)
        if {"input_ids", "labels"} <= cols:
            kinds.add("tok")
        elif {"lang", "raw", "clean"} <= cols:
            kinds.add("raw")
        else:
            bad(f"{os.path.basename(f)}: kolom aneh -- {sorted(cols)}")
            problems.append("schema shard gak dikenal")
            return
        rows_per_file.append(md.num_rows)
    if len(kinds) > 1:
        bad(f"{d}: schema campur aduk antar shard -- {kinds}")
        problems.append("schema antar-shard beda")
        return
    kind = kinds.pop()
    total = sum(rows_per_file)
    ok(f"schema: {kind} ({', '.join(sorted(pq.ParquetFile(files[0]).schema_arrow.names))})")
    ok(f"rows: {total:,} (per shard {min(rows_per_file):,}..{max(rows_per_file):,})")

    if kind == "raw" and total != EXPECTED_TOTAL:
        warn(f"total {total:,} != corpus-cleaning-v1 utuh ({EXPECTED_TOTAL:,})")
        warnings.append("dataset gak utuh")
    if kind == "tok" and total > EXPECTED_TOTAL:
        warn(f"total {total:,} > sumber ({EXPECTED_TOTAL:,}) -- aneh")
        warnings.append("rows tok > sumber")
    elif kind == "tok":
        print(f"         (sumber {EXPECTED_TOTAL:,} -- selisih = raw > max-input yg di-skip pretokenize)")

    if not deep:
        print("         (pakai --deep buat decode isi + stat panjang sekuens)")
        return

    # --- deep: decode semua row group
    import pyarrow.compute as pc
    stat = {"in_min": 10**9, "in_max": 0, "lab_min": 10**9, "lab_max": 0,
            "in_empty": 0, "lab_empty": 0}
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg)
            for col, pmin, pmax, pempty in (
                ("input_ids", "in_min", "in_max", "in_empty"),
                ("labels", "lab_min", "lab_max", "lab_empty"),
            ):
                arr = t.column(col).combine_chunks()
                lens = pc.list_value_length(arr)
                mn, mx = pc.min_max(lens)["min"].as_py(), pc.min_max(lens)["max"].as_py()
                stat[pmin] = min(stat[pmin], mn)
                stat[pmax] = max(stat[pmax], mx)
                stat[pempty] += pc.sum(pc.equal(lens, 0)).as_py() or 0
        print(f"  {os.path.basename(f)} decoded")
    bad_rows = stat["in_empty"] + stat["lab_empty"]
    if bad_rows:
        bad(f"{bad_rows:,} row kosong (input {stat['in_empty']:,} / labels {stat['lab_empty']:,})")
        problems.append("ada row sekuens kosong")
    else:
        ok("gak ada row dgn sekuens kosong")
    ok(f"panjang input {stat['in_min']}..{stat['in_max']} byte | labels {stat['lab_min']}..{stat['lab_max']} byte")


def check_lora_grad_checkpointing():
    """Smoke test yang dulu bikin orang nambahin enable_input_require_grads().

    Crash "element 0 of tensors does not require grad" hanya terjadi di stack
    lama (torch < 2.1 / transformers < 4.35) yang default-nya
    use_reentrant=True. Stack pin kita (default use_reentrant=False) gak
    kena -- buktiin langsung pake mini-T5 (forward + backward + grad
    mengalir), bukan cuma teori.
    """
    global cuda_ok
    if not cuda_ok:
        return
    head("LoRA + gradient checkpointing smoke test")
    try:
        import torch
        from transformers import T5Config, T5ForConditionalGeneration
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = T5Config(vocab_size=384, d_model=64, d_ff=128, num_layers=2,
                       num_heads=4, d_kv=16, decoder_start_token_id=0)
        model = T5ForConditionalGeneration(cfg)
        model.config.use_cache = False
        lora = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
                          target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"])
        model = get_peft_model(model, lora)
        model.train()
        # persis jalur finetune.py: Trainer(gradient_checkpointing=True)
        # -> gradient_checkpointing_enable() tanpa kwargs -> default stack
        model.gradient_checkpointing_enable()
        model.to("cuda")
        # BACKWARD PASS beneran (bukan cuma forward): klaim "tanpa
        # enable_input_require_grads -> RuntimeError element 0" kalau bener
        # harus meledak persis di baris ini
        out = model(input_ids=torch.randint(3, 250, (2, 16), device="cuda"),
                    labels=torch.randint(3, 250, (2, 8), device="cuda"))
        out.loss.backward()
        n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.parameters() if p.requires_grad)
        if n_grad == 0:
            bad("forward-backward lalu tapi gak ada grad yang mengalir")
            problems.append("LoRA+grad-checkpoint: grad kosong")
            return
        ok(f"forward-backward OK, {n_grad} trainable param dapat grad")
        model.zero_grad(set_to_none=True)
    except Exception as e:
        bad(f"crash: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        problems.append("LoRA+grad-checkpoint crash")
        print("         kalau 'does not require grad': stack lama (use_reentrant=True) --")
        print("         update transformers, atau panggil model.enable_input_require_grads().")
    finally:
        # buang model mini dari VRAM biar gak makan tempat buat training
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass


def check_disk():
    head("Disk")
    try:
        free = shutil.disk_usage(ROOT).free / 2**30
    except OSError as e:
        warn(f"gak bisa cek disk: {e}")
        return
    if free < 25:
        warn(f"sisa {free:.1f} GB di {ROOT} -- model byt5-medium + checkpoint butuh ~25 GB")
        warnings.append(f"disk tinggal {free:.1f} GB")
    else:
        ok(f"sisa {free:.1f} GB di {ROOT}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="folder parquet buat dicek shard-nya")
    ap.add_argument("--deep", action="store_true", help="decode semua shard (lambat)")
    args = ap.parse_args()

    print("byt5-finetuned-cleaner -- environment doctor")
    check_python()
    check_torch_and_gpu()
    check_libs()
    check_flash_attn()
    check_lora_grad_checkpointing()
    check_disk()
    check_dataset(args.data_dir, args.deep)

    head("VERDICT")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        print("\n  Perbaiki dulu, lalu jalankan lagi: python src/check_env.py")
        sys.exit(1)

    for w in warnings:
        print(f"  WARN: {w}")
    if cuda_ok:
        print(f"\n  SIAP TRAINING. Saran flag: {suggest}")
    sys.exit(0)


if __name__ == "__main__":
    main()
