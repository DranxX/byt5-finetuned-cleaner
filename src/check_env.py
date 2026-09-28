"""
Environment doctor — cek kesiapan mesin sebelum training.

Dicek:
  1. Python + OS
  2. PyTorch: build CPU vs CUDA, cuDNN, CUDA runtime smoke test
  3. GPU: nama, compute capability (bf16?), VRAM, driver NVIDIA
  4. Library: versi terinstall vs pin di requirements.txt
  5. Sisa disk (model + checkpoint butuh ~25 GB)

Usage:
  python src/check_env.py

Exit code 0 = siap training, 1 = ada masalah (baca output).
Pakai exit code ini buat gate agent/CI.
"""
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

problems = []
warnings = []
cuda_ok = False
suggest = ""


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
    print("byt5-finetuned-cleaner -- environment doctor")
    check_python()
    check_torch_and_gpu()
    check_libs()
    check_disk()

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
