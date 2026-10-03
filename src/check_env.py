import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIGS

LIBS = ["transformers", "peft", "accelerate", "datasets", "sentencepiece",
        "tokenizers", "pyarrow", "numpy", "bitsandbytes"]

EXPECTED_TOTAL = 1_431_369
DEEP_BATCH_SIZE = 4096
SUPPORTED_LANGUAGES = {"id", "en", "zh"}
SHARD_RE = re.compile(r"^train-(\d+)-of-(\d+)\.parquet$")

problems = []
warnings = []
cuda_ok = False
suggest = ""
gpu_caps = []


def head(title):
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)))


def ok(msg):   print(f"  [OK  ] {msg}")
def bad(msg):  print(f"  [FAIL] {msg}")
def warn(msg): print(f"  [WARN] {msg}")


def pip_version(dist):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def read_pins():
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
        bad(f"Python {v} terlalu tua — butuh >= 3.9")
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

    if torch.version.cuda is None:
        bad(f"torch {torch.__version__} = build CPU-ONLY <-- biang kerok OOM-selalu & lemot")
        problems.append("torch build CPU-only")
        print("""       Di Windows, `pip install torch` dari PyPI = CPU build.
       Fix (dengan python/venv yang SAMA dengan yang dipakai training):
         python -m pip uninstall -y torch torchvision torchaudio
         python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
       Lalu jalankan ulang script ini.""")
        return

    ok(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        bad("torch build CUDA tapi torch.cuda.is_available() = False")
        problems.append("CUDA tidak tersedia (driver/WSL?)")
        print("       cek: `nvidia-smi` — kalau gagal, driver NVIDIA belum terpasang/bukan WSL passthrough")
        return

    ok(f"CUDA available — {torch.cuda.device_count()} GPU")
    try:
        x = torch.ones(64, device="cuda")
        (x @ x).sum().item()
        ok("CUDA runtime smoke test lulus")
    except Exception as e:
        bad(f"CUDA runtime error: {str(e).splitlines()[0]}")
        problems.append("CUDA runtime error")
        return

    for i in range(torch.cuda.device_count()):
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
        if not bf16:
            warn(f"GPU {i}: gak support bf16 — JANGAN pakai fp16 utk umt5/T5-family (overflow); "
                 f"mulai dari --precision fp32 --batch 1 --accum 16")
        if total < 8:
            warn(f"GPU {i}: VRAM < 8 GB — pakai batch kecil")
    bf16_flag = " --bf16" if (gpu_caps and min(gpu_caps) >= (8, 0)) else ""
    suggest = f"--model-family umt5-base --batch 1 --accum 16{bf16_flag} --data-dir dataset --out models/umt5-full"
    cuda_ok = True


def check_libs():
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
            bad(f"{name} {v}{note} — versi beda dgn environment yang diuji")
            problems.append(f"{name} {v} != pin {pin}")
        else:
            ok(f"{name} {v}".rstrip())

    if cuda_ok:
        try:
            import bitsandbytes.optim as bnb_optim
            if hasattr(bnb_optim, "PagedAdamW8bit"):
                ok("bitsandbytes: PagedAdamW8bit tersedia")
        except Exception as e:
            bad(f"bitsandbytes gagal: {str(e).splitlines()[0]}")
            print("         alternatif: training dgn --optim adamw_torch (lihat README)")


def check_flash_attn():
    head("Attention configuration")
    try:
        __import__("flash_attn")
        warn("flash-attn terinstall tapi TIDAK akan dipakai:")
        print("         UMT5 pada pin 4.55.4 memakai eager; T5Gemma 2 dikonfigurasi SDPA.")
    except ImportError:
        ok("flash-attn gak terinstall; tidak dibutuhkan konfigurasi ini")


def check_lora_grad_checkpointing():
    global cuda_ok
    if not cuda_ok:
        return
    head("LoRA + gradient checkpointing smoke test")
    model = None
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
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        out = model(input_ids=torch.randint(3, 250, (2, 16), device="cuda"),
                    labels=torch.randint(3, 250, (2, 8), device="cuda"))
        out.loss.backward()
        n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.parameters() if p.requires_grad)
        if n_grad == 0:
            bad("backward lulus tapi gak ada grad mengalir")
            problems.append("LoRA+grad-checkpoint: grad kosong")
        else:
            ok(f"forward-backward OK, {n_grad} trainable param dapat grad")
    except Exception as e:
        bad(f"crash: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        problems.append("LoRA+grad-checkpoint crash")
        if "does not require grad" in str(e):
            print("         use_reentrant=True kena — finetune.py udah set False;")
            print("         berarti transformers lu terlalu tua, cek requirements.txt")
    finally:
        try:
            del model
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def check_models(model_family="umt5-base"):
    head("Model HF (config arsitektur)")
    from huggingface_hub import hf_hub_download
    import transformers
    for key, c in CONFIGS.items():
        if key != model_family:
            continue
        if c.model_cls != "auto" and not hasattr(transformers, c.model_cls):
            bad(f"{key}: transformers {transformers.__version__} belum memiliki {c.model_cls}")
            problems.append(f"{key}: arsitektur tidak tersedia pada dependency terinstall")
            continue
        try:
            hf_hub_download(c.repo, "config.json")
            ok(f"{key}: {c.repo} bisa diakses")
        except Exception as e:
            msg = str(e).splitlines()[0][:100]
            if "401" in msg or "gated" in msg.lower() or "Access" in msg:
                bad(f"{key}: {c.repo} GATED — wajib `huggingface-cli login` + accept terms")
                problems.append(f"{key}: gated repo butuh token")
            else:
                bad(f"{key}: {msg}")
                problems.append(f"{key}: gagal resolve config")


def check_dataset(data_dir=None, deep=False):
    head("Dataset shards")
    if data_dir:
        cands = [data_dir]
    else:
        cands = [os.path.join(ROOT, c) for c in ("dataset/tok", "dataset")]
        cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        print("  [INFO] belum ada dataset/ — download/pretokenize dulu (lihat README)")
        return

    for d in cands:
        _check_one_dir(d, deep)


def _check_one_dir(d, deep):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(d, "**", "train-*.parquet"), recursive=True))
    tmps = glob.glob(os.path.join(d, "**", "tmp-*.parquet"), recursive=True)
    if tmps:
        bad(f"{d}: ada {len(tmps)} tmp-*.parquet — pretokenize ke-interupted")
        problems.append("pretokenize ke-interupted")
        return
    if not files:
        other = [f for f in glob.glob(os.path.join(d, "*.parquet"))
                 if not SHARD_RE.match(os.path.basename(f))]
        if other:
            warn(f"{d}: {len(other)} parquet gak match pola train-*-of-* (diabaikan finetune):")
            for f in other[:5]:
                print(f"           - {os.path.basename(f)}")
            problems.append("nama shard gak sesuai pola")
        else:
            print(f"  [INFO] {d}: kosong")
        return

    idxs, of_vals = [], set()
    for f in files:
        m = SHARD_RE.match(os.path.basename(f))
        if m is None:
            bad("nama shard tidak valid")
            problems.append("nama shard tidak valid")
            return
        idxs.append(int(m.group(1)))
        of_vals.add(int(m.group(2)))
    n = len(files)
    missing = sorted(set(range(max(idxs) + 1)) - set(idxs))
    if len(idxs) != len(set(idxs)) or missing or of_vals != {n}:
        bad(f"{d}: penamaan shard gak konsisten")
        problems.append("shard bolong / penamaan kacau")
        if missing:
            print(f"         shard hilang: {missing}")
        if of_vals != {n}:
            print(f"         'of-M': {sorted(of_vals)} vs faktanya {n} file")
        return
    ok(f"{d}: {n} shard, urutan 0..{n - 1} lengkap")

    kinds, rows_per_file = set(), []
    for f in files:
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:
            bad(f"{os.path.basename(f)}: parquet rusak — {e}")
            problems.append("parquet corrupt")
            return
        cols = set(pf.schema_arrow.names)
        if {"input_ids", "labels"} <= cols:
            kinds.add("tok")
        elif {"lang", "raw", "clean"} <= cols:
            kinds.add("raw")
        else:
            bad(f"{os.path.basename(f)}: kolom aneh — {sorted(cols)}")
            problems.append("schema gak dikenal")
            return
        rows_per_file.append(pf.metadata.num_rows)
    if len(kinds) > 1:
        bad(f"{d}: schema campur antar shard — {kinds}")
        problems.append("schema antar-shard beda")
        return
    kind = kinds.pop()
    total = sum(rows_per_file)
    ok(f"schema: {kind} ({', '.join(sorted(pq.ParquetFile(files[0]).schema_arrow.names))})")
    ok(f"rows: {total:,} (per shard {min(rows_per_file):,}..{max(rows_per_file):,})")
    if kind == "raw" and total != EXPECTED_TOTAL:
        warn(f"total {total:,} != corpus utuh ({EXPECTED_TOTAL:,}) — download belum lengkap?")
        warnings.append("dataset gak utuh")
    elif kind == "tok":
        print(f"         (sumber {EXPECTED_TOTAL:,}; selisih = rows ke-skip pretokenize)")

    if not deep:
        print("         (pakai --deep buat decode isi + stat panjang sekuens)")
        return

    import pyarrow.compute as pc
    if kind == "raw":
        for filename in files:
            parquet = pq.ParquetFile(filename)
            for batch in parquet.iter_batches(batch_size=DEEP_BATCH_SIZE, columns=["lang", "raw", "clean"]):
                columns = batch.to_pydict()
                for lang, raw, clean in zip(columns["lang"], columns["raw"], columns["clean"]):
                    if (not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES or
                            not isinstance(raw, str) or not isinstance(clean, str)):
                        bad("lang/raw/clean mengandung nilai yang tidak valid")
                        problems.append("nilai dataset mentah tidak valid")
                        return
        ok("decode dataset mentah dan validasi lang/raw/clean lulus")
        return
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
                mnmx = pc.min_max(lens)
                stat[pmin] = min(stat[pmin], mnmx["min"].as_py())
                stat[pmax] = max(stat[pmax], mnmx["max"].as_py())
                stat[pempty] += pc.sum(pc.equal(lens, 0)).as_py() or 0
        print(f"  {os.path.basename(f)} decoded")
    bad_rows = stat["in_empty"] + stat["lab_empty"]
    if bad_rows:
        bad(f"{bad_rows:,} row sekuens kosong")
        problems.append("ada row kosong")
    else:
        ok("gak ada row sekuens kosong")
    ok(f"panjang input {stat['in_min']}..{stat['in_max']} | labels {stat['lab_min']}..{stat['lab_max']}")


def check_disk():
    head("Disk")
    try:
        free = shutil.disk_usage(ROOT).free / 2**30
    except OSError as e:
        warn(f"gak bisa cek disk: {e}")
        return
    if free < 30:
        warn(f"sisa {free:.1f} GB — model + checkpoints butuh ~30 GB")
        warnings.append(f"disk tinggal {free:.1f} GB")
    else:
        ok(f"sisa {free:.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="skip cek akses HF")
    ap.add_argument("--model-family", choices=sorted(CONFIGS), default="umt5-base")
    args = ap.parse_args()

    print("corpus-cleaner — environment doctor")
    print("WAJIB exit 0 sebelum finetune/inference. (AGENTS.md: gate ini buat semua agent)")
    check_python()
    check_torch_and_gpu()
    check_libs()
    check_flash_attn()
    check_lora_grad_checkpointing()
    if not args.no_model:
        check_models(args.model_family)
    check_disk()
    check_dataset(args.data_dir, args.deep)

    head("VERDICT")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        print("\n  Perbaiki dulu, jalankan lagi: python src/check_env.py")
        sys.exit(1)
    for w in warnings:
        print(f"  WARN: {w}")
    print("\n  SIAP. Contoh command training:")
    print(f"    python src/finetune.py {suggest}")
    sys.exit(0)


if __name__ == "__main__":
    main()
