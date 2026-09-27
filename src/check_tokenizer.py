"""
Verifikasi tokenizer ByT5 terhadap corpus parquet.

ByT5 = byte-level statis: token_id = byte + 3 (specials 0..2, extra_id 259..383).

  1. Compare manual encoding vs ByT5Tokenizer HF (harus identik)
  2. Scan parquet shards, cek coverage byte corpus

Usage:
  python src/check_tokenizer.py --data-dir /path/ke/parquet-folder
"""
import argparse
import glob
import os
import sys

import pyarrow.parquet as pq

BYTE_OFFSET = 3


def compare_with_hf() -> None:
    try:
        from transformers import ByT5Tokenizer
    except ImportError:
        print("[skip] transformers gak terinstall — skip compare dgn HF")
        return
    tok = ByT5Tokenizer.from_pretrained("google/byt5-small")
    probes = [
        "Sam menyewa kamar kost di rumah angker.",
        "corrupted \ufeff text^ with 0x00 artifact~",
        "中文测试文本 🚀",
        "ISSN 9754-7018 E-ISSN:2898-2017",
    ]
    ok = True
    for p in probes:
        manual = [b + BYTE_OFFSET for b in p.encode("utf-8")]
        hf = tok(p, add_special_tokens=False)["input_ids"]
        match = manual == hf
        ok &= match
        print(f"  {'OK  ' if match else 'FAIL'} {p[:40]!r}")
    print("SEMUA IDENTIK DGN HF:", ok)
    if not ok:
        sys.exit(1)


def scan_bytes(data_dir: str) -> set:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        sys.exit(f"tidak ada parquet di {data_dir}")
    print(f"scanning {len(files)} file parquet ...")
    seen = set()
    for fn in files:
        t = pq.read_table(fn, columns=["raw", "clean"])
        for col in ["raw", "clean"]:
            for s in t.column(col).to_pylist():
                seen.update(s.encode("utf-8"))
        print(f"  {os.path.basename(fn)} | unique bytes: {len(seen)}")
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    print("=== 1. Compare dgn ByT5Tokenizer HF ===")
    compare_with_hf()

    if args.data_dir:
        print("\n=== 2. Scan corpus parquet ===")
        seen = scan_bytes(args.data_dir)
        missing = sorted(b for b in range(256) if b not in seen)
        print(f"unique bytes di corpus : {len(seen)}/256")
        print(f"byte tidak terpakai    : {missing if missing else 'TIDAK ADA'}")
        print("coverage vocab (byte+3): 256/256 = 100% -> OK")


if __name__ == "__main__":
    main()
