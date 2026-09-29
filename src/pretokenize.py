"""
Pre-tokenisasi dataset corpus-cleaning-v1 -> parquet siap-train.

Skema byte-level ByT5 (token_id = byte + 3) — terbukti identik dgn
ByT5Tokenizer HF (check_tokenizer.py), tapi ~440x lebih cepat.

Input : folder parquet dgn kolom lang, raw, clean
Output: folder parquet dgn kolom input_ids, labels

Usage:
  python src/pretokenize.py --data-dir /path/parquet --out data_tok \
      --max-input 1024 --max-target 512
"""
import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

BYTE_OFFSET = 3
EOS_ID = 1


def encode(s: str, max_len: int) -> tuple[list[int], bool]:
    """string -> (ids, truncated). ids = byte+3, +EOS di akhir (jika muat)."""
    data = s.encode("utf-8")
    truncated = len(data) > max_len
    ids = [b + BYTE_OFFSET for b in data[:max_len]]
    if len(ids) < max_len:
        ids.append(EOS_ID)
    return ids, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-input", type=int, default=512,
                    help="byte budget input; 512 nyangkut ~99%% baris corpus")
    ap.add_argument("--max-target", type=int, default=256,
                    help="byte budget target; 256 nyangkut ~99%% baris")
    ap.add_argument("--rows-per-shard", type=int, default=100_000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"tidak ada parquet di {args.data_dir}")
    os.makedirs(args.out, exist_ok=True)

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("labels", pa.list_(pa.int32())),
    ])

    state = {"shard": 0, "buf_in": [], "buf_lab": [], "total": 0, "skipped": 0}

    def flush():
        if not state["buf_in"]:
            return
        tbl = pa.table({
            "input_ids": pa.array(state["buf_in"], pa.list_(pa.int32())),
            "labels": pa.array(state["buf_lab"], pa.list_(pa.int32())),
        }, schema=schema)
        path = os.path.join(args.out, f"tmp-{state['shard']:05d}.parquet")
        pq.write_table(tbl, path)
        print(f"  wrote {os.path.basename(path)} ({len(state['buf_in'])} rows)")
        state["shard"] += 1
        state["buf_in"], state["buf_lab"] = [], []

    for fn in files:
        pf = pq.ParquetFile(fn)
        for batch in pf.iter_batches(batch_size=10_000, columns=["raw", "clean"]):
            for raw, clean in zip(batch.column("raw").to_pylist(),
                                  batch.column("clean").to_pylist()):
                inp_ids, in_trunc = encode(raw, args.max_input)
                lab_ids, _ = encode(clean, args.max_target)
                if in_trunc and len(inp_ids) >= args.max_input:
                    state["skipped"] += 1
                    continue
                state["buf_in"].append(inp_ids)
                state["buf_lab"].append(lab_ids)
                state["total"] += 1
                if len(state["buf_in"]) >= args.rows_per_shard:
                    flush()
        print(f"{os.path.basename(fn)} done | total rows so far: {state['total']}")

    flush()

    # rename sesuai jumlah shard final
    tmp_files = sorted(glob.glob(os.path.join(args.out, "tmp-*.parquet")))
    n = len(tmp_files)
    for i, f in enumerate(tmp_files):
        os.rename(f, os.path.join(args.out, f"train-{i:05d}-of-{n:05d}.parquet"))
    print(f"\nDONE: {state['total']} rows -> {n} shard di {args.out}")
    print(f"skipped (input > {args.max_input} byte): {state['skipped']}")


if __name__ == "__main__":
    main()
