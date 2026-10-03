import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config

CASES = [
    ("ood-id-jurnal", "id",
     "JurnalTeknolo5giInform  asi  Vol.12  No.30x00Maret2021  E-ISSN:2721-88  22X~0x00AnalisisPerforma  nsiSistem",
     "repair hanya artefak yang pasti; jangan menebak ISSN, nama, atau digit ambigu"),
    ("ood-id-webdump", "id",
     "Beranda | Tentang Kami | Kontak | FAQ\nArtikel Populer\n18 September 2025\nBandung - Pemerintah kota mengumumkan program baru untuk perpustakaan daerah.\nBaca Juga: 5 Tips Hemat Listrik\n(c) 2025 Portal Berita Contoh. All rights reserved.\nMenu\nSearch...",
     "nav/menu/footer dibuang; kalimat berita dan tanggal yang bermakna tetap utuh"),
    ("ood-id-mojibake", "id",
     "﻿KetuaÂ panitiaÂ menyatakan–“acara ini akan berjalan sesuai rencana” kataÂ beliauÂ kemarin.",
     "BOM + mojibake (Â nbsp) dibenerin; konten TIDAK diduplikasi"),
    ("ood-en-webjunk", "en",
     "Skip to content Home Products Pricing Blog Login\nMARCH 14, 2024\n6 MIN READ\nOur team shipped a new feature that reduces latency by 40% across all endpoints.\nShare on Twitter Share on LinkedIn\nSubscribe to our newsletter\n(c) 2024 ExampleCorp Inc. Privacy Policy Terms of Service Cookie Settings",
     "klaim 40% dan tanggal tetap utuh; nav/share/footer dibuang"),
    ("ood-zh-webjunk", "zh",
     "首页 | 最新消息 | 热门排行\n云南目标2025年3月15日\n北京报道（记者王某）昨日发布的数据显示，本季度增速回落。\n版权所有 (c) 2025 示例网站\n关于我们 联系我们",
     "nav/footer dibuang; judul, tanggal, byline, dan kalimat berita tetap utuh"),
    ("indomain-id-header", "id",
     "No. 24; Diperbarui Maret 2011  \nKlik di sini untuk mengunduh dan mencetak versi PDF dokumen ini.  \nOrang tua biasanya yang pertama mengenali bahwa anak mereka mengalami masalah emosional.",
     "kontrol PDF dibuang; kalimat substantif dan metadata yang bermakna tetap utuh"),
    ("indomain-id-issn", "id",
     "Dijual  Ce1paISSN9754-701  8~E,-ISSN:2898-20717S.~  R.PratiwiK--a,j  ia--n0x00P,sikolo--giSosial",
     "konten tetap utuh; karakter dan identifier ambigu tidak ditebak"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-family", default="umt5-base")
    ap.add_argument("--base", default=None, help="override repo base (mis. snapshot offline)")
    ap.add_argument("--beams", type=int, default=1,
                    help="jumlah beam; bandingkan fidelity dan waktu generation")
    ap.add_argument("--min-new", type=int, default=0,
                    help="0 membolehkan output pendek tanpa memaksa tambahan isi")
    ap.add_argument("--max-new", type=int, default=4096)
    args = ap.parse_args()

    from inference import load_model, clean

    cfg = get_config(args.model_family)
    model, tok, device = load_model(args.adapter, cfg, args.base)
    print(f"model: {cfg.repo} + adapter {args.adapter} | device={device}\n")

    for name, lang, raw, expect in CASES:
        out = clean(model, tok, raw, lang, device,
                    max_new_tokens=args.max_new, num_beams=args.beams,
                    min_new_tokens=args.min_new)
        print(f"== {name} [{lang}]")
        print(f"   RAW   : {raw[:110]}{'...' if len(raw) > 110 else ''}")
        print(f"   CLEAN : {out[:220]}{'...' if len(out) > 220 else ''}")
        print(f"   EXPECT: {expect}\n")


if __name__ == "__main__":
    main()
