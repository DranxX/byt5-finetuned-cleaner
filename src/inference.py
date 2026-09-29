"""
inference.py — load base + adapter LoRA, clean dirty text.

Pakai config dari config.py (sama dengan finetune.py):
  python src/inference.py --model-family mt5-small --adapter models/mt5s1/final --text "..."
  python src/inference.py --model-family t5gemma-270m --adapter models/t5g1/final --text "..."
  python src/inference.py --model-family mt5-small --base /local/path --text "..."   # offline

Legacy ByT5 (byte-level) juga didukung utk pembanding adapter lama.
"""
import argparse

import torch

BYTE_OFFSET = 3


def load_model(adapter_path, family_cfg, base_override=None, device=None):
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base = base_override or family_cfg.repo
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=dtype, attn_implementation=family_cfg.attn)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval().to(device)

    tok = AutoTokenizer.from_pretrained(adapter_path)  # adapter save menyertakan tokenizer
    return model, tok, device


@torch.no_grad()
def clean(model, tok, text, lang, device, max_new_tokens=512, num_beams=1,
          min_new_tokens=0, repetition_penalty=1.2, no_repeat_ngram=4):
    """Prefix <lang> wajib sama dgn format training.

    Default mitigation (hasil pilot T4):
    - repetition_penalty + no_repeat_ngram: cegah repetition loop di teks susah
    - num_beams>1 + min_new_tokens: cegah EOS premature (length prior data synthetic)
    - sentinel <extra_id_N> (prior span-corruption mT5) di-strip otomatis
    """
    src = f"<{lang}> {text}"
    max_len = getattr(model.config, "encoder_max_length", None) or 512
    ids = tok(src, return_tensors="pt", truncation=True,
              max_length=max_len).to(device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens,
                         num_beams=num_beams, do_sample=False,
                         min_new_tokens=min_new_tokens,
                         repetition_penalty=repetition_penalty,
                         no_repeat_ngram_size=no_repeat_ngram)
    import re
    txt = tok.decode(out[0], skip_special_tokens=True)
    return re.sub(r"\s*<extra_id_\d+>\s*", " ", txt).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-family", default="mt5-small")
    ap.add_argument("--base", default=None, help="override base model repo/local path")
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="id", choices=["id", "en", "zh"])
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--beams", type=int, default=1)
    args = ap.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import get_config
    cfg = get_config(args.model_family)

    model, tok, device = load_model(args.adapter, cfg, args.base)
    result = clean(model, tok, args.text, args.lang, device, args.max_new, args.beams)
    print("LANG :", args.lang)
    print("RAW  :", args.text[:200])
    print("CLEAN:", result)


if __name__ == "__main__":
    main()
