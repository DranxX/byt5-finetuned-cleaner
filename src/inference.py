import argparse
import json
import math
from pathlib import Path

import torch

SUPPORTED_LANGUAGES = {"id", "en", "zh"}


def load_model(adapter_path, family_cfg, base_override=None, device=None):
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_device = torch.device(device)
    base = base_override or family_cfg.repo
    is_native_bf16 = runtime_device.type == "cuda" and torch.cuda.get_device_capability(runtime_device) >= (8, 0)
    dtype = torch.bfloat16 if is_native_bf16 else torch.float32
    max_input = family_cfg.max_input
    config_path = Path(adapter_path) / "cleaning_config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            cleaning_config = json.load(handle)
        if not isinstance(cleaning_config, dict) or cleaning_config.get("model_family") != family_cfg.key:
            raise ValueError("adapter dan model-family tidak sesuai")
        max_input = cleaning_config.get("max_input")
        if type(max_input) is not int or max_input <= 0:
            raise ValueError("budget encoder adapter tidak valid")


    if family_cfg.model_cls == "auto":
        model = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype=dtype, attn_implementation=family_cfg.attn)
    else:
        import transformers
        if not hasattr(transformers, family_cfg.model_cls):
            raise ValueError("versi transformers belum mendukung model-family yang dipilih")
        cls = getattr(transformers, family_cfg.model_cls)
        model = cls.from_pretrained(base, torch_dtype=dtype, attn_implementation=family_cfg.attn)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval().to(device)
    model.config.encoder_max_length = max_input

    tok = AutoTokenizer.from_pretrained(adapter_path)
    return model, tok, device


@torch.no_grad()
def clean(model, tok, text, lang, device, max_new_tokens=4096, num_beams=1,
          min_new_tokens=0, repetition_penalty=1.0, no_repeat_ngram=0):
    if not isinstance(text, str) or not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES:
        raise ValueError("text harus string dan bahasa harus id/en/zh")
    if (type(max_new_tokens) is not int or max_new_tokens <= 0 or
            type(num_beams) is not int or num_beams <= 0 or
            type(min_new_tokens) is not int or not 0 <= min_new_tokens <= max_new_tokens or
            type(no_repeat_ngram) is not int or no_repeat_ngram < 0 or
            type(repetition_penalty) not in (int, float) or
            not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
        raise ValueError("parameter generation tidak valid")
    if not text:
        return ""
    src = f"<{lang}> {text}"

    max_len = getattr(model.config, "encoder_max_length", None) or 4096
    ids = tok(src, return_tensors="pt", truncation=False)
    if ids["input_ids"].shape[-1] > max_len:
        raise ValueError("input melebihi budget encoder; pecah dokumen sebelum inference")
    ids = ids.to(device)
    is_native_bf16 = torch.device(device).type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=is_native_bf16):
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             num_beams=num_beams, do_sample=False,
                             min_new_tokens=min_new_tokens,
                             repetition_penalty=repetition_penalty,
                             no_repeat_ngram_size=no_repeat_ngram)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-family", default="umt5-base")
    ap.add_argument("--base", default=None, help="override base model repo/local path")
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="id", choices=["id", "en", "zh"])
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--min-new", type=int, default=0,
                    help="0 menjaga output pendek agar tidak dipaksa menambah isi")
    args = ap.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import get_config
    cfg = get_config(args.model_family)

    model, tok, device = load_model(args.adapter, cfg, args.base)
    result = clean(model, tok, args.text, args.lang, device, args.max_new,
                   args.beams, min_new_tokens=args.min_new)
    print("LANG :", args.lang)
    print("RAW  :", args.text[:200])
    print("CLEAN:", result)


if __name__ == "__main__":
    main()
