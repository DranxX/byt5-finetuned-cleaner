"""
Inference: load base ByT5 + adapter LoRA, clean dirty text.

Usage:
  python src/inference.py --adapter output/m1/final --text "dirty text here"
  python src/inference.py --adapter user/repo-on-hf --text "..."
"""
import argparse

import torch

BYTE_OFFSET = 3


def load_model(adapter_path: str, base_model: str = "google/byt5-medium", device: str | None = None):
    from peft import PeftModel
    from transformers import T5ForConditionalGeneration

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base = T5ForConditionalGeneration.from_pretrained(base_model)
    base.config.tie_word_embeddings = False
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval().to(device)
    return model, device


def encode(text: str, max_len: int = 1024) -> list[int]:
    """byte+3 (identik ByT5Tokenizer), tanpa EOS (decoder yang generate)."""
    return [b + BYTE_OFFSET for b in text.encode("utf-8")[:max_len]]


def decode(ids) -> str:
    """ids -> string (skip special: id di luar range byte)."""
    return bytes(b - BYTE_OFFSET for b in ids if 3 <= b <= 258).decode("utf-8", errors="replace")


@torch.no_grad()
def clean(model, device: str, text: str, max_new_tokens: int = 512, num_beams: int = 1) -> str:
    input_ids = torch.tensor([encode(text)], dtype=torch.long, device=device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    return decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="google/byt5-medium")
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--beams", type=int, default=1)
    args = ap.parse_args()

    model, device = load_model(args.adapter, args.base)
    result = clean(model, device, args.text, args.max_new, args.beams)
    print("RAW  :", args.text[:200])
    print("CLEAN:", result)


if __name__ == "__main__":
    main()
