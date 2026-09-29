"""
config.py — definisi arsitektur model yang didukung project ini.

Dua config resmi (pilihan final dari diskusi base-model selection):
  1. mt5-small     — google/mt5-small      (enc-dec, SentencePiece 250K, 300M)
  2. t5gemma-270m  — google/t5gemma-2-270m-270m (enc-dec, Gemma3-based, ~370M)

Setiap config nentuinin:
  - base model HF repo id
  - apakah tokenizer byte-level manual (ByT5 legacy) atau HF tokenizer
  - target modules LoRA yang benar utk arsitekturnya
  - budget sekuens default (token)
  - flag khusus (mis. attn_implementation, use_cache)

Script lain (check_env / finetune / inference) WAJIB ambil setting dari sini,
JANGAN hardcode nama model.
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    key: str                    # nama pendek utk --model-family
    repo: str                   # HF repo id
    family: str                 # "t5" (enc-dec subword) | "byt5" (byte-level legacy)
    tokenizer: str              # "hf" = pakai AutoTokenizer | "byte" = byte+3 manual
    lora_targets: list          # module names untuk LoraConfig.target_modules
    max_input: int              # default token budget encoder
    max_target: int             # default token budget decoder
    attn: str                   # attention implementation default
    notes: str = ""


# ---------------------------------------------------------------
# CONFIG 1: mT5-small — pilihan aman, zero-dependency, tercepat di 8 GB
# ---------------------------------------------------------------
MT5_SMALL = ModelConfig(
    key="mt5-small",
    repo="google/mt5-small",
    family="t5",
    tokenizer="hf",
    # mT5 = T5 arch: attention q,k,v,o + FFN wi_0/wi_1/wo (Gated GELU)
    lora_targets=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    max_input=512,
    max_target=2048,            # full-scan: p99 target id=149 tok, en=6037 tok.
                                # 2048 tok = ~96% row utuh (mt5). batch 16 mepet di
                                # long-tail; ckpting menahan, batch 8 kalau OOM.
    attn="sdpa",                # T5-family GAK support FA2 — SDPA = tercepat tersedia
    notes="300M params | 250K vocab | median 92 tok utk rows id (73.6% korpus) | "
          "tanpa dependency kernel eksternal | "
          "PILOT T4 SUKSES (val loss 25.5->2.25, lr 3e-5 r16, fp32); "
          "fp16 DILARANG (overflow); lr 2e-4 DILARANG (diverge)",
)

# ---------------------------------------------------------------
# CONFIG 2: T5Gemma-2 270M-270M — enc-dec modern, sliding window 4096
# ---------------------------------------------------------------
T5GEMMA_270M = ModelConfig(
    key="t5gemma-270m",
    repo="google/t5gemma-2-270m-270m",
    family="t5",
    tokenizer="hf",
    # Gemma3-style attention+MLP: q_proj,k_proj,v_proj,o_proj + gate_proj/up_proj/down_proj
    # (T5Gemma2 merge self/cross attention — target tetap sama)
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    max_input=512,
    max_target=2048,            # sliding window 4096 di sisi attention; budget 2048
                                # = ~97% row utuh (est).
    attn="sdpa",                # FA2 gak didukung arch ini (verified _supports_flash_attn=False)
    notes="~370M params | vocab 262K | sliding_window 4096 + full-attn layers | "
          "GATED repo: wajib `huggingface-cli login` + accept Gemma terms dulu | "
          "verified load+LoRA fwd/bwd+generate OK (transformers 4.55.x)",
)

# legacy: ByT5 (byte-level) — dipertahankan sebagai baseline/kontrol, BUKAN target
BYT5_MEDIUM = ModelConfig(
    key="byt5-medium",
    repo="google/byt5-medium",
    family="byt5",
    tokenizer="byte",
    lora_targets=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    max_input=512,
    max_target=256,
    attn="sdpa",
    notes="LEGACY baseline. Byte-level: 1 byte = 1 token -> selalu OOM di 8 GB "
          "utk korpus ini. Hanya dipakai buat pembanding/ablation.",
)

CONFIGS = {c.key: c for c in (MT5_SMALL, T5GEMMA_270M, BYT5_MEDIUM)}


def get_config(key: str) -> ModelConfig:
    if key not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise SystemExit(f"model-family '{key}' gak dikenal. Pilihan: {valid}")
    return CONFIGS[key]
