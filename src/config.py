"""
config.py — definisi arsitektur model yang didukung project ini.

Config resmi (pilihan final dari diskusi base-model selection + pilot):
  1. umt5-base     — google/umt5-base      (enc-dec, 580M, PIPELINE UTAMA)
  2. t5gemma-270m  — google/t5gemma-2-270m-270m (enc-dec, Gemma3-based, ~370M; lain waktu)

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
# CONFIG 2: T5Gemma-2 270M-270M (opsional — dipakai lain waktu) — enc-dec modern, sliding window 4096
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
    max_input=4096,
    max_target=4096,            # sliding window 4096 di sisi attention — sepadan.
                                # ~99% row utuh (est).
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


# ---------------------------------------------------------------
# CONFIG 1: UMT5-base (PIPELINE UTAMA) — versi "upgrade" mT5: params sama kayak mt5-base (580M)
# tapi pretrained berbeda (mC4 + EMA/scalable attention + max_distance 128),
# vocab 256,384 (+300 extra_id). Tokenizer lebih padat utk id (5.37 vs 4.43
# B/tok — terukur). Model_type 'umt5' tersedia sejak transformers lama (4.46+).
# ---------------------------------------------------------------
UMT5_BASE = ModelConfig(
    key="umt5-base",
    repo="google/umt5-base",
    family="t5",
    tokenizer="hf",
    # arsitektur = T5 family (verified: modeling_umt5 = clone modeling_t5
    # dgn class rename; bedanya cuma config: relative_attention_max_distance
    # 128 & scalable_attention). LoRA targets sama dgn mt5.
    lora_targets=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    max_input=4096,
    max_target=4096,
    attn="sdpa",                # FA2 gak didukung T5-family (sama dgn mt5)
    notes="580M params | 256K vocab (+300 sentinel) | EMA-pretrained (lebih stabil) | "
          "tie_word_embeddings=False | fp16 DILARANG (T5-family overflow, sama mt5) | "
          "VRAM: weights bf16 1.2GB — batch 8 aman di 8GB",
)

CONFIGS = {c.key: c for c in (UMT5_BASE, T5GEMMA_270M, BYT5_MEDIUM)}


def get_config(key: str) -> ModelConfig:
    if key == "mt5-small":
        # dihapus — pakai umt5-base (upgrade langsung, params lebih besar, tokenizer
        # lebih padat). Pesan ini biar command lama gak diem-diem jatuh ke default lain.
        raise SystemExit("mt5-small udah dihapus dari config. Pakai --model-family umt5-base "
                         "(upgrade langsung: 580M, pretraining lebih baik, tokenizer lebih padat).")
    if key not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise SystemExit(f"model-family '{key}' gak dikenal. Pilihan: {valid}")
    return CONFIGS[key]
