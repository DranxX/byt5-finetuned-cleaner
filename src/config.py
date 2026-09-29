"""
config.py — definisi arsitektur model yang didukung project ini.

Dua config resmi (hasil base-model selection + pilot):
  1. umt5-base     — google/umt5-base           (PIPELINE UTAMA, 580M)
  2. t5gemma-270m  — google/t5gemma-2-270m-270m (opsional, dipakai lain waktu)

Setiap config nentuinin:
  - base model HF repo id
  - target modules LoRA yang benar utk arsitekturnya
  - budget sekuens default (token)
  - attention implementation default

Script lain (check_env / finetune / inference) WAJIB ambil setting dari sini,
JANGAN hardcode nama model.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    key: str                    # nama pendek utk --model-family
    repo: str                   # HF repo id
    family: str                 # "t5" (enc-dec subword) — dua-duanya
    tokenizer: str              # "hf" = AutoTokenizer
    lora_targets: list          # module names untuk LoraConfig.target_modules
    max_input: int              # default token budget encoder
    max_target: int             # default token budget decoder
    attn: str                   # attention implementation default
    model_cls: str = "auto"     # nama class model utk import langsung.
                                # "auto" = AutoModelForSeq2SeqLM (butuh model_type
                                # di config.json). umt5 WAJIB "UMT5ForConditionalGeneration"
                                # karena config.json resmi google gak punya model_type
                                # (bug repo) -> AutoConfig raise "Unrecognized model".
    notes: str = ""


# ---------------------------------------------------------------
# CONFIG 1: UMT5-base — PIPELINE UTAMA (580M, pretraining EMA lebih stabil)
# ---------------------------------------------------------------
UMT5_BASE = ModelConfig(
    key="umt5-base",
    repo="google/umt5-base",
    family="t5",
    tokenizer="hf",
    # arsitektur = T5 family (verified: modeling_umt5 = clone modeling_t5;
    # bedanya cuma config: relative_attention_max_distance 128 + scalable_attention)
    lora_targets=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    max_input=4096,             # full-scan: input p99 id=185 tok, en median 1056 tok.
                                # 4096 = 98.7% row utuh (512 cuma 82.6% — en kepotong!)
    max_target=4096,            # p99.9 target id=174 tok, en p95 3.4K tok -> ~99% utuh
    attn="sdpa",                # FA2 gak didukung T5-family (verified)
    model_cls="UMT5ForConditionalGeneration",   # WAJIB: config.json google gak punya model_type
    notes="580M params | vocab 256K (+300 sentinel) | EMA-pretrained (lebih stabil) | "
          "tokenizer paling padat utk id (5.37 B/tok terukur) | "
          "tie_word_embeddings=False | fp16 DILARANG (T5-family overflow) | "
          "weights bf16 1.2GB — batch 8 aman di 8GB | "
          "pilot T4 (val 2.25 @178K rows) + WSL2 owner (loss 2.9/val 2.8 @100K rows) "
          "jalan di mt5-small — hyperparam & arch identik, hasil transfer",
)

# ---------------------------------------------------------------
# CONFIG 2: T5Gemma-2 270M-270M — opsional, dipakai lain waktu
# ---------------------------------------------------------------
T5GEMMA_270M = ModelConfig(
    key="t5gemma-270m",
    repo="google/t5gemma-2-270m-270m",
    family="t5",
    tokenizer="hf",
    # Gemma3-style attention+MLP: q_proj,k_proj,v_proj,o_proj + gate_proj/up_proj/down_proj
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    max_input=4096,
    max_target=4096,            # sliding window 4096 di sisi attention — sepadan
    attn="sdpa",                # FA2 gak didukung arch ini (verified _supports_flash_attn=False)
    model_cls="auto",           # T5Gemma2ForConditionalGeneration — config.json-nya benar
    notes="~370M params | vocab 262K | sliding_window 4096 + full-attn layers | "
          "GATED repo: wajib `huggingface-cli login` + accept Gemma terms | "
          "verified load+LoRA fwd/bwd+generate OK (transformers 4.55.x)",
)

CONFIGS = {c.key: c for c in (UMT5_BASE, T5GEMMA_270M)}

REMOVED = {
    "mt5-small": "mt5-small dihapus — pakai umt5-base (upgrade langsung: 580M, pretraining "
                 "lebih baik, tokenizer lebih padat). Pilot T4 & WSL2 tetap valid (arch sama).",
    "byt5-medium": "byt5-medium dihapus — byte-level selalu OOM di 8 GB utk korpus ini.",
}


def get_config(key: str) -> ModelConfig:
    if key in REMOVED:
        raise SystemExit(REMOVED[key])
    if key not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise SystemExit(f"model-family '{key}' gak dikenal. Pilihan: {valid}")
    return CONFIGS[key]
