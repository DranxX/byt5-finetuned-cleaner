from dataclasses import dataclass


@dataclass
class ModelConfig:
    key: str
    repo: str
    family: str
    tokenizer: str
    lora_targets: list[str]
    max_input: int
    max_target: int
    attn: str
    model_cls: str = "auto"
    notes: str = ""


UMT5_BASE = ModelConfig(
    key="umt5-base",
    repo="google/umt5-base",
    family="t5",
    tokenizer="hf",
    lora_targets=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    max_input=4096,
    max_target=512,
    attn="eager",
    model_cls="UMT5ForConditionalGeneration",
    notes="580M params | vocab 256K | eager attention in transformers 4.55.4 | "
          "T4: fp32; native Ampere+: bf16 | fp16 risks forward overflow | "
          "measure VRAM and throughput at the selected sequence budget",
)

T5GEMMA_270M = ModelConfig(
    key="t5gemma-270m",
    repo="google/t5gemma-2-270m-270m",
    family="t5",
    tokenizer="hf",
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
    max_input=4096,
    max_target=512,
    attn="sdpa",
    model_cls="T5Gemma2ForConditionalGeneration",
    notes="T5Gemma 2, not first-generation T5Gemma | gated model | "
          "unsupported by the project's transformers 4.55.4 pin | "
          "requires an approved compatible training environment | "
          "decoder logits scale with target length and vocabulary size",
)

CONFIGS = {config.key: config for config in (UMT5_BASE, T5GEMMA_270M)}

REMOVED = {
    "mt5-small": "mt5-small bukan config aktif; jalur utama memakai umt5-base. "
                 "Hasil pilot mt5-small tidak membuktikan kualitas atau VRAM umt5-base.",
    "byt5-medium": "byt5-medium bukan config aktif; gunakan salah satu model subword yang tersedia.",
}


def get_config(key: str) -> ModelConfig:
    if key in REMOVED:
        raise SystemExit(REMOVED[key])
    if key not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise SystemExit(f"model-family '{key}' gak dikenal. Pilihan: {valid}")
    return CONFIGS[key]
