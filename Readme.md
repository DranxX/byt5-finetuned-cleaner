# corpus-cleaner

LoRA fine-tuning for raw → clean text pairs. The active model is `google/umt5-base`; earlier pilots used `mt5-small`. UMT5 quality, VRAM requirements, and throughput need separate validation from those mT5 pilots.

## Precision and attention

| Model/GPU | Precision | Notes |
|---|---|---|
| UMT5 on T4 | FP32 | T4 lacks native BF16; FP16 is avoided because of forward overflow risk |
| UMT5 on RTX 30xx | BF16 | `--precision auto` selects native BF16 and loads base weights in BF16 |
| T5Gemma 2 on T4 | FP32 as a numerical baseline | The current dependency pin does not support the T5Gemma 2 architecture |
| T5Gemma 2 on RTX 30xx | BF16 after approval of a compatible environment | Transformers 4.55.4 supports first-generation T5Gemma; this configuration uses T5Gemma 2 |

UMT5 uses eager attention on Transformers 4.55.4. That implementation does not support SDPA, so requesting it cannot provide a more memory-efficient fused kernel. Non-reentrant gradient checkpointing remains enabled.

T5Gemma 2 is available in Transformers 5.0.0, while this project remains pinned to 4.55.4. `config.py` specifies the correct model class, and the scripts reject unsupported models before tokenization. Dependencies are not upgraded automatically; a Trainer/PEFT/Accelerate migration requires approval and validation of the complete environment.

References: [UMT5 4.55.4 source](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/umt5/modeling_umt5.py), [T5Gemma 2 in Transformers 5.0.0](https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/t5gemma2), [NVIDIA Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

## Environment gate

Use the same Python environment for installation, checks, and training. On Windows, keep the virtual environment, models, datasets, and caches on drive D as specified by the owner. Preserve the working PyTorch runtime.

```bash
python src/check_env.py --model-family umt5-base --data-dir dataset
```

Run training or inference only after the check exits with code 0. It verifies the CUDA build, pinned versions, gradient checkpointing, access to the selected model, dataset shards recursively, schema, and disk space. `--deep` validates both raw and tokenized datasets; the training script accepts the raw `lang/raw/clean` schema.

`--no-model` skips Hub access checks for offline environments. CUDA and library version checks still apply.

## UMT5 training

After the environment gate passes, select precision for the available GPU:

```bash
python src/finetune.py --data-dir dataset --out models/umt5-full --precision auto
```

On T4, start measurements with a small batch:

```bash
python src/finetune.py --data-dir dataset --out models/umt5-t4 --precision fp32 --batch 1 --accum 16
```

Script defaults: input length 4,096, target length 512, LoRA r32/alpha64, learning rate 3e-5, batch size 2, accumulation 8, and two epochs. `--bf16` remains available; do not combine it with `--precision fp32`.

- Tokenization is batched and stored in an Arrow cache. Cache identity includes the dataset, tokenizer vocabulary, special tokens, sequence budgets, and preprocessing version.
- Trainer receives the original Arrow dataset with its length column, so grouping avoids scanning the wrapper again for each row.
- Padding is aligned to multiples of eight. Label padding uses -100 and is excluded from the loss.
- Standard processes default to one GPU. Trainer utilities select the latest checkpoint numerically.
- Base weights are loaded in the selected dtype. LoRA adapters may remain FP32 while computation uses the selected mixed precision.
- `cleaning_config.json` in the final adapter stores the model family and encoder budget for inference.

Sequence budgets limit resource usage; rows can still exceed them. Truncated targets can teach the cleaner to discard document endings, while truncated inputs can lose evidence required by the target. Choose between skipping oversized pairs and truncating them before interpreting training fidelity. Verify alignment before chunking raw and clean text at matching token offsets.

Use a new run directory when the model, dataset, sequence budget, LoRA rank, or optimizer configuration changes. Resuming an existing checkpoint does not automatically validate all these changes.

## Inference

After the environment gate passes and an adapter is available:

```bash
python src/inference.py --adapter models/umt5-full/final --lang en --text "source text"
```

Generation defaults to min-new 0, one beam, repetition penalty 1, and no-repeat-ngram 0. These settings allow short outputs and meaningful source repetition. Inputs beyond the encoder budget are rejected, preventing silent truncation. Supported languages remain id/en/zh, with the same language prefix used during training.

If cleaning quality remains poor, evaluate the loss of numbers, hedges, negations, list/table structure, and document endings. Avoid masking these defects with a minimum output length or a global repetition ban.

## Resource measurements

Attention and vocabulary logits still consume substantial memory with LoRA. UMT5 eager attention has quadratic cost; decoder logits scale with batch size × target length × vocabulary size. Increase batch size only after measuring peak VRAM on long batches that actually occur in the dataset.

Record precision, row count, truncation, iterations per second, peak VRAM, evaluation loss, and content preservation checks. The two-shard notebook is an Indonesian pilot; its results do not represent the full English/Chinese distribution.
