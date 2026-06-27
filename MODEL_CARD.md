---
license: apache-2.0
base_model: JetBrains/Mellum2-12B-A2.5B-Thinking
quantized_by: barryke
tags:
- fp8
- fp8-static
- vllm
- compressed-tensors
- llm-compressor
- quantization
- text-generation
- reasoning
- thinking
- code
- mellum
- moe
- mixture-of-experts
language:
- en
library_name: transformers
pipeline_tag: text-generation
---

# Mellum2-12B-A2.5B-Thinking-FP8

## Model Description

This is an **FP8 statically-quantized** version of [JetBrains/Mellum2-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking), JetBrains' reasoning-augmented MoE coding model. Mellum2-Thinking is produced from [`Mellum2-12B-A2.5B-Base`](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base) via supervised fine-tuning followed by RL with verifiable rewards (RLVR) on a harder data mix that includes long-form math. The model emits its reasoning inside `<think>...</think>` blocks before the final answer.

Quantization was performed using [LLM Compressor](https://github.com/vllm-project/llm-compressor) v0.12+ via a post-training **one-shot** RTN method with **static activation calibration** on 512 samples from UltraChat-200K. The checkpoint is saved in the [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) format, natively supported by **vLLM** and **transformers**.

## Quantization Details

| Property | Value |
|---|---|
| **Base model** | `JetBrains/Mellum2-12B-A2.5B-Thinking` |
| **Quantization method** | `compressed-tensors` (via LLM Compressor `oneshot`, RTN) |
| **Scheme** | `FP8` (per-channel weights + **static** per-tensor activations) |
| **Weight quantization** | FP8 (float8 E4M3), per-channel, symmetric |
| **Activation quantization** | FP8 (float8 E4M3), per-tensor, **static** (calibrated) |
| **Targets** | All `Linear` layers (attention projections + all 64 experts per layer) |
| **Ignored layers** | `lm_head`, `re:.*mlp.gate$` (router kept at full precision) |
| **LLM Compressor version** | `>=0.12.0` |
| **compressed-tensors version** | `>=0.17.1` |
| **Calibration dataset** | `HuggingFaceH4/ultrachat_200k`, split `train_sft` |
| **Calibration samples** | 512 (seed=42, shuffled) |
| **Calibration sequence length** | 2048 tokens |
| **Total size on disk** | ~12 GB (down from ~24 GB BF16, ~50% reduction) |

### Why static FP8 (not dynamic)?

- **Faster inference** — static activation scales are baked into the checkpoint, so vLLM skips per-token scale computation at runtime.
- **Calibrated accuracy** — running 512 real chat samples through the model captures the activation distribution the model will see in production, giving more accurate scales than dynamic per-token estimation for outlier-heavy MoE expert layers.
- **MoE-aware** — `load_context()` linearizes the 64 experts per layer so the quantizer can actually reach every expert MLP. Without it, only the attention projections would be quantized (a common silent failure mode for MoE PTQ).

### Quantization Recipe

```yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, "re:.*mlp.gate$"]
      scheme: FP8
```

### Quantization Code

```python
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Thinking"

# load_context() linearizes MoE experts so they're visible to the quantizer.
with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Calibration data.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(512))
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
ds = ds.map(
    lambda s: tokenizer(s["text"], padding=False, max_length=2048, truncation=True, add_special_tokens=False),
    remove_columns=ds.column_names,
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

model.save_pretrained("./Mellum2-12B-A2.5B-Thinking-FP8")
tokenizer.save_pretrained("./Mellum2-12B-A2.5B-Thinking-FP8")
```

## Model Architecture

Mellum2-Thinking is a decoder-only **Mixture-of-Experts** transformer with sliding-window + full attention layers, GQA, RoPE, and RMSNorm. It uses a custom `MellumForCausalLM` architecture (Qwen3-family derived, with a `MellumTopKRouter`).

| Hyperparameter | Value |
|---|---|
| Architecture | `MellumForCausalLM` |
| Total parameters | ~12 B |
| Active parameters per token | ~2.5 B |
| Layers | 28 |
| Hidden size | 2304 |
| MLP intermediate size (dense) | 7168 |
| MoE intermediate size (per expert) | 896 |
| **Number of experts** | **64** |
| **Activated experts per token** | **8** |
| Router | `MellumTopKRouter` (`model.layers.N.mlp.gate`) |
| Attention heads (Q) | 32 |
| KV heads (GQA) | 4 |
| Context length | 131,072 (128K) |
| Sliding window | 1,024 |
| Vocabulary size | 98,304 |
| Original dtype | bfloat16 |

## Capabilities

Use this quantized checkpoint when you want **explicit chain-of-thought before the final answer** — complex debugging, multi-step planning, agentic workflows, and math- or reasoning-heavy tasks. For direct, low-latency answers without reasoning traces, use [JetBrains/Mellum2-12B-A2.5B-Instruct](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct) instead.

### Representative base-model benchmarks (from the official Mellum2 Thinking card)

| Benchmark | Mellum2 Thinking (BF16) |
|---|---|
| LiveCodeBench v6 (pass@1) | 69.9 |
| BFCL v3 (accuracy) | 69.4 |
| BFCL v4 (macro-avg, 5 subtasks) | 45.6 |
| AIME 2025+2026 (mean, 30 q each) | 58.4 |
| GSM-Plus (exact match) | 87.0 |
| MMLU-Redux (accuracy) | 86.2 |
| GPQA Diamond (accuracy) | 57.6 |
| IFEval | 76.5 |
| HarmBench (↓, lower is better) | 20.6 |
| XSTest | 89.6 |

> All values self-reported by JetBrains on the BF16 base. The FP8 quantized version should recover near-full accuracy (typically within ~1 pp on these benchmarks for FP8 static), but no formal re-eval has been run.

## How to Use

### vLLM (recommended for production)

```bash
pip install vllm>=0.10.0
vllm serve barryke/Mellum2-12B-A2.5B-Thinking-FP8 \
  --max-model-len 131072 \
  --reasoning-parser qwen3
```

With tool calling:

```bash
vllm serve barryke/Mellum2-12B-A2.5B-Thinking-FP8 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "barryke/Mellum2-12B-A2.5B-Thinking-FP8"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Is 1024 a power of 2? Explain your reasoning."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=8192,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    pad_token_id=tokenizer.eos_token_id,
)
# The response includes <think>...</think> blocks -- strip them for the final answer.
print(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### OpenAI-compatible API (via vLLM)

```python
from openai import OpenAI
# Configured by environment variables (vllm serve)
client = OpenAI()

response = client.chat.completions.create(
    model="barryke/Mellum2-12B-A2.5B-Thinking-FP8",
    messages=[{"role": "user", "content": "Is 1024 a power of 2? Explain your reasoning."}],
    max_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    extra_body={"top_k": 20},
)
print(response.choices[0].message.reasoning)  # CoT trace
print(response.choices[0].message.content)    # final answer
```

## Recommendations

- **Always apply the chat template** — call `tokenizer.apply_chat_template(...)` with `add_generation_prompt=True`. Raw prompts to a chat-tuned model produce garbage.
- **Sampling for Thinking models** — use `temperature=0.6, top_p=0.95, top_k=20` (from the Mellum2 card). Greedy decoding tends to produce degraded, repetitive reasoning traces.
- **Hardware requirement** — FP8 inference requires an NVIDIA GPU with compute capability >= 8.9 (Ada Lovelace / Hopper / Blackwell). For other GPUs, use the original BF16 model or a W4A16 quantization variant.
- **KV-cache budget** — at 128K context, KV-cache dominates memory; size `--gpu-memory-utilization` / `--max-model-len` accordingly when serving.
- **`<think>` blocks** — the model emits long CoT traces before the answer. In vLLM, use `--reasoning-parser qwen3` to split these automatically. In transformers, strip them with a regex.

## Known Limitations

- **Calibration data mismatch** — UltraChat-200K is general chat data; Mellum2-Thinking's training distribution skews toward code, math, and reasoning. A domain-matched calibration set (e.g. Open-Platypus + code samples) may recover additional accuracy. Re-running with `--num-samples 1024 --max-seq-len 4096` is also worth trying.
- **No formal re-evaluation** — accuracy claims rest on FP8 being near-lossless; verify on your own benchmarks before deployment.
- **Reasoning traces are long** — a single response can emit thousands of CoT tokens. Budget for this in `max_tokens` and latency planning.

## License

This model inherits the [Apache License 2.0](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking/blob/main/LICENSE) from the base model.

## Citation

```bibtex
@misc{mellum2,
  title  = {Mellum2: A 12B Mixture-of-Experts Model by JetBrains},
  author = {{JetBrains}},
  url    = {https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking},
  note   = {Apache 2.0 licensed MoE reasoning model with 64 experts (8 active) and 128K context}
}
```

```bibtex
@software{llm-compressor,
  title  = {{LLM Compressor: An easy-to-use library for compressing LLMs}},
  author = {{Red Hat AI and vLLM Project}},
  url    = {https://github.com/vllm-project/llm-compressor},
  note   = {Used v0.12+ with load_context() to produce this FP8-static MoE checkpoint}
}
```
