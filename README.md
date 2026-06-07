# LLM Compressor Quickstart

A minimal, copy-paste-runnable repo for **quantizing** LLMs with [LLM Compressor](https://github.com/vllm-project/llm-compressor), then **verifying** and **running** them in [vLLM](https://github.com/vllm-project/vllm) — locally or on cloud GPUs via [Modal](https://modal.com).

This quickstart quantizes **TinyLlama-1.1B-Chat** to **FP8 (dynamic activations, per-channel weights)** — the simplest scheme that requires no calibration data and recovers near-full accuracy on most models. The same recipe works for any Hugging Face causal LM.

---

## Requirements

- Python ≥ 3.10
- For **local** FP8 quantization: CUDA GPU with compute capability ≥ 8.9 (Ada Lovelace / Hopper / Blackwell)
- For **Modal** quantization: just a Modal account + $1 of credits (no GPU needed locally)
- ~3 GB disk for the model cache
- [Conda](https://docs.conda.io/projects/miniconda/) (Miniconda or Miniforge recommended)

> **No FP8 hardware?** Use the Modal scripts (see [Run on Modal](#-run-on-modal-cloud-gpu)) — they give you on-demand H100 access for ~$0.15/run.

---

## Setup (Conda — recommended)

The `environment.yml` pins everything via conda-forge and pip, so the setup is reproducible across machines.

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate it
conda activate llm-compressor-quickstart
```

To update an existing environment after pulling changes:
```bash
conda env update -f environment.yml --prune
```

To remove it entirely:
```bash
conda env remove -n llm-compressor-quickstart
```

### What gets installed

| Package | Channel | Why |
|---|---|---|
| `pytorch` + `pytorch-cuda` | conda-forge | GPU framework |
| `transformers`, `tokenizers` | conda-forge | Load HF model + tokenizer |
| `accelerate`, `safetensors` | conda-forge | Model loading helpers |
| `llmcompressor` | pip | Quantization library (not yet on conda-forge) |
| `compressed-tensors` | pip | Save format used by vLLM |
| `modal` | pip | Cloud GPU runner (quantize / infer / verify on H100 / L40S) |

> **Why no local vLLM?** vLLM has no Windows wheels and hard-pins `compressed-tensors` to a version incompatible with `llmcompressor`. So vLLM runs **inside the Modal container only** (see `modal_inference.py`). Locally you can still quantize (`quantize.py`) and run the HF-only check (`scripts/verify.py`).

### Alternative: plain pip
A `requirements.txt` is kept for users who can't use conda:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start (local — needs FP8 GPU)

> **Tip:** This repo ships a `Makefile`. Run `make help` to see every available
> command (setup, quantize, inference, cleanup, etc.). The snippets below show
> the raw commands for reference.

### 1. Quantize the model

```bash
python quantize.py
# or: make quantize
```

This will:
1. Download `TinyLlama/TinyLlama-1.1B-Chat-v1.0` from Hugging Face
2. Apply FP8_DYNAMIC quantization to all `Linear` layers (except `lm_head`)
3. Run a sanity-check generation on the quantized model
4. Save the result to `outputs/TinyLlama-1.1B-Chat-v1.0-FP8-Dynamic/`

Expected runtime: **~30–60 seconds** on a modern GPU.

### 2. Run inference

vLLM has no Windows wheels, so inference runs in the Modal container
(no local GPU required). See the [Modal section](#-run-on-modal-cloud-gpu)
for setup; once authenticated:

```bash
make inference-modal
# or: modal run modal_inference.py
```

---

## ☁️ Run on Modal (cloud GPU)

No FP8-capable GPU at home? The Modal scripts give you on-demand H100 access and bill per second. TinyLlama quantization costs roughly **$0.15/run**.

### One-time setup

```bash
conda activate llm-compressor-quickstart
modal token new     # opens browser to authenticate
```

### Quantize on Modal

```bash
make quantize-modal
# Override model or scheme (defaults: EssentialAI/rnj-1-instruct + FP8_DYNAMIC):
make quantize-modal MODEL=meta-llama/Meta-Llama-3-8B-Instruct SCHEME=FP8_BLOCK
# or directly:
modal run modal_quantize.py --model-id <org>/<model> --scheme <SCHEME>
```

This:
1. Spins up an H100 container on Modal
2. Downloads the model (cached in a Modal Volume across runs)
3. Runs `oneshot` quantization
4. **Saves the checkpoint to a Modal Volume** (before the sanity-check generation, to avoid a slow offloaded save)
5. Runs a sanity-check generation
6. **Downloads the checkpoint to your local `outputs/`** so you have a copy

### Re-download a checkpoint (no GPU cost)

The checkpoint persists on the Modal volume, so if a download stalls or you want
to pull a previously-quantized model without re-paying for quantization:

```bash
make download-modal
# or for a specific model/scheme:
make download-modal MODEL=EssentialAI/rnj-1-instruct SCHEME=FP8_DYNAMIC
# or directly:
modal run modal_quantize.py --download-only --model-id <org>/<model> --scheme <SCHEME>
```

### Run inference on Modal

```bash
make inference-modal
# or:
modal run modal_inference.py
modal run modal_inference.py --prompt "The capital of France is"
```

Or spin up a live HTTP endpoint (billed while running):
```bash
make serve-modal
# or:
modal run modal_inference.py --serve-mode
# Then POST to the printed URL:
curl -X POST https://<your-workspace>--llm-compressor-quickstart-inference-generate.modal.run \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Hello"}'
```

### Verify the checkpoint

Confirm the quantized model (a) still generates sensible output and (b) actually
shrank on disk — both compared against the original HF model:

```bash
make verify-modal
# or: modal run modal_verify.py
# Point at a non-default checkpoint:
modal run modal_verify.py --model-name TinyLlama-1.1B-Chat-v1.0-FP8-BLOCK
```

Sample output:
```
ORIGINAL  -> The capital of France is Paris, the largest city in France...
QUANTIZED -> The capital of France is Paris, the largest city in France...

Weight size on disk:
  ORIGINAL  : 2.190 GB
  QUANTIZED : 1.132 GB
  SHRINK    : 48.3% smaller
```

### Cost breakdown (TinyLlama)

| Step | H100 time | Approx cost |
|---|---|---|
| Cold start + image build | ~30 s | $0.03 |
| Model download (cached after first run) | ~60 s | $0.06 |
| Quantization | ~30 s | $0.03 |
| Sanity-check generation | ~10 s | $0.01 |
| **Total per run** | **~2 min** | **~$0.13** |

A $30 credit gets you ~200+ quantization runs. Larger models (Llama-3-8B) take ~3–5 min and cost ~$0.30 each.

---

## How it works

The whole pipeline is three lines of meaningful code:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)
```

- `QuantizationModifier` — declares *what* to quantize (algorithm + scheme)
- `oneshot` — applies the recipe to the model in place
- Saved checkpoints use the [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) format, which vLLM loads directly

---

## Customizing

### Swap the model
Edit `MODEL_ID` at the top of `quantize.py`:
```python
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # any HF causal LM
```

### Pick a different scheme
Common alternatives (see [LLM Compressor docs](https://docs.vllm.ai/projects/llm-compressor/)):
```python
scheme="FP8_BLOCK"      # block-wise FP8 weights (better accuracy, same HW)
scheme="W4A16_ASYM"     # INT4 weights, FP16 activations (older GPUs OK)
scheme="W8A8_INT8"      # INT8 weights + activations
```

### Skip more layers
Some layers (e.g. MoE routers) are sensitive — add them to `ignore`:
```python
ignore=["lm_head", "re:.*mlp.gate$"]
```

---

## Project layout

```
llm-compressor-quickstart/
├── README.md
├── Makefile               # `make help` lists all commands
├── environment.yml        # conda-forge environment (recommended)
├── requirements.txt       # pip-only fallback
├── .gitignore
├── quantize.py            # local quantization script (needs FP8 GPU)
├── modal_quantize.py      # Modal: quantize on H100 (+ --download-only to re-pull)
├── modal_inference.py     # Modal: run inference on L40S (or serve)
├── modal_verify.py        # Modal: generation + weight-size comparison
├── scripts/
│   └── verify.py          # extra: local HF-only checks (no vLLM)
└── outputs/               # quantized checkpoints land here (gitignored)
```

---

## Troubleshooting

**`RuntimeError: ... not supported on the current GPU`**
Your GPU doesn't support FP8. Use an INT4 (W4A16) scheme instead.

**`OSError: ... gated repo`**
Some models (e.g. Llama) require a HF token. Run:
```bash
huggingface-cli login
```

**Out of memory**
For larger models, see the [disk offloading](https://github.com/vllm-project/llm-compressor/tree/main/examples/disk_offloading) and [sequential onloading](https://github.com/vllm-project/llm-compressor/tree/main/examples/big_models_with_sequential_onloading) examples.

**`RuntimeError: Could not find nvcc`** *(Modal inference)*
vLLM's V1 engine JIT-compiles kernels at startup and needs the CUDA toolkit (`nvcc`). `modal_inference.py` already bases its image on `nvidia/cuda:...-cudnn-devel-...` to provide it — if you customized the image, use a `-devel` variant, not `debian_slim`.

**`OSError: Repo id must be in the form 'repo_name'...`** *(Modal inference / verify)*
The checkpoint directory name didn't match what the script looked for. Run `make results` to list what's in the volume, then pass `--model-name <dir>` — e.g. `...-FP8-BLOCK` vs the default `...-FP8-DYNAMIC`.

**Modal `FunctionTimeoutError` during save** *(Modal quantize)*
Quantization finished but `Saving checkpoint shards` stalled at 0% until the 30-min timeout. This happens when `save_pretrained` runs on a model with CPU-offloaded layers. `modal_quantize.py` now saves **before** the sanity-check dispatch to avoid this. If you customized the script and hit it again, make sure `save_pretrained` runs before `dispatch_model`. Either way the checkpoint is already on the volume — re-pull it free with `make download-modal`.

---

## Further reading

- [LLM Compressor docs](https://docs.vllm.ai/projects/llm-compressor/)
- [Step-by-step compression guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-model/)
- [Source repo (this library)](https://github.com/vllm-project/llm-compressor)
