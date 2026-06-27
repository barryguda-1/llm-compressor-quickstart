"""
Run inference on the quantized model using vLLM on a Modal H100 GPU.

Two modes:
    # 1. One-shot generation (returns immediately)
    make inference-modal
    modal run modal_inference.py --prompt "Is 1024 a power of 2?"

    # 2. Live web endpoint (stays up, billed while running)
    make serve-modal
    modal run modal_inference.py --serve-mode
    # then: curl https://<your-workspace>--llm-compressor-quickstart-inference-generate.modal.run \\
    #            -d '{"prompt": "Hello"}'

Defaults target the Mellum2-Thinking FP8 checkpoint. The chat template is
applied automatically (the previous version passed raw prompts to a
chat-tuned model -- broken). Reasoning traces inside <think>...</think>
blocks are parsed out by vLLM's qwen3 reasoning parser so that
`outputs[0].text` contains only the final answer; the reasoning itself is
available via `outputs[0].reasoning_text` if you want it.
"""

from pathlib import Path

import modal

results_vol = modal.Volume.from_name("llm-compressor-results", create_if_missing=True)

# vLLM's V1 engine JIT-compiles kernels (torch.compile / inductor) at startup,
# which requires nvcc + cuDNN. debian_slim has no CUDA toolkit, so we base on
# the NVIDIA CUDA devel image instead. (The quantize image is unaffected --
# llmcompressor/PyTorch wheels bundle the CUDA runtime and need no compiler.)
#
# vLLM >=0.20 is required for transformers v5 support (Mellum2 needs v5). We
# do NOT pin compressed-tensors or transformers here -- vLLM has its own
# strict pins (==0.17.0 for vllm 0.23) and the checkpoint written by
# llmcompressor 0.12 (compressed-tensors 0.17.1) is file-compatible.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.20.0",
        "accelerate>=1.0.0",
        "torchvision",  # required by transformers v5 lazy-import chain
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(image=image, name="llm-compressor-quickstart-inference")
GPU = "H100"
MODEL_NAME = "Mellum2-12B-A2.5B-Thinking-FP8"
PROMPT = "Is 1024 a power of 2? Explain your reasoning."


@app.function(
    image=image,
    gpu=GPU,
    timeout=20 * 60,
    volumes={"/results": results_vol},
)
def generate(
    prompt: str,
    model_name: str = MODEL_NAME,
    max_tokens: int = 512,
    temperature: float = 0.6,
) -> dict:
    """Load the quantized model in vLLM and return a single generation.

    Returns both the final answer and the reasoning trace (if any) so callers
    can see what the model "thought" before answering.
    """
    from vllm import LLM, SamplingParams

    model_dir = f"/results/{model_name}"
    print(f"Loading vLLM model from {model_dir}...")
    # reasoning_parser="qwen3" parses <think>...</think> blocks so that
    # outputs[i].text holds the final answer and outputs[i].reasoning_text
    # holds the CoT. Without it the answer is polluted with the trace.
    llm = LLM(
        model=model_dir,
        dtype="auto",
        enforce_eager=True,
        reasoning_parser="qwen3",
    )

    messages = [{"role": "user", "content": prompt}]
    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=max_tokens,
    )
    # llm.chat() applies the chat template internally -- safer than calling
    # apply_chat_template ourselves (return type varies across transformers
    # versions, and a raw-prompt generate() breaks chat-tuned models).
    outputs = llm.chat(messages=[messages], sampling_params=sampling)
    out = outputs[0].outputs[0]
    return {
        "answer": out.text,
        "reasoning": getattr(out, "reasoning_text", None) or "",
    }


@app.function(image=image, gpu=GPU)
@modal.web_server(port=8000, startup_timeout=180)
def serve():
    """Long-lived web endpoint. Hit /generate with JSON: {"prompt": "..."}."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from vllm import LLM, SamplingParams

    model_dir = f"/results/{MODEL_NAME}"
    llm = LLM(
        model=model_dir,
        dtype="auto",
        reasoning_parser="qwen3",
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = body.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]
            params = SamplingParams(
                temperature=body.get("temperature", 0.6),
                top_p=0.95,
                top_k=20,
                max_tokens=body.get("max_tokens", 512),
            )
            out = llm.chat(messages=[messages], sampling_params=params)[0].outputs[0]
            payload = json.dumps(
                {
                    "answer": out.text,
                    "reasoning": getattr(out, "reasoning_text", None) or "",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass  # silence default access logging

    server = HTTPServer(("0.0.0.0", 8000), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


@app.local_entrypoint()
def main(
    prompt: str = PROMPT,
    model_name: str = MODEL_NAME,
    max_tokens: int = 512,
    serve_mode: bool = False,
):
    if serve_mode:
        print("Starting web server... press Ctrl+C to stop.")
        serve.remote()
        return

    print(f"Running on Modal {GPU} with prompt: {prompt!r}")
    r = generate.remote(
        prompt=prompt, model_name=model_name, max_tokens=max_tokens
    )
    print(f"\nReasoning:\n  {r['reasoning']}")
    print(f"\nAnswer:\n  {r['answer']}")
