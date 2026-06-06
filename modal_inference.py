"""
Run inference on the quantized model using vLLM on a Modal L40S GPU.

Two modes:
    # 1. One-shot generation (returns immediately)
    modal run modal_inference.py
    modal run modal_inference.py --prompt "The capital of France is"

    # 2. Live web endpoint (stays up, billed while running)
    modal run modal_inference.py --serve-mode
    # then: curl https://<your-workspace>--llm-compressor-quickstart-inference-generate.modal.run \\
    #            -d '{"prompt": "Hello"}'
"""

from pathlib import Path

import modal

results_vol = modal.Volume.from_name("llm-compressor-results", create_if_missing=True)

# vLLM's V1 engine JIT-compiles kernels (torch.compile / inductor) at startup,
# which requires nvcc + cuDNN. debian_slim has no CUDA toolkit, so we base on
# the NVIDIA CUDA devel image instead. (The quantize image is unaffected —
# llmcompressor/PyTorch wheels bundle the CUDA runtime and need no compiler.)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.6.0",
        "transformers>=4.45.0",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(image=image, name="llm-compressor-quickstart-inference")

@app.function(
    image=image,
    gpu="L40S",
    timeout=10 * 60,
    volumes={"/results": results_vol},
)
def generate(
    prompt: str = "The capital of France is",
    model_dir: str = "/results/TinyLlama-1.1B-Chat-v1.0-FP8-BLOCK",
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> str:
    """Load the quantized model in vLLM and return a single generation."""
    from vllm import LLM, SamplingParams

    print(f"Loading vLLM model from {model_dir}...")
    llm = LLM(model=model_dir, dtype="auto", enforce_eager=False)

    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sampling)
    return outputs[0].outputs[0].text

@app.function(image=image, gpu="L40S")
@modal.web_server(port=8000, startup_timeout=120)
def serve():
    """Long-lived web endpoint. Hit /generate with JSON: {"prompt": "..."}."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/results/TinyLlama-1.1B-Chat-v1.0-FP8-BLOCK",
        dtype="auto",
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = body.get("prompt", "")
            params = SamplingParams(
                temperature=body.get("temperature", 0.0),
                max_tokens=body.get("max_tokens", 100),
            )
            out = llm.generate([prompt], params)[0].outputs[0].text
            payload = json.dumps({"output": out}).encode()
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
    prompt: str = "The capital of France is",
    model_name: str = "TinyLlama-1.1B-Chat-v1.0-FP8-BLOCK",
    max_tokens: int = 50,
    serve_mode: bool = False,
):
    if serve_mode:
        print("Starting web server... press Ctrl+C to stop.")
        serve.remote()
        return

    remote_model_dir = f"/results/{model_name}"
    print(f"Running on Modal L40S with prompt: {prompt!r}")
    output = generate.remote(
        prompt=prompt, model_dir=remote_model_dir, max_tokens=max_tokens
    )
    print(f"\nOutput: {output}")

