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
        "vllm>=0.19.1",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(image=image, name="llm-compressor-quickstart-inference")
GPU = "H100"


def _extract_response(output) -> dict:
    """Extract reasoning + answer from a vLLM CompletionOutput.

    vLLM populates `reasoning_content` when a reasoning parser is active;
    otherwise the <think> block stays inline in `.text`. Handle both, plus the
    truncated case where generation ends before the closing </think>.
    """
    reasoning = (getattr(output, "reasoning_content", None) or "").strip()
    answer = (output.text or "").strip()
    if not reasoning and "</think>" in answer:
        r, a = answer.split("</think>", 1)
        reasoning = r.replace("<think>", "").strip()
        answer = a.strip()
    elif "<think>" in answer:
        # Truncated mid-reasoning: no </think> yet — strip the opener and treat
        # what we have as reasoning so the caller sees it isn't a real answer.
        reasoning = answer.replace("<think>", "").strip()
        answer = ""
    return {"reasoning": reasoning, "answer": answer}


@app.function(
    image=image,
    gpu=GPU,
    timeout=20 * 60,
    volumes={"/results": results_vol},
)
def generate(
    prompt: str = "Tell me a poem about Nairobi",
    model_dir: str = "/results/Ornith-1.0-9B-FP8-DYNAMIC",
    max_tokens: int = 4096,
    temperature: float = 0.6,
) -> dict:
    """Load the quantized model in vLLM and return reasoning + answer.

    Ornith is a reasoning model: it emits a <think>...</think> block before the
    final answer. We use llm.chat() so the model's chat template is applied,
    and the model-card sampling defaults (temp=0.6, top_p=0.95, top_k=20).
    max_tokens is generous because the reasoning trace alone can run long.
    """
    from vllm import LLM, SamplingParams

    print(f"Loading vLLM model from {model_dir}...")
    llm = LLM(model=model_dir, dtype="auto", enforce_eager=True, trust_remote_code=True)

    sampling = SamplingParams(
        temperature=temperature, top_p=0.95, top_k=20, max_tokens=max_tokens
    )
    messages = [{"role": "user", "content": prompt}]
    outputs = llm.chat(messages, sampling)
    return _extract_response(outputs[0].outputs[0])

@app.function(image=image, gpu=GPU, volumes={"/results": results_vol})
@modal.web_server(port=8000, startup_timeout=120)
def serve():
    """Long-lived web endpoint. Hit /generate with JSON: {"prompt": "..."}.

    The model is chosen via the INFERENCE_MODEL_DIR env var (set from the
    local entrypoint through `serve.with_options(env=...)`), defaulting to the
    rnj-1 checkpoint. `@modal.web_server` functions must be nullary, so the
    model can't be a regular parameter — env-var injection is the workaround.
    """
    import json
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from vllm import LLM, SamplingParams

    model_dir = os.environ.get(
        "INFERENCE_MODEL_DIR", "/results/Ornith-1.0-9B-FP8-DYNAMIC"
    )
    print(f"Loading vLLM model from {model_dir}...")
    llm = LLM(
        model=model_dir,
        dtype="auto",
        trust_remote_code=True,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = body.get("prompt", "")
            params = SamplingParams(
                temperature=body.get("temperature", 0.6),
                top_p=0.95,
                top_k=20,
                max_tokens=body.get("max_tokens", 4096),
            )
            messages = [{"role": "user", "content": prompt}]
            out = llm.chat(messages, params)[0].outputs[0]
            payload = json.dumps(_extract_response(out)).encode()
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
    prompt: str = "Share with me a poem about Nairobi",
    model_dir: str = "Ornith-1.0-9B-FP8-DYNAMIC",
    max_tokens: int = 4096,
    serve_mode: bool = False,
):
    remote_model_dir = f"/results/{model_dir}"
    if serve_mode:
        print(f"Starting web server with model {remote_model_dir}... press Ctrl+C to stop.")
        serve.with_options(env={"INFERENCE_MODEL_DIR": remote_model_dir}).remote()
        return

    print(f"Running on Modal {GPU} with prompt: {prompt!r}")
    output = generate.remote(
        prompt=prompt, model_dir=remote_model_dir, max_tokens=max_tokens
    )
    print(f"\nReasoning:\n{output['reasoning']}")
    print(f"\nAnswer:\n{output['answer']}")

