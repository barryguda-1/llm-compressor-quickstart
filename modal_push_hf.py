"""
Push a quantized checkpoint from the Modal results volume to the HuggingFace Hub.

Why a separate script? Uploading from inside a Modal container reads the
checkpoint off the volume (fast in-container I/O, no local download needed),
and `huggingface_hub` parallelizes multi-file uploads.

Setup (one-time):
    1. Create a read access token at https://huggingface.co/settings/tokens
       (needs `write_repos` scope to push a model).
    2. Register it as a Modal secret named `huggingface`:
           modal secret create huggingface HF_TOKEN=hf_xxx

Run:
    modal run modal_push_hf.py --repo-id <user>/rnj-1-instruct-FP8-DYNAMIC
    modal run modal_push_hf.py \\
        --model-id EssentialAI/rnj-1-instruct \\
        --scheme FP8_DYNAMIC \\
        --repo-id <user>/rnj-1-instruct-FP8-DYNAMIC \\
        --private

Prereq: a checkpoint must already exist on the results volume -- run
`make quantize-modal` first.
"""

from pathlib import Path

import modal

RESULTS_VOLUME_NAME = "llm-compressor-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.25.0",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(image=image, name="llm-compressor-quickstart-push")
results_vol = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _save_name(model_id: str, scheme: str) -> str:
    """Build the on-disk directory name for a quantized checkpoint."""
    return f"{model_id.split('/')[-1]}-{scheme.replace('_', '-')}"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/results": results_vol},
    timeout=20 * 60,
)
def push(save_name: str, repo_id: str, private: bool) -> dict:
    """Upload `/results/<save_name>` to the HuggingFace Hub as `repo_id`."""
    import os
    from huggingface_hub import HfApi

    src_dir = Path("/results") / save_name
    # Require actual weight files, not just the directory. A prior quantize
    # run that crashed before `results_vol.commit()` can leave a stale
    # snapshot where Path(...).exists() is True but no weights ever landed.
    weight_files = (
        list(src_dir.glob("*.safetensors")) + list(src_dir.glob("*.bin"))
        if src_dir.exists()
        else []
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No weight files (.safetensors / .bin) found in {src_dir}. "
            f"The quantization either did not complete or was never run. "
            f"Re-run `modal run modal_quantize.py`."
        )

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    who = api.whoami()
    ns = who["name"] if isinstance(who, dict) else str(who)
    print(f"Authenticated as: {ns}")

    repo_owner = repo_id.split("/")[0]
    if repo_owner != ns:
        print(
            f"WARNING: repo-id namespace '{repo_owner}' does not match the "
            f"authenticated user '{ns}'. Push will fail unless you have write "
            f"access to that namespace."
        )

    # create_repo is idempotent: existing_repo_ok=True means no error if it
    # already exists. Private flag is only applied on first creation.
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"Uploading {src_dir} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(src_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {save_name} (FP8 quantized)",
    )

    # Refresh volume metadata so subsequent reads in this container stay valid.
    results_vol.commit()

    url = f"https://huggingface.co/{repo_id}"
    return {"repo_id": repo_id, "url": url, "file_count": len(weight_files)}


@app.local_entrypoint()
def main(
    model_id: str = "EssentialAI/rnj-1-instruct",
    scheme: str = "FP8_DYNAMIC",
    repo_id: str = "",
    private: bool = False,
):
    if not repo_id or "/" not in repo_id:
        raise SystemExit(
            "--repo-id is required and must be of the form '<owner>/<name>', "
            "e.g. --repo-id your-username/rnj-1-instruct-FP8-DYNAMIC"
        )

    save_name = _save_name(model_id, scheme)
    print(
        f"Submitting push job to Modal "
        f"(checkpoint={save_name}, repo_id={repo_id}, private={private})..."
    )
    r = push.remote(save_name=save_name, repo_id=repo_id, private=private)
    print(f"\nPushed {r['file_count']} weight file(s).")
    print(f"View at: {r['url']}")
