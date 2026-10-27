"""Step 1: download the Fast-WAM Optional-IDM LIBERO checkpoint straight into the Modal Volume "hf-cache".

  modal run setup/download_ckpt.py

The download runs on a small CPU-only Modal machine (no GPU), so the 12 GB file lands where the GPU jobs will read it
and is never downloaded twice. The Mac only launches the job. Pinned to repo revision 8eaceeb (2026-08-19); the
.pt file's SHA-256 is checked against the value Hugging Face lists for it.
"""
import modal

REPO = "yuanty/fastwam"
REVISION = "8eaceeb24c3cc92ff2a9c9a9d266a4941b836705"
FILES = {  # file -> expected SHA-256 (None: small JSON, not stored with a checksum)
    "libero_optional_idm_2cam224.pt": "26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4",
    "libero_optional_idm_2cam224_dataset_stats.json": None,
}

image = modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub==0.36.0")
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("fastwam-download", image=image)


@app.function(cpu=2.0, memory=4096, timeout=2 * 3600, volumes={"/root/.cache/huggingface": hf_vol})
def download():
    import hashlib
    import os
    import time

    from huggingface_hub import hf_hub_download

    for name, sha in FILES.items():
        t0 = time.time()
        path = hf_hub_download(REPO, name, revision=REVISION)
        size = os.path.getsize(path)
        print(f"{name}: {size / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {path}", flush=True)
        if sha:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            ok = h.hexdigest() == sha
            print(f"  SHA-256 {'OK' if ok else 'MISMATCH ' + h.hexdigest()}", flush=True)
            if not ok:
                raise RuntimeError(f"checksum mismatch for {name}")
    hf_vol.commit()


@app.local_entrypoint()
def main():
    download.remote()
