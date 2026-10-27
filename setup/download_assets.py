"""Setup step 2: download what Fast-WAM needs besides its checkpoint, into the Modal Volume "fastwam-assets".

  modal run setup/download_assets.py

CPU-only Modal machine; nothing lands on the Mac. Layout inside the volume (matches FastWAM's
DIFFSYNTH_MODEL_BASE_PATH convention, <base>/<model_id>/<file>):
  /assets/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth                      2.82 GB  video encoder (VAE)
  /assets/models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth     11.36 GB  umT5 text encoder
  /assets/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/*                  ~21 MB  tokenizer
  /assets/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot/           0.84 GB  LIBERO-Goal demos (extracted)
The 20 GB Wan2.2 video DiT is NOT downloaded: the Fast-WAM checkpoint replaces it (skip_dit_load_from_pretrain).
"""
import modal

WAN22 = ("Wan-AI/Wan2.2-TI2V-5B", "921dbaf3f1674a56f47e83fb80a34bac8a8f203e")
WAN21 = ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a")
LIBERO = ("yuanty/LIBERO-fastwam", "ee018b997c430bb12b5bf3c892d744798c5a2f91")

image = modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub==0.36.0")
assets = modal.Volume.from_name("fastwam-assets", create_if_missing=True)
app = modal.App("fastwam-assets-download", image=image)


@app.function(cpu=4.0, memory=8192, timeout=3 * 3600, volumes={"/assets": assets})
def download():
    import os
    import subprocess
    import time

    from huggingface_hub import hf_hub_download, snapshot_download

    def fetch(repo, rev, filename, dest, repo_type="model"):
        t0 = time.time()
        path = hf_hub_download(repo, filename, revision=rev, repo_type=repo_type, local_dir=dest)
        print(f"{repo}/{filename}: {os.path.getsize(path) / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
        return path

    base = "/assets/models"
    fetch(WAN22[0], WAN22[1], "Wan2.2_VAE.pth", f"{base}/{WAN22[0]}")
    fetch(WAN22[0], WAN22[1], "models_t5_umt5-xxl-enc-bf16.pth", f"{base}/{WAN22[0]}")
    snapshot_download(WAN21[0], revision=WAN21[1], allow_patterns="google/umt5-xxl/*", local_dir=f"{base}/{WAN21[0]}")
    print("tokenizer:", sorted(os.listdir(f"{base}/{WAN21[0]}/google/umt5-xxl")), flush=True)
    assets.commit()

    data = "/assets/data/libero_mujoco3.3.2"
    os.makedirs(data, exist_ok=True)
    tar = fetch(LIBERO[0], LIBERO[1], "libero_goal_no_noops_lerobot.tar.gz", "/tmp/libero", repo_type="dataset")
    subprocess.run(["tar", "-xzf", tar, "-C", data], check=True)
    for root, dirs, files in os.walk(data):
        depth = root[len(data):].count(os.sep)
        if depth <= 3:
            print("  " * depth + os.path.basename(root) + "/", f"({len(files)} files)", files[:4], flush=True)
    assets.commit()


@app.local_entrypoint()
def main():
    download.remote()
