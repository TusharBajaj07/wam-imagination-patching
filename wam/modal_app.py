"""Run a script from wam/ on a Modal GPU. The Mac only launches; all model code runs remotely.

  modal run wam/modal_app.py --script step0 [--args "..."]

Machine: Debian + Python 3.10, torch 2.7.1+cu128 (FastWAM's pinned version), 1x L40S (48 GB).
Volumes: hf-cache (Fast-WAM checkpoint, from setup/download_ckpt.py), fastwam-assets (Wan VAE, umT5 text encoder,
tokenizer, LIBERO-Goal demos, from setup/download_assets.py), wam-interp-out (results).
"""
import os
from pathlib import Path

import modal

HERE = Path(__file__).parent
# Official FastWAM code (https://github.com/yuantianyuan01/FastWAM, commit 7faa711), used unmodified.
FASTWAM = Path(os.environ.get("FASTWAM_DIR", HERE.parent / "third_party" / "FastWAM"))
GPU = "L40S"

base = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("accelerate==1.12.0", "av==15.1.0", "einops==0.8.1", "hydra-core==1.3.2", "omegaconf==2.3.0",
                 "numpy==2.2.6", "pandas==2.2.3", "pyarrow==24.0.0", "pillow==12.0.0", "safetensors==0.5.3",
                 "transformers==4.49.0", "regex==2025.11.3", "tqdm==4.68.3", "imageio==2.37.0",
                 "imageio-ffmpeg==0.6.0", "huggingface-hub==0.29.2", "ftfy==6.3.1", "sentencepiece==0.2.0",
                 "protobuf==5.29.3", "scipy==1.15.2",
                 # remaining FastWAM imports (pyproject.toml pins), so no module is missing at import time
                 "gitpython==3.1.45", "datasets==4.8.5", "jsonlines==4.0.0", "rich==14.2.0", "termcolor==2.5.0",
                 "packaging==25.0", "torchcodec==0.4.0", "boto3==1.35.99", "wandb==0.23.1", "modelscope==1.34.0")
    .env({"PYTHONPATH": "/root/FastWAM/src:/root/wam",
          "DIFFSYNTH_MODEL_BASE_PATH": "/assets/models",
          "DIFFSYNTH_DOWNLOAD_SOURCE": "huggingface",
          "DIFFSYNTH_SKIP_DOWNLOAD": "true",          # fail loudly instead of downloading anything at run time
          "HF_HUB_OFFLINE": "1"})
)
# Same image + the LIBERO simulator (MuJoCo 3.3.2, as the FastWAM README requires; robosuite; LIBERO from GitHub at a
# pinned commit). Rendering is headless through EGL on the GPU. LIBERO asks for its paths interactively on first import,
# so its config file is written in advance.
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"  # LIBERO master, 2025-03-15
sim_base = (
    base.apt_install("libegl1", "libgl1", "libglvnd0", "libgles2", "libosmesa6", "git")
    .run_commands("git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /root/LIBERO",
                  f"cd /root/LIBERO && git checkout {LIBERO_COMMIT}")
    .pip_install("mujoco==3.3.2", "robosuite==1.4.0", "bddl==1.0.1", "easydict==1.13", "gym==0.25.2", "future==1.0.0",
                 "cloudpickle==3.1.1", "opencv-python-headless==4.11.0.86", "matplotlib==3.10.1")
    .run_commands("pip install --no-deps -e /root/LIBERO",
                  "mkdir -p /root/.libero && printf 'benchmark_root: /root/LIBERO/libero/libero\\n"
                  "bddl_files: /root/LIBERO/libero/libero/bddl_files\\ninit_states: /root/LIBERO/libero/libero/init_files\\n"
                  "datasets: /root/LIBERO/libero/datasets\\nassets: /root/LIBERO/libero/libero/assets\\n' "
                  "> /root/.libero/config.yaml")
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
)


def with_code(img):
    return (img.add_local_dir(str(FASTWAM), "/root/FastWAM", ignore=["**/__pycache__/**", "**/*.egg-info/**", "**/.git/**"])
               .add_local_dir(str(HERE), "/root/wam", ignore=["**/__pycache__/**", "results/**", "logs/**"]))


image, sim_image = with_code(base), with_code(sim_base)
hf_vol = modal.Volume.from_name("hf-cache")
assets = modal.Volume.from_name("fastwam-assets")
out_vol = modal.Volume.from_name("wam-interp-out", create_if_missing=True)
app = modal.App("wam-interp", image=image)


VOLS = {"/root/.cache/huggingface": hf_vol, "/assets": assets, "/out": out_vol}


@app.function(gpu=GPU, cpu=8.0, memory=65536, timeout=4 * 3600, volumes=VOLS)
def run(script: str, args: list):
    _execute(script, args)


@app.function(image=sim_image, gpu=GPU, cpu=8.0, memory=65536, timeout=8 * 3600, volumes=VOLS)
def run_sim(script: str, args: list):
    _execute(script, args)


def _execute(script: str, args: list):
    import importlib
    import subprocess
    import sys

    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout, flush=True)
    sys.path.insert(0, "/root/wam")
    mod = importlib.import_module(script)
    try:
        mod.main(args + ["--out", "/out"])
    except Exception as exc:  # re-raise as a plain error so the Mac (no hydra/torch installed) can print it
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from None
    finally:
        out_vol.commit()


@app.local_entrypoint()
def main(script: str, args: str = "", sim: bool = False):
    (run_sim if sim else run).remote(script, args.split() if args else [])
