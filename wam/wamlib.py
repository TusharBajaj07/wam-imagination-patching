"""Core library for the Fast-WAM imagination study. Runs inside the Modal GPU container (see modal_app.py).

Everything here mirrors FastWAM's own inference code; where we copy logic, the source line is cited so the copy can be
checked. The one structural change: FastWAMIDM.infer_action (fastwam_idm.py:365-589) is split into
    prepare() -> imagine() -> prefill() -> act()
so that the per-layer video KV cache can be edited between prefill() and act(). Step 0 verifies that running the four
pieces unedited reproduces model.infer_action(..., action_infer_mode="idm") exactly.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

FASTWAM = "/root/FastWAM"
CKPT_DIR = "/root/.cache/huggingface/hub/models--yuanty--fastwam/snapshots/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705"
CKPT = f"{CKPT_DIR}/libero_optional_idm_2cam224.pt"
STATS = f"{CKPT_DIR}/libero_optional_idm_2cam224_dataset_stats.json"
SIGMA_SHIFT = 1.0        # README: the released Optional-IDM checkpoint is evaluated with EVALUATION.sigma_shift=1.0
NUM_STEPS = 10           # configs/train.yaml eval_num_inference_steps
NUM_VIDEO_FRAMES = 9     # (num_frames 33 - 1) // action_video_freq_ratio 4 + 1
ACTION_HORIZON = 32
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"  # robot_video_dataset.py:23


# ----------------------------------------------------------------------------------------------------------------- load
def load_model_and_processor(device: str = "cuda"):
    """Build the model exactly as eval_libero_single.py does (sim_libero.yaml + task=libero_optional_idm_2cam224_1e-4)."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    import fastwam.utils.config_resolvers  # noqa: F401  (registers OmegaConf resolvers used by the configs)
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    with initialize_config_dir(config_dir=f"{FASTWAM}/configs", version_base="1.3"):
        # redirect_common_files=false: use the official Wan-AI .pth files (the "converted safetensors" repo the default
        # points to is not on Hugging Face). The loader's hash registry is written for exactly these .pth files.
        cfg = compose(config_name="sim_libero", overrides=["task=libero_optional_idm_2cam224_1e-4", f"ckpt={CKPT}",
                                                           "model.redirect_common_files=false"])
    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    t_build = time.time() - t0

    # load_checkpoint() uses strict=False (fastwam.py:1214), so count missing/unexpected keys ourselves.
    payload = torch.load(CKPT, map_location="cpu")
    res = model.mot.load_state_dict(payload["mot"], strict=False)
    load_report = {
        "payload_keys": sorted(payload.keys()),
        "mot_tensors_in_ckpt": len(payload["mot"]),
        "mot_missing": list(res.missing_keys),
        "mot_unexpected": list(res.unexpected_keys),
    }
    if model.proprio_encoder is not None and "proprio_encoder" in payload:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        load_report["proprio_encoder"] = "loaded strict"
    else:
        load_report["proprio_encoder"] = "NOT loaded"
    del payload
    model = model.to(device).eval()

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(STATS))
    load_report["build_seconds"] = round(t_build, 1)
    load_report["params_video_expert"] = sum(p.numel() for p in model.video_expert.parameters())
    load_report["params_action_expert"] = sum(p.numel() for p in model.action_expert.parameters())
    load_report["dtype"] = str(model.torch_dtype)
    return model, processor, cfg, load_report


# --------------------------------------------------------------------------------------------------------------- data
DATA = "/assets/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot"   # LeRobot v2.1, 433 episodes, 20 fps


def load_tasks() -> dict[int, str]:
    with open(f"{DATA}/meta/tasks.jsonl") as f:
        return {d["task_index"]: d["task"] for d in map(json.loads, f)}


def decode_video(path: str) -> np.ndarray:
    import av
    with av.open(path) as c:
        return np.stack([fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)])


def load_episode(ep: int) -> dict:
    """One demonstration: frames from both cameras [T,512,512,3], states [T,8], actions [T,7], task index."""
    import pandas as pd
    df = pd.read_parquet(f"{DATA}/data/chunk-000/episode_{ep:06d}.parquet")
    main = decode_video(f"{DATA}/videos/chunk-000/observation.images.image/episode_{ep:06d}.mp4")
    wrist = decode_video(f"{DATA}/videos/chunk-000/observation.images.wrist_image/episode_{ep:06d}.mp4")
    n = min(len(df), len(main), len(wrist))
    return {"ep": ep, "task_index": int(df["task_index"].iloc[0]), "main": main[:n], "wrist": wrist[:n],
            "state": np.stack(df["observation.state"].to_numpy())[:n].astype(np.float32),
            "action": np.stack(df["action"].to_numpy())[:n].astype(np.float32), "length": n}


def episodes_by_task() -> dict[int, list[int]]:
    with open(f"{DATA}/meta/episodes.jsonl") as f:
        eps = [json.loads(line) for line in f]
    tasks = {v: k for k, v in load_tasks().items()}
    out: dict[int, list[int]] = {}
    for e in eps:
        out.setdefault(tasks[e["tasks"][0]], []).append(e["episode_index"])
    return out


@torch.no_grad()
def encode_clip(model, main: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
    """VAE-encode 9 real frames (current + 8 future) into [1,48,3,14,28] latents, the same way the current frame is
    encoded (fastwam.py:263-273, with T=9 instead of 1)."""
    frames = [to_model_image(m, w, model.device, model.torch_dtype)[0] for m, w in zip(main, wrist)]  # 9 x [3,224,448]
    video = torch.stack(frames, dim=1).unsqueeze(0)                                                   # [1,3,9,224,448]
    return model.vae.model.encode(video, model.vae.scale)


@torch.no_grad()
def decode_frames(model, latents: torch.Tensor):
    return model._decode_latents(latents)   # list of PIL images, fastwam.py:275


# ------------------------------------------------------------------------------------------------------------- inputs
def to_model_image(main_rgb: np.ndarray, wrist_rgb: np.ndarray, device, dtype) -> torch.Tensor:
    """Each camera resized to 224x224, concatenated side by side, scaled to [-1, 1] (eval_libero_single.py:185-236)."""
    def resize(a):
        img = Image.fromarray(a)
        scale = max(224 / img.width, 224 / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), resample=Image.BILINEAR)
        left, top = max((img.width - 224) // 2, 0), max((img.height - 224) // 2, 0)
        return np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.uint8)

    rgb = np.concatenate([resize(main_rgb), resize(wrist_rgb)], axis=1)       # [224, 448, 3]
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return x * (2.0 / 255.0) - 1.0


def normalize_state(state8: np.ndarray, processor) -> torch.Tensor:
    """eval_libero_single.py:167-182."""
    key = processor.shape_meta["state"][0]["key"]
    batch = {"state": {key: torch.as_tensor(state8, dtype=torch.float32).unsqueeze(0)}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][key]


def normalize_actions(actions: np.ndarray, processor) -> torch.Tensor:
    """Demo actions [T,7] into the model's normalized action space (for comparing with predictions)."""
    key = processor.shape_meta["action"][0]["key"]
    norm = processor.normalizer.normalizers["action"][key]
    return norm.forward(torch.as_tensor(actions, dtype=torch.float32).unsqueeze(0))[0]


# ------------------------------------------------------------------------------------------- split imagine-then-act
@dataclass
class Prepared:
    first_frame_latents: torch.Tensor
    latents_video0: torch.Tensor     # initial noise for the video (frame 0 already set to the current frame)
    latents_action0: torch.Tensor    # initial noise for the action (same seed => same noise in every edit)
    context: torch.Tensor
    context_mask: torch.Tensor
    tokens_per_frame: int


@torch.no_grad()
def prepare(model, image, proprio, prompt, seed, rand_device="cpu") -> Prepared:
    """fastwam_idm.py:384-467: noise, VAE-encode the current frame, encode the prompt, append proprio."""
    _, _, h, w = image.shape
    latent_t = (NUM_VIDEO_FRAMES - 1) // model.vae.temporal_downsample_factor + 1
    lh, lw = h // model.vae.upsampling_factor, w // model.vae.upsampling_factor
    gv = torch.Generator(device=rand_device).manual_seed(seed)
    ga = torch.Generator(device=rand_device).manual_seed(seed)
    lv = torch.randn((1, model.vae.model.z_dim, latent_t, lh, lw), generator=gv, device=rand_device,
                     dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)
    la = torch.randn((1, ACTION_HORIZON, model.action_expert.action_dim), generator=ga, device=rand_device,
                     dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)
    ffl = model._encode_input_image_latents_tensor(input_image=image.to(model.device, model.torch_dtype), tiled=False)
    lv[:, :, 0:1] = ffl.clone()
    context, context_mask = model.encode_prompt(prompt)
    proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
    context, context_mask = model._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)
    ph, pw = int(model.video_expert.patch_size[1]), int(model.video_expert.patch_size[2])
    return Prepared(ffl, lv, la, context, context_mask, (lh // ph) * (lw // pw))


@torch.no_grad()
def imagine(model, p: Prepared) -> torch.Tensor:
    """Stage 1, fastwam_idm.py:469-506: denoise the video alone; frame 0 is re-clamped to the real frame every step."""
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    lv = p.latents_video0.clone()
    seq = (lv.shape[2] // int(model.video_expert.patch_size[0])) * p.tokens_per_frame
    mask = model.video_expert.build_video_to_video_mask(video_seq_len=seq, video_tokens_per_frame=p.tokens_per_frame,
                                                        device=model.device)
    ts, ds = model.infer_video_scheduler.build_inference_schedule(num_inference_steps=NUM_STEPS, device=model.device,
                                                                  dtype=lv.dtype, shift_override=SIGMA_SHIFT)
    for t, d in zip(ts, ds):
        pred = model._denoise_video(latents_video=lv, timestep_video=t.unsqueeze(0).to(lv.dtype), context=p.context,
                                    context_mask=p.context_mask, video_self_attn_mask=mask,
                                    fuse_vae_embedding_in_latents=fuse)
        lv = model.infer_video_scheduler.step(pred, d, lv)
        lv[:, :, 0:1] = p.first_frame_latents.clone()
    return lv


@torch.no_grad()
def prefill(model, p: Prepared, latents_video: torch.Tensor):
    """Stage 2a, fastwam_idm.py:508-560: run the clean video once (timestep 0) and keep every layer's K and V."""
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    t0 = torch.zeros((latents_video.shape[0],), dtype=latents_video.dtype, device=model.device)
    (tok, _t, tmod, vctx, vctx_mask, freqs, _f, _h, _w, tpf) = model.video_expert.prepare(
        x=latents_video, timestep=t0, context=p.context, context_mask=p.context_mask, action=None,
        fuse_vae_embedding_in_latents=fuse)
    vseq = int(tok.shape[1])
    mask = model._build_mot_attention_mask(video_seq_len=vseq, action_seq_len=p.latents_action0.shape[1],
                                           video_tokens_per_frame=tpf, device=tok.device)
    k, v = model.mot.prefill_video_cache_tensor(video_tokens=tok, video_freqs=freqs, video_t_mod=tmod,
                                                video_context=vctx, video_context_mask=vctx_mask,
                                                video_attention_mask=mask[:vseq, :vseq])
    return list(k), list(v), mask[vseq:, :], vseq


@torch.no_grad()
def act(model, p: Prepared, cache_k, cache_v, action_mask) -> torch.Tensor:
    """Stage 2b, fastwam_idm.py:562-589: 10 action denoising steps reading the (possibly edited) cache.
    Returns the normalized action chunk [32, 7] in float32 (before denormalization)."""
    la = p.latents_action0.clone()
    ts, ds = model.infer_action_scheduler.build_inference_schedule(num_inference_steps=NUM_STEPS, device=model.device,
                                                                   dtype=la.dtype, shift_override=SIGMA_SHIFT)
    for t, d in zip(ts, ds):
        pred = model._denoise_action_with_video_cache(latents_action=la, timestep_action=t.unsqueeze(0).to(la.dtype),
                                                      context=p.context, context_mask=p.context_mask,
                                                      video_cache_k=cache_k, video_cache_v=cache_v,
                                                      action_attention_mask=action_mask)
        la = model.infer_action_scheduler.step(pred, d, la)
    return la[0].float()


# ------------------------------------------------------------------------------------------------ attention recorder
class AttnRecorder:
    """Wraps the one attention function the MoT uses (mot.py imports flash_attention from wan_video_dit.py).
    While `on`, every call made by the ACTION expert (32 query tokens) during act() is recorded. Calls arrive in order
    step 0 layer 0..29, step 1 layer 0..29, ..., so call index -> (step, layer) = divmod(idx, 30).

    Per call we recompute the attention weights in fp32 (SDPA does not return them):
        alpha = softmax(q k^T / sqrt(128) + mask)              [heads=24, queries=32, keys=326]
    and for each key group g (current 0-97, future 98-293, action 294-325):
        M[g]  = mean over queries of sum_{j in g} alpha_ij      (share of attention, sums to 1 over groups)
        O[g]  = mean over queries of || sum_{j in g} alpha_ij v_j ||   (size of what group g passes on, per head)
    and check that the three group outputs add up to the model's own attention output (reconstruction error)."""
    GROUPS = {"cur": slice(0, 98), "fut": slice(98, 294), "act": slice(294, 326)}

    def __init__(self):
        import fastwam.models.wan22.mot as mot
        self.mot, self.orig = mot, mot.flash_attention
        self.on, self.calls = False, []
        mot.flash_attention = self._wrapped

    def _wrapped(self, q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
        out = self.orig(q, k, v, num_heads, ctx_mask=ctx_mask, compatibility_mode=compatibility_mode)
        if self.on and q.shape[1] == ACTION_HORIZON and k.shape[1] == 326:
            H = num_heads
            qh = q.float().view(1, -1, H, 128).transpose(1, 2)            # [1,24,32,128]
            kh = k.float().view(1, -1, H, 128).transpose(1, 2)            # [1,24,326,128]
            vh = v.float().view(1, -1, H, 128).transpose(1, 2)
            s = qh @ kh.transpose(-1, -2) / 128 ** 0.5
            if ctx_mask is not None:
                s = s.masked_fill(~ctx_mask.to(s.device).bool(), float("-inf"))
            a = s.softmax(-1)                                              # [1,24,32,326]
            M, O, total = [], [], 0
            for g in self.GROUPS.values():
                M.append(a[..., g].sum(-1).mean(-1)[0])                   # [24]
                og = a[..., g] @ vh[:, :, g]                              # [1,24,32,128]
                O.append(og.norm(dim=-1).mean(-1)[0])                     # [24]
                total = total + og
            ref = out.float().view(1, -1, H, 128).transpose(1, 2)
            err = float((total - ref).abs().max() / ref.abs().max())
            self.calls.append((torch.stack(M).cpu(), torch.stack(O).cpu(), err))
        return out

    def take(self):
        """Return M, O as [steps=10, layers=30, groups=3, heads=24] and the worst reconstruction error; reset."""
        assert len(self.calls) == NUM_STEPS * 30, len(self.calls)
        M = torch.stack([c[0] for c in self.calls]).view(NUM_STEPS, 30, 3, -1)
        O = torch.stack([c[1] for c in self.calls]).view(NUM_STEPS, 30, 3, -1)
        err = max(c[2] for c in self.calls)
        self.calls = []
        return M.numpy(), O.numpy(), err


# -------------------------------------------------------------------------------------------------------------- edits
BANDS = {"all": range(0, 30), "L1-10": range(0, 10), "L11-20": range(10, 20), "L21-30": range(20, 30)}


def edit_slice(cache, layers, sl, source):
    """Return a new cache list where tokens `sl` of the chosen layers come from `source` (a list of tensors, or a
    callable layer_idx -> tensor shaped like cache[i][:, sl]). Untouched layers are shared, not copied."""
    out = list(cache)
    for i in layers:
        t = cache[i].clone()
        t[:, sl] = source(i) if callable(source) else source[i][:, sl]
        out[i] = t
    return out


def knockout_mask(action_mask: torch.Tensor, tpf: int, vseq: int) -> torch.Tensor:
    """Forbid action -> imagined-future attention (columns tpf..vseq). The same mask is used by all 30 layers
    (mot.py:561), so this is an all-layer knockout; the softmax renormalizes over the remaining keys."""
    m = action_mask.clone()
    m[:, tpf:vseq] = False
    return m


def dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Euclidean distance between two [32,7] normalized action chunks, i.e. 224 numbers."""
    return float(torch.linalg.vector_norm(a - b))


def cuda_sync_time():
    torch.cuda.synchronize()
    return time.time()
