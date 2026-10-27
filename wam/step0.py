"""Step 0: checks before any result counts. Every check prints PASS/FAIL with its numbers; report -> /out/step0/.

  modal run wam/modal_app.py --script step0

Checks (see Lesson 4, section 4.2):
  1 load      checkpoint loads; count missing/unexpected weights (load_checkpoint is strict=False)
  2 shapes    cache = 30 layers; tokens per frame = 98; video tokens = 294 (98 current + 196 future); 32 action tokens
  3 faithful  our split prepare/imagine/prefill/act == model.infer_action(mode="idm"), same seed
  4 determ    our pipeline run twice gives identical actions
  5 selfpatch replacing every layer's cache with a copy of itself changes nothing
  6 floor     seed spread: (a) whole pipeline with 4 other seeds, (b) same cache, 4 other action-noise seeds
  7 modes     act-directly (first_frame) vs imagine (idm) on the same state, same seed
  8 timing    seconds per stage
  9 frames    decode the imagined future and save it next to the real future (visual sanity check)
"""
import argparse
import dataclasses
import json
import os

import numpy as np
import torch

import wamlib as W


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--episodes", default="0,1,2")
    ap.add_argument("--t", type=int, default=20)
    a = ap.parse_args(argv)
    out = os.path.join(a.out, "step0")
    os.makedirs(out, exist_ok=True)
    rep = {}

    def log(k, v, ok=None):
        rep[k] = v
        tag = "" if ok is None else ("PASS " if ok else "FAIL ")
        print(f"[{tag}{k}] {json.dumps(v, default=str)[:900]}", flush=True)

    # 1 load -------------------------------------------------------------------------------------------------------
    model, proc, cfg, load = W.load_model_and_processor()
    log("1_load", load, ok=(not load["mot_missing"] and not load["mot_unexpected"]))
    log("gpu_mem_after_load_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
    tasks = W.load_tasks()

    def state_inputs(ep, t):
        e = W.load_episode(ep)
        img = W.to_model_image(e["main"][t], e["wrist"][t], model.device, model.torch_dtype)
        prop = W.normalize_state(e["state"][t], proc)
        prompt = W.PROMPT.format(task=tasks[e["task_index"]])
        return e, img, prop, prompt

    eps = [int(x) for x in a.episodes.split(",")]
    e, img, prop, prompt = state_inputs(eps[0], a.t)
    log("state", {"episode": eps[0], "t": a.t, "length": e["length"], "task": tasks[e["task_index"]],
                  "image": list(img.shape), "proprio_norm": [round(x, 3) for x in prop[0].tolist()]})

    # 2-3 our split pipeline vs the official function ---------------------------------------------------------------
    tm = {}
    t0 = W.cuda_sync_time(); p = W.prepare(model, img, prop, prompt, seed=0); tm["prepare"] = W.cuda_sync_time() - t0
    t0 = W.cuda_sync_time(); lv = W.imagine(model, p); tm["imagine"] = W.cuda_sync_time() - t0
    t0 = W.cuda_sync_time(); K, V, amask, vseq = W.prefill(model, p, lv); tm["prefill"] = W.cuda_sync_time() - t0
    t0 = W.cuda_sync_time(); a_ours = W.act(model, p, K, V, amask); tm["act"] = W.cuda_sync_time() - t0
    log("2_shapes", {"layers": len(K), "k_shape": list(K[0].shape), "v_shape": list(V[0].shape),
                     "tokens_per_frame": p.tokens_per_frame, "video_tokens": vseq,
                     "action_mask_shape": list(amask.shape), "latents_video": list(lv.shape),
                     "action": list(a_ours.shape)},
        ok=(len(K) == 30 and p.tokens_per_frame == 98 and vseq == 294 and tuple(a_ours.shape) == (32, 7)))
    # which keys may the action read? (should be all 294 video + 32 action in imagine mode)
    log("2_action_mask_allowed_keys", {"video_current": int(amask[0, :98].sum()), "video_future": int(amask[0, 98:vseq].sum()),
                                       "action": int(amask[0, vseq:].sum())})

    t0 = W.cuda_sync_time()
    ref = model.infer_action(prompt=prompt, input_image=img, action_horizon=W.ACTION_HORIZON,
                             num_video_frames=W.NUM_VIDEO_FRAMES, proprio=prop, num_inference_steps=W.NUM_STEPS,
                             sigma_shift=W.SIGMA_SHIFT, seed=0, rand_device="cpu", action_infer_mode="idm")
    tm["official_idm_total"] = W.cuda_sync_time() - t0
    a_ref = ref["action"].to(a_ours.device).float()
    log("3_faithful", {"max_abs_diff_vs_official": float((a_ours - a_ref).abs().max()),
                       "dist_vs_official": W.dist(a_ours, a_ref),
                       "video_latent_max_abs_diff": float((lv - ref["video_latents"].to(lv.device)).abs().max())},
        ok=W.dist(a_ours, a_ref) < 1e-3)

    # 4 determinism --------------------------------------------------------------------------------------------------
    p2 = W.prepare(model, img, prop, prompt, seed=0)
    K2, V2, _, _ = W.prefill(model, p2, W.imagine(model, p2))
    a_again = W.act(model, p2, K2, V2, amask)
    log("4_determinism", {"dist_rerun": W.dist(a_again, a_ours)}, ok=W.dist(a_again, a_ours) == 0.0)

    # 5 self-patch -----------------------------------------------------------------------------------------------------
    sl = slice(0, vseq)
    Ks, Vs = W.edit_slice(K, range(30), sl, K), W.edit_slice(V, range(30), sl, V)
    a_self = W.act(model, p, Ks, Vs, amask)
    log("5_selfpatch", {"dist": W.dist(a_self, a_ours)}, ok=W.dist(a_self, a_ours) == 0.0)

    # 6 noise floor -----------------------------------------------------------------------------------------------------
    full, actonly = [], []
    for s in (1, 2, 3, 4):
        ps = W.prepare(model, img, prop, prompt, seed=s)
        Ks_, Vs_, _, _ = W.prefill(model, ps, W.imagine(model, ps))
        full.append(W.dist(W.act(model, ps, Ks_, Vs_, amask), a_ours))
        pa = dataclasses.replace(p, latents_action0=ps.latents_action0)        # same cache, other action noise
        actonly.append(W.dist(W.act(model, pa, K, V, amask), a_ours))
    a_true = W.normalize_actions(e["action"][a.t:a.t + 32], proc).to(a_ours.device)
    log("6_noise_floor", {"seed_spread_full_pipeline": [round(x, 4) for x in full],
                          "seed_spread_action_noise_only": [round(x, 4) for x in actonly],
                          "dist_to_demo_actions": round(W.dist(a_ours, a_true), 4),
                          "norm_of_action_chunk": round(float(torch.linalg.vector_norm(a_ours)), 4)})

    # 7 modes -----------------------------------------------------------------------------------------------------------
    ff = model.infer_action(prompt=prompt, input_image=img, action_horizon=W.ACTION_HORIZON, proprio=prop,
                            num_inference_steps=W.NUM_STEPS, sigma_shift=W.SIGMA_SHIFT, seed=0, rand_device="cpu",
                            action_infer_mode="first_frame", num_video_frames=W.NUM_VIDEO_FRAMES)
    a_ff = ff["action"].to(a_ours.device).float()
    log("7_modes", {"dist_firstframe_vs_idm": round(W.dist(a_ff, a_ours), 4),
                    "dist_firstframe_to_demo": round(W.dist(a_ff, a_true), 4)})

    # 8 timing ----------------------------------------------------------------------------------------------------------
    log("8_timing_s", {k: round(v, 3) for k, v in tm.items()})
    log("gpu_mem_peak_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

    # 9 frames: imagined vs real future ----------------------------------------------------------------------------------
    idx = [min(a.t + 4 * i, e["length"] - 1) for i in range(9)]
    real = W.encode_clip(model, e["main"][idx], e["wrist"][idx])
    imagined = W.decode_frames(model, lv)
    realdec = W.decode_frames(model, real)
    for i, (im, rd) in enumerate(zip(imagined, realdec)):
        im.save(f"{out}/imagined_{i}.png")
        rd.save(f"{out}/real_{i}.png")
    lat_err = float(torch.linalg.vector_norm((lv - real.to(lv.dtype))[:, :, 1:].float()) /
                    torch.linalg.vector_norm(real[:, :, 1:].float()))
    log("9_frames", {"n_decoded": len(imagined), "future_latent_rel_error_imagined_vs_real": round(lat_err, 4),
                     "current_latent_max_abs_diff": float((lv[:, :, 0] - real[:, :, 0].to(lv.dtype)).abs().max())})

    with open(f"{out}/report.json", "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print("STEP0 DONE", flush=True)
