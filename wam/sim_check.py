"""Simulator setup check (closed loop), run in the sim image:

  modal run wam/modal_app.py --sim --script sim_check

1 LIBERO imports, the env builds, renders headless (EGL), 30 no-op steps; save both camera views; time per step.
2 Episodes on LIBERO-Goal task 0 ("open the middle drawer of the cabinet"), starting positions 0..n-1, in imagine
  mode and in act-directly mode: success, steps, seconds per planning decision, seconds per episode.
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

import sim as S
import wamlib as W


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--task", type=int, default=0)
    a = ap.parse_args(argv)
    out = os.path.join(a.out, "sim_check")
    os.makedirs(out, exist_ok=True)
    rep = {}

    t0 = time.time()
    task, inits = S.suite_task("libero_goal", a.task)
    env, desc = S.get_libero_env(task, S.RES, S.SEED)
    env.reset()
    obs = env.set_init_state(inits[0])
    rep["env_build_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    for _ in range(30):
        obs, _, done, _ = env.step(S.get_libero_dummy_action())
    rep["step_ms"] = round((time.time() - t0) / 30 * 1000, 1)
    imgs = S.get_libero_image(obs)
    Image.fromarray(np.concatenate([imgs["image"], imgs["wrist_image"]], axis=1)).save(f"{out}/render_task{a.task}.png")
    rep.update(task=desc, n_init_states=len(inits), obs_keys=sorted(obs.keys()),
               image_shape=list(imgs["image"].shape), eef_pos=obs["robot0_eef_pos"].round(3).tolist())
    env.close()
    print("[1_env]", json.dumps(rep), flush=True)

    model, proc, _, load = W.load_model_and_processor()
    assert not load["mot_missing"] and not load["mot_unexpected"]
    eps = []
    for mode in ("idm", "first_frame"):
        for i in range(a.n):
            t0 = time.time()
            r, _ = S.run_episode(model, proc, "libero_goal", a.task, i, mode=mode)
            r["episode_s"] = round(time.time() - t0, 1)
            eps.append(r)
            print("[2_episode]", json.dumps(r), flush=True)
    with open(f"{out}/report.json", "w") as f:
        json.dump({"env": rep, "episodes": eps}, f, indent=1)
    print("SIM CHECK DONE", flush=True)
