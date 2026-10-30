"""Closed-loop flip test on LIBERO-Goal (all 10 tasks share one scene, so another task's goal can be checked on the
same simulator state).

  modal run wam/modal_app.py --sim --script flip_test --args "--cond clean --inits 5"

For task A and B = (A+5) % 10, from LIBERO's fixed starting states 0..inits-1, imagine mode, seed 42:
  clean    imagine with A, act with A                      (the normal model; also saves every decision's inputs)
  futureB  imagine with B, act with A   -- future tokens 98-293 swapped in all 30 layers at every decision
  textB    imagine with A, act with B   -- the action reads B's instruction, but A's imagined future
  noimagine  first-frame mode with A    -- no imagined future at all (official action_infer_mode="first_frame")
Recorded per episode: A succeeded? (the env's own check), B's goal reached? (B's goal checked on the same state at
every decision), steps, decisions. --save-states stores each decision's inputs (for the held-out open-loop pilot).

Phase 1 options (futureB only): --layers swaps the future in some layers only ("21-30", "26,28"; 1-based);
--alpha blends instead of swapping (future = (1-alpha)*A + alpha*B in the cache). --seed changes env.seed() and the
model noise. --video N saves the main camera (every 2 steps, 128 px) for starts 0..N-1 to flip_test/videos/.
"""
import argparse
import json
import os
import time

import numpy as np

import sim as S
import wamlib as W


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--cond", choices=["clean", "futureB", "textB", "noimagine"], required=True)
    ap.add_argument("--inits", type=int, default=5)
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--file", default=None, help="results file name (default: <cond>.jsonl); parallel parts use their own")
    ap.add_argument("--skip", default="", help="file(s) whose (task, init) episodes are already done, comma-separated")
    ap.add_argument("--layers", default="1-30", help='1-based layers for the future swap, e.g. "21-30" or "26,28"')
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=S.SEED)
    ap.add_argument("--video", type=int, default=0, help="save videos for starts 0..N-1")
    ap.add_argument("--save-states", action="store_true")
    ap.add_argument("--stop-on-b", action="store_true", help="end an episode once B's goal is seen (A success then False)")
    a = ap.parse_args(argv)
    out = os.path.join(a.out, "flip_test")
    os.makedirs(out, exist_ok=True)
    if a.save_states:
        os.makedirs(os.path.join(out, "clean_states"), exist_ok=True)
    if a.video:
        os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    layers = sorted({l - 1 for part in a.layers.split(",") for l in
                     (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [int(part)])})
    assert all(0 <= l < 30 for l in layers), layers
    if a.cond != "futureB":
        assert a.layers == "1-30" and a.alpha == 1.0, "--layers/--alpha only apply to futureB"

    fname = a.file or f"{a.cond}.jsonl"
    done = set()
    for f in filter(None, a.skip.split(",")):
        if os.path.exists(f"{out}/{f}"):
            done |= {(r["task_id"], r["init"]) for r in map(json.loads, open(f"{out}/{f}"))}
    model, proc, _, load = W.load_model_and_processor()
    assert not load["mot_missing"] and not load["mot_unexpected"]
    t_start = time.time()
    for A in map(int, a.tasks.split(",")):
        B = (A + 5) % 10
        for i in range(a.inits):
            if (A, i) in done:
                continue
            t0 = time.time()
            r, states = S.run_episode(model, proc, "libero_goal", A, i,
                                      mode="first_frame" if a.cond == "noimagine" else "idm",
                                      imagine_task=B if a.cond == "futureB" else None,
                                      act_task=B if a.cond == "textB" else None,
                                      check_task=B, keep_states=a.save_states, swap_layers=layers,
                                      alpha=a.alpha, seed=a.seed, video_every=2 if i < a.video else 0, stop_on_check=a.stop_on_b)
            frames = r.pop("_frames", None)
            r.update(cond=a.cond, task_B=B, episode_s=round(time.time() - t0, 1))
            if frames is not None:
                np.savez_compressed(f"{out}/videos/{fname[:-6]}_task{A}_init{i}.npz", frames=frames,
                                    success=r["success"], b_step=-1 if r["check_success_step"] is None
                                    else r["check_success_step"])
            with open(f"{out}/{fname}", "a") as f:
                f.write(json.dumps(r) + "\n")
            if states:
                np.savez_compressed(f"{out}/clean_states/task{A}_init{i}.npz",
                                    main=np.stack([s["main"] for s in states]),
                                    wrist=np.stack([s["wrist"] for s in states]),
                                    state=np.stack([s["state"] for s in states]),
                                    t=np.array([s["t"] for s in states]),
                                    a_norm=np.stack([s["a_norm"] for s in states]),
                                    task=A, init=i, success=r["success"])
            print(f"[{a.cond}] A={A} B={B} init={i} A_success={r['success']} B_reached_at={r['check_success_step']} "
                  f"steps={r['steps']} {r['episode_s']}s  err={r['check_error']}  [{fname}]  total {time.time() - t_start:.0f}s",
                  flush=True)
    print("FLIP DONE", flush=True)
