"""Phase 2 pilot (open loop): replace the imagined future inside the KV cache and measure how much the action changes.

  modal run wam/modal_app.py --script pilot_future_swap --args "--eps-per-task 2 --t-per-ep 5"

States: LIBERO-Goal demonstration frames (training data; a pipeline pilot, NOT a paper result -- the paper uses
closed-loop rollouts). 10 tasks x eps-per-task episodes x t-per-ep timesteps, t spread evenly over [0, length-33].

Per state, everything uses seed 0, so the action's starting noise is identical in every run and any change is caused
by the edit. The cache is 30 layers x [1, 294, 3072] for K and for V; tokens 0-97 = current frame, 98-293 = future.

Edits (rows of the results table):
  future_donor   tokens 98-293 <- the same slice from a state of a DIFFERENT task (paired: state i <-> i+N/2)
  future_mean    tokens 98-293 <- position-wise mean of that slice over all pilot states (information-free future)
  future_noise   tokens 98-293 <- Gaussian noise with the slice's own per-channel mean and std (garbage)
  future_static  re-prefill with a "nothing moves" video (current frame x9, VAE-encoded); take its tokens 98-293
  future_true    re-prefill with the REAL next frames from the demo (VAE-encoded); take its tokens 98-293
  future_knockout  forbid action->future attention in all layers (future present but unreadable)
  current_donor  tokens 0-97 <- the paired state's current frame (comparison: how much does the PRESENT matter?)
Each cache edit is applied to bands: all 30 layers, layers 1-10, 11-20, 21-30 (knockout: all layers only).
References: act-directly mode (official first_frame), and 2 other action-noise seeds with the clean cache (noise floor).
"""
import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

import wamlib as W

FUT = slice(98, 294)
CUR = slice(0, 98)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--eps-per-task", type=int, default=2)
    ap.add_argument("--t-per-ep", type=int, default=5)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0, help="seed for the video AND action starting noise")
    ap.add_argument("--single-layer", action="store_true", help="also swap the future in each single layer 1..30")
    ap.add_argument("--attn", action="store_true", help="record action->video attention during the clean run (RQ1)")
    ap.add_argument("--source", choices=["demo", "rollouts"], default="demo",
                    help="demo = LIBERO-Goal training frames; rollouts = fresh closed-loop states from flip_test clean")
    ap.add_argument("--per-ep", type=int, default=2, help="rollouts: decisions sampled per episode")
    a = ap.parse_args(argv)
    out = os.path.join(a.out, f"pilot_future_swap_{a.tag}")
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    model, proc, _, load = W.load_model_and_processor()
    assert not load["mot_missing"] and not load["mot_unexpected"], load
    rec = W.AttnRecorder() if a.attn else None
    attn_M, attn_O, attn_err = [], [], []
    tasks = W.load_tasks()
    by_task = W.episodes_by_task()

    # ---- choose states: ordered by task so state i and i+N/2 always belong to different tasks
    states, episodes = [], {}
    if a.source == "demo":
        for task in sorted(by_task):
            for ep in by_task[task][: a.eps_per_task]:
                e = W.load_episode(ep)
                episodes[ep] = e
                for t in np.linspace(0, e["length"] - 33, a.t_per_ep).round().astype(int):
                    states.append((task, ep, int(t)))
    else:  # fresh states saved by flip_test.py --cond clean (images already rotated like training frames)
        import glob
        files = sorted(glob.glob(f"{a.out}/flip_test/clean_states/task*_init*.npz"),
                       key=lambda f: (int(np.load(f)["task"]), int(np.load(f)["init"])))
        for f in files:
            z = np.load(f)
            ep = os.path.basename(f)[:-4]
            episodes[ep] = {"main": z["main"], "wrist": z["wrist"], "state": z["state"], "action": None,
                            "length": len(z["t"]), "success": bool(z["success"])}
            for t in np.linspace(0, len(z["t"]) - 1, a.per_ep + 2)[1:-1].round().astype(int):
                states.append((int(z["task"]), ep, int(t)))
    N = len(states)
    print(f"{N} states from {len(episodes)} episodes", flush=True)

    def inputs(i):
        task, ep, t = states[i]
        e = episodes[ep]
        img = W.to_model_image(e["main"][t], e["wrist"][t], model.device, model.torch_dtype)
        prop = W.normalize_state(e["state"][t], proc)
        p = W.prepare(model, img, prop, W.PROMPT.format(task=tasks[task]), seed=a.seed)
        return e, t, img, prop, p

    # ---- phase A: imagine once per state; keep the cache on CPU; running mean of the future slice
    caches, clean, sums = [], [], None
    for i in range(N):
        e, t, img, prop, p = inputs(i)
        K, V, amask, vseq = W.prefill(model, p, W.imagine(model, p))
        if rec:
            rec.on = True
        clean.append(W.act(model, p, K, V, amask).cpu())
        if rec:
            rec.on = False
            M_, O_, err_ = rec.take()
            attn_M.append(M_); attn_O.append(O_); attn_err.append(err_)
        caches.append(([k.cpu() for k in K], [v.cpu() for v in V]))
        fk = [k[:, FUT].float() for k in K]
        fv = [v[:, FUT].float() for v in V]
        sums = (fk, fv) if sums is None else ([s + x for s, x in zip(sums[0], fk)], [s + x for s, x in zip(sums[1], fv)])
        if i % 10 == 0:
            print(f"A {i}/{N}  {time.time() - t_start:.0f}s", flush=True)
    meanK = [(s / N).to(torch.bfloat16) for s in sums[0]]
    meanV = [(s / N).to(torch.bfloat16) for s in sums[1]]

    # ---- phase B: edits
    rows = []
    gen = torch.Generator(device="cuda").manual_seed(1234 + a.seed)
    for i in range(N):
        task, ep, t = states[i]
        e, t, img, prop, p = inputs(i)
        K = [k.cuda() for k in caches[i][0]]
        V = [v.cuda() for v in caches[i][1]]
        j = (i + N // 2) % N
        while states[j][0] == task:                       # donor must be a different task
            j = (j + 1) % N
        Kd = [k.cuda() for k in caches[j][0]]
        Vd = [v.cuda() for v in caches[j][1]]
        a0 = clean[i].cuda()
        a_true = None if e["action"] is None else W.normalize_actions(e["action"][t:t + 32], proc).cuda()
        a_donor = clean[j].cuda()
        r = {"i": i, "task": task, "episode": ep, "t": t, "donor_i": j, "donor_task": states[j][0],
             "clean_to_donor_action": W.dist(a0, a_donor)}
        if a_true is not None:
            r["clean_to_demo"] = W.dist(a0, a_true)

        def run(name, Ke, Ve, mask=amask):
            a1 = W.act(model, p, Ke, Ve, mask)
            r[name] = W.dist(a1, a0)
            if a_true is not None:
                r[name + "__to_demo"] = W.dist(a1, a_true)
            if "donor" in name:
                r[name + "__to_donor_action"] = W.dist(a1, a_donor)

        # references
        ff = model.infer_action(prompt=W.PROMPT.format(task=tasks[task]), input_image=img,
                                action_horizon=W.ACTION_HORIZON, proprio=prop, num_inference_steps=W.NUM_STEPS,
                                sigma_shift=W.SIGMA_SHIFT, seed=a.seed, rand_device="cpu", action_infer_mode="first_frame",
                                num_video_frames=W.NUM_VIDEO_FRAMES)["action"].cuda().float()
        r["ref_first_frame_mode"] = W.dist(ff, a0)
        if a_true is not None:
            r["ref_first_frame_mode__to_demo"] = W.dist(ff, a_true)
        for s in (1, 2):
            g = torch.Generator(device="cpu").manual_seed(100 * (a.seed + 1) + s)
            la = torch.randn(p.latents_action0.shape, generator=g, dtype=torch.float32).to(p.latents_action0)
            r[f"ref_seed{s}"] = W.dist(W.act(model, dataclasses.replace(p, latents_action0=la), K, V, amask), a0)

        # re-prefilled caches for the static and true futures
        lat_static = W.encode_clip(model, e["main"][[t] * 9], e["wrist"][[t] * 9]).to(model.torch_dtype)
        Ks, Vs, _, _ = W.prefill(model, p, lat_static)
        Kt = None
        if a_true is not None:   # the real future exists only for demonstrations
            idx = [min(t + 4 * k, e["length"] - 1) for k in range(9)]
            lat_true = W.encode_clip(model, e["main"][idx], e["wrist"][idx]).to(model.torch_dtype)
            Kt, Vt, _, _ = W.prefill(model, p, lat_true)
        # sanity: the current-frame slice must be untouched by any change to the future (first_frame_causal mask)
        r["check_current_slice_unchanged"] = max(float((Ks[l][:, CUR] - K[l][:, CUR]).abs().max()) for l in range(30))

        for band, layers in W.BANDS.items():
            run(f"future_donor|{band}", W.edit_slice(K, layers, FUT, Kd), W.edit_slice(V, layers, FUT, Vd))
            run(f"future_mean|{band}", W.edit_slice(K, layers, FUT, lambda l: meanK[l]),
                W.edit_slice(V, layers, FUT, lambda l: meanV[l]))

            def noise_like(cache):
                def f(l):
                    x = cache[l][:, FUT].float()
                    mu, sd = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
                    return (mu + sd * torch.randn(x.shape, generator=gen, device=x.device)).to(cache[l].dtype)
                return f
            run(f"future_noise|{band}", W.edit_slice(K, layers, FUT, noise_like(K)), W.edit_slice(V, layers, FUT, noise_like(V)))
            run(f"future_static|{band}", W.edit_slice(K, layers, FUT, Ks), W.edit_slice(V, layers, FUT, Vs))
            if Kt is not None:
                run(f"future_true|{band}", W.edit_slice(K, layers, FUT, Kt), W.edit_slice(V, layers, FUT, Vt))
            run(f"current_donor|{band}", W.edit_slice(K, layers, CUR, Kd), W.edit_slice(V, layers, CUR, Vd))
        run("future_knockout|all", K, V, mask=W.knockout_mask(amask, 98, vseq))
        if a.single_layer:
            for l in range(30):
                run(f"future_donor|L{l + 1}", W.edit_slice(K, [l], FUT, Kd), W.edit_slice(V, [l], FUT, Vd))

        rows.append(r)
        with open(f"{out}/rows.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        if i % 10 == 0:
            print(f"B {i}/{N}  {time.time() - t_start:.0f}s  donor|all={r['future_donor|all']:.3f} "
                  f"noise|all={r['future_noise|all']:.3f} cur_donor|all={r['current_donor|all']:.3f} "
                  f"seed1={r['ref_seed1']:.3f}", flush=True)

    # ---- summary: mean over states, 95% bootstrap CI over EPISODES (states within an episode are correlated)
    eps = sorted({r["episode"] for r in rows})
    keys = [k for k in rows[0] if k not in ("i", "task", "episode", "t", "donor_i", "donor_task")]
    rng = np.random.default_rng(0)
    per_ep = {k: np.array([np.mean([r[k] for r in rows if r["episode"] == ep]) for ep in eps]) for k in keys}
    summary = {}
    for k in keys:
        x = per_ep[k]
        boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
        summary[k] = {"mean": float(x.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    summary["_meta"] = {"n_states": N, "n_episodes": len(eps), "seconds": round(time.time() - t_start),
                        "states": states, "gpu_mem_peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    if rec:
        np.savez_compressed(f"{out}/attn.npz", M=np.stack(attn_M), O=np.stack(attn_O), err=np.array(attn_err),
                            groups=np.array(list(W.AttnRecorder.GROUPS)))
        summary["_meta"]["attn_max_reconstruction_error"] = float(max(attn_err))
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY (mean distance from the clean action, 95% CI over episodes)", flush=True)
    for k in keys:
        if "__" not in k:
            s = summary[k]
            print(f"  {k:28s} {s['mean']:.4f}  [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]", flush=True)
    print("PILOT DONE", flush=True)
