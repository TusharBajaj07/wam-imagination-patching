"""Phase 2 (open loop, fresh states): WHAT part of the imagined future is read, WHERE the decision forms, WHICH heads.

  python wam/launch.py phase2 "--per-ep 2 --seed 0 --tag s0"

States: the same 100 fresh closed-loop states as pilot_future_swap --source rollouts (2 decisions x 50 clean
episodes, flip_test/clean_states). Donor = a state of a different task (same pairing rule as the pilot). Seed 0 for all
starting noise, so any change comes from the edit. Every edit swaps the donor's FUTURE K/V into the state's own cache.

Token map of the future slice (98-293): latent frame 1 = 98-195, latent frame 2 = 196-293. Inside a frame, token
index = row * 14 + col on a 7 x 14 grid (einops "b c f h w -> b (f h w) c", wan_video_dit.py:718); the model image is
main camera (cols 0-6) | wrist camera (cols 7-13), wamlib.to_model_image.

Exp 4, which part of the future (all 30 layers, plus L21-30 for the frame / camera splits):
  frame1, frame2, main-camera half, wrist-camera half;
  diff-ranked: rank the 196 future tokens by how much the two imagined futures differ there (||z_donor - z_own|| of
  the imagined latent patch, 48 ch x 2 x 2 = 192 numbers) and swap the top 10/25/50 % or the bottom 50 %;
  random 10/25/50 % of tokens as the matched control.
Exp 5, action lens: record the action stream after every layer at every denoising step (clean run, all-layer swap,
  L21-30 swap) and decode it with the model's own head: v = head(h), x0 = a_k - sigma_k * v (flow matching:
  a_sigma = x0 + sigma * v). Raw lens = head on h directly; tuned lens = a per-layer affine map h_l -> h_30 fitted by
  ridge least squares on the clean runs of half the episodes (even episode index), evaluated on the other half.
Exp 6, heads: swap the future in ONE of the 24 heads (channels h*128..h*128+127 of K and V) of L26 or of L28.
Outputs (out/phase2_<tag>/): rows.jsonl (distances per edit), lens.npz (decoded x0 per run/step/layer), summary.json.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import wamlib as W

FUT = slice(98, 294)
L2130 = range(20, 30)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--per-ep", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N states (smoke test)")
    ap.add_argument("--lens-only", action="store_true", help="skip Exp 4 and 6; run only the action lens")
    ap.add_argument("--lens-bands", default="all,L21-30", help="swap bands for the lens: all, L1-20, L21-30")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)                      # inference only (the lens calls the head directly)
    out = os.path.join(a.out, f"phase2_{a.tag}")
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    model, proc, _, load = W.load_model_and_processor()
    assert not load["mot_missing"] and not load["mot_unexpected"], load
    tasks = W.load_tasks()
    head = model.action_expert.head
    n_train_t = float(model.infer_action_scheduler.num_train_timesteps)

    # ---- action-stream recorder: the MoT applies each action block's post-attention part through this one method
    rec = {"on": False, "buf": []}
    orig_post = model.mot._apply_expert_post_block_tensor

    def post_wrapped(*args, **kw):
        x = orig_post(*args, **kw)
        if rec["on"] and x.shape[1] == W.ACTION_HORIZON:
            rec["buf"].append(x.detach())
        return x
    model.mot._apply_expert_post_block_tensor = post_wrapped

    @torch.no_grad()
    def act_lens(p, K, V, amask):
        """wamlib.act + the action stream after every layer. Returns final action [32,7], H [10,30,32,1024] (bf16),
        a_k [10,32,7] and sigma_k [10] (the sample and noise level each step starts from)."""
        la = p.latents_action0.clone()
        ts, ds = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=W.NUM_STEPS, device=model.device, dtype=la.dtype, shift_override=W.SIGMA_SHIFT)
        A, rec["buf"], rec["on"] = [], [], True
        for t, d in zip(ts, ds):
            A.append(la[0].float())
            pred = model._denoise_action_with_video_cache(latents_action=la, timestep_action=t.unsqueeze(0).to(la.dtype),
                                                          context=p.context, context_mask=p.context_mask,
                                                          video_cache_k=K, video_cache_v=V, action_attention_mask=amask)
            la = model.infer_action_scheduler.step(pred, d, la)
        rec["on"] = False
        assert len(rec["buf"]) == W.NUM_STEPS * 30, len(rec["buf"])
        H = torch.stack([b[0] for b in rec["buf"]]).view(W.NUM_STEPS, 30, W.ACTION_HORIZON, -1)
        rec["buf"] = []
        sig = ts.float() / n_train_t
        # sanity: the head on the last layer reproduces the model's own prediction at the last step
        return la[0].float(), H, torch.stack(A), sig, pred[0].float()

    @torch.no_grad()
    def decode(H, A, sig, T=None):
        """x0 estimate per step and layer: a_k - sigma_k * head(h). T: optional tuned-lens maps (list of 30)."""
        h = H.float()
        if T is not None:
            h = torch.stack([h[:, l] @ T[l][:-1] + T[l][-1] for l in range(30)], dim=1)
        v = head(h.to(head.weight.dtype)).float()                                  # [10,30,32,7]
        return A[:, None] - sig[:, None, None, None] * v

    # ---- states: identical selection to pilot_future_swap.py --source rollouts
    files = sorted(glob.glob(f"{a.out}/flip_test/clean_states/task*_init*.npz"),
                   key=lambda f: (int(np.load(f)["task"]), int(np.load(f)["init"])))
    assert len(files) == 50, len(files)
    states, episodes = [], {}
    for f in files:
        z = np.load(f)
        ep = os.path.basename(f)[:-4]
        episodes[ep] = {"main": z["main"], "wrist": z["wrist"], "state": z["state"], "n": len(z["t"])}
        for t in np.linspace(0, len(z["t"]) - 1, a.per_ep + 2)[1:-1].round().astype(int):
            states.append((int(z["task"]), ep, int(t)))
    ep_index = {ep: k for k, ep in enumerate(sorted(episodes))}
    if a.limit:
        states = states[: a.limit // 2] + states[len(states) // 2: len(states) // 2 + a.limit - a.limit // 2]
    N = len(states)
    print(f"{N} states", flush=True)

    def inputs(i):
        task, ep, t = states[i]
        e = episodes[ep]
        img = W.to_model_image(e["main"][t], e["wrist"][t], model.device, model.torch_dtype)
        prop = W.normalize_state(e["state"][t], proc)
        return W.prepare(model, img, prop, W.PROMPT.format(task=tasks[task]), seed=a.seed)

    def fut_tokens(lv):
        """Imagined future latents [1,48,3,14,28] -> per-token patches [196, 192] in token order (f h w)."""
        z = lv[0, :, 1:].float()                                               # [48, 2, 14, 28]
        z = z.view(48, 2, 7, 2, 14, 2).permute(1, 2, 4, 0, 3, 5).reshape(196, 192)
        return z

    # ---- phase A: imagine, prefill, clean action with the action stream recorded
    caches, lat, clean, lensA = [], [], [], []
    check_head = 0.0
    for i in range(N):
        p = inputs(i)
        lv = W.imagine(model, p)
        K, V, amask, vseq = W.prefill(model, p, lv)       # amask depends only on sequence lengths: same for all
        a0, H, Ak, sig, pred_last = act_lens(p, K, V, amask)
        a0_ref = W.act(model, p, K, V, amask)
        assert float((a0 - a0_ref).abs().max()) == 0.0, "recording changed the action"
        check_head = max(check_head, float((head(H[-1, -1].to(head.weight.dtype)).float() - pred_last).abs().max()))
        caches.append(([k.cpu() for k in K], [v.cpu() for v in V]))
        lat.append(fut_tokens(lv).cpu())
        clean.append(a0.cpu())
        lensA.append((H.cpu(), Ak.cpu(), sig.cpu()))
        if i % 10 == 0:
            print(f"A {i}/{N}  {time.time() - t_start:.0f}s", flush=True)
    print(f"head(h_30) vs model prediction, max abs diff: {check_head:.2e}", flush=True)

    # ---- tuned lens: ridge fit h_l -> h_30 on clean runs of even-index episodes
    train = [i for i in range(N) if ep_index[states[i][1]] % 2 == 0]
    T = []
    Y = torch.cat([lensA[i][0][:, -1].reshape(-1, lensA[i][0].shape[-1]) for i in train]).cuda().float()
    for l in range(30):
        X = torch.cat([lensA[i][0][:, l].reshape(-1, Y.shape[1]) for i in train]).cuda().float()
        X1 = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
        G = X1.T @ X1
        G += 1e-3 * G.diagonal().mean() * torch.eye(G.shape[0], device=G.device)
        T.append(torch.linalg.solve(G, X1.T @ Y))                              # [1025, 1024]
    del X, X1, Y, G
    print(f"tuned lens fitted on {len(train)} states  {time.time() - t_start:.0f}s", flush=True)

    # ---- phase B
    rows, lens_raw, lens_tuned = [], [], []
    BANDS = {"all": range(30), "L21-30": L2130, "L1-20": range(0, 20)}
    lens_bands = a.lens_bands.split(",")

    def lens_bands_run(i, p, K, V, Kd, Vd, r):
        H0, A0, s0 = (x.cuda() for x in lensA[i])
        lr, lt = [decode(H0, A0, s0)], [decode(H0, A0, s0, T)]
        for b in lens_bands:
            Ke, Ve = W.edit_slice(K, BANDS[b], FUT, Kd), W.edit_slice(V, BANDS[b], FUT, Vd)
            a1, H1, A1, s1, _ = act_lens(p, Ke, Ve, amask)
            r[f"lens_swap|{b}"] = W.dist(a1, clean[i].cuda())
            lr.append(decode(H1, A1, s1))
            lt.append(decode(H1, A1, s1, T))
        lens_raw.append(torch.stack(lr).detach().half().cpu().numpy())                  # [1+bands,10,30,32,7]
        lens_tuned.append(torch.stack(lt).detach().half().cpu().numpy())
    for i in range(N):
        task, ep, t = states[i]
        p = inputs(i)
        K = [k.cuda() for k in caches[i][0]]
        V = [v.cuda() for v in caches[i][1]]
        j = (i + N // 2) % N
        while states[j][0] == task:
            j = (j + 1) % N
        Kd = [k.cuda() for k in caches[j][0]]
        Vd = [v.cuda() for v in caches[j][1]]
        a0, a_d = clean[i].cuda(), clean[j].cuda()
        r = {"i": i, "task": task, "episode": ep, "t": t, "donor_i": j, "donor_task": states[j][0],
             "train": ep_index[ep] % 2 == 0, "clean_to_donor_action": W.dist(a0, a_d)}

        def run(name, Ke, Ve):
            a1 = W.act(model, p, Ke, Ve, amask)
            r[name] = W.dist(a1, a0)
            r[name + "__to_donor"] = W.dist(a1, a_d)

        def swap_tok(layers, idx):
            """Future tokens `idx` (positions inside the 196-token future) <- donor, in `layers`."""
            pos = torch.as_tensor(idx, device="cuda") + 98
            return W.edit_slice(K, layers, pos, Kd), W.edit_slice(V, layers, pos, Vd)

        if a.lens_only:
            lens_bands_run(i, p, K, V, Kd, Vd, r)
            rows.append(r)
            with open(f"{out}/rows.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            if i % 10 == 0:
                print(f"B {i}/{N}  {time.time() - t_start:.0f}s", flush=True)
            continue

        # Exp 4
        f1, f2 = list(range(0, 98)), list(range(98, 196))
        main_ = [k for k in range(196) if (k % 98) % 14 < 7]
        wrist = [k for k in range(196) if (k % 98) % 14 >= 7]
        for band, layers in (("all", range(30)), ("L21-30", L2130)):
            run(f"fut|{band}", *swap_tok(layers, range(196)))
            for nm, idx in (("frame1", f1), ("frame2", f2), ("main", main_), ("wrist", wrist)):
                run(f"{nm}|{band}", *swap_tok(layers, idx))
        diff = (lat[j] - lat[i]).norm(dim=1)                                    # [196]
        order = torch.argsort(diff, descending=True).tolist()
        r["diff_top10_share_main"] = float(np.mean([(k % 98) % 14 < 7 for k in order[:20]]))
        r["diff_top10_share_frame2"] = float(np.mean([k >= 98 for k in order[:20]]))
        g = torch.Generator().manual_seed(1000 + i)
        for pct in (10, 25, 50):
            n = round(196 * pct / 100)
            run(f"top{pct}|all", *swap_tok(range(30), order[:n]))
            run(f"rand{pct}|all", *swap_tok(range(30), torch.randperm(196, generator=g)[:n].tolist()))
        run("bottom50|all", *swap_tok(range(30), order[98:]))

        # Exp 5: action lens on the clean run (phase A) and the swap bands
        lens_bands_run(i, p, K, V, Kd, Vd, r)

        # Exp 6: one head of L26 or L28
        run("L26+L28|all_heads", W.edit_slice(K, [25, 27], FUT, Kd), W.edit_slice(V, [25, 27], FUT, Vd))
        for l in (25, 27):
            for h in range(24):
                ch = slice(h * 128, (h + 1) * 128)

                def one_head(cache, donor):
                    out_ = list(cache)
                    x = cache[l].clone()
                    x[:, FUT, ch] = donor[l][:, FUT, ch]
                    out_[l] = x
                    return out_
                run(f"head|L{l + 1}|h{h}", one_head(K, Kd), one_head(V, Vd))

        rows.append(r)
        with open(f"{out}/rows.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        if i % 10 == 0:
            print(f"B {i}/{N}  {time.time() - t_start:.0f}s  fut|all={r['fut|all']:.2f} frame1={r['frame1|all']:.2f} "
                  f"frame2={r['frame2|all']:.2f} main={r['main|all']:.2f} wrist={r['wrist|all']:.2f} "
                  f"top10={r['top10|all']:.2f} rand10={r['rand10|all']:.2f}", flush=True)

    a0s = np.stack([c.detach().numpy() for c in clean])
    np.savez_compressed(f"{out}/lens.npz", raw=np.stack(lens_raw), tuned=np.stack(lens_tuned), clean=a0s,
                        donor=np.array([r["donor_i"] for r in rows]), train=np.array([r["train"] for r in rows]),
                        runs=np.array(["clean"] + [f"swap_{b}" for b in lens_bands]))
    summary = {"n_states": N, "seconds": round(time.time() - t_start), "head_check_max_abs": check_head,
               "states": states, "gpu_mem_peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float)}}
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for k, v in summary["mean"].items():
        if "__" not in k and not k.startswith("head|"):
            print(f"  {k:24s} {v:.3f}", flush=True)
    print("PHASE2 DONE", flush=True)
