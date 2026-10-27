"""Combine the seeded pilot runs (laptop, numpy only).

  python wam/analyze_pilot.py wam/results/pilot_future_swap_goal100_s0 ..._s1 ..._s2

Prints, with 95% bootstrap intervals over episodes (seeds averaged within each episode first):
  - every edit's mean action change, and the same divided by the seed floor
  - the single-layer future swap, layer by layer (effect and pull toward the other task's action)
  - RQ1: attention share M, size-corrected share R and output size O of the imagined future, per layer
Writes analysis.json next to the first run.
"""
import json
import sys
from pathlib import Path

import numpy as np

runs = [Path(p) for p in sys.argv[1:]]
rows = [[json.loads(l) for l in open(r / "rows.jsonl")] for r in runs]
for r, rs in zip(runs, rows):
    print(f"{r.name}: {len(rs)} states")
eps = sorted({r["episode"] for r in rows[0]})
keys = [k for k in rows[0][0] if isinstance(rows[0][0][k], float)]
rng = np.random.default_rng(0)


def per_episode(key):
    """[episodes] mean over states and seeds."""
    return np.array([np.mean([r[key] for rs in rows for r in rs if r["episode"] == ep]) for ep in eps])


def boot(x, n=5000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


out = {"n_seeds": len(runs), "n_states_per_seed": len(rows[0]), "n_episodes": len(eps), "edits": {}}
floor = per_episode("ref_seed1") / 2 + per_episode("ref_seed2") / 2
out["seed_floor"] = boot(floor)
print(f"\nseed floor (action-noise only): {out['seed_floor'][0]:.3f} [{out['seed_floor'][1]:.3f}, {out['seed_floor'][2]:.3f}]")
print(f"\n{'edit':30s} {'change':>22s} {'x floor':>8s}")
for k in keys:
    if "__" in k or k.startswith("ref_") or k.startswith("check") or k.startswith("clean"):
        continue
    if k.startswith("future_donor|L") and "-" not in k:
        continue
    m = boot(per_episode(k))
    out["edits"][k] = m
    print(f"{k:30s} {m[0]:7.3f} [{m[1]:.3f}, {m[2]:.3f}] {m[0] / out['seed_floor'][0]:8.1f}")
for k in ("clean_to_demo", "future_true|all__to_demo", "ref_first_frame_mode", "ref_first_frame_mode__to_demo",
          "clean_to_donor_action", "future_donor|all__to_donor_action", "current_donor|all__to_donor_action"):
    if k not in rows[0][0]:
        continue
    out["edits"][k] = boot(per_episode(k))
    m = out["edits"][k]
    print(f"{k:38s} {m[0]:7.3f} [{m[1]:.3f}, {m[2]:.3f}]")

if f"future_donor|L1" in rows[0][0]:
    print("\nsingle-layer future swap (other task):  layer  change  [CI]   pull toward other task's action")
    full = per_episode("future_donor|all")
    base = per_episode("clean_to_donor_action")
    out["single_layer"] = []
    for l in range(1, 31):
        ch = boot(per_episode(f"future_donor|L{l}"))
        pull = boot(1 - per_episode(f"future_donor|L{l}__to_donor_action") / base)
        out["single_layer"].append({"layer": l, "change": ch, "pull": pull})
        bar = "#" * int(round(40 * ch[0] / full.mean()))
        print(f"  L{l:<3d} {ch[0]:6.3f} [{ch[1]:.3f}, {ch[2]:.3f}]  pull {pull[0]:+.3f}  {bar}")

attn = [r / "attn.npz" for r in runs if (r / "attn.npz").exists()]
if attn:
    M = np.concatenate([np.load(a)["M"] for a in attn])        # [states, steps, layers, groups, heads]
    O = np.concatenate([np.load(a)["O"] for a in attn])
    err = max(float(np.load(a)["err"].max()) for a in attn)
    n = {"cur": 98, "fut": 196, "act": 32}
    even = {g: n[g] / 326 for g in n}
    print(f"\nRQ1 attention ({M.shape[0]} state-runs; reconstruction error max {err:.2e})")
    print("  layer   M_cur  M_fut  M_act   R_fut   O_fut/O_all")
    out["attention"] = []
    for l in range(30):
        m = M[:, :, l].mean(axis=(0, 1, 3))                    # groups
        o = O[:, :, l].mean(axis=(0, 1, 3))
        row = {"layer": l + 1, "M": dict(zip(n, m.round(4).tolist())), "R_fut": float(m[1] / even["fut"]),
               "O_fut_share": float(o[1] / o.sum())}
        out["attention"].append(row)
        print(f"  L{l + 1:<4d}  {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}   {m[1] / even['fut']:.2f}    {o[1] / o.sum():.3f}")
    steps = M[:, :, :, 1].mean(axis=(0, 2, 3))
    out["attention_by_step_M_fut"] = steps.round(4).tolist()
    print("  M_fut by denoising step (1 = noisiest):", " ".join(f"{x:.3f}" for x in steps))
    out["attention_reconstruction_error_max"] = err

(runs[0] / "analysis.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {runs[0] / 'analysis.json'}")
