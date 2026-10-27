"""Phase 2 analysis (laptop, numpy only).

  python wam/analyze_phase2.py wam/results/phase2_s0 [wam/results/pilot_future_swap_rollouts_s0/attn.npz]
  python wam/analyze_phase2.py wam/results/phase2_lens3          # --lens-only run

Exp 4: action change (distance from the clean action) and distance to the donor's own action, for each part of the
  future swapped; 95% bootstrap CI over episodes.
Exp 5: action lens. For each layer l and denoising step k, x0 = the action decoded from the action stream. Pull toward
  the other task = d(x0, own clean action) / (d(x0, own) + d(x0, donor action)): 0 = reads as task A, 1 = as task B.
  Reported on held-out states (episodes not used to fit the tuned lens), raw and tuned lens, for every run in lens.npz.
Exp 6: single-head swaps in L26 and L28; optionally next to that head's attention share on the future (RQ1 recording).
Writes analysis.json next to the rows.
"""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "rows.jsonl")]
eps = sorted({r["episode"] for r in rows})
rng = np.random.default_rng(0)
out = {"n_states": len(rows), "n_episodes": len(eps)}


def per_ep(key):
    return np.array([np.mean([r[key] for r in rows if r["episode"] == ep]) for ep in eps])


def boot(x, n=5000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


base = per_ep("clean_to_donor_action")
print(f"{len(rows)} states, {len(eps)} episodes. Clean action -> donor action: {base.mean():.2f}")


def exp4():
    print("\nExp 4: which part of the future           change [95% CI]          to donor action   pull")
    out["exp4"] = {}
    for k in ["fut|all", "frame1|all", "frame2|all", "main|all", "wrist|all",
              "fut|L21-30", "frame1|L21-30", "frame2|L21-30", "main|L21-30", "wrist|L21-30",
              "top10|all", "rand10|all", "top25|all", "rand25|all", "top50|all", "rand50|all", "bottom50|all"]:
        c, t = boot(per_ep(k)), boot(per_ep(k + "__to_donor"))
        pull = boot(1 - per_ep(k + "__to_donor") / base)
        out["exp4"][k] = {"change": c, "to_donor": t, "pull": pull}
        print(f"  {k:16s} {c[0]:6.2f} [{c[1]:.2f}, {c[2]:.2f}]   {t[0]:6.2f}            {pull[0]:+.2f}")
    for k in ("diff_top10_share_main", "diff_top10_share_frame2"):
        out["exp4"][k] = boot(per_ep(k))
        print(f"  {k}: {out['exp4'][k][0]:.2f}")


def exp6():
    print("\nExp 6: one head swapped (change, pull toward other task)")
    att = None
    if len(sys.argv) > 2:
        A = np.load(sys.argv[2])["M"]              # [states, steps, layers, groups, heads]
        att = A[:, :, :, 1].mean(axis=(0, 1))      # [layers, heads] share of attention on the future
    out["exp6"] = {"L26+L28_all_heads": boot(per_ep("L26+L28|all_heads"))}
    print(f"  L26+L28 all heads: {out['exp6']['L26+L28_all_heads'][0]:.2f}")
    for L in (26, 28):
        ch = [boot(per_ep(f"head|L{L}|h{h}")) for h in range(24)]
        pl = [boot(1 - per_ep(f"head|L{L}|h{h}__to_donor") / base) for h in range(24)]
        out["exp6"][f"L{L}"] = [{"head": h, "change": ch[h], "pull": pl[h],
                                 "attn_fut": None if att is None else float(att[L - 1, h])} for h in range(24)]
        s = sorted(range(24), key=lambda h: -ch[h][0])
        print(f"  L{L}: sum of single heads {sum(c[0] for c in ch):.2f}; top heads:")
        for h in s[:6]:
            a_ = "" if att is None else f"  attn on future {att[L - 1, h]:.2f}"
            print(f"     h{h:<2d} {ch[h][0]:.3f} [{ch[h][1]:.3f}, {ch[h][2]:.3f}]  pull {pl[h][0]:+.3f}{a_}")
        if att is not None:
            r = np.corrcoef([c[0] for c in ch], att[L - 1])[0, 1]
            out["exp6"][f"L{L}_corr_change_vs_attn"] = float(r)
            print(f"     correlation (head effect vs head attention on future): {r:+.2f}")


def exp5():
    z = np.load(d / "lens.npz")
    runs = [str(x) for x in z["runs"]]
    clean, donor, train = z["clean"].astype(np.float32), z["donor"], z["train"]
    test = np.where(~train)[0]
    a_own = clean[test][:, None, None, None]                 # [n,1,1,1,32,7]
    a_don = clean[donor[test]][:, None, None, None]
    out["exp5"] = {"n_test_states": int(len(test)), "runs": runs}
    for lens in ("raw", "tuned"):
        x = z[lens][test].astype(np.float32)                # [n, runs, steps=10, layers=30, 32, 7]
        do = np.linalg.norm((x - a_own).reshape(*x.shape[:4], -1), axis=-1)
        dd = np.linalg.norm((x - a_don).reshape(*x.shape[:4], -1), axis=-1)
        pull = do / (do + dd)                               # [n, runs, 10, 30]
        lo, hi = [], []
        for _ in range(2000):                               # 95% CI over held-out states, step 1
            b = pull[rng.integers(0, len(test), len(test)), :, 0].mean(0)
            lo.append(b)
        lo = np.array(lo)
        out["exp5"][lens] = {"pull_mean": pull.mean(0).round(4).tolist(),       # [runs, steps, layers]
                             "pull_step1_ci95": [np.percentile(lo, 2.5, 0).round(4).tolist(),
                                                 np.percentile(lo, 97.5, 0).round(4).tolist()]}
        print(f"\nExp 5 ({lens} lens, {len(test)} held-out states), denoising step 1: pull toward the other task "
              f"(0 = A, 1 = B)")
        print("  layer " + "".join(f"{nm:>14s}" for nm in runs))
        for l in range(30):
            print(f"  L{l + 1:<4d}" + "".join(f"{pull[:, r_, 0, l].mean():14.2f}" for r_ in range(len(runs))))


if "fut|all" in rows[0]:
    exp4()
    exp6()
exp5()
(d / "analysis.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {d / 'analysis.json'}")
