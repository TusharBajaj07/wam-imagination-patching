"""Figures for the paper, built only from result files in wam/results/ (laptop).

  python paper/make_figs.py

Writes paper/figs/*.pdf and paper/figs/numbers.tex (every number quoted in the text, as LaTeX macros).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parent.parent / "wam" / "results"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
J = lambda p: json.loads((R / p).read_text())  # noqa: E731

A_COL, B_COL, C_COL, D_COL, GREY = "#2f6db0", "#c25e17", "#7b4fb3", "#23866f", "#9aa3ab"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "axes.titlesize": 8, "axes.titleweight": "bold"})
W1 = 3.45   # IEEE single column, inches

train = J("pilot_future_swap_goal100_s0/analysis.json")       # 3 seeds pooled, demo (training) frames
fresh = J("pilot_future_swap_rollouts_s0/analysis.json")      # fresh closed-loop states
flip = J("flip_test/flip_summary.json")
p1 = J("phase1/flip_test/phase1_layers_dose.json")
seeds = J("phase1/flip_test/phase1_seeds.json")
seeds.update({k.replace("noimag", "noimg"): v for k, v in J("phase1/flip_test/noimagine_summary.json").items()
              if k.startswith("noimag")})   # imagination off (first-frame mode), same 150 episodes
p2 = J("phase2_s0/analysis.json")
lens = J("phase2_lens3/analysis.json")

# ---------------------------------------------------------------- Fig. 2: closed-loop flip, three seeds
fig, ax = plt.subplots(figsize=(W1, 1.85))
conds = [("clean", "Normal\n\n(imagine A,\ntold A)"), ("noimg", "Imagination\noff\n(no future,\ntold A)"),
         ("textB", "Swap\ninstruction\n(imagine A,\ntold B)"), ("futB", "Swap\nimagination\n(imagine B,\ntold A)")]
x = np.arange(len(conds))
for k, (key, col, lab) in enumerate((("A", A_COL, "task A done"), ("B", B_COL, "task B done"))):
    vals = [[seeds[f"{c}_{s}"][key][1] for s in ("s42", "s0", "s1")] for c, _ in conds]
    m = [np.mean(v) for v in vals]
    ax.bar(x + (k - 0.5) * 0.36, m, 0.34, color=col, label=lab, zorder=2)
    for i, v in enumerate(vals):
        ax.scatter([x[i] + (k - 0.5) * 0.36] * 3, v, s=6, color="k", zorder=3, linewidths=0)
        ax.text(x[i] + (k - 0.5) * 0.36, max(v) + 0.05, f"{m[i]:.0%}", ha="center", fontsize=6.5)
ax.set_xticks(x, [c[1] for c in conds], fontsize=6.3)
ax.set_ylim(0, 1.32)
ax.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
ax.set_ylabel("episodes (of 50)")
ax.legend(loc="upper center", fontsize=6.5, ncol=2, bbox_to_anchor=(0.5, 1.04))
ax.grid(axis="y", lw=0.4, color="#e3e6e9", zorder=0)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "flip_seeds.pdf")

# ---------------------------------------------------------------- Fig. 3: attention vs use, per layer (fresh states)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 2.3), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
L = np.arange(1, 31)
Rf = np.array([r["R_fut"] for r in fresh["attention"]])
ch = np.array([r["change"][0] for r in fresh["single_layer"]])
lo = np.array([r["change"][1] for r in fresh["single_layer"]])
hi = np.array([r["change"][2] for r in fresh["single_layer"]])
a1.bar(L, Rf, 0.75, color=[A_COL if v > 1 else "#a9c3e3" for v in Rf], zorder=2)
a1.axhline(1, ls="--", lw=0.6, color="k")
a1.set_ylabel("attention to\nfuture (×chance)")
a1.set_ylim(0, 1.45)
a2.bar(L, ch, 0.75, color=[B_COL if v > 0.4 else "#e8b48f" for v in ch], zorder=2)
a2.errorbar(L, ch, yerr=[ch - lo, hi - ch], fmt="none", ecolor="k", elinewidth=0.5, capsize=0)
a2.axhline(fresh["seed_floor"][0], ls=":", lw=0.6, color="k")
a2.set_ylabel("action change,\nthis layer only")
a2.set_xlabel("layer")
a2.set_xticks([1, 5, 10, 15, 20, 25, 30])
for a in (a1, a2):
    a.axvspan(20.5, 30.5, color="#f5e0cf", alpha=0.45, zorder=0, lw=0)
a1.text(25.5, 1.32, "layers 21–30", ha="center", fontsize=6.5)
fig.tight_layout(pad=0.3, h_pad=0.4)
fig.savefig(OUT / "attention_vs_use.pdf")

# ---------------------------------------------------------------- Fig. 4: closed-loop localization + dose
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 1.75), gridspec_kw={"width_ratios": [1.25, 1]})
rows = [("none", seeds["clean_s42"]), ("L1–20", p1["L1-20"]), ("L26+28", p1["L26+28"]),
        ("L21–30", p1["L21-30"]), ("all", p1["all_s42"])]
y = np.arange(len(rows))[::-1]
for k, (key, col) in enumerate((("A", A_COL), ("B", B_COL))):
    v = [r[key][1] for _, r in rows]
    e = [[r[key][1] - r[key][2] for _, r in rows], [r[key][3] - r[key][1] for _, r in rows]]
    a1.barh(y + (0.5 - k) * 0.36, v, 0.34, color=col, xerr=e, error_kw={"elinewidth": 0.5}, zorder=2,
            label=f"task {key} done")
a1.set_yticks(y, [r[0] for r in rows])
a1.set_xlim(0, 1.05)
a1.set_xticks([0, .5, 1], ["0", "50%", "100%"])
a1.set_title("B's future swapped into", loc="left", fontsize=7)
h_, l_ = a1.get_legend_handles_labels()
fig.legend(h_, l_, fontsize=6.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
al = [0, .25, .5, .75, 1]
keys = ["clean_s42", "a0.25", "a0.5", "a0.75", "all_s42"]
src = {**p1, "clean_s42": seeds["clean_s42"]}
for key, col in (("A", A_COL), ("B", B_COL)):
    m = [src[k][key][1] for k in keys]
    l_ = [src[k][key][2] for k in keys]
    h_ = [src[k][key][3] for k in keys]
    a2.fill_between(al, l_, h_, color=col, alpha=0.15, lw=0)
    a2.plot(al, m, "-o", color=col, ms=3, lw=1.3)
a2.set_xticks(al, ["0", ".25", ".5", ".75", "1"])
a2.set_xlabel("B's share α")
a2.set_yticks([0, .5, 1], ["0", "50%", "100%"])
a2.set_title("blended, all layers", loc="left", fontsize=7)
fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.1, 1, 1))
fig.savefig(OUT / "localize_dose.pdf")

# ---------------------------------------------------------------- Fig. 5: action lens
t = lens["exp5"]["tuned"]
runs = lens["exp5"]["runs"]
fig, ax = plt.subplots(figsize=(W1, 1.75))
style = {"clean": (A_COL, "-", "normal run"), "swap_all": (B_COL, "-", "B's future, all layers"),
         "swap_L21-30": (C_COL, "--", "B's future, L21–30"), "swap_L1-20": (D_COL, ":", "B's future, L1–20")}
for r, name in enumerate(runs):
    col, ls, lab = style[name]
    m = np.array([t["pull_mean"][r][0][l] for l in range(30)])
    ax.fill_between(L, t["pull_step1_ci95"][0][r], t["pull_step1_ci95"][1][r], color=col, alpha=0.15, lw=0)
    ax.plot(L, m, ls, color=col, lw=1.4, label=lab)
ax.axvspan(20.5, 30.5, color="#f5e0cf", alpha=0.45, zorder=0, lw=0)
ax.axhline(0.5, lw=0.5, color="#9aa3ab")
ax.set_xlim(1, 30)
ax.set_ylim(0, 1)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("layer")
ax.set_ylabel("decoded action\n(0 = task A, 1 = task B)")
ax.legend(fontsize=6, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "action_lens.pdf")

# ---------------------------------------------------------------- numbers used in the text
N = {}


def put(name, v, fmt="{:.2f}"):
    N[name] = fmt.format(v)


put("floorTrain", train["seed_floor"][0]); put("floorFresh", fresh["seed_floor"][0])
for tag, src_ in (("Train", train), ("Fresh", fresh)):
    e = src_["edits"]
    for k, nm in (("future_donor|all", "Donor"), ("current_donor|all", "Present"), ("future_mean|all", "Mean"),
                  ("future_noise|all", "Noise"), ("future_static|all", "Static"), ("future_knockout|all", "Knock"),
                  ("ref_first_frame_mode", "FF"), ("future_donor|L1-10", "BandOne"),
                  ("future_donor|L11-20", "BandTwo"), ("future_donor|L21-30", "BandThree"),
                  ("clean_to_donor_action", "CleanToDonor"), ("future_donor|all__to_donor_action", "DonorToDonor")):
        put(nm + tag, e[k][0])
    put("Ratio" + tag, e["future_donor|all"][0] / src_["seed_floor"][0], "{:.0f}")
e = train["edits"]
put("TrueToDemo", e["future_true|all__to_demo"][0]); put("CleanToDemo", e["clean_to_demo"][0])
put("FFToDemo", e["ref_first_frame_mode__to_demo"][0])
sl = {r["layer"]: r["change"][0] for r in fresh["single_layer"]}
put("LTwentyEight", sl[28]); put("LTwentySix", sl[26]); put("LTwentyFive", sl[25])
put("SingleSum", sum(sl.values()))
put("RfutPeak", max(Rf)); N["RfutPeakLayer"] = str(int(L[np.argmax(Rf)]))
pct = lambda v: f"{100 * v:.0f}"  # noqa: E731
for k, nm in (("L21-30", "BandLate"), ("L26+28", "BandTwoLayers"), ("L1-20", "BandEarly"), ("a0.25", "AlphaQ"),
              ("a0.5", "AlphaH"), ("a0.75", "AlphaTQ"), ("all_s42", "AlphaOne")):
    N[f"{nm}A"], N[f"{nm}B"] = pct(p1[k]["A"][1]), pct(p1[k]["B"][1])
sB = [seeds[f"futB_{s}"]["B"][0] for s in ("s42", "s0", "s1")]
sA = [seeds[f"futB_{s}"]["A"][0] for s in ("s42", "s0", "s1")]
tA = [seeds[f"textB_{s}"]["A"][0] for s in ("s42", "s0", "s1")]
cA = [seeds[f"clean_{s}"]["A"][0] for s in ("s42", "s0", "s1")]
nA = [seeds[f"noimg_{s}"]["A"][0] for s in ("s42", "s0", "s1")]
nB = [seeds[f"noimg_{s}"]["B"][0] for s in ("s42", "s0", "s1")]


def wilson(k, n, z=1.96):
    c, h = (k + z * z / 2) / (n + z * z), z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return pct(c - h), pct(c + h)


N.update(FutBPooled=str(sum(sB)), FutBPooledPct=pct(sum(sB) / 150), FutAPooled=str(sum(sA)),
         TextAPooled=str(sum(tA)), TextAPooledPct=pct(sum(tA) / 150), CleanAPooled=str(sum(cA)),
         CleanAPooledPct=pct(sum(cA) / 150), FutBSeeds="/".join(map(str, sB)), TextASeeds="/".join(map(str, tA)),
         CleanASeeds="/".join(map(str, cA)), NoImgAPooled=str(sum(nA)), NoImgAPooledPct=pct(sum(nA) / 150),
         NoImgASeeds="/".join(map(str, nA)), NoImgBPooled=str(sum(nB)))
N["CleanACILo"], N["CleanACIHi"] = wilson(sum(cA), 150)
N["NoImgACILo"], N["NoImgACIHi"] = wilson(sum(nA), 150)
st = lambda c: [x for s in ("s42", "s0", "s1") for x in seeds[f"{c}_{s}"]["A_steps_success"]]  # noqa: E731
N["MedStepsCleanPooled"], N["MedStepsNoImgPooled"] = f"{np.median(st('clean')):.0f}", f"{np.median(st('noimg')):.0f}"
e4 = p2["exp4"]
for k, nm in (("fut|all", "PAll"), ("top50|all", "PTop"), ("rand50|all", "PRand"), ("bottom50|all", "PBottom"),
              ("frame1|all", "PFOne"), ("frame2|all", "PFTwo"), ("main|all", "PMain"), ("wrist|all", "PWrist")):
    put(nm, e4[k]["change"][0]); put(nm + "Pull", e4[k]["pull"][0])
h = p2["exp6"]
put("HeadsBoth", h["L26+L28_all_heads"][0])
for L_ in (26, 28):
    top = sorted(h[f"L{L_}"], key=lambda r: -r["change"][0])
    N[f"HeadTop{'A' if L_ == 26 else 'B'}"] = ", ".join(f"h{r['head']} ({r['change'][0]:.2f})" for r in top[:3])
    put(f"HeadSum{'A' if L_ == 26 else 'B'}", sum(r["change"][0] for r in top))
    put(f"HeadCorr{'A' if L_ == 26 else 'B'}", h[f"L{L_}_corr_change_vs_attn"], "{:+.2f}")
put("HeadMaxPull", max(r["pull"][0] for L_ in (26, 28) for r in h[f"L{L_}"]))
m = {nm: [t["pull_mean"][r][0][l] for l in range(30)] for r, nm in enumerate(runs)}
put("LensEarlyTwenty", m["swap_L1-20"][19]); put("LensEarlyThirty", m["swap_L1-20"][29])
put("LensEarlyNine", m["swap_L1-20"][8]); put("LensLateTwenty", m["swap_L21-30"][19])
put("LensLateThirty", m["swap_L21-30"][29]); put("LensAllThirty", m["swap_all"][29])
put("LensCleanThirty", m["clean"][29])
N["MedStepsClean"] = f"{np.median(flip['clean']['A_steps_success']):.0f}"
N["MedStepsFutB"] = f"{np.median(flip['futureB']['B_step']):.0f}"
mact = [r["M"]["act"] for r in fresh["attention"][26:30]]
N["MactLateLo"], N["MactLateHi"] = f"{min(mact):.2f}", f"{max(mact):.2f}"
N["MactMidMax"] = f"{max(r['M']['act'] for r in fresh['attention'][1:26]):.2f}"   # layers 2-26
stp = fresh["attention_by_step_M_fut"]
N["MfutStepFirst"], N["MfutStepLast"] = f"{stp[0]:.2f}", f"{stp[-1]:.2f}"
N["ReconErr"] = f"{fresh['attention_reconstruction_error_max']:.0e}".replace("e-0", "\\times10^{-") + "}"
with open(OUT / "numbers.tex", "w") as f:
    f.write("% generated by make_figs.py from wam/results -- do not edit\n")
    for k, v in N.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
print(f"wrote {len(N)} numbers and 4 figures to {OUT}")
