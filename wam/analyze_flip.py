"""Closed-loop flip test summary (laptop, stdlib + numpy).

  python wam/analyze_flip.py wam/results/flip_test [name=glob ...] [--out name.json]

Default conditions: clean.jsonl, textB.jsonl, futureB.jsonl + futureB_part_*.jsonl (the seed-42 all-layer run).
Or name them: e.g. "L21-30=futureB_L21-30.jsonl". Parallel parts are merged; a duplicated (task, init) is an error.
Prints per condition: A succeeded, B's goal reached, with 95% Wilson intervals; then per task pair; then the step at
which B was reached. Writes flip_summary.json in the same folder.
"""
import glob
import json
import math
import os
import sys

import numpy as np

d = sys.argv[1]
args = sys.argv[2:]
out_name = "flip_summary.json"
if "--out" in args:
    out_name = args[args.index("--out") + 1]
    args = [x for i, x in enumerate(args) if x != "--out" and (i == 0 or args[i - 1] != "--out")]
TASKS = ["open the middle drawer", "bowl on stove", "wine bottle on top of cabinet", "open top drawer, bowl inside",
         "bowl on top of cabinet", "push plate to front of stove", "cream cheese in bowl", "turn on stove",
         "bowl on plate", "wine bottle on rack"]  # LIBERO-Goal task order (matches the env's task names in the rows)


def load(*patterns):
    files = sorted({f for p in patterns for f in glob.glob(os.path.join(d, p))})
    assert files, patterns
    rows = [json.loads(l) for f in files for l in open(f)]
    keys = [(r["task_id"], r["init"]) for r in rows]
    assert len(keys) == len(set(keys)), f"duplicate episodes in {patterns}"
    return sorted(rows, key=lambda r: (r["task_id"], r["init"]))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


if args:
    conds = {n: load(*g.split("+")) for n, g in (x.split("=", 1) for x in args)}
else:
    conds = {"clean": load("clean.jsonl"), "textB": load("textB.jsonl"),
             "futureB": load("futureB.jsonl", "futureB_part_*.jsonl")}
out = {}
W_ = max(9, max(map(len, conds)))
print(f"{'condition':{W_}s} {'n':>3s}  {'A succeeded':>24s}  {'B goal reached':>24s}  errors")
for c, rows in conds.items():
    n = len(rows)
    ka = sum(r["success"] for r in rows)
    kb = sum(r["check_success_step"] is not None for r in rows)
    err = sum(r["check_error"] is not None for r in rows)
    wa, wb = wilson(ka, n), wilson(kb, n)
    out[c] = {"n": n, "A": [ka, *wa], "B": [kb, *wb], "check_errors": err,
              "B_step": [r["check_success_step"] for r in rows if r["check_success_step"] is not None],
              "A_steps_success": [r["steps"] for r in rows if r["success"]]}
    print(f"{c:{W_}s} {n:3d}  {ka:2d}/{n} {wa[0]:5.0%} [{wa[1]:.0%},{wa[2]:.0%}]  "
          f"{kb:2d}/{n} {wb[0]:5.0%} [{wb[1]:.0%},{wb[2]:.0%}]  {err}")

print("\nper task (A -> B): A succeeded / B reached, out of 5 starts")
print(f"  {'A':32s} {'B':32s} " + " ".join(f"{c:>9s}" for c in conds))
out["per_task"] = []
for A in range(10):
    B = (A + 5) % 10
    cell, row = [], {"A": A, "B": B}
    for c, rows in conds.items():
        rs = [r for r in rows if r["task_id"] == A]
        ka = sum(r["success"] for r in rs)
        kb = sum(r["check_success_step"] is not None for r in rs)
        row[c] = {"n": len(rs), "A": ka, "B": kb}
        cell.append(f"{ka}/{kb}" if rs else "-")
    out["per_task"].append(row)
    print(f"  {A} {TASKS[A]:30s} {B} {TASKS[B]:30s} " + " ".join(f"{x:>9s}" for x in cell))

for c in conds:
    s = out[c]["A_steps_success"]
    if s:
        print(f"\n{c}: steps to A success  median {np.median(s):.0f}  (range {min(s)}-{max(s)})")
    s = out[c]["B_step"]
    if s:
        print(f"{c}: step B goal first seen  median {np.median(s):.0f}  (range {min(s)}-{max(s)})")

json.dump(out, open(os.path.join(d, out_name), "w"), indent=1)
print(f"\nwrote {os.path.join(d, out_name)}")
