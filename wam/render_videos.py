"""Turn flip-test frame files (flip_test/videos/*.npz, main camera every 2 steps, 128 px) into .mp4 (laptop).

  python wam/render_videos.py wam/results/flip_test/videos [--side-by-side]

Each video plays at 10 fps (= 20 sim steps per second, real time is 20 fps). A caption bar shows the run, the step,
and turns orange from the step where task B's goal was first seen. --side-by-side also writes one grid per run tag
(all tasks of that run in one video).
"""
import glob
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

TASKS = ["open middle drawer", "bowl on stove", "bottle on cabinet", "top drawer + bowl", "bowl on cabinet",
         "plate to stove front", "cheese in bowl", "turn on stove", "bowl on plate", "bottle on rack"]  # LIBERO-Goal order


def caption(frame, text, hit):
    im = Image.fromarray(frame).resize((256, 256), Image.NEAREST)
    canvas = Image.new("RGB", (256, 288), (194, 94, 23) if hit else (27, 32, 38))
    canvas.paste(im, (0, 32))
    ImageDraw.Draw(canvas).text((6, 8), text, fill=(255, 255, 255))
    return np.asarray(canvas)


def render(path):
    d = np.load(path)
    frames, b_step = d["frames"], int(d["b_step"])
    name = os.path.basename(path)[:-4]
    tag, rest = name.rsplit("_task", 1)
    A = int(rest.split("_")[0])
    B = (A + 5) % 10
    out = []
    for k, f in enumerate(frames):
        step = 2 * (k + 1)
        hit = 0 <= b_step <= step
        out.append(caption(f, f"{tag}  A:{TASKS[A]}  B:{TASKS[B]}  t={step}" + ("  B DONE" if hit else ""), hit))
    imageio.mimsave(path[:-4] + ".mp4", out, fps=10)
    return tag, A, out


def main():
    root = sys.argv[1]
    by_tag = {}
    for p in sorted(glob.glob(os.path.join(root, "*.npz"))):
        tag, A, frames = render(p)
        by_tag.setdefault(tag, {})[A] = frames
        print("wrote", p[:-4] + ".mp4")
    if "--side-by-side" in sys.argv:
        for tag, vids in by_tag.items():
            n = max(len(v) for v in vids.values())
            blank = np.zeros_like(next(iter(vids.values()))[0])
            tiles = [vids.get(A) for A in range(10)]
            grid = []
            for k in range(n):
                cells = [(v[min(k, len(v) - 1)] if v else blank) for v in tiles]
                grid.append(np.concatenate([np.concatenate(cells[:5], 1), np.concatenate(cells[5:], 1)], 0))
            imageio.mimsave(os.path.join(root, f"grid_{tag}.mp4"), grid, fps=10)
            print("wrote grid", tag)


if __name__ == "__main__":
    main()
