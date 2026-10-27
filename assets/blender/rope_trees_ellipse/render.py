"""Render frames of the rope animation.

    python render.py OUTDIR FIRST LAST [STEP] [--pct N] [--samples N]

Frames whose scene state matches an already-rendered frame are copied
instead of rendered (the held shots).
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import matplotlib  # noqa: E402

import geom as g  # noqa: E402
import scene  # noqa: E402


def key_of(st):
    def r(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, (tuple, list)):
            return [r(x) for x in v]
        return v
    return json.dumps({k: r(v) for k, v in st.items() if k != "t"}, sort_keys=True)


def main():
    args = sys.argv[1:]
    pct = 100
    samples = None
    if "--pct" in args:
        i = args.index("--pct")
        pct = int(args[i + 1])
        del args[i:i + 2]
    if "--samples" in args:
        i = args.index("--samples")
        samples = int(args[i + 1])
        del args[i:i + 2]
    frames = None
    if "--frames" in args:
        i = args.index("--frames")
        frames = [int(x) for x in args[i + 1].split(",")]
        del args[i:i + 2]
    out = args[0]
    if frames is None:
        first, last = int(args[1]), int(args[2])
        step = int(args[3]) if len(args) > 3 else 1
        frames = range(first, last + 1, step)
    os.makedirs(out, exist_ok=True)

    font_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    scene.build(font_dir)
    sc = bpy.context.scene
    sc.render.resolution_percentage = pct
    if samples:
        sc.cycles.samples = samples
    if "RENDER_THREADS" in os.environ:
        sc.render.threads_mode = "FIXED"
        sc.render.threads = int(os.environ["RENDER_THREADS"])

    seen = {}
    t_start = time.time()
    n_rendered = 0
    for f in frames:
        path = os.path.join(out, f"f{f:04d}.png")
        st = g.state(f)
        k = key_of(st)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            seen.setdefault(k, path)
            continue
        if k in seen:
            shutil.copyfile(seen[k], path)
            continue
        scene.apply(st)
        sc.render.filepath = path
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        n_rendered += 1
        seen[k] = path
        print(f"frame {f} rendered in {time.time() - t0:.1f}s "
              f"(total {time.time() - t_start:.0f}s, {n_rendered} renders)", flush=True)


main()
