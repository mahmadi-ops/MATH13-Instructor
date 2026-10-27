"""Text overlays for the rope animation, composited over the Blender frames.

    python overlay.py FRAMEDIR OUT.mp4            # whole video
    python overlay.py FRAMEDIR OUTDIR --frames 0,300,...   # stills to check
"""
import functools
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as g  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "DejaVu Sans"

W, H = 1920, 1080
DARK = "#1B1B1B"
BLUE = "#0072B2"
ORANGE = "#C04F00"
GRAY = "#6E6E6E"
PANEL = (255, 255, 255, 222)
PANEL_EDGE = (0, 0, 0, 40)
TPAD = 4


# ---------------------------------------------------------------- text
@functools.lru_cache(maxsize=8192)
def text_img(s, px, color=DARK, weight="normal"):
    """RGBA image of one line of text, and the baseline's distance from
    its top edge.  Heights are padded to the font's ascent/descent so
    lines of the same size share a baseline."""
    fig = Figure(figsize=(1, 1), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    t = fig.text(0, 0, s, fontsize=px * 0.72, color=color, weight=weight,
                 va="baseline", ha="left")
    r = canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    pad = TPAD
    asc = max(bb.y1, 0.95 * px)
    desc = max(-bb.y0, 0.30 * px)
    w = int(math.ceil(bb.x1 - min(bb.x0, 0))) + 2 * pad
    h = int(math.ceil(asc + desc)) + 2 * pad
    fig.set_size_inches(w / 100, h / 100)
    t.set_position(((pad - min(bb.x0, 0)) / w, (pad + desc) / h))
    canvas.draw()
    im = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy())
    return im, int(round(pad + asc))


def rounded(size, fill, radius=18, outline=PANEL_EDGE):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius,
                        fill=fill, outline=outline, width=2)
    return im


def fade_img(im, a):
    if a >= 0.997:
        return im
    im = im.copy()
    al = np.asarray(im.getchannel("A"), dtype=np.float32) * max(a, 0.0)
    im.putalpha(Image.fromarray(al.astype(np.uint8)))
    return im


def stack(rows, padx=30, pady=22, align="left", gap=1.0, min_w=0, row_alpha=None):
    """Panel holding rows.  A row is a list of (text, px, color, weight)
    pieces laid out left to right, or ("rule",) for a thin line, or
    ("img", image) for a ready-made image."""
    lines = []
    for row in rows:
        if row and row[0] == "rule":
            lines.append(("rule", None, 0, 14))
            continue
        if row and row[0] == "img":
            im = row[1]
            lines.append(("img", im, im.size[0], im.size[1] + 8))
            continue
        pieces = [text_img(*p) for p in row]
        width = sum(im.size[0] - 2 * TPAD for im, _ in pieces) + 2 * TPAD
        px = max(p[1] for p in row)
        # tall math (a radical, say) gets a taller row instead of
        # running into the row above
        asc = max(bl - TPAD for _, bl in pieces)
        desc = max(im.size[1] - bl - TPAD for im, bl in pieces)
        pitch = max(int(px * 1.42 * gap), asc + desc + 12)
        lines.append(("text", (pieces, max(int(pitch * 0.74), asc + 4)), width, pitch))
    inner_w = max([ln[2] for ln in lines] + [min_w])
    total_h = sum(ln[3] for ln in lines)
    panel = rounded((inner_w + 2 * padx, total_h + 2 * pady), PANEL)
    y = pady
    for li, (kind, data, width, pitch) in enumerate(lines):
        ra = 1.0 if row_alpha is None else row_alpha[li]
        if align == "center":
            x = padx + (inner_w - width) // 2
        else:
            x = padx
        if kind == "rule":
            d = ImageDraw.Draw(panel)
            d.line((padx, y + pitch // 2, padx + inner_w, y + pitch // 2),
                   fill=(0, 0, 0, 70), width=2)
        elif kind == "img":
            panel.alpha_composite(data, (x, y + 4))
        else:
            pieces, base_off = data
            base = y + base_off
            for im, bl in pieces:
                if ra > 0.003:
                    panel.alpha_composite(fade_img(im, ra), (x, max(0, base - bl)))
                x += im.size[0] - 2 * TPAD
        y += pitch
    return panel


def chip(label, px, style):
    im, bl = text_img(label, px, {"plain": DARK, "on": "#FFFFFF", "off": "#9A9A9A"}[style],
                      "bold" if style == "on" else "normal")
    w, h = im.size[0] + 34, int(px * 1.9)
    fill = {"plain": (255, 255, 255, 255), "on": (0, 114, 178, 255),
            "off": (240, 240, 240, 255)}[style]
    edge = {"plain": (0, 0, 0, 90), "on": (0, 90, 140, 255), "off": (0, 0, 0, 40)}[style]
    c = rounded((w, h), fill, radius=h // 2, outline=edge)
    c.alpha_composite(im, (17, max(0, (h - im.size[1]) // 2)))
    if style == "off":
        d = ImageDraw.Draw(c)
        d.line((14, h // 2, w - 14, h // 2), fill=(150, 150, 150, 255), width=2)
    return c


def chip_row(px, highlight=None):
    names = ["line", "ellipse", "parabola", "hyperbola"]
    chips = []
    for n in names:
        style = "plain" if highlight is None else ("on" if n == highlight else "off")
        chips.append(chip(n, px, style))
    gap = 22
    w = sum(c.size[0] for c in chips) + gap * (len(chips) - 1)
    h = max(c.size[1] for c in chips)
    row = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for c in chips:
        row.alpha_composite(c, (x, 0))
        x += c.size[0] + gap
    return row


# ---------------------------------------------------------------- content
@functools.lru_cache(maxsize=None)
def title_card():
    return stack([
        [("Assignment 6", 34, BLUE, "bold")],
        [("The rope between two trees", 60, DARK, "bold")],
    ], padx=46, pady=26, align="center")


SETUP_LINES = [
    (4.6, [("Two trees stand 100 ft apart.", 32)]),
    (5.6, [("A rope is tied at ", 32), ("$A$", 36, ORANGE), (", 100 ft up the first tree,", 32)]),
    (6.4, [("and at ", 32), ("$B$", 36, ORANGE), (", 50 ft up the second.", 32)]),
    (7.6, [("The rope is 120 ft long and always taut.", 32)]),
    (9.0, [("Question: ", 32, BLUE, "bold"), ("what curve does the fold point ", 32, BLUE),
           ("$P$", 36, ORANGE), (" trace?", 32, BLUE)]),
]


@functools.lru_cache(maxsize=None)
def setup_panel(alphas):
    """All rows are laid out from the start (so the panel never resizes);
    each line fades in with its own alpha."""
    rows = [[("The setup", 34, DARK, "bold")]]
    for _, pieces in SETUP_LINES:
        rows.append([(p[0], p[1], p[2] if len(p) > 2 else DARK, p[3] if len(p) > 3 else "normal")
                     for p in pieces])
    return stack(rows, row_alpha=(1.0,) + alphas)


def readout_panel(P, highlight_sum):
    A, B = (g.AX, g.AZ), (g.BX, g.BZ)
    ap, pb = g.dist(A, P), g.dist(P, B)
    px = 40
    left = [r"$|AP|$", r"$|PB|$", r"$|AP| + |PB|$"]
    right = [rf"$=\ {ap:5.1f}\ \mathrm{{ft}}$", rf"$=\ {pb:5.1f}\ \mathrm{{ft}}$",
             rf"$=\ {ap + pb:5.1f}\ \mathrm{{ft}}$"]
    sum_col = BLUE if highlight_sum else DARK
    imgs_l = [text_img(s, px, sum_col if i == 2 else DARK) for i, s in enumerate(left)]
    imgs_r = [text_img(s, px, sum_col if i == 2 else DARK) for i, s in enumerate(right)]
    wl = max(im.size[0] for im, _ in imgs_l)
    # fixed right-column width (widest possible value) so the panel is steady
    wr = text_img(r"$=\ 888.8\ \mathrm{ft}$", px)[0].size[0]
    padx, pady, pitch = 30, 18, int(px * 1.45)
    hgt = 2 * pady + 3 * pitch + 14
    panel = rounded((2 * padx + wl + wr, hgt), PANEL)
    y = pady
    for i in range(3):
        if i == 2:
            d = ImageDraw.Draw(panel)
            d.line((padx, y + 7, padx + wl + wr, y + 7), fill=(0, 0, 0, 70), width=2)
            y += 14
        base = y + int(pitch * 0.74)
        (il, bl), (ir, br) = imgs_l[i], imgs_r[i]
        panel.alpha_composite(il, (padx + wl - il.size[0], max(0, base - bl)))
        panel.alpha_composite(ir, (padx + wl, max(0, base - br)))
        y += pitch
    return panel


CAPTIONS = [
    # (start, end, rows)
    (11.8, 28.6, [[("The bottle slides along the taut rope; the fold point ", 32),
                   ("$P$", 36, ORANGE), (" traces the ", 32), ("blue path", 32, BLUE, "bold"), (".", 32)]]),
    (29.4, 32.6, [[("Now look straight at the vertical plane of the two trees.", 32)]]),
    (36.2, 39.8, [[("The taut rope is always two straight pieces, so", 32)],
                  [("$|AP| + |PB| = 120\\ \\mathrm{ft}$", 40, BLUE), (",  the same at every moment.", 32)]]),
    (40.0, 47.8, [[("Keep the rope taut and take ", 32), ("$P$", 36, ORANGE),
                   (" all the way round: it draws an ", 32), ("ellipse", 32, BLUE, "bold"), (",", 32)],
                  [("the curve of points whose distances to ", 32), ("$A$", 36, ORANGE), (" and ", 32),
                   ("$B$", 36, ORANGE), (" add up to a constant.", 32)]]),
    (48.0, 52.0, [[("The foci are ", 32), ("$A$", 36, ORANGE), (" and ", 32), ("$B$", 36, ORANGE),
                   (".  The bottle hangs below the rope, between the trees,", 32)],
                  [("so it travels only the ", 32), ("solid arc", 32, BLUE, "bold"),
                   ("; the dashed part is out of its reach.", 32)]]),
    (52.2, 56.0, [[("The ellipse exists because the rope is longer than the gap between its ends:", 32)],
                  [(r"$120\ \mathrm{ft} > |AB| = \sqrt{100^2 + 50^2} \approx 111.8\ \mathrm{ft}$", 40)]]),
]


@functools.lru_cache(maxsize=None)
def caption_panel(i):
    rows = [[(p[0], p[1], p[2] if len(p) > 2 else DARK, p[3] if len(p) > 3 else "normal")
             for p in row] for row in CAPTIONS[i][2]]
    return stack(rows, align="center")


@functools.lru_cache(maxsize=None)
def question_panel():
    return stack([
        [("Ignoring the numbers: what kind of curve is the path?", 34, DARK, "bold")],
        ("img", chip_row(32)),
    ], align="center", pady=20)


@functools.lru_cache(maxsize=None)
def answer_panel():
    return stack([
        [("Answer:  an ", 44, DARK, "bold"), ("ellipse", 44, BLUE, "bold")],
        ("img", chip_row(30, "ellipse")),
        [("$P$", 34, ORANGE), (" moves on an arc of the ellipse with foci ", 30), ("$A$", 34, ORANGE),
         (" and ", 30), ("$B$", 34, ORANGE), (", in the vertical plane of the trees.", 30)],
    ], align="center", pady=22)


def place(canvas, im, anchor, xy, opacity):
    if opacity <= 0.003:
        return
    w, h = im.size
    x, y = xy
    if anchor == "tc":
        x -= w // 2
    elif anchor == "tr":
        x -= w
    elif anchor == "bc":
        x -= w // 2
        y -= h
    if opacity < 0.997:
        im = im.copy()
        a = np.asarray(im.getchannel("A"), dtype=np.float32) * opacity
        im.putalpha(Image.fromarray(a.astype(np.uint8)))
    canvas.alpha_composite(im, (int(x), int(y)))


def compose(frame, base):
    t = frame / g.FPS
    st = g.state(frame)
    img = base.convert("RGBA")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)

    place(img, title_card(), "tc", (W // 2, 70), g.fade(t, 0.3, 0.9, 3.4, 3.9))

    alphas = tuple(round(g.ramp(t, t0, t0 + 0.5), 2) for t0, _ in SETUP_LINES)
    place(img, setup_panel(alphas), "tr", (W - 40, 40), g.fade(t, 4.3, 4.7, 10.7, 11.2))

    ro = g.fade(t, 11.3, 11.9, g.S7, g.S7 + 0.5)
    if ro > 0:
        place(img, readout_panel(st["P"], t >= 36.0), "tr", (W - 40, 40), ro)

    for i, (t0, t1, _) in enumerate(CAPTIONS):
        o = g.fade(t, t0, t0 + 0.4, t1 - 0.4, t1)
        if o > 0:
            place(img, caption_panel(i), "bc", (W // 2, H - 36), o)

    place(img, question_panel(), "bc", (W // 2, H - 36), g.fade(t, 32.6, 33.0, 35.8, 36.2))
    place(img, answer_panel(), "bc", (W // 2, H - 36), g.fade(t, 57.4, 58.0))
    # fade from / to black at the very ends
    k = min(g.ramp(t, 0.0, 0.5), 1 - g.ramp(t, g.END - 0.7, g.END - 0.05))
    out = img.convert("RGB")
    if k < 0.999:
        out = Image.blend(Image.new("RGB", out.size, (0, 0, 0)), out, k)
    return out


def main():
    args = sys.argv[1:]
    frames_dir, out = args[0], args[1]
    if "--frames" in args:
        fl = [int(x) for x in args[args.index("--frames") + 1].split(",")]
        os.makedirs(out, exist_ok=True)
        for f in fl:
            base = Image.open(os.path.join(frames_dir, f"f{f:04d}.png"))
            compose(f, base).save(os.path.join(out, f"o{f:04d}.png"))
        return
    n = int(round(g.END * g.FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(g.FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(n):
        base = Image.open(os.path.join(frames_dir, f"f{f:04d}.png"))
        p.stdin.write(compose(f, base).tobytes())
        if f % 150 == 0:
            print("overlay frame", f, flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    main()
