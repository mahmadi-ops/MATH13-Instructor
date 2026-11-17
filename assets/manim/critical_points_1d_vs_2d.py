r"""Critical points in one and two variables: a curve y = f(x) beside a surface
z = f(x, y).

    manim -qh --fps 20 --disable_caching -o critical_points_1d_vs_2d_1080p \
        critical_points_1d_vs_2d.py CriticalPoints1D2D

Video of fig-critical-1d-2d-video, at the end of the subsection "Saddle
Point" of extreme-values-saddle-points.ptx, copied to assets/videos.  It
uses the examples of the section and their one-variable counterparts:

A. Title.
B. The two definitions of a critical point side by side.
C. Local minimum: y = x^2 beside z = x^2 + y^2 (horizontal tangent line and
   plane, open interval and open disk); the slice y = 0 of the surface is the
   curve on the left.
D. Local maximum: y = -x^2 beside z = -x^2 - y^2, the same way.
E. A critical point where the derivative does not exist: the corner of
   y = |x| beside the tip of the cone z = sqrt(x^2 + y^2), whose trace in
   y = 0 is |x|.
F. The saddle z = y^2/3 - x^2/3: it bends down along the x-axis and up along
   the y-axis.  A curve y = f(x) has one direction through the point, so it
   bends one way only (f''(0) > 0 or < 0); y = x^3, neither a maximum nor a
   minimum, does not bend at 0 at all.  The saddle has no one-variable
   counterpart.
G. Summary table.
"""

import os

import numpy as np
from manim import *

TEX = TexTemplate(preamble=(
    r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}" "\n"
    r"\AtBeginDocument{\setlength{\textwidth}{1500pt}\setlength{\linewidth}{1500pt}"
    r"\setlength{\hsize}{1500pt}\setlength{\columnwidth}{1500pt}}"))
config.tex_template = TEX
config.background_color = BLACK

FAST = os.environ.get("FAST", "") == "1"
STAGES = os.environ.get("STAGES", "ABCDEFG")

# ---- colours ----------------------------------------------------------------
CURVE_C = "#5CC8FF"     # the curve on the left and its twin, the slice y = 0
TAN_C = "#E0B43C"       # horizontal tangent line and plane
NBHD_C = "#7FC97F"      # open interval and open disk
UP_C = "#5CC8FF"        # saddle: the trace that bends up (blue, as in the notes)
DOWN_C = "#E8625A"      # saddle: the trace that bends down (red, as in the notes)
CORNER_C = "#7FC97F"    # the trace |x| of the cone (green, as in the notes)
MIN_C = "#7FC97F"
MAX_C = "#E8625A"
SAD_C = "#F2C94C"
DIM = GREY_B
SURF = ["#4A5BA6", "#36447F"]
CONE = ["#4F7A9A", "#3A5C75"]
SADDLE = ["#8C8C8C", "#5A5A5A"]


def T(s, fs=32, color=WHITE):
    return Tex(s, font_size=fs, color=color)


def W(t):
    return 0.3 if FAST else t


# ---- the 3D picture ---------------------------------------------------------
PHI, THETA = 66 * DEGREES, -52 * DEGREES
RIGHT3 = np.array([-np.sin(THETA), np.cos(THETA), 0.0])
O3 = 3.25 * RIGHT3
XY = 1.15
ZOFF = [0.0]
ZS = [0.6]


def W3(x, y, z):
    return O3 + np.array([XY * x, XY * y, ZS[0] * z + ZOFF[0]])


# ---- small pictures for the summary table -----------------------------------
def _view(phi=PHI, th=THETA):
    right = np.array([-np.sin(th), np.cos(th), 0.0])
    up = np.array([-np.cos(phi) * np.cos(th), -np.cos(phi) * np.sin(th), np.sin(phi)])
    d = np.array([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)])
    return right, up, d


def mini_surface(f, kind, colors, width=1.7, height=1.08):
    """A small shaded surface, projected like the 3D scenes (depth-sorted)."""
    right, up, d = _view()
    light = d + 0.6 * up
    light = light / np.linalg.norm(light)
    if kind == "polar":
        P = lambda u, v: np.array([u * np.cos(v), u * np.sin(v), 1.0 * f(u, v)])
        us, vs = np.linspace(0, 1.0, 7), np.linspace(0, TAU, 29)
    else:
        P = lambda u, v: np.array([u, v, 1.0 * f(u, v)])
        us, vs = np.linspace(-1, 1, 11), np.linspace(-1, 1, 11)
    faces = []
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            q = [P(us[i], vs[j]), P(us[i + 1], vs[j]), P(us[i + 1], vs[j + 1]), P(us[i], vs[j + 1])]
            n = np.cross(q[1] - q[0], q[3] - q[0])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                n = np.cross(q[2] - q[1], q[3] - q[1]); nn = np.linalg.norm(n)
            shade = 0.45 + 0.55 * abs(n @ light) / max(nn, 1e-9)
            depth = np.mean([p @ d for p in q])
            pts = [np.array([p @ right, p @ up, 0.0]) for p in q]
            base = colors[(i + j) % 2]
            faces.append((depth, Polygon(*pts, stroke_color=BLACK, stroke_width=0.4,
                                         fill_color=interpolate_color(BLACK, ManimColor(base), shade),
                                         fill_opacity=1.0)))
    faces.sort(key=lambda t: t[0])
    g = VGroup(*[p for _, p in faces])
    g.scale_to_fit_width(width)
    if g.height > height:
        g.scale_to_fit_height(height)
    return g


def mini_curve(f, color, w=1.45, h=1.0):
    ax = Axes(x_range=[-1.15, 1.15], y_range=[-1.2, 1.2], x_length=w, y_length=h, tips=False,
              axis_config=dict(color=GREY_C, stroke_width=1.5, include_ticks=False))
    c = ax.plot(f, x_range=[-1.05, 1.05], color=color, stroke_width=3.5)
    dot = Dot(ax.c2p(0, 0), radius=0.045, color=WHITE)
    return VGroup(ax, c, dot)


# ---- the 2D picture ---------------------------------------------------------
LEFT_O = np.array([-3.75, -0.15, 0.0])


class CriticalPoints1D2D(ThreeDScene):

    # ------------------------------------------------------------- helpers
    def fix(self, *mobs):
        self.add_fixed_in_frame_mobjects(*mobs)
        self.remove(*mobs)
        return mobs

    def clear_all(self, rt=0.8):
        ms = list(self.mobjects)
        if ms:
            self.play(*[FadeOut(m) for m in ms], run_time=rt)
        self.remove(*self.mobjects)

    def flat_view(self):
        self.set_camera_orientation(phi=0, theta=-PI / 2)

    def space_view(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)

    def headers(self, right=True):
        l = T(r"one variable: \;$y = f(x)$", 34).move_to([-3.75, 3.5, 0])
        g = VGroup(l)
        if right:
            r = T(r"two variables: \;$z = f(x,y)$", 34).move_to([3.35, 3.5, 0])
            g.add(r)
        div = DashedLine([-0.25, 3.15, 0], [-0.25, -2.85, 0], color=GREY_D,
                         dash_length=0.12, stroke_width=1.5)
        g.add(div)
        self.fix(*g)
        return g

    def left_axes(self):
        ax = Axes(x_range=[-2, 2, 1], y_range=[-2.6, 2.8, 1], x_length=5.0,
                  y_length=4.0, tips=True,
                  axis_config=dict(color=GREY_B, stroke_width=2, include_ticks=False,
                                   tip_width=0.18, tip_height=0.18))
        ax.shift(LEFT_O - ax.c2p(0, 0))
        xl = MathTex("x", font_size=34).next_to(ax.x_axis.get_end(), DOWN, buff=0.15)
        yl = MathTex("y", font_size=34).next_to(ax.y_axis.get_end(), LEFT, buff=0.15)
        g = VGroup(ax, xl, yl)
        self.fix(g)
        return ax, g

    def axes3d(self, zr):
        ax = VGroup()
        for a, b in [((-2.0, 0, 0), (2.3, 0, 0)), ((0, -2.0, 0), (0, 2.3, 0)),
                     ((0, 0, zr[0]), (0, 0, zr[1]))]:
            ax.add(Arrow3D(W3(*a), W3(*b), color=GREY_A, thickness=0.006,
                           height=0.18, base_radius=0.05))
        labs = [MathTex("x", font_size=34).move_to(W3(2.6, 0, 0)),
                MathTex("y", font_size=34).move_to(W3(0, 2.6, 0)),
                MathTex("z", font_size=34).move_to(W3(0, 0, zr[1] + 0.35))]
        self.add_fixed_orientation_mobjects(*labs)
        self.remove(*labs)
        return ax, labs

    def polar_surface(self, h, R, colors):
        return Surface(lambda r, t: W3(r * np.cos(t), r * np.sin(t), h(r)),
                       u_range=[0.0, R], v_range=[0, TAU], resolution=(12, 36),
                       checkerboard_colors=colors, fill_opacity=0.9,
                       stroke_width=0.2, stroke_color=BLACK)

    def square_surface(self, f, R, colors):
        return Surface(lambda u, v: W3(u, v, f(u, v)), u_range=[-R, R], v_range=[-R, R],
                       resolution=(22, 22), checkerboard_colors=colors, fill_opacity=0.9,
                       stroke_width=0.2, stroke_color=BLACK)

    def tangent_plane(self, a=0.95):
        return Polygon(W3(-a, -a, 0), W3(a, -a, 0), W3(a, a, 0), W3(-a, a, 0),
                       fill_color=TAN_C, fill_opacity=0.45, stroke_color=TAN_C,
                       stroke_width=1)

    def disk(self, r=0.8):
        pts = [W3(r * np.cos(t), r * np.sin(t), 0) for t in np.linspace(0, TAU, 49)[:-1]]
        return Polygon(*pts, fill_color=NBHD_C, fill_opacity=0.55, stroke_color=NBHD_C,
                       stroke_width=2)

    def slice_plane(self, zlo, zhi, along="x", r=2.0):
        if along == "x":
            pts = [W3(-r, 0, zlo), W3(r, 0, zlo), W3(r, 0, zhi), W3(-r, 0, zhi)]
        else:
            pts = [W3(0, -r, zlo), W3(0, r, zlo), W3(0, r, zhi), W3(0, -r, zhi)]
        return Polygon(*pts, fill_color=GREY_E, fill_opacity=0.4, stroke_width=0)

    def trace(self, g, R, color, along="x"):
        if along == "x":
            fn = lambda s: W3(s, 0, g(s))
        else:
            fn = lambda s: W3(0, s, g(s))
        return ParametricFunction(fn, t_range=[-R, R], color=color, stroke_width=7)

    # ------------------------------------------------------------- A
    def title(self):
        self.flat_view()
        t1 = T(r"\textbf{Critical Points in One and Two Variables}", 56)
        t2 = T(r"a curve $y = f(x)$ in the plane \quad and \quad a surface $z = f(x,y)$ in space",
               34, CURVE_C)
        t3 = T(r"MATH 13 \;---\; Extreme Values and Saddle Points", 26, DIM)
        g = VGroup(t1, t2, t3).arrange(DOWN, buff=0.5)
        self.play(FadeIn(t1, shift=0.2 * UP), run_time=1.2)
        self.play(FadeIn(t2), FadeIn(t3), run_time=1.0)
        self.wait(W(2.5))
        self.play(FadeOut(g), run_time=0.8)

    # ------------------------------------------------------------- B
    def definitions(self):
        self.flat_view()
        left = VGroup(
            T(r"\textbf{One variable}", 40, CURVE_C),
            T(r"$a$ is a critical point of $f(x)$ if", 32),
            T(r"$f'(a) = 0$", 32),
            T(r"or $f'(a)$ does not exist", 32),
            T(r"the tangent line is horizontal, or there is none", 28, DIM),
        ).arrange(DOWN, buff=0.4)
        right = VGroup(
            T(r"\textbf{Two variables}", 40, CURVE_C),
            T(r"$(a,b)$ is a critical point of $f(x,y)$ if", 32),
            T(r"$f_x(a,b) = f_y(a,b) = 0$", 32),
            T(r"or one of them does not exist", 32),
            T(r"the tangent plane is horizontal, or there is none", 28, DIM),
        ).arrange(DOWN, buff=0.4)
        left.move_to([-3.6, 0.7, 0])
        right.move_to([3.4, 0.7, 0])
        div = DashedLine([-0.25, 2.6, 0], [-0.25, -1.3, 0], color=GREY_D,
                         dash_length=0.12, stroke_width=1.5)
        foot = T(r"Critical points are the candidates for local maxima and minima.", 30)
        foot.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(left[0]), FadeIn(right[0]), Create(div), run_time=0.8)
        self.play(FadeIn(left[1:4]), run_time=1.0)
        self.play(FadeIn(right[1:4]), run_time=1.0)
        self.play(FadeIn(left[4]), FadeIn(right[4]), run_time=0.8)
        self.play(FadeIn(foot), run_time=0.8)
        self.wait(W(4.0))
        self.clear_all()

    # ------------------------------------------------------------- C, D
    def extremum(self, kind):
        is_min = kind == "min"
        sgn = 1.0 if is_min else -1.0
        name = "Local minimum" if is_min else "Local maximum"
        col = MIN_C if is_min else MAX_C
        f1 = (lambda x: x * x) if is_min else (lambda x: -x * x)
        h = (lambda r: r * r) if is_min else (lambda r: -r * r)
        tex1 = r"y = x^2" if is_min else r"y = -x^2"
        tex2 = r"z = x^2 + y^2" if is_min else r"z = -x^2 - y^2"
        ineq1 = r"f(0) \le f(x)" if is_min else r"f(0) \ge f(x)"
        ineq2 = r"f(0,0) \le f(x,y)" if is_min else r"f(0,0) \ge f(x,y)"
        ZOFF[0] = -0.75 if is_min else 0.75
        ZS[0] = 0.6

        self.space_view()
        hdr = self.headers()
        ttl = T(r"\textbf{" + name + "}", 40, col).move_to([0, -3.55, 0])
        self.fix(ttl)
        ax, axg = self.left_axes()
        curve = ax.plot(f1, x_range=[-1.6, 1.6], color=CURVE_C, stroke_width=5)
        clab = MathTex(tex1, font_size=36, color=CURVE_C).next_to(
            ax.c2p(1.6, f1(1.6)), RIGHT, buff=0.12)
        dot2 = Dot(ax.c2p(0, 0), radius=0.08, color=WHITE)
        tline = Line(ax.c2p(-0.95, 0), ax.c2p(0.95, 0), color=TAN_C, stroke_width=6)
        tlab = MathTex(r"f'(0) = 0", font_size=34, color=TAN_C).next_to(
            ax.c2p(0, 0), -sgn * UP * 1.0 + RIGHT * 0.0, buff=0.35).shift(1.35 * RIGHT)
        intv = Line(ax.c2p(-0.75, 0), ax.c2p(0.75, 0), color=NBHD_C, stroke_width=11)
        ends = VGroup(*[Circle(radius=0.07, color=NBHD_C, stroke_width=3).set_fill(BLACK, 1)
                        .move_to(ax.c2p(e, 0)) for e in (-0.75, 0.75)])
        ilab = T(rf"${ineq1}$ on an open interval", 30, NBHD_C).move_to([-3.75, -2.6, 0])
        self.fix(curve, clab, dot2, tline, tlab, intv, ends, ilab)

        zr = (-0.3, 3.3) if is_min else (-3.2, 0.9)
        ax3, labs = self.axes3d(zr)
        surf = self.polar_surface(h, 1.6, SURF)
        dot3 = Dot3D(W3(0, 0, 0), radius=0.07, color=WHITE)
        tp = self.tangent_plane()
        dk = self.disk()
        tlab3 = T(r"$f_x(0,0) = f_y(0,0) = 0$", 30, TAN_C).move_to([3.35, -2.0, 0])
        dlab = T(rf"${ineq2}$ on an open disk", 30, NBHD_C).move_to([3.35, -2.6, 0])
        slab = MathTex(tex2, font_size=36, color=WHITE).move_to([5.6, 2.8, 0])
        self.fix(tlab3, dlab, slab)

        # the two graphs
        self.play(FadeIn(hdr), FadeIn(ttl), FadeIn(axg), run_time=0.8)
        self.play(Create(curve), FadeIn(clab), Create(ax3), FadeIn(surf), FadeIn(slab),
                  *[FadeIn(l) for l in labs], run_time=2.0)
        self.add(dot2, dot3)
        self.wait(W(1.0))
        # horizontal tangent line and plane
        self.play(Create(tline), FadeIn(tlab), FadeIn(tp), FadeIn(tlab3), run_time=1.2)
        self.wait(W(2.5))
        # open interval and open disk
        self.play(FadeOut(tline), FadeOut(tp), FadeIn(intv), FadeIn(ends), FadeIn(ilab),
                  FadeIn(dk), FadeIn(dlab), run_time=1.2)
        self.add(dot2, dot3)
        self.wait(W(3.0))
        # the slice y = 0 is the curve
        sp = self.slice_plane(*((-0.4, 3.0) if is_min else (-3.0, 0.4)))
        tr = self.trace(f1, 1.6, CURVE_C)
        say = T(r"the slice $y = 0$ of the surface is the curve on the left", 28, CURVE_C)
        say.move_to([3.35, -2.6, 0])
        self.fix(say)
        self.play(FadeOut(dk), FadeOut(dlab), FadeOut(tlab3), FadeIn(sp), Create(tr),
                  FadeIn(say), run_time=1.6)
        self.add(dot3)
        self.wait(W(3.5))
        self.clear_all()

    # ------------------------------------------------------------- E
    def corner(self):
        ZOFF[0] = -0.55
        ZS[0] = 0.75
        self.space_view()
        hdr = self.headers()
        ttl = T(r"\textbf{Critical point where the derivative does not exist}", 38, MIN_C)
        ttl.move_to([0, -3.55, 0])
        self.fix(ttl)
        ax, axg = self.left_axes()
        curve = VGroup(ax.plot(lambda x: -x, x_range=[-1.6, 0], color=CORNER_C, stroke_width=5),
                       ax.plot(lambda x: x, x_range=[0, 1.6], color=CORNER_C, stroke_width=5))
        clab = MathTex(r"y = \lvert x \rvert", font_size=36, color=CORNER_C).next_to(ax.c2p(1.6, 1.6), RIGHT, buff=0.12)
        dot2 = Dot(ax.c2p(0, 0), radius=0.08, color=WHITE)
        sl = T(r"slope $-1$", 28).next_to(ax.c2p(-0.85, 0.85), LEFT, buff=0.25)
        sr = T(r"slope $1$", 28).next_to(ax.c2p(0.85, 0.85), RIGHT, buff=0.25)
        nd = T(r"$f'(0)$ does not exist: the slope jumps at the corner", 30, TAN_C)
        nd.move_to([-3.75, -2.6, 0])
        self.fix(curve, clab, dot2, sl, sr, nd)

        ax3, labs = self.axes3d((-0.3, 2.8))
        surf = self.polar_surface(lambda r: r, 1.8, CONE)
        dot3 = Dot3D(W3(0, 0, 0), radius=0.07, color=WHITE)
        slab = MathTex(r"z = \sqrt{x^2 + y^2}", font_size=36).move_to([5.5, 2.8, 0])
        nd3 = T(r"$f_x(0,0)$, $f_y(0,0)$ do not exist: a sharp tip", 30, TAN_C)
        nd3.move_to([3.35, -2.0, 0])
        sp = self.slice_plane(-0.3, 2.2)
        tr = self.trace(lambda s: abs(s), 1.8, CORNER_C)
        say = T(r"its slice $y = 0$ is the curve $y = \lvert x \rvert$", 30, CORNER_C)
        say.move_to([3.35, -2.6, 0])
        self.fix(slab, nd3, say)

        self.play(FadeIn(hdr), FadeIn(ttl), FadeIn(axg), run_time=0.8)
        self.play(Create(curve), FadeIn(clab), Create(ax3), FadeIn(surf), FadeIn(slab),
                  *[FadeIn(l) for l in labs], run_time=2.0)
        self.add(dot2, dot3)
        self.play(FadeIn(sl), FadeIn(sr), run_time=0.8)
        self.play(FadeIn(nd), FadeIn(nd3), run_time=1.0)
        self.wait(W(3.0))
        self.play(FadeIn(sp), Create(tr), FadeIn(say), run_time=1.4)
        self.add(dot3)
        self.wait(W(2.0))
        also = T(r"(and since $f(0) = 0$ is the smallest value, both are also local minima)", 26, DIM)
        also.move_to([0, -3.05, 0])
        self.fix(also)
        self.play(FadeIn(also), run_time=0.8)
        self.wait(W(3.0))
        self.clear_all()

    # ------------------------------------------------------------- F
    def saddle(self):
        ZOFF[0] = 0.15
        ZS[0] = 0.95
        self.space_view()
        hdr = self.headers()
        ttl = T(r"\textbf{Saddle point}", 40, SAD_C).move_to([0, -3.55, 0])
        self.fix(ttl)

        fx = lambda x: -x * x / 3.0
        fy = lambda y: y * y / 3.0
        ax3, labs = self.axes3d((-1.6, 1.8))
        surf = self.square_surface(lambda x, y: (y * y - x * x) / 3.0, 1.8, SADDLE)
        dot3 = Dot3D(W3(0, 0, 0), radius=0.07, color=WHITE)
        slab = MathTex(r"z = \tfrac{y^2}{3} - \tfrac{x^2}{3}", font_size=36).move_to([5.5, 2.8, 0])
        tp = self.tangent_plane(0.8)
        tx = self.trace(fx, 1.8, DOWN_C, "x")
        ty = self.trace(fy, 1.8, UP_C, "y")
        lx = T(r"along the $x$-axis: \;$z = -x^2/3$, bends down", 28, DOWN_C)
        ly = T(r"along the $y$-axis: \;$z = y^2/3$, bends up", 28, UP_C)
        VGroup(lx, ly).arrange(DOWN, buff=0.15).move_to([3.35, -2.35, 0])
        self.fix(slab, lx, ly)

        self.play(FadeIn(hdr), FadeIn(ttl), run_time=0.8)
        self.play(Create(ax3), FadeIn(surf), FadeIn(slab), *[FadeIn(l) for l in labs],
                  run_time=2.0)
        self.add(dot3)
        self.play(FadeIn(tp), run_time=0.8)
        self.wait(W(1.5))
        self.play(FadeOut(tp), Create(tx), FadeIn(lx), run_time=1.4)
        self.wait(W(1.5))
        self.play(Create(ty), FadeIn(ly), run_time=1.4)
        self.add(dot3)
        self.wait(W(3.0))

        # the left side: is there a counterpart?
        ax, axg = self.left_axes()
        q = T(r"a counterpart in one variable?", 32, SAD_C).move_to([-3.75, 2.75, 0])
        c_dn = ax.plot(fx, x_range=[-1.9, 1.9], color=DOWN_C, stroke_width=5)
        c_up = ax.plot(fy, x_range=[-1.9, 1.9], color=UP_C, stroke_width=5)
        dot2 = Dot(ax.c2p(0, 0), radius=0.08, color=WHITE)
        t_dn = T(r"$y = -x^2/3$: bends down, a maximum", 28, DOWN_C)
        t_up = T(r"$y = x^2/3$: bends up, a minimum", 28, UP_C)
        VGroup(t_dn, t_up).arrange(DOWN, buff=0.12).move_to([-3.75, -2.45, 0])
        self.fix(q, c_dn, c_up, dot2, t_dn, t_up)
        self.play(FadeIn(axg), FadeIn(q), run_time=0.8)
        self.play(Create(c_dn), FadeIn(t_dn), run_time=1.2)
        self.add(dot2)
        self.wait(W(1.5))
        self.play(Create(c_up), FadeIn(t_up), run_time=1.2)
        self.add(dot2)
        self.wait(W(2.0))

        one = T(r"one function $y = f(x)$ gives one curve, with one direction through $0$:", 30)
        one2 = T(r"it bends one way only, \;$f''(0) > 0$ (minimum) or $f''(0) < 0$ (maximum)", 30, CURVE_C)
        VGroup(one, one2).arrange(DOWN, buff=0.12).move_to([0, -3.0, 0])
        self.fix(one, one2)
        self.play(FadeOut(ttl), FadeOut(t_dn), FadeOut(t_up), FadeOut(lx), FadeOut(ly),
                  FadeIn(one), FadeIn(one2), run_time=1.0)
        self.wait(W(4.5))

        # y = x^3 is no saddle either
        cube = ax.plot(lambda x: x ** 3, x_range=[-1.37, 1.37], color=WHITE, stroke_width=5)
        tcube = T(r"$y = x^3$: neither a maximum nor a minimum,", 30)
        tcube2 = T(r"but not a saddle: \;$f''(0) = 0$, it does not bend at $0$, it only flattens out", 30, DIM)
        VGroup(tcube, tcube2).arrange(DOWN, buff=0.12).move_to([0, -3.0, 0])
        clab = MathTex(r"y = x^3", font_size=36).next_to(ax.c2p(1.37, 1.37 ** 3), RIGHT, buff=0.12)
        self.fix(cube, tcube, tcube2, clab)
        self.play(FadeOut(c_dn), FadeOut(c_up), FadeOut(one), FadeOut(one2),
                  Create(cube), FadeIn(clab), FadeIn(tcube), FadeIn(tcube2), run_time=1.4)
        self.add(dot2)
        self.wait(W(4.5))

        fin1 = T(r"A saddle bends up in one direction and down in another.", 32, SAD_C)
        fin2 = T(r"A curve has only one direction: \;the saddle has no one-variable counterpart.", 32, SAD_C)
        VGroup(fin1, fin2).arrange(DOWN, buff=0.12).move_to([0, -3.0, 0])
        self.fix(fin1, fin2)
        cross = VGroup(Line(ax.c2p(-1.9, -2.4), ax.c2p(1.9, 2.6)),
                       Line(ax.c2p(-1.9, 2.6), ax.c2p(1.9, -2.4))).set_stroke(DOWN_C, 6, opacity=0.8)
        nocp = T(r"no saddle point", 34, DOWN_C).move_to([-3.75, 2.75, 0])
        self.fix(cross, nocp)
        self.play(FadeOut(tcube), FadeOut(tcube2), FadeOut(cube), FadeOut(clab), FadeOut(q),
                  FadeIn(c_dn), FadeIn(c_up), run_time=1.0)
        self.add(dot2)
        self.play(Create(cross), FadeIn(nocp), FadeIn(fin1), FadeIn(fin2), run_time=1.2)
        self.wait(W(5.0))
        self.clear_all()

    # ------------------------------------------------------------- G
    def summary(self):
        self.flat_view()
        head = T(r"\textbf{Critical points: one variable and two variables}", 44).to_edge(UP, buff=0.45)
        rows = [
            (r"local minimum", r"$y = x^2$", r"$z = x^2 + y^2$", MIN_C,
             lambda x: x * x, lambda r, t: r * r, "polar", SURF, CURVE_C),
            (r"local maximum", r"$y = -x^2$", r"$z = -x^2 - y^2$", MAX_C,
             lambda x: -x * x, lambda r, t: -r * r, "polar", SURF, CURVE_C),
            (r"no derivative", r"$y = \lvert x \rvert$, a corner", r"$z = \sqrt{x^2+y^2}$, a tip", CORNER_C,
             lambda x: abs(x), lambda r, t: r, "polar", CONE, CORNER_C),
            (r"saddle point", r"none", r"$z = \tfrac{y^2}{3} - \tfrac{x^2}{3}$", SAD_C,
             None, lambda x, y: (y * y - x * x) / 1.5, "square", SADDLE, None),
        ]
        X_LAB, X_EQ1, X_G1, X_EQ2, X_G2 = -5.6, -3.05, -0.75, 1.85, 4.95
        hdr = VGroup(T(r"\textbf{one variable}", 32, CURVE_C).move_to([(X_EQ1 + X_G1) / 2, 2.45, 0]),
                     T(r"\textbf{two variables}", 32, CURVE_C).move_to([(X_EQ2 + X_G2) / 2, 2.45, 0]))
        rule = Line([-6.6, 2.1, 0], [6.6, 2.1, 0], color=GREY_D, stroke_width=1.5)
        grid = VGroup()
        y = 1.38
        for (lab, e1, e2, col, f1, f2, kind, cols, ccol) in rows:
            cells = VGroup(T(lab, 32, col).move_to([X_LAB, y, 0]),
                           T(e1, 32, DOWN_C if f1 is None else WHITE).move_to([X_EQ1, y, 0]),
                           T(e2, 32).move_to([X_EQ2, y, 0]))
            if f1 is not None:
                cells.add(mini_curve(f1, ccol).move_to([X_G1, y, 0]))
            cells.add(mini_surface(f2, kind, cols).move_to([X_G2, y, 0]))
            grid.add(cells)
            y -= 1.22
        foot = T(r"A saddle needs two directions to bend in opposite ways; \;one variable has only one.", 30, DIM)
        foot.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(head), FadeIn(hdr), Create(rule), run_time=1.0)
        for r in grid:
            self.play(FadeIn(r), run_time=0.8)
        self.play(FadeIn(foot), run_time=0.8)
        self.wait(W(7.0))
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    def construct(self):
        if "A" in STAGES:
            self.title()
        if "B" in STAGES:
            self.definitions()
        if "C" in STAGES:
            self.extremum("min")
        if "D" in STAGES:
            self.extremum("max")
        if "E" in STAGES:
            self.corner()
        if "F" in STAGES:
            self.saddle()
        if "G" in STAGES:
            self.summary()
