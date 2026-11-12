r"""The tangent plane to z = f(x,y) at P0, from two tangent lines and a cross
product.

    manim -qh --fps 30 --disable_caching -o tangent_plane_cross_product \
        tangent_plane_cross_product.py TangentPlane

It follows "Equation of the Tangent Plane" of the course notes, step by step:

1. The plane y = y0 cuts the surface in a curve whose tangent line at P0 has
   slope f_x(x0, y0): a run of 1 in x comes with a rise of f_x, so
   v1 = i + f_x k is a direction vector of that line.
2. The same in the plane x = x0: v2 = j + f_y k.
3. Both vectors lie in the tangent plane, so v1 x v2 = -f_x i - f_y j + k is
   normal to it.
4. The plane through P0 with that normal vector; multiplied by -1 it is
   f_x (x - x0) + f_y (y - y0) - (z - z0) = 0.
5. Its coefficients are the components of n = v2 x v1 = grad w at P0, where
   w = f(x,y) - z; n is also the direction vector of the normal line.

The 3D picture is drawn with a fixed orthographic projection inside an
ordinary Scene (faces depth-sorted once), as in dirderiv_partials.py; z is
drawn at its true scale here, so that the normal vector looks normal.  The
surface is only a prop for "a general f"; its formula is never shown.  Both
slopes are negative at the chosen point, so both rises point down, as in the
figure of the notes.
"""

import os

import numpy as np
from manim import *

# One long line per Tex object: widen the page so that nothing wraps.
TEX = TexTemplate(preamble=(
    r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}" "\n"
    r"\AtBeginDocument{\setlength{\textwidth}{1500pt}\setlength{\linewidth}{1500pt}"
    r"\setlength{\hsize}{1500pt}\setlength{\columnwidth}{1500pt}}"))
config.tex_template = TEX
config.background_color = BLACK

# ---- the construction -------------------------------------------------------
X0, Y0 = 0.75, 1.3                      # P0 (shown only as (x0, y0))
R_S = 2.6                               # the surface is drawn over the quarter disc r <= R_S
C_F = 0.25


def f(x, y):
    """The surface: a prop for a general f, never named on screen."""
    return 2.6 - C_F * (x * x + y * y)


Z0 = f(X0, Y0)
FX, FY = -2 * C_F * X0, -2 * C_F * Y0   # the two slopes at P0 (both negative)
P0 = np.array([X0, Y0, Z0])
V1 = np.array([1.0, 0.0, FX])
V2 = np.array([0.0, 1.0, FY])
CR = np.cross(V1, V2)                   # (-f_x, -f_y, 1)
NV = -CR                                # n = v2 x v1 = (f_x, f_y, -1)

# ---- colours ----------------------------------------------------------------
V1_C = "#7CE07C"                        # the slice y = y0, its tangent line, v1
V2_C = "#FFB000"                        # the slice x = x0, its tangent line, v2
CR_C = YELLOW                           # v1 x v2
N_C = "#FF7AD9"                         # n and the normal line
PL_C = "#E88A8A"                        # the tangent plane
KEY_C = YELLOW

FAST = os.environ.get("FAST", "") == "1"


class View3:
    """Orthographic camera at azimuth az, elevation el: centre + scale * proj."""

    def __init__(self, center, scale, az=25 * DEGREES, el=22 * DEGREES):
        self.center = np.array([center[0], center[1], 0.0])
        self.scale = float(scale)
        self.az, self.el = az, el

    def proj(self, x, y, z=0.0):
        ca, sa = np.cos(self.az), np.sin(self.az)
        sx = -x * sa + y * ca
        sy = z * np.cos(self.el) - (x * ca + y * sa) * np.sin(self.el)
        return np.array([sx, sy, 0.0])

    def p(self, x, y, z=0.0):
        return self.center + self.scale * self.proj(x, y, z)

    def q(self, pt):
        return self.p(pt[0], pt[1], pt[2])

    def qo(self, pt, dx, dy):
        """The point pt, then moved by (dx, dy) on screen (screen units)."""
        return self.q(pt) + np.array([dx, dy, 0.0])

    def depth(self, x, y, z=0.0):
        ca, sa = np.cos(self.az), np.sin(self.az)
        return (x * ca + y * sa) * np.cos(self.el) + z * np.sin(self.el)

    def cam(self):
        return np.array([np.cos(self.az) * np.cos(self.el),
                         np.sin(self.az) * np.cos(self.el), np.sin(self.el)])


def arrow(a, b, color, sw=6, tip=0.26, opacity=1.0):
    m = Arrow(a, b, buff=0, color=color, stroke_width=sw, tip_length=tip,
              max_tip_length_to_length_ratio=0.6, max_stroke_width_to_length_ratio=30)
    if opacity < 1.0:
        m.set_opacity(opacity)
    return m


def T(*parts, fs=29, **kw):
    return Tex(*parts, font_size=fs, **kw)


def M(*parts, fs=31, **kw):
    return MathTex(*parts, font_size=fs, **kw)


FXT = r"f_x(x_0,y_0)"
FYT = r"f_y(x_0,y_0)"


class TangentPlane(Scene):
    def w(self, t):
        self.wait(0.15 if FAST else t)

    def construct(self):
        LEFT_X = -6.8
        v = View3((2.55, -0.9), 1.55)

        def put(m, y, x=LEFT_X, limit=None):
            # the upper left of the figure is empty, so the top rows may run longer
            if limit is None:
                limit = 1.5 if y > 1.4 else -0.1
            m.move_to(np.array([x, y, 0]), aligned_edge=LEFT)
            if m.get_right()[0] > limit:
                print("TOO WIDE:", round(m.width, 2), "right edge", round(m.get_right()[0], 2),
                      "limit", limit)
            return m

        # ---- the figure ---------------------------------------------------------
        ax_x = arrow(v.p(0, 0, 0), v.p(2.6, 0, 0), GREY_A, sw=2.5, tip=0.18)
        ax_y = arrow(v.p(0, 0, 0), v.p(0, 2.85, 0), GREY_A, sw=2.5, tip=0.18)
        ax_z = arrow(v.p(0, 0, 0), v.p(0, 0, 2.85), GREY_A, sw=2.5, tip=0.18)
        lx = M("x", fs=32).move_to(v.qo((2.6, 0, 0), -0.24, -0.04))
        ly = M("y", fs=32).move_to(v.qo((0, 2.85, 0), 0.22, -0.08))
        lz = M("z", fs=32).move_to(v.qo((0, 0, 2.85), -0.22, 0.02))
        axes = VGroup(ax_x, ax_y, ax_z, lx, ly, lz)

        NR, NT = 11, 14
        rs = np.linspace(0.0, R_S, NR + 1)
        ts = np.linspace(0.0, PI / 2, NT + 1)
        cam = v.cam()
        e_r = np.array([-np.sin(v.az), np.cos(v.az), 0.0])
        light = cam + 0.9 * np.array([0, 0, 1.0]) - 0.7 * e_r
        light /= np.linalg.norm(light)
        faces = []
        for i in range(NR):
            for j in range(NT):
                quad = [(rs[i], ts[j]), (rs[i + 1], ts[j]), (rs[i + 1], ts[j + 1]),
                        (rs[i], ts[j + 1])]
                pts3 = [(r * np.cos(t), r * np.sin(t), f(r * np.cos(t), r * np.sin(t)))
                        for r, t in quad]
                if i == 0:
                    pts3 = pts3[1:]
                c = np.mean(pts3, axis=0)
                n = np.array([2 * C_F * c[0], 2 * C_F * c[1], 1.0])
                n /= np.linalg.norm(n)
                b = 0.2 + 0.8 * max(0.0, float(n @ light))
                poly = Polygon(*[v.p(*pt) for pt in pts3],
                               stroke_color=WHITE, stroke_width=0.6, stroke_opacity=0.2,
                               fill_color=interpolate_color(ManimColor("#123A66"),
                                                            ManimColor("#4FA3E8"), b),
                               fill_opacity=0.4)
                faces.append((v.depth(c[0], c[1], c[2]), poly))
        faces.sort(key=lambda t: t[0])
        surface = VGroup(*[poly for _, poly in faces])
        edge = [(r, 0.0) for r in rs] + [(R_S, t) for t in ts] + [(r, PI / 2) for r in rs[::-1]]
        rim = VMobject(color=BLUE_B, stroke_width=1.5, stroke_opacity=0.7)
        rim.set_points_as_corners([v.p(r * np.cos(t), r * np.sin(t),
                                       f(r * np.cos(t), r * np.sin(t))) for r, t in edge])
        surface.add(rim)
        surf_label = M("z = f(x,y)", fs=30, color=BLUE_B).move_to(v.qo((1.9, 0, f(1.9, 0)), -0.95, 0.28))

        g_dot = Dot(v.p(X0, Y0, 0), radius=0.055, color=WHITE)
        g_lab = M("(x_0,y_0)", fs=26).move_to(v.qo((X0, Y0, 0), 0.0, -0.27))
        riser = DashedLine(v.p(X0, Y0, 0), v.q(P0), color=WHITE, stroke_width=2,
                           dash_length=0.07)
        p_dot = Dot(v.q(P0), radius=0.075, color=WHITE).set_z_index(6)
        p_lab = M("P_0", fs=32).move_to(v.qo(P0, -0.02, 0.42)).set_z_index(6)

        # the two slices, their curves and tangent lines
        def slice_parts(along_x, color):
            fixed = Y0 if along_x else X0
            hi = np.sqrt(R_S ** 2 - fixed ** 2)
            us = np.linspace(0.0, hi, 40)
            pt = (lambda u: (u, Y0, f(u, Y0))) if along_x else (lambda u: (X0, u, f(X0, u)))
            gr = (lambda u: (u, Y0, 0.0)) if along_x else (lambda u: (X0, u, 0.0))
            curtain = VMobject(fill_color=color, fill_opacity=0.13, stroke_color=color,
                               stroke_width=1.2, stroke_opacity=0.5)
            curtain.set_points_as_corners([v.p(*gr(0.0))] + [v.p(*pt(u)) for u in us]
                                          + [v.p(*gr(hi)), v.p(*gr(0.0))])
            curve = VMobject(color=color, stroke_width=4.5)
            curve.set_points_as_corners([v.p(*pt(u)) for u in us])
            d = V1 if along_x else V2
            line = Line(v.q(P0 - 1.0 * d), v.q(P0 + 1.45 * d), color=color, stroke_width=3,
                        stroke_opacity=0.85)
            return curtain, curve, line

        cur_y, crv_y, tan_y = slice_parts(True, V1_C)
        cur_x, crv_x, tan_x = slice_parts(False, V2_C)
        lab_py = M(r"\text{plane } y=y_0", fs=27, color=V1_C).move_to(v.qo((2.3, Y0, 0), -0.95, -0.12))
        lab_px = M(r"\text{plane } x=x_0", fs=27, color=V2_C).move_to(v.qo((X0, 2.3, 0), 0.5, -0.3))

        # run 1, rise f_x (or f_y), and the direction vector
        def step_parts(d, color, run_off, rise_off, rise_tex, v_tex, v_off):
            a = P0 + np.array([d[0], d[1], 0.0])
            b = P0 + d
            run = DashedLine(v.q(P0), v.q(a), color=WHITE, stroke_width=2.5, dash_length=0.07)
            rise = DashedLine(v.q(a), v.q(b), color=WHITE, stroke_width=2.5, dash_length=0.07)
            l1 = M("1", fs=27).move_to(v.qo((P0 + a) / 2, *run_off))
            l2 = M(rise_tex, fs=27).move_to(v.qo((a + b) / 2, *rise_off))
            vec = arrow(v.q(P0), v.q(b), color, sw=6.5, tip=0.24).set_z_index(5)
            lv = M(v_tex, fs=32, color=color).move_to(v.qo(b, *v_off)).set_z_index(5)
            return run, rise, l1, l2, vec, lv

        run1, rise1, l11, l1f, vec1, lv1 = step_parts(V1, V1_C, (-0.05, 0.2), (-0.24, 0.0), "f_x",
                                                     r"\mathbf{v}_1", (-0.3, -0.1))
        run2, rise2, l21, l2f, vec2, lv2 = step_parts(V2, V2_C, (0.05, 0.2), (0.24, 0.0), "f_y",
                                                     r"\mathbf{v}_2", (0.3, -0.12))

        para = Polygon(v.q(P0), v.q(P0 + V1), v.q(P0 + V1 + V2), v.q(P0 + V2),
                       stroke_color=WHITE, stroke_width=1.5, stroke_opacity=0.6,
                       fill_color=PL_C, fill_opacity=0.35).set_z_index(3)
        # the tangent plane: the part P0 + a v1 + b v2 with a, b in [LO, HI], ruled like
        # a board so that its tilt can be read
        LO, HI = -0.75, 1.15
        patch = VGroup(Polygon(v.q(P0 + LO * V1 + LO * V2), v.q(P0 + HI * V1 + LO * V2),
                               v.q(P0 + HI * V1 + HI * V2), v.q(P0 + LO * V1 + HI * V2),
                               stroke_color=PL_C, stroke_width=2, stroke_opacity=0.9,
                               fill_color=PL_C, fill_opacity=0.27))
        for a in np.linspace(LO, HI, 6)[1:-1]:
            patch.add(Line(v.q(P0 + a * V1 + LO * V2), v.q(P0 + a * V1 + HI * V2),
                           color=PL_C, stroke_width=1, stroke_opacity=0.45))
            patch.add(Line(v.q(P0 + LO * V1 + a * V2), v.q(P0 + HI * V1 + a * V2),
                           color=PL_C, stroke_width=1, stroke_opacity=0.45))
        patch.set_z_index(2)
        patch_lab = T("tangent plane", fs=26, color=PL_C).move_to(
            v.qo(P0 + HI * V1 + LO * V2, -1.15, 0.38)).set_z_index(4)

        # right-angle marks between v1 x v2 and each of v1, v2
        def right_angle(d):
            e1 = 0.2 * d / np.linalg.norm(d)
            e3 = 0.2 * CR / np.linalg.norm(CR)
            m = VMobject(color=CR_C, stroke_width=2)
            m.set_points_as_corners([v.q(P0 + e1), v.q(P0 + e1 + e3), v.q(P0 + e3)])
            return m.set_z_index(5)

        marks = VGroup(right_angle(V1), right_angle(V2))
        cross = arrow(v.q(P0), v.q(P0 + 1.05 * CR), CR_C, sw=6.5, tip=0.24).set_z_index(5)
        cross_lab = M(r"\mathbf{v}_1\times\mathbf{v}_2", fs=30, color=CR_C).move_to(
            v.qo(P0 + 1.05 * CR, 0.72, 0.05)).set_z_index(5)
        n_arr = arrow(v.q(P0), v.q(P0 + 1.05 * NV), N_C, sw=6.5, tip=0.24, opacity=0.9).set_z_index(5)
        n_lab = M(r"\mathbf{n}", fs=32, color=N_C).move_to(v.qo(P0 + 1.05 * NV, 0.27, -0.05)).set_z_index(5)
        n_line = DashedLine(v.q(P0 + 1.75 * NV), v.q(P0 - 1.45 * NV), color=N_C, stroke_width=2.5,
                            dash_length=0.1).set_z_index(4)
        nl_lab = T("normal line", fs=26, color=N_C).move_to(v.qo(P0 - 1.45 * NV, 0.95, 0.0)).set_z_index(4)

        # ---- opening --------------------------------------------------------------
        title = put(T(r"\textbf{The Tangent Plane: Two Tangent Lines and a Cross Product}",
                      fs=32, color=BLUE_B), 3.62, limit=6.9)
        i1 = put(T(r"We want the plane that touches the surface"), 2.8)
        i2 = put(T(r"$z=f(x,y)$ at $P_0(x_0,y_0,z_0)$, where $z_0=f(x_0,y_0)$."), 2.3)
        self.play(FadeIn(title), run_time=0.9)
        self.play(LaggedStart(*[Create(m) for m in axes[:3]], lag_ratio=0.2), FadeIn(axes[3:]),
                  run_time=1.0)
        self.play(FadeIn(surface, lag_ratio=0.004), FadeIn(surf_label), FadeIn(i1), FadeIn(i2),
                  run_time=1.4)
        self.play(FadeIn(g_dot), FadeIn(g_lab), Create(riser), run_time=0.8)
        self.play(FadeIn(p_dot, scale=1.6), FadeIn(p_lab), run_time=0.7)
        self.w(1.6)

        # ---- one tangent line and its direction vector ------------------------------
        def step(num, plane_tex, slope_tex, var, curtain, curve, line, plane_lab, run, rise,
                 l1, l2, vec, lv, v_parts, y_keep, color, first):
            e1 = put(T(r"\textbf{%d.} The plane $%s$ cuts the surface in a curve." % (num, plane_tex)), 2.8)
            e2 = put(T(r"Its tangent line at $P_0$ has slope $%s$:" % slope_tex), 2.3)
            e3 = put(T(r"a run of $1$ in $%s$ comes with a rise of $%s$." % (var, slope_tex)), 1.8)
            out = [FadeOut(i1), FadeOut(i2)] if first else []
            self.play(*out, FadeIn(e1), FadeIn(curtain), FadeIn(plane_lab), run_time=1.0)
            self.play(Create(curve), run_time=1.3)
            self.play(FadeIn(e2), Create(line), run_time=1.3)
            self.w(1.3)
            self.play(FadeIn(e3), Create(run), FadeIn(l1), run_time=1.0)
            self.play(Create(rise), FadeIn(l2), run_time=1.0)
            formula = M(*v_parts, fs=34)
            formula[0].set_color(color)
            put(formula, y_keep)
            self.play(GrowArrow(vec), FadeIn(lv), run_time=1.0)
            self.play(Write(formula), run_time=1.5)
            self.w(2.8)
            self.play(FadeOut(VGroup(e1, e2, e3)), FadeOut(curtain), FadeOut(plane_lab),
                      line.animate.set_stroke(opacity=0.45), run_time=0.9)
            return formula

        f1 = step(1, "y=y_0", FXT, "x", cur_y, crv_y, tan_y, lab_py, run1, rise1, l11, l1f,
                  vec1, lv1, [r"\mathbf{v}_1", "=", r"\hat{\imath}+" + FXT + r"\,\hat{k}"],
                  1.05, V1_C, True)
        f2 = step(2, "x=x_0", FYT, "y", cur_x, crv_x, tan_x, lab_px, run2, rise2, l21, l2f,
                  vec2, lv2, [r"\mathbf{v}_2", "=", r"\hat{\jmath}+" + FYT + r"\,\hat{k}"],
                  0.45, V2_C, False)

        # ---- the cross product --------------------------------------------------------
        e1 = put(T(r"\textbf{3.} The tangent plane contains both tangent lines,"), 2.8)
        e2 = put(T(r"so $\mathbf{v}_1$ and $\mathbf{v}_2$ are parallel to it, and their"), 2.3)
        e3 = put(T(r"cross product is normal to the tangent plane."), 1.8)
        self.play(FadeIn(e1), FadeIn(e2), FadeIn(para),
                  FadeOut(VGroup(run1, rise1, l11, l1f, run2, rise2, l21, l2f)), run_time=1.3)
        self.play(FadeIn(e3), run_time=0.8)
        det = M(r"\mathbf{v}_1\times\mathbf{v}_2", "=",
                r"\begin{vmatrix}\hat{\imath} & \hat{\jmath} & \hat{k}\\ 1 & 0 & f_x\\ 0 & 1 & f_y\end{vmatrix}",
                fs=32)
        det[0].set_color(CR_C)
        put(det, -0.55)
        res = M("=", r"-" + FXT + r"\,\hat{\imath}-" + FYT + r"\,\hat{\jmath}+\hat{k}", fs=32)
        put(res, -1.72, x=LEFT_X + (det[1].get_left()[0] - det.get_left()[0]))
        res[1].set_color(CR_C)
        self.play(Write(det), run_time=2.2)
        self.w(1.6)
        self.play(GrowArrow(cross), FadeIn(cross_lab), run_time=1.1)
        self.play(Create(marks), run_time=0.7)
        self.play(Write(res), run_time=1.8)
        self.w(3.0)

        # ---- the equation of the plane -------------------------------------------------
        g1 = put(T(r"\textbf{4.} The plane through $P_0$ with normal vector $\langle A,B,C\rangle$"), 2.8)
        g2 = put(T(r"is $A(x-x_0)+B(y-y_0)+C(z-z_0)=0$. Here"), 2.3)
        g3 = put(T(r"$\langle A,B,C\rangle=\mathbf{v}_1\times\mathbf{v}_2$:"), 1.8)
        self.play(FadeOut(VGroup(e1, e2, e3)), run_time=0.5)
        self.play(FadeIn(g1), FadeIn(g2), run_time=1.0)
        self.play(FadeIn(g3), run_time=0.7)
        row1 = M(r"-" + FXT + r"(x-x_0)-" + FYT + r"(y-y_0)+(z-z_0)=0", fs=32)
        put(row1, -2.87, limit=6.9)
        self.play(Write(row1), FadeIn(patch), FadeIn(patch_lab), FadeOut(para), run_time=2.2)
        self.w(1.6)
        times = put(T(r"Multiply by $-1$:", fs=28, color=GREY_A), -3.56, limit=6.9)
        key = T(r"\textbf{Tangent plane:}", color=KEY_C)
        row2 = M(FXT + r"(x-x_0)+" + FYT + r"(y-y_0)-(z-z_0)=0", fs=34)
        key.next_to(times, RIGHT, buff=0.55)
        row2.next_to(key, RIGHT, buff=0.3)
        box = SurroundingRectangle(VGroup(key, row2), color=KEY_C, buff=0.12, stroke_width=2.5,
                                   corner_radius=0.08)
        if box.get_right()[0] > 6.95:
            print("TOO WIDE: tangent plane row", round(box.get_right()[0], 2))
        self.play(FadeIn(times), run_time=0.6)
        self.play(FadeIn(key), TransformFromCopy(row1, row2), run_time=1.6)
        self.play(Create(box), run_time=0.8)
        self.w(3.6)

        # ---- the normal vector as a gradient, and the normal line ------------------------
        self.play(FadeOut(VGroup(g1, g2, g3, f1, f2, det, res, row1, times)),
                  FadeOut(VGroup(vec1, lv1, vec2, lv2, crv_y, crv_x, marks)),
                  tan_y.animate.set_stroke(opacity=0.25), tan_x.animate.set_stroke(opacity=0.25),
                  run_time=1.0)
        h1 = put(T(r"\textbf{5.} The coefficients of this equation are the"), 2.8)
        h2 = put(T(r"components of another normal vector,"), 2.3)
        h3 = M(r"\mathbf{n}", r"=\mathbf{v}_2\times\mathbf{v}_1=" + FXT + r"\,\hat{\imath}+" + FYT
               + r"\,\hat{\jmath}-\hat{k}.", fs=31)
        h3[0].set_color(N_C)
        put(h3, 1.7)
        self.play(FadeIn(h1), FadeIn(h2), run_time=1.0)
        self.play(Write(h3), GrowArrow(n_arr), FadeIn(n_lab), run_time=2.0)
        self.w(2.4)
        h4 = put(T(r"With $w(x,y,z)=f(x,y)-z$, it is a gradient:"), 0.95)
        h5 = M(r"\mathbf{n}", r"=\nabla w\Big|_{P_0}", fs=34)
        h5[0].set_color(N_C)
        put(h5, 0.3)
        self.play(FadeIn(h4), run_time=0.9)
        self.play(Write(h5), run_time=1.3)
        self.w(2.0)
        h6 = put(T(r"\textbf{Normal line} to the surface at $P_0$:"), -0.55)
        nl = VGroup(M(r"x(t)=x_0+" + FXT + r"\,t", fs=30),
                    M(r"y(t)=y_0+" + FYT + r"\,t", fs=30),
                    M(r"z(t)=f(x_0,y_0)-t", fs=30))
        for m, y in zip(nl, (-1.1, -1.6, -2.1)):
            put(m, y, x=LEFT_X + 0.3)
        self.play(FadeIn(h6), Create(n_line), FadeIn(nl_lab), FadeOut(cross), FadeOut(cross_lab),
                  run_time=1.3)
        self.play(LaggedStart(*[Write(m) for m in nl], lag_ratio=0.5), run_time=2.4)
        self.w(6.0)
