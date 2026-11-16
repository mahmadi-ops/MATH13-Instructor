r"""Why the second derivative test works: take the directional derivative twice.

    manim -qh --fps 20 --disable_caching -o second_derivative_test_1080p \
        second_derivative_test.py SecondDerivativeTest

Video of fig-why-sdt-video (Subsection "Why the Second Derivative Test
Works", why-second-derivative-test.ptx), copied to assets/videos.

A. Title.
B. At a critical point the tangent plane is horizontal, so the first
   derivative cannot tell a minimum from a maximum from a saddle; the idea is
   to leave the point in every unit direction u and watch the slice curve.
C. Three cases, z = x^2 + y^2, z = -(x^2 + y^2) and z = x^2 - y^2: the
   vertical plane through the critical point turns with u, the slice curve it
   cuts is drawn on the surface and, flat, in an inset, and D^2_u f is
   plotted against the direction angle.
D. The derivation, with no parameter t and no function g(t): D_u f, then
   D^2_u f = f_xx u1^2 + 2 f_xy u1 u2 + f_yy u2^2, which then stays at the top
   of the screen; the completed square
   f_xx D^2_u f = (f_xx u1 + f_xy u2)^2 + (f_xx f_yy - f_xy^2) u2^2, which stays
   right under it; and the sign of each of its terms, marked under the term,
   as the cases H > 0, H < 0 and H = 0 are argued.  Then the test itself.

Instructor request (Nov 2026): keep D^2_u f and the completed square on
screen throughout the derivation and point out the sign of each term; no t
dependence and no g(t); the 2D slices are called slice curves.
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

FAST = os.environ.get("FAST", "") == "1"     # quick previews: shorter waits
STAGES = os.environ.get("STAGES", "ABCD")

# ---- colours ----------------------------------------------------------------
UP_C = "#7FC97F"        # curves up, D^2_u f > 0
DOWN_C = "#E8625A"      # curves down, D^2_u f < 0
SAD_C = "#F2C94C"       # saddle
U_C = "#F2C94C"         # the unit vector u
EQ_C = "#7FD3E8"        # the main equations
GRAPH_C = "#5BC0DE"     # the curve D^2_u f against theta
DIM = GREY_B

U = r"\mathbf{u}"
D2 = r"D^2_{\mathbf{u}}f"


def T(s, fs=34, color=WHITE):
    """A line of text, set by LaTeX like the mathematics."""
    return Tex(s, font_size=fs, color=color)


def W(t):
    """Wait, shortened in quick previews."""
    return 0.3 if FAST else t


# ---- the three examples -----------------------------------------------------
CASES = [
    dict(name="Local minimum", color=UP_C, f=lambda x, y: x * x + y * y,
         zoff=-0.35, zaxis=(-0.3, 5.2), plane=(-0.6, 4.6),
         d2=lambda th: 2.0 + 0 * th, surf=["#2B8FA8", "#1E6C80"],
         lines=[r"$z = x^2 + y^2$", rf"${D2} > 0$ for every ${U}$",
                r"$f_{xx} = 2 > 0$", r"$H = f_{xx}f_{yy} - f_{xy}^2 = 4 > 0$"],
         say=["every slice curve through the point bends up",
              "positive curvature in every direction $\\;\\to\\;$ local minimum"]),
    dict(name="Local maximum", color=DOWN_C, f=lambda x, y: -(x * x + y * y),
         zoff=1.45, zaxis=(-5.0, 1.0), plane=(-4.6, 0.6),
         d2=lambda th: -2.0 + 0 * th, surf=["#2B8FA8", "#1E6C80"],
         lines=[r"$z = -(x^2 + y^2)$", rf"${D2} < 0$ for every ${U}$",
                r"$f_{xx} = -2 < 0$", r"$H = f_{xx}f_{yy} - f_{xy}^2 = 4 > 0$"],
         say=["every slice curve through the point bends down",
              "negative curvature in every direction $\\;\\to\\;$ local maximum"]),
    dict(name="Saddle point", color=SAD_C, f=lambda x, y: x * x - y * y,
         zoff=0.1, zaxis=(-2.8, 3.0), plane=(-2.6, 2.6),
         d2=lambda th: 2.0 * np.cos(2 * th), surf=["#8C8C8C", "#5A5A5A"],
         lines=[r"$z = x^2 - y^2$", rf"${D2}$ changes sign with ${U}$",
                r"up along $x$, down along $y$", r"$H = (2)(-2) - 0^2 = -4 < 0$"],
         say=["the slice curve bends up in some directions, down in others",
              "both signs occur $\\;\\to\\;$ saddle point"]),
]

XY = 1.25               # scene units per unit of x and y
ZS = 0.55               # scene units per unit of z
R = 1.5                 # the surface is drawn for |x|, |y| <= R
PHI, THETA = 66 * DEGREES, -50 * DEGREES
RIGHT3 = np.array([-np.sin(THETA), np.cos(THETA), 0.0])   # screen right, in 3D
O3 = 2.55 * RIGHT3                                       # where (0, 0, 0) goes
ZOFF = [-0.35]          # vertical shift of the picture, set for each surface


def W3(x, y, z):
    return O3 + np.array([XY * x, XY * y, ZS * z + ZOFF[0]])


class SecondDerivativeTest(ThreeDScene):

    # ------------------------------------------------------------------ A
    def title(self):
        self.set_camera_orientation(phi=0, theta=-PI / 2)
        t1 = T(r"\textbf{Why the Second Derivative Test Works}", 60)
        t2 = T(rf"taking the directional derivative twice: ${D2}$", 38, EQ_C)
        t3 = T(r"MATH 13 \;---\; Extreme Values and Saddle Points", 26, DIM)
        g = VGroup(t1, t2, t3).arrange(DOWN, buff=0.5)
        self.play(FadeIn(t1, shift=0.2 * UP), run_time=1.2)
        self.play(FadeIn(t2), FadeIn(t3), run_time=1.0)
        self.wait(W(2.5))
        self.play(FadeOut(g), run_time=0.8)

    # ------------------------------------------------------------------ 3D parts
    def axes3d(self, zr=(-0.3, 5.2)):
        ax = VGroup()
        for a, b in [((-2.0, 0, 0), (2.3, 0, 0)), ((0, -2.0, 0), (0, 2.3, 0)),
                     ((0, 0, zr[0]), (0, 0, zr[1]))]:
            ax.add(Arrow3D(W3(*a), W3(*b), color=GREY_A, thickness=0.006,
                           height=0.18, base_radius=0.05))
        labs = [MathTex("x", font_size=36).move_to(W3(2.6, 0, 0)),
                MathTex("y", font_size=36).move_to(W3(0, 2.6, 0)),
                MathTex("z", font_size=36).move_to(W3(0, 0, zr[1] + 0.4))]
        return ax, labs

    def surface(self, f, colors, zclip=None):
        def p(u, v):
            return W3(u, v, f(u, v))
        s = Surface(p, u_range=[-R, R], v_range=[-R, R], resolution=(22, 22),
                    checkerboard_colors=colors, fill_opacity=0.92,
                    stroke_width=0.2, stroke_color=BLACK)
        return s

    def slice_plane(self, th, zlo, zhi):
        d = np.array([np.cos(th), np.sin(th), 0.0])
        r = 1.95
        o = W3(0, 0, 0)
        pts = [o + XY * (-r) * d + ZS * zlo * OUT, o + XY * r * d + ZS * zlo * OUT,
               o + XY * r * d + ZS * zhi * OUT, o + XY * (-r) * d + ZS * zhi * OUT]
        return Polygon(*pts, fill_color=GREY_E, fill_opacity=0.45, stroke_width=0)

    def slice_curve3d(self, case, th):
        k = case["d2"](th)
        c = UP_C if k > 1e-6 else (DOWN_C if k < -1e-6 else WHITE)
        d = np.array([np.cos(th), np.sin(th)])
        return ParametricFunction(
            lambda s: W3(s * d[0], s * d[1], 0.5 * k * s * s),
            t_range=[-R, R], color=c, stroke_width=7)

    def u_arrow(self, th):
        d = np.array([np.cos(th), np.sin(th), 0.0])
        o = W3(0, 0, 0)
        return Arrow(o, o + 1.55 * XY * d, buff=0, color=U_C,
                     stroke_width=6, max_tip_length_to_length_ratio=0.18)

    def tangent_plane(self):
        a = 1.15
        return Polygon(W3(-a, -a, 0), W3(a, -a, 0), W3(a, a, 0), W3(-a, a, 0),
                       fill_color="#C9A227", fill_opacity=0.45,
                       stroke_color="#C9A227", stroke_width=1)

    # ------------------------------------------------------------------ B
    def critical_point(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)
        case = CASES[0]
        ZOFF[0] = case["zoff"]
        ax, labs = self.axes3d(case["zaxis"])
        surf = self.surface(case["f"], case["surf"])
        tp = self.tangent_plane()
        dot = Dot3D(W3(0, 0, 0), radius=0.07, color=WHITE)
        zl = T(r"$z = x^2 + y^2$", 34, EQ_C).to_corner(UL, buff=0.6)
        self.add_fixed_orientation_mobjects(*labs)
        self.add_fixed_in_frame_mobjects(zl)
        self.play(Create(ax), FadeIn(surf), FadeIn(zl), *[FadeIn(l) for l in labs],
                  run_time=2.0)
        c1 = T(r"At a critical point $f_x(a,b) = f_y(a,b) = 0$: the tangent plane is horizontal.", 32)
        c2 = T("the first derivative cannot tell a minimum from a maximum from a saddle", 28, DIM)
        cap = VGroup(c1, c2).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.45)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(FadeIn(tp), FadeIn(dot), FadeIn(cap), run_time=1.5)
        self.wait(W(4.0))
        c3 = T(rf"Idea: leave the point in every unit direction ${U} = (u_1, u_2)$ and watch how the slice curve bends.", 32)
        c3.to_edge(DOWN, buff=0.55)
        self.add_fixed_in_frame_mobjects(c3)
        self.play(FadeOut(cap), FadeOut(tp), FadeIn(c3), run_time=1.0)
        self.wait(W(3.0))
        self.play(*[FadeOut(m) for m in [ax, surf, dot, zl, c3, *labs]], run_time=0.8)

    # ------------------------------------------------------------------ C
    def one_case(self, case, sweep, dur):
        th = ValueTracker(0.0)
        k_of = case["d2"]

        # left: the summary box
        title = T(r"\textbf{" + case["name"] + "}", 40, case["color"])
        rows = VGroup(*[T(s, 30) for s in case["lines"]]).arrange(DOWN, buff=0.14)
        rows[1].set_color(case["color"])
        inner = VGroup(title, rows).arrange(DOWN, buff=0.2)
        box = SurroundingRectangle(inner, buff=0.24, corner_radius=0.15,
                                   color=GREY_B, stroke_width=2)
        box.set_fill(GREY_E, opacity=0.35)
        panel = VGroup(box, inner).move_to([-4.6, 2.3, 0])

        # left: D^2_u f against the angle
        gax = Axes(x_range=[0, TAU, PI], y_range=[-2.8, 2.8, 2], x_length=4.2,
                   y_length=1.75, tips=False,
                   axis_config=dict(color=GREY_B, stroke_width=2, include_ticks=True),
                   y_axis_config=dict(numbers_to_include=[-2, 0, 2], font_size=22))
        gax.move_to([-4.6, -0.72, 0])
        gframe = SurroundingRectangle(gax, buff=0.05, color=GREY_B, stroke_width=1.5)
        xl = VGroup(MathTex("0", font_size=22).next_to(gax.c2p(0, -2.8), DOWN, buff=0.12),
                    MathTex(r"\pi", font_size=22).next_to(gax.c2p(PI, -2.8), DOWN, buff=0.12),
                    MathTex(r"2\pi", font_size=22).next_to(gax.c2p(TAU, -2.8), DOWN, buff=0.12))
        gtitle = T(rf"${D2}$ \;vs.\; direction angle $\theta$", 28).next_to(gframe, UP, buff=0.12)
        curve = gax.plot(k_of, x_range=[0, TAU], color=GRAPH_C, stroke_width=4)
        shades = VGroup()
        xs = np.linspace(0, TAU, 241)
        ys = k_of(xs)
        start = 0
        for i in range(1, len(xs)):
            if i == len(xs) - 1 or np.sign(ys[i]) != np.sign(ys[start]):
                seg = xs[start:i + 1]
                pos = np.mean(ys[start:i + 1]) > 0
                shades.add(gax.get_area(curve, x_range=[seg[0], seg[-1]],
                                        color=UP_C if pos else DOWN_C, opacity=0.35))
                start = i
        gdot = Dot(gax.c2p(0, k_of(0.0)), radius=0.08, color=WHITE)

        def upd_gdot(m):
            t = th.get_value() % TAU
            k = float(k_of(np.array(t)))
            m.move_to(gax.c2p(t, k))
            m.set_color(UP_C if k > 1e-6 else (DOWN_C if k < -1e-6 else WHITE))
        gdot.add_updater(upd_gdot)

        # left: the slice curve, flat
        ibox = Rectangle(width=4.2, height=1.5, color=GREY_B, stroke_width=1.5)
        ibox.move_to([-4.6, -3.18, 0])
        ititle = T(rf"slice curve in the direction ${U}$", 28).next_to(ibox, UP, buff=0.1)
        vtx = ibox.get_center() + 0.12 * UP

        def flat_points():
            k = float(k_of(np.array(th.get_value())))
            s = np.linspace(-1.6, 1.6, 61)
            return [vtx + np.array([x, 0.1 * k * x * x, 0]) for x in s]
        icurve = VMobject(stroke_width=5)
        icurve.set_points_smoothly(flat_points())

        def upd_icurve(m):
            m.set_points_smoothly(flat_points())
            k = float(k_of(np.array(th.get_value())))
            m.set_stroke(UP_C if k > 1e-6 else (DOWN_C if k < -1e-6 else WHITE), width=5)
        icurve.add_updater(upd_icurve)
        idot = Dot(vtx, radius=0.07, color=WHITE)
        lab_up = T(rf"bends up: ${D2} > 0$", 26, UP_C)
        lab_dn = T(rf"bends down: ${D2} < 0$", 26, DOWN_C)
        lab_fl = T(rf"flat: ${D2} = 0$", 26, WHITE)
        for l in (lab_up, lab_dn, lab_fl):
            l.move_to(ibox.get_bottom() + 0.27 * UP)

        def lab_upd(which):
            def f(m):
                k = float(k_of(np.array(th.get_value())))
                on = {"up": k > 1e-6, "dn": k < -1e-6, "fl": abs(k) <= 1e-6}[which]
                m.set_opacity(1.0 if on else 0.0)
            return f
        lab_up.add_updater(lab_upd("up"))
        lab_dn.add_updater(lab_upd("dn"))
        lab_fl.add_updater(lab_upd("fl"))

        left = VGroup(panel, gframe, gax, xl, gtitle, shades, curve, gdot,
                      ibox, ititle, icurve, idot, lab_up, lab_dn, lab_fl)
        self.add_fixed_in_frame_mobjects(*left)
        for m in left:
            self.remove(m)

        # right: the surface, the turning plane and the slice curve
        ZOFF[0] = case["zoff"]
        ax, labs = self.axes3d(case["zaxis"])
        surf = self.surface(case["f"], case["surf"])
        zlo, zhi = case["plane"]
        plane = always_redraw(lambda: self.slice_plane(th.get_value(), zlo, zhi))
        sc3 = always_redraw(lambda: self.slice_curve3d(case, th.get_value()))
        uarr = always_redraw(lambda: self.u_arrow(th.get_value()))
        ulab = MathTex(r"\mathbf{u}", font_size=40, color=U_C)
        ulab.add_updater(lambda m: m.move_to(
            W3(0, 0, 0) + 1.85 * XY * np.array([np.cos(th.get_value()), np.sin(th.get_value()), 0])))
        dot = Dot3D(W3(0, 0, 0), radius=0.07, color=WHITE)
        self.add_fixed_orientation_mobjects(*labs, ulab)

        cap = T(case["say"][0], 32).move_to([2.55, -3.45, 0])
        cap2 = T(case["say"][1], 30, case["color"]).next_to(cap, DOWN, buff=0.12)
        self.add_fixed_in_frame_mobjects(cap, cap2)
        self.remove(cap, cap2)

        self.play(FadeIn(panel), Create(ax), FadeIn(surf), *[FadeIn(l) for l in labs],
                  run_time=1.6)
        self.play(FadeIn(VGroup(gframe, gax, xl, gtitle, shades, curve)),
                  FadeIn(VGroup(ibox, ititle)), run_time=1.0)
        self.add(plane, sc3, uarr, ulab, dot, gdot, icurve, idot, lab_up, lab_dn, lab_fl)
        self.play(FadeIn(cap), run_time=0.6)
        for target, t in sweep:
            if target is None:
                self.wait(W(t))
            else:
                self.play(th.animate.set_value(target), run_time=t if not FAST else 0.6,
                          rate_func=linear)
        self.play(FadeIn(cap2), run_time=0.6)
        self.wait(W(dur))
        for m in (gdot, icurve, lab_up, lab_dn, lab_fl, ulab):
            m.clear_updaters()
        everything = [m for m in self.mobjects]
        self.play(*[FadeOut(m) for m in everything], run_time=0.8)
        self.remove(*everything)

    def three_cases(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)
        self.one_case(CASES[0], [(TAU, 6.0)], 1.5)
        self.one_case(CASES[1], [(TAU, 6.0)], 1.5)
        self.one_case(CASES[2], [(PI / 2, 3.0), (None, 1.0), (PI, 3.0), (None, 1.0),
                                 (TAU, 6.0)], 2.0)

    # ------------------------------------------------------------------ D
    def derivation(self):
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        # D_u f, once
        h1 = T(r"\textbf{Take the directional derivative once \dots}", 44)
        h1.to_edge(UP, buff=0.5)
        e1 = MathTex(r"D_{\mathbf{u}}f", "=", r"f_x\,u_1 + f_y\,u_2", font_size=48, color=EQ_C)
        t1 = T(r"at the critical point $f_x = f_y = 0$, so $D_{\mathbf{u}}f = 0$ for every $\mathbf{u}$:", 32)
        t2 = T("every slice curve is flat there; the first derivative sees no difference", 32, DIM)
        g1 = VGroup(e1, t1, t2).arrange(DOWN, buff=0.45).next_to(h1, DOWN, buff=0.8)
        self.play(FadeIn(h1), run_time=0.8)
        self.play(Write(e1), run_time=1.2)
        self.play(FadeIn(t1), run_time=0.8)
        self.play(FadeIn(t2), run_time=0.8)
        self.wait(W(3.0))

        # D^2_u f, twice
        h2 = T(r"\textbf{\dots\ and twice}", 44).to_edge(UP, buff=0.5)
        e2a = MathTex(D2, "=", r"D_{\mathbf{u}}\bigl(f_x\,u_1 + f_y\,u_2\bigr)", font_size=46, color=EQ_C)
        e2b = MathTex(r"=", r"\bigl(f_{xx}u_1 + f_{xy}u_2\bigr)u_1 + \bigl(f_{yx}u_1 + f_{yy}u_2\bigr)u_2",
                      font_size=42, color=EQ_C)
        e2a.move_to([0, 1.6, 0])
        e2b.next_to(e2a, DOWN, buff=0.35, aligned_edge=LEFT).shift(0.95 * RIGHT)
        note = T(r"apply $D_{\mathbf{u}}$ to each partial derivative; $f_{xy} = f_{yx}$", 30, DIM)
        note.next_to(e2b, DOWN, buff=0.45)
        self.play(FadeOut(VGroup(h1, e1, t1, t2)), FadeIn(h2), run_time=0.8)
        self.play(Write(e2a), run_time=1.2)
        self.play(Write(e2b), FadeIn(note), run_time=1.4)
        self.wait(W(2.5))

        # the second directional derivative: to the top, where it stays
        top = MathTex(D2, "=", r"f_{xx}\,u_1^2", "+", r"2f_{xy}\,u_1u_2", "+", r"f_{yy}\,u_2^2",
                      font_size=50, color=EQ_C)
        top.move_to([0, 0.0, 0])
        self.play(FadeOut(VGroup(e2a, e2b, note)), Write(top), run_time=1.4)
        w1 = T(rf"the curvature of the slice curve in the direction ${U}$", 32)
        w2 = T(r"one sign for every $\mathbf{u}$ $\;\to\;$ local min or max", 32, UP_C)
        w3 = T(r"both signs $\;\to\;$ saddle.\quad But there are infinitely many directions to check.", 32, DOWN_C)
        ws = VGroup(w1, w2, w3).arrange(DOWN, buff=0.3).next_to(top, DOWN, buff=0.6)
        self.play(FadeIn(w1), run_time=0.8)
        self.play(FadeIn(w2), run_time=0.8)
        self.play(FadeIn(w3), run_time=0.8)
        self.wait(W(3.5))
        topbox_target = top.copy().scale(0.86).to_edge(UP, buff=0.35)
        self.play(FadeOut(h2), FadeOut(ws), Transform(top, topbox_target), run_time=1.0)
        frame_top = SurroundingRectangle(top, buff=0.18, color=EQ_C, stroke_width=1.5,
                                         corner_radius=0.1)
        self.play(Create(frame_top), run_time=0.6)

        # complete the square: the second line, which stays
        hcs = T(r"multiply by $f_{xx}$ and complete the square in $u_1$:", 32, DIM)
        hcs.next_to(frame_top, DOWN, buff=0.35)
        sq = MathTex(r"f_{xx}\,", D2, "=", r"\bigl(f_{xx}u_1 + f_{xy}u_2\bigr)^2", "+",
                     r"\bigl(f_{xx}f_{yy} - f_{xy}^2\bigr)", r"\,u_2^2",
                     font_size=48, color=EQ_C)
        sq.next_to(hcs, DOWN, buff=0.3)
        self.play(FadeIn(hcs), run_time=0.6)
        self.play(Write(sq), run_time=1.6)
        self.wait(W(1.5))
        lhs = VGroup(sq[0], sq[1])
        b_lhs = Brace(lhs, DOWN, buff=0.12, color=GREY_B)
        b_sq = Brace(sq[3], DOWN, buff=0.12, color=GREY_B)
        b_h = Brace(sq[5], DOWN, buff=0.12, color=GREY_B)
        b_u = Brace(sq[6], DOWN, buff=0.12, color=GREY_B)
        l_sq = T(r"$\ge 0$ always", 30, UP_C).next_to(b_sq, DOWN, buff=0.08)
        l_h = MathTex("H", font_size=34, color=WHITE).next_to(b_h, DOWN, buff=0.08)
        l_u = T(r"$\ge 0$", 28, UP_C).next_to(b_u, DOWN, buff=0.08)
        l_lhs = T("sign?", 28, WHITE).next_to(b_lhs, DOWN, buff=0.08)
        self.play(GrowFromCenter(b_sq), FadeIn(l_sq), run_time=0.8)
        self.wait(W(1.0))
        self.play(GrowFromCenter(b_u), FadeIn(l_u), run_time=0.8)
        self.wait(W(1.0))
        self.play(GrowFromCenter(b_h), FadeIn(l_h), run_time=0.8)
        say = T(r"only $H$ can be negative, so $H$ decides the sign of the right-hand side", 30)
        say.next_to(VGroup(l_sq, l_h, l_u), DOWN, buff=0.45)
        self.play(GrowFromCenter(b_lhs), FadeIn(l_lhs), FadeIn(say), run_time=0.9)
        self.wait(W(3.0))
        self.play(FadeOut(hcs), run_time=0.4)

        area_top = say.get_top()[1] + 0.1

        def body(*lines):
            g = VGroup(*lines).arrange(DOWN, buff=0.28)
            g.next_to(np.array([0, area_top, 0]), DOWN, buff=0.0)
            return g

        # Case H > 0
        self.play(FadeOut(say), run_time=0.4)
        hcase = T(r"\textbf{Case $H > 0$}", 38, UP_C)
        l_h2 = MathTex("H > 0", font_size=34, color=UP_C).move_to(l_h)
        l_u2 = T(r"$H\,u_2^2 \ge 0$", 28, UP_C).next_to(VGroup(b_h, b_u), DOWN, buff=0.08)
        a1 = T(r"both terms are $\ge 0$, and they are never both $0$:", 30)
        a2 = T(r"if $u_2 \ne 0$ the second term is $> 0$; \; if $u_2 = 0$ then $u_1 = \pm 1$ and the square is $f_{xx}^2 > 0$", 28, DIM)
        a3 = T(rf"so $f_{{xx}}\,{D2} > 0$ for every ${U}$: \; ${D2}$ has the sign of $f_{{xx}}$", 32, EQ_C)
        a4 = T(r"$f_{xx} > 0$: every slice curve bends up $\;\to\;$ local minimum", 30, UP_C)
        a5 = T(r"$f_{xx} < 0$: every slice curve bends down $\;\to\;$ local maximum", 30, DOWN_C)
        g = body(hcase, a1, a2, a3, a4, a5)
        b_hu = Brace(VGroup(sq[5], sq[6]), DOWN, buff=0.12, color=GREY_B)
        self.play(FadeIn(hcase), Transform(l_h, l_h2), run_time=0.8)
        self.wait(W(1.0))
        self.play(ReplacementTransform(VGroup(b_h, b_u), b_hu),
                  ReplacementTransform(VGroup(l_h, l_u), l_u2), run_time=0.9)
        self.play(FadeIn(a1), FadeIn(a2), run_time=0.9)
        self.wait(W(2.0))
        l_lhs2 = T(r"$> 0$", 28, UP_C).move_to(l_lhs)
        self.play(Transform(l_lhs, l_lhs2), FadeIn(a3), run_time=0.9)
        self.wait(W(2.0))
        self.play(FadeIn(a4), run_time=0.7)
        self.play(FadeIn(a5), run_time=0.7)
        self.wait(W(3.5))

        # Case H < 0
        hcase2 = T(r"\textbf{Case $H < 0$} \quad (with $f_{xx} \ne 0$)", 38, SAD_C)
        l_u3 = T(r"$H\,u_2^2 \le 0$", 28, DOWN_C).move_to(l_u2)
        c1 = T("the two terms now compete:", 30)
        c2 = T(rf"${U} = (1,0)$: \; square $= f_{{xx}}^2 > 0$, \; $H u_2^2 = 0$ \; $\Rightarrow$ \; $f_{{xx}}\,{D2} > 0$", 30, UP_C)
        c3 = T(rf"${U}$ along $(-f_{{xy}},\, f_{{xx}})$: \; square $= 0$, \; $H u_2^2 < 0$ \; $\Rightarrow$ \; $f_{{xx}}\,{D2} < 0$", 30, DOWN_C)
        c4 = T(rf"${D2}$ takes both signs $\;\to\;$ saddle point", 32, SAD_C)
        g2 = body(hcase2, c1, c2, c3, c4)
        l_lhs3 = T("both signs", 28, SAD_C).move_to(l_lhs)
        self.play(*[FadeOut(m) for m in (hcase, a1, a2, a3, a4, a5)], run_time=0.6)
        self.play(FadeIn(hcase2), Transform(l_u2, l_u3), run_time=0.8)
        self.play(FadeIn(c1), run_time=0.6)
        self.play(FadeIn(c2), run_time=0.9)
        self.wait(W(2.0))
        self.play(FadeIn(c3), run_time=0.9)
        self.wait(W(2.0))
        self.play(Transform(l_lhs, l_lhs3), FadeIn(c4), run_time=0.8)
        self.wait(W(3.5))

        # Case H = 0
        hcase3 = T(r"\textbf{Case $H = 0$}", 38, GREY_A)
        l_u4 = T(r"$H\,u_2^2 = 0$", 28, GREY_A).move_to(l_u2)
        d1 = T(r"only the square is left, and it is $0$ for $\mathbf{u}$ along $(-f_{xy},\, f_{xx})$:", 30)
        d2 = T(rf"${D2} = 0$ in that direction, a flat slice curve $\;\to\;$ the test is inconclusive", 30, GREY_A)
        g3 = body(hcase3, d1, d2)
        l_lhs4 = T(r"$\ge 0$, can be $0$", 28, GREY_A).move_to(l_lhs)
        self.play(*[FadeOut(m) for m in (hcase2, c1, c2, c3, c4)], run_time=0.6)
        self.play(FadeIn(hcase3), Transform(l_u2, l_u4), Transform(l_lhs, l_lhs4), run_time=0.8)
        self.play(FadeIn(d1), run_time=0.8)
        self.play(FadeIn(d2), run_time=0.8)
        self.wait(W(3.0))

        # the test
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        th = T(r"\textbf{The Second Derivative Test \;(Theorem II)}", 48).to_edge(UP, buff=0.7)
        rows = VGroup(
            T(r"$H > 0,\; f_{xx} > 0 \;\Rightarrow\;$ local minimum", 36, UP_C),
            T(r"$H > 0,\; f_{xx} < 0 \;\Rightarrow\;$ local maximum", 36, DOWN_C),
            T(r"$H < 0 \;\Rightarrow\;$ saddle point", 36, SAD_C),
            T(r"$H = 0 \;\Rightarrow\;$ inconclusive", 36, GREY_A),
            T(r"$H = f_{xx}f_{yy} - f_{xy}^2$ \; evaluated at the critical point", 32),
        ).arrange(DOWN, buff=0.35)
        bx = SurroundingRectangle(rows, buff=0.45, corner_radius=0.15, color=GREY_B,
                                  stroke_width=2).set_fill(GREY_E, opacity=0.35)
        VGroup(bx, rows).next_to(th, DOWN, buff=0.6)
        self.play(FadeIn(th), run_time=0.8)
        self.play(FadeIn(bx), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.35),
                  run_time=2.4)
        self.wait(W(5.0))

    def construct(self):
        if "A" in STAGES:
            self.title()
        if "B" in STAGES:
            self.critical_point()
        if "C" in STAGES:
            self.three_cases()
        if "D" in STAGES:
            self.derivation()
