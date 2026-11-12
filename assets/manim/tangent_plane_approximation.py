r"""Approximating a function with its tangent plane: the differential df against
the actual change Delta f, and the linearization L(x,y).

    manim -qh --fps 30 --disable_caching -o tangent_plane_approximation \
        tangent_plane_approximation.py TangentPlaneApproximation

It follows "Estimating the Change in a Particular Direction" and "Linearizing
z = f(x,y)" of the course notes.

1. One variable: moving dx away from x changes f by Delta f = f(x+dx) - f(x);
   walking along the tangent line instead gives df = f'(x) dx, which is close
   to Delta f when dx is small.
2. Two variables: moving away from P0 takes a distance ds and a direction u.
   The actual change is Delta f = f(x0 + u1 ds, y0 + u2 ds) - f(x0, y0); the
   rate of change in the direction u is grad f . u, the slope of the tangent
   line of the slice, so Delta f is about df = (grad f . u) ds.  The error
   shrinks much faster than ds.
3. Every direction: as u turns, the tangent line sweeps out the tangent plane.
   With x = x0 + u1 ds and y = y0 + u2 ds the differential is
   f_x (x - x0) + f_y (y - y0), so f(x,y) is about
   L(x,y) = f_x (x - x0) + f_y (y - y0) + f(x0, y0), whose graph is the tangent
   plane: it hugs the surface near P0 and the surface curves away farther out.

The colours are those of the interactive figure of the notes: blue for the
tangent line and df, orange for Delta f, pink for the error.  The 3D picture
is drawn with a fixed orthographic projection inside an ordinary Scene (faces
depth-sorted once), as in dirderiv_partials.py.  The surface is only a prop for
"a general f"; its formula is never shown.
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
X0, Y0 = 2.0, 1.4                       # P0 (shown only as (x0, y0))
R_S = 3.2                               # the surface is drawn over the quarter disc r <= R_S
C_F = 0.27
ANG0 = 200 * DEGREES                    # the direction of u shown first
S_BIG, S_SMALL = 1.0, 0.12
R_P = 0.9                               # radius (in the xy-plane) of the tangent plane patch


def f(x, y):
    """The surface: a prop for a general f, never named on screen."""
    return 3.0 - C_F * (x * x + y * y)


Z0 = f(X0, Y0)
FX, FY = -2 * C_F * X0, -2 * C_F * Y0

# ---- colours ----------------------------------------------------------------
DF_C = "#58C4DD"                        # the tangent line and df
DEL_C = "#FF9F43"                       # Delta f
ERR_C = "#E07AC8"                       # the error
U_C = "#7CE07C"                         # the unit vector u
INK = "#ECE6E2"
PL_C = "#E88A8A"                        # the tangent plane, as in tangent_plane_cross_product.py
KEY_C = YELLOW

FAST = os.environ.get("FAST", "") == "1"


class View3:
    """Orthographic camera at azimuth az, elevation el: centre + scale * proj."""

    def __init__(self, center, scale, az=75 * DEGREES, el=20 * DEGREES):
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

    def po(self, x, y, z, dx, dy):
        """The point (x, y, z), then moved by (dx, dy) on screen (screen units)."""
        return self.p(x, y, z) + np.array([dx, dy, 0.0])

    def depth(self, x, y, z=0.0):
        ca, sa = np.cos(self.az), np.sin(self.az)
        return (x * ca + y * sa) * np.cos(self.el) + z * np.sin(self.el)

    def cam(self):
        return np.array([np.cos(self.az) * np.cos(self.el),
                         np.sin(self.az) * np.cos(self.el), np.sin(self.el)])


def arrow(a, b, color, sw=6, tip=0.26):
    return Arrow(a, b, buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=0.6, max_stroke_width_to_length_ratio=30)


def ramp(x, lo, width):
    return float(np.clip((x - lo) / width, 0.0, 1.0))


def T(*parts, fs=29, **kw):
    return Tex(*parts, font_size=fs, **kw)


def M(*parts, fs=31, **kw):
    return MathTex(*parts, font_size=fs, **kw)


def follow(m, pos, op=None):
    """m stays at pos() with opacity op(), both re-read every frame."""
    base = m.copy()

    def upd(mm):
        mm.become(base.copy().move_to(pos()).set_opacity(1.0 if op is None else op()))

    upd(m)
    m.add_updater(upd)
    return m


def seg(a, b, color, sw, op=1.0, dashed=False):
    """A segment that may have zero length or zero opacity."""
    if np.linalg.norm(np.array(b) - np.array(a)) < 1e-4:
        b = np.array(a) + np.array([1e-4, 0, 0])
    if dashed:
        return DashedLine(a, b, color=color, stroke_width=sw, stroke_opacity=op, dash_length=0.07)
    return Line(a, b, color=color, stroke_width=sw, stroke_opacity=op)


FXT = r"f_x(x_0,y_0)"
FYT = r"f_y(x_0,y_0)"
GRAD = r"\nabla f\Big|_{P_0}"


class TangentPlaneApproximation(Scene):
    def w(self, t):
        self.wait(0.15 if FAST else t)

    def construct(self):
        LEFT_X = -6.8
        ease = rate_functions.ease_in_out_sine

        def put(m, y, x=LEFT_X, limit=None):
            if limit is None:
                limit = 2.1 if y > 1.4 else 1.0
            m.move_to(np.array([x, y, 0]), aligned_edge=LEFT)
            if m.get_right()[0] > limit:
                print("TOO WIDE:", round(m.width, 2), "right edge", round(m.get_right()[0], 2),
                      "limit", limit)
            return m

        title = put(T(r"\textbf{Approximating with the Tangent Plane}", fs=32, color=BLUE_B),
                    3.62, limit=6.9)
        self.play(FadeIn(title), run_time=0.9)

        # =====================================================================
        # 1. one variable
        # =====================================================================
        ox, oy, sc = 2.0, -2.25, 1.08

        def q(x, y):
            return np.array([ox + sc * x, oy + sc * y, 0.0])

        def h(x):
            return 0.5 + 0.2 * x * x

        XA = 1.45
        HP = 0.4 * XA
        dx = ValueTracker(1.75)
        DX = dx.get_value
        a_x = arrow(q(-0.2, 0), q(4.45, 0), GREY_A, sw=3, tip=0.2)
        a_y = arrow(q(0, -0.2), q(0, 4.6), GREY_A, sw=3, tip=0.2)
        curve = VMobject(color=INK, stroke_width=4.5)
        curve.set_points_as_corners([q(x, h(x)) for x in np.linspace(0, 3.75, 60)])
        c_lab = M("y=f(x)", fs=30).move_to(q(3.75, h(3.75)) + np.array([0.15, 0.3, 0]))
        g0 = DashedLine(q(XA, 0), q(XA, h(XA)), color=GREY_B, stroke_width=2, dash_length=0.08)
        t0 = M("x", fs=28).move_to(q(XA, -0.28))
        d0 = Dot(q(XA, h(XA)), radius=0.07, color=WHITE).set_z_index(5)
        fig1 = VGroup(a_x, a_y, curve, c_lab, g0, t0, d0)

        tan1 = always_redraw(lambda: Line(q(XA - 1.25, h(XA) - 1.25 * HP),
                                          q(XA + DX() + 0.35, h(XA) + HP * (DX() + 0.35)),
                                          color=DF_C, stroke_width=4))
        g1 = always_redraw(lambda: DashedLine(q(XA + DX(), 0), q(XA + DX(), h(XA)), color=GREY_B,
                                              stroke_width=2, dash_length=0.08))
        run1 = always_redraw(lambda: Line(q(XA, h(XA)), q(XA + DX(), h(XA)), color=GREY_A,
                                          stroke_width=2.5))
        leg1 = always_redraw(lambda: Line(q(XA + DX(), h(XA)), q(XA + DX(), h(XA) + HP * DX()),
                                          color=DF_C, stroke_width=7))
        err1 = always_redraw(lambda: Line(q(XA + DX(), h(XA) + HP * DX()),
                                          q(XA + DX(), h(XA + DX())), color=ERR_C, stroke_width=7))

        def bracket1():
            xb = XA + DX() + 0.2
            m = VMobject(color=DEL_C, stroke_width=4)
            m.set_points_as_corners([q(XA + DX() + 0.06, h(XA)), q(xb, h(XA)),
                                     q(xb, h(XA + DX())), q(XA + DX() + 0.06, h(XA + DX()))])
            return m

        brk1 = always_redraw(bracket1)
        db1 = always_redraw(lambda: Dot(q(XA + DX(), h(XA) + HP * DX()), radius=0.06, color=DF_C))
        do1 = always_redraw(lambda: Dot(q(XA + DX(), h(XA + DX())), radius=0.065, color=DEL_C))
        big1 = lambda: ramp(DX(), 0.55, 0.35)
        t1 = follow(M("x+dx", fs=28), lambda: q(XA + DX() + 0.12, -0.28), lambda: ramp(DX(), 0.5, 0.3))
        ldx = follow(M("dx", fs=28), lambda: q(XA + DX() / 2, h(XA) - 0.24), big1)
        ldf = follow(M("df", fs=32, color=DF_C),
                     lambda: q(XA + DX() - 0.3, h(XA) + 0.3 * HP * DX()), big1)
        lDf = follow(M(r"\Delta f", fs=32, color=DEL_C),
                     lambda: q(XA + DX() + 0.58, (h(XA) + h(XA + DX())) / 2 - 0.15), big1)
        lerr = follow(T("error", fs=27, color=ERR_C),
                      lambda: q(XA + DX() - 0.62, h(XA + DX()) + 0.12), big1)
        dyn1 = [g1, run1, tan1, leg1, err1, brk1, db1, do1, t1, ldx, ldf, lDf, lerr]

        p1 = put(T(r"\textbf{One variable.} Move a small distance $dx$ away from $x$."), 2.8, limit=1.6)
        p2 = put(T(r"Along the curve, the \textbf{actual change} is"), 1.9, limit=1.6)
        p3 = put(M(r"\Delta f", r"=f(x+dx)-f(x).", fs=33), 1.3, limit=1.6)
        p3[0].set_color(DEL_C)
        p4 = put(T(r"Along the tangent line, the change is the \textbf{differential}"), 0.4, limit=1.6)
        p5 = put(M("df", r"=f'(x)\,dx.", fs=33), -0.2, limit=1.6)
        p5[0].set_color(DF_C)
        p6 = put(T(r"When $dx$ is small, $df$ closely approximates $\Delta f$."), -1.15, limit=1.6)

        self.play(Create(a_x), Create(a_y), FadeIn(p1), run_time=1.0)
        self.play(Create(curve), FadeIn(c_lab), FadeIn(VGroup(g0, t0, d0)), run_time=1.4)
        self.add(g1, run1, t1, ldx, brk1, do1, lDf)
        self.play(*[FadeIn(m) for m in (g1, run1, t1, ldx)], run_time=0.8)
        self.play(FadeIn(p2), Write(p3), FadeIn(brk1), FadeIn(do1), FadeIn(lDf), run_time=1.8)
        self.w(1.6)
        self.add(tan1, leg1, db1, ldf)
        self.play(FadeIn(p4), Create(tan1), run_time=1.3)
        self.play(Write(p5), FadeIn(leg1), FadeIn(db1), FadeIn(ldf), run_time=1.5)
        self.w(1.2)
        self.add(err1, lerr)
        self.play(FadeIn(err1), FadeIn(lerr), run_time=0.8)
        self.w(1.2)
        self.play(FadeIn(p6), dx.animate.set_value(0.2), run_time=4.5, rate_func=ease)
        self.w(2.0)
        for m in dyn1:
            m.clear_updaters()
        self.play(FadeOut(VGroup(fig1, *dyn1, p1, p2, p3, p4, p5, p6)), run_time=0.9)

        # =====================================================================
        # 2. two variables: a distance ds in a direction u
        # =====================================================================
        v = View3((5.3, -0.92), 1.3)
        ang = ValueTracker(ANG0)
        s = ValueTracker(0.02)
        A, S = ang.get_value, s.get_value
        sl = ValueTracker(0.0)          # the slice: curtain and curve
        tl = ValueTracker(0.0)          # the tangent line
        brk = ValueTracker(0.0)         # Delta f
        tri = ValueTracker(0.0)         # run, df and the error
        gap = ValueTracker(0.0)         # plane point against surface point (the last part)
        back = ValueTracker(0.5)        # how far the tangent line reaches behind P0
        pv = ValueTracker(1.0)          # P, its label and the displacement (hidden in the sweep)

        ax_x = arrow(v.p(0, 0, 0), v.p(3.45, 0, 0), GREY_A, sw=2.5, tip=0.18)
        ax_y = arrow(v.p(0, 0, 0), v.p(0, 3.45, 0), GREY_A, sw=2.5, tip=0.18)
        ax_z = arrow(v.p(0, 0, 0), v.p(0, 0, 3.3), GREY_A, sw=2.5, tip=0.18)
        lx = M("x", fs=32).move_to(v.po(3.45, 0, 0, -0.2, -0.12))
        ly = M("y", fs=32).move_to(v.po(0, 3.45, 0, 0.2, -0.14))
        lz = M("z", fs=32).move_to(v.po(0, 0, 3.3, 0.22, 0.02))
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
        surf_label = M("z = f(x,y)", fs=30, color=BLUE_B).move_to(v.po(2.4, 0, f(2.4, 0), -0.95, 0.3))

        g_dot = Dot(v.p(X0, Y0, 0), radius=0.06, color=WHITE).set_z_index(5)
        g_lab = M("P_0", fs=30).move_to(v.po(X0, Y0, 0, -0.3, -0.16))
        riser0 = DashedLine(v.p(X0, Y0, 0), v.p(X0, Y0, Z0), color=WHITE, stroke_width=2,
                            dash_length=0.07)
        a_dot = Dot(v.p(X0, Y0, Z0), radius=0.07, color=WHITE).set_z_index(6)

        def ray(t):
            return X0 + t * np.cos(A()), Y0 + t * np.sin(A())

        def D():
            return FX * np.cos(A()) + FY * np.sin(A())

        def gz(t):
            return f(*ray(t))

        def S3(t, z):
            return v.p(*ray(t), z)

        def t_range():
            c, sn = np.cos(A()), np.sin(A())
            b = X0 * c + Y0 * sn
            rt = np.sqrt(b * b + R_S ** 2 - X0 ** 2 - Y0 ** 2)
            lo, hi = -b - rt, -b + rt
            for p0, d in ((X0, c), (Y0, sn)):
                if d > 1e-9:
                    lo = max(lo, -p0 / d)
                elif d < -1e-9:
                    hi = min(hi, -p0 / d)
            return lo, hi

        def trace_pts():
            lo, hi = t_range()
            return [S3(t, gz(t)) for t in np.linspace(lo, hi, 44)]

        def mk_curtain():
            lo, hi = t_range()
            m = VMobject(fill_color=WHITE, fill_opacity=0.09 * sl.get_value(), stroke_color=WHITE,
                         stroke_width=1.2, stroke_opacity=0.35 * sl.get_value())
            m.set_points_as_corners([S3(lo, 0)] + trace_pts() + [S3(hi, 0), S3(lo, 0)])
            return m

        def mk_trace():
            m = VMobject(color=INK, stroke_width=4.5, stroke_opacity=sl.get_value())
            m.set_points_as_corners(trace_pts())
            return m

        def mk_tangent():
            tb = back.get_value()
            tf = max(S() + 0.3, tb) if tb < R_P - 1e-6 else R_P
            return Line(S3(-tb, Z0 - D() * tb), S3(tf, Z0 + D() * tf), color=DF_C, stroke_width=4.5,
                        stroke_opacity=tl.get_value())

        def mk_bracket():
            o = 0.16
            m = VMobject(color=DEL_C, stroke_width=4, stroke_opacity=brk.get_value())
            m.set_points_as_corners([S3(S() + 0.05, Z0), S3(S() + o, Z0), S3(S() + o, gz(S())),
                                     S3(S() + 0.05, gz(S()))])
            return m

        curtain = always_redraw(mk_curtain)
        trace = always_redraw(mk_trace)
        tangent = always_redraw(mk_tangent)
        u_arr = always_redraw(lambda: arrow(v.p(X0, Y0, 0), v.p(*ray(1.0), 0), U_C, sw=5, tip=0.2))
        disp = always_redraw(lambda: seg(v.p(X0, Y0, 0), v.p(*ray(S()), 0), WHITE, 7,
                                         0.9 * pv.get_value()))
        p_dot = always_redraw(lambda: Dot(v.p(*ray(S()), 0), radius=0.06, color=WHITE).set_opacity(
            pv.get_value()))
        riser = always_redraw(lambda: seg(v.p(*ray(S()), 0), S3(S(), gz(S())), WHITE, 2,
                                          max(sl.get_value(), gap.get_value()) * pv.get_value(),
                                          dashed=True))
        run = always_redraw(lambda: seg(S3(0, Z0), S3(S(), Z0), GREY_A, 2.5, tri.get_value()))
        leg = always_redraw(lambda: seg(S3(S(), Z0), S3(S(), Z0 + D() * S()), DF_C, 7,
                                        tri.get_value()))
        err = always_redraw(lambda: seg(S3(S() + 0.055, Z0 + D() * S()), S3(S() + 0.055, gz(S())),
                                        ERR_C, 7, tri.get_value()))
        gapl = always_redraw(lambda: seg(S3(S(), Z0 + D() * S()), S3(S(), gz(S())), ERR_C, 7,
                                         gap.get_value()))
        bracket = always_redraw(mk_bracket)
        b_dot = always_redraw(lambda: Dot(S3(S(), Z0 + D() * S()), radius=0.06, color=DF_C).set_opacity(
            max(tri.get_value(), gap.get_value())))
        o_dot = always_redraw(lambda: Dot(S3(S(), gz(S())), radius=0.065, color=DEL_C).set_opacity(
            max(brk.get_value(), gap.get_value())))

        big = lambda: ramp(S(), 0.5, 0.3)
        l_u = follow(M(r"\hat{u}", fs=32, color=U_C), lambda: v.po(*ray(0.62), 0, 0.05, -0.27))
        mode = ValueTracker(0.0)        # 0: P is labelled P;  1: it is labelled (x, y)
        l_p = follow(M("P", fs=30), lambda: v.po(*ray(S()), 0, 0.26, -0.12),
                     lambda: ramp(S(), 0.25, 0.25) * (1 - mode.get_value()) * pv.get_value())
        l_xy = follow(M("(x,y)", fs=27), lambda: v.po(*ray(S()), 0, 0.42, -0.14),
                      lambda: ramp(S(), 0.25, 0.25) * mode.get_value())
        l_ds = follow(M("ds", fs=28), lambda: v.po(*ray(S() / 2), 0, -0.1, 0.24),
                      lambda: big() * (1 - mode.get_value()))
        l_df = follow(M("df", fs=32, color=DF_C),
                      lambda: S3(S(), Z0 + 0.3 * D() * S()) + np.array([-0.3, 0.0, 0.0]),
                      lambda: big() * tri.get_value())
        l_Df = follow(M(r"\Delta f", fs=32, color=DEL_C),
                      lambda: S3(S() + 0.16, (Z0 + gz(S())) / 2) + np.array([0.4, 0.0, 0.0]),
                      lambda: big() * brk.get_value())
        l_err = follow(T("error", fs=27, color=ERR_C),
                       lambda: S3(S() + 0.055, (Z0 + D() * S() + gz(S())) / 2) + np.array([0.52, 0.0, 0.0]),
                       lambda: big() * tri.get_value())
        far = lambda: ramp(S(), 0.45, 0.25) * gap.get_value()
        l_L = follow(M("L(x,y)", fs=28, color=DF_C),
                     lambda: S3(S(), Z0 + D() * S()) + np.array([0.0, 0.3, 0.0]), far)
        l_f = follow(M("f(x,y)", fs=28, color=DEL_C),
                     lambda: S3(S(), gz(S())) + np.array([0.62, -0.1, 0.0]), far)
        dyn = [curtain, trace, riser, disp, u_arr, p_dot, tangent, run, leg, err, gapl, bracket,
               b_dot, o_dot, l_u, l_p, l_xy, l_ds, l_df, l_Df, l_err, l_L, l_f]
        for m in (tangent, run, leg, err, gapl, bracket):
            m.set_z_index(4)
        for m in (b_dot, o_dot, l_df, l_Df, l_err, l_L, l_f):
            m.set_z_index(6)

        c1 = put(T(r"\textbf{Two variables.} Moving away from $P_0(x_0,y_0)$ takes"), 2.8)
        c2 = put(T(r"a distance $ds$ and a direction $\hat{u}=u_1\hat{\imath}+u_2\hat{\jmath}$."), 2.3)
        c3 = put(T(r"We arrive at $P(x_0+u_1\,ds,\;y_0+u_2\,ds)$."), 1.75)
        c4 = put(T(r"The \textbf{actual change} is"), 0.95)
        c5 = M(r"\Delta f", r"=f(x_0+u_1\,ds,\;y_0+u_2\,ds)-f(x_0,y_0).", fs=29)
        c5[0].set_color(DEL_C)
        put(c5, 0.4)
        c6 = put(T(r"The \textbf{rate of change} in the direction $\hat{u}$ is"), -0.45)
        c7 = M(r"\left(\frac{df}{ds}\right)_{\hat{u},P_0}=" + GRAD + r"\cdot\hat{u},", fs=31)
        put(c7, -1.15)
        c8 = put(T(r"the slope of the tangent line (blue)."), -1.85)

        self.play(LaggedStart(*[Create(m) for m in axes[:3]], lag_ratio=0.2), FadeIn(axes[3:]),
                  FadeIn(c1), run_time=1.0)
        self.play(FadeIn(surface, lag_ratio=0.004), FadeIn(surf_label), run_time=1.3)
        self.play(FadeIn(g_dot), FadeIn(g_lab), run_time=0.6)
        self.add(*[m for m in dyn if m is not u_arr and m is not l_u])
        self.play(FadeIn(c2), FadeIn(u_arr), FadeIn(l_u), run_time=1.0)
        self.play(FadeIn(c3), s.animate.set_value(S_BIG), run_time=2.2, rate_func=ease)
        self.w(1.2)
        # the slice, and the actual change
        self.play(sl.animate.set_value(1.0), Create(riser0), FadeIn(a_dot), run_time=1.4)
        self.play(FadeIn(c4), Write(c5), brk.animate.set_value(1.0), run_time=2.0)
        self.w(2.0)
        # the tangent line of the slice, and the differential
        self.play(FadeIn(c6), tl.animate.set_value(1.0), run_time=1.3)
        self.play(Write(c7), FadeIn(c8), run_time=1.8)
        self.w(1.5)
        key = T(r"\textbf{Estimate:}", color=KEY_C)
        est = M(r"\Delta f", r"\approx", "df", r"=\left(\nabla f\big|_{P_0}\cdot\hat{u}\right)ds", fs=32)
        est[0].set_color(DEL_C)
        est[2].set_color(DF_C)
        put(key, -2.72, limit=6.9)
        est.next_to(key, RIGHT, buff=0.3)
        box = SurroundingRectangle(VGroup(key, est), color=KEY_C, buff=0.12, stroke_width=2.5,
                                   corner_radius=0.08)
        self.play(FadeIn(key), Write(est), tri.animate.set_value(1.0), run_time=2.0)
        self.play(Create(box), run_time=0.7)
        self.w(2.2)
        note = put(T(r"As $ds$ shrinks, the error shrinks much faster than $ds$ itself.", fs=28,
                     color=GREY_A), -3.6, limit=6.9)
        self.play(FadeIn(note), s.animate.set_value(S_SMALL), run_time=5.0, rate_func=ease)
        self.w(1.6)
        self.play(s.animate.set_value(S_BIG), run_time=1.8, rate_func=ease)
        self.w(0.8)

        # =====================================================================
        # 3. every direction: the tangent plane and the linearization
        # =====================================================================
        d1 = put(T(r"\textbf{Every direction.} As $\hat{u}$ turns, the tangent line"), 2.8)
        d2 = put(T(r"sweeps out one plane: the tangent plane at $P_0$."), 2.3)
        self.play(FadeOut(VGroup(c1, c2, c3, c4, c5, c6, c7, c8, note)), run_time=0.7)
        self.play(FadeIn(d1), FadeIn(d2), tri.animate.set_value(0.0), brk.animate.set_value(0.0),
                  back.animate.set_value(R_P), s.animate.set_value(0.5), pv.animate.set_value(0.0),
                  run_time=1.3)

        def rim_pt(t):
            return v.p(X0 + R_P * np.cos(t), Y0 + R_P * np.sin(t),
                       Z0 + R_P * (FX * np.cos(t) + FY * np.sin(t)))

        def mk_sweep():
            span = min(A() - ANG0, PI)
            m = VGroup()
            if span < 1e-3:
                return m
            n = max(3, int(40 * span / PI))
            for a0 in (ANG0, ANG0 + PI):
                pts = [v.p(X0, Y0, Z0)] + [rim_pt(t) for t in np.linspace(a0, a0 + span, n)]
                m.add(Polygon(*pts, stroke_width=0, fill_color=PL_C, fill_opacity=0.26))
            return m

        sweep = always_redraw(mk_sweep).set_z_index(3)
        self.add(sweep)
        self.play(ang.animate.set_value(ANG0 + TAU), run_time=9.0, rate_func=ease)
        patch = Polygon(*[rim_pt(t) for t in np.linspace(0, TAU, 60, endpoint=False)],
                        stroke_color=PL_C, stroke_width=2, stroke_opacity=0.9,
                        fill_color=PL_C, fill_opacity=0.26).set_z_index(3)
        top = max((rim_pt(t) for t in np.linspace(0, TAU, 90)), key=lambda p: p[1])
        patch_lab = T("tangent plane", fs=26, color=PL_C).move_to(
            top + np.array([-1.25, 0.3, 0])).set_z_index(6)
        sweep.clear_updaters()
        self.remove(sweep)
        self.add(patch)
        ang.set_value(ANG0)
        self.play(FadeIn(patch_lab), run_time=0.6)
        self.w(1.0)

        d3 = M("df", "=" + FXT + r"\,u_1\,ds+" + FYT + r"\,u_2\,ds.", fs=30)
        d3[0].set_color(DF_C)
        put(d3, 1.55)
        d4 = put(T(r"At $P(x,y)$: $\;u_1\,ds=x-x_0\;$ and $\;u_2\,ds=y-y_0$, so"), 0.95)
        d5 = M("df", "=" + FXT + "(x-x_0)+" + FYT + "(y-y_0).", fs=30)
        d5[0].set_color(DF_C)
        put(d5, 0.35)
        d6 = put(T(r"Then $f(x,y)=f(x_0,y_0)+\Delta f\approx f(x_0,y_0)+df$:"), -0.5)
        self.play(back.animate.set_value(0.5), s.animate.set_value(S_BIG), pv.animate.set_value(1.0),
                  run_time=1.2, rate_func=ease)
        self.play(Write(d3), tri.animate.set_value(1.0), brk.animate.set_value(1.0), run_time=2.0)
        self.w(1.2)
        self.play(FadeIn(d4), mode.animate.set_value(1.0), run_time=1.2)
        self.play(Write(d5), run_time=1.8)
        self.w(1.8)
        self.play(FadeIn(d6), run_time=1.0)
        key2 = T(r"\textbf{Linearization:}", color=KEY_C)
        lin = M("f(x,y)", r"\approx", "L(x,y)", "=" + FXT + "(x-x_0)+" + FYT + "(y-y_0)+f(x_0,y_0)",
                fs=31)
        lin[0].set_color(DEL_C)
        lin[2].set_color(DF_C)
        put(key2, -3.6, limit=6.9)
        lin.next_to(key2, RIGHT, buff=0.3)
        box2 = SurroundingRectangle(VGroup(key2, lin), color=KEY_C, buff=0.13, stroke_width=2.5,
                                    corner_radius=0.08)
        if box2.get_right()[0] > 7.0:
            print("TOO WIDE: linearization row", round(box2.get_right()[0], 2))
        self.play(FadeIn(key2), Write(lin), run_time=2.4)
        self.play(Create(box2), run_time=0.7)
        self.w(2.5)

        # the plane against the surface, near P0 and farther out
        e1 = put(T(r"The graph of $z=L(x,y)$ is the tangent plane."), 1.55)
        e2 = put(T(r"Near $P_0$ the plane hugs the surface, so $L(x,y)$"), 0.75)
        e3 = put(T(r"approximates $f(x,y)$; farther out, the surface"), 0.25)
        e4 = put(T(r"curves away from the plane."), -0.25)
        u_arr.clear_updaters()
        l_u.clear_updaters()
        self.play(FadeOut(VGroup(d3, d4, d5, d6)), tri.animate.set_value(0.0),
                  brk.animate.set_value(0.0), tl.animate.set_value(0.0), sl.animate.set_value(0.0),
                  FadeOut(u_arr), FadeOut(l_u), run_time=1.0)
        self.play(FadeIn(e1), gap.animate.set_value(1.0), run_time=1.2)
        self.w(1.0)
        self.play(FadeIn(e2), s.animate.set_value(0.15), run_time=3.0, rate_func=ease)
        self.w(0.6)
        self.play(FadeIn(e3), FadeIn(e4), s.animate.set_value(1.1),
                  ang.animate.set_value(ANG0 + 60 * DEGREES), run_time=3.5, rate_func=ease)
        self.play(ang.animate.set_value(ANG0 - 60 * DEGREES), run_time=4.0, rate_func=ease)
        self.play(s.animate.set_value(0.15), run_time=2.5, rate_func=ease)
        self.play(s.animate.set_value(0.9), ang.animate.set_value(ANG0), run_time=3.0,
                  rate_func=ease)
        self.w(6.0)
