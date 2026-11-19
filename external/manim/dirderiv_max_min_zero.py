r"""Opening of the lecture on directional derivatives: in which directions is
the rate of change maximum, minimum, and zero?

    manim -qh --fps 30 --disable_caching -o dirderiv_max_min_zero \
        dirderiv_max_min_zero.py MaxMinZero

The video follows the instructor's board plan:

A. Reminders, (I): (D_u f)_{P0} is the rate of change of z = f(x,y)
   in the direction u at P0.  Shown on z = 4 - x^2 - y^2 at P0(1,1): the unit
   vector u turns in the xy-plane, and the tangent line of the vertical slice
   through P0 along u turns with it; its slope is the directional derivative.
   Then reminder (II): (D_u f)_{P0} = (grad f)_{P0} . u.
B. The question, a 5 second wait, the hint, another 5 second wait.
C. The answer (D_u f)_{P0} = ||(grad f)_{P0}|| cos(theta); the board's 2D
   figure animated in the centre; then that figure for the example next to the
   3D graph, the two driven by the same angle and paused together at
   theta = 0, pi/2, pi, and pi/2 on the other side.
D. The summary at the bottom of the board, ending with the open question about
   the two directions of zero change (not answered).

The 3D picture is drawn with a fixed orthographic projection inside an
ordinary Scene (faces depth-sorted once), so it can sit beside flat figures
and be moved and scaled like one.  Notation follows the course notes:
\left(D_{\hat{u}} f\right)_{P_0}, \left(\nabla f\right)_{P_0}, \hat{u} =
u_1\hat{\imath} + u_2\hat{\jmath}.
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

# ---- the example ------------------------------------------------------------
X0, Y0 = 1.0, 1.0                       # P0


def f(x, y):
    return 4.0 - x * x - y * y


Z0 = f(X0, Y0)                          # 2
GX, GY = -2.0 * X0, -2.0 * Y0           # (grad f)_{P0} = <-2, -2>
G_ANG = np.arctan2(GY, GX)              # its direction, 225 degrees
R_DOME = 2.0                            # the surface is drawn for z >= 0


def dd(ang):
    """The directional derivative at P0 in the direction at angle ang."""
    return GX * np.cos(ang) + GY * np.sin(ang)


# ---- colours ----------------------------------------------------------------
U_C = "#5CC8FF"                         # the unit vector u (blue, as in the notes)
G_C = "#FF7B3D"                         # the gradient (vermillion, as in the notes)
TAN_C = "#7CE07C"                       # tangent line
TRACE_C = GOLD
TH_C = YELLOW
KEY_C = YELLOW                          # (I), (II), Question, Hint, Answer

STAGES = os.environ.get("STAGES", "ABCD")
FAST = os.environ.get("FAST", "") == "1"


def v3(c):
    c = np.array(c, dtype=float)
    return c if c.size == 3 else np.array([c[0], c[1], 0.0])


# ---- views: where a figure sits on screen -----------------------------------
class View:
    """Maps figure coordinates to the screen: centre + scale * proj(x, y, z).

    The centre and scale are interpolated by a tracker, so a figure can be
    moved and resized while its updater-driven parts keep following.
    """

    def __init__(self, center, scale):
        self.c0 = self.c1 = v3(center)
        self.s0 = self.s1 = float(scale)
        self.t = ValueTracker(1.0)

    @property
    def center(self):
        a = self.t.get_value()
        return (1 - a) * self.c0 + a * self.c1

    @property
    def scale(self):
        a = self.t.get_value()
        return (1 - a) * self.s0 + a * self.s1

    def retarget(self, center, scale):
        self.c0, self.s0 = self.center.copy(), self.scale
        self.c1, self.s1 = v3(center), float(scale)
        self.t.set_value(0.0)

    def proj(self, x, y, z=0.0):
        return np.array([x, y, 0.0])

    def p(self, x, y, z=0.0):
        return self.center + self.scale * self.proj(x, y, z)

    def po(self, x, y, z, dx, dy):
        """The point (x, y, z), then moved by (dx, dy) on screen (figure units)."""
        return self.p(x, y, z) + self.scale * np.array([dx, dy, 0.0])


class View3(View):
    """Orthographic camera at azimuth az, elevation el; z drawn at zk."""

    def __init__(self, center, scale, az=-15 * DEGREES, el=22 * DEGREES, zk=0.72):
        super().__init__(center, scale)
        self.az, self.el, self.zk = az, el, zk

    def proj(self, x, y, z=0.0):
        ca, sa = np.cos(self.az), np.sin(self.az)
        sx = -x * sa + y * ca
        sy = self.zk * z * np.cos(self.el) - (x * ca + y * sa) * np.sin(self.el)
        return np.array([sx, sy, 0.0])

    def depth(self, x, y, z=0.0):
        ca, sa = np.cos(self.az), np.sin(self.az)
        return (x * ca + y * sa) * np.cos(self.el) + self.zk * z * np.sin(self.el)


def move_fig(view, static, center, scale):
    """Animations that move a figure (its static group and its view) together."""
    c0, s0 = view.center.copy(), view.scale
    view.retarget(center, scale)
    return [
        static.animate.scale(scale / s0, about_point=c0).shift(v3(center) - c0),
        view.t.animate.set_value(1.0),
    ]


def follow(tex, view, pos_fn, opacity_fn=None):
    """Keep a label at pos_fn(), sized with the view."""
    base = tex.copy()
    s_ref = view.scale

    def upd(m):
        new = base.copy().scale(view.scale / s_ref).move_to(pos_fn())
        if opacity_fn is not None:
            new.set_opacity(opacity_fn())
        m.become(new)

    upd(tex)
    tex.add_updater(upd)
    return tex


def arrow(a, b, color, sw=6, tip=0.26):
    return Arrow(a, b, buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=0.6,
                 max_stroke_width_to_length_ratio=30)


def wrap(a):
    """Angle reduced to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


# ---- parts shared by the flat and the 3D figure -----------------------------
def u_arrow(view, p0, ang_fn):
    def mk():
        a = ang_fn()
        return arrow(view.p(p0[0], p0[1]),
                     view.p(p0[0] + np.cos(a), p0[1] + np.sin(a)),
                     U_C, sw=6, tip=0.24 * min(1.0, view.scale))
    return always_redraw(mk)


def u_label(view, p0, ang_fn, fs=38, r=1.28):
    tex = MathTex(r"\hat{u}", font_size=fs, color=U_C)
    return follow(tex, view, lambda: view.p(p0[0] + r * np.cos(ang_fn()),
                                            p0[1] + r * np.sin(ang_fn())))


def theta_arc(view, p0, g_ang, psi_fn, rho=0.42):
    def mk():
        psi = wrap(psi_fn())
        m = VMobject(color=TH_C, stroke_width=3.5)
        if abs(psi) < 0.03:
            return m
        angs = np.linspace(g_ang, g_ang + psi, 30)
        m.set_points_as_corners([view.p(p0[0] + rho * np.cos(a),
                                        p0[1] + rho * np.sin(a)) for a in angs])
        return m
    return always_redraw(mk)


def theta_label(view, p0, g_ang, psi_fn, fs=34, r=0.72):
    tex = MathTex(r"\theta", font_size=fs, color=TH_C)

    def pos():
        a = g_ang + wrap(psi_fn()) / 2
        return view.p(p0[0] + r * np.cos(a), p0[1] + r * np.sin(a))

    def op():
        return float(np.clip((abs(wrap(psi_fn())) - 0.20) / 0.25, 0.0, 1.0))

    return follow(tex, view, pos, op)


# ---- the flat figure from the board (general f, P0(x0, y0)) -----------------
class BoardFig:
    P0 = (1.55, 1.45)
    G_ANG = 34 * DEGREES
    G_LEN = 2.05

    def __init__(self, view, psi):
        v, p0 = view, self.P0
        ga = self.G_ANG
        self.view = v
        ang = lambda: ga + psi.get_value()

        ax_x = arrow(v.p(-0.25, 0), v.p(4.3, 0), GREY_A, sw=3, tip=0.2)
        ax_y = arrow(v.p(0, -0.25), v.p(0, 3.55), GREY_A, sw=3, tip=0.2)
        lx = MathTex("x", font_size=34).move_to(v.p(4.5, -0.05))
        ly = MathTex("y", font_size=34).move_to(v.p(-0.05, 3.75))
        self.axes = VGroup(ax_x, ax_y, lx, ly)

        dx = DashedLine(v.p(p0[0], 0), v.p(*p0), color=GREY_B, stroke_width=2,
                        dash_length=0.08)
        dy = DashedLine(v.p(0, p0[1]), v.p(*p0), color=GREY_B, stroke_width=2,
                        dash_length=0.08)
        tx = MathTex("x_0", font_size=32).move_to(v.p(p0[0], -0.3))
        ty = MathTex("y_0", font_size=32).move_to(v.p(-0.32, p0[1]))
        dot = Dot(v.p(*p0), radius=0.07, color=WHITE)
        tp = MathTex("P_0", font_size=34).move_to(v.p(p0[0] + 0.3, p0[1] - 0.3))
        self.point = VGroup(dx, dy, tx, ty, dot, tp)

        tip = (p0[0] + self.G_LEN * np.cos(ga), p0[1] + self.G_LEN * np.sin(ga))
        g = arrow(v.p(*p0), v.p(*tip), G_C, sw=6, tip=0.26)
        tg = MathTex(r"\left(\nabla f\right)_{P_0}", font_size=36, color=G_C)
        tg.move_to(v.p(tip[0] + 0.25, tip[1] + 0.33))
        self.grad = VGroup(g, tg)

        self.static = VGroup(self.axes, self.point, self.grad)
        self.u = u_arrow(v, p0, ang)
        self.ul = u_label(v, p0, ang)
        self.arc = theta_arc(v, p0, ga, psi.get_value)
        self.thl = theta_label(v, p0, ga, psi.get_value)
        self.dynamic = [self.arc, self.thl, self.u, self.ul]


# ---- the flat figure for the example ----------------------------------------
class ExampleFig:
    def __init__(self, view, psi):
        v, p0 = view, (X0, Y0)
        self.view = v
        ang = lambda: G_ANG + psi.get_value()
        ax_x = arrow(v.p(-1.5, 0), v.p(2.6, 0), GREY_A, sw=3, tip=0.2)
        ax_y = arrow(v.p(0, -1.45), v.p(0, 2.4), GREY_A, sw=3, tip=0.2)
        lx = MathTex("x", font_size=34).move_to(v.p(2.8, -0.03))
        ly = MathTex("y", font_size=34).move_to(v.p(-0.03, 2.6))
        ticks = VGroup()
        for k in (-1, 1, 2):
            ticks.add(Line(v.p(k, -0.06), v.p(k, 0.06), color=GREY_A, stroke_width=2))
            ticks.add(Line(v.p(-0.06, k), v.p(0.06, k), color=GREY_A, stroke_width=2))
        self.axes = VGroup(ax_x, ax_y, lx, ly, ticks)

        dx = DashedLine(v.p(X0, 0), v.p(*p0), color=GREY_B, stroke_width=2,
                        dash_length=0.08)
        dy = DashedLine(v.p(0, Y0), v.p(*p0), color=GREY_B, stroke_width=2,
                        dash_length=0.08)
        tx = MathTex("1", font_size=30).move_to(v.p(X0, -0.3))
        ty = MathTex("1", font_size=30).move_to(v.p(-0.28, Y0))
        dot = Dot(v.p(*p0), radius=0.07, color=WHITE)
        tp = MathTex("P_0", font_size=32).move_to(v.p(X0 + 0.42, Y0 + 0.02))
        self.point = VGroup(dx, dy, tx, ty, dot, tp)

        g = arrow(v.p(*p0), v.p(X0 + GX, Y0 + GY), G_C, sw=6, tip=0.26)
        tg = MathTex(r"\left(\nabla f\right)_{P_0}", font_size=34, color=G_C)
        tg.move_to(v.p(X0 + GX - 0.3, Y0 + GY + 0.6))
        self.grad = VGroup(g, tg)

        self.static = VGroup(self.axes, self.point, self.grad)
        self.u = u_arrow(v, p0, ang)
        self.ul = u_label(v, p0, ang, fs=36)
        self.arc = theta_arc(v, p0, G_ANG, psi.get_value)
        self.thl = theta_label(v, p0, G_ANG, psi.get_value, fs=32)
        self.dynamic = [self.arc, self.thl, self.u, self.ul]


# ---- the 3D figure: z = 4 - x^2 - y^2 over P0(1,1) --------------------------
class DomeFig:
    ELL = 1.25                          # half-length of the tangent segment

    def __init__(self, view, ang_fn, psi_fn=None, with_grad=False):
        v = view
        self.view = v
        p0 = (X0, Y0)

        # axes
        ax_x = arrow(v.p(-2.4, 0, 0), v.p(2.9, 0, 0), GREY_A, sw=2.5, tip=0.18)
        ax_y = arrow(v.p(0, -2.4, 0), v.p(0, 3.0, 0), GREY_A, sw=2.5, tip=0.18)
        ax_z = arrow(v.p(0, 0, 0), v.p(0, 0, 4.6), GREY_A, sw=2.5, tip=0.18)
        lx = MathTex("x", font_size=32).move_to(v.po(2.9, 0, 0, 0.17, -0.1))
        ly = MathTex("y", font_size=32).move_to(v.po(0, 3.0, 0, 0.22, 0.02))
        lz = MathTex("z", font_size=32).move_to(v.po(0, 0, 4.6, -0.2, 0.05))
        self.axes = VGroup(ax_x, ax_y, ax_z, lx, ly, lz)

        # the surface: depth-sorted quads of a polar mesh, lit from the upper left
        NR, NT = 10, 44
        rs = np.linspace(0.0, R_DOME, NR + 1)
        ts = np.linspace(0.0, TAU, NT + 1)
        light = np.array([0.45, -0.55, 0.70])
        light /= np.linalg.norm(light)
        faces = []
        for i in range(NR):
            for j in range(NT):
                corners = [(rs[i], ts[j]), (rs[i + 1], ts[j]),
                           (rs[i + 1], ts[j + 1]), (rs[i], ts[j + 1])]
                pts3 = [(r * np.cos(t), r * np.sin(t), 4.0 - r * r) for r, t in corners]
                if i == 0:
                    pts3 = pts3[1:]
                c = np.mean(pts3, axis=0)
                n = np.array([2 * c[0], 2 * c[1], 1.0])
                n /= np.linalg.norm(n)
                b = 0.25 + 0.75 * max(0.0, float(n @ light))
                poly = Polygon(*[v.p(*q) for q in pts3],
                               stroke_color=WHITE, stroke_width=0.6,
                               stroke_opacity=0.22,
                               fill_color=interpolate_color(ManimColor("#123A66"),
                                                            ManimColor("#4FA3E8"), b),
                               fill_opacity=0.50)
                faces.append((v.depth(*c), poly))
        faces.sort(key=lambda t: t[0])
        self.surface = VGroup(*[poly for _, poly in faces])
        rim = VMobject(color=BLUE_B, stroke_width=1.5, stroke_opacity=0.7)
        rim.set_points_as_corners([v.p(R_DOME * np.cos(t), R_DOME * np.sin(t), 0)
                                   for t in np.linspace(0, TAU, 90)])
        self.surface.add(rim)
        self.surf_label = MathTex("z = 4 - x^2 - y^2", font_size=34, color=BLUE_B)
        self.surf_label.move_to(v.po(0, 0, 4.0, -1.75, -0.05))

        # P0 on the ground and the point above it on the surface
        self.p0_dot = Dot(v.p(X0, Y0, 0), radius=0.065, color=WHITE)
        self.p0_lab = MathTex("P_0", font_size=32).move_to(v.po(X0, Y0, 0, 0.06, -0.3))
        self.riser = DashedLine(v.p(X0, Y0, 0), v.p(X0, Y0, Z0), color=WHITE,
                                stroke_width=2, dash_length=0.07)
        self.s_dot = Dot(v.p(X0, Y0, Z0), radius=0.075, color=WHITE)
        self.point = VGroup(self.riser, self.p0_dot, self.p0_lab, self.s_dot)

        self.static = VGroup(self.axes, self.surface, self.surf_label, self.point)

        # the gradient on the ground
        self.grad = VGroup()
        if with_grad:
            g = arrow(v.p(X0, Y0, 0), v.p(X0 + GX, Y0 + GY, 0), G_C, sw=5, tip=0.22)
            tg = MathTex(r"\left(\nabla f\right)_{P_0}", font_size=32, color=G_C)
            tg.move_to(v.po(X0 + GX, Y0 + GY, 0, -1.27, 0.4))
            self.grad = VGroup(g, tg)
            self.static.add(self.grad)

        # the slice through P0 along u, its trace, and the tangent line
        def trace_pts():
            a = ang_fn()
            c, s = np.cos(a), np.sin(a)
            b = X0 * c + Y0 * s
            rt = np.sqrt(b * b + R_DOME ** 2 - X0 * X0 - Y0 * Y0)
            return [v.p(X0 + t * c, Y0 + t * s, f(X0 + t * c, Y0 + t * s))
                    for t in np.linspace(-b - rt, -b + rt, 44)]

        def mk_slice():
            pts = trace_pts()
            m = VMobject(fill_color=TRACE_C, fill_opacity=0.20,
                         stroke_color=TRACE_C, stroke_width=1.5, stroke_opacity=0.6)
            m.set_points_as_corners(pts + [pts[0]])
            return m

        def mk_trace():
            m = VMobject(color=TRACE_C, stroke_width=4.5)
            m.set_points_as_corners(trace_pts())
            return m

        def tan_ends(horizontal=False):
            a = ang_fn()
            c, s = np.cos(a), np.sin(a)
            d = 0.0 if horizontal else dd(a)
            L = self.ELL / np.sqrt(1.0 + (v.zk * d) ** 2)
            if horizontal:
                L *= 0.8
            return (v.p(X0 - L * c, Y0 - L * s, Z0 - L * d),
                    v.p(X0 + L * c, Y0 + L * s, Z0 + L * d))

        self.slice = always_redraw(mk_slice)
        self.trace = always_redraw(mk_trace)
        self.level = always_redraw(lambda: DashedLine(
            *tan_ends(True), color=WHITE, stroke_width=2, stroke_opacity=0.75,
            dash_length=0.07))
        self.tangent = always_redraw(lambda: Line(
            *tan_ends(), color=TAN_C, stroke_width=7))
        self.u = u_arrow(v, p0, ang_fn)
        self.ul = u_label(v, p0, ang_fn, fs=34, r=1.3)
        self.top_dot = always_redraw(lambda: Dot(v.p(X0, Y0, Z0), radius=0.075 * min(1, v.scale),
                                                 color=WHITE))
        self.dynamic = [self.slice, self.trace, self.level, self.tangent,
                        self.top_dot, self.u, self.ul]
        if psi_fn is not None:
            self.arc = theta_arc(v, p0, G_ANG, psi_fn, rho=0.5)
            self.dynamic.insert(0, self.arc)


def T(*parts, fs=32, **kw):
    return Tex(*parts, font_size=fs, **kw)


DU = r"\left(D_{\hat{u}} f\right)_{P_0}"
GR = r"\left(\nabla f\right)_{P_0}"
NG = r"\left\|\left(\nabla f\right)_{P_0}\right\|"


class MaxMinZero(Scene):
    def w(self, t):
        self.wait(0.15 if FAST else t)

    def construct(self):
        LEFT_X = -6.75

        def put(m, y, x=LEFT_X):
            m.move_to(np.array([x, y, 0]), aligned_edge=LEFT)
            return m

        # =====================================================================
        # A.  Reminder (I), with the 3D picture, then (II)
        # =====================================================================
        hdr = put(T(r"\textbf{Reminders:}", fs=34, color=BLUE_B), 3.6)
        I = T(r"\textbf{(I)}", r"\quad $" + DU + r"$ gives the rate of change of "
              r"$z=f(x,y)$ in the direction $\hat{u}$ at the point $P_0$.", fs=30)
        I[0].set_color(KEY_C)
        put(I, 2.95, LEFT_X + 0.25)
        II = T(r"\textbf{(II)}", r"\quad $" + DU + "=" + GR + r"\cdot\hat{u}$,"
               r"\quad where $\nabla f=\dfrac{\partial f}{\partial x}\,\hat{\imath}"
               r"+\dfrac{\partial f}{\partial y}\,\hat{\jmath}$,\quad "
               r"$\hat{u}=u_1\hat{\imath}+u_2\hat{\jmath}$\quad and\quad $u_1^2+u_2^2=1$.",
               fs=30)
        II[0].set_color(KEY_C)
        put(II, 2.1, LEFT_X + 0.25)

        phiA = ValueTracker(20 * DEGREES)
        if "A" in STAGES:
            self.play(FadeIn(hdr), run_time=0.8)
            self.play(Write(I), run_time=2.6)
            self.w(1.5)

            vA = View3((2.45, -2.1), 1.35)
            A = DomeFig(vA, phiA.get_value)
            ex = VGroup(
                T(r"For example:", fs=30, color=GREY_A),
                MathTex(r"f(x,y)=4-x^2-y^2", font_size=34),
                MathTex(r"P_0(1,1)", font_size=34),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            put(ex, 1.45, LEFT_X + 0.25)

            self.play(FadeIn(ex[0]), Write(ex[1]), run_time=1.4)
            self.play(LaggedStart(*[Create(m) for m in A.axes[:3]], lag_ratio=0.2),
                      FadeIn(A.axes[3:]), run_time=1.3)
            self.play(FadeIn(A.surface, lag_ratio=0.002), FadeIn(A.surf_label),
                      run_time=2.0)
            self.play(Write(ex[2]), FadeIn(A.p0_dot, scale=2), FadeIn(A.p0_lab),
                      run_time=1.0)
            self.play(Create(A.riser), run_time=0.7)
            self.add(A.top_dot)
            self.play(FadeIn(A.s_dot, scale=2), run_time=0.5)
            self.w(0.4)

            # the direction u, the slice it cuts, the tangent line
            self.play(GrowArrow(A.u), run_time=0.9)
            self.add(A.ul)
            self.play(FadeIn(A.ul), run_time=0.4)
            self.add(A.slice, A.trace, A.u, A.ul, A.top_dot)
            self.play(FadeIn(A.slice), Create(A.trace), run_time=1.5)
            self.add(A.level, A.tangent, A.top_dot)
            self.play(Create(A.tangent), FadeIn(A.level), run_time=1.1)

            cap = VGroup(
                T(r"The slope of the tangent line", fs=30),
                T(r"in the direction $\hat{u}$ is the", fs=30),
                T(r"directional derivative:", fs=30),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            put(cap, -0.55, LEFT_X + 0.25)
            cap[0][0][-11:].set_color(TAN_C)
            slope = MathTex(r"\text{slope}", "=", DU, "=", font_size=36)
            slope[0].set_color(TAN_C)
            put(slope, -1.8, LEFT_X + 0.25)
            val = DecimalNumber(dd(phiA.get_value()), num_decimal_places=2,
                                include_sign=True, font_size=38, color=TAN_C)
            val.next_to(slope, RIGHT, buff=0.2)
            val.add_updater(lambda m: m.set_value(dd(phiA.get_value()))
                            .next_to(slope, RIGHT, buff=0.2))
            self.play(FadeIn(cap), run_time=1.0)
            self.play(Write(slope), FadeIn(val), run_time=1.3)
            self.w(1.2)

            # turn u once around P0
            self.play(phiA.animate.set_value(20 * DEGREES + TAU), run_time=14,
                      rate_func=rate_functions.ease_in_out_sine)
            self.w(1.0)
            every = T(r"Each direction $\hat{u}$ has its own tangent line,", fs=28,
                      color=GREY_A)
            every2 = T(r"and its own slope $" + DU + "$.", fs=28, color=GREY_A)
            eg = VGroup(every, every2).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            put(eg, -3.05, LEFT_X + 0.25)
            self.play(FadeIn(eg), run_time=0.8)
            self.w(2.5)

            val.clear_updaters()
            for m in A.dynamic:
                m.clear_updaters()
            self.play(FadeOut(VGroup(A.static, *A.dynamic, ex, cap, slope, val, eg)),
                      run_time=1.0)
        else:
            self.add(hdr, I)

        self.play(Write(II), run_time=3.2)
        self.w(3.0)

        # =====================================================================
        # B.  The question, 5 s, the hint, 5 s
        # =====================================================================
        Q1 = T(r"\textbf{Question:}", r"\quad For a given function $z=f(x,y)$ and a point "
               r"$P_0(x_0,y_0)$, can we find the directions", fs=30)
        Q1[0].set_color(KEY_C)
        Q2 = T(r"in which the rate of change in ", r"\textbf{(I)}",
               r" is maximum, minimum, and zero?", fs=30)
        Q2[1].set_color(KEY_C)
        put(Q1, 1.0)
        Q2.move_to(Q1[1].get_left() + DOWN * 0.58, aligned_edge=LEFT).shift(RIGHT * 0.28)
        H = T(r"\textbf{Hint:}", r"\quad Use ", r"\textbf{(II)}",
              r" and the definition of the dot product.", fs=30)
        H[0].set_color(KEY_C)
        H[2].set_color(KEY_C)
        put(H, -0.5)

        if "B" in STAGES:
            self.play(Write(Q1), run_time=2.6)
            self.play(Write(Q2), run_time=2.0)
            self.w(5.0)
            self.play(Write(H), run_time=1.6)
            self.w(5.0)
        else:
            self.add(Q1, Q2, H)

        # =====================================================================
        # C.  The answer
        # =====================================================================
        ans_k = T(r"\textbf{Answer:}", fs=32, color=KEY_C)
        ans = MathTex(DU, "=", GR + r"\cdot\hat{u}", "=",
                      NG + r"\,\left\|\hat{u}\right\|\cos\theta", "=",
                      NG + r"\cos\theta", font_size=36)
        put(ans_k, -1.7)
        ans.next_to(ans_k, RIGHT, buff=0.35)
        note = T(r"($\theta$ is the angle between $" + GR + r"$ and $\hat{u}$,\quad "
                 r"$\left\|\hat{u}\right\|=1$)", fs=28, color=GREY_A)
        note.next_to(ans, DOWN, buff=0.35, aligned_edge=LEFT)

        if "C" not in STAGES and "D" not in STAGES:
            return

        self.play(FadeIn(ans_k), run_time=0.5)
        self.play(Write(ans[0:3]), run_time=1.8)
        self.w(1.0)
        self.play(Write(ans[3:5]), run_time=1.8)
        self.play(FadeIn(note), run_time=0.8)
        self.w(1.5)
        self.play(Write(ans[5:7]), run_time=1.5)
        box = SurroundingRectangle(ans[6], color=TH_C, buff=0.12, stroke_width=2.5)
        self.play(Create(box), run_time=0.8)
        self.w(2.0)

        # clear the board down to the answer, which moves to the top
        top = VGroup(ans_k, ans, box)
        self.play(FadeOut(VGroup(hdr, I, II, Q1, Q2, H, note)),
                  top.animate.move_to(np.array([0, 3.45, 0])), run_time=1.5)

        # ---- the board's figure, in the centre -------------------------------
        psi = ValueTracker(56 * DEGREES)
        vB = View((-3.35, -3.1), 1.5)
        B = BoardFig(vB, psi)
        self.play(LaggedStart(Create(B.axes[0]), Create(B.axes[1]), lag_ratio=0.3),
                  FadeIn(B.axes[2:]), run_time=1.2)
        self.play(FadeIn(B.point), run_time=1.0)
        self.play(GrowArrow(B.grad[0]), FadeIn(B.grad[1]), run_time=1.0)
        self.play(GrowArrow(B.u), run_time=0.8)
        self.add(B.arc, B.thl, B.u, B.ul)
        self.play(FadeIn(B.ul), FadeIn(B.thl), Create(B.arc), run_time=0.8)
        turn = VGroup(T(r"As $\hat{u}$ turns,", fs=30, color=GREY_A),
                      T(r"only $\cos\theta$ changes.", fs=30, color=GREY_A))
        turn.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        turn.move_to(np.array([5.2, -0.3, 0]))
        self.play(FadeIn(turn), run_time=0.7)
        self.w(0.8)
        self.play(psi.animate.set_value(56 * DEGREES - TAU), run_time=10,
                  rate_func=rate_functions.ease_in_out_sine)
        psi.set_value(56 * DEGREES)
        self.w(1.0)

        # ---- side by side with the 3D graph, for the example -----------------
        for m in B.dynamic:
            m.clear_updaters()
        ex_line = VGroup(
            MathTex(r"\text{Example:}\quad f(x,y)=4-x^2-y^2,\quad P_0(1,1)", font_size=30),
            MathTex(GR + r"=-2\,\hat{\imath}-2\,\hat{\jmath},\qquad "
                    + NG + r"=2\sqrt{2}", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        put(ex_line, 2.36, LEFT_X + 0.1)
        vE = View((-4.0, -0.75), 0.93)
        E = ExampleFig(vE, psi)
        vD = View3((3.4, -0.72), 1.05)
        D = DomeFig(vD, lambda: G_ANG + psi.get_value(), psi.get_value, with_grad=True)

        self.play(FadeOut(VGroup(B.static, *B.dynamic, turn)), run_time=0.9)
        self.play(FadeIn(ex_line), run_time=0.9)
        self.play(FadeIn(E.static), FadeIn(D.static, lag_ratio=0.001), run_time=1.6)
        self.add(*E.dynamic, *D.dynamic)
        self.play(*[FadeIn(m) for m in E.dynamic + D.dynamic], run_time=1.0)

        # live readout along the bottom
        def theta():
            return abs(wrap(psi.get_value()))

        th_tex = MathTex(r"\theta", "=", font_size=34, color=TH_C)
        th_val = DecimalNumber(theta() / PI, num_decimal_places=2, font_size=34, color=TH_C)
        th_pi = MathTex(r"\pi", font_size=34, color=TH_C)
        dv_tex = MathTex(DU, "=", NG + r"\cos\theta", "=", r"2\sqrt{2}\,\cos\theta", "=",
                         font_size=34)
        dv_val = DecimalNumber(dd(G_ANG + psi.get_value()), num_decimal_places=2,
                               include_sign=True, font_size=36, color=TAN_C)
        th_tex.move_to(np.array([-5.6, -3.6, 0]))
        dv_tex.move_to(np.array([0.9, -3.6, 0]))

        def upd_th(m):
            m.set_value(theta() / PI).next_to(th_tex, RIGHT, buff=0.15)
            th_pi.next_to(m, RIGHT, buff=0.06, aligned_edge=DOWN)

        def upd_dv(m):
            m.set_value(dd(G_ANG + psi.get_value())).next_to(dv_tex, RIGHT, buff=0.18)

        upd_th(th_val)
        upd_dv(dv_val)
        th_val.add_updater(upd_th)
        dv_val.add_updater(upd_dv)
        readout = VGroup(th_tex, th_val, th_pi, dv_tex, dv_val)
        self.play(FadeIn(readout), run_time=0.8)
        self.w(1.5)

        def right_angle(view, sgn):
            """A small right-angle mark at P0 between the gradient and u."""
            k = 0.26
            g = np.array([np.cos(G_ANG), np.sin(G_ANG)])
            u = np.array([np.cos(G_ANG + sgn * PI / 2), np.sin(G_ANG + sgn * PI / 2)])
            q = [np.array([X0, Y0]) + k * g, np.array([X0, Y0]) + k * (g + u),
                 np.array([X0, Y0]) + k * u]
            m = VMobject(color=TH_C, stroke_width=3)
            m.set_points_as_corners([view.p(*pt) for pt in q])
            return m

        def stop(target, run, cap_tex, hold=4.0, mark=0):
            self.play(psi.animate.set_value(target), run_time=run,
                      rate_func=rate_functions.ease_in_out_sine)
            c = MathTex(*cap_tex, font_size=36, color=TH_C)
            c.move_to(np.array([0, -2.75, 0]))
            bg = SurroundingRectangle(c, color=TH_C, buff=0.16, stroke_width=2,
                                      corner_radius=0.1)
            extra = []
            if mark:
                extra = [right_angle(vE, mark), right_angle(vD, mark)]
            self.play(FadeIn(c), Create(bg), *[Create(m) for m in extra], run_time=0.7)
            self.w(hold)
            self.play(FadeOut(c), FadeOut(bg), *[FadeOut(m) for m in extra], run_time=0.5)

        stop(0.0, 2.4,
             [r"\theta=0:", r"\quad " + DU + "=" + NG + r"\quad\text{(maximum)}"])
        stop(-PI / 2, 3.2,
             [r"\theta=\tfrac{\pi}{2}:", r"\quad " + DU + r"=0"], mark=-1)
        stop(-PI, 3.2,
             [r"\theta=\pi:", r"\quad " + DU + "=-" + NG + r"\quad\text{(minimum)}"])
        stop(-3 * PI / 2, 3.2,
             [r"\theta=\tfrac{\pi}{2}\ \text{again, on the other side}:",
              r"\quad " + DU + r"=0"], mark=1)
        self.play(psi.animate.set_value(56 * DEGREES - TAU), run_time=2.6,
                  rate_func=rate_functions.ease_in_out_sine)
        psi.set_value(56 * DEGREES)
        self.w(1.0)

        if "D" not in STAGES:
            return

        # =====================================================================
        # D.  Summary at the bottom of the board, and the open question
        # =====================================================================
        th_val.clear_updaters()
        dv_val.clear_updaters()
        self.play(FadeOut(readout), FadeOut(ex_line),
                  *move_fig(vE, E.static, (-3.7, 0.58), 0.85),
                  *move_fig(vD, D.static, (2.9, 0.25), 0.78),
                  run_time=1.6)

        s0 = MathTex(DU, font_size=38)
        put(s0, -2.4, -6.8)
        l1 = T(r"maximum value $=" + NG + r"$,\quad which occurs in\quad $\hat{u}_{\max}="
               r"\dfrac{" + GR + "}{" + NG + "}$", fs=30)
        l2 = T(r"minimum value $=-" + NG + r"$,\quad which happens in\quad $\hat{u}_{\min}="
               r"-\dfrac{" + GR + "}{" + NG + "}$", fs=30)
        l3 = T(r"zero in two directions", fs=30)
        put(l1, -1.4, -4.15)
        put(l2, -2.4, -4.15)
        put(l3, -3.4, -4.15)
        src = s0.get_right() + RIGHT * 0.12
        a1 = arrow(src + UP * 0.12, l1.get_left() + LEFT * 0.15 + DOWN * 0.05, WHITE, sw=3, tip=0.18)
        a2 = arrow(src, l2.get_left() + LEFT * 0.15, WHITE, sw=3, tip=0.18)
        a3 = arrow(src + DOWN * 0.12, l3.get_left() + LEFT * 0.15 + UP * 0.05, WHITE, sw=3, tip=0.18)
        l1[0][:12].set_color(TAN_C)
        l2[0][:12].set_color(RED_B)
        l3[0][:4].set_color(TH_C)

        self.play(Write(s0), run_time=1.0)
        self.play(GrowArrow(a1), psi.animate.set_value(0.0), run_time=1.3)
        self.play(Write(l1), run_time=2.6)
        self.w(2.5)
        self.play(GrowArrow(a2), psi.animate.set_value(-PI), run_time=1.6)
        self.play(Write(l2), run_time=2.6)
        self.w(2.5)
        self.play(GrowArrow(a3), psi.animate.set_value(-PI / 2), run_time=1.6)
        self.play(Write(l3), run_time=1.2)

        # the second zero direction, as a ghost, in both figures
        def ghost(view):
            a = G_ANG + PI / 2
            return arrow(view.p(X0, Y0), view.p(X0 + np.cos(a), Y0 + np.sin(a)),
                         U_C, sw=5, tip=0.2 * view.scale).set_opacity(0.55)

        g1, g2 = ghost(vE), ghost(vD)
        self.play(GrowArrow(g1), GrowArrow(g2), run_time=1.0)
        self.w(1.5)

        to_q = arrow(l3.get_right() + RIGHT * 0.25, l3.get_right() + RIGHT * 1.2, WHITE,
                     sw=3, tip=0.18)
        fq = T(r"\textbf{Question:}", r"\quad What are these two vectors $\hat{u}$?", fs=30)
        fq[0].set_color(KEY_C)
        fq.next_to(to_q, RIGHT, buff=0.3)
        self.play(GrowArrow(to_q), run_time=0.7)
        self.play(Write(fq), run_time=1.8)
        fbox = SurroundingRectangle(fq, color=KEY_C, buff=0.18, stroke_width=2.5,
                                    corner_radius=0.1)
        self.play(Create(fbox), run_time=0.8)
        self.play(Indicate(fq[1], color=WHITE, scale_factor=1.06), run_time=1.2)
        self.w(8.0)
