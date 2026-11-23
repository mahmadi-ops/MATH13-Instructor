"""Recap of the session on the directional derivative: the concept, with no
worked examples.  It follows the course notes (section "Directional Derivatives
and Gradient Vectors", subsections "The Definition of the Directional
Derivative" and "The Gradient Vector").

    manim -qh --fps 30 --disable_caching -o dirderiv_definition_chain_rule \
        dirderiv_definition_chain_rule.py DefinitionAndChainRule

The video runs in five parts:

1. The definition.  In the xy-plane: P0, the unit vector u, and the point
   P(x0 + s u1, y0 + s u2) a distance s from P0 along u; then the limit.
2. The slope of the tangent line.  Over a general surface z = f(x,y): the
   vertical plane through P0 along u cuts the surface in a curve; the secant
   through (P0, f(P0)) and (P, f(P)) has slope (f(P) - f(P0))/s, rise over
   run, and turns onto the tangent line as s -> 0.  Statement (I).
3. The line through P0 and P, back in the xy-plane, and why it is
   parametrized by x(s) = x0 + s u1, y(s) = y0 + s u2: the point P runs along
   the line as s changes, and its coordinates are read off on the axes.  On
   that line f depends on s alone.
4. The chain rule, with the dependency diagram of the notes, giving the dot
   product of the gradient vector with u.  Statement (II).
5. Summary: (I) and (II), the two reminders that open the next lecture.

The 3D picture is drawn with a fixed orthographic projection inside an
ordinary Scene (faces depth-sorted once), as in dirderiv_max_min_zero.py.  The
surface is only a prop for "a general f"; its formula is never shown.
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

# ---- colours ----------------------------------------------------------------
U_C = "#5CC8FF"                         # the unit vector u
G_C = "#FF7B3D"                         # the gradient
TAN_C = "#7CE07C"                       # tangent line
SEC_C = "#FF6060"                       # secant line
DISP_C = GOLD                           # the displacement s u and the point P
KEY_C = YELLOW

STAGES = os.environ.get("STAGES", "12345")
FAST = os.environ.get("FAST", "") == "1"


def v3(c):
    c = np.array(c, dtype=float)
    return c if c.size == 3 else np.array([c[0], c[1], 0.0])


# ---- views: where a figure sits on screen -----------------------------------
class View:
    """Maps figure coordinates to the screen: centre + scale * proj(x, y, z)."""

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

    def __init__(self, center, scale, az=-25 * DEGREES, el=20 * DEGREES, zk=0.55):
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

    def cam(self):
        return np.array([np.cos(self.az) * np.cos(self.el),
                         np.sin(self.az) * np.cos(self.el), np.sin(self.el)])


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


def ramp(x, lo, width):
    """0 below lo, 1 above lo + width."""
    return float(np.clip((x - lo) / width, 0.0, 1.0))


def arrow(a, b, color, sw=6, tip=0.26):
    return Arrow(a, b, buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=0.6,
                 max_stroke_width_to_length_ratio=30)


def T(*parts, fs=30, **kw):
    return Tex(*parts, font_size=fs, **kw)


DU = r"\left(D_{\hat{u}} f\right)_{P_0}"
GR = r"\left(\nabla f\right)_{P_0}"
DFDS = r"\left(\frac{df}{ds}\right)_{\hat{u},\,P_0}"
FX0 = r"\left(\frac{\partial f}{\partial x}\right)_{P_0}"
FY0 = r"\left(\frac{\partial f}{\partial y}\right)_{P_0}"


# ---- the construction in the xy-plane (parts 1 and 3) -----------------------
class PlaneFig:
    P0 = (1.35, 0.95)
    ANG = 33 * DEGREES

    def __init__(self, view, s, with_line=False):
        v, p0, a = view, self.P0, self.ANG
        u = (np.cos(a), np.sin(a))
        self.view = v

        ax_x = arrow(v.p(-0.25, 0), v.p(5.25, 0), GREY_A, sw=3, tip=0.2)
        ax_y = arrow(v.p(0, -0.25), v.p(0, 3.45), GREY_A, sw=3, tip=0.2)
        lx = MathTex("x", font_size=34).move_to(v.p(5.45, -0.03))
        ly = MathTex("y", font_size=34).move_to(v.p(-0.03, 3.65))
        self.axes = VGroup(ax_x, ax_y, lx, ly)

        dot = Dot(v.p(*p0), radius=0.075, color=WHITE)
        if with_line:
            tp = MathTex("P_0", font_size=32).move_to(v.p(p0[0] + 0.33, p0[1] - 0.27))
        else:
            tp = MathTex("P_0(x_0,y_0)", font_size=32).move_to(
                v.p(p0[0] - 0.15, p0[1] - 0.36))
        self.point = VGroup(dot, tp)

        self.u = arrow(v.p(*p0), v.p(p0[0] + u[0], p0[1] + u[1]), U_C, sw=6, tip=0.24)
        self.ul = MathTex(r"\hat{u}", font_size=38, color=U_C).move_to(
            v.p(p0[0] + 0.55 * u[0] - 0.22, p0[1] + 0.55 * u[1] + 0.3))
        self.uvec = VGroup(self.u, self.ul)
        self.static = VGroup(self.axes, self.point, self.uvec)

        def P():
            t = s.get_value()
            return p0[0] + t * u[0], p0[1] + t * u[1]

        self.disp = always_redraw(lambda: Line(v.p(*p0), v.p(*P()), color=DISP_C,
                                               stroke_width=8, stroke_opacity=0.75))
        self.leg_x = always_redraw(lambda: DashedLine(
            v.p(*p0), v.p(P()[0], p0[1]), color=GREY_A, stroke_width=2.5, dash_length=0.09))
        self.leg_y = always_redraw(lambda: DashedLine(
            v.p(P()[0], p0[1]), v.p(*P()), color=GREY_A, stroke_width=2.5, dash_length=0.09))
        self.pdot = always_redraw(lambda: Dot(v.p(*P()), radius=0.075 * min(1, v.scale),
                                              color=DISP_C))
        far = lambda: ramp(s.get_value(), 1.5, 0.5)
        self.sl = follow(MathTex(r"s\hat{u}", font_size=34, color=DISP_C), v,
                         lambda: v.p(p0[0] + 0.62 * (P()[0] - p0[0]) - 0.2,
                                     p0[1] + 0.62 * (P()[1] - p0[1]) + 0.33), far)
        self.lxl = follow(MathTex(r"su_1\hat{\imath}", font_size=30), v,
                          lambda: v.p((p0[0] + P()[0]) / 2 + 0.25, p0[1] - 0.27),
                          lambda: ramp(s.get_value(), 0.9, 0.5))
        self.lyl = follow(MathTex(r"su_2\hat{\jmath}", font_size=30), v,
                          lambda: v.p(P()[0] + 0.5, (p0[1] + P()[1]) / 2),
                          lambda: ramp(s.get_value(), 0.9, 0.5))

        if not with_line:
            self.pl = follow(MathTex(r"P(x_0+su_1,\;y_0+su_2)", font_size=32, color=DISP_C),
                             v, lambda: v.p(P()[0] + 0.75, P()[1] + 0.34
                                            + 0.5 * (1 - ramp(s.get_value(), 0.5, 1.0))))
            self.dynamic = [self.disp, self.leg_x, self.leg_y, self.pdot,
                            self.pl, self.sl, self.lxl, self.lyl]
            return

        # part 3: the whole line through P0 and P, and the coordinates of P on the axes
        t1, t2 = -1.45, 4.25
        self.line = Line(v.p(p0[0] + t1 * u[0], p0[1] + t1 * u[1]),
                         v.p(p0[0] + t2 * u[0], p0[1] + t2 * u[1]),
                         color=DISP_C, stroke_width=2.5, stroke_opacity=0.8)
        d0x = DashedLine(v.p(p0[0], 0), v.p(*p0), color=GREY_B, stroke_width=2,
                         dash_length=0.08)
        d0y = DashedLine(v.p(0, p0[1]), v.p(*p0), color=GREY_B, stroke_width=2,
                         dash_length=0.08)
        t0x = MathTex("x_0", font_size=30).move_to(v.p(p0[0], -0.3))
        t0y = MathTex("y_0", font_size=30).move_to(v.p(-0.33, p0[1]))
        self.coords0 = VGroup(d0x, d0y, t0x, t0y)
        self.static.add(self.line, self.coords0)

        self.drop_x = always_redraw(lambda: DashedLine(
            v.p(P()[0], 0), v.p(P()[0], p0[1]), color=DISP_C, stroke_width=2,
            stroke_opacity=0.8, dash_length=0.08))
        self.drop_y = always_redraw(lambda: DashedLine(
            v.p(0, P()[1]), v.p(*P()), color=DISP_C, stroke_width=2,
            stroke_opacity=0.8, dash_length=0.08))
        apart = lambda: ramp(s.get_value(), 1.15, 0.45)
        self.px = follow(MathTex("x_0+su_1", font_size=28, color=DISP_C), v,
                         lambda: v.p(P()[0] + 0.12, -0.3), apart)
        self.py = follow(MathTex("y_0+su_2", font_size=28, color=DISP_C), v,
                         lambda: v.p(-0.82, P()[1]), lambda: ramp(s.get_value(), 0.75, 0.4))
        self.pl = follow(MathTex(r"P\big(x(s),\,y(s)\big)", font_size=32, color=DISP_C),
                         v, lambda: v.p(P()[0] - 0.75, P()[1] + 0.42),
                         lambda: ramp(s.get_value(), 1.3, 0.5))
        self.dynamic = [self.drop_x, self.drop_y, self.disp, self.leg_x, self.leg_y,
                        self.pdot, self.px, self.py, self.pl, self.sl, self.lxl, self.lyl]


# ---- part 2: a general surface z = f(x,y) -----------------------------------
SX0, SY0 = 0.9, 0.2                     # P0, off the axis of the bowl
S_ANG = 55 * DEGREES                    # the direction of u, not radial
SU = (np.cos(S_ANG), np.sin(S_ANG))
R_B = 3.0                               # the surface is drawn for r <= R_B
S_START, S_END = 1.8, 0.03


def fs(x, y):
    """The surface: a prop for a general f, never named on screen."""
    return 0.8 + 0.5 * (x * x + y * y)


def g(t):
    """f along the line through P0 in the direction u."""
    return fs(SX0 + t * SU[0], SY0 + t * SU[1])


SZ0 = g(0.0)
G1 = SX0 * SU[0] + SY0 * SU[1]          # g'(0)


def sec_slope(s):
    return (g(s) - g(0.0)) / s


class SurfFig:
    def __init__(self, view, s):
        v = view
        self.view = v

        ax_x = arrow(v.p(-3.1, 0, 0), v.p(3.5, 0, 0), GREY_A, sw=2.5, tip=0.18)
        ax_y = arrow(v.p(0, -2.6, 0), v.p(0, 3.6, 0), GREY_A, sw=2.5, tip=0.18)
        ax_z = arrow(v.p(0, 0, 0), v.p(0, 0, 6.2), GREY_A, sw=2.5, tip=0.18)
        lx = MathTex("x", font_size=32).move_to(v.po(3.5, 0, 0, 0.14, -0.14))
        ly = MathTex("y", font_size=32).move_to(v.po(0, 3.6, 0, 0.2, 0.05))
        lz = MathTex("z", font_size=32).move_to(v.po(0, 0, 6.2, -0.2, 0.02))
        self.axes = VGroup(ax_x, ax_y, ax_z, lx, ly, lz)

        # the surface: depth-sorted quads of a polar mesh
        NR, NT = 11, 44
        rs = np.linspace(0.0, R_B, NR + 1)
        ts = np.linspace(0.0, TAU, NT + 1)
        cam = v.cam()
        e_r = np.array([-np.sin(v.az), np.cos(v.az), 0.0])
        light = cam + 0.9 * np.array([0, 0, 1.0]) - 0.7 * e_r
        light /= np.linalg.norm(light)
        faces = []
        for i in range(NR):
            for j in range(NT):
                q = [(rs[i], ts[j]), (rs[i + 1], ts[j]), (rs[i + 1], ts[j + 1]),
                     (rs[i], ts[j + 1])]
                pts3 = [(r * np.cos(t), r * np.sin(t), fs(r * np.cos(t), r * np.sin(t)))
                        for r, t in q]
                if i == 0:
                    pts3 = pts3[1:]
                c = np.mean(pts3, axis=0)
                n = np.array([-c[0] * v.zk, -c[1] * v.zk, 1.0])
                n /= np.linalg.norm(n)
                if n @ cam < 0:
                    n = -n
                b = 0.2 + 0.8 * max(0.0, float(n @ light))
                poly = Polygon(*[v.p(*pt) for pt in pts3],
                               stroke_color=WHITE, stroke_width=0.6, stroke_opacity=0.2,
                               fill_color=interpolate_color(ManimColor("#123A66"),
                                                            ManimColor("#4FA3E8"), b),
                               fill_opacity=0.36)
                faces.append((v.depth(c[0], c[1], c[2]), poly))
        faces.sort(key=lambda t: t[0])
        self.surface = VGroup(*[poly for _, poly in faces])
        rim = VMobject(color=BLUE_B, stroke_width=1.5, stroke_opacity=0.7)
        rim.set_points_as_corners([v.p(R_B * np.cos(t), R_B * np.sin(t), fs(R_B, 0))
                                   for t in np.linspace(0, TAU, 90)])
        self.surface.add(rim)
        self.surf_label = MathTex("z = f(x,y)", font_size=34, color=BLUE_B)
        self.surf_label.move_to(v.po(0, 0, 0, 2.2, 4.12))

        # the vertical plane through P0 along u, and its trace on the surface
        def ray(t):
            return SX0 + t * SU[0], SY0 + t * SU[1]

        rt = np.sqrt(G1 ** 2 + R_B ** 2 - SX0 ** 2 - SY0 ** 2)
        t_lo, t_hi = -G1 - rt, -G1 + rt
        tr = [v.p(*ray(t), g(t)) for t in np.linspace(t_lo, t_hi, 60)]
        self.curtain = VMobject(fill_color=GOLD, fill_opacity=0.13, stroke_width=0)
        self.curtain.set_points_as_corners(
            [v.p(*ray(t_lo), 0)] + tr + [v.p(*ray(t_hi), 0), v.p(*ray(t_lo), 0)])
        self.ground_line = Line(v.p(*ray(t_lo), 0), v.p(*ray(t_hi), 0), color=GOLD,
                                stroke_width=1.5, stroke_opacity=0.6)
        self.trace = VMobject(color=GOLD, stroke_width=4.5)
        self.trace.set_points_as_corners(tr)
        self.plane = VGroup(self.curtain, self.ground_line, self.trace)

        # P0 on the ground, u, and the point of the surface above P0
        self.p0_dot = Dot(v.p(SX0, SY0, 0), radius=0.065, color=WHITE)
        self.p0_lab = MathTex("P_0", font_size=30).move_to(v.po(SX0, SY0, 0, -0.12, -0.3))
        self.riser0 = DashedLine(v.p(SX0, SY0, 0), v.p(SX0, SY0, SZ0), color=WHITE,
                                 stroke_width=2, dash_length=0.07)
        self.a_dot = Dot(v.p(SX0, SY0, SZ0), radius=0.07, color=WHITE)
        self.a_lab = MathTex("f(P_0)", font_size=28).move_to(v.po(SX0, SY0, SZ0, -0.5, 0.24))
        self.point = VGroup(self.riser0, self.p0_dot, self.p0_lab, self.a_dot, self.a_lab)
        self.u = arrow(v.p(SX0, SY0, 0), v.p(SX0 + SU[0], SY0 + SU[1], 0), U_C, sw=5,
                       tip=0.2)
        self.ul = MathTex(r"\hat{u}", font_size=32, color=U_C).move_to(
            v.po(SX0 + 0.5 * SU[0], SY0 + 0.5 * SU[1], 0, -0.05, 0.3))

        def line_at(m, color, sw):
            """The line through (P0, f(P0)) in the plane, with slope m."""
            k = np.sqrt(1.0 + (v.zk * m) ** 2)
            t1, t2 = -1.1 / k, 3.0 / k
            return Line(v.p(*ray(t1), SZ0 + m * t1), v.p(*ray(t2), SZ0 + m * t2),
                        color=color, stroke_width=sw)

        self.tangent = line_at(G1, TAN_C, 6)
        self.static = VGroup(self.axes, self.surface, self.surf_label, self.plane,
                             self.point, self.u, self.ul)

        # the moving point P = P0 + s u, the point above it, the secant, rise and run
        def P():
            return ray(s.get_value())

        far = lambda: ramp(s.get_value(), 0.55, 0.45)
        self.disp = always_redraw(lambda: Line(v.p(SX0, SY0, 0), v.p(*P(), 0), color=DISP_C,
                                               stroke_width=7, stroke_opacity=0.8))
        self.p_dot = always_redraw(lambda: Dot(v.p(*P(), 0), radius=0.065, color=DISP_C))
        self.p_lab = follow(MathTex("P", font_size=30, color=DISP_C), v,
                            lambda: v.po(*P(), 0, 0.2, -0.24), far)
        self.riser = always_redraw(lambda: DashedLine(
            v.p(*P(), 0), v.p(*P(), g(s.get_value())), color=DISP_C, stroke_width=2,
            dash_length=0.07))
        self.b_dot = always_redraw(lambda: Dot(v.p(*P(), g(s.get_value())), radius=0.07,
                                               color=DISP_C))
        self.b_lab = follow(MathTex("f(P)", font_size=28, color=DISP_C), v,
                            lambda: v.po(*P(), g(s.get_value()), -0.45, 0.22), far)
        self.secant = always_redraw(lambda: line_at(sec_slope(s.get_value()), SEC_C, 6))
        self.run = always_redraw(lambda: Line(
            v.p(SX0, SY0, SZ0), v.p(*P(), SZ0), color=WHITE, stroke_width=3))
        self.rise = always_redraw(lambda: Line(
            v.p(*P(), SZ0), v.p(*P(), g(s.get_value())), color=WHITE, stroke_width=3))
        self.run_lab = follow(MathTex("s", font_size=30), v,
                              lambda: v.po(*ray(0.5 * s.get_value()), SZ0, 0.05, -0.2), far)
        self.rise_lab = follow(
            MathTex("f(P)-f(P_0)", font_size=28), v,
            lambda: v.po(*P(), 0.5 * (SZ0 + g(s.get_value())), 0.9, 0.0), far)
        self.triangle = [self.run, self.rise, self.run_lab, self.rise_lab]
        self.dynamic = [self.disp, self.p_dot, self.p_lab, self.riser, self.run, self.rise,
                        self.run_lab, self.rise_lab, self.secant, self.b_dot, self.b_lab]


def chain_tree(center, w=1.45, h=1.25):
    """The chain rule dependency diagram of the notes."""
    c = v3(center)
    z = MathTex("z=f(x,y)", font_size=32).move_to(c + UP * h)
    x = MathTex("x", font_size=34).move_to(c + LEFT * w)
    y = MathTex("y", font_size=34).move_to(c + RIGHT * w)
    s = MathTex("s", font_size=34).move_to(c + DOWN * h)
    a1 = arrow(z.get_bottom() + LEFT * 0.35 + DOWN * 0.08, x.get_top() + UP * 0.12 + RIGHT * 0.1,
               U_C, sw=3.5, tip=0.18)
    a2 = arrow(z.get_bottom() + RIGHT * 0.35 + DOWN * 0.08, y.get_top() + UP * 0.12 + LEFT * 0.1,
               U_C, sw=3.5, tip=0.18)
    a3 = arrow(x.get_bottom() + DOWN * 0.12 + RIGHT * 0.1, s.get_top() + UP * 0.1 + LEFT * 0.22,
               G_C, sw=3.5, tip=0.18)
    a4 = arrow(y.get_bottom() + DOWN * 0.12 + LEFT * 0.1, s.get_top() + UP * 0.1 + RIGHT * 0.22,
               G_C, sw=3.5, tip=0.18)
    l1 = MathTex(r"\dfrac{\partial f}{\partial x}", font_size=28, color=U_C).move_to(
        a1.get_center() + LEFT * 0.55 + UP * 0.12)
    l2 = MathTex(r"\dfrac{\partial f}{\partial y}", font_size=28, color=U_C).move_to(
        a2.get_center() + RIGHT * 0.55 + UP * 0.12)
    l3 = MathTex(r"\dfrac{dx}{ds}", font_size=28, color=G_C).move_to(
        a3.get_center() + LEFT * 0.55 + DOWN * 0.12)
    l4 = MathTex(r"\dfrac{dy}{ds}", font_size=28, color=G_C).move_to(
        a4.get_center() + RIGHT * 0.55 + DOWN * 0.12)
    return VGroup(z, x, y, s, a1, a2, a3, a4, l1, l2, l3, l4)


class DefinitionAndChainRule(Scene):
    def w(self, t):
        self.wait(0.15 if FAST else t)

    def construct(self):
        LEFT_X = -6.75
        ease = rate_functions.ease_in_out_sine

        def put(m, y, x=LEFT_X):
            m.move_to(np.array([x, y, 0]), aligned_edge=LEFT)
            if m.get_right()[0] > 6.95:
                print("TOO WIDE:", round(m.width, 2), "right edge", round(m.get_right()[0], 2))
            return m

        title = put(T(r"\textbf{Directional Derivatives and Gradient Vectors}",
                      fs=32, color=BLUE_B), 3.6)
        self.play(FadeIn(title), run_time=0.9)

        # =====================================================================
        # 1.  The definition
        # =====================================================================
        q = T(r"How fast does $z=f(x,y)$ change at $P_0$ in the direction of a unit vector"
              r"\quad $\hat{u}=u_1\hat{\imath}+u_2\hat{\jmath}$\ ?")
        put(q, 2.75)
        dkey = T(r"\textbf{Definition:}", color=KEY_C)
        dfn = MathTex(DU, "=", DFDS, "=", r"\lim_{s\to 0}",
                      r"\frac{f(x_0+su_1,\;y_0+su_2)-f(x_0,y_0)}{s}", font_size=34)
        put(dkey, 2.55)
        dfn.next_to(dkey, RIGHT, buff=0.4)

        if "1" in STAGES:
            s2d = ValueTracker(2.75)
            vS = View((-3.0, -3.35), 1.22)
            S = PlaneFig(vS, s2d)

            self.play(Write(q), run_time=2.8)
            self.play(LaggedStart(Create(S.axes[0]), Create(S.axes[1]), lag_ratio=0.3),
                      FadeIn(S.axes[2:]), run_time=1.1)
            self.play(FadeIn(S.point), run_time=0.8)
            self.play(GrowArrow(S.u), FadeIn(S.ul), run_time=1.0)
            self.w(1.5)

            # move a distance s along u, from P0 to P
            self.add(S.disp, S.u, S.ul, S.point)
            self.play(Create(S.disp), run_time=1.0)
            self.add(S.pdot, S.pl, S.sl)
            self.play(FadeIn(S.pdot, scale=2), FadeIn(S.pl), FadeIn(S.sl), run_time=0.9)
            self.add(S.leg_x, S.leg_y, S.lxl, S.lyl, S.disp, S.u, S.ul, S.point, S.pdot)
            self.play(Create(S.leg_x), Create(S.leg_y), FadeIn(S.lxl), FadeIn(S.lyl),
                      run_time=1.2)
            self.w(2.5)

            self.play(FadeOut(q), run_time=0.6)
            self.play(FadeIn(dkey), Write(dfn), run_time=3.2)
            self.w(1.5)
            leg = VGroup(
                T(r"$f(x_0+su_1,\;y_0+su_2)-f(x_0,y_0)$\,:", color=DISP_C, fs=28),
                T(r"the change in $f$ from $P_0$ to $P$", fs=28),
                T(r"$s$\,:", color=DISP_C, fs=28),
                T(r"the distance from $P_0$ to $P$", fs=28),
                T(r"As $s\to 0$, the point $P$ slides back to $P_0$.", fs=28, color=GREY_A),
            )
            leg[1].next_to(leg[0], DOWN, buff=0.14, aligned_edge=LEFT)
            leg[2].next_to(leg[1], DOWN, buff=0.42, aligned_edge=LEFT)
            leg[3].next_to(leg[2], RIGHT, buff=0.2)
            leg[4].next_to(leg[2], DOWN, buff=0.5, aligned_edge=LEFT)
            leg.move_to(np.array([1.0, -0.7, 0]), aligned_edge=LEFT)
            self.play(*move_fig(vS, S.static, (-6.3, -3.35), 1.12), run_time=1.3)
            self.play(FadeIn(leg[0]), FadeIn(leg[1]), run_time=0.9)
            self.play(FadeIn(leg[2]), FadeIn(leg[3]), run_time=0.9)
            self.w(1.2)
            self.play(FadeIn(leg[4]), run_time=0.7)
            self.play(s2d.animate.set_value(0.35), run_time=4.0, rate_func=ease)
            self.w(0.6)
            self.play(s2d.animate.set_value(2.75), run_time=2.2, rate_func=ease)
            self.w(1.2)
            for m in S.dynamic:
                m.clear_updaters()
            self.play(FadeOut(VGroup(S.static, *S.dynamic, leg)), run_time=0.9)
        else:
            self.add(dkey, dfn)

        # =====================================================================
        # 2.  The slope of the tangent line
        # =====================================================================
        if "2" in STAGES:
            sv = ValueTracker(S_START)
            vF = View3((3.0, -2.5), 0.9)
            F = SurfFig(vF, sv)

            h2 = T(r"\textbf{What does this number measure?}", color=KEY_C)
            put(h2, 1.45)
            self.play(FadeIn(h2), run_time=0.7)
            self.play(LaggedStart(*[Create(m) for m in F.axes[:3]], lag_ratio=0.2),
                      FadeIn(F.axes[3:]), run_time=1.2)
            self.play(FadeIn(F.surface, lag_ratio=0.004), FadeIn(F.surf_label), run_time=1.8)
            self.play(FadeIn(F.p0_dot, scale=2), FadeIn(F.p0_lab), run_time=0.7)
            self.play(GrowArrow(F.u), FadeIn(F.ul), run_time=0.8)
            n1 = VGroup(T(r"The vertical plane through $P_0$", fs=28),
                        T(r"along $\hat{u}$ cuts the surface in a curve.", fs=28))
            n1.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            put(n1, 0.55)
            n1[1][0][-6:-1].set_color(GOLD)
            self.play(FadeIn(n1), FadeIn(F.curtain), Create(F.ground_line), Create(F.trace),
                      run_time=1.6)
            self.play(Create(F.riser0), FadeIn(F.a_dot, scale=2), FadeIn(F.a_lab),
                      run_time=0.9)
            self.w(1.0)

            # P, the point above it, and the secant with its rise and run
            self.add(F.disp, F.u, F.ul, F.p0_dot)
            self.play(Create(F.disp), run_time=0.8)
            self.add(F.p_dot, F.p_lab, F.riser, F.b_dot, F.b_lab)
            self.play(FadeIn(F.p_dot, scale=2), FadeIn(F.p_lab), Create(F.riser),
                      FadeIn(F.b_dot, scale=2), FadeIn(F.b_lab), run_time=1.0)
            self.add(F.secant, F.a_dot, F.b_dot)
            self.play(Create(F.secant), run_time=1.0)
            sec = MathTex(r"\text{slope of the secant line}", "=", r"\frac{f(P)-f(P_0)}{s}",
                          font_size=32)
            sec[0].set_color(SEC_C)
            put(sec, -0.75)
            self.add(*F.triangle, F.secant, F.a_dot, F.b_dot)
            self.play(Create(F.run), Create(F.rise), FadeIn(F.run_lab), FadeIn(F.rise_lab),
                      Write(sec), run_time=1.8)
            rr = T(r"(rise over run, in that plane)", fs=26, color=GREY_A)
            rr.next_to(sec, DOWN, buff=0.2, aligned_edge=LEFT)
            self.play(FadeIn(rr), run_time=0.6)
            self.w(2.5)

            # s -> 0
            lim_note = T(r"As $s\to 0$, the secant line turns onto the tangent line.", fs=28)
            put(lim_note, -2.1)
            lim_note[0][-12:-1].set_color(TAN_C)
            self.add(F.tangent, F.secant, F.a_dot, F.b_dot)
            self.play(FadeIn(lim_note), Create(F.tangent), run_time=1.2)
            self.play(sv.animate.set_value(S_END), run_time=9, rate_func=ease)
            self.w(1.0)
            tanl = MathTex(r"\text{slope of the tangent line}", "=", r"\lim_{s\to 0}",
                           r"\frac{f(P)-f(P_0)}{s}", "=", DU, font_size=32)
            tanl[0].set_color(TAN_C)
            put(tanl, -3.2)
            self.play(Write(tanl), run_time=2.4)
            self.w(1.5)
            # once more, so that the turning is seen with the conclusion on screen
            self.play(sv.animate.set_value(1.7), run_time=2.2, rate_func=ease)
            self.play(sv.animate.set_value(S_END), run_time=5.5, rate_func=ease)
            self.w(1.5)

            # statement (I)
            for m in F.dynamic:
                m.clear_updaters()
            I3 = VGroup(
                T(r"\textbf{(I)}", r"\quad $" + DU + r"$ gives the rate of change"),
                T(r"of $z=f(x,y)$ in the direction $\hat{u}$"),
                T(r"at the point $P_0$."),
            )
            I3[0][0].set_color(KEY_C)
            put(I3[0], 0.6)
            I3[1].next_to(I3[0][1], DOWN, buff=0.2, aligned_edge=LEFT).shift(RIGHT * 0.28)
            I3[2].next_to(I3[1], DOWN, buff=0.2, aligned_edge=LEFT)
            self.play(FadeOut(VGroup(h2, n1, sec, rr, lim_note, *F.dynamic)), run_time=0.8)
            self.play(Write(I3), run_time=3.0)
            self.w(4.5)
            self.play(FadeOut(VGroup(F.static, F.tangent, I3, tanl)), run_time=1.0)

        # =====================================================================
        # 3.  The line through P0 and P, and its parametrization
        # =====================================================================
        px = MathTex("x(s)=x_0+su_1", font_size=36)
        py = MathTex("y(s)=y_0+su_2", font_size=36)
        if "3" in STAGES:
            s3 = ValueTracker(2.75)
            vL = View((-5.45, -3.3), 1.0)
            L = PlaneFig(vL, s3, with_line=True)

            h3 = T(r"\textbf{How do we compute it without the limit?}", color=KEY_C)
            put(h3, 1.6)
            self.play(FadeIn(h3), run_time=0.8)
            self.play(LaggedStart(Create(L.axes[0]), Create(L.axes[1]), lag_ratio=0.3),
                      FadeIn(L.axes[2:]), run_time=1.0)
            self.play(FadeIn(L.point), FadeIn(L.coords0), GrowArrow(L.u), FadeIn(L.ul),
                      run_time=1.2)
            self.add(L.disp, L.u, L.ul, L.point)
            self.play(Create(L.disp), run_time=0.8)
            self.add(L.pdot, L.pl, L.sl)
            self.play(FadeIn(L.pdot, scale=2), FadeIn(L.pl), FadeIn(L.sl), run_time=0.8)
            self.w(0.8)

            # the line through the two points
            RX = 1.25
            t1 = T(r"Draw the line through $P_0$ and $P$.", fs=29)
            put(t1, 1.0, RX)
            self.add(L.line, L.disp, L.u, L.ul, L.point, L.pdot)
            self.play(FadeIn(t1), Create(L.line), run_time=1.6)
            self.w(1.2)

            # the coordinates of P
            t2 = VGroup(T(r"From $P_0$, the point $P$ is $s\hat{u}$ away:", fs=29),
                        T(r"$su_1$ in $x$, and $su_2$ in $y$.", fs=29))
            t2.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            put(t2, 0.25, RX)
            self.add(L.leg_x, L.leg_y, L.lxl, L.lyl, L.drop_x, L.drop_y, L.px, L.py,
                     L.disp, L.u, L.ul, L.point, L.pdot)
            self.play(FadeIn(t2), Create(L.leg_x), Create(L.leg_y), FadeIn(L.lxl),
                      FadeIn(L.lyl), run_time=1.5)
            self.play(Create(L.drop_x), Create(L.drop_y), FadeIn(L.px), FadeIn(L.py),
                      run_time=1.2)
            self.w(1.0)
            par = VGroup(px, py).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            put(par, -0.95, RX + 0.5)
            pbox = SurroundingRectangle(par, color=DISP_C, buff=0.2, stroke_width=2.5,
                                        corner_radius=0.1)
            self.play(Write(px), run_time=1.3)
            self.play(Write(py), run_time=1.3)
            self.play(Create(pbox), run_time=0.7)
            self.w(1.5)

            # every point of the line comes from one value of s
            sread = MathTex("s", "=", font_size=34, color=DISP_C)
            sread.next_to(pbox, RIGHT, buff=0.55)
            snum = DecimalNumber(s3.get_value(), num_decimal_places=2, font_size=34,
                                 color=DISP_C)
            snum.next_to(sread, RIGHT, buff=0.15)
            snum.add_updater(lambda m: m.set_value(s3.get_value())
                             .next_to(sread, RIGHT, buff=0.15))
            t3 = VGroup(T(r"As $s$ changes, $P$ runs along the line;", fs=29),
                        T(r"$s=0$ gives $P_0$.", fs=29))
            t3.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            put(t3, -2.15, RX)
            self.play(FadeIn(t3), FadeIn(sread), FadeIn(snum), run_time=0.9)
            self.play(s3.animate.set_value(0.0), run_time=4.5, rate_func=ease)
            self.w(0.8)
            self.play(s3.animate.set_value(3.6), run_time=4.5, rate_func=ease)
            self.play(s3.animate.set_value(2.75), run_time=1.6, rate_func=ease)
            self.w(1.0)
            t4 = VGroup(T(r"So on this line, $f$ depends on $s$ alone:", fs=29),
                        MathTex(r"z=f\big(x(s),\,y(s)\big)", font_size=34))
            t4.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            t4[1].shift(RIGHT * 0.5)
            put(t4, -3.2, RX)
            self.play(Write(t4), run_time=2.2)
            self.w(3.5)

            snum.clear_updaters()
            for m in L.dynamic:
                m.clear_updaters()
            self.play(FadeOut(VGroup(L.static, *L.dynamic, h3, t1, t2, t3, t4, pbox, sread,
                                     snum, dkey, dfn)), run_time=1.0)
        else:
            self.remove(dkey, dfn)

        # =====================================================================
        # 4.  The chain rule
        # =====================================================================
        II = MathTex(r"\textbf{(II)}", r"\quad", DU, "=", GR, r"\cdot", r"\hat{u}",
                     font_size=40)
        if "4" in STAGES:
            h4 = T(r"\textbf{The chain rule}", r" along the line", color=KEY_C)
            h4[1].set_color(WHITE)
            put(h4, 2.85)
            comma = MathTex(",", font_size=36).set_opacity(0)
            target = VGroup(px.copy(), comma, py.copy()).arrange(RIGHT, buff=0.12)
            target[2].shift(RIGHT * 0.45)
            target.next_to(h4, RIGHT, buff=0.35)
            if "3" in STAGES:
                self.play(FadeIn(h4), px.animate.move_to(target[0]),
                          py.animate.move_to(target[2]), FadeIn(target[1]), run_time=1.4)
            else:
                px.move_to(target[0])
                py.move_to(target[2])
                self.play(FadeIn(h4), FadeIn(px), FadeIn(py), FadeIn(target[1]), run_time=1.0)
            comma = target[1]
            zs = T(r"$z=f\big(x(s),\,y(s)\big)$ is a function of $s$, and its derivative at "
                   r"$s=0$ is the limit in the definition.", fs=29)
            put(zs, 2.05)
            self.play(Write(zs), run_time=2.8)
            self.w(1.5)

            tree = chain_tree((5.0, -0.25))
            d1 = MathTex(DU, "=", DFDS, "=", FX0 + r"\frac{dx}{ds}+" + FY0 + r"\frac{dy}{ds}",
                         font_size=32)
            d2 = MathTex("=", FX0 + r"\,u_1+" + FY0 + r"\,u_2", font_size=32)
            d3 = MathTex("=", r"\left[" + FX0 + r"\hat{\imath}+" + FY0 + r"\hat{\jmath}\right]",
                         r"\cdot", r"\left(u_1\hat{\imath}+u_2\hat{\jmath}\right)",
                         font_size=32)
            put(d1, 0.55)
            for d, y in ((d2, -0.65), (d3, -1.85)):
                d.shift(d1[3].get_center() - d[0].get_center())
                d.set_y(y)
            n2 = MathTex(r"\left(\dfrac{dx}{ds}=u_1,\quad \dfrac{dy}{ds}=u_2\right)",
                         font_size=26, color=GREY_A)
            n2.next_to(d2[0], LEFT, buff=0.45)

            self.play(FadeIn(tree[:4]), run_time=0.8)
            self.play(LaggedStart(*[GrowArrow(a) for a in tree[4:8]], lag_ratio=0.25),
                      FadeIn(tree[8:]), run_time=1.8)
            self.w(1.5)
            self.play(Write(d1), run_time=3.0)
            self.w(2.0)
            self.play(Indicate(px, color=G_C, scale_factor=1.08),
                      Indicate(py, color=G_C, scale_factor=1.08), FadeIn(n2), run_time=1.4)
            self.play(Write(d2), run_time=2.0)
            self.w(2.0)
            self.play(Write(d3), run_time=2.6)
            self.w(1.2)
            b1 = Brace(d3[1], DOWN, buff=0.1, color=G_C)
            b1l = MathTex(GR, r"\ \text{: the gradient vector}", font_size=30, color=G_C)
            b1l.next_to(b1, DOWN, buff=0.12)
            b2 = Brace(d3[3], DOWN, buff=0.1, color=U_C)
            b2l = MathTex(r"\hat{u}", font_size=32, color=U_C).next_to(b2, DOWN, buff=0.12)
            self.play(GrowFromCenter(b1), FadeIn(b1l), run_time=1.1)
            self.play(GrowFromCenter(b2), FadeIn(b2l), run_time=0.9)
            self.w(3.0)

            # the gradient vector and statement (II)
            self.play(FadeOut(VGroup(h4, px, py, comma, zs, tree, d1, d2, d3, n2, b1, b1l,
                                     b2, b2l)), run_time=1.0)
            gk = T(r"\textbf{Gradient vector:}", color=KEY_C)
            gd = MathTex(GR, "=", FX0 + r"\hat{\imath}+" + FY0 + r"\hat{\jmath}", font_size=36)
            gd[0].set_color(G_C)
            grp = VGroup(gk, gd).arrange(RIGHT, buff=0.4).move_to(np.array([0, 1.6, 0]))
            II[0].set_color(KEY_C)
            II[4].set_color(G_C)
            II[6].set_color(U_C)
            II.move_to(np.array([0, -0.3, 0]))
            box = SurroundingRectangle(II, color=KEY_C, buff=0.22, stroke_width=2.5,
                                       corner_radius=0.1)
            note = T(r"One dot product, and no limit to compute.", fs=28, color=GREY_A)
            note.next_to(box, DOWN, buff=0.4)
            self.play(FadeIn(gk), Write(gd), run_time=2.2)
            self.w(1.2)
            self.play(Write(II), run_time=1.8)
            self.play(Create(box), FadeIn(note), run_time=0.9)
            self.w(4.0)
            self.play(FadeOut(VGroup(grp, II, box, note)), run_time=1.0)
        else:
            self.remove(px, py)

        # =====================================================================
        # 5.  Summary
        # =====================================================================
        if "5" in STAGES:
            sk = T(r"\textbf{Summary}", fs=34, color=KEY_C)
            put(sk, 2.3)
            sI = T(r"\textbf{(I)}", r"\quad $" + DU + r"$ gives the rate of change of "
                   r"$z=f(x,y)$ in the direction $\hat{u}$ at the point $P_0$.", fs=31)
            sI[0].set_color(KEY_C)
            sII = T(r"\textbf{(II)}", r"\quad $" + DU + "=" + GR + r"\cdot\hat{u}$,"
                    r"\quad where\quad $\nabla f=\dfrac{\partial f}{\partial x}\,\hat{\imath}"
                    r"+\dfrac{\partial f}{\partial y}\,\hat{\jmath}$,\quad "
                    r"$\hat{u}=u_1\hat{\imath}+u_2\hat{\jmath}$\quad and\quad $u_1^2+u_2^2=1$.",
                    fs=31)
            sII[0].set_color(KEY_C)
            put(sI, 1.1)
            put(sII, -0.3)
            self.play(FadeIn(sk), run_time=0.6)
            self.play(Write(sI), run_time=3.0)
            self.w(1.5)
            self.play(Write(sII), run_time=3.5)
            self.w(8.0)
