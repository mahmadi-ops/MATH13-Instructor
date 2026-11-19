r"""The partial derivatives are directional derivatives: taking u = i in the limit
definition gives f_x, and u = j gives f_y.

    manim -qh --fps 30 --disable_caching -o dirderiv_partials \
        dirderiv_partials.py PartialsAreDirectional

It follows the activity "The Partial Derivatives Are Directional Derivatives"
of the course notes.  The xy-plane (left) and the surface z = f(x,y) (right)
are driven by the same two numbers: the direction of u, and the distance s
from P0 to P.

1. The definition, with a general unit vector u in both pictures (briefly).
2. u turns onto i: u1 = 1, u2 = 0.  The point P is (x0 + s, y0), so only x
   changes and y is held at y0; the vertical plane is y = y0.  The limit left
   in the definition is the one that defines f_x(x0, y0), and as s -> 0 the
   secant turns onto the tangent line of the curve cut by that plane.
3. u turns onto j: u1 = 0, u2 = 1.  P is (x0, y0 + s), the plane is x = x0, and
   the limit is f_y(x0, y0).
4. Both results, with both tangent lines on the surface.

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

# ---- the construction -------------------------------------------------------
X0, Y0 = 0.7, 0.8                       # P0 (shown only as (x0, y0))
R_S = 3.3                               # the surface is drawn over the quarter disc r <= R_S
ANG0 = 40 * DEGREES                     # the general direction shown first
S_START, S_END = 1.5, 0.03


def f(x, y):
    """The surface: a prop for a general f, never named on screen."""
    return 3.2 - 0.28 * (x * x + y * y)


Z0 = f(X0, Y0)


def g(ang, t):
    """f along the line through P0 in the direction at angle ang."""
    return f(X0 + t * np.cos(ang), Y0 + t * np.sin(ang))


def tan_slope(ang):
    return -0.56 * (X0 * np.cos(ang) + Y0 * np.sin(ang))


def sec_slope(ang, s):
    return (g(ang, s) - Z0) / s


# ---- colours ----------------------------------------------------------------
U_C = "#5CC8FF"                         # the unit vector u
TAN_C = "#7CE07C"                       # tangent line
SEC_C = "#FF6060"                       # secant line
DISP_C = GOLD                           # the displacement s u and the point P
KEY_C = YELLOW

FAST = os.environ.get("FAST", "") == "1"


def v3(c):
    c = np.array(c, dtype=float)
    return c if c.size == 3 else np.array([c[0], c[1], 0.0])


class View:
    """Maps figure coordinates to the screen: centre + scale * proj(x, y, z)."""

    def __init__(self, center, scale):
        self.center = v3(center)
        self.scale = float(scale)

    def proj(self, x, y, z=0.0):
        return np.array([x, y, 0.0])

    def p(self, x, y, z=0.0):
        return self.center + self.scale * self.proj(x, y, z)

    def po(self, x, y, z, dx, dy):
        """The point (x, y, z), then moved by (dx, dy) on screen (figure units)."""
        return self.p(x, y, z) + self.scale * np.array([dx, dy, 0.0])


class View3(View):
    """Orthographic camera at azimuth az, elevation el; z drawn at zk."""

    def __init__(self, center, scale, az=35 * DEGREES, el=20 * DEGREES, zk=0.8):
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


def arrow(a, b, color, sw=6, tip=0.26):
    return Arrow(a, b, buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=0.6,
                 max_stroke_width_to_length_ratio=30)


def ramp(x, lo, width):
    return float(np.clip((x - lo) / width, 0.0, 1.0))


def T(*parts, fs=30, **kw):
    return Tex(*parts, font_size=fs, **kw)


def variants(texs, mode, pos_fns, extra_opacity=None):
    """Labels of which the one numbered mode.get_value() is shown; they cross-fade
    as mode moves from one whole number to the next, and each follows its own
    position function."""
    out = []
    for k, (tex, pos) in enumerate(zip(texs, pos_fns)):
        base = tex.copy()

        def upd(m, k=k, base=base, pos=pos):
            op = max(0.0, 1.0 - abs(mode.get_value() - k))
            if extra_opacity is not None:
                op *= extra_opacity()
            m.become(base.copy().move_to(pos()).set_opacity(op))

        upd(tex)
        tex.add_updater(upd)
        out.append(tex)
    return out


# ---- the xy-plane ------------------------------------------------------------
class PlaneFig:
    def __init__(self, view, ang, s, mode):
        v = view
        A = ang.get_value
        S = s.get_value

        ax_x = arrow(v.p(-0.2, 0), v.p(2.95, 0), GREY_A, sw=3, tip=0.2)
        ax_y = arrow(v.p(0, -0.2), v.p(0, 2.9), GREY_A, sw=3, tip=0.2)
        lx = MathTex("x", font_size=32).move_to(v.p(3.12, -0.02))
        ly = MathTex("y", font_size=32).move_to(v.p(-0.02, 3.08))
        self.axes = VGroup(ax_x, ax_y, lx, ly)

        d0x = DashedLine(v.p(X0, 0), v.p(X0, Y0), color=GREY_B, stroke_width=2,
                         dash_length=0.08)
        d0y = DashedLine(v.p(0, Y0), v.p(X0, Y0), color=GREY_B, stroke_width=2,
                         dash_length=0.08)
        t0x = MathTex("x_0", font_size=28).move_to(v.p(X0, -0.26))
        t0y = MathTex("y_0", font_size=28).move_to(v.p(-0.28, Y0))
        dot = Dot(v.p(X0, Y0), radius=0.07, color=WHITE)
        tp = MathTex("P_0", font_size=30).move_to(v.p(X0 - 0.26, Y0 - 0.22))
        self.point = VGroup(d0x, d0y, t0x, t0y, dot, tp)
        self.static = VGroup(self.axes, self.point)

        def P():
            return X0 + S() * np.cos(A()), Y0 + S() * np.sin(A())

        def unit(k=1.0):
            return X0 + k * np.cos(A()), Y0 + k * np.sin(A())

        self.u = always_redraw(lambda: arrow(v.p(X0, Y0), v.p(*unit()), U_C, sw=6, tip=0.22))
        self.disp = always_redraw(lambda: Line(v.p(X0, Y0), v.p(*P()), color=DISP_C,
                                               stroke_width=8, stroke_opacity=0.75))
        self.pdot = always_redraw(lambda: Dot(v.p(*P()), radius=0.07, color=DISP_C))
        self.drop_x = always_redraw(lambda: DashedLine(
            v.p(P()[0], 0), v.p(*P()), color=DISP_C, stroke_width=2, stroke_opacity=0.8,
            dash_length=0.08))
        self.drop_y = always_redraw(lambda: DashedLine(
            v.p(0, P()[1]), v.p(*P()), color=DISP_C, stroke_width=2, stroke_opacity=0.8,
            dash_length=0.08))

        def n_off(k, d):
            """A point k along u from P0, then d to the left of u."""
            a = A()
            return v.p(X0 + k * np.cos(a) - d * np.sin(a), Y0 + k * np.sin(a) + d * np.cos(a))

        self.ul = variants(
            [MathTex(r"\hat{u}", font_size=34, color=U_C),
             MathTex(r"\hat{u}=\hat{\imath}", font_size=32, color=U_C),
             MathTex(r"\hat{u}=\hat{\jmath}", font_size=32, color=U_C)],
            mode,
            [lambda: n_off(0.5, 0.3), lambda: n_off(0.5, 0.3), lambda: n_off(0.5, -0.55)])
        far = lambda: ramp(S(), 0.55, 0.4)
        self.pl = variants(
            [MathTex(r"P(x_0+su_1,\;y_0+su_2)", font_size=28, color=DISP_C),
             MathTex(r"P(x_0+s,\;y_0)", font_size=28, color=DISP_C),
             MathTex(r"P(x_0,\;y_0+s)", font_size=28, color=DISP_C)],
            mode,
            [lambda: v.p(P()[0] + 0.1, P()[1] + 0.3), lambda: v.p(P()[0] + 0.25, P()[1] + 0.3),
             lambda: v.p(P()[0] + 0.85, P()[1] + 0.05)], far)
        self.px = variants(
            [MathTex("x_0+su_1", font_size=26, color=DISP_C),
             MathTex("x_0+s", font_size=26, color=DISP_C),
             MathTex("", font_size=26)],
            mode, [lambda: v.p(P()[0] + 0.1, -0.26)] * 3, lambda: ramp(S(), 0.9, 0.4))
        self.py = variants(
            [MathTex("y_0+su_2", font_size=26, color=DISP_C),
             MathTex("", font_size=26),
             MathTex("y_0+s", font_size=26, color=DISP_C)],
            mode, [lambda: v.p(-0.6, P()[1] + 0.02), lambda: v.p(-0.6, P()[1]),
                   lambda: v.p(-0.48, P()[1])], lambda: ramp(S(), 0.65, 0.4))
        self.dynamic = [self.drop_x, self.drop_y, self.disp, self.u, self.pdot,
                        *self.ul, *self.pl, *self.px, *self.py]

        # the lines y = y0 and x = x0, shown when one variable is held fixed
        self.hold_y = VGroup(
            Line(v.p(0, Y0), v.p(2.85, Y0), color=TAN_C, stroke_width=2.5, stroke_opacity=0.9),
            MathTex("y=y_0", font_size=28, color=TAN_C).move_to(v.p(2.45, Y0 - 0.24)))
        self.hold_x = VGroup(
            Line(v.p(X0, 0), v.p(X0, 2.8), color=TAN_C, stroke_width=2.5, stroke_opacity=0.9),
            MathTex("x=x_0", font_size=28, color=TAN_C).move_to(v.p(X0 + 0.5, 2.65)))
        self.i_arrow = VGroup(
            arrow(v.p(X0, Y0), v.p(X0 + 1, Y0), U_C, sw=6, tip=0.22),
            MathTex(r"\hat{\imath}", font_size=34, color=U_C).move_to(v.p(X0 + 0.55, Y0 - 0.28)))
        self.j_arrow = VGroup(
            arrow(v.p(X0, Y0), v.p(X0, Y0 + 1), U_C, sw=6, tip=0.22),
            MathTex(r"\hat{\jmath}", font_size=34, color=U_C).move_to(v.p(X0 + 0.28, Y0 + 0.6)))


# ---- the surface --------------------------------------------------------------
class SurfFig:
    def __init__(self, view, ang, s, mode):
        v = view
        self.view = v
        A = ang.get_value
        S = s.get_value

        ax_x = arrow(v.p(0, 0, 0), v.p(3.9, 0, 0), GREY_A, sw=2.5, tip=0.18)
        ax_y = arrow(v.p(0, 0, 0), v.p(0, 3.9, 0), GREY_A, sw=2.5, tip=0.18)
        ax_z = arrow(v.p(0, 0, 0), v.p(0, 0, 3.75), GREY_A, sw=2.5, tip=0.18)
        lx = MathTex("x", font_size=32).move_to(v.po(3.9, 0, 0, -0.18, -0.12))
        ly = MathTex("y", font_size=32).move_to(v.po(0, 3.9, 0, 0.2, -0.05))
        lz = MathTex("z", font_size=32).move_to(v.po(0, 0, 3.75, -0.2, 0.05))
        self.axes = VGroup(ax_x, ax_y, ax_z, lx, ly, lz)

        # the surface over the quarter disc: depth-sorted quads of a polar mesh
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
                q = [(rs[i], ts[j]), (rs[i + 1], ts[j]), (rs[i + 1], ts[j + 1]),
                     (rs[i], ts[j + 1])]
                pts3 = [(r * np.cos(t), r * np.sin(t), f(r * np.cos(t), r * np.sin(t)))
                        for r, t in q]
                if i == 0:
                    pts3 = pts3[1:]
                c = np.mean(pts3, axis=0)
                n = np.array([0.56 * c[0] * v.zk, 0.56 * c[1] * v.zk, 1.0])
                n /= np.linalg.norm(n)
                b = 0.2 + 0.8 * max(0.0, float(n @ light))
                poly = Polygon(*[v.p(*pt) for pt in pts3],
                               stroke_color=WHITE, stroke_width=0.6, stroke_opacity=0.22,
                               fill_color=interpolate_color(ManimColor("#123A66"),
                                                            ManimColor("#4FA3E8"), b),
                               fill_opacity=0.42)
                faces.append((v.depth(c[0], c[1], c[2]), poly))
        faces.sort(key=lambda t: t[0])
        self.surface = VGroup(*[poly for _, poly in faces])
        edge = [(r, 0.0) for r in rs] + [(R_S, t) for t in ts] + [(r, PI / 2) for r in rs[::-1]]
        rim = VMobject(color=BLUE_B, stroke_width=1.5, stroke_opacity=0.7)
        rim.set_points_as_corners([v.p(r * np.cos(t), r * np.sin(t),
                                       f(r * np.cos(t), r * np.sin(t))) for r, t in edge])
        self.surface.add(rim)
        self.surf_label = MathTex("z = f(x,y)", font_size=32, color=BLUE_B)
        self.surf_label.move_to(v.po(0, 0, 3.2, 1.75, 0.3))

        # P0 on the ground and the point of the surface above it
        self.p0_dot = Dot(v.p(X0, Y0, 0), radius=0.06, color=WHITE)
        self.p0_lab = MathTex("P_0", font_size=28).move_to(v.po(X0, Y0, 0, 0.36, 0.12))
        self.riser0 = DashedLine(v.p(X0, Y0, 0), v.p(X0, Y0, Z0), color=WHITE,
                                 stroke_width=2, dash_length=0.07)
        self.a_dot = Dot(v.p(X0, Y0, Z0), radius=0.065, color=WHITE)
        self.point = VGroup(self.riser0, self.p0_dot, self.p0_lab)
        self.static = VGroup(self.axes, self.surface, self.surf_label, self.point)

        def ray(t):
            return X0 + t * np.cos(A()), Y0 + t * np.sin(A())

        def t_range():
            c, sn = np.cos(A()), np.sin(A())
            b = X0 * c + Y0 * sn
            rt = np.sqrt(b * b + R_S ** 2 - X0 ** 2 - Y0 ** 2)
            lo = -b - rt
            if c > 1e-6:
                lo = max(lo, -X0 / c)
            if sn > 1e-6:
                lo = max(lo, -Y0 / sn)
            return lo, -b + rt

        def trace_pts():
            lo, hi = t_range()
            return [v.p(*ray(t), g(A(), t)) for t in np.linspace(lo, hi, 44)]

        def mk_curtain():
            lo, hi = t_range()
            pts = trace_pts()
            m = VMobject(fill_color=GOLD, fill_opacity=0.16, stroke_color=GOLD,
                         stroke_width=1.5, stroke_opacity=0.55)
            m.set_points_as_corners([v.p(*ray(lo), 0)] + pts + [v.p(*ray(hi), 0),
                                                                 v.p(*ray(lo), 0)])
            return m

        def mk_trace():
            m = VMobject(color=GOLD, stroke_width=4.5)
            m.set_points_as_corners(trace_pts())
            return m

        def line_at(m, color, sw):
            k = np.sqrt(1.0 + (v.zk * m) ** 2)
            t1, t2 = -1.0 / k, 2.2 / k
            return Line(v.p(*ray(t1), Z0 + m * t1), v.p(*ray(t2), Z0 + m * t2),
                        color=color, stroke_width=sw)

        self.curtain = always_redraw(mk_curtain)
        self.trace = always_redraw(mk_trace)
        self.u = always_redraw(lambda: arrow(v.p(X0, Y0, 0), v.p(*ray(1.0), 0), U_C, sw=5,
                                             tip=0.2))
        self.disp = always_redraw(lambda: Line(v.p(X0, Y0, 0), v.p(*ray(S()), 0),
                                               color=DISP_C, stroke_width=7,
                                               stroke_opacity=0.8))
        self.p_dot = always_redraw(lambda: Dot(v.p(*ray(S()), 0), radius=0.06, color=DISP_C))
        self.riser = always_redraw(lambda: DashedLine(
            v.p(*ray(S()), 0), v.p(*ray(S()), g(A(), S())), color=DISP_C, stroke_width=2,
            dash_length=0.07))
        self.b_dot = always_redraw(lambda: Dot(v.p(*ray(S()), g(A(), S())), radius=0.065,
                                               color=DISP_C))
        self.secant = always_redraw(lambda: line_at(sec_slope(A(), S()), SEC_C, 5.5))
        self.tangent = always_redraw(lambda: line_at(tan_slope(A()), TAN_C, 5.5))
        self.top = always_redraw(lambda: Dot(v.p(X0, Y0, Z0), radius=0.065, color=WHITE))
        far = lambda: ramp(S(), 0.55, 0.4)
        self.p_lab = variants([MathTex("P", font_size=28, color=DISP_C)], ValueTracker(0),
                              [lambda: v.po(*ray(S()), 0, 0.02, -0.26)], far)
        self.ul = variants(
            [MathTex(r"\hat{u}", font_size=30, color=U_C),
             MathTex(r"\hat{\imath}", font_size=32, color=U_C),
             MathTex(r"\hat{\jmath}", font_size=32, color=U_C)],
            mode, [lambda: v.po(*ray(0.6), 0, 0.0, -0.28)] * 3)
        self.dynamic = [self.curtain, self.trace, self.disp, self.u, self.p_dot, self.riser,
                        self.tangent, self.secant, self.top, self.b_dot, *self.p_lab,
                        *self.ul]

        # labels of the two special planes
        self.plane_y = MathTex(r"\text{plane } y=y_0", font_size=28, color=GOLD)
        self.plane_y.move_to(v.po(2.2, Y0, f(2.2, Y0), -1.95, 0.1))
        self.plane_x = MathTex(r"\text{plane } x=x_0", font_size=28, color=GOLD)
        self.plane_x.move_to(v.po(X0, 2.2, f(X0, 2.2), 1.5, 0.1))

        # for the closing picture: both curves and both tangent lines at once
        def static_pair(a):
            c, sn = np.cos(a), np.sin(a)
            b = X0 * c + Y0 * sn
            hi = -b + np.sqrt(b * b + R_S ** 2 - X0 ** 2 - Y0 ** 2)
            lo = -X0 / c if c > 0.5 else -Y0 / sn
            tr = VMobject(color=GOLD, stroke_width=4)
            tr.set_points_as_corners([v.p(X0 + t * c, Y0 + t * sn, f(X0 + t * c, Y0 + t * sn))
                                      for t in np.linspace(lo, hi, 40)])
            m = tan_slope(a)
            k = np.sqrt(1.0 + (v.zk * m) ** 2)
            t1, t2 = -1.0 / k, 2.2 / k
            tl = Line(v.p(X0 + t1 * c, Y0 + t1 * sn, Z0 + m * t1),
                      v.p(X0 + t2 * c, Y0 + t2 * sn, Z0 + m * t2), color=TAN_C,
                      stroke_width=5.5)
            ar = arrow(v.p(X0, Y0, 0), v.p(X0 + c, Y0 + sn, 0), U_C, sw=5, tip=0.2)
            return VGroup(tr, tl, ar)

        self.pair_i = static_pair(0.0)
        self.pair_j = static_pair(PI / 2)
        self.lab_i = MathTex(r"\hat{\imath}", font_size=32, color=U_C).move_to(
            v.po(X0 + 0.6, Y0, 0, 0.0, -0.28))
        self.lab_j = MathTex(r"\hat{\jmath}", font_size=32, color=U_C).move_to(
            v.po(X0, Y0 + 0.6, 0, 0.0, -0.28))
        self.slope_x = MathTex(r"\text{slope } f_x(x_0,y_0)", font_size=28, color=TAN_C)
        self.slope_x.move_to(v.po(X0 + 2.0, Y0, Z0 + 2.0 * tan_slope(0.0), -1.75, -0.12))
        self.slope_y = MathTex(r"\text{slope } f_y(x_0,y_0)", font_size=28, color=TAN_C)
        self.slope_y.move_to(v.po(X0, Y0 + 2.0, Z0 + 2.0 * tan_slope(PI / 2), 1.5, 0.05))


DU = r"\left(D_{\hat{u}} f\right)_{P_0}"
DI = r"\left(D_{\hat{\imath}} f\right)_{P_0}"
DJ = r"\left(D_{\hat{\jmath}} f\right)_{P_0}"


class PartialsAreDirectional(Scene):
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

        title = put(T(r"\textbf{The Partial Derivatives Are Directional Derivatives}",
                      fs=32, color=BLUE_B), 3.62)
        dkey = T(r"\textbf{Definition:}", color=KEY_C)
        dfn = MathTex(DU, "=", r"\lim_{s\to 0}",
                      r"\frac{f(x_0+su_1,\;y_0+su_2)-f(x_0,y_0)}{s}", r",\qquad",
                      r"\hat{u}=u_1\hat{\imath}+u_2\hat{\jmath}", font_size=32)
        put(dkey, 2.8)
        dfn.next_to(dkey, RIGHT, buff=0.4)
        self.play(FadeIn(title), run_time=0.9)
        self.play(FadeIn(dkey), Write(dfn), run_time=2.4)
        self.w(0.5)

        # ---- the two pictures, for a general direction -----------------------
        ang = ValueTracker(ANG0)
        s = ValueTracker(S_START)
        mode = ValueTracker(0.0)
        vP = View((-5.7, -1.4), 1.15)
        vS = View3((2.55, -0.55), 0.85)
        Pl = PlaneFig(vP, ang, s, mode)
        Sf = SurfFig(vS, ang, s, mode)

        self.play(LaggedStart(Create(Pl.axes[0]), Create(Pl.axes[1]), lag_ratio=0.3),
                  FadeIn(Pl.axes[2:]),
                  LaggedStart(*[Create(m) for m in Sf.axes[:3]], lag_ratio=0.2),
                  FadeIn(Sf.axes[3:]), run_time=1.0)
        self.play(FadeIn(Pl.point), FadeIn(Sf.surface, lag_ratio=0.004),
                  FadeIn(Sf.surf_label), FadeIn(Sf.point), run_time=1.3)
        self.add(*Pl.dynamic, *Sf.dynamic)
        self.play(*[FadeIn(m) for m in Pl.dynamic + Sf.dynamic], run_time=1.0)
        self.w(1.5)

        # ---- one coordinate direction ----------------------------------------
        def case(k, target, case_tex, d_tex, sub_tex, simp_tex, res_tex, note_parts,
                 hold, plane_lab):
            cl = MathTex(*case_tex, font_size=32)
            cl[0].set_color(KEY_C)
            put(cl, -2.05)
            self.play(FadeIn(cl), ang.animate.set_value(target), mode.animate.set_value(k),
                      run_time=2.0, rate_func=ease)
            self.play(FadeIn(hold), FadeIn(plane_lab), run_time=0.7)

            row = MathTex(d_tex, "=", r"\lim_{s\to 0}", sub_tex, font_size=32)
            put(row, -2.85)
            self.play(Write(row), run_time=2.0)
            self.w(1.2)
            simp = MathTex(simp_tex, font_size=32)
            simp.move_to(row[3], aligned_edge=LEFT)
            self.play(ReplacementTransform(row[3], simp), run_time=1.1)
            note = MathTex(*note_parts, font_size=30, color=GREY_A)
            put(note, -3.65)
            # s -> 0 in both pictures
            self.play(FadeIn(note), s.animate.set_value(S_END), run_time=4.5, rate_func=ease)
            res = MathTex("=", res_tex, font_size=34)
            res.next_to(simp, RIGHT, buff=0.25)
            res[1].set_color(TAN_C)
            box = SurroundingRectangle(VGroup(row[0], res), color=TAN_C, buff=0.14,
                                       stroke_width=2, corner_radius=0.08)
            self.play(Write(res), Create(box), run_time=1.2)
            self.w(2.5)
            self.play(s.animate.set_value(S_START),
                      FadeOut(VGroup(cl, row[:3], simp, res, box, note, hold, plane_lab)),
                      run_time=1.2, rate_func=ease)

        case(1, 0.0,
             [r"\textbf{Take }\hat{u}=\hat{\imath}:", r"\qquad u_1=1,\quad u_2=0."],
             DI,
             r"\frac{f(x_0+s\cdot 1,\;y_0+s\cdot 0)-f(x_0,y_0)}{s}",
             r"\frac{f(x_0+s,\;y_0)-f(x_0,y_0)}{s}",
             r"f_x(x_0,y_0)",
             [r"\text{Only } x \text{ changes; } y \text{ is held at } y_0."],
             Pl.hold_y, Sf.plane_y)

        case(2, PI / 2,
             [r"\textbf{Take }\hat{u}=\hat{\jmath}:", r"\qquad u_1=0,\quad u_2=1."],
             DJ,
             r"\frac{f(x_0+s\cdot 0,\;y_0+s\cdot 1)-f(x_0,y_0)}{s}",
             r"\frac{f(x_0,\;y_0+s)-f(x_0,y_0)}{s}",
             r"f_y(x_0,y_0)",
             [r"\text{Only } y \text{ changes; } x \text{ is held at } x_0."],
             Pl.hold_x, Sf.plane_x)

        # ---- both results ------------------------------------------------------
        for m in Pl.dynamic + Sf.dynamic:
            m.clear_updaters()
        self.play(FadeOut(VGroup(*Pl.dynamic, *Sf.dynamic)), run_time=0.6)
        self.play(FadeIn(Pl.i_arrow), FadeIn(Pl.j_arrow), FadeIn(Pl.hold_y), FadeIn(Pl.hold_x),
                  FadeIn(Sf.pair_i), FadeIn(Sf.pair_j), FadeIn(Sf.lab_i), FadeIn(Sf.lab_j),
                  FadeIn(Sf.a_dot), FadeIn(Sf.slope_x), FadeIn(Sf.slope_y), run_time=1.2)
        r1 = MathTex(DI, "=", r"f_x(x_0,y_0)", font_size=36)
        r2 = MathTex(DJ, "=", r"f_y(x_0,y_0)", font_size=36)
        r1[2].set_color(TAN_C)
        r2[2].set_color(TAN_C)
        res = VGroup(r1, r2).arrange(RIGHT, buff=1.4).move_to(np.array([0, -2.55, 0]))
        rb = SurroundingRectangle(res, color=KEY_C, buff=0.22, stroke_width=2.5,
                                  corner_radius=0.1)
        end = T(r"The partial derivatives are the directional derivatives in the two "
                r"coordinate directions.", fs=29)
        end.move_to(np.array([0, -3.55, 0]))
        self.play(Write(r1), Write(r2), run_time=1.6)
        self.play(Create(rb), FadeIn(end), run_time=0.9)
        self.w(6.0)
