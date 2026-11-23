"""Building the two cylinders of the Quadric Surfaces and Cylinders section
from the definition of a cylinder.

Two scenes, one per example of the Cylinders subsection of QuadricSurface.ptx:

    ParabolicCylinderRulings   z = x^2          (y is the missing variable)
    CircularCylinderRulings    y^2 + z^2 = 4    (x is the missing variable)

Each follows the definition rather than the slicing method of the six
quadric-surface videos: the generating curve is drawn in its coordinate
plane; a point P is marked on it and the line through P parallel to the
axis of the missing variable is drawn, because the missing variable is free;
P then travels along the curve, its line sweeping out all the other parallel
lines, and the surface they make up fades in.  The look matches the other
videos of the section: the equation stays in the top-left corner, captions
sit at the bottom of the frame, and the conclusion closes in yellow while the
camera turns.

Render, from the repository root:

    manim -qm --disable_caching -o parabolic_cylinder_rulings_manim \
        assets/manim/cylinder_rulings.py ParabolicCylinderRulings
    manim -qm --disable_caching -o circular_cylinder_rulings_manim \
        assets/manim/cylinder_rulings.py CircularCylinderRulings

(-qm is 1280x720 at 30 fps, the format of the section's other videos.)  The
still-frame previews in assets/videos/previews/ are the frame one third of
the way through each video:

    ffmpeg -ss <duration/3> -i <video>.mp4 -frames:v 1 -q:v 3 <preview>.jpg

Only amsmath and amssymb are needed by the formulas, so a minimal TeX
preamble is used; the full manim default preamble would require several
extra LaTeX font packages.
"""
import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}")
config.tex_template = TEX

TEXT_FONT = "DejaVu Sans"
SURFACE_COLORS = [BLUE_D, BLUE_E]
CURVE_COLOR = ORANGE
LINE_COLOR = YELLOW
RULING_COLOR = BLUE_B
CAPTION_COLOR = GREY_B


def mtex(s, **kw):
    kw.setdefault("tex_template", TEX)
    return MathTex(s, **kw)


class CylinderRulings(ThreeDScene):
    """Shared storyboard; the subclasses supply the geometry."""

    EQUATION = ""
    AXES = {}
    CAMERA = dict(phi=66 * DEGREES, theta=-48 * DEGREES)
    FINAL_THETA = -20 * DEGREES
    T_RANGE = (0.0, 1.0)
    T_START = 0.5
    N_RULINGS = 24
    SURFACE_RES = (20, 24)
    CAPTIONS = {}
    CONCLUSION = ("", "")

    # --- geometry supplied by the subclasses -------------------------------
    def curve_point(self, t):
        """A point of the generating curve, as (x, y, z)."""
        raise NotImplementedError

    def ruling_ends(self, t):
        """The two ends of the ruling through curve_point(t)."""
        raise NotImplementedError

    def surface_point(self, u, v):
        raise NotImplementedError

    # --- helpers ----------------------------------------------------------
    def c2p(self, p):
        return self.axes.c2p(*p)

    def make_ruling(self, t, color=LINE_COLOR, width=5, opacity=1.0):
        a, b = self.ruling_ends(t)
        return Line3D(self.c2p(a), self.c2p(b), color=color, thickness=0.012 * width / 5,
                      ).set_opacity(opacity)

    def caption(self, lines):
        parts = []
        for kind, s in lines:
            if kind == "tex":
                parts.append(mtex(s, font_size=38, color=CAPTION_COLOR))
            else:
                parts.append(Text(s, font=TEXT_FONT, font_size=26, color=CAPTION_COLOR))
        block = VGroup(*parts).arrange(DOWN, buff=0.2)
        block.to_edge(DOWN, buff=0.35)
        return block

    def show_caption(self, cap, old=None):
        if isinstance(cap, (list, tuple)):
            cap = self.caption(cap)
        self.add_fixed_in_frame_mobjects(cap)
        self.remove(cap)
        anims = [FadeIn(cap)]
        if old is not None:
            anims.append(FadeOut(old))
        self.play(*anims, run_time=0.7)
        return cap

    # --- the storyboard ---------------------------------------------------
    def construct(self):
        self.axes = ThreeDAxes(**self.AXES)
        self.axes.set_stroke(width=2)
        self.set_camera_orientation(**self.CAMERA)

        equation = mtex(self.EQUATION, font_size=48)
        equation.to_corner(UL, buff=0.5)
        self.add_fixed_in_frame_mobjects(equation)
        self.remove(equation)
        self.play(Create(self.axes), FadeIn(equation), run_time=1.6)

        # 1. the missing variable, and the generating curve in its plane
        cap = self.show_caption(self.CAPTIONS["missing"])
        self.wait(1.2)
        t0, t1 = self.T_RANGE
        gen = ParametricFunction(lambda t: self.c2p(self.curve_point(t)),
                                 t_range=[t0, t1], color=CURVE_COLOR, stroke_width=6)
        cap = self.show_caption(self.CAPTIONS["curve"], cap)
        self.play(Create(gen), run_time=2.2)
        self.wait(0.8)

        # 2. a point P on it, and the line through P parallel to the axis
        tv = ValueTracker(self.T_START)
        dot = always_redraw(lambda: Dot3D(self.c2p(self.curve_point(tv.get_value())),
                                          color=LINE_COLOR, radius=0.07))
        p_label = mtex("P", font_size=36, color=LINE_COLOR)
        self.add_fixed_orientation_mobjects(p_label)
        self.remove(p_label)
        p_label.add_updater(lambda m: m.move_to(
            self.c2p(self.curve_point(tv.get_value())) + 0.35 * OUT + 0.25 * LEFT))
        cap = self.show_caption(self.CAPTIONS["point"], cap)
        self.play(FadeIn(dot), FadeIn(p_label), run_time=0.6)
        self.wait(0.8)
        line = always_redraw(lambda: self.make_ruling(tv.get_value()))
        cap = self.show_caption(self.CAPTIONS["line"], cap)
        self.play(FadeIn(line), run_time=1.0)
        self.wait(1.4)

        # 3. P travels along the curve; its line leaves the other rulings behind
        cap = self.show_caption(self.CAPTIONS["sweep"], cap)
        trail = VGroup()
        self.add(trail)
        ts = np.linspace(t0, t1, self.N_RULINGS)
        left = [t for t in ts if t < self.T_START][::-1]
        right = [t for t in ts if t >= self.T_START]
        for seq in (left, right):
            if not seq:
                continue
            self.play(tv.animate.set_value(seq[0]), run_time=0.9 if seq is left else 1.2,
                      rate_func=rate_functions.ease_in_out_sine)
            for tt in seq:
                self.play(tv.animate.set_value(tt), run_time=0.22, rate_func=linear)
                trail.add(self.make_ruling(tt, color=RULING_COLOR, width=2.5, opacity=0.55))
        self.wait(0.6)

        # 4. the surface the lines make up
        surface = Surface(
            lambda u, v: self.c2p(self.surface_point(u, v)),
            u_range=self.U_RANGE, v_range=self.V_RANGE,
            resolution=self.SURFACE_RES, fill_opacity=0.55,
            checkerboard_colors=SURFACE_COLORS,
            stroke_color=BLUE_B, stroke_width=0.3)
        cap = self.show_caption(self.CAPTIONS["surface"], cap)
        self.play(FadeIn(surface), run_time=1.8)
        # keep the generating curve, P and its line visible on top of the surface
        self.add(gen, line, dot, p_label)
        self.wait(0.8)

        # 5. the conclusion, and a slow turn
        line1 = Text(self.CONCLUSION[0], font=TEXT_FONT, font_size=28)
        line2 = VGroup(
            mtex(r"\Rightarrow", font_size=36, color=YELLOW),
            Text(self.CONCLUSION[1], font=TEXT_FONT, font_size=28, color=YELLOW),
        ).arrange(RIGHT, buff=0.2)
        conclusion = VGroup(line1, line2).arrange(DOWN, buff=0.25)
        conclusion.to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(conclusion)
        self.remove(conclusion)
        self.play(FadeOut(cap), FadeIn(line1), run_time=0.8)
        self.play(FadeIn(line2), run_time=0.8)
        self.move_camera(theta=self.FINAL_THETA, run_time=5.0)
        self.wait(1.5)


class ParabolicCylinderRulings(CylinderRulings):
    """z = x^2: the parabola z = x^2 in the xz-plane, lines parallel to the y-axis."""

    EQUATION = r"z = x^2"
    AXES = dict(x_range=[-3, 3, 1], y_range=[-4, 4, 1], z_range=[-1, 5, 1],
                x_length=4.8, y_length=5.8, z_length=3.8)
    T_RANGE = (-2.0, 2.0)
    T_START = 1.0
    N_RULINGS = 25
    U_RANGE = [-2, 2]
    V_RANGE = [-3, 3]
    CAPTIONS = {
        "missing": [("text", "y is missing from the equation: it is free to take any value")],
        "curve": [("tex", r"y = 0:\quad z = x^2"),
                  ("text", "the generating curve, a parabola in the xz-plane")],
        "point": [("tex", r"P = (a,\, 0,\, a^2)"),
                  ("text", "a point of the generating curve")],
        "line": [("tex", r"(a,\, y,\, a^2) \text{ for every } y"),
                 ("text", "the whole line through P parallel to the y-axis is on the surface")],
        "sweep": [("text", "the same is true at every point of the parabola"),
                  ("text", "all the lines parallel to the y-axis through the generating curve")],
        "surface": [("text", "together the lines make up the surface")],
    }
    CONCLUSION = ("a parabola as generating curve, rulings parallel to the y-axis",
                  "a parabolic cylinder")

    def curve_point(self, t):
        return (t, 0.0, t * t)

    def ruling_ends(self, t):
        return (t, -3.0, t * t), (t, 3.0, t * t)

    def surface_point(self, u, v):
        return (u, v, u * u)


class CircularCylinderRulings(CylinderRulings):
    """y^2 + z^2 = 4: the circle of radius 2 in the yz-plane, lines parallel to the x-axis."""

    EQUATION = r"y^2 + z^2 = 4"
    AXES = dict(x_range=[-5, 5, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
                x_length=6.4, y_length=3.9, z_length=3.9)
    CAMERA = dict(phi=64 * DEGREES, theta=-52 * DEGREES)
    FINAL_THETA = -24 * DEGREES
    T_RANGE = (0.0, TAU)
    T_START = PI / 3
    N_RULINGS = 30
    U_RANGE = [0, TAU]
    V_RANGE = [-4, 4]
    SURFACE_RES = (24, 16)
    CAPTIONS = {
        "missing": [("text", "x is missing from the equation: it is free to take any value")],
        "curve": [("tex", r"x = 0:\quad y^2 + z^2 = 4"),
                  ("text", "the generating curve, a circle of radius 2 in the yz-plane")],
        "point": [("tex", r"P = (0,\, b,\, c),\quad b^2 + c^2 = 4"),
                  ("text", "a point of the generating curve")],
        "line": [("tex", r"(x,\, b,\, c) \text{ for every } x"),
                 ("text", "the whole line through P parallel to the x-axis is on the surface")],
        "sweep": [("text", "the same is true at every point of the circle"),
                  ("text", "all the lines parallel to the x-axis through the generating curve")],
        "surface": [("text", "together the lines make up the surface")],
    }
    CONCLUSION = ("a circle as generating curve, rulings parallel to the x-axis",
                  "a circular cylinder, with the x-axis as its axis")

    def curve_point(self, t):
        return (0.0, 2 * np.cos(t), 2 * np.sin(t))

    def ruling_ends(self, t):
        return (-4.0, 2 * np.cos(t), 2 * np.sin(t)), (4.0, 2 * np.cos(t), 2 * np.sin(t))

    def surface_point(self, u, v):
        return (v, 2 * np.cos(u), 2 * np.sin(u))
