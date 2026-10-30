"""The quadric surfaces, told by their slices: the summary video of the
Quadric Surfaces and Cylinders section.

One scene, QuadricSurfacesSummary, in five parts, each on the parent graph
whose coefficients are all +1 or -1:

    z = x^2 + y^2          two parabolas and a circle      elliptical paraboloid
    z^2 = x^2 + y^2        two pairs of crossing lines     cone
                           and circles
    x^2 + y^2 - z^2 = 1    two hyperbolas and a circle     hyperboloid of one sheet
    z^2 - x^2 - y^2 = 1    two hyperbolas and a circle     hyperboloid of two sheets
    z = x^2 - y^2          two parabolas, one opening up   hyperbolic paraboloid
                           and one down, and hyperbolas

Between the hyperboloids and the hyperbolic paraboloid an interlude puts the
two hyperboloids side by side and rearranges their equations: for one sheet
x^2 + y^2 = 1 + z^2 has a circle at every height, so z is unrestricted; for
two sheets x^2 + y^2 = z^2 - 1 needs z^2 >= 1, and the slab -1 < z < 1 is
empty.  A closing card lists all five.  The look matches the section's other
videos: the equation stays in the top-left corner, captions sit at the bottom
of the frame, and each conclusion is in yellow while the camera turns.

Render, from the repository root:

    manim -qm --disable_caching -o quadric_surfaces_summary_manim \
        assets/manim/quadric_summary.py QuadricSurfacesSummary

(-qm is 1280x720 at 30 fps, the format of the section's other videos.)  The
still-frame preview in assets/videos/previews/ is the frame one third of the
way through the video:

    ffmpeg -ss <duration/3> -i <video>.mp4 -frames:v 1 -q:v 3 <preview>.jpg

Only amsmath and amssymb are needed by the formulas, so a minimal TeX
preamble is used.
"""
import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}")
config.tex_template = TEX

TEXT_FONT = "DejaVu Sans"
SURFACE_COLORS = [BLUE_D, BLUE_E]
PLANE_COLOR = GREY_B
CAPTION_COLOR = GREY_B
TRACE_COLOR = {"x": ORANGE, "y": RED_C, "z": YELLOW}
SCREEN_RIGHT = np.array([np.cos(PI / 4), np.sin(PI / 4), 0.0])  # for theta = -45 degrees


def mtex(s, **kw):
    kw.setdefault("tex_template", TEX)
    return MathTex(s, **kw)


def circle(r, z):
    return lambda t: (r * np.cos(t), r * np.sin(t), z)


class QuadricSurfacesSummary(ThreeDScene):

    CAMERA = dict(phi=66 * DEGREES, theta=-45 * DEGREES)

    # --- fixed-in-frame text ------------------------------------------------
    def fixed(self, m):
        self.add_fixed_in_frame_mobjects(m)
        self.remove(m)
        return m

    def caption(self, lines):
        parts = []
        for kind, s in lines:
            if kind == "tex":
                parts.append(mtex(s, font_size=38, color=CAPTION_COLOR))
            else:
                parts.append(Text(s, font=TEXT_FONT, font_size=26, color=CAPTION_COLOR))
        block = VGroup(*parts).arrange(DOWN, buff=0.2)
        block.to_edge(DOWN, buff=0.35)
        return self.fixed(block)

    def show_caption(self, lines, old=None, run_time=0.7):
        cap = self.caption(lines)
        anims = [FadeIn(cap)]
        if old is not None:
            anims.append(FadeOut(old))
        self.play(*anims, run_time=run_time)
        return cap

    def conclusion(self, first, second, old=None):
        line1 = Text(first, font=TEXT_FONT, font_size=28)
        line2 = VGroup(
            mtex(r"\Rightarrow", font_size=36, color=YELLOW),
            Text(second, font=TEXT_FONT, font_size=28, color=YELLOW),
        ).arrange(RIGHT, buff=0.2)
        block = VGroup(line1, line2).arrange(DOWN, buff=0.25)
        block.to_edge(DOWN, buff=0.35)
        self.fixed(block)
        anims = [FadeIn(line1)]
        if old is not None:
            anims.append(FadeOut(old))
        self.play(*anims, run_time=0.8)
        self.play(FadeIn(line2), run_time=0.8)
        return block

    # --- 3d helpers -----------------------------------------------------------
    def c2p(self, p):
        return self.axes.c2p(*p)

    def make_axes(self, z_range):
        axes = ThreeDAxes(x_range=[-2.5, 2.5, 1], y_range=[-2.5, 2.5, 1], z_range=z_range,
                          x_length=6.2, y_length=6.2,
                          z_length=1.05 * (z_range[1] - z_range[0]))
        axes.set_stroke(width=2)
        return axes

    def axis_labels(self, axes, z_top):
        labels = VGroup(mtex("x", font_size=30).move_to(axes.c2p(2.85, 0, 0)),
                        mtex("y", font_size=30).move_to(axes.c2p(0, 2.85, 0)),
                        mtex("z", font_size=30).move_to(axes.c2p(0, 0, z_top + 0.35)))
        self.add_fixed_orientation_mobjects(*labels)
        self.remove(*labels)
        return labels

    def make_surface(self, f, u_range, v_range, resolution=(24, 24), opacity=0.55):
        return Surface(lambda u, v: self.c2p(f(u, v)), u_range=u_range, v_range=v_range,
                       resolution=resolution, fill_opacity=opacity,
                       checkerboard_colors=SURFACE_COLORS,
                       stroke_color=BLUE_B, stroke_width=0.3)

    def make_plane(self, axis, c, span, z_range):
        if axis == "x":
            f = lambda u, v: self.c2p((c, u, v))
            u_range, v_range = [-span, span], list(z_range)
        elif axis == "y":
            f = lambda u, v: self.c2p((u, c, v))
            u_range, v_range = [-span, span], list(z_range)
        else:
            f = lambda u, v: self.c2p((u, v, c))
            u_range, v_range = [-span, span], [-span, span]
        return Surface(f, u_range=u_range, v_range=v_range, resolution=(2, 2),
                       fill_opacity=0.22, checkerboard_colors=[PLANE_COLOR, PLANE_COLOR],
                       stroke_color=PLANE_COLOR, stroke_width=0.8)

    def make_trace(self, f, t_range, color, width=6):
        return ParametricFunction(lambda t: self.c2p(f(t)), t_range=list(t_range),
                                  color=color, stroke_width=width)

    def slice_with(self, axis, c, span, z_range, traces, lines, cap, hold=1.6):
        """Show the plane axis = c, draw the traces on it, and leave the traces behind."""
        plane = self.make_plane(axis, c, span, z_range)
        curves = VGroup(*[self.make_trace(f, tr, TRACE_COLOR[axis]) for f, tr in traces])
        cap = self.show_caption(lines, cap)
        self.play(FadeIn(plane), run_time=0.8)
        if len(curves) > 0:
            self.play(*[Create(cv) for cv in curves], run_time=1.6)
        self.wait(hold)
        self.play(FadeOut(plane), run_time=0.5)
        return curves, cap

    def equation(self, s, name=None):
        eq = mtex(s, font_size=48)
        eq.to_corner(UL, buff=0.5)
        self.fixed(eq)
        return eq

    def clear_scene(self, *mobs, run_time=0.9):
        self.play(*[FadeOut(m) for m in mobs if m is not None], run_time=run_time)

    def part(self, eq_tex, z_range, surface, slices, concl, final_theta=-15 * DEGREES,
             extra=None):
        """One surface: axes, equation, the surface, its slices, the conclusion."""
        self.axes = self.make_axes(z_range)
        self.set_camera_orientation(**self.CAMERA)
        eq = self.equation(eq_tex)
        labels = self.axis_labels(self.axes, z_range[1])
        surf = surface()
        self.play(Create(self.axes), FadeIn(labels), FadeIn(eq), run_time=1.2)
        self.play(FadeIn(surf), run_time=1.4)
        self.wait(0.5)
        cap = None
        kept = VGroup()
        for axis, c, span, traces, lines in slices:
            curves, cap = self.slice_with(axis, c, span, z_range, traces, lines, cap)
            kept.add(curves)
        if extra is not None:
            cap = extra(cap)
        block = self.conclusion(concl[0], concl[1], cap)
        self.move_camera(theta=final_theta, run_time=4.5)
        self.wait(1.2)
        self.clear_scene(self.axes, labels, surf, kept, eq, block)

    # --- the storyboard -----------------------------------------------------------
    def construct(self):
        self.intro()
        self.paraboloid()
        self.cone()
        self.one_sheet()
        self.two_sheets()
        self.one_or_two()
        self.hyperbolic_paraboloid()
        self.outro()

    def intro(self):
        title = Text("The quadric surfaces, told by their slices",
                     font=TEXT_FONT, font_size=40)
        sub = Text("the parent graphs: every coefficient is 1 or −1",
                   font=TEXT_FONT, font_size=28, color=CAPTION_COLOR)
        rule = Text("slice with x = 0, y = 0 and z = c, and read off the curves",
                    font=TEXT_FONT, font_size=28, color=CAPTION_COLOR)
        block = VGroup(title, sub, rule).arrange(DOWN, buff=0.4)
        self.fixed(block)
        self.play(FadeIn(title), run_time=1.0)
        self.play(FadeIn(sub), run_time=0.8)
        self.play(FadeIn(rule), run_time=0.8)
        self.wait(1.8)
        self.play(FadeOut(block), run_time=0.7)

    # 1. the elliptical paraboloid
    def paraboloid(self):
        z_range = [-0.6, 2.6, 1]
        surface = lambda: self.make_surface(
            lambda u, v: (v * np.cos(u), v * np.sin(u), v * v), [0, TAU], [0, 1.5])
        slices = [
            ("x", 0, 1.9, [(lambda t: (0, t, t * t), (-1.5, 1.5))],
             [("tex", r"x = 0:\quad z = y^2"), ("text", "a parabola")]),
            ("y", 0, 1.9, [(lambda t: (t, 0, t * t), (-1.5, 1.5))],
             [("tex", r"y = 0:\quad z = x^2"), ("text", "a second parabola")]),
            ("z", 1, 1.9, [(circle(1, 1), (0, TAU))],
             [("tex", r"z = 1:\quad x^2 + y^2 = 1"), ("text", "a circle")]),
        ]
        self.part(r"z = x^2 + y^2", z_range, surface, slices,
                  ("two parabolas and a circle", "an elliptical paraboloid"))

    # 2. the cone
    def cone(self):
        z_range = [-2.2, 2.2, 1]
        surface = lambda: self.make_surface(
            lambda u, v: (v * np.cos(u), v * np.sin(u), v), [0, TAU], [-1.8, 1.8],
            resolution=(24, 16))
        slices = [
            ("x", 0, 2.0, [(lambda t: (0, t, t), (-1.8, 1.8)), (lambda t: (0, t, -t), (-1.8, 1.8))],
             [("tex", r"x = 0:\quad z^2 = y^2,\ \text{so } z = \pm y"),
              ("text", "two lines crossing at the origin")]),
            ("y", 0, 2.0, [(lambda t: (t, 0, t), (-1.8, 1.8)), (lambda t: (t, 0, -t), (-1.8, 1.8))],
             [("tex", r"y = 0:\quad z^2 = x^2,\ \text{so } z = \pm x"),
              ("text", "a second pair of crossing lines")]),
            ("z", 1, 2.0, [(circle(1, 1), (0, TAU))],
             [("tex", r"z = 1:\quad x^2 + y^2 = 1"), ("text", "a circle")]),
            ("z", -1, 2.0, [(circle(1, -1), (0, TAU))],
             [("tex", r"z = -1:\quad x^2 + y^2 = 1"),
              ("text", "another circle: every plane z = c cuts a circle of radius |c|")]),
        ]
        self.part(r"z^2 = x^2 + y^2", z_range, surface, slices,
                  ("two pairs of crossing lines and circles", "a cone"))

    # 3. the hyperboloid of one sheet
    def one_sheet(self):
        z_range = [-2.0, 2.0, 1]
        surface = lambda: self.make_surface(
            lambda u, v: (np.cosh(v) * np.cos(u), np.cosh(v) * np.sin(u), np.sinh(v)),
            [0, TAU], [-1.25, 1.25], resolution=(24, 16))
        H = 1.25
        slices = [
            ("x", 0, 2.1, [(lambda t: (0, np.cosh(t), np.sinh(t)), (-H, H)),
                           (lambda t: (0, -np.cosh(t), np.sinh(t)), (-H, H))],
             [("tex", r"x = 0:\quad y^2 - z^2 = 1"), ("text", "a hyperbola")]),
            ("y", 0, 2.1, [(lambda t: (np.cosh(t), 0, np.sinh(t)), (-H, H)),
                           (lambda t: (-np.cosh(t), 0, np.sinh(t)), (-H, H))],
             [("tex", r"y = 0:\quad x^2 - z^2 = 1"), ("text", "a second hyperbola")]),
            ("z", 0, 2.1, [(circle(1, 0), (0, TAU))],
             [("tex", r"z = 0:\quad x^2 + y^2 = 1"), ("text", "a circle, the waist of the surface")]),
        ]
        self.part(r"x^2 + y^2 - z^2 = 1", z_range, surface, slices,
                  ("two hyperbolas and a circle", "a hyperboloid of one sheet"))

    # 4. the hyperboloid of two sheets
    def two_sheets(self):
        z_range = [-2.4, 2.4, 1]
        H = 1.3

        def surface():
            up = self.make_surface(
                lambda u, v: (np.sinh(v) * np.cos(u), np.sinh(v) * np.sin(u), np.cosh(v)),
                [0, TAU], [0, H], resolution=(24, 12))
            down = self.make_surface(
                lambda u, v: (np.sinh(v) * np.cos(u), np.sinh(v) * np.sin(u), -np.cosh(v)),
                [0, TAU], [0, H], resolution=(24, 12))
            return VGroup(up, down)

        r15 = np.sqrt(1.5 ** 2 - 1)
        slices = [
            ("x", 0, 2.1, [(lambda t: (0, np.sinh(t), np.cosh(t)), (-H, H)),
                           (lambda t: (0, np.sinh(t), -np.cosh(t)), (-H, H))],
             [("tex", r"x = 0:\quad z^2 - y^2 = 1"), ("text", "a hyperbola, opening up and down")]),
            ("y", 0, 2.1, [(lambda t: (np.sinh(t), 0, np.cosh(t)), (-H, H)),
                           (lambda t: (np.sinh(t), 0, -np.cosh(t)), (-H, H))],
             [("tex", r"y = 0:\quad z^2 - x^2 = 1"), ("text", "a second hyperbola")]),
            ("z", 1.5, 2.1, [(circle(r15, 1.5), (0, TAU))],
             [("tex", r"z = \tfrac32:\quad x^2 + y^2 = \tfrac94 - 1 = \tfrac54"), ("text", "a circle")]),
            ("z", 0.5, 2.1, [],
             [("tex", r"z = \tfrac12:\quad x^2 + y^2 = \tfrac14 - 1 < 0"),
              ("text", "no points at all: the plane passes between the sheets")]),
        ]
        self.part(r"z^2 - x^2 - y^2 = 1", z_range, surface, slices,
                  ("two hyperbolas and a circle (where there is one)",
                   "a hyperboloid of two sheets"))

    # 5. one sheet or two?  rearrange and look at z
    def one_or_two(self):
        self.axes = self.make_axes([-2.3, 2.3, 1])
        self.axes.scale(0.6).shift(-1.0 * OUT)
        self.set_camera_orientation(**self.CAMERA)
        left = self.axes.copy().shift(-3.0 * SCREEN_RIGHT)
        right = self.axes.copy().shift(3.0 * SCREEN_RIGHT)
        H1, H2 = 1.2, 1.3

        one = Surface(lambda u, v: left.c2p(np.cosh(v) * np.cos(u), np.cosh(v) * np.sin(u), np.sinh(v)),
                      u_range=[0, TAU], v_range=[-H1, H1], resolution=(24, 14),
                      fill_opacity=0.5, checkerboard_colors=SURFACE_COLORS,
                      stroke_color=BLUE_B, stroke_width=0.3)
        two = VGroup(*[Surface(lambda u, v, s=s: right.c2p(np.sinh(v) * np.cos(u), np.sinh(v) * np.sin(u), s * np.cosh(v)),
                               u_range=[0, TAU], v_range=[0, H2], resolution=(24, 10),
                               fill_opacity=0.5, checkerboard_colors=SURFACE_COLORS,
                               stroke_color=BLUE_B, stroke_width=0.3) for s in (1, -1)])

        title = Text("one sheet or two?  rearrange, and look at z", font=TEXT_FONT, font_size=30)
        title.to_edge(UP, buff=0.3)
        self.fixed(title)
        eqL = VGroup(mtex(r"x^2 + y^2 - z^2 = 1", font_size=36),
                     mtex(r"x^2 + y^2 = 1 + z^2", font_size=36),
                     mtex(r"1 + z^2 \ge 1 \text{ for every } z", font_size=32, color=YELLOW),
                     ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        eqR = VGroup(mtex(r"z^2 - x^2 - y^2 = 1", font_size=36),
                     mtex(r"x^2 + y^2 = z^2 - 1", font_size=36),
                     mtex(r"\text{needs } z^2 - 1 \ge 0", font_size=32, color=YELLOW),
                     mtex(r"z \le -1 \ \text{ or } \ z \ge 1", font_size=32, color=YELLOW),
                     ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        eqL.to_corner(UL, buff=0.45).shift(0.55 * DOWN)
        eqR.to_corner(UR, buff=0.45).shift(0.55 * DOWN)
        self.fixed(eqL)
        self.fixed(eqR)

        self.play(FadeIn(title), Create(left), Create(right), run_time=1.2)
        self.play(FadeIn(one), FadeIn(two), FadeIn(eqL[0]), FadeIn(eqR[0]), run_time=1.4)
        self.wait(0.6)
        cap = self.show_caption([("text", "move the z term to the other side")])
        self.play(FadeIn(eqL[1]), FadeIn(eqR[1]), run_time=1.0)
        self.wait(1.2)

        # a plane z = c sweeps both surfaces
        cv = ValueTracker(-1.9)

        def plane_at(ax):
            return always_redraw(lambda: Surface(
                lambda u, v: ax.c2p(u, v, cv.get_value()), u_range=[-2.0, 2.0], v_range=[-2.0, 2.0],
                resolution=(2, 2), fill_opacity=0.2, checkerboard_colors=[PLANE_COLOR, PLANE_COLOR],
                stroke_color=PLANE_COLOR, stroke_width=0.8))

        def circle_one():
            c = cv.get_value()
            r = np.sqrt(1 + c * c)
            return ParametricFunction(lambda t: left.c2p(r * np.cos(t), r * np.sin(t), c),
                                      t_range=[0, TAU], color=YELLOW, stroke_width=6)

        def circle_two():
            c = cv.get_value()
            if c * c < 1:
                return VGroup()
            r = np.sqrt(c * c - 1)
            return ParametricFunction(lambda t: right.c2p(r * np.cos(t), r * np.sin(t), c),
                                      t_range=[0, TAU], color=YELLOW, stroke_width=6)

        planes = VGroup(plane_at(left), plane_at(right))
        circ1 = always_redraw(circle_one)
        circ2 = always_redraw(circle_two)
        cap = self.show_caption([("text", "slide a plane z = c up through both surfaces")], cap)
        self.play(FadeIn(planes), run_time=0.6)
        self.add(circ1, circ2)
        self.play(cv.animate.set_value(-1.0), run_time=1.6, rate_func=linear)
        cap = self.show_caption([("text", "left: a circle at every height, x² + y² = 1 + c² is never empty"),
                                 ("text", "right: for −1 < c < 1 the right side is negative, no points")], cap, run_time=0.5)
        self.play(cv.animate.set_value(1.0), run_time=3.2, rate_func=linear)
        self.play(cv.animate.set_value(1.9), run_time=1.6, rate_func=linear)
        self.play(FadeOut(planes), FadeOut(circ1), FadeOut(circ2), run_time=0.5)

        self.play(FadeIn(eqL[2]), run_time=0.8)
        self.play(FadeIn(eqR[2]), FadeIn(eqR[3]), run_time=0.8)
        # mark the empty slab on the right
        slab = VGroup(*[Surface(lambda u, v, s=s: right.c2p(u, v, s), u_range=[-1.6, 1.6], v_range=[-1.6, 1.6],
                                resolution=(2, 2), fill_opacity=0.18, checkerboard_colors=[YELLOW, YELLOW],
                                stroke_color=YELLOW, stroke_width=1.0) for s in (-1.0, 1.0)])
        block = self.conclusion("every z allowed: one sheet.   a gap −1 < z < 1: two sheets",
                                "rearranging the equation tells them apart", cap)
        self.play(FadeIn(slab), run_time=0.8)
        self.move_camera(theta=-20 * DEGREES, run_time=4.0)
        self.wait(1.2)
        self.clear_scene(left, right, one, two, slab, title, eqL, eqR, block)

    # 6. the hyperbolic paraboloid
    def hyperbolic_paraboloid(self):
        z_range = [-2.5, 2.5, 1]
        surface = lambda: self.make_surface(
            lambda u, v: (u, v, u * u - v * v), [-1.5, 1.5], [-1.5, 1.5], resolution=(20, 20))
        T = 0.95
        slices = [
            ("x", 0, 1.9, [(lambda t: (0, t, -t * t), (-1.5, 1.5))],
             [("tex", r"x = 0:\quad z = -y^2"), ("text", "a parabola opening down")]),
            ("y", 0, 1.9, [(lambda t: (t, 0, t * t), (-1.5, 1.5))],
             [("tex", r"y = 0:\quad z = x^2"), ("text", "a parabola opening up: the other type")]),
            ("z", 1, 1.9, [(lambda t: (np.cosh(t), np.sinh(t), 1), (-T, T)),
                           (lambda t: (-np.cosh(t), np.sinh(t), 1), (-T, T))],
             [("tex", r"z = 1:\quad x^2 - y^2 = 1"), ("text", "a hyperbola, opening along x")]),
            ("z", -1, 1.9, [(lambda t: (np.sinh(t), np.cosh(t), -1), (-T, T)),
                            (lambda t: (np.sinh(t), -np.cosh(t), -1), (-T, T))],
             [("tex", r"z = -1:\quad y^2 - x^2 = 1"),
              ("text", "a hyperbola opening along y (and z = 0 gives the two lines y = ±x)")]),
        ]
        self.part(r"z = x^2 - y^2", z_range, surface, slices,
                  ("two parabolas of opposite types, and hyperbolas",
                   "a hyperbolic paraboloid, the saddle"), final_theta=-10 * DEGREES)

    # 7. the summary card
    def outro(self):
        rows = [
            (r"z = x^2 + y^2", "parabola, parabola, circle", "elliptical paraboloid"),
            (r"z^2 = x^2 + y^2", "crossing lines, crossing lines, circle", "cone"),
            (r"x^2 + y^2 - z^2 = 1", "hyperbola, hyperbola, circle;  every z", "hyperboloid of one sheet"),
            (r"z^2 - x^2 - y^2 = 1", "hyperbola, hyperbola, circle;  |z| ≥ 1", "hyperboloid of two sheets"),
            (r"z = x^2 - y^2", "parabola down, parabola up, hyperbola", "hyperbolic paraboloid"),
        ]
        header = Text("x = 0,  y = 0,  z = c slices", font=TEXT_FONT, font_size=24, color=CAPTION_COLOR)
        table = VGroup()
        for eq, sl, name in rows:
            table.add(VGroup(mtex(eq, font_size=34),
                             Text(sl, font=TEXT_FONT, font_size=22, color=CAPTION_COLOR),
                             Text(name, font=TEXT_FONT, font_size=24, color=YELLOW)))
        cols = [VGroup(*[r[i] for r in table]).arrange(DOWN, buff=0.38, aligned_edge=LEFT) for i in range(3)]
        grid = VGroup(*cols).arrange(RIGHT, buff=0.7, aligned_edge=UP)
        # align the rows of the three columns
        for i in range(len(rows)):
            y = cols[0][i].get_center()[1]
            for c in cols[1:]:
                c[i].move_to([c[i].get_center()[0], y, 0])
        header.next_to(cols[1], UP, buff=0.35)
        card = VGroup(header, grid).move_to(ORIGIN)
        if card.width > 12.6:
            card.scale_to_fit_width(12.6)
        self.fixed(card)
        self.play(FadeIn(header), run_time=0.6)
        for i in range(len(rows)):
            self.play(*[FadeIn(c[i]) for c in cols], run_time=0.7)
        self.wait(3.0)
        self.play(FadeOut(card), run_time=0.8)
