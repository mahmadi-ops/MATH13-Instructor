"""Why the rates add: the two-variable chain rule, seen on a hill.

The moving version of "Seeing the chain rule geometrically" in 4.4.1
(fig-cr-hiker-split): a hiker on terrain w = f(x, y), x east and y north.

Render from the repository root with Manim Community 0.19.0:

    manim -qm --disable_caching -o chain_rule_hiker_manim \
        assets/manim/chain_rule_hiker.py ChainRuleHiker

``-qm`` is 1280x720 at 30 fps, matching the other course animations.  Put the
finished movie in ``assets/videos/chain_rule_hiker_manim.mp4`` and a still in
``assets/videos/previews/chain_rule_hiker_manim.jpg``.

The screen is split in two for the whole video: on the left the hill in 3D,
on the right the same hill from above as a contour map, with height gauges
beside it.  The acts:

1. the hill and its map;
2. two hikers leave A together, one walking straight to C, the other east
   to B and then north to C, and they arrive together: the second hiker's
   gauge is built from a blue (east) and an orange (north) piece, and at
   the finish it matches the first hiker's, so the climb depends only on
   the endpoints;
3. the east leg: its rise is about f_x times the eastward step;
4. the north leg: its rise is about f_y times the northward step, so
   Delta w is about f_x Delta x + f_y Delta y;
5. the step shrinks: the true rate Delta w / Delta t and the estimate
   (f_x Delta x + f_y Delta y) / Delta t converge, giving the chain rule.

Design notes:

* The 3D view is drawn with a fixed oblique projection (``proj``) in an
  ordinary Scene, patches painted back to front, rather than with
  ThreeDScene: that keeps the hill in the left half of the frame while the
  flat map, gauges and lettering share the right half without any camera
  bookkeeping.  The camera never moves, and every path is on the near slope,
  so nothing the eye follows is ever hidden.
* The hill is a radial bump, f = H exp(-r^2 / (2 s^2)) about (XC, YC).  Its
  formula is never shown, in keeping with the text ("without finding an
  equation for the hill").  The map's contour bands use the same height
  colours as the surface, so the two panels read as one object.
* Colours follow the book's figure: the east leg BLUE, the north leg ORANGE.
  The straight-line hiker is PINK, on a white dashed path, and the hiker who
  walks the two legs is GOLD.  The black background, DejaVu Sans captions at the bottom and the
  equation at the top follow the repository's other animations.
"""

from manim import *
import numpy as np


TEX = TexTemplate(preamble=r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}")
config.tex_template = TEX
config.background_color = BLACK

TEXT_FONT = "DejaVu Sans"
EAST_COLOR = BLUE
NORTH_COLOR = ORANGE
STRAIGHT_COLOR = PINK
STRAIGHT_PATH = WHITE
TWO_LEG_COLOR = GOLD
CAPTION_COLOR = GREY_B
MUTED_COLOR = GREY_C

# ---- the hill ------------------------------------------------------------------
H, XC, YC, SIG = 2.0, 3.6, 3.3, 1.4      # f = H exp(-((x-XC)^2+(y-YC)^2)/(2 SIG^2))
DOM = 4.5                                # the drawn square is [0, DOM] x [0, DOM]
MID = DOM / 2


def f(x, y):
    return H * np.exp(-((x - XC) ** 2 + (y - YC) ** 2) / (2 * SIG ** 2))


def fx(x, y):
    return -f(x, y) * (x - XC) / SIG ** 2


def fy(x, y):
    return -f(x, y) * (y - YC) / SIG ** 2


# ---- the walk ------------------------------------------------------------------
A = np.array([1.4, 1.3])
STEP = np.array([1.2, 0.8])              # the displacement A -> C, also the velocity
B = A + np.array([STEP[0], 0.0])
C = A + STEP

# ---- the 3D view: a fixed oblique projection -----------------------------------
PANEL3D = np.array([-3.45, -0.55, 0.0])
SC = 1.0                                 # scene units per unit of x or y
AZ = 20 * DEGREES                        # turn of the ground plane
EL = 30 * DEGREES                        # the camera's height above the horizon
ZK = 1.1                                 # vertical scale


def _turn(x, y):
    X, Y = x - MID, y - MID
    u = X * np.cos(AZ) + Y * np.sin(AZ)
    d = -X * np.sin(AZ) + Y * np.cos(AZ)
    return u, d


def proj(x, y, z):
    u, d = _turn(x, y)
    v = d * np.sin(EL) + z * ZK * np.cos(EL)
    return PANEL3D + SC * np.array([u, v, 0.0])


def depth(x, y, z):
    u, d = _turn(x, y)
    return d * np.cos(EL) - z * ZK * np.sin(EL)


def on_hill(x, y, lift=0.0):
    return proj(x, y, f(x, y) + lift)


# ---- the map --------------------------------------------------------------------
MAP_CENTER = np.array([2.85, -0.3, 0.0])
MAP_SIDE = 4.1
MS = MAP_SIDE / DOM


def on_map(x, y):
    return MAP_CENTER + MS * np.array([x - MID, y - MID, 0.0])


LOW, HIGH = ManimColor("#22402b"), ManimColor("#d3ebc9")


def height_color(h):
    return interpolate_color(LOW, HIGH, float(np.clip(h / (H * 1.02), 0, 1)))


# ---- the gauges -----------------------------------------------------------------
GAUGE_X = (5.55, 6.45)                   # the two bars' centres
GAUGE_BASE = -0.95
GAUGE_SCALE = 2.6
GAUGE_W = 0.46


def mtex(*parts, **kwargs):
    kwargs.setdefault("tex_template", TEX)
    return MathTex(*parts, **kwargs)


def fit_width(mobject, width):
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    return mobject


def bar(x_center, bottom, height, color, width=GAUGE_W):
    """A gauge bar standing on ``bottom`` (hanging below it when height < 0)."""
    h = max(abs(height), 1e-3)
    rect = Rectangle(width=width, height=h, stroke_width=0,
                     fill_color=color, fill_opacity=0.95)
    if height >= 0:
        rect.move_to([x_center, bottom, 0], aligned_edge=DOWN)
    else:
        rect.move_to([x_center, bottom, 0], aligned_edge=UP)
    return rect


def surface_path(points, color, width=5, dashed=False, lift=0.02):
    """A curve on the hill through the ground points ``points`` (x, y)."""
    path = VMobject(stroke_color=color, stroke_width=width)
    path.set_points_as_corners([on_hill(x, y, lift) for x, y in points])
    if dashed:
        path = DashedVMobject(path, num_dashes=max(6, len(points) // 3),
                              dashed_ratio=0.55)
        path.set_stroke(color, width)
    return path


def segment(p, q, n=40):
    return [p + (q - p) * t for t in np.linspace(0, 1, n)]


def two_leg_point(start, mid, end, s):
    """Where the two-leg hiker is after a fraction s of the walk."""
    l1 = np.linalg.norm(mid - start)
    l2 = np.linalg.norm(end - mid)
    arc = s * (l1 + l2)
    if arc <= l1:
        return start + (mid - start) * (arc / l1 if l1 else 0)
    return mid + (end - mid) * ((arc - l1) / l2 if l2 else 0)


class ChainRuleHiker(Scene):
    def construct(self):
        self.current_caption = None
        self.current_equation = None
        self.build_panels()
        self.race()
        self.east_leg()
        self.north_leg()
        self.shrink()
        self.conclusion()

    # ------------------------------------------------------------------
    # layout helpers
    def caption(self, text):
        new = fit_width(Text(text, font=TEXT_FONT, font_size=24,
                             color=CAPTION_COLOR), 13.4)
        new.to_edge(DOWN, buff=0.3)
        anims = [FadeIn(new)]
        if self.current_caption is not None:
            anims.append(FadeOut(self.current_caption))
        self.play(*anims, run_time=0.55)
        self.current_caption = new

    def equation(self, *parts, colors=None, font_size=40):
        eq = mtex(*parts, font_size=font_size)
        if colors:
            for i, c in colors.items():
                eq[i].set_color(c)
        fit_width(eq, 12.5)
        eq.move_to([0, 3.3, 0])
        return eq

    def show_equation(self, eq, transform=True):
        if self.current_equation is None:
            self.play(Write(eq), run_time=1.0)
        elif transform:
            self.play(TransformMatchingTex(self.current_equation, eq), run_time=1.1)
        else:
            self.play(FadeOut(self.current_equation), FadeIn(eq), run_time=0.8)
        self.current_equation = eq

    # ------------------------------------------------------------------
    # act 1: the hill and its map
    def build_panels(self):
        n = 30
        g = np.linspace(0, DOM, n + 1)
        light = np.array([-0.45, -0.6, 0.75])
        light = light / np.linalg.norm(light)
        patches = []
        for i in range(n):
            for j in range(n):
                xs = (g[i], g[i + 1], g[i + 1], g[i])
                ys = (g[j], g[j], g[j + 1], g[j + 1])
                cx, cy = (g[i] + g[i + 1]) / 2, (g[j] + g[j + 1]) / 2
                h = f(cx, cy)
                normal = np.array([-fx(cx, cy) * ZK, -fy(cx, cy) * ZK, 1.0])
                shade = 0.45 + 0.55 * max(0.0, float(normal @ light) / np.linalg.norm(normal))
                col = interpolate_color(BLACK, height_color(h), shade)
                poly = Polygon(*[proj(x, y, f(x, y)) for x, y in zip(xs, ys)],
                               fill_color=col, fill_opacity=1,
                               stroke_color=BLACK, stroke_width=0.5, stroke_opacity=0.35)
                patches.append((depth(cx, cy, h), poly))
        patches.sort(key=lambda t: -t[0])
        self.hill = VGroup(*[p for _, p in patches])

        corner = np.array([-0.35, -0.35])
        east_arrow = Arrow(proj(*corner, 0), proj(corner[0] + 1.3, corner[1], 0),
                           buff=0, stroke_width=3, color=GREY_B,
                           max_tip_length_to_length_ratio=0.18)
        north_arrow = Arrow(proj(*corner, 0), proj(corner[0], corner[1] + 1.3, 0),
                            buff=0, stroke_width=3, color=GREY_B,
                            max_tip_length_to_length_ratio=0.18)
        east_lab = Text("east", font=TEXT_FONT, font_size=18, color=GREY_B
                        ).next_to(east_arrow.get_end(), DOWN, buff=0.08)
        north_lab = Text("north", font=TEXT_FONT, font_size=18, color=GREY_B
                         ).next_to(north_arrow.get_end(), UP, buff=0.06)
        self.compass3d = VGroup(east_arrow, north_arrow, east_lab, north_lab)

        # the map: contour bands in the surface's colours
        square = Square(side_length=MAP_SIDE).move_to(MAP_CENTER)
        bands = [square.copy().set_fill(height_color(0.12), 1).set_stroke(width=0)]
        for level in (0.25, 0.5, 0.9, 1.4, 1.85):
            r = SIG * np.sqrt(2 * np.log(H / level))
            disc = Circle(radius=r * MS).move_to(on_map(XC, YC))
            band = Intersection(disc, square)
            band.set_fill(height_color(level + 0.18), 1)
            band.set_stroke(BLACK, 1, opacity=0.35)
            bands.append(band)
        frame = square.copy().set_fill(opacity=0).set_stroke(GREY_C, 2)
        m0 = on_map(0.3, DOM - 1.0)
        map_compass = VGroup(
            Arrow(m0, m0 + RIGHT * 0.65, buff=0, stroke_width=3, color=GREY_A,
                  max_tip_length_to_length_ratio=0.25),
            Arrow(m0, m0 + UP * 0.65, buff=0, stroke_width=3, color=GREY_A,
                  max_tip_length_to_length_ratio=0.25),
        )
        map_compass.add(
            Text("E", font=TEXT_FONT, font_size=16, color=GREY_A).next_to(map_compass[0], RIGHT, buff=0.06),
            Text("N", font=TEXT_FONT, font_size=16, color=GREY_A).next_to(map_compass[1], UP, buff=0.04),
        )
        self.map = VGroup(*bands, frame, map_compass)

        tag3d = Text("the hill", font=TEXT_FONT, font_size=20, color=MUTED_COLOR
                     ).move_to([PANEL3D[0], 2.72, 0])
        tagmap = Text("seen from above", font=TEXT_FONT, font_size=20, color=MUTED_COLOR
                      ).next_to(square, UP, buff=0.14)
        self.tags = VGroup(tag3d, tagmap)

        title = Text("Why do the rates add?", font=TEXT_FONT, font_size=40)
        title.move_to([0, 3.42, 0])
        self.play(FadeIn(title), run_time=0.8)
        self.play(FadeIn(self.hill, lag_ratio=0.0), FadeIn(self.compass3d),
                  FadeIn(self.map), FadeIn(self.tags), run_time=1.6)
        self.caption("A hiker walks on a hill w = f(x, y): x measures east, y measures north.")
        self.wait(1.4)

        # the points A, B, C on both panels
        self.marks = VGroup()
        for name, p, direction in (("A", A, DL), ("B", B, DR), ("C", C, UR)):
            d3 = Dot(on_hill(*p, 0.02), radius=0.06, color=WHITE)
            l3 = mtex(name, font_size=30).next_to(d3, direction, buff=0.08)
            l3.add_background_rectangle(opacity=0.55, buff=0.04)
            d2 = Dot(on_map(*p), radius=0.06, color=WHITE)
            l2 = mtex(name, font_size=30).next_to(d2, direction, buff=0.08)
            l2.add_background_rectangle(opacity=0.55, buff=0.04)
            self.marks.add(VGroup(d3, l3, d2, l2))
        self.play(FadeOut(title), FadeIn(self.marks[0]), FadeIn(self.marks[2]),
                  run_time=0.8)

    # ------------------------------------------------------------------
    # act 2: two hikers, two routes, one climb
    def race(self):
        self.caption("Two hikers leave A together: one walks straight to C, "
                     "the other goes east to B, then north to C.")
        self.play(FadeIn(self.marks[1]), run_time=0.5)

        s = ValueTracker(0.0)
        w0 = f(*A)

        def pos1():
            return A + (C - A) * s.get_value()

        def pos2():
            return two_leg_point(A, B, C, s.get_value())

        # trails on the hill and on the map
        def trail1_3d():
            q = pos1()
            return surface_path(segment(A, q, 30), STRAIGHT_PATH, 4, dashed=True)

        def trail2_3d():
            q = pos2()
            if s.get_value() * 2.0 <= STEP[0] or np.allclose(q, A):
                return surface_path(segment(A, q, 30), EAST_COLOR, 6)
            return VGroup(surface_path(segment(A, B, 30), EAST_COLOR, 6),
                          surface_path(segment(B, q, 30), NORTH_COLOR, 6))

        def trail1_map():
            q = pos1()
            if np.allclose(q, A):
                return VGroup()
            return DashedLine(on_map(*A), on_map(*q), color=STRAIGHT_PATH,
                              stroke_width=4, dash_length=0.1)

        def trail2_map():
            q = pos2()
            if np.allclose(q, A):
                return VGroup()
            if q[1] <= A[1] + 1e-9:
                return Line(on_map(*A), on_map(*q), color=EAST_COLOR, stroke_width=6)
            return VGroup(Line(on_map(*A), on_map(*B), color=EAST_COLOR, stroke_width=6),
                          Line(on_map(*B), on_map(*q), color=NORTH_COLOR, stroke_width=6))

        t1, t2 = always_redraw(trail1_3d), always_redraw(trail2_3d)
        m1, m2 = always_redraw(trail1_map), always_redraw(trail2_map)
        h1 = always_redraw(lambda: Dot(on_hill(*pos1(), 0.03), radius=0.11,
                                       color=STRAIGHT_COLOR, stroke_color=WHITE,
                                       stroke_width=2))
        h2 = always_redraw(lambda: Dot(on_hill(*pos2(), 0.03), radius=0.11,
                                       color=TWO_LEG_COLOR, stroke_color=WHITE,
                                       stroke_width=2))
        d1 = always_redraw(lambda: Dot(on_map(*pos1()), radius=0.1,
                                       color=STRAIGHT_COLOR, stroke_color=WHITE,
                                       stroke_width=2))
        d2 = always_redraw(lambda: Dot(on_map(*pos2()), radius=0.1,
                                       color=TWO_LEG_COLOR, stroke_color=WHITE,
                                       stroke_width=2))

        # the gauges: height gained so far
        gx1, gx2 = GAUGE_X
        base_line = Line([gx1 - 0.45, GAUGE_BASE, 0], [gx2 + 0.45, GAUGE_BASE, 0],
                         color=GREY_C, stroke_width=2)
        g_title = Text("height gained", font=TEXT_FONT, font_size=17,
                       color=MUTED_COLOR).move_to([(gx1 + gx2) / 2, 1.85, 0])
        g_lab1 = Text("straight", font=TEXT_FONT, font_size=14, color=STRAIGHT_COLOR
                      ).next_to([gx1, GAUGE_BASE, 0], DOWN, buff=0.14)
        g_lab2 = Text("two legs", font=TEXT_FONT, font_size=14,
                      color=TWO_LEG_COLOR
                      ).next_to([gx2, GAUGE_BASE, 0], DOWN, buff=0.14)
        self.gauge_frame = VGroup(base_line, g_title, g_lab1, g_lab2)

        gauge1 = always_redraw(lambda: bar(gx1, GAUGE_BASE,
                                           (f(*pos1()) - w0) * GAUGE_SCALE,
                                           STRAIGHT_COLOR))

        def gauge2_bars():
            q = pos2()
            east_rise = f(*(q if q[1] <= A[1] + 1e-9 else B)) - w0
            north_rise = 0.0 if q[1] <= A[1] + 1e-9 else f(*q) - f(*B)
            b1 = bar(gx2, GAUGE_BASE, east_rise * GAUGE_SCALE, EAST_COLOR)
            b2 = bar(gx2, GAUGE_BASE + east_rise * GAUGE_SCALE,
                     north_rise * GAUGE_SCALE, NORTH_COLOR)
            return VGroup(b1, b2)

        gauge2 = always_redraw(gauge2_bars)

        self.play(FadeIn(self.gauge_frame), run_time=0.6)
        self.add(t1, t2, m1, m2, gauge1, gauge2, h1, h2, d1, d2)
        self.wait(0.4)
        self.play(s.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        self.wait(0.5)

        dw = (f(*C) - w0) * GAUGE_SCALE
        level = DashedLine([gx1 - 0.4, GAUGE_BASE + dw, 0], [gx2 + 0.4, GAUGE_BASE + dw, 0],
                           color=WHITE, stroke_width=2, dash_length=0.08)
        dw_lab = mtex(r"\Delta w", font_size=32).next_to(level, UP, buff=0.08)
        self.caption("They arrive together, and they have climbed exactly the same amount.")
        self.play(Create(level), FadeIn(dw_lab), run_time=0.9)
        self.wait(1.0)
        self.caption("The change in height depends only on where you start and where you end.")
        self.wait(2.2)

        # freeze everything at the finish
        for mob in (t1, t2, m1, m2, gauge1, gauge2, h1, h2, d1, d2):
            mob.clear_updaters()
        self.race_mobs = VGroup(t1, t2, m1, m2, h1, h2, d1, d2)
        self.straight_mobs = VGroup(t1, m1, h1, d1)
        self.gauge1, self.gauge2 = gauge1, gauge2
        self.level = VGroup(level, dw_lab)

    # ------------------------------------------------------------------
    # act 3: the east leg
    def east_leg(self):
        self.play(self.straight_mobs.animate.set_opacity(0.25),
                  self.gauge1.animate.set_opacity(0.35), run_time=0.7)
        self.caption("So follow the second hiker one leg at a time. First east, from A to B:")
        rise = f(*B) - f(*A)
        slope = fx(*A)
        tip = A + np.array([STEP[0], 0.0])
        tangent = DashedLine(on_hill(*A, 0.02),
                             proj(tip[0], tip[1], f(*A) + slope * STEP[0] + 0.02),
                             color=WHITE, stroke_width=3, dash_length=0.09)
        tangent_lab = mtex(r"\text{slope } f_x", font_size=26, color=EAST_COLOR)
        tangent_lab.add_background_rectangle(opacity=0.7, buff=0.05)
        tangent_lab.next_to(on_hill(*(A + B) / 2), DOWN, buff=0.3)
        dx_arrow = Arrow(on_map(*A), on_map(*B), buff=0.06, color=EAST_COLOR,
                         stroke_width=5, max_tip_length_to_length_ratio=0.12)
        dx_lab = mtex(r"\Delta x", font_size=30, color=EAST_COLOR
                      ).next_to(dx_arrow, DOWN, buff=0.1)
        dx_lab.add_background_rectangle(opacity=0.6, buff=0.04)
        eq = self.equation(r"\text{rise from } A \text{ to } B", r"\approx",
                           r"f_x\,\Delta x", colors={2: EAST_COLOR})
        self.show_equation(eq)
        self.play(Create(tangent), FadeIn(tangent_lab), GrowArrow(dx_arrow),
                  FadeIn(dx_lab), run_time=1.3)
        self.play(Indicate(self.gauge2[0], color=EAST_COLOR, scale_factor=1.12),
                  run_time=1.0)
        self.caption("Over a short eastward step, the eastward slope f_x gives a rise of about f_x Δx.")
        self.wait(2.0)
        self.east_bits = VGroup(tangent, tangent_lab, dx_arrow, dx_lab)

    # ------------------------------------------------------------------
    # act 4: the north leg
    def north_leg(self):
        slope = fy(*B)
        tip = B + np.array([0.0, STEP[1]])
        tangent = DashedLine(on_hill(*B, 0.02),
                             proj(tip[0], tip[1], f(*B) + slope * STEP[1] + 0.02),
                             color=WHITE, stroke_width=3, dash_length=0.09)
        tangent_lab = mtex(r"\text{slope } f_y", font_size=26, color=NORTH_COLOR)
        tangent_lab.add_background_rectangle(opacity=0.7, buff=0.05)
        tangent_lab.next_to(on_hill(*(B + C) / 2), RIGHT, buff=0.35)
        dy_arrow = Arrow(on_map(*B), on_map(*C), buff=0.06, color=NORTH_COLOR,
                         stroke_width=5, max_tip_length_to_length_ratio=0.18)
        dy_lab = mtex(r"\Delta y", font_size=30, color=NORTH_COLOR
                      ).next_to(dy_arrow, RIGHT, buff=0.1)
        dy_lab.add_background_rectangle(opacity=0.6, buff=0.04)
        self.caption("Then north, from B to C: the northward slope f_y gives a rise of about f_y Δy.")
        eq = self.equation(r"\text{rise from } B \text{ to } C", r"\approx",
                           r"f_y\,\Delta y", colors={2: NORTH_COLOR})
        self.show_equation(eq, transform=False)
        self.play(Create(tangent), FadeIn(tangent_lab), GrowArrow(dy_arrow),
                  FadeIn(dy_lab), run_time=1.3)
        self.play(Indicate(self.gauge2[1], color=NORTH_COLOR, scale_factor=1.12),
                  run_time=1.0)
        self.wait(1.2)
        eq2 = self.equation(r"\Delta w", r"\approx", r"f_x\,\Delta x", "+",
                            r"f_y\,\Delta y",
                            colors={0: STRAIGHT_COLOR, 2: EAST_COLOR, 4: NORTH_COLOR})
        self.show_equation(eq2, transform=False)
        self.play(self.gauge1.animate.set_opacity(1.0), Indicate(self.level, scale_factor=1.1),
                  run_time=1.0)
        self.caption("The two rises stack up to the whole climb Δw: the contributions add.")
        self.wait(2.4)
        self.north_bits = VGroup(tangent, tangent_lab, dy_arrow, dy_lab)

    # ------------------------------------------------------------------
    # act 5: shrink the step
    def shrink(self):
        self.play(FadeOut(self.race_mobs), FadeOut(self.east_bits),
                  FadeOut(self.north_bits), FadeOut(self.gauge1), FadeOut(self.gauge2),
                  FadeOut(self.level), FadeOut(self.marks[1]), FadeOut(self.marks[2]),
                  FadeOut(self.gauge_frame), run_time=0.8)
        self.caption("Now let the walk last a time Δt, and make Δt smaller and smaller.")

        h = ValueTracker(1.0)
        w0 = f(*A)

        def pts():
            k = h.get_value()
            return A + np.array([STEP[0] * k, 0.0]), A + STEP * k

        def legs3d():
            b, c = pts()
            return VGroup(surface_path(segment(A, b, 20), EAST_COLOR, 6),
                          surface_path(segment(b, c, 20), NORTH_COLOR, 6),
                          surface_path(segment(A, c, 20), STRAIGHT_PATH, 4, dashed=True))

        def legs_map():
            b, c = pts()
            return VGroup(Line(on_map(*A), on_map(*b), color=EAST_COLOR, stroke_width=6),
                          Line(on_map(*b), on_map(*c), color=NORTH_COLOR, stroke_width=6),
                          DashedLine(on_map(*A), on_map(*c), color=STRAIGHT_PATH,
                                     stroke_width=4, dash_length=0.08),
                          Dot(on_map(*c), radius=0.06, color=WHITE))

        gx1, gx2 = GAUGE_X
        rate_scale = GAUGE_SCALE

        def true_rate():
            b, c = pts()
            return (f(*c) - w0) / h.get_value()

        def estimate_parts():
            b, c = pts()
            return fx(*A) * STEP[0], fy(*b) * STEP[1]

        def rate_bars():
            e1, e2 = estimate_parts()
            return VGroup(
                bar(gx1, GAUGE_BASE, true_rate() * rate_scale, STRAIGHT_COLOR),
                bar(gx2, GAUGE_BASE, e1 * rate_scale, EAST_COLOR),
                bar(gx2, GAUGE_BASE + e1 * rate_scale, e2 * rate_scale, NORTH_COLOR),
            )

        base_line = Line([gx1 - 0.45, GAUGE_BASE, 0], [gx2 + 0.45, GAUGE_BASE, 0],
                         color=GREY_C, stroke_width=2)
        r_title = Text("rate of climbing", font=TEXT_FONT, font_size=17,
                       color=MUTED_COLOR).move_to([(gx1 + gx2) / 2, 1.85, 0])
        r_lab1 = Text("true", font=TEXT_FONT, font_size=14, color=STRAIGHT_COLOR
                      ).next_to([gx1, GAUGE_BASE, 0], DOWN, buff=0.14)
        r_lab2 = Text("slopes", font=TEXT_FONT, font_size=14,
                      color=TWO_LEG_COLOR).next_to([gx2, GAUGE_BASE, 0], DOWN, buff=0.14)
        clock = VGroup(mtex(r"\Delta t =", font_size=30),
                       DecimalNumber(1.0, num_decimal_places=2, font_size=30)
                       ).arrange(RIGHT, buff=0.15)
        clock.move_to([(gx1 + gx2) / 2, 2.45, 0])
        clock[1].add_updater(lambda m: m.set_value(h.get_value()))

        L3, LM, R = always_redraw(legs3d), always_redraw(legs_map), always_redraw(rate_bars)
        eq = self.equation(r"\frac{\Delta w}{\Delta t}", r"\approx",
                           r"f_x\,\frac{\Delta x}{\Delta t}", "+",
                           r"f_y\,\frac{\Delta y}{\Delta t}",
                           colors={0: STRAIGHT_COLOR, 2: EAST_COLOR, 4: NORTH_COLOR})
        self.show_equation(eq, transform=False)
        self.play(FadeIn(L3), FadeIn(LM), FadeIn(R), FadeIn(base_line), FadeIn(r_title),
                  FadeIn(r_lab1), FadeIn(r_lab2), FadeIn(clock), run_time=1.0)
        self.wait(0.8)
        self.caption("Dividing by Δt compares rates. As Δt shrinks, the slopes along the "
                     "two legs settle down and the two rates agree.")
        self.play(h.animate.set_value(0.04), run_time=6.0, rate_func=smooth)
        self.wait(1.0)
        final = self.equation(r"\frac{dw}{dt}", "=",
                              r"\frac{\partial f}{\partial x}\frac{dx}{dt}", "+",
                              r"\frac{\partial f}{\partial y}\frac{dy}{dt}",
                              colors={0: STRAIGHT_COLOR, 2: EAST_COLOR, 4: NORTH_COLOR})
        box = SurroundingRectangle(final, color=WHITE, buff=0.12, stroke_width=2)
        self.play(FadeOut(self.current_equation), FadeIn(final), Create(box), run_time=1.1)
        self.current_equation = VGroup(final, box)
        self.caption("In the limit Δt → 0 this is the chain rule.")
        self.wait(2.5)
        clock[1].clear_updaters()
        for mob in (L3, LM, R):
            mob.clear_updaters()
        self.play(FadeOut(VGroup(L3, LM, R, base_line, r_title, r_lab1, r_lab2, clock)),
                  run_time=0.7)

    # ------------------------------------------------------------------
    def conclusion(self):
        final = self.equation(r"\frac{dw}{dt}", "=",
                              r"\frac{\partial f}{\partial x}\frac{dx}{dt}", "+",
                              r"\frac{\partial f}{\partial y}\frac{dy}{dt}",
                              colors={0: STRAIGHT_COLOR, 2: EAST_COLOR, 4: NORTH_COLOR})
        box = SurroundingRectangle(final, color=WHITE, buff=0.12, stroke_width=2)
        self.play(FadeOut(self.current_equation), FadeIn(final), Create(box), run_time=0.9)
        self.current_equation = VGroup(final, box)
        self.caption("Each coordinate motion contributes to the rate of climbing, "
                     "and those contributions add.")
        self.wait(3.5)
