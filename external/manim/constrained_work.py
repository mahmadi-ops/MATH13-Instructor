"""Work done by F = <2,-1,3> on a particle constrained to the plane
2x - y + 2z = 0 (the exercise dp-constrained-work: Assignment 4 in this
book, Assignment 6 in the student book M13-Skeletal-Notes).

Renders to assets/videos/constrained_work_1080p.mp4:

    manim -r 1920,1080 --fps 30 --disable_caching -o constrained_work_1080p \
        assets/manim/constrained_work.py ConstrainedWork

Design notes:

* The frame.  As in M13-Skeletal-Notes' normal_vector_3d.py the scene is
  drawn in a rotated coordinate system: its z-axis is the unit normal n/3,
  its x-axis is the direction of F_par = (1/9)<-4,2,5>, and its y-axis is
  n/3 x F_par/|F_par|, which works out to -<1,2,0>/sqrt5.  So the plane
  is the horizontal plane, F_par points along +x and the angle theta
  between d and F_par is simply the polar angle of d.  In these coordinates
  F = (sqrt5/3, 0, 11/3), proj_n F = (0, 0, 11/3), F_par = (sqrt5/3, 0, 0).
  No axes are drawn, so the rotation changes nothing the viewer is told.

* Scale.  F is almost normal to the plane (|proj_n F| = 11/3 against
  |F_par| = sqrt5/3), so the decomposition is shown zoomed out, to scale,
  and F, proj_n F and n are then faded before the camera zooms in on the
  plane, where the rest of the story happens.

* The point of the video.  A fan of unit vectors shows that the allowed
  directions d are all the unit vectors in the plane, infinitely many.
  Then a single d turns once around the circle while the work
  W = F . d = F_par . d = |F_par| cos theta is shown three ways: as a
  readout, as a dot running along the graph of W against theta, and as the
  shadow of F_par on the line of d.  It then turns back, theta falling from
  150 degrees to 0, and stops where d points along F_par and W reaches its
  largest value sqrt5/3.  About 95 s in all.

* Arrowheads are flat triangles, not Cones.  The cairo 3D camera draws
  every shaded surface before every flat mobject, so a cone head would be
  painted over by the translucent plane; flat heads keep the drawing order
  the order in which things are added.
"""

from manim import *
import numpy as np

S = 2.0                                  # scene units per unit of length
F_PAR = np.sqrt(5) / 3                   # |F_par|, about 0.745
PROJ = 11 / 3                            # |proj_n F|
O = ORIGIN
F_TIP = S * np.array([F_PAR, 0, PROJ])
PROJ_TIP = S * np.array([0, 0, PROJ])
PAR_TIP = S * np.array([F_PAR, 0, 0])
R_DISK = 2.8                             # radius of the drawn piece of plane
N_TAIL = np.array([-1.7, 1.6, 0])        # n is drawn off to the side
N_LEN = 1.3

# camera for the decomposition (zoomed out) and for the plane (zoomed in)
PHI1, THETA1, ZOOM1 = 64 * DEGREES, -78 * DEGREES, 0.4
PHI2, ZOOM2 = 52 * DEGREES, 1.0

C_PLANE = "#56B4E9"
C_F = "#E69F00"
C_PROJ = "#D55E00"
C_PAR = "#2BD99F"
C_D = "#F0E442"
C_W = "#CC79A7"


def cam_dir(phi, theta):
    return np.array([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta), np.cos(phi)])


def flat_arrow(start, end, color, width=6, head=0.3, side=None,
               opacity=1.0):
    """Line plus a flat triangular head.  `side` is the direction the head
    spreads in; by default the in-plane perpendicular, right for arrows
    lying in the plane."""
    d = end - start
    u = d / np.linalg.norm(d)
    if side is None:
        side = np.cross(OUT, u)
    side = side / np.linalg.norm(side)
    base = end - head * u
    shaft = Line(start, base, color=color, stroke_width=width,
                 stroke_opacity=opacity)
    tip = Polygon(end, base + 0.42 * head * side, base - 0.42 * head * side,
                  color=color, fill_color=color, fill_opacity=opacity,
                  stroke_width=0)
    return VGroup(shaft, tip)


def facing(u, phi=PHI1, theta=THETA1):
    """Head direction for an arrow along u that faces the camera."""
    return np.cross(u, cam_dir(phi, theta))


class ConstrainedWork(ThreeDScene):
    def construct(self):
        tmpl = TexTemplate()
        tmpl.add_to_preamble(r"\usepackage{amsmath,amssymb}")

        def tex(s, **kw):
            return MathTex(s, tex_template=tmpl, **kw)

        def fixed(*mobs):
            self.add_fixed_in_frame_mobjects(*mobs)
            self.remove(*mobs)

        def label3d(s, pos, color=WHITE, scale=0.8):
            m = tex(s, color=color).scale(scale).move_to(pos)
            self.add_fixed_orientation_mobjects(m)
            self.remove(m)
            return m

        self.set_camera_orientation(phi=PHI1, theta=THETA1, zoom=ZOOM1)

        # -- title and the given data, fixed to the frame ---------------------
        title = Tex(r"Which direction does the most work?",
                    tex_template=tmpl).scale(0.6).to_corner(UL, buff=0.35)
        given = VGroup(
            tex(r"\mathbf F = \langle 2,\,-1,\,3\rangle", color=C_F),
            Tex(r"plane: $2x - y + 2z = 0$", tex_template=tmpl,
                color=C_PLANE),
            tex(r"\mathbf n = \langle 2,\,-1,\,2\rangle"),
        ).scale(0.6).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        given.next_to(title, DOWN, buff=0.45).to_edge(LEFT, buff=0.4)
        split = VGroup(
            tex(r"\mathbf F = \operatorname{proj}_{\mathbf n}\mathbf F"
                r" + \mathbf F_{\parallel}"),
            tex(r"\mathbf F_{\parallel} = \tfrac19\langle -4,\,2,\,5\rangle",
                color=C_PAR),
            tex(r"\|\mathbf F_{\parallel}\| = \tfrac{\sqrt5}{3}"
                r"\approx 0.745", color=C_PAR),
        ).scale(0.6).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        split.next_to(given, DOWN, buff=0.45, aligned_edge=LEFT)
        fixed(title, given, split)

        caption = [None]

        def say(s, run_time=1.2):
            """Replace the caption along the bottom of the frame."""
            new = Tex(s, tex_template=tmpl).scale(0.6)
            new.to_edge(DOWN, buff=0.4)
            fixed(new)
            anims = [FadeIn(new)]
            if caption[0] is not None:
                anims.append(FadeOut(caption[0]))
            self.play(*anims, run_time=run_time)
            caption[0] = new

        self.play(Write(title), run_time=1.5)

        # -- the plane, F and n ----------------------------------------------
        disk = Circle(radius=R_DISK, color=C_PLANE, stroke_width=2,
                      fill_color=C_PLANE, fill_opacity=0.22)
        origin = Dot(O, radius=0.06, color=WHITE)
        o_lab = label3d("O", np.array([-0.3, -0.3, 0]))
        self.play(FadeIn(disk), Write(given[1]), run_time=2)
        self.play(FadeIn(origin), FadeIn(o_lab), run_time=0.8)

        n_arrow = flat_arrow(N_TAIL, N_TAIL + N_LEN * OUT, WHITE, width=4,
                             side=facing(OUT))
        n_lab = label3d(r"\mathbf n", N_TAIL + (N_LEN + 0.45) * OUT)
        self.play(Create(n_arrow), FadeIn(n_lab), Write(given[2]),
                  run_time=1.5)

        f_arrow = flat_arrow(O, F_TIP, C_F, width=6, head=0.45,
                             side=facing(F_TIP))
        f_lab = label3d(r"\mathbf F", F_TIP + np.array([0.6, 0, 0.1]),
                        color=C_F, scale=1.0)
        self.play(GrowFromPoint(f_arrow, O), Write(given[0]), run_time=2)
        self.play(FadeIn(f_lab), run_time=0.6)
        self.wait(1)

        # -- the decomposition -----------------------------------------------
        say(r"Split $\mathbf F$ into a piece along $\mathbf n$ and a piece "
            r"$\mathbf F_{\parallel}$ lying in the plane.")
        proj_arrow = flat_arrow(O, PROJ_TIP, C_PROJ, width=6, head=0.45,
                                side=facing(OUT))
        proj_lab = label3d(r"\operatorname{proj}_{\mathbf n}\mathbf F",
                           PROJ_TIP + np.array([-2.4, 0, 0.1]), color=C_PROJ,
                           scale=1.0)
        par_arrow = flat_arrow(O, PAR_TIP, C_PAR, width=8, head=0.4)
        par_lab = label3d(r"\mathbf F_{\parallel}",
                          PAR_TIP + np.array([1.3, -0.2, 0]), color=C_PAR,
                          scale=0.9)
        dashes = VGroup(DashedLine(PROJ_TIP, F_TIP, color=GREY_B),
                        DashedLine(PAR_TIP, F_TIP, color=GREY_B))
        self.play(GrowFromPoint(proj_arrow, O), FadeIn(proj_lab),
                  run_time=1.8)
        self.play(GrowFromPoint(par_arrow, O), FadeIn(par_lab),
                  Create(dashes), run_time=1.8)
        self.play(Write(split[0]), run_time=1.5)
        self.play(Write(split[1]), Write(split[2]), run_time=2)
        self.wait(1.5)

        say(r"$\operatorname{proj}_{\mathbf n}\mathbf F$ is perpendicular to "
            r"every direction in the plane, so it does no work:\\"
            r"$W = \mathbf F\cdot\mathbf d = \mathbf F_{\parallel}\cdot"
            r"\mathbf d$ for every allowed direction $\mathbf d$.")
        self.wait(4)

        # -- zoom in on the plane --------------------------------------------
        self.play(FadeOut(VGroup(f_arrow, proj_arrow, dashes, n_arrow)),
                  FadeOut(VGroup(f_lab, proj_lab, n_lab, o_lab)),
                  FadeOut(caption[0]), run_time=1.5)
        caption[0] = None
        # F_par's label moves outside the circle on diameter O F_par, where
        # the shadow and the dashed drop of the next part live
        self.move_camera(phi=PHI2, zoom=ZOOM2, run_time=3, added_anims=[
            par_lab.animate.move_to(np.array([1.55, -0.5, 0]))])
        self.wait(0.5)

        # -- infinitely many directions --------------------------------------
        say(r"Every unit vector $\mathbf d$ lying in the plane is an allowed "
            r"direction:\\ there are infinitely many of them.")

        def unit(a):
            return S * np.array([np.cos(a), np.sin(a), 0])
        fan = VGroup(*[flat_arrow(O, unit(a), C_D, width=3, head=0.22,
                                  opacity=0.55)
                       for a in np.arange(0, TAU, TAU / 12)])
        fan2 = VGroup(*[flat_arrow(O, unit(a), C_D, width=3, head=0.22,
                                   opacity=0.55)
                        for a in np.arange(0, TAU, TAU / 36)
                        if round(np.degrees(a)) % 30])
        fan3 = VGroup(*[Line(O, unit(a), color=C_D, stroke_width=1.5,
                             stroke_opacity=0.4)
                        for a in np.arange(0, TAU, TAU / 180)
                        if round(np.degrees(a)) % 10])
        self.play(LaggedStart(*[GrowFromPoint(a, O) for a in fan],
                              lag_ratio=0.15), run_time=2.5)
        self.play(LaggedStart(*[GrowFromPoint(a, O) for a in fan2],
                              lag_ratio=0.05), run_time=2)
        self.play(LaggedStart(*[GrowFromPoint(a, O) for a in fan3],
                              lag_ratio=0.01), run_time=2)
        circle = Circle(radius=S, color=C_D, stroke_width=3)
        c_lab = label3d(r"\|\mathbf d\| = 1", (S + 0.55) * np.array(
            [np.cos(-1.75), np.sin(-1.75), 0]), color=C_D, scale=0.7)
        self.play(Create(circle), FadeIn(c_lab), run_time=2)
        self.wait(1.5)
        fans = VGroup(fan, fan2, fan3)
        self.play(fans.animate.set_opacity(0.1), FadeOut(c_lab),
                  run_time=1.2)

        # -- one direction d, turning ----------------------------------------
        t = ValueTracker(150.0)                # polar angle of d, in degrees

        def signed():
            return (t.get_value() + 180.0) % 360.0 - 180.0

        def u_d():
            a = np.radians(t.get_value())
            return np.array([np.cos(a), np.sin(a), 0.0])

        def work():
            return F_PAR * np.cos(np.radians(t.get_value()))

        d_arrow = always_redraw(
            lambda: flat_arrow(O, S * u_d(), C_D, width=6, head=0.32))
        d_lab = label3d(r"\mathbf d", (S + 0.4) * u_d(), color=C_D,
                        scale=1.0)
        d_lab.add_updater(lambda m: m.move_to((S + 0.4) * u_d()))
        arc = always_redraw(lambda: Arc(
            radius=0.6, start_angle=0, angle=np.radians(signed()),
            color=WHITE, stroke_width=3))
        th_lab = label3d(r"\theta", np.zeros(3), scale=0.9)

        def place_theta(m):
            h = np.radians(signed()) / 2
            m.move_to(0.95 * np.array([np.cos(h), np.sin(h), 0]))
            m.set_opacity(1 if abs(signed()) > 16 else 0)
        th_lab.add_updater(place_theta)
        place_theta(th_lab)

        def foot():
            return S * work() * u_d()

        def segment(a, b, maker):
            """maker(a, b), or nothing while a and b (nearly) coincide."""
            if np.linalg.norm(b - a) < 0.03:
                return VMobject()
            return maker(a, b)

        d_line = always_redraw(lambda: DashedLine(
            -S * u_d(), S * u_d(), color=C_D, stroke_width=1.5,
            stroke_opacity=0.6, dash_length=0.1))
        shadow = always_redraw(lambda: segment(O, foot(), lambda a, b: Line(
            a, b, color=C_W, stroke_width=14)))
        drop = always_redraw(lambda: segment(
            PAR_TIP, foot(), lambda a, b: DashedLine(
                a, b, color=GREY_B, stroke_width=2, dash_length=0.08)))

        # live readouts and the graph of W against theta
        th_read = VGroup(tex(r"\theta ="),
                         DecimalNumber(abs(signed()), num_decimal_places=0,
                                       unit=r"^\circ"))
        w_read = VGroup(tex(r"W = \|\mathbf F_{\parallel}\|\cos\theta =",
                            color=C_W),
                        DecimalNumber(work(), num_decimal_places=3,
                                      color=C_W))
        for r in (th_read, w_read):
            r.arrange(RIGHT, buff=0.15)
        readout = VGroup(th_read, w_read).scale(0.62)
        readout.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        readout.to_corner(UR, buff=0.4).shift(DOWN * 0.35)
        fixed(readout)

        def refresh(num, value):
            def upd(m):
                m.set_value(value())
                self.renderer.camera.add_fixed_in_frame_mobjects(m)
            num.add_updater(upd)
        refresh(th_read[1], lambda: abs(signed()))
        refresh(w_read[1], work)

        axes = Axes(x_range=[0, 180, 45], y_range=[-0.8, 0.8, 0.4],
                    x_length=3.0, y_length=2.4, tips=False,
                    axis_config={"stroke_width": 2, "tick_size": 0.05})
        axes.move_to(np.array([5.0, -0.55, 0]))
        curve = axes.plot(lambda x: F_PAR * np.cos(np.radians(x)),
                          x_range=[0, 180], color=C_W, stroke_width=3)
        # above the axis, where the curve (falling through 90) is not
        ticks = VGroup(
            tex(r"90^\circ").scale(0.45).next_to(axes.c2p(90, 0), UR,
                                                  buff=0.06),
            tex(r"180^\circ").scale(0.45).next_to(axes.c2p(180, 0), UP,
                                                   buff=0.1))
        x_lab = tex(r"\theta").scale(0.55).next_to(axes.c2p(180, 0), RIGHT,
                                                    buff=0.12)
        y_lab = tex("W", color=C_W).scale(0.55).next_to(
            axes.c2p(0, 0.8), UP, buff=0.1)
        top = DashedLine(axes.c2p(0, F_PAR), axes.c2p(180, F_PAR),
                         color=GREY_B, stroke_width=1.5, dash_length=0.06)
        top_lab = tex(r"\|\mathbf F_{\parallel}\|").scale(0.45).next_to(
            top, UP, buff=0.06).align_to(top, RIGHT)
        graph = VGroup(axes, curve, ticks, x_lab, y_lab, top, top_lab)
        g_dot = Dot(axes.c2p(abs(signed()), work()), radius=0.07,
                    color=C_D)
        g_dot.add_updater(
            lambda m: m.move_to(axes.c2p(abs(signed()), work())))
        fixed(graph, g_dot)

        say(r"$W = \mathbf F_{\parallel}\cdot\mathbf d = "
            r"\|\mathbf F_{\parallel}\|\,\|\mathbf d\|\cos\theta = "
            r"\|\mathbf F_{\parallel}\|\cos\theta$,\\ where $\theta$ is "
            r"the angle between $\mathbf d$ and $\mathbf F_{\parallel}$.\\"
            r"It is the signed length of the shadow of "
            r"$\mathbf F_{\parallel}$ on the line of $\mathbf d$.")
        self.play(Create(d_line), GrowFromPoint(d_arrow, O), FadeIn(d_lab),
                  run_time=1.5)
        self.play(Create(arc), FadeIn(th_lab), FadeIn(readout), run_time=1.2)
        # F_par stays on top of the shadow and of d
        self.add(shadow, drop)
        self.bring_to_front(par_arrow)
        self.play(FadeIn(graph), FadeIn(g_dot), run_time=1.5)
        self.wait(3)

        # a full turn: theta runs 150 -> 180 -> 90 -> 0 -> 90 -> 150
        self.play(t.animate.set_value(510.0), run_time=18, rate_func=linear)
        self.wait(1)

        # ... and turn back to the maximum, theta falling 150 -> 0
        say(r"$W$ is largest when $\cos\theta = 1$, that is when "
            r"$\theta = 0$: $\mathbf d$ points the same way as "
            r"$\mathbf F_{\parallel}$.")
        self.play(t.animate.set_value(360.0), run_time=6,
                  rate_func=smooth)
        self.wait(1)
        best = Circle(radius=0.16, color=C_D, stroke_width=3).move_to(
            axes.c2p(0, F_PAR))
        fixed(best)
        self.play(Create(best), Indicate(par_arrow, color=C_PAR,
                                         scale_factor=1.15), run_time=1.5)
        self.wait(2)

        say(r"$\displaystyle \mathbf d_{\max} = "
            r"\frac{\mathbf F_{\parallel}}{\|\mathbf F_{\parallel}\|} = "
            r"\frac{\langle -4,\,2,\,5\rangle}{3\sqrt5},\qquad "
            r"W_{\max} = \|\mathbf F_{\parallel}\| = \frac{\sqrt5}{3}"
            r"\approx 0.745$", run_time=1.5)
        self.wait(3)

        # -- a slow look around ----------------------------------------------
        self.begin_ambient_camera_rotation(rate=0.09)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        self.wait(2)
