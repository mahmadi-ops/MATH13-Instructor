"""Why the gradients of f and g are collinear at a constrained extremum:
the general argument, step by step.

The moving picture behind the paragraph of "Why the Gradients Must Be
Collinear" that compares (eq-lagrange-gradg-perp) and
(eq-lagrange-gradf-perp): both gradients are perpendicular to the unit
tangent T-hat, and in the plane only one line through P is perpendicular
to T-hat, so the gradients lie on it, are collinear, and
grad f = lambda grad g.
No particular f or g is used; the constraint curve is the wavy curve of
fig-lagrange-tangent, c(t) = (1.4 t, 1.1 sin t), with P at t = 0.7, the
point that figure uses.

Renders to assets/videos/lagrange_gradients_perpendicular_manim.mp4:

    manim -qh --disable_caching -o lagrange_gradients_perpendicular_manim \
        assets/manim/lagrange_gradients_perpendicular.py \
        LagrangeGradientsPerpendicular

It is fig-lagrange-perpendicular-video of the completed notes, right after
(eq-lagrange-collinear), and fig-skel-lagrange-perpendicular-video of the
skeletal notes, right after the summary that holds that equation.

Acts:

1. the constraint curve C: g(x,y) = 0, a point P where f has a constrained
   max or min, and the unit tangent T-hat there;
2. grad g . T-hat = 0: g = 0 all along C, so a point moving along C
   sees no change in g and D_T g = grad g . T-hat = 0; grad g is drawn
   crossing the neighbouring level curves g = c and g = -c at right angles;
3. grad f . T-hat = 0: the rate of change of f along C is
   D_T f = grad f . T-hat, and if it were not 0, f would rise one way along
   C and fall the other (a tilted grad f, with its shadow on T-hat, turns
   until the shadow is gone);
4. pure geometry: both gradients are perpendicular to the same T-hat; in
   the plane only one line through P, the normal line, is perpendicular to
   T-hat, so both gradients lie on it and are collinear;
5. collinear means one is a multiple of the other, grad f = lambda grad g: a
   copy lambda grad g is stretched from grad g onto grad f, through 0 since
   here the two point opposite ways (lambda < 0), and then grad f is turned
   round to show lambda > 0;
6. the Lagrange equations grad f = lambda grad g, g(x,y) = 0, component by
   component.

Colours follow the chapter's other Lagrange animation (lagrange_collinear.py):
C and grad g GREEN, grad f RED, T-hat BLUE, right angles and the normal line
YELLOW.
"""

from manim import *
import numpy as np

TEX = TexTemplate(preamble=(
    r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}" "\n"
    r"\AtBeginDocument{\setlength{\textwidth}{1500pt}\setlength{\linewidth}{1500pt}"
    r"\setlength{\hsize}{1500pt}\setlength{\columnwidth}{1500pt}}"))
config.tex_template = TEX
config.background_color = BLACK

CURVE_G = GREEN          # the constraint curve, and grad g
GRAD_F = RED             # grad f
DIR_T = BLUE             # the unit tangent
MARK = YELLOW            # right angles, the normal line, the shadow
GLOSS = GREY_B

# ---- the construction (data units) ------------------------------------------
T0 = 0.7                 # the parameter of P, as in fig-lagrange-tangent


def C(t):
    return np.array([1.4 * t, 1.1 * np.sin(t)])


def unit(v):
    return v / np.linalg.norm(v)


P0 = C(T0)
TH = unit(np.array([1.4, 1.1 * np.cos(T0)]))   # T-hat
NH = np.array([-TH[1], TH[0]])                 # the unit normal, grad g's side
TH_ANGLE = float(np.arctan2(TH[1], TH[0]))

LT, LG, LF = 1.15, 1.0, 1.45   # drawn lengths of T-hat, grad g, grad f
SH = 0.07                      # how far the shadow sits off T-hat
LAM = -LF / LG                 # grad f = LAM grad g in the picture

S = 1.15                       # screen units per data unit
COL_X = 0.62                   # left edge of the text column
TOP_Y = 3.72                   # top of the text column


class LagrangeGradientsPerpendicular(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-2.75, 3.6, 1], y_range=[-1.65, 2.55, 1],
            x_length=6.35 * S, y_length=4.2 * S,
            axis_config={"stroke_width": 2, "include_ticks": False,
                         "stroke_opacity": 0.45, "color": GREY_C},
            tips=False,
        ).move_to([-3.45, 0.45, 0])

        def pt(xy):
            return ax.c2p(xy[0], xy[1])

        def arrow(tail, head, color, width=6, z=5):
            v = np.asarray(head) - np.asarray(tail)
            if np.linalg.norm(v) < 0.03:
                return VMobject()
            return Arrow(pt(tail), pt(head), buff=0, color=color,
                         stroke_width=width, tip_length=0.22,
                         max_tip_length_to_length_ratio=0.45,
                         max_stroke_width_to_length_ratio=40, z_index=z)

        def seg(a, b, color, width, opacity=1.0, z=3):
            if np.linalg.norm(np.asarray(b) - np.asarray(a)) < 0.01:
                return VMobject()
            return Line(pt(a), pt(b), color=color, stroke_width=width,
                        stroke_opacity=opacity, z_index=z)

        def dashed(a, b, color=GREY_B, width=2):
            if np.linalg.norm(np.asarray(b) - np.asarray(a)) < 0.06:
                return VMobject()
            return DashedLine(pt(a), pt(b), color=color, stroke_width=width,
                              dash_length=0.07)

        def square(u, v, d=0.21):
            return Polygon(pt(P0), pt(P0 + d * u), pt(P0 + d * (u + v)),
                           pt(P0 + d * v), color=MARK, stroke_width=3,
                           z_index=4)

        def lines(*rows, size=30, color=WHITE, buff=0.2):
            g = VGroup(*[MathTex(r, font_size=size, color=color) for r in rows])
            g.arrange(DOWN, aligned_edge=LEFT, buff=buff)
            return g

        def head(*rows, size=30):
            g = lines(*rows, size=size)
            g.move_to([COL_X, TOP_Y, 0], aligned_edge=UL)
            return g

        def place(m, y):
            return m.move_to([COL_X, y, 0], aligned_edge=LEFT)

        def swap(old, new, rt=0.5):
            self.play(FadeOut(old), run_time=rt * 0.7)
            self.play(FadeIn(new), run_time=rt)

        # ================= act one: the set-up ===============================
        axis_labels = VGroup(
            MathTex("x", font_size=28, color=GREY_B).move_to(pt([3.45, -0.24])),
            MathTex("y", font_size=28, color=GREY_B).move_to(pt([0.24, 2.36])),
        )
        curve = ParametricFunction(lambda s: pt(C(s)), t_range=[-1.95, 2.4],
                                   color=CURVE_G, stroke_width=5)
        c_label = MathTex(r"C\colon\ g(x,y) = 0", font_size=30, color=CURVE_G)
        c_label.move_to(pt([-1.75, -1.42]))

        dot_p = Dot(pt(P0), radius=0.075, color=WHITE, z_index=8)
        lab_p = MathTex("P", font_size=32).move_to(
            pt(P0 + 0.36 * unit(-TH + 0.55 * NH)))

        vec_t = arrow(P0, P0 + LT * TH, DIR_T, width=7)
        lab_t = MathTex(r"\hat{T}", font_size=34, color=DIR_T)
        lab_t.move_to(pt(P0 + (LT + 0.22) * TH + 0.2 * NH))
        tline = DashedLine(pt(P0 - 2.3 * TH), pt(P0 + 2.3 * TH), color=DIR_T,
                           stroke_width=2, stroke_opacity=0.55,
                           dash_length=0.08)

        h1 = head(r"C\text{: the constraint curve } g(x,y) = 0",
                  r"P\text{: a point where } f \text{ has a max or min on } C",
                  r"\hat{T}\text{: the unit tangent to } C \text{ at } P")
        g1 = lines(r"\text{a small move along } C \text{ from } P",
                   r"\text{is a move along } \hat{T} \text{ or } -\hat{T}",
                   size=27, color=GLOSS)
        g1.next_to(h1, DOWN, buff=0.35, aligned_edge=LEFT)

        self.play(Create(ax), FadeIn(axis_labels), run_time=0.9)
        self.play(FadeIn(h1[0]), Create(curve), run_time=1.6)
        self.play(FadeIn(c_label), run_time=0.6)
        self.wait(0.6)
        self.play(FadeIn(h1[1]), FadeIn(dot_p, scale=1.5), FadeIn(lab_p),
                  run_time=0.9)
        self.wait(0.8)
        self.play(FadeIn(h1[2]), GrowArrow(vec_t), FadeIn(lab_t), run_time=1.0)
        self.wait(0.5)
        self.play(FadeIn(g1), Create(tline), run_time=1.0)
        self.wait(2.2)

        # ================= act two: grad g is perpendicular to T-hat ==========
        # Shown, not just quoted: g is 0 all along C, so a point moving along C
        # sees no change in g, and the rate of change D_T g = grad g . T-hat is
        # 0.  The neighbouring level curves g = c and g = -c are drawn near P
        # (for the drawing, g(x,y) = y - 1.1 sin(x/1.4), whose gradient at P is
        # along NH) so that grad g is seen crossing them.
        h2 = head(r"\text{first: why is } \nabla g \text{ perpendicular to }"
                  r" \hat{T}\text{?}")
        swap(VGroup(h1, g1), h2)

        KG = 0.6

        def level_g(k):
            return ParametricFunction(
                lambda x: pt([x, 1.1 * np.sin(x / 1.4) + k]),
                t_range=[-0.6, 3.2], color=CURVE_G, stroke_width=3,
                stroke_opacity=0.45)

        nb = VGroup(level_g(KG), level_g(-KG))
        nb_labels = VGroup(
            MathTex(r"g = c", font_size=26, color=CURVE_G)
            .move_to(pt([-1.12, 0.22])),
            MathTex(r"g = -c", font_size=26, color=CURVE_G)
            .move_to(pt([2.5, -0.2])),
        ).set_opacity(0.75)
        g_note = MathTex(r"\text{nearby level curves of } g", font_size=26,
                         color=GLOSS)
        g_note.next_to(h2, DOWN, buff=0.22, aligned_edge=LEFT)
        self.play(Create(nb), FadeIn(nb_labels), FadeIn(g_note), run_time=1.4)
        self.wait(0.6)

        def nrm(t):
            return unit(np.array([-1.1 * np.cos(t), 1.4]))

        u = ValueTracker(T0)
        mover = always_redraw(lambda: Dot(pt(C(u.get_value())), radius=0.085,
                                          color=PINK, z_index=9))
        g_read = MathTex(r"g = 0", font_size=28, color=CURVE_G)
        g_read.add_updater(lambda m: m.move_to(pt(
            C(u.get_value()) - 0.34 * nrm(u.get_value()))))

        g2 = lines(r"g = 0 \text{ at every point of } C\text{, so as we move}",
                   r"\text{along } C\text{, } g \text{ does not change at all}",
                   size=30)
        g2.next_to(g_note, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(mover), FadeIn(g_read), FadeIn(g2[0]), run_time=0.8)
        self.play(u.animate.set_value(T0 + 0.62), run_time=1.6,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeIn(g2[1]), u.animate.set_value(T0 - 0.62), run_time=2.6,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(u.animate.set_value(T0), run_time=1.4,
                  rate_func=rate_functions.ease_in_out_sine)
        g_read.clear_updaters()
        self.play(FadeOut(mover), FadeOut(g_read), run_time=0.4)

        g2b = MathTex(r"\text{so } D_{\hat{T}} g = \nabla g \cdot \hat{T} = 0",
                      font_size=34)
        g2b.next_to(g2, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(Write(g2b), run_time=1.2)
        self.wait(0.8)

        eq1 = MathTex(r"\nabla g", r"\cdot", r"\hat{T}", r"= 0", font_size=40)
        eq1[0].set_color(CURVE_G)
        eq1[2].set_color(DIR_T)
        place(eq1, 0.15).shift(RIGHT * 0.25)
        why1 = MathTex(r"\text{(}g \text{ is constant on } C\text{)}",
                       font_size=26, color=GLOSS)
        why1.next_to(eq1, RIGHT, buff=0.4)

        vec_g = arrow(P0, P0 + LG * NH, CURVE_G, width=6)
        lab_g = MathTex(r"\nabla g", font_size=32, color=CURVE_G)
        lab_g.move_to(pt(P0 + (LG + 0.08) * NH - 0.36 * TH))
        sq_g = square(TH, NH)
        g2c = MathTex(r"\nabla g \text{ crosses the level curves of } g"
                      r" \text{ at right angles}", font_size=26, color=GLOSS)
        g2c.next_to(g2b, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(GrowArrow(vec_g), FadeIn(lab_g), run_time=1.0)
        self.play(Create(sq_g), FadeIn(g2c), run_time=0.8)
        self.play(Write(eq1), run_time=1.0)
        self.play(FadeIn(why1), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(nb), FadeOut(nb_labels), run_time=0.6)

        # ================= act three: grad f is perpendicular to T-hat ========
        h3 = head(r"\text{unlike } g\text{, } f \text{ can change along } C\text{;}",
                  r"\text{its rate of change at } P \text{ is }"
                  r" D_{\hat{T}} f = \nabla f \cdot \hat{T}")
        swap(VGroup(h2, g_note, g2, g2b, g2c), h3)

        phi = ValueTracker(52 * DEGREES)

        def fdir(a):
            return np.cos(a) * (-NH) + np.sin(a) * TH

        ghost_f = always_redraw(lambda: arrow(
            P0, P0 + LF * fdir(phi.get_value()), GRAD_F, width=6
        ).set_opacity(0.6))
        shadow_f = always_redraw(lambda: seg(
            P0 - SH * NH, P0 - SH * NH + LF * np.sin(phi.get_value()) * TH,
            MARK, 9, 0.8, z=6))
        drop_f = always_redraw(lambda: dashed(
            P0 + LF * fdir(phi.get_value()),
            P0 - SH * NH + LF * np.sin(phi.get_value()) * TH))
        lab_f = MathTex(r"\nabla f", font_size=32, color=GRAD_F)
        lab_f.add_updater(lambda m: m.move_to(pt(
            P0 + (LF + 0.3) * fdir(phi.get_value()))))

        rises = MathTex(r"f \text{ rises}", font_size=26, color=GLOSS)
        rises.move_to(pt(P0 + 2.55 * TH + 0.32 * NH))
        falls = MathTex(r"f \text{ falls}", font_size=26, color=GLOSS)
        falls.move_to(pt(P0 - 2.45 * TH + 0.36 * NH))

        g3 = lines(r"\text{suppose it were not } 0\text{: } \nabla f"
                   r"\text{ would have a}",
                   r"\text{shadow on } \hat{T}\text{, and } f \text{ would rise one way}",
                   r"\text{along } C \text{ and fall the other}",
                   size=27, color=GLOSS)
        g3.next_to(h3, DOWN, buff=0.3, aligned_edge=LEFT)
        g3b = MathTex(r"\text{so } P \text{ would be no max or min}",
                      font_size=27, color=GLOSS)
        g3b.next_to(g3, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(FadeIn(g3[0]), run_time=0.6)
        self.add(ghost_f, shadow_f, drop_f)
        self.play(FadeIn(lab_f), run_time=0.6)
        self.play(FadeIn(g3[1:]), FadeIn(rises), FadeIn(falls), run_time=0.9)
        self.play(FadeIn(g3b), run_time=0.6)
        self.wait(2.2)

        h3b = head(r"\text{at a max or min of } f \text{ on } C\text{, then,}",
                   r"\text{the shadow must be gone:}")
        self.play(FadeOut(h3), FadeOut(g3), FadeOut(g3b), run_time=0.5)
        self.play(FadeIn(h3b), run_time=0.5)
        self.play(phi.animate.set_value(0.0), run_time=2.4,
                  rate_func=rate_functions.ease_in_out_sine)
        lab_f.clear_updaters()
        vec_f = arrow(P0, P0 - LF * NH, GRAD_F, width=6)
        F_SIDE = 0.45
        self.remove(ghost_f, shadow_f, drop_f)
        self.add(vec_f)
        self.play(FadeOut(rises), FadeOut(falls),
                  lab_f.animate.move_to(pt(P0 - (LF - 0.05) * NH + F_SIDE * TH)),
                  run_time=0.5)

        eq2 = MathTex(r"\nabla f", r"\cdot", r"\hat{T}", r"= 0", font_size=40)
        eq2[0].set_color(GRAD_F)
        eq2[2].set_color(DIR_T)
        place(eq2, -0.55).shift(RIGHT * 0.25)
        why2 = MathTex(r"\text{(}f \text{ is stationary along } C\text{)}",
                       font_size=26, color=GLOSS)
        why2.next_to(eq2, RIGHT, buff=0.4)
        why2.align_to(why1, LEFT)
        sq_f = square(TH, -NH)
        self.play(Create(sq_f), run_time=0.7)
        self.play(Write(eq2), run_time=1.0)
        self.play(FadeIn(why2), run_time=0.6)
        self.wait(2.0)

        # ================= act four: one line, so collinear ====================
        # Pure geometry: in the plane only one line through P is perpendicular
        # to T-hat, and both gradients, being perpendicular to T-hat, lie on it.
        h4 = head(r"\text{so both gradients are perpendicular}",
                  r"\text{to the same vector } \hat{T}")
        swap(h3b, h4)
        both = SurroundingRectangle(VGroup(eq1, eq2), color=MARK, buff=0.15)
        self.play(Create(both), run_time=0.8)
        self.wait(1.4)
        self.play(FadeOut(both), run_time=0.4)

        normal = DashedLine(pt(P0 - 2.0 * NH), pt(P0 + 1.85 * NH), color=MARK,
                            stroke_width=3, dash_length=0.1, z_index=2)
        normal_lab = MathTex(r"\text{normal line}", font_size=26, color=MARK)
        normal_lab.move_to(pt(P0 + 1.9 * NH + 0.85 * TH))

        h5 = head(r"\text{in the plane, only one line through } P",
                  r"\text{is perpendicular to } \hat{T}\text{: the normal line}")
        swap(h4, h5)
        self.play(Create(normal), FadeIn(normal_lab), run_time=1.2)
        self.wait(1.4)

        h6 = head(r"\text{so } \nabla g \text{ and } \nabla f \text{ both lie on it:}",
                  r"\text{they are collinear}")
        swap(h5, h6)
        self.play(Indicate(VGroup(vec_g, lab_g), color=CURVE_G,
                           scale_factor=1.1), run_time=0.9)
        self.play(Indicate(VGroup(vec_f, lab_f), color=GRAD_F,
                           scale_factor=1.1), run_time=0.9)
        self.wait(1.4)

        # ================= act five: grad f = lambda grad g ===================
        h7 = head(r"\text{collinear: one is a multiple of the other}")
        self.play(FadeOut(h6), FadeOut(VGroup(eq1, eq2, why1, why2)),
                  FadeOut(sq_f), FadeOut(sq_g), run_time=0.6)
        self.play(FadeIn(h7), run_time=0.6)

        boxed = MathTex(r"\nabla f = \lambda\,\nabla g", font_size=50)
        boxed[0][0:2].set_color(GRAD_F)
        boxed[0][-2:].set_color(CURVE_G)
        boxed.move_to([COL_X + 3.0, 1.75, 0])
        box = SurroundingRectangle(boxed, color=MARK, buff=0.2)

        # the multiplier, read off a dial under the picture
        dial = NumberLine(x_range=[-2, 2, 1], length=4.4, include_ticks=True,
                          tick_size=0.07, stroke_width=2.5, color=GREY_B)
        dial.move_to([-3.45, -3.2, 0])
        dial_lab = MathTex(r"\lambda", font_size=32).next_to(dial, LEFT,
                                                             buff=0.25)
        dial_nums = VGroup(
            MathTex("0", font_size=24, color=GREY_B).next_to(dial.n2p(0), DOWN,
                                                              buff=0.14),
            MathTex("1", font_size=24, color=GREY_B).next_to(dial.n2p(1), DOWN,
                                                              buff=0.14),
        )
        lam = ValueTracker(1.0)
        marker = always_redraw(lambda: Triangle(
            fill_opacity=1, color=MARK, stroke_width=0
        ).scale(0.09).rotate(PI).next_to(dial.n2p(lam.get_value()), UP,
                                         buff=0.04))

        copy = always_redraw(lambda: arrow(
            P0, P0 + lam.get_value() * LG * NH, MARK, width=4, z=7))
        lab_copy = MathTex(r"\lambda\nabla g", font_size=30, color=MARK)

        def place_copy(m):
            v = lam.get_value()
            side = 1.0 if v >= 0 else -1.0
            m.move_to(pt(P0 + (v * LG + side * 0.1) * NH - 0.62 * TH))
            m.set_opacity(0.0 if abs(v) < 0.18 else 1.0)

        lab_copy.add_updater(place_copy)

        self.play(FadeIn(dial), FadeIn(dial_lab), FadeIn(dial_nums),
                  FadeIn(marker), run_time=0.8)
        self.add(copy)
        self.play(FadeIn(lab_copy), run_time=0.5)
        self.wait(0.6)
        self.play(lam.animate.set_value(LAM), run_time=4.0,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(Flash(pt(P0 - LF * NH), color=MARK, line_length=0.16,
                        flash_radius=0.26), run_time=0.7)
        self.play(Write(boxed), Create(box), run_time=1.1)

        neg = MathTex(r"\text{here } \lambda < 0\text{: the gradients point"
                      r" opposite ways}", font_size=27, color=GLOSS)
        neg.next_to(box, DOWN, buff=0.45)
        neg.align_to(h7, LEFT)
        self.play(FadeIn(neg), run_time=0.6)
        self.wait(2.0)

        # grad f turned round: lambda > 0
        sf = ValueTracker(-1.0)
        self.remove(vec_f)
        vec_f2 = always_redraw(lambda: arrow(
            P0, P0 + sf.get_value() * LF * NH, GRAD_F, width=10, z=5))
        self.add(vec_f2)
        lab_f.add_updater(lambda m: m.move_to(pt(
            P0 + (sf.get_value() * LF + np.sign(sf.get_value() + 1e-9) * 0.05)
            * NH + F_SIDE * TH)).set_opacity(
                0.0 if abs(sf.get_value()) < 0.15 else 1.0))
        pos = MathTex(r"\text{had } \nabla f \text{ pointed the same way as }"
                      r" \nabla g\text{, } \lambda > 0", font_size=27,
                      color=GLOSS)
        pos.next_to(neg, DOWN, buff=0.2, aligned_edge=LEFT)
        lab_copy.clear_updaters()
        self.play(FadeOut(lab_copy), run_time=0.3)
        self.play(FadeIn(pos), run_time=0.6)
        self.play(sf.animate.set_value(1.0), lam.animate.set_value(-LAM),
                  run_time=3.0, rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.6)
        only = MathTex(r"\text{only the collinearity matters}",
                       font_size=27, color=GLOSS)
        only.next_to(pos, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(only), run_time=0.6)
        self.wait(1.4)
        self.play(sf.animate.set_value(-1.0), lam.animate.set_value(LAM),
                  run_time=2.4, rate_func=rate_functions.ease_in_out_sine)
        self.wait(1.0)

        # ================= act six: the Lagrange equations ====================
        h8 = head(r"\text{so a max or min of } f \text{ on } C \text{ solves}")
        col = VGroup(h7, boxed, box, neg, pos, only)
        self.play(FadeOut(col), run_time=0.7)
        self.play(FadeIn(h8), run_time=0.6)

        def at_equals(rows, size, buff):
            g = VGroup(*[MathTex(a, "=", b, font_size=size) for a, b in rows])
            g.arrange(DOWN, buff=buff)
            for r in g[1:]:
                r.shift(RIGHT * (g[0][1].get_x() - r[1].get_x()))
            return g

        system = at_equals([(r"\nabla f", r"\lambda\,\nabla g"),
                            (r"g(x,y)", r"0")], 48, 0.3)
        system.move_to([COL_X + 3.0, 1.85, 0])
        sbox = SurroundingRectangle(system, color=MARK, buff=0.25)
        self.play(Write(system), Create(sbox), run_time=1.4)
        self.wait(1.4)

        comp = lines(r"\text{component by component:}", size=28, color=GLOSS)
        place(comp, 0.25)
        eqs = at_equals([(r"f_x", r"\lambda\, g_x"), (r"f_y", r"\lambda\, g_y"),
                         (r"g(x,y)", r"0")], 38, 0.25)
        eqs.move_to([COL_X + 3.0, -0.95, 0])
        eqs.shift(RIGHT * (system[0][1].get_x() - eqs[0][1].get_x()))
        count = MathTex(r"\text{three equations for the three unknowns }"
                        r" x,\ y,\ \lambda", font_size=28, color=GLOSS)
        place(count, -2.35)
        self.play(FadeIn(comp), run_time=0.6)
        self.play(Write(eqs), run_time=1.4)
        self.play(FadeIn(count), run_time=0.7)
        self.wait(4.0)
