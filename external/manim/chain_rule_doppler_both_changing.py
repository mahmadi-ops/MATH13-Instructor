"""Conceptual animation for Part (f) of the boat-and-wave activity.

Render from the repository root with Manim Community 0.19.0:

    manim -qm --disable_caching -o chain_rule_doppler_both_changing_manim \
        assets/manim/chain_rule_doppler_both_changing.py \
        ChainRuleDopplerBothChanging

The animation deliberately stays symbolic.  It distinguishes the change in
the wave itself from the change in the phase sampled by the boat, first keeps
the boat and crest paths separate, and then combines them through relative
motion.  The symbol ``k`` is used for crest speed throughout.
"""

from manim import *
import numpy as np


TEX = TexTemplate(preamble=r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}")
config.tex_template = TEX
config.background_color = BLACK

TEXT_FONT = "DejaVu Sans"
BOAT_COLOR = ORANGE
CREST_COLOR = BLUE
WAVE_COLOR = TEAL
OUTPUT_COLOR = GOLD
HIGHLIGHT_COLOR = YELLOW
AMPLITUDE_COLOR = PURPLE_B
CAPTION_COLOR = GREY_B
MUTED_COLOR = GREY_C
GUIDE_COLOR = GREY_D


def mtex(*parts, **kwargs):
    """MathTex using the compact template shared by the Doppler videos."""
    kwargs.setdefault("tex_template", TEX)
    return MathTex(*parts, **kwargs)


def fit_width(mobject, width=13.2):
    """Keep displayed material inside the 16:9 title-safe area."""
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    return mobject


def boat_at(point, scale=1.0):
    """Filled, closed-hull boat matching the other activity animations."""
    hull = Polygon(
        [-0.28, 0.11, 0],
        [0.28, 0.11, 0],
        [0.18, 0, 0],
        [-0.18, 0, 0],
        stroke_color=BOAT_COLOR,
        stroke_width=2.2,
        fill_color=BOAT_COLOR,
        fill_opacity=1,
    )
    mast = Line([0, 0.11, 0], [0, 0.52, 0], color=GREY_A, stroke_width=2)
    sail = Polygon(
        [0.025, 0.49, 0],
        [0.025, 0.16, 0],
        [0.24, 0.16, 0],
        stroke_color=BOAT_COLOR,
        stroke_width=1.2,
        fill_color=BOAT_COLOR,
        fill_opacity=0.35,
    )
    return VGroup(hull, mast, sail).scale(scale, about_point=ORIGIN).shift(point)


class ChainRuleDopplerBothChanging(Scene):
    """Show the amplitude and relative-position paths conceptually."""

    def construct(self):
        self.current_caption = None
        self.introduction()
        self.three_paths()
        self.changing_wave_and_phase()
        self.combine_contributions()
        self.conclusion()

    # ------------------------------------------------------------------
    # Shared layout helpers
    def section_tag(self, text):
        tag = Text(text, font=TEXT_FONT, font_size=20, color=MUTED_COLOR)
        tag.to_corner(UR, buff=0.42)
        return tag

    def corner_equation(self, tex, font_size=38, color=WHITE, max_width=9.3):
        equation = fit_width(
            mtex(tex, font_size=font_size, color=color), max_width
        )
        equation.to_corner(UL, buff=0.42)
        return equation

    def caption(self, text, color=CAPTION_COLOR):
        new_caption = Text(text, font=TEXT_FONT, font_size=25, color=color)
        fit_width(new_caption, 13.4)
        new_caption.to_edge(DOWN, buff=0.28)
        animations = [FadeIn(new_caption)]
        if self.current_caption is not None:
            animations.append(FadeOut(self.current_caption))
        self.play(*animations, run_time=0.55)
        self.current_caption = new_caption
        return new_caption

    def clear_scene(self):
        for mobject in self.mobjects:
            mobject.clear_updaters(recursive=True)
        shown = list(self.mobjects)
        if shown:
            self.play(*[FadeOut(mobject) for mobject in shown], run_time=0.55)
        self.clear()
        self.current_caption = None

    @staticmethod
    def node(tex, color, width=1.75):
        label = mtex(tex, font_size=40, color=color)
        box = RoundedRectangle(
            width=width,
            height=0.88,
            corner_radius=0.12,
            color=color,
            stroke_width=2.2,
            fill_color=BLACK,
            fill_opacity=0.92,
        )
        return VGroup(box, label)

    # ------------------------------------------------------------------
    # Storyboard
    def introduction(self):
        title = VGroup(
            Text(
                "WHEN BOTH EFFECTS CHANGE",
                font=TEXT_FONT,
                font_size=43,
                color=WHITE,
            ),
            Text(
                "Amplitude change and relative motion",
                font=TEXT_FONT,
                font_size=29,
                color=WAVE_COLOR,
            ),
        ).arrange(DOWN, buff=0.34)
        rule = Line(LEFT * 3.7, RIGHT * 3.7, color=GUIDE_COLOR, stroke_width=1.5)
        rule.next_to(title, DOWN, buff=0.42)

        self.play(FadeIn(title[0], shift=0.15 * UP), run_time=0.9)
        self.play(FadeIn(title[1]), Create(rule), run_time=0.75)
        self.caption("Two mechanisms can change the height measured by the boat.")
        self.wait(2.5)
        self.clear_scene()

    def three_paths(self):
        tag = self.section_tag("THREE PATHS FROM TIME")
        model = self.corner_equation(
            r"w=f(A,x_b,x_c),\qquad \phi=\frac{2\pi}{\lambda}(x_b-x_c)",
            font_size=35,
            max_width=9.6,
        )
        self.play(FadeIn(tag), Write(model), run_time=1.0)

        time_node = self.node(r"t", WHITE, width=1.25).move_to([-5.1, 0.25, 0])
        amplitude_node = self.node(r"A(t)", AMPLITUDE_COLOR).move_to([-1.75, 1.55, 0])
        boat_node = self.node(r"x_b(t)", BOAT_COLOR).move_to([-1.75, 0.25, 0])
        crest_node = self.node(r"x_c(t)", CREST_COLOR).move_to([-1.75, -1.05, 0])
        output_node = self.node(r"w(t)", OUTPUT_COLOR, width=1.65).move_to([2.65, 0.25, 0])

        inputs = [amplitude_node, boat_node, crest_node]
        colors = [AMPLITUDE_COLOR, BOAT_COLOR, CREST_COLOR]
        incoming = VGroup(
            *[
                Arrow(
                    time_node.get_right(),
                    node.get_left(),
                    buff=0.14,
                    color=color,
                    stroke_width=3,
                    max_tip_length_to_length_ratio=0.12,
                )
                for node, color in zip(inputs, colors)
            ]
        )
        outgoing = VGroup(
            *[
                Arrow(
                    node.get_right(),
                    output_node.get_left(),
                    buff=0.14,
                    color=color,
                    stroke_width=3,
                    max_tip_length_to_length_ratio=0.12,
                )
                for node, color in zip(inputs, colors)
            ]
        )

        self.caption("Time reaches the measured height along three paths.")
        self.play(FadeIn(time_node), FadeIn(output_node), run_time=0.65)
        for node, arrow_in, arrow_out in zip(inputs, incoming, outgoing):
            self.play(FadeIn(node), GrowArrow(arrow_in), GrowArrow(arrow_out), run_time=0.65)
        self.wait(1.0)

        chain = fit_width(
            mtex(
                r"\frac{dw}{dt}",
                "=",
                r"\frac{\partial f}{\partial A}A'",
                "+",
                r"\frac{\partial f}{\partial x_b}v",
                "+",
                r"\frac{\partial f}{\partial x_c}k",
                font_size=39,
            ),
            12.2,
        ).move_to([0, -2.08, 0])
        chain[2].set_color(AMPLITUDE_COLOR)
        chain[4].set_color(BOAT_COLOR)
        chain[6].set_color(CREST_COLOR)
        self.play(Write(chain), run_time=1.4)
        self.wait(2.0)
        self.clear_scene()

    def changing_wave_and_phase(self):
        tag = self.section_tag("TWO CHANGES AT ONCE")
        model = self.corner_equation(
            r"w(t)=A(t)\cos\phi(t),\qquad \phi(t)=\frac{2\pi}{\lambda}(x_b-x_c)",
            font_size=34,
            max_width=10.1,
        )
        self.play(FadeIn(tag), Write(model), run_time=1.0)

        time = ValueTracker(0.0)
        left, right = -6.05, 6.05
        mean_y = 0.15
        axis_y = -1.55
        samples = np.linspace(left, right, 260)
        wave_number = 1.22

        def amplitude():
            return 0.48 + 0.07 * time.get_value()

        def crest_x():
            return -2.15 + 0.53 * time.get_value()

        def boat_x():
            return -4.65 + 0.31 * time.get_value()

        def height(x_value):
            return mean_y + amplitude() * np.cos(
                wave_number * (x_value - crest_x())
            )

        def wave_now():
            values = np.array([height(x_value) for x_value in samples])
            points = np.column_stack((samples, values, np.zeros_like(samples)))
            return VMobject(color=WAVE_COLOR, stroke_width=3.5).set_points_as_corners(points)

        def crest_point():
            return np.array([crest_x(), mean_y + amplitude(), 0])

        def boat_point():
            return np.array([boat_x(), height(boat_x()) + 0.02, 0])

        wave = always_redraw(wave_now)
        mean_level = DashedLine(
            [left, mean_y, 0],
            [right, mean_y, 0],
            color=GUIDE_COLOR,
            stroke_width=1.2,
            dash_length=0.10,
        )
        axis = Line(
            [left, axis_y, 0],
            [right, axis_y, 0],
            color=GUIDE_COLOR,
            stroke_width=1.5,
        )
        boat = always_redraw(lambda: boat_at(boat_point(), 0.78))
        crest = always_redraw(lambda: Dot(crest_point(), radius=0.075, color=CREST_COLOR))
        boat_guide = always_redraw(
            lambda: DashedLine(
                [boat_x(), axis_y, 0],
                [boat_x(), height(boat_x()), 0],
                color=BOAT_COLOR,
                stroke_width=1.7,
                dash_length=0.07,
            )
        )
        crest_guide = always_redraw(
            lambda: DashedLine(
                [crest_x(), axis_y, 0],
                crest_point(),
                color=CREST_COLOR,
                stroke_width=1.7,
                dash_length=0.07,
            )
        )

        def separation_arrow_now():
            return DoubleArrow(
                [boat_x(), axis_y - 0.22, 0],
                [crest_x(), axis_y - 0.22, 0],
                buff=0.03,
                color=HIGHLIGHT_COLOR,
                stroke_width=2.5,
                tip_length=0.13,
            )

        separation_arrow = always_redraw(separation_arrow_now)
        separation_label = mtex(
            r"r(t)=x_b(t)-x_c(t)", font_size=30, color=HIGHLIGHT_COLOR
        )
        separation_label.add_updater(
            lambda label: label.move_to(
                [(boat_x() + crest_x()) / 2, axis_y - 0.62, 0]
            )
        )

        def amplitude_arrow_now():
            x_value = 5.55
            return DoubleArrow(
                [x_value, mean_y, 0],
                [x_value, mean_y + amplitude(), 0],
                buff=0,
                color=AMPLITUDE_COLOR,
                stroke_width=2.8,
                tip_length=0.14,
            )

        amplitude_arrow = always_redraw(amplitude_arrow_now)
        amplitude_label = mtex(r"A(t)", font_size=29, color=AMPLITUDE_COLOR)
        amplitude_label.add_updater(
            lambda label: label.move_to(
                [5.98, mean_y + amplitude() / 2, 0]
            )
        )
        boat_label = mtex(r"x_b(t)", font_size=27, color=BOAT_COLOR)
        boat_label.add_updater(
            lambda label: label.move_to([boat_x(), axis_y + 0.24, 0])
        )
        crest_label = mtex(r"x_c(t)", font_size=27, color=CREST_COLOR)
        crest_label.add_updater(
            lambda label: label.move_to([crest_x(), axis_y + 0.24, 0])
        )

        dynamic = VGroup(
            wave,
            mean_level,
            axis,
            boat_guide,
            crest_guide,
            separation_arrow,
            amplitude_arrow,
            boat,
            crest,
            separation_label,
            amplitude_label,
            boat_label,
            crest_label,
        )
        self.play(FadeIn(dynamic), run_time=1.0)
        self.caption("Changing A(t) changes the wave itself.")
        self.play(time.animate.set_value(3.0), run_time=3.5, rate_func=linear)
        self.caption("Changing relative position changes which phase the boat samples.")
        self.play(time.animate.set_value(6.0), run_time=4.2, rate_func=linear)
        self.wait(1.2)
        self.clear_scene()

    def combine_contributions(self):
        tag = self.section_tag("KEEP THE PATHS SEPARATE")
        total = self.corner_equation(
            r"\frac{dw}{dt}=C_A+f_{x_b}v+f_{x_c}k",
            font_size=39,
            color=OUTPUT_COLOR,
            max_width=7.3,
        )
        self.play(FadeIn(tag), Write(total), run_time=0.9)

        amplitude_term = mtex(
            r"C_A=A'(t)\cos\phi", font_size=35, color=AMPLITUDE_COLOR
        )
        boat_term = mtex(
            r"f_{x_b}v=-\frac{2\pi A(t)}{\lambda}v\sin\phi",
            font_size=33,
            color=BOAT_COLOR,
        )
        crest_term = mtex(
            r"f_{x_c}k=+\frac{2\pi A(t)}{\lambda}k\sin\phi",
            font_size=33,
            color=CREST_COLOR,
        )

        terms = VGroup(amplitude_term, boat_term, crest_term).arrange(
            DOWN, buff=0.46, aligned_edge=LEFT
        ).move_to([0, 0.45, 0])
        boxes = VGroup(
            *[
                SurroundingRectangle(
                    term,
                    color=color,
                    buff=0.18,
                    corner_radius=0.06,
                    stroke_width=2,
                )
                for term, color in zip(
                    terms, [AMPLITUDE_COLOR, BOAT_COLOR, CREST_COLOR]
                )
            ]
        )

        self.caption("Amplitude growth, boat motion, and crest motion each contribute a rate.")
        for term, box in zip(terms, boxes):
            self.play(Write(term), Create(box), run_time=0.72)
        self.wait(1.6)

        position_sum = fit_width(
            mtex(
                r"C_r=f_{x_b}v+f_{x_c}k",
                "=",
                r"\frac{2\pi A(t)}{\lambda}(k-v)\sin\phi",
                font_size=39,
            ),
            11.4,
        ).move_to([0, -1.93, 0])
        position_sum[0].set_color(HIGHLIGHT_COLOR)
        position_sum[2].set_color(HIGHLIGHT_COLOR)
        sum_box = SurroundingRectangle(
            position_sum,
            color=HIGHLIGHT_COLOR,
            buff=0.17,
            corner_radius=0.06,
            stroke_width=2.2,
        )
        self.caption("The boat and crest paths combine into one relative-motion contribution.")
        self.play(
            TransformFromCopy(VGroup(boat_term, crest_term), position_sum),
            Create(sum_box),
            run_time=1.25,
        )
        self.wait(2.2)
        self.clear_scene()

    def conclusion(self):
        tag = self.section_tag("ADD THE TWO EFFECTS")
        formula = fit_width(
            mtex(
                r"\frac{dw}{dt}",
                "=",
                r"\underbrace{A'(t)\cos\phi}_{\text{wave shape changes}}",
                "+",
                r"\underbrace{\frac{2\pi A(t)}{\lambda}(k-v)\sin\phi}_{\text{sampling phase changes}}",
                font_size=38,
            ),
            12.35,
        ).move_to([0, 0.75, 0])
        formula[0].set_color(OUTPUT_COLOR)
        formula[2].set_color(AMPLITUDE_COLOR)
        formula[4].set_color(HIGHLIGHT_COLOR)
        formula_box = SurroundingRectangle(
            formula,
            color=OUTPUT_COLOR,
            buff=0.22,
            corner_radius=0.07,
            stroke_width=2.2,
        )

        amplitude_card = VGroup(
            mtex(r"A'(t)\cos\phi", font_size=34, color=AMPLITUDE_COLOR),
            Text("wave changes", font=TEXT_FONT, font_size=22, color=MUTED_COLOR),
        ).arrange(DOWN, buff=0.16)
        motion_card = VGroup(
            mtex(
                r"\frac{2\pi A(t)}{\lambda}(k-v)\sin\phi",
                font_size=32,
                color=HIGHLIGHT_COLOR,
            ),
            Text("sampled phase changes", font=TEXT_FONT, font_size=22, color=MUTED_COLOR),
        ).arrange(DOWN, buff=0.16)
        cards = VGroup(amplitude_card, motion_card).arrange(RIGHT, buff=1.25)
        fit_width(cards, 11.8).move_to([0, -1.17, 0])
        plus = mtex("+", font_size=43, color=WHITE).move_to(
            (amplitude_card.get_right() + motion_card.get_left()) / 2
        )

        self.play(FadeIn(tag), Write(formula), Create(formula_box), run_time=1.5)
        self.caption("The measured rate adds how the wave changes and how the boat moves through it.")
        self.play(FadeIn(cards), FadeIn(plus), run_time=0.9)
        self.wait(2.4)
        self.caption("Their signs decide whether the two effects reinforce or oppose one another.")
        self.play(
            Indicate(amplitude_card[0], color=AMPLITUDE_COLOR),
            Indicate(motion_card[0], color=HIGHLIGHT_COLOR),
            run_time=1.1,
        )
        self.wait(3.2)
