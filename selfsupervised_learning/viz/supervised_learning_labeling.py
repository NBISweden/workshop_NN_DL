import numpy as np
import manim
from manim import (
    Scene, Text, Tex,
    Rectangle, Dot, Flash,
    FadeIn, FadeOut, Transform,
    UP,
    GREY_B, WHITE,
)

PIXEL_WIDTH  = 900
PIXEL_HEIGHT = 900

manim.config.pixel_height = PIXEL_HEIGHT
manim.config.pixel_width  = PIXEL_WIDTH

Text.set_default(font="Ubuntu")

_CA = "#5ab4e8"   # blue — class A
_CB = "#ffa060"   # orange — class B


class SupervisedLearningLabelingScene(Scene):
    def construct(self):
        # ── Data (same seed as all other scenes) ─────────────────────────────
        rng = np.random.default_rng(seed=1729)
        n    = 80
        r_a  = rng.uniform(0.00, 0.95, n)
        r_b  = rng.uniform(1.05, 2.00, n)
        th_a = rng.uniform(0, 2 * np.pi, n)
        th_b = rng.uniform(0, 2 * np.pi, n)

        cx_a, cy_a = r_a * np.cos(th_a), r_a * np.sin(th_a)
        cx_b, cy_b = r_b * np.cos(th_b), r_b * np.sin(th_b)

        # ── Layout ────────────────────────────────────────────────────────────
        PSIZ   = 13
        DSCL   = PSIZ / 4.8
        CENTER = np.array([0.0, 0.0, 0.0])
        DR     = 0.1   # dot radius, scaled for the larger panel

        def d2s(x, y):
            return CENTER + np.array([x * DSCL, y * DSCL, 0.0])

        panel = Rectangle(width=PSIZ, height=PSIZ,
                          stroke_color=GREY_B, stroke_width=1.5,
                          fill_color="#101028", fill_opacity=0.4)
        panel.move_to(CENTER)

        # All dots start white — no labels yet
        dots_a = [Dot(d2s(x, y), radius=DR, color=WHITE, fill_opacity=0.8)
                  for x, y in zip(cx_a, cy_a)]
        dots_b = [Dot(d2s(x, y), radius=DR, color=WHITE, fill_opacity=0.8)
                  for x, y in zip(cx_b, cy_b)]

        # Counter at the bottom of the panel
        counter_pos = CENTER + np.array([0.0, -PSIZ / 2 + 0.5, 0.0])
        counter = Tex(f"Labeled: 0 / {2 * n}", font_size=48)
        counter.move_to(counter_pos)

        # ── Scene setup ───────────────────────────────────────────────────────
        self.play(FadeIn(panel), run_time=0.5)
        self.play(*[FadeIn(d) for d in dots_a + dots_b], run_time=0.8)
        self.play(FadeIn(counter), run_time=0.4)

        # ── Labeling sequence ─────────────────────────────────────────────────
        # Visit 40 points from each class (interleaved A, B, A, B, …),
        # totalling 80 / 160 labeled — the counter at the end makes the
        # incompleteness obvious.
        n_each = 40
        step   = n // n_each   # spread evenly across each class
        seq = []
        for i in range(n_each):
            seq.append(('a', i * step))
            seq.append(('b', i * step))

        for labeled_count, (cls, idx) in enumerate(seq, start=1):
            if cls == 'a':
                dot, color, label_text = dots_a[idx], _CA, "This is Class A"
            else:
                dot, color, label_text = dots_b[idx], _CB, "This is Class B"

            new_counter = Tex(f"Labeled: {labeled_count} / {2 * n}", font_size=48)
            new_counter.move_to(counter_pos)

            label = Tex(label_text, font_size=48, color=color)
            label.next_to(dot, UP, buff=0.2)

            # Ping: flash burst + recolor the dot
            self.play(
                Flash(dot.get_center(), color=color,
                      flash_radius=0.4, line_length=0.2, num_lines=10),
                dot.animate.set_color(color),
                run_time=0.4,
            )
            # Label appears; counter ticks up
            self.play(FadeIn(label), Transform(counter, new_counter), run_time=0.3)
            self.wait(0.25)
            self.play(FadeOut(label), run_time=0.35)

        self.wait(5)


def supervised_learning_labeling(width: int = PIXEL_WIDTH) -> None:
    """Embed the pre-rendered SupervisedLearningLabelingScene inline (Jupyter / Quarto)."""
    from ._media import _embed_render
    _embed_render("SupervisedLearningLabelingScene", width, __file__)
