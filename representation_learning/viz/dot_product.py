"""
Dot-product / cosine-similarity animation.

Render:
    python -m manim -ql viz/dot_product.py DotProductScene

Display in a Quarto notebook cell (eval: true, echo: false):
    from viz import show_dot_product
    show_dot_product()
"""
from __future__ import annotations

import numpy as np
from manim import (
    BLUE, GREY_B, GREY_C, ORIGIN, RED, WHITE, YELLOW, YELLOW_D,
    PI, UP, DOWN, LEFT, RIGHT,
    Scene, ValueTracker,
    NumberPlane, Arrow, Line, DashedLine, Angle, MathTex,
    GrowArrow, Write, FadeIn,
    always_redraw, smooth,
)


class DotProductScene(Scene):
    def construct(self) -> None:
        # ── Background grid ──────────────────────────────────────────────────
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": GREY_C, "stroke_opacity": 0.35},
            axis_config={"stroke_color": GREY_B, "stroke_opacity": 0.7},
        )
        self.add(plane)

        # ── Vectors ──────────────────────────────────────────────────────────
        A_angle = np.pi/8
        A_MAG = 2
        A = np.array([np.cos(A_angle)*A_MAG, np.sin(A_angle)*A_MAG, 0.0])
        B_MAG = 2.2
        angle = ValueTracker(PI / 3)        # start at 60°

        def b_tip() -> np.ndarray:
            θ = angle.get_value()
            return np.array([B_MAG * np.cos(θ), B_MAG * np.sin(θ), 0.0])

        def _arrow(start, end, color, **kw):
            return Arrow(start, end, buff=0, color=color,
                         stroke_width=5, max_tip_length_to_length_ratio=0.12, **kw)

        vec_a = _arrow(ORIGIN, A, BLUE)
        vec_b = _arrow(ORIGIN, b_tip(), RED)

        label_a = MathTex(r"\vec{a}", color=BLUE, font_size=44).next_to(
            A + np.array([0.1, 0.1, 0]), buff=0.1
        )
        label_b = always_redraw(
            lambda: MathTex(r"\vec{b}", color=RED, font_size=44).next_to(
                b_tip() + np.array([0.1, 0.1, 0]), buff=0.1
            )
        )

        # ── Angle arc + θ label ──────────────────────────────────────────────
        def _arc():
            return Angle(
                Line(ORIGIN, A), Line(ORIGIN, b_tip()),
                radius=0.75, color=YELLOW, stroke_width=3,
            )

        arc = always_redraw(_arc)
        theta_lbl = always_redraw(
            lambda: MathTex(r"\theta", color=YELLOW, font_size=34).move_to(
                _arc().point_from_proportion(0.5) * 1.55
            )
        )

        # ── Projection (dashed drop from b onto a) ───────────────────────────
        def _proj_line():
            b = b_tip()
            a_hat = A / np.linalg.norm(A)
            proj = np.dot(b, a_hat) * a_hat
            return DashedLine(b, proj, color=YELLOW_D,
                              stroke_width=2, dash_length=0.12)

        proj = always_redraw(_proj_line)

        # ── Info panel ───────────────────────────────────────────────────────
        formula = MathTex(
            r"\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta",
            font_size=30, color=WHITE,
        ).to_corner(UP + LEFT).shift(np.array([0.4, -0.5, 0]))

        def _dot() -> float:
            return float(np.dot(A, b_tip()))

        def _cos() -> float:
            return float(np.dot(A, b_tip()) / (np.linalg.norm(A) * B_MAG))

        dot_readout = always_redraw(
            lambda: MathTex(
                rf"\vec{{a}} \cdot \vec{{b}} = {_dot():+.2f}",
                font_size=30,
            ).next_to(formula, DOWN, buff=0.35).align_to(formula, LEFT)
        )
        cos_readout = always_redraw(
            lambda: MathTex(
                rf"\cos\theta = {_cos():+.2f}",
                font_size=30, color=YELLOW,
            ).next_to(dot_readout, DOWN, buff=0.2).align_to(formula, LEFT)
        )

        # ── Intro animations ─────────────────────────────────────────────────
        self.play(GrowArrow(vec_a), Write(label_a), run_time=0.8)
        self.play(GrowArrow(vec_b), run_time=0.7)
        # attach updater *after* the grow animation to avoid jitter
        vec_b.add_updater(lambda m: m.become(_arrow(ORIGIN, b_tip(), RED)))
        self.add(label_b, arc, theta_lbl)
        self.play(Write(formula), run_time=0.7)
        self.play(FadeIn(dot_readout), FadeIn(cos_readout), run_time=0.5)
        self.wait(0.8)
        
        self.add(proj)                          # reveal projection at near-parallel
        
        # ── Sweep: toward parallel (θ → ~0) ─────────────────────────────────
        self.play(angle.animate.set_value(A_angle+0.1), run_time=3.0, rate_func=smooth)
        self.wait(0.3)
        
        self.wait(0.4)

        # ── Sweep: orthogonal (θ = π/2) ──────────────────────────────────────
        self.play(angle.animate.set_value(A_angle + (PI / 2)), run_time=3.0, rate_func=smooth)
        self.wait(1)

        # ── Sweep: anti-parallel (θ → π) ─────────────────────────────────────
        self.play(angle.animate.set_value(A_angle + PI), run_time=3.0, rate_func=smooth)
        self.wait(1)

        # ── Return to a nice angle ────────────────────────────────────────────
        self.play(angle.animate.set_value(PI / 4), run_time=1.5, rate_func=smooth)
        self.wait(5.0)


def show_dot_product(width: int = 700) -> None:
    """Embed the pre-rendered DotProductScene inline (Jupyter / Quarto)."""
    from ._media import _embed_render
    _embed_render("DotProductScene", width, __file__)
