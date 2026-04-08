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
        A = np.array([2.5, 0.5, 0.0])
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
        ).to_corner(UP + LEFT).shift(np.array([-0.4, -0.5, 0]))

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
        self.wait(0.4)

        # ── Sweep: toward parallel (θ → ~0) ─────────────────────────────────
        self.play(angle.animate.set_value(0.08), run_time=2.0, rate_func=smooth)
        self.wait(0.3)
        self.add(proj)                          # reveal projection at near-parallel
        self.wait(0.4)

        # ── Sweep: orthogonal (θ = π/2) ──────────────────────────────────────
        self.play(angle.animate.set_value(PI / 2), run_time=2.0, rate_func=smooth)
        self.wait(0.5)

        # ── Sweep: anti-parallel (θ → π) ─────────────────────────────────────
        self.play(angle.animate.set_value(PI - 0.08), run_time=2.0, rate_func=smooth)
        self.wait(0.5)

        # ── Return to a nice angle ────────────────────────────────────────────
        self.play(angle.animate.set_value(PI / 4), run_time=1.5, rate_func=smooth)
        self.wait(1.0)


# ── Display helper (used in Quarto cells) ────────────────────────────────────

_MIME_TYPES = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".gif":  "image/gif",
}


def show_dot_product(width: int = 800) -> None:
    """Embed the pre-rendered DotProductScene inline (Jupyter / Quarto).

    Accepts any format produced by Manim (mp4, webm, mov, gif).
    The most recently modified file is used, so switching FORMAT in the
    Makefile and re-running ``make media`` will automatically pick up the
    new output.  The video must have been rendered first via ``make media``.
    """
    from pathlib import Path
    from IPython.display import HTML, display as _display

    media_root = Path(__file__).parent.parent / "media"
    matches = [
        p for p in media_root.rglob("DotProductScene.*")
        if p.suffix in _MIME_TYPES
    ]
    if not matches:
        raise FileNotFoundError(
            f"No DotProductScene render found under media/. "
            f"Run 'make media' (or 'make') before rendering the slides."
        )
    media_path = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        rel = media_path.relative_to(Path.cwd())
    except ValueError:
        rel = media_path

    mime = _MIME_TYPES[media_path.suffix]
    if media_path.suffix == ".gif":
        html = (
            f'<img src="{rel}" width="{width}" '
            f'style="display:block;margin:auto">'
        )
    else:
        html = (
            f'<video width="{width}" autoplay loop muted playsinline controls '
            f'style="display:block;margin:auto">'
            f'<source src="{rel}" type="{mime}">'
            f"</video>"
        )
    _display(HTML(html))
