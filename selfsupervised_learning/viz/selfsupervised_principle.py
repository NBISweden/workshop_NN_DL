"""selfsupervised_principle.py
Animation paraphrase of LeCun's self-supervised learning principles slide.

A pseudo-3-D time-volume block cycles through six prediction strategies:
  0. Predict any part from any other part  (overview)
  1. Predict the future from the past
  2. Predict the future from the recent past
  3. Predict the past from the present
  4. Predict the top from the bottom
  5. Predict the occluded from the visible

Scene class: SelfsupervisedPrincipleScene
Embed helper: selfsupervised_principle_visualization()
"""


import numpy as np
import manim
from manim import (
    Scene, VGroup, Polygon, Text, Arrow,
    FadeIn, FadeOut, Create, Write, ReplacementTransform,
    DOWN, UP,
    WHITE, GREY_A, GREY_B, config,
)

# ── Render config ────────────────────────────────────────────────────────────
manim.config.pixel_width  = 1920
manim.config.pixel_height = 1080
manim.config.frame_width  = 16.0
manim.config.frame_height = 9.0

Text.set_default(font="Ubuntu")

# ── Palette ──────────────────────────────────────────────────────────────────
_BODY    = "#2a2a4e"   # block front face
_TOP_F   = "#1e1e40"   # top parallelogram
_SIDE_F  = "#181836"   # right-side parallelogram
_STROKE  = "#7a7aaa"   # edge lines
_CONTEXT = "#4a72cc"   # input / context region (blue)
_TARGET  = "#bf4fd8"   # predicted region        (purple-pink)
_PRESENT = "#d09030"   # present-moment marker   (amber)

# ── Block geometry ────────────────────────────────────────────────────────────
BW, BH   = 9.5, 2.8   # front-face width and height (scene units)
DX, DY   = 1.1, 0.50  # pseudo-perspective depth offsets
BCX, BCY = 0.0, -0.30  # front-face centre


def _front_corners(cx=BCX, cy=BCY, bw=BW, bh=BH):
    """Return (bl, br, tr, tl) corners of the front face."""
    hw, hh = bw / 2, bh / 2
    return (
        np.array([cx - hw, cy - hh, 0]),  # bottom-left
        np.array([cx + hw, cy - hh, 0]),  # bottom-right
        np.array([cx + hw, cy + hh, 0]),  # top-right
        np.array([cx - hw, cy + hh, 0]),  # top-left
    )


def make_block():
    """Three-face pseudo-3D block: front, top, right side."""
    bl, br, tr, tl = _front_corners()
    d = np.array([DX, DY, 0])
    kw = dict(stroke_color=_STROKE, stroke_width=1.8)
    front = Polygon(bl, br, tr, tl,     fill_color=_BODY,   fill_opacity=0.92, **kw)
    top   = Polygon(tl, tr, tr+d, tl+d, fill_color=_TOP_F,  fill_opacity=0.92, **kw)
    side  = Polygon(tr, tr+d, br+d, br, fill_color=_SIDE_F, fill_opacity=0.92, **kw)
    return VGroup(front, top, side)


def strip(x0, x1, y0=0.0, y1=1.0, color=_TARGET, alpha=0.83):
    """
    Colored rectangular region on the block front face.
    x0, x1, y0, y1 are fractions of BW / BH in [0, 1].
    """
    bl, *_ = _front_corners()
    left   = bl[0] + x0 * BW
    right  = bl[0] + x1 * BW
    bottom = bl[1] + y0 * BH
    top_y  = bl[1] + y1 * BH
    return Polygon(
        np.array([left,  bottom, 0]),
        np.array([right, bottom, 0]),
        np.array([right, top_y,  0]),
        np.array([left,  top_y,  0]),
        fill_color=color, fill_opacity=alpha,
        stroke_width=0,
    )


class SelfsupervisedPrincipleScene(Scene):
    def construct(self):
        block = make_block()

        # Time axis
        ax_y    = BCY - BH / 2 - 0.60
        t_start = np.array([BCX - BW / 2, ax_y, 0])
        t_end   = np.array([BCX + BW / 2, ax_y, 0])
        t_arrow = Arrow(t_start, t_end, color=WHITE, stroke_width=2.5,
                        buff=0, max_tip_length_to_length_ratio=0.04)
        lbl_past   = Text("Past",   font_size=26, color=GREY_B
                          ).next_to(t_start, DOWN, buff=0.2)
        lbl_future = Text("Future", font_size=26, color=GREY_B
                          ).next_to(t_end,   DOWN, buff=0.2)

        title = Text(
            "Predict any part  from any other part",
            font_size=46, color=WHITE,
        ).to_edge(UP, buff=0.50)

        # ── Intro ─────────────────────────────────────────────────────────────
        self.play(FadeIn(block), run_time=0.8)
        self.play(Create(t_arrow), FadeIn(lbl_past), FadeIn(lbl_future), run_time=0.7)
        self.play(Write(title), run_time=0.7)
        self.wait(1.2)

        # ── Strategy helper ───────────────────────────────────────────────────
        cur_title  = title
        cur_strips = VGroup()

        def show(label, strips, wait=2.2):
            nonlocal cur_title, cur_strips
            new_title  = Text(label, font_size=46, color=WHITE).to_edge(UP, buff=0.50)
            new_strips = VGroup(*strips)
            self.play(
                ReplacementTransform(cur_title, new_title),
                FadeOut(cur_strips),
                FadeIn(new_strips),
                run_time=0.75,
            )
            self.wait(wait)
            cur_title  = new_title
            cur_strips = new_strips

        # ── Strategies ────────────────────────────────────────────────────────
        show("1.  Predict the future  from the past", [
            strip(0.0, 0.5,  color=_CONTEXT),
            strip(0.5, 1.0,  color=_TARGET),
        ])

        show("2.  Predict the future  from the recent past", [
            strip(0.34, 0.52, color=_CONTEXT),   # recent-past window
            strip(0.52, 1.00, color=_TARGET),     # future
        ])

        show("3.  Predict the past  from the present", [
            strip(0.00, 0.47, color=_TARGET),     # past (to predict)
            strip(0.47, 0.57, color=_PRESENT),    # present (context)
        ])

        show("4.  Predict the top  from the bottom", [
            strip(0, 1, y0=0.00, y1=0.50, color=_CONTEXT),
            strip(0, 1, y0=0.50, y1=1.00, color=_TARGET),
        ])

        show("5.  Predict the occluded  from the visible", [
            strip(0.28, 0.62, y0=0.12, y1=0.88, color=_TARGET),
        ], wait=2.8)

        # ── Outro ─────────────────────────────────────────────────────────────
        outro = Text(
            "\"Pretend part of the input is unknown — predict it.\"",
            font_size=36, color=GREY_A,
        ).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(outro), run_time=0.7)
        self.wait(3.5)


def selfsupervised_principle_visualization(width: int = 1920) -> None:
    """Embed the pre-rendered SelfsupervisedPrincipleScene inline."""
    from ._media import _embed_render
    _embed_render("SelfsupervisedPrincipleScene", width, __file__)
