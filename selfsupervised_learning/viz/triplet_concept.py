"""
Contrastive / triplet learning concept — Manim animation.

Three real MNIST images (anchor, positive, negative) are shown on the left
with colour-coded borders.  An encoder box sits in the centre.  For each
image a coloured dot travels from the image → encoder input → encoder output
→ embedding grid, where the resulting arrow then grows.  Contrastive loss
then pulls same-class vectors together and pushes different-class ones apart.

Pure Manim — no training cache required (loads MNIST directly from
~/.cache/mnist, already downloaded by the training scripts).

Render:
    python -m manim -ql viz/triplet_concept.py TripletConceptScene
"""
from __future__ import annotations

import struct
import numpy as np

import manim
from manim import (
    BLACK, WHITE, GREY_B, GREY_C, YELLOW,
    ORIGIN, UP, DOWN, LEFT, RIGHT,
    Scene, Group, VGroup, ValueTracker,
    NumberPlane, Arrow, Dot, Line, Rectangle, Angle, MathTex, ImageMobject, Text,
    GrowArrow, Write, FadeIn, FadeOut,
    always_redraw, smooth, ManimColor,
)

# ── Render config ──────────────────────────────────────────────────────────────

PIXEL_WIDTH  = 1280
PIXEL_HEIGHT = 720
FRAME_WIDTH  = 14.2

manim.config.pixel_width  = PIXEL_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.frame_width  = FRAME_WIDTH

Text.set_default(font="Ubuntu")

# ── Colors — three distinct hues, one per role ─────────────────────────────────

ANCHOR_COLOR = ManimColor("#4C72B0")   # blue
POS_COLOR    = ManimColor("#55A868")   # green
NEG_COLOR    = ManimColor("#DD8452")   # orange

# ── Layout ─────────────────────────────────────────────────────────────────────

VEC_MAG = 2.3     # arrow length in scene units
IMG_H   = 1.4     # image height in scene units

IMG_X    = -5.5   # image column centre x
ENC_X    = -2.0   # encoder box centre x
ENC_W    =  1.8
ENC_H    =  5.0

# Grid is a NumberPlane shifted so its origin sits at GRID_CENTER.
GRID_CENTER = np.array([3.5, 0.0, 0.0])

ENC_LEFT  = np.array([ENC_X - ENC_W / 2, 0.0, 0.0])
ENC_RIGHT = np.array([ENC_X + ENC_W / 2, 0.0, 0.0])

Y_ANCHOR =  1.8
Y_POS    =  0.0
Y_NEG    = -1.8

# Angles of the three vectors in the embedding plane.
A_ANCHOR    = np.radians(25)
A_POS_INIT  = np.radians(115)   # 90° from anchor — orthogonal, cos = 0
A_NEG_INIT  = np.radians(255)   # 230° CCW from anchor (130° CW), cos ≈ −0.64
A_POS_FINAL = np.radians(30)    # 5° from anchor — tightly clustered, cos ≈ +1
A_NEG_FINAL = np.radians(200)   # 175° from anchor — nearly anti-parallel, cos ≈ −1


def _vec(angle: float, mag: float = VEC_MAG) -> np.ndarray:
    return GRID_CENTER + np.array([mag * np.cos(angle), mag * np.sin(angle), 0.0])


def _arrow(tip: np.ndarray, color: ManimColor) -> Arrow:
    return Arrow(GRID_CENTER, tip, buff=0, color=color,
                 stroke_width=5, max_tip_length_to_length_ratio=0.12)


def _border_rect(img: ImageMobject, color: ManimColor, buff: float = 0.07) -> Rectangle:
    return Rectangle(
        width=img.get_width()  + 2 * buff,
        height=img.get_height() + 2 * buff,
        color=color, stroke_width=3,
    ).move_to(img.get_center())


# ── MNIST loader ───────────────────────────────────────────────────────────────

def _load_mnist_images() -> dict[str, np.ndarray]:
    """Read MNIST test images directly from the raw binary cache."""
    from pathlib import Path

    raw = Path.home() / ".cache" / "mnist" / "MNIST" / "raw"

    def _read(fname: str) -> bytes:
        p = raw / fname
        if p.exists():
            return p.read_bytes()
        import gzip
        with gzip.open(raw / (fname + ".gz"), "rb") as f:
            return f.read()

    img_bytes = _read("t10k-images-idx3-ubyte")
    lbl_bytes = _read("t10k-labels-idx1-ubyte")

    _, n, h, w = struct.unpack_from(">IIII", img_bytes)
    imgs = np.frombuffer(img_bytes, dtype=np.uint8, offset=16).reshape(n, h, w)
    lbls = np.frombuffer(lbl_bytes, dtype=np.uint8, offset=8)

    sevens = np.where(lbls == 7)[0]
    threes = np.where(lbls == 3)[0]

    return {
        "anchor":   imgs[sevens[0]],
        "positive": imgs[sevens[3]],
        "negative": imgs[threes[0]],
    }


def _as_mob(arr: np.ndarray, height: float = IMG_H) -> ImageMobject:
    """Grayscale MNIST array → Manim ImageMobject."""
    rgba = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    mob  = ImageMobject(rgba)
    mob.set(height=height)
    return mob


# ── Scene ──────────────────────────────────────────────────────────────────────

class TripletConceptScene(Scene):
    def construct(self) -> None:
        self.camera.background_color = BLACK

        # ── Cartesian grid — right half of frame ──────────────────────────────
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-5, 5, 1],
            background_line_style={"stroke_color": GREY_C, "stroke_opacity": 0.35},
            axis_config={"stroke_color": GREY_B, "stroke_opacity": 0.7},
        ).shift(RIGHT * 3.5)
        self.add(plane)

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("Contrastive Learning", font_size=28, color=WHITE
                     ).to_edge(UP, buff=0.25)
        self.play(FadeIn(title), run_time=0.5)

        # ── Encoder box (middle column) — stays visible throughout ─────────────
        enc_box  = Rectangle(width=ENC_W, height=ENC_H, color=GREY_B, stroke_width=2
                             ).move_to([ENC_X, 0, 0])
        enc_lbl  = Text("Encoder", font_size=18, color=GREY_B
                        ).move_to([ENC_X,  0.28, 0])
        enc_func = MathTex(r"f(\cdot)", color=GREY_B, font_size=26
                           ).move_to([ENC_X, -0.28, 0])
        self.play(FadeIn(enc_box), FadeIn(enc_lbl), FadeIn(enc_func), run_time=0.5)

        # ── Input images with colour-coded borders — stay visible throughout ───
        img_arrs = _load_mnist_images()

        anchor_img = _as_mob(img_arrs["anchor"]).move_to([IMG_X, Y_ANCHOR, 0])
        pos_img    = _as_mob(img_arrs["positive"]).move_to([IMG_X, Y_POS,    0])
        neg_img    = _as_mob(img_arrs["negative"]).move_to([IMG_X, Y_NEG,    0])

        anchor_bdr = _border_rect(anchor_img, ANCHOR_COLOR)
        pos_bdr    = _border_rect(pos_img,    POS_COLOR)
        neg_bdr    = _border_rect(neg_img,    NEG_COLOR)

        a_role = Text("Anchor",   font_size=15, color=ANCHOR_COLOR
                      ).next_to(anchor_img, RIGHT, buff=0.15)
        p_role = Text("Positive", font_size=15, color=POS_COLOR
                      ).next_to(pos_img,    RIGHT, buff=0.15)
        n_role = Text("Negative", font_size=15, color=NEG_COLOR
                      ).next_to(neg_img,    RIGHT, buff=0.15)

        same_lbl = Text("same class", font_size=12, color=GREY_B
                        ).next_to(Group(anchor_img, pos_img), LEFT, buff=0.12)
        diff_lbl = Text("diff. class", font_size=12, color=GREY_B
                        ).next_to(neg_img, LEFT, buff=0.12)

        self.play(
            FadeIn(anchor_img), FadeIn(anchor_bdr),
            FadeIn(pos_img),    FadeIn(pos_bdr),
            FadeIn(neg_img),    FadeIn(neg_bdr),
            FadeIn(a_role), FadeIn(p_role), FadeIn(n_role),
            run_time=0.8,
        )
        self.play(FadeIn(same_lbl), FadeIn(diff_lbl), run_time=0.4)
        self.wait(1.0)

        # ── Encode: dot travels image → encoder (in) → encoder (out) → grid ───
        # Images and encoder box remain visible throughout.

        def encode(img: ImageMobject, angle: float, color: ManimColor) -> Arrow:
            # Dot enters encoder from the image side
            dot_in = Dot(img.get_right(), color=color, radius=0.13)
            self.play(FadeIn(dot_in), run_time=0.15)
            self.play(dot_in.animate.move_to(ENC_LEFT), run_time=0.38, rate_func=smooth)
            self.play(FadeOut(dot_in), run_time=0.12)

            # Dot exits encoder on the grid side and sweeps to GRID_CENTER
            dot_out = Dot(ENC_RIGHT, color=color, radius=0.13)
            self.play(FadeIn(dot_out), run_time=0.12)
            self.play(dot_out.animate.move_to(GRID_CENTER), run_time=0.60, rate_func=smooth)
            self.play(FadeOut(dot_out), run_time=0.12)

            # Arrow grows from GRID_CENTER outward
            arrow = _arrow(_vec(angle), color)
            self.play(GrowArrow(arrow), run_time=0.5)
            return arrow

        anchor_arrow = encode(anchor_img, A_ANCHOR,   ANCHOR_COLOR)
        self.wait(0.2)
        pos_arrow    = encode(pos_img,    A_POS_INIT,  POS_COLOR)
        self.wait(0.2)
        neg_arrow    = encode(neg_img,    A_NEG_INIT,  NEG_COLOR)

        self.wait(0.6)

        # ── Static anchor label ────────────────────────────────────────────────
        anchor_lbl = MathTex(r"f(x_a)", color=ANCHOR_COLOR, font_size=36
                             ).next_to(_vec(A_ANCHOR) + np.array([0.1, 0.1, 0]), buff=0.08)
        self.play(Write(anchor_lbl), run_time=0.4)

        # ── Attach updaters — pos and neg arrows follow ValueTrackers ──────────
        pos_tracker = ValueTracker(A_POS_INIT)
        neg_tracker = ValueTracker(A_NEG_INIT)

        pos_arrow.add_updater(
            lambda m: m.become(_arrow(_vec(pos_tracker.get_value()), POS_COLOR))
        )
        neg_arrow.add_updater(
            lambda m: m.become(_arrow(_vec(neg_tracker.get_value()), NEG_COLOR))
        )

        pos_lbl = always_redraw(lambda: MathTex(
            r"f(x_+)", color=POS_COLOR, font_size=36,
        ).next_to(_vec(pos_tracker.get_value()) + np.array([0.1, 0.1, 0]), buff=0.08))

        neg_lbl = always_redraw(lambda: MathTex(
            r"f(x_-)", color=NEG_COLOR, font_size=36,
        ).next_to(_vec(neg_tracker.get_value()) + np.array([0.1, 0.1, 0]), buff=0.08))

        self.add(pos_lbl, neg_lbl)

        # ── Angle arcs ─────────────────────────────────────────────────────────
        anchor_line = Line(GRID_CENTER, _vec(A_ANCHOR))

        pos_arc = always_redraw(lambda: Angle(
            anchor_line,
            Line(GRID_CENTER, _vec(pos_tracker.get_value())),
            radius=0.55, color=YELLOW, stroke_width=3,
        ))
        neg_arc = always_redraw(lambda: Angle(
            anchor_line,
            Line(GRID_CENTER, _vec(neg_tracker.get_value())),
            radius=0.90, color=NEG_COLOR, stroke_width=3,
        ))

        def _arc_label_pos(tracker: ValueTracker, radius: float, scale: float) -> np.ndarray:
            arc_mid = Angle(
                anchor_line,
                Line(GRID_CENTER, _vec(tracker.get_value())),
                radius=radius,
            ).point_from_proportion(0.5)
            return GRID_CENTER + (arc_mid - GRID_CENTER) * scale

        pos_theta = always_redraw(lambda: MathTex(
            r"\theta_+", color=YELLOW, font_size=30,
        ).move_to(_arc_label_pos(pos_tracker, 0.55, 1.75)))

        neg_theta = always_redraw(lambda: MathTex(
            r"\theta_-", color=NEG_COLOR, font_size=30,
        ).move_to(_arc_label_pos(neg_tracker, 0.90, 1.65)))

        self.play(FadeIn(pos_arc), FadeIn(neg_arc), run_time=0.5)
        self.add(pos_theta, neg_theta)

        # ── Cosine readouts ────────────────────────────────────────────────────
        cos_pos = always_redraw(lambda: MathTex(
            rf"\cos\,\theta_+ = {np.cos(pos_tracker.get_value() - A_ANCHOR):+.2f}",
            color=YELLOW, font_size=26,
        ).to_corner(UP + RIGHT, buff=0.4))

        cos_neg = always_redraw(lambda: MathTex(
            rf"\cos\,\theta_- = {np.cos(neg_tracker.get_value() - A_ANCHOR):+.2f}",
            color=NEG_COLOR, font_size=26,
        ).next_to(cos_pos, DOWN, buff=0.25).align_to(cos_pos, RIGHT))

        self.play(FadeIn(cos_pos), FadeIn(cos_neg), run_time=0.5)
        self.wait(1.5)

        # ── Loss annotation ────────────────────────────────────────────────────
        loss_lbl = Text(
            "Contrastive loss:  ↑ cos θ₊   (same class)     ↓ cos θ₋   (different class)",
            font_size=18, color=WHITE,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(loss_lbl), run_time=0.5)
        self.wait(1.0)

        # ── Optimization: pull positive close, push negative away ──────────────
        self.play(
            pos_tracker.animate.set_value(A_POS_FINAL),
            neg_tracker.animate.set_value(A_NEG_FINAL),
            run_time=3.2, rate_func=smooth,
        )
        self.wait(2.0)

        # ── Final note ─────────────────────────────────────────────────────────
        self.play(FadeOut(loss_lbl), run_time=0.3)
        note = Text(
            "High cos θ₊  →  same class         Low cos θ₋  →  different class",
            font_size=18, color=GREY_B,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(4.0)


def show_triplet_concept(width: int = 800) -> None:
    """Embed the pre-rendered TripletConceptScene inline (Jupyter / Quarto)."""
    from ._media import _embed_render
    _embed_render("TripletConceptScene", width, __file__)
