import matplotlib.pyplot as plt
import numpy as np

import manim
from manim import (
    Scene, Tex, VGroup,
    Rectangle, Text, Arrow, Dot, Line,
    FadeIn, GrowArrow, Transform,
    DOWN, UP, LEFT, RIGHT,
    GREY_B, TEAL_C, YELLOW, WHITE, BLUE, RED, config,
    smooth,
)

PIXEL_WIDTH = 1920
PIXEL_HEIGHT = 350
FRAME_WIDTH = 22

manim.config.frame_width = FRAME_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.pixel_width = PIXEL_WIDTH

Text.set_default(font="Ubuntu")


def cartesian_vs_polar():
    """Visualize cartesian vs. polar representation of two-class data."""
    rng = np.random.default_rng(seed=1729)
    r_a = rng.uniform(0, 0.95, 200)
    r_b = rng.uniform(1.05, 2, 200)
    theta_a = rng.uniform(0, 2 * np.pi, size=200)
    theta_b = rng.uniform(0, 2 * np.pi, size=200)

    x_a = r_a * np.cos(theta_a);  y_a = r_a * np.sin(theta_a)
    x_b = r_b * np.cos(theta_b);  y_b = r_b * np.sin(theta_b)

    fix, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    axes[0].scatter(x_a, y_a, color="#1f77b4", alpha=0.7, label="Class A")
    axes[0].scatter(x_b, y_b, color="#ff7f0e", alpha=0.7, label="Class B")
    axes[0].set_title("Cartesian Coordinates")
    axes[0].set_xticks([]);  axes[0].set_yticks([]);  axes[0].legend()
    axes[1].scatter(r_a, theta_a, color="#1f77b4", alpha=0.7, label="Class A")
    axes[1].scatter(r_b, theta_b, color="#ff7f0e", alpha=0.7, label="Class B")
    axes[1].set_title("Polar Coordinates")
    axes[1].set_xticks([]);  axes[1].set_yticks([]);  axes[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()


class AutoencoderScene(Scene):
    def construct(self):
        # ── Data ────────────────────────────────────────────────────────────
        rng = np.random.default_rng(seed=1729)
        n = 80
        r_a     = rng.uniform(0.00, 0.95, n)
        r_b     = rng.uniform(1.05, 2.00, n)
        theta_a = rng.uniform(0, 2 * np.pi, n)
        theta_b = rng.uniform(0, 2 * np.pi, n)

        # Cartesian (input / output)
        cx_a, cy_a = r_a * np.cos(theta_a), r_a * np.sin(theta_a)
        cx_b, cy_b = r_b * np.cos(theta_b), r_b * np.sin(theta_b)

        # Polar (bottleneck representation), scaled to ±2 range
        #   r ∈ [0, 2]  → x: (r − 1)·2  ∈ [−2,  2]
        #   θ ∈ [0, 2π] → y: θ/π·2 − 2  ∈ [−2,  2]
        px_a = (r_a - 1.0) * 2.0;  py_a = theta_a / np.pi * 2.0 - 2.0
        px_b = (r_b - 1.0) * 2.0;  py_b = theta_b / np.pi * 2.0 - 2.0

        # ── Layout ──────────────────────────────────────────────────────────
        PSIZ = 2.4             # panel side length (scene units)
        PANEL_CLEARANCE = 0.6  # space between panels
        Y0   = 0.3             # shift up for label clearance
        N_PANELS = 7           # Input + Enc1 + Enc2 + Bottleneck + Dec1 + Dec2 + Output
        DSCL = PSIZ / 4.8      # maps ±2.4 data units to just inside panel

        ARRAY_LENGTH = N_PANELS * PSIZ + (N_PANELS - 1) * PANEL_CLEARANCE
        LC_RC_DISTANCE = ARRAY_LENGTH - PSIZ
        LC = np.array([-LC_RC_DISTANCE / 2, Y0, 0.0])
        NN_OFFSET = np.array([PSIZ + PANEL_CLEARANCE, 0.0, 0.0])

        ENC1 = LC + NN_OFFSET
        ENC2 = LC + NN_OFFSET * 2
        BN   = LC + NN_OFFSET * 3   # Bottleneck (centre)
        DEC1 = LC + NN_OFFSET * 4
        DEC2 = LC + NN_OFFSET * 5
        RC   = LC + NN_OFFSET * 6

        def d2s(x, y, ctr):
            """Convert data coordinates to a Manim scene position."""
            return ctr + np.array([x * DSCL, y * DSCL, 0.0])

        # ── Panels ──────────────────────────────────────────────────────────
        def make_panel(ctr, lbl_txt, highlight=False):
            stroke_col = TEAL_C if highlight else GREY_B
            fill_col   = "#102828" if highlight else "#101028"
            stroke_w   = 2.5 if highlight else 1.5
            rect = Rectangle(width=PSIZ, height=PSIZ,
                             stroke_color=stroke_col, stroke_width=stroke_w,
                             fill_color=fill_col, fill_opacity=0.4)
            rect.move_to(ctr)
            lbl = Tex(lbl_txt, font_size=36).next_to(rect, DOWN, buff=0.2)
            return rect, lbl

        in_rect,   in_lbl   = make_panel(LC,   "Input")
        enc1_rect, enc1_lbl = make_panel(ENC1, "Encoder 1")
        enc2_rect, enc2_lbl = make_panel(ENC2, "Encoder 2")
        bn_rect,   bn_lbl   = make_panel(BN,   "Bottleneck", highlight=True)
        dec1_rect, dec1_lbl = make_panel(DEC1, "Decoder 1")
        dec2_rect, dec2_lbl = make_panel(DEC2, "Decoder 2")
        out_rect,  out_lbl  = make_panel(RC,   "Output")

        # ── Arrows ──────────────────────────────────────────────────────────
        AKW = dict(color=YELLOW, stroke_width=6,
                   max_tip_length_to_length_ratio=0.2, buff=0.12)
        arr_in_e1  = Arrow(in_rect.get_right(),   enc1_rect.get_left(),  **AKW)
        arr_e1_e2  = Arrow(enc1_rect.get_right(), enc2_rect.get_left(),  **AKW)
        arr_e2_bn  = Arrow(enc2_rect.get_right(), bn_rect.get_left(),    **AKW)
        arr_bn_d1  = Arrow(bn_rect.get_right(),   dec1_rect.get_left(),  **AKW)
        arr_d1_d2  = Arrow(dec1_rect.get_right(), dec2_rect.get_left(),  **AKW)
        arr_d2_out = Arrow(dec2_rect.get_right(), out_rect.get_left(),   **AKW)

        # ── Intermediate representations ─────────────────────────────────────
        # Encoder: linearly interpolate cartesian → polar
        e1_alpha = 0.2
        e1_xa_t = (1 - e1_alpha) * cx_a + e1_alpha * px_a
        e1_ya_t = (1 - e1_alpha) * cy_a + e1_alpha * py_a
        e1_xb_t = (1 - e1_alpha) * cx_b + e1_alpha * px_b
        e1_yb_t = (1 - e1_alpha) * cy_b + e1_alpha * py_b

        e2_alpha = 0.5
        e2_xa_t = (1 - e2_alpha) * cx_a + e2_alpha * px_a
        e2_ya_t = (1 - e2_alpha) * cy_a + e2_alpha * py_a
        e2_xb_t = (1 - e2_alpha) * cx_b + e2_alpha * px_b
        e2_yb_t = (1 - e2_alpha) * cy_b + e2_alpha * py_b

        # Bottleneck = full polar
        bn_xa_t, bn_ya_t = px_a, py_a
        bn_xb_t, bn_yb_t = px_b, py_b

        # Decoder: mirror encoder steps back to cartesian
        d1_xa_t, d1_ya_t = e2_xa_t, e2_ya_t   # mirrors Enc2 target
        d1_xb_t, d1_yb_t = e2_xb_t, e2_yb_t
        d2_xa_t, d2_ya_t = e1_xa_t, e1_ya_t   # mirrors Enc1 target
        d2_xb_t, d2_yb_t = e1_xb_t, e1_yb_t
        noise_scale = 0.18
        out_xa_t = cx_a + rng.normal(0, noise_scale, n)   # imperfect reconstruction
        out_ya_t = cy_a + rng.normal(0, noise_scale, n)
        out_xb_t = cx_b + rng.normal(0, noise_scale, n)
        out_yb_t = cy_b + rng.normal(0, noise_scale, n)

        # ── Dots ────────────────────────────────────────────────────────────
        DR = 0.03
        CA, CB = "#5ab4e8", "#ffa060"   # blue / orange

        def dot_group(xs, ys, col, ctr):
            return VGroup(*[
                Dot(d2s(x, y, ctr), radius=DR, color=col, fill_opacity=0.8)
                for x, y in zip(xs, ys)
            ])

        def move_anims(dots_a, dots_b, xa_t, ya_t, xb_t, yb_t, ctr):
            return (
                [d.animate.move_to(d2s(x, y, ctr)) for d, x, y in zip(dots_a, xa_t, ya_t)]
                + [d.animate.move_to(d2s(x, y, ctr)) for d, x, y in zip(dots_b, xb_t, yb_t)]
            )

        # Each panel's dots start at the coordinates the previous panel ended at
        in_dots_a   = dot_group(cx_a,    cy_a,    CA, LC)
        in_dots_b   = dot_group(cx_b,    cy_b,    CB, LC)

        enc1_dots_a = dot_group(cx_a,    cy_a,    CA, ENC1)
        enc1_dots_b = dot_group(cx_b,    cy_b,    CB, ENC1)

        enc2_dots_a = dot_group(e1_xa_t, e1_ya_t, CA, ENC2)
        enc2_dots_b = dot_group(e1_xb_t, e1_yb_t, CB, ENC2)

        bn_dots_a   = dot_group(e2_xa_t, e2_ya_t, CA, BN)
        bn_dots_b   = dot_group(e2_xb_t, e2_yb_t, CB, BN)

        dec1_dots_a = dot_group(bn_xa_t, bn_ya_t, CA, DEC1)
        dec1_dots_b = dot_group(bn_xb_t, bn_yb_t, CB, DEC1)

        dec2_dots_a = dot_group(d1_xa_t, d1_ya_t, CA, DEC2)
        dec2_dots_b = dot_group(d1_xb_t, d1_yb_t, CB, DEC2)

        out_dots_a  = dot_group(d2_xa_t, d2_ya_t, CA, RC)
        out_dots_b  = dot_group(d2_xb_t, d2_yb_t, CB, RC)

        # ── Scene assembly ───────────────────────────────────────────────────
        # Input
        self.play(FadeIn(in_rect), FadeIn(in_lbl), run_time=0.8)
        self.play(FadeIn(in_dots_a), FadeIn(in_dots_b), run_time=0.7)

        # Encoder 1
        self.play(FadeIn(enc1_rect), FadeIn(enc1_lbl),
                  FadeIn(enc1_dots_a), FadeIn(enc1_dots_b), run_time=0.7)
        self.play(GrowArrow(arr_in_e1),
                  *move_anims(enc1_dots_a, enc1_dots_b,
                              e1_xa_t, e1_ya_t, e1_xb_t, e1_yb_t, ENC1),
                  run_time=1.5, rate_func=smooth)

        # Encoder 2
        self.play(FadeIn(enc2_rect), FadeIn(enc2_lbl),
                  FadeIn(enc2_dots_a), FadeIn(enc2_dots_b), run_time=0.7)
        self.play(GrowArrow(arr_e1_e2),
                  *move_anims(enc2_dots_a, enc2_dots_b,
                              e2_xa_t, e2_ya_t, e2_xb_t, e2_yb_t, ENC2),
                  run_time=1.5, rate_func=smooth)

        # Bottleneck
        self.play(FadeIn(bn_rect), FadeIn(bn_lbl),
                  FadeIn(bn_dots_a), FadeIn(bn_dots_b), run_time=0.7)
        self.play(GrowArrow(arr_e2_bn),
                  *move_anims(bn_dots_a, bn_dots_b,
                              bn_xa_t, bn_ya_t, bn_xb_t, bn_yb_t, BN),
                  run_time=1.5, rate_func=smooth)

        self.wait(1.0)

        # Decoder 1
        self.play(FadeIn(dec1_rect), FadeIn(dec1_lbl),
                  FadeIn(dec1_dots_a), FadeIn(dec1_dots_b), run_time=0.7)
        self.play(GrowArrow(arr_bn_d1),
                  *move_anims(dec1_dots_a, dec1_dots_b,
                              d1_xa_t, d1_ya_t, d1_xb_t, d1_yb_t, DEC1),
                  run_time=1.5, rate_func=smooth)

        # Decoder 2
        self.play(FadeIn(dec2_rect), FadeIn(dec2_lbl),
                  FadeIn(dec2_dots_a), FadeIn(dec2_dots_b), run_time=0.7)
        self.play(GrowArrow(arr_d1_d2),
                  *move_anims(dec2_dots_a, dec2_dots_b,
                              d2_xa_t, d2_ya_t, d2_xb_t, d2_yb_t, DEC2),
                  run_time=1.5, rate_func=smooth)

        # Output (reconstruction)
        self.play(FadeIn(out_rect), FadeIn(out_lbl),
                  FadeIn(out_dots_a), FadeIn(out_dots_b), run_time=0.7)
        self.play(GrowArrow(arr_d2_out),
                  *move_anims(out_dots_a, out_dots_b,
                              out_xa_t, out_ya_t, out_xb_t, out_yb_t, RC),
                  run_time=1.5, rate_func=smooth)

        self.wait(4)


def autoencoder_visualization(width: int = PIXEL_WIDTH) -> None:
    """Embed the pre-rendered AutoencoderScene inline (Jupyter / Quarto)."""
    from ._media import _embed_render
    _embed_render("AutoencoderScene", width, __file__)


if __name__ == "__main__":
    cartesian_vs_polar()
