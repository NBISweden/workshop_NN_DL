import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from typing import Literal
import numpy as np

import manim
from manim import (
    Scene, Tex, VGroup,
    Rectangle, Text, Arrow, Dot, Line, SurroundingRectangle,
    FadeIn, Create, Write, GrowArrow, Transform,
    DOWN, UP, LEFT, RIGHT,
    GREY_B, TEAL_C, YELLOW, WHITE, BLUE, RED, config,
    smooth,
)

PIXEL_WIDTH = 1280
PIXEL_HEIGHT = 300
FRAME_WIDTH = 17

manim.config.frame_width = FRAME_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.pixel_width = PIXEL_WIDTH
#manim.config.background_color = "#262669"

Text.set_default(font="Ubuntu")

def cartesian_vs_polar():
    """
    Visualize the difference between representing data cartesian vs. polar coordinates.
    This is a simplified example of how way data is represented affect how easy a problem is to solve
    """
    rng = np.random.default_rng(seed=1729)
    r_a = rng.uniform(0, 0.95, 200)
    r_b = rng.uniform(1.05, 2, 200)
    theta_a = rng.uniform(0, 2 * np.pi, size=200)
    theta_b = rng.uniform(0, 2 * np.pi, size=200)

    x_a = r_a * np.cos(theta_a)
    y_a = r_a * np.sin(theta_a)
    x_b = r_b * np.cos(theta_b)
    y_b = r_b * np.sin(theta_b)

    fix, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    axes[0].scatter(x_a, y_a, color="#1f77b4", alpha=0.7, label="Class A")
    axes[0].scatter(x_b, y_b, color="#ff7f0e", alpha=0.7, label="Class B")
    axes[0].set_title("Cartesian Coordinates")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend()
    axes[1].scatter(r_a, theta_a, color="#1f77b4", alpha=0.7, label="Class A")
    axes[1].scatter(r_b, theta_b, color="#ff7f0e", alpha=0.7, label="Class B")
    axes[1].set_title("Polar Coordinates")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()


class RepresentationLearningScene(Scene):
    def construct(self):
        # ── Data (same seed as cartesian_vs_polar) ───────────────────────────
        rng = np.random.default_rng(seed=1729)
        n = 80
        r_a     = rng.uniform(0.00, 0.95, n)
        r_b     = rng.uniform(1.05, 2.00, n)
        theta_a = rng.uniform(0, 2 * np.pi, n)
        theta_b = rng.uniform(0, 2 * np.pi, n)

        # Cartesian (input)
        cx_a, cy_a = r_a * np.cos(theta_a), r_a * np.sin(theta_a)
        cx_b, cy_b = r_b * np.cos(theta_b), r_b * np.sin(theta_b)

        # Polar target — scaled to the same ±2 range as cartesian so the
        # linear interpolation through the 3 layers stays within the panel.
        #   r ∈ [0, 2]   →  x: (r − 1)·2  ∈ [−2,  2]
        #   θ ∈ [0, 2π]  →  y: θ/π·2 − 2  ∈ [−2,  2]
        px_a = (r_a - 1.0) * 2.0;  py_a = theta_a / np.pi * 2.0 - 2.0
        px_b = (r_b - 1.0) * 2.0;  py_b = theta_b / np.pi * 2.0 - 2.0

        # ── Layout ───────────────────────────────────────────────────────────

        PSIZ = 2.4          # panel side length (scene units)
        DSCL = PSIZ / 4.8    # maps ±2.4 data units to just inside the panel
        Y0   = 0.25           # shift everything up slightly for label clearance
        PANEL_CLEARANCE = 1     # space between panel edge and data points
        N_PANELS = 5    # The input panel + 3 layers + output panel
        ARRAY_LENGTH = N_PANELS * PSIZ + (N_PANELS - 1) * PANEL_CLEARANCE
        LC_RC_DISTANCE = ARRAY_LENGTH - PSIZ  # distance between panel centres
        LC   = np.array([-LC_RC_DISTANCE/2, Y0, 0.0])   # left  panel centre
        RC   = np.array([LC_RC_DISTANCE/2, Y0, 0.0])   # right panel centre
        
        NN_OFFSET = np.array([PSIZ + PANEL_CLEARANCE, 0.0, 0.0])  # from left panel centre to NN centre
        L1 = LC + NN_OFFSET
        L2 = LC + NN_OFFSET * 2
        L3 = LC + NN_OFFSET * 3

        def d2s(x, y, ctr):
            """Convert data coordinates to a Manim scene position."""
            return ctr + np.array([x * DSCL, y * DSCL, 0.0])

        # ── Panels ───────────────────────────────────────────────────────────
        def make_panel(ctr, lbl_txt):
            rect = Rectangle(width=PSIZ, height=PSIZ,
                             stroke_color=GREY_B, stroke_width=1.5,
                             fill_color="#101028", fill_opacity=0.4)
            rect.move_to(ctr)
            lbl = Tex(lbl_txt, font_size=42).next_to(rect, DOWN, buff=0.2)
            return rect, lbl

        l_rect, l_lbl = make_panel(LC, "Input ")
        r_rect, r_lbl = make_panel(RC, "Classifier")

        # ── Neural-network block (3 layer boxes in a frame) ──────────────────
        # BOX_W, BOX_H, GAP = 1.2, 2.4, 0.1

        
        # nn_labels = VGroup()
        # for i, name in enumerate(["Layer 1", "Layer 2", "Layer 3"]):
        #     box = Rectangle(width=BOX_W, height=BOX_H,
        #                     fill_color=TEAL_C, fill_opacity=0.2,
        #                     stroke_color=TEAL_C, stroke_width=2.0)
        #     box.move_to(np.array([(i - 1) * (BOX_W + GAP), Y0, 0.0]))
        #     nn_boxes.add(box)
        #     nn_labels.add(Text(name, font_size=11).next_to(box, DOWN, buff=0.10))

        nn_boxes  = VGroup()
        nn_labels = VGroup()

        layer_1, layer1_lbl = make_panel(L1, "Layer 1")
        layer_2, layer2_lbl = make_panel(L2, "Layer 2")
        layer_3, layer3_lbl = make_panel(L3, "Layer 3")

        #nn_frame = SurroundingRectangle(nn_boxes, color=GREY_B, buff=0.2, corner_radius=0.1)
        #nn_title = Text("Neural Network", font_size=16).next_to(nn_frame, DOWN, buff=0.15)

        # ── Arrows ───────────────────────────────────────────────────────────
        AKW = dict(color=YELLOW, stroke_width=6,
                   max_tip_length_to_length_ratio=0.2, buff=0.12)
        arr_l_l1 = Arrow(l_rect.get_right(), layer_1.get_left(), **AKW)
        arr_l1_l2 = Arrow(layer_1.get_right(), layer_2.get_left(), **AKW)
        arr_l2_l3 = Arrow(layer_2.get_right(), layer_3.get_left(), **AKW)
        arr_l3_r = Arrow(layer_3.get_right(), r_rect.get_left(), **AKW)
        

        # ── Dots ─────────────────────────────────────────────────────────────
        DR = 0.03
        CA, CB = "#5ab4e8", "#ffa060"   # blue / orange — visible on dark panel

        def dot_group(xs, ys, col, ctr):
            return VGroup(*[
                Dot(d2s(x, y, ctr), radius=DR, color=col, fill_opacity=0.8)
                for x, y in zip(xs, ys)
            ])

        # Left panel — always shows the cartesian input (static)
        l_dots_a = dot_group(cx_a, cy_a, CA, LC)
        l_dots_b = dot_group(cx_b, cy_b, CB, LC)
        r_dots_a = dot_group(px_a, py_a, CA, RC)
        r_dots_b = dot_group(px_b, py_b, CB, RC)
        
        # First layer dots
        l1_dots_a = dot_group(cx_a, cy_a, CA, L1)
        l1_dots_b = dot_group(cx_b, cy_b, CB, L1)

        l1_alpha = 0.2
        l1_xa_t = (1-l1_alpha) * cx_a + l1_alpha * px_a
        l1_ya_t = (1-l1_alpha) * cy_a + l1_alpha * py_a
        l1_xb_t = (1-l1_alpha) * cx_b + l1_alpha * px_b
        l1_yb_t = (1-l1_alpha) * cy_b + l1_alpha * py_b
        
        # Second layer dots
        l2_dots_a = dot_group(l1_xa_t, l1_ya_t, CA, L2)
        l2_dots_b = dot_group(l1_xb_t, l1_yb_t, CB, L2)
        l2_alpha = 0.5
        l2_xa_t = (1-l2_alpha) * cx_a + l2_alpha * px_a
        l2_ya_t = (1-l2_alpha) * cy_a + l2_alpha * py_a
        l2_xb_t = (1-l2_alpha) * cx_b + l2_alpha * px_b
        l2_yb_t = (1-l2_alpha) * cy_b + l2_alpha * py_b
        
        # Third layer dots
        l3_dots_a = dot_group(l2_xa_t, l2_ya_t, CA, L3)
        l3_dots_b = dot_group(l2_xb_t, l2_yb_t, CB, L3)
        l3_xa_t = px_a
        l3_ya_t = py_a
        l3_xb_t = px_b
        l3_yb_t = py_b

        l1_move_anims = (
                 [d.animate.move_to(d2s(x, y, L1)) for d, x, y in zip(l1_dots_a, l1_xa_t, l1_ya_t)]
                 + [d.animate.move_to(d2s(x, y, L1)) for d, x, y in zip(l1_dots_b, l1_xb_t, l1_yb_t)]
             )
        l2_move_anims = (
                 [d.animate.move_to(d2s(x, y, L2)) for d, x, y in zip(l2_dots_a, l2_xa_t, l2_ya_t)]
                 + [d.animate.move_to(d2s(x, y, L2)) for d, x, y in zip(l2_dots_b, l2_xb_t, l2_yb_t)]
             )
        l3_move_anims = (
                 [d.animate.move_to(d2s(x, y, L3)) for d, x, y in zip(l3_dots_a, l3_xa_t, l3_ya_t)]
                 + [d.animate.move_to(d2s(x, y, L3)) for d, x, y in zip(l3_dots_b, l3_xb_t, l3_yb_t)]
             )

        # ── Separator line ───────────────────────────────────────────────────
        # In polar space class A has r < 1 (px < 0) and class B has r > 1 (px > 0),
        # so the true decision boundary is a vertical line at data x=0 → RC centre.
        # We first show it as a diagonal (wrong orientation) then rotate it into place.
        half = PSIZ / 2 * 0.88   # stay just inside the panel edges
        sep_initial = Line(
            RC + np.array([-half, -half, 0.0]),
            RC + np.array([ half,  half, 0.0]),
            color=WHITE, stroke_width=4,
        )
        sep_final = Line(
            RC + np.array([0.0, -half, 0.0]),
            RC + np.array([0.0,  half, 0.0]),
            color=WHITE, stroke_width=4,
        )

        # ── Scene assembly ────────────────────────────────────────────────────
        self.play(
            FadeIn(l_rect), FadeIn(l_lbl),
            #FadeIn(r_rect), FadeIn(r_lbl),
            run_time=0.8,
        )
        self.play(FadeIn(l_dots_a), FadeIn(l_dots_b), run_time=0.7)
        self.play(FadeIn(layer_1), FadeIn(layer1_lbl), FadeIn(l1_dots_a), FadeIn(l1_dots_b), run_time=0.7)
        self.play(GrowArrow(arr_l_l1), *l1_move_anims, run_time=1.5, rate_func=smooth)
        self.play(FadeIn(layer_2), FadeIn(layer2_lbl), FadeIn(l2_dots_a), FadeIn(l2_dots_b),run_time=0.7)
        self.play(GrowArrow(arr_l1_l2), *l2_move_anims, run_time=1.5, rate_func=smooth)
        self.play(FadeIn(layer_3), FadeIn(layer3_lbl), FadeIn(l3_dots_a), FadeIn(l3_dots_b),run_time=0.7)
        self.play(GrowArrow(arr_l2_l3), *l3_move_anims, run_time=1.5, rate_func=smooth)
        self.play(
            FadeIn(r_rect), FadeIn(r_lbl),
            FadeIn(r_dots_a), FadeIn(r_dots_b),
            FadeIn(sep_initial),
            run_time=0.8,
        )
        self.play(
            GrowArrow(arr_l3_r),
            Transform(sep_initial, sep_final),
            run_time=1.5, rate_func=smooth,
        )
        
        #self.play(GrowArrow(arr_l), GrowArrow(arr_r), run_time=0.7)
        #self.play(FadeIn(r_dots_a), FadeIn(r_dots_b), run_time=0.5)
        self.wait(4)

        # ── Layer-by-layer transformation ─────────────────────────────────────
        # Each step linearly interpolates 1/3 of the way from cartesian to polar.
        # After all 3 steps the right panel shows the fully polar representation,
        # where the two classes are separated along the x-axis (radius dimension).
        # for step in range(3):
        #     t = (step + 1) / 3.0
        #     box = nn_boxes[step]

        #     # Highlight the active layer
        #     self.play(
        #         box.animate.set_fill(YELLOW, opacity=0.45).set_stroke(YELLOW, width=3.0),
        #         run_time=0.4,
        #     )

        #     # Interpolated coordinates at this step
        #     xa_t = (1 - t) * cx_a + t * px_a;  ya_t = (1 - t) * cy_a + t * py_a
        #     xb_t = (1 - t) * cx_b + t * px_b;  yb_t = (1 - t) * cy_b + t * py_b

        #     move_anims = (
        #         [d.animate.move_to(d2s(x, y, RC)) for d, x, y in zip(r_dots_a, xa_t, ya_t)]
        #         + [d.animate.move_to(d2s(x, y, RC)) for d, x, y in zip(r_dots_b, xb_t, yb_t)]
        #     )
        #     self.play(*move_anims, run_time=1.5, rate_func=smooth)

        #     # Dim back to show layer is "done" but processed
        #     self.play(
        #         box.animate.set_fill(TEAL_C, opacity=0.45).set_stroke(TEAL_C, width=2.0),
        #         run_time=0.3,
        #     )
        #     self.wait(0.25)

        # self.wait(1.0)


def neural_network_representation_learning(width: int = PIXEL_WIDTH) -> None:
    """Embed the pre-rendered RepresentationLearningScene inline (Jupyter / Quarto)."""
    from ._media import _embed_render
    _embed_render("RepresentationLearningScene", width, __file__)


if __name__ == "__main__":
    cartesian_vs_polar()
