import numpy as np
import manim
from manim import (
    Scene, Tex, VGroup, AnimationGroup,
    Text, Rectangle, Arrow, Dot, Line,
    FadeIn, FadeOut, GrowArrow, Transform,
    DOWN, UP, LEFT, RIGHT,
    GREY_B, TEAL_C, YELLOW, WHITE, config,
    smooth,
)

ratio = 1280/780
PIXEL_WIDTH  = 1280
PIXEL_HEIGHT = int(PIXEL_WIDTH // ratio)
FRAME_WIDTH  = 15

manim.config.frame_width  = FRAME_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.pixel_width  = PIXEL_WIDTH

Text.set_default(font="Ubuntu")

_PSIZ            = 2.0
_PANEL_CLEARANCE = 0.8
_N_PANELS        = 5
_ARRAY_LEN       = _N_PANELS * _PSIZ + (_N_PANELS - 1) * _PANEL_CLEARANCE
_LC_RC_DIST      = _ARRAY_LEN - _PSIZ   # distance between first and last panel centres
_DSCL            = _PSIZ / 4.8          # data ±2.4 → panel interior
_NN_STEP         = np.array([_PSIZ + _PANEL_CLEARANCE, 0.0, 0.0])
_HALF_SEP        = _PSIZ / 2 * 0.88    # separator line half-length (stays inside panel)
_ROW_1_Y       = 2.0
_ROW_2_Y       = -2.0
_CA, _CB, _CC = "#5ab4e8", "#ffa060", "#7bc87b"
_DOT_R = 0.03
_AKW   = dict(color=YELLOW, stroke_width=5,
              max_tip_length_to_length_ratio=0.2, buff=0.12)

# Interpolation alphas (shared across rows — same network transformations)
_A1, _A2 = 0.2, 0.5


class TransferLearningScene(Scene):
    def construct(self):
        rng = np.random.default_rng(seed=1729)

        # ── Layout helpers ────────────────────────────────────────────────────
        def row_centers(y0):
            lc = np.array([-_LC_RC_DIST / 2, y0, 0.0])
            return lc, lc + _NN_STEP, lc + _NN_STEP * 2, lc + _NN_STEP * 3, lc + _NN_STEP * 4

        LC1, L11, L21, L31, RC1 = row_centers(_ROW_1_Y)
        LC2, L12, L22, L32, RC2 = row_centers(_ROW_2_Y)

        def d2s(x, y, ctr):
            return ctr + np.array([x * _DSCL, y * _DSCL, 0.0])

        def dot_group(xs, ys, col, ctr):
            return VGroup(*[
                Dot(d2s(x, y, ctr), radius=_DOT_R, color=col, fill_opacity=0.8)
                for x, y in zip(xs, ys)
            ])

        def make_panel(ctr, lbl_txt, next_to_direction=UP, border=GREY_B):
            rect = Rectangle(width=_PSIZ, height=_PSIZ,
                             stroke_color=border, stroke_width=1.5,
                             fill_color="#101028", fill_opacity=0.4)
            rect.move_to(ctr)
            lbl = Tex(lbl_txt, font_size=34).next_to(rect, next_to_direction, buff=0.2)
            return rect, lbl

        def make_lr_arrow(src, dst):
            return Arrow(src.get_right(), dst.get_left(), **_AKW)
        
        def make_ud_arrow(src, dst):
            return Arrow(src.get_bottom(), dst.get_top(), **_AKW)
        
        def make_label(arrow, text, color=YELLOW, font_size=22):
            return Tex(text, font_size=font_size, color=color).next_to(arrow, RIGHT, buff=0.15)

        def interp(xs_c, xs_p, alpha):
            return (1 - alpha) * xs_c + alpha * xs_p

        # ── Row 1 data: 2-class circular ─────────────────────────────────────
        n = 80
        r_a  = rng.uniform(0.00, 0.95, n)
        r_b  = rng.uniform(1.05, 2.00, n)
        th_a = rng.uniform(0, 2 * np.pi, n)
        th_b = rng.uniform(0, 2 * np.pi, n)

        cx_a, cy_a = r_a * np.cos(th_a), r_a * np.sin(th_a)
        cx_b, cy_b = r_b * np.cos(th_b), r_b * np.sin(th_b)
        px_a = (r_a - 1.0) * 2.0;  py_a = th_a / np.pi * 2.0 - 2.0
        px_b = (r_b - 1.0) * 2.0;  py_b = th_b / np.pi * 2.0 - 2.0

        # ── Row 2 data: 3-class circular ─────────────────────────────────────
        # In polar space: class 1 (inner) → px ≈ [-2, -0.96],
        #                 class 2 (middle) → px ≈ [-0.36, 0.36],
        #                 class 3 (outer)  → px ≈ [ 0.96,  2.0]
        n2 = 55
        r_1  = rng.uniform(0.00, 0.52, n2)
        r_2  = rng.uniform(0.82, 1.18, n2)
        r_3  = rng.uniform(1.48, 2.00, n2)
        th_1 = rng.uniform(0, 2 * np.pi, n2)
        th_2 = rng.uniform(0, 2 * np.pi, n2)
        th_3 = rng.uniform(0, 2 * np.pi, n2)

        cx_1, cy_1 = r_1 * np.cos(th_1), r_1 * np.sin(th_1)
        cx_2, cy_2 = r_2 * np.cos(th_2), r_2 * np.sin(th_2)
        cx_3, cy_3 = r_3 * np.cos(th_3), r_3 * np.sin(th_3)
        px_1 = (r_1 - 1.0) * 2.0;  py_1 = th_1 / np.pi * 2.0 - 2.0
        px_2 = (r_2 - 1.0) * 2.0;  py_2 = th_2 / np.pi * 2.0 - 2.0
        px_3 = (r_3 - 1.0) * 2.0;  py_3 = th_3 / np.pi * 2.0 - 2.0

        # ── Row 1 panels and arrows ───────────────────────────────────────────
        in1,   in1_lbl  = make_panel(LC1, "Input")
        nn1_1, nn1_1l  = make_panel(L11, "Layer 1")
        nn1_2, nn1_2l  = make_panel(L21, "Layer 2")
        nn1_3, nn1_3l  = make_panel(L31, "Layer 3")
        out1,  out1_lbl = make_panel(RC1, "Classifier")

        arr1_in_l1  = make_lr_arrow(in1,   nn1_1)
        arr1_l1_l2  = make_lr_arrow(nn1_1, nn1_2)
        arr1_l2_l3  = make_lr_arrow(nn1_2, nn1_3)
        arr1_l3_out = make_lr_arrow(nn1_3, out1)

        # ── Row 1 dots ────────────────────────────────────────────────────────
        in1_da = dot_group(cx_a, cy_a, _CA, LC1)
        in1_db = dot_group(cx_b, cy_b, _CB, LC1)

        l1a_xa = interp(cx_a, px_a, _A1);  l1a_ya = interp(cy_a, py_a, _A1)
        l1a_xb = interp(cx_b, px_b, _A1);  l1a_yb = interp(cy_b, py_b, _A1)
        l1_da  = dot_group(cx_a, cy_a, _CA, L11)
        l1_db  = dot_group(cx_b, cy_b, _CB, L11)

        l2a_xa = interp(cx_a, px_a, _A2);  l2a_ya = interp(cy_a, py_a, _A2)
        l2a_xb = interp(cx_b, px_b, _A2);  l2a_yb = interp(cy_b, py_b, _A2)
        l2_da  = dot_group(l1a_xa, l1a_ya, _CA, L21)
        l2_db  = dot_group(l1a_xb, l1a_yb, _CB, L21)

        l3_da    = dot_group(l2a_xa, l2a_ya, _CA, L31)
        l3_db    = dot_group(l2a_xb, l2a_yb, _CB, L31)
        out1_da  = dot_group(px_a, py_a, _CA, RC1)
        out1_db  = dot_group(px_b, py_b, _CB, RC1)

        # ── Row 1 separator ───────────────────────────────────────────────────
        h = _HALF_SEP
        sep1_i = Line(RC1 + np.array([-h, -h, 0.0]), RC1 + np.array([h,  h, 0.0]),
                      color=WHITE, stroke_width=4)
        sep1_f = Line(RC1 + np.array([0.0, -h, 0.0]), RC1 + np.array([0.0,  h, 0.0]),
                      color=WHITE, stroke_width=4)

        # ── Row 2 panels and arrows ───────────────────────────────────────────
        # Frozen layers (pretrained) shown with teal border
        in2,   in2_lbl  = make_panel(LC2, "Input", next_to_direction=DOWN)
        nn2_1, nn2_1l  = make_panel(L12, "Layer 1", next_to_direction=DOWN, border=TEAL_C)
        nn2_2, nn2_2l  = make_panel(L22, "Layer 2", next_to_direction=DOWN, border=TEAL_C)
        nn2_3, nn2_3l  = make_panel(L32, "Layer 3", next_to_direction=DOWN, border=TEAL_C)
        out2,  out2_lbl = make_panel(RC2, "New Classifier", next_to_direction=DOWN)

        arr2_in_l1  = make_lr_arrow(in2,   nn2_1)
        arr2_l1_l2  = make_lr_arrow(nn2_1, nn2_2)
        arr2_l2_l3  = make_lr_arrow(nn2_2, nn2_3)
        arr2_l3_out = make_lr_arrow(nn2_3, out2)


        arr2_l1_l1 = make_ud_arrow(nn1_1, nn2_1)
        arr2_l2_l2 = make_ud_arrow(nn1_2, nn2_2)
        arr2_l3_l3 = make_ud_arrow(nn1_3, nn2_3)
        tl_l1_l1_lbl = make_label(arr2_l1_l1, "Transfer pretrained weights")
        tl_l2_l2_lbl = make_label(arr2_l2_l2, "Transfer pretrained weights")
        tl_l3_l3_lbl = make_label(arr2_l3_l3, "Transfer pretrained weights")

        # ── Row 2 dots ────────────────────────────────────────────────────────
        in2_d1 = dot_group(cx_1, cy_1, _CA, LC2)
        in2_d2 = dot_group(cx_2, cy_2, _CB, LC2)
        in2_d3 = dot_group(cx_3, cy_3, _CC, LC2)

        l1b_x1 = interp(cx_1, px_1, _A1);  l1b_y1 = interp(cy_1, py_1, _A1)
        l1b_x2 = interp(cx_2, px_2, _A1);  l1b_y2 = interp(cy_2, py_2, _A1)
        l1b_x3 = interp(cx_3, px_3, _A1);  l1b_y3 = interp(cy_3, py_3, _A1)
        l1_d1  = dot_group(cx_1, cy_1, _CA, L12)
        l1_d2  = dot_group(cx_2, cy_2, _CB, L12)
        l1_d3  = dot_group(cx_3, cy_3, _CC, L12)

        l2b_x1 = interp(cx_1, px_1, _A2);  l2b_y1 = interp(cy_1, py_1, _A2)
        l2b_x2 = interp(cx_2, px_2, _A2);  l2b_y2 = interp(cy_2, py_2, _A2)
        l2b_x3 = interp(cx_3, px_3, _A2);  l2b_y3 = interp(cy_3, py_3, _A2)
        l2_d1  = dot_group(l1b_x1, l1b_y1, _CA, L22)
        l2_d2  = dot_group(l1b_x2, l1b_y2, _CB, L22)
        l2_d3  = dot_group(l1b_x3, l1b_y3, _CC, L22)

        l3_d1   = dot_group(l2b_x1, l2b_y1, _CA, L32)
        l3_d2   = dot_group(l2b_x2, l2b_y2, _CB, L32)
        l3_d3   = dot_group(l2b_x3, l2b_y3, _CC, L32)
        out2_d1 = dot_group(px_1, py_1, _CA, RC2)
        out2_d2 = dot_group(px_2, py_2, _CB, RC2)
        out2_d3 = dot_group(px_3, py_3, _CC, RC2)

        # ── Row 2 separator lines (2 boundaries for 3 classes) ────────────────
        # Boundaries in polar x at midpoints of class gaps:
        #   bnd1: midpoint between r=0.52 and r=0.82 → r=0.67 → px = (0.67-1)*2 = -0.66
        #   bnd2: midpoint between r=1.18 and r=1.48 → r=1.33 → px = (1.33-1)*2 = +0.66
        # Convert to scene offset: bnd_px * _DSCL
        bnd1 = -0.66 * _DSCL
        bnd2 = +0.66 * _DSCL

        # Initial state: X pattern (two diagonals — clearly wrong)
        sep2a_i = Line(RC2 + np.array([-h, -h, 0.0]), RC2 + np.array([h,  h, 0.0]),
                       color=WHITE, stroke_width=4)
        sep2b_i = Line(RC2 + np.array([-h,  h, 0.0]), RC2 + np.array([h, -h, 0.0]),
                       color=WHITE, stroke_width=4)
        # Final state: two vertical lines at the class boundaries
        sep2a_f = Line(RC2 + np.array([bnd1, -h, 0.0]), RC2 + np.array([bnd1, h, 0.0]),
                       color=WHITE, stroke_width=4)
        sep2b_f = Line(RC2 + np.array([bnd2, -h, 0.0]), RC2 + np.array([bnd2, h, 0.0]),
                       color=WHITE, stroke_width=4)

        # ── Transition label ──────────────────────────────────────────────────
        tl = Text("Transfer Learning \n new task (3 classes)", font_size=22, color=YELLOW, should_center=True)
        tl.move_to(( LC1 + LC2 )/2)

        # ── Animation: Row 1 ─────────────────────────────────────────────────
        self.play(FadeIn(in1), FadeIn(in1_lbl), run_time=0.8)
        self.play(FadeIn(in1_da), FadeIn(in1_db), run_time=0.7)
        self.play(FadeIn(nn1_1), FadeIn(nn1_1l), FadeIn(l1_da), FadeIn(l1_db), run_time=0.7)
        self.play(GrowArrow(arr1_in_l1), 
                  *[d.animate.move_to(d2s(x, y, L11)) for d, x, y in zip(l1_da, l1a_xa, l1a_ya)],
                  *[d.animate.move_to(d2s(x, y, L11)) for d, x, y in zip(l1_db, l1a_xb, l1a_yb)],
                  run_time=1.5, rate_func=smooth)
        self.play(FadeIn(nn1_2), FadeIn(nn1_2l), FadeIn(l2_da), FadeIn(l2_db), run_time=0.7)
        self.play(GrowArrow(arr1_l1_l2), 
                  *[d.animate.move_to(d2s(x, y, L21)) for d, x, y in zip(l2_da, l2a_xa, l2a_ya)],
                  *[d.animate.move_to(d2s(x, y, L21)) for d, x, y in zip(l2_db, l2a_xb, l2a_yb)],
                  run_time=1.5, rate_func=smooth)
        self.play(FadeIn(nn1_3), FadeIn(nn1_3l), FadeIn(l3_da), FadeIn(l3_db), run_time=0.7)
        self.play(GrowArrow(arr1_l2_l3), 
                  *[d.animate.move_to(d2s(x, y, L31)) for d, x, y in zip(l3_da, px_a, py_a)],
                  *[d.animate.move_to(d2s(x, y, L31)) for d, x, y in zip(l3_db, px_b, py_b)],
                  run_time=1.5, rate_func=smooth)
        self.play(FadeIn(out1), FadeIn(out1_lbl),
                  FadeIn(out1_da), FadeIn(out1_db), FadeIn(sep1_i), run_time=0.8)
        self.play(GrowArrow(arr1_l3_out), Transform(sep1_i, sep1_f),
                  run_time=1.5, rate_func=smooth)
        self.wait(0.5)

        # ── Animation: Row 2 ─────────────────────────────────────────────────
        #self.play(FadeIn(tl), run_time=0.6)
        self.play(FadeIn(tl), FadeIn(in2), FadeIn(in2_lbl), run_time=1)
        
        self.play(FadeIn(in2_d1), FadeIn(in2_d2), FadeIn(in2_d3), run_time=0.6)
        # Frozen layers appear quickly — they're reused from row 1
        self.play(FadeIn(nn2_1), FadeIn(nn2_1l), FadeIn(l1_d1), FadeIn(l1_d2), FadeIn(l1_d3),
                  run_time=0.5)
        self.play(AnimationGroup(GrowArrow(arr2_in_l1), GrowArrow(arr2_l1_l1), FadeIn(tl_l1_l1_lbl, run_time=0.6),
                                 *[d.animate.move_to(d2s(x, y, L12)) for d, x, y in zip(l1_d1, l1b_x1, l1b_y1)],
                                 *[d.animate.move_to(d2s(x, y, L12)) for d, x, y in zip(l1_d2, l1b_x2, l1b_y2)],
                                *[d.animate.move_to(d2s(x, y, L12)) for d, x, y in zip(l1_d3, l1b_x3, l1b_y3)],
                                run_time=1.6, rate_func=smooth))
        self.play(FadeOut(tl_l1_l1_lbl, run_time=0.6))
        self.play(FadeIn(nn2_2), FadeIn(nn2_2l), FadeIn(l2_d1), FadeIn(l2_d2), FadeIn(l2_d3),
                  run_time=0.5)
        self.play(AnimationGroup(GrowArrow(arr2_l1_l2), GrowArrow(arr2_l2_l2), FadeIn(tl_l2_l2_lbl, run_time=0.6),
                  *[d.animate.move_to(d2s(x, y, L22)) for d, x, y in zip(l2_d1, l2b_x1, l2b_y1)],
                  *[d.animate.move_to(d2s(x, y, L22)) for d, x, y in zip(l2_d2, l2b_x2, l2b_y2)],
                  *[d.animate.move_to(d2s(x, y, L22)) for d, x, y in zip(l2_d3, l2b_x3, l2b_y3)],
                  run_time=1.6, rate_func=smooth))
        self.play(FadeOut(tl_l2_l2_lbl, run_time=0.6))
        self.play(FadeIn(nn2_3), FadeIn(nn2_3l), FadeIn(l3_d1), FadeIn(l3_d2), FadeIn(l3_d3),
                  run_time=0.5)
        self.play(AnimationGroup(GrowArrow(arr2_l2_l3), GrowArrow(arr2_l3_l3), FadeIn(tl_l3_l3_lbl, run_time=0.6),
                  *[d.animate.move_to(d2s(x, y, L32)) for d, x, y in zip(l3_d1, px_1, py_1)],
                  *[d.animate.move_to(d2s(x, y, L32)) for d, x, y in zip(l3_d2, px_2, py_2)],
                  *[d.animate.move_to(d2s(x, y, L32)) for d, x, y in zip(l3_d3, px_3, py_3)],
                  run_time=1.6, rate_func=smooth))
        self.play(FadeOut(tl_l3_l3_lbl, run_time=0.6))
        self.play(FadeIn(out2), FadeIn(out2_lbl),
                  FadeIn(out2_d1), FadeIn(out2_d2), FadeIn(out2_d3),
                  FadeIn(sep2a_i), FadeIn(sep2b_i), run_time=0.8)
        self.play(GrowArrow(arr2_l3_out),
                  Transform(sep2a_i, sep2a_f), Transform(sep2b_i, sep2b_f),
                  run_time=1.5, rate_func=smooth)
        self.wait(4)


_MIME_TYPES = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".gif":  "image/gif",
}


def neural_network_transfer_learning(width: int = PIXEL_WIDTH) -> None:
    """Embed the pre-rendered TransferLearningScene inline (Jupyter / Quarto).

    The scene must have been rendered first via ``make media``.
    Accepts any format produced by Manim (mp4, webm, mov, gif); the most
    recently modified file is used automatically.
    """
    from pathlib import Path
    from IPython.display import HTML, display as _display

    media_root = Path(__file__).parent.parent / "media"
    matches = [
        p for p in media_root.rglob("TransferLearningScene.*")
        if p.suffix in _MIME_TYPES
    ]
    if not matches:
        raise FileNotFoundError(
            "No TransferLearningScene render found under media/. "
            "Run 'make media' before rendering the slides."
        )
    media_path = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        rel = media_path.relative_to(Path.cwd())
    except ValueError:
        rel = media_path

    mime = _MIME_TYPES[media_path.suffix]
    if media_path.suffix == ".gif":
        html = f'<img src="{rel}" width="{width}" style="display:block;margin:auto">'
    else:
        html = (
            f'<video width="{width}" autoplay loop muted playsinline controls '
            f'style="display:block;margin:auto">'
            f'<source src="{rel}" type="{mime}">'
            f"</video>"
        )
    _display(HTML(html))


if __name__ == "__main__":
    neural_network_transfer_learning()
