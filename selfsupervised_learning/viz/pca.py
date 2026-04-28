"""pca.py — animated PCA visualization for the self-supervised learning lecture.

Animation beats:
  1. 2D scatter with visible covariance
  2. PC arrows grow in, scaled by captured variance; "Principal Components" label
  3. Data rotates smoothly to align with the PCs (axis-aligned)
  4a. Shadow dots collapse onto the PC1 axis; PC1 1-D strip fades in
  4b. Shadow dots collapse onto the PC2 axis; PC2 1-D strip fades in

Scene class : PCAScene
Embed helper: pca_visualization()
"""
from pathlib import Path

import numpy as np
import manim
from manim import (
    Scene, VGroup, AnimationGroup, Succession, Wait,
    Dot, Arrow, Line, Text, Tex,
    FadeIn, FadeOut, GrowArrow, Write,
    ReplacementTransform, Transform,
    UP, RIGHT,
    WHITE, GREY_D, YELLOW, smooth,
)

# ── Render config ─────────────────────────────────────────────────────────────
PIXEL_WIDTH  = 800
PIXEL_HEIGHT = 600
FRAME_SIZE  = 11.0

manim.config.pixel_width  = PIXEL_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.frame_width  = FRAME_SIZE

Text.set_default(font="Ubuntu")

# ── Palette ───────────────────────────────────────────────────────────────────
_PC1  = "#ff6060"   # PC1 (dominant)  — red
_PC2  = "#4daaff"   # PC2             — blue
_DOT  = "#9999cc"   # data dots       — soft indigo
_AX   = "#444466"   # reference axes  — dim

# ── Geometry ──────────────────────────────────────────────────────────────────
_CX, _CY      = 0.0, 0.0   # centre of main scatter panel (scene units)
_SCL          = 0.82        # data unit → scene unit
_ARROW_SCALE  = 2.5         # multiply std to get arrow length (increase to lengthen arrows)
_STRIP1_Y     = -2.0        # y of PC1 projection strip axis
_STRIP2_Y     = -3.5        # y of PC2 projection strip axis
_STRIP_HX     = 4.6         # half-width of strip axis lines
_DR           = 0.055       # dot radius


def sp(x, y):
    """Data → scene coords, relative to main scatter centre."""
    return np.array([_CX + x * _SCL, _CY + y * _SCL, 0.0])


class PCAScene(Scene):
    def construct(self):
        # ── Data ──────────────────────────────────────────────────────────────
        rng = np.random.default_rng(seed=42)
        n   = 120
        cov = np.array([[1.0, 0.8], [0.8, 1]])   # clear off-diagonal covariance
        X   = rng.multivariate_normal([0.0, 0.0], cov, n)

        # PCA via eigendecomposition of sample covariance
        vals, vecs = np.linalg.eigh(np.cov(X.T))
        order = np.argsort(vals)[::-1]             # descending variance
        vals, vecs = vals[order], vecs[:, order]
        std1, std2  = float(np.sqrt(vals[0])), float(np.sqrt(vals[1]))

        # Canonical sign convention for minimal rotation angle:
        #   1. PC1 first component positive → rotation angle in (−90°, +90°)
        #   2. det = +1 → proper rotation, no reflection
        if vecs[0, 0] < 0:
            vecs[:, 0] *= -1
        if np.linalg.det(vecs) < 0:
            vecs[:, 1] *= -1

        pc1, pc2 = vecs[:, 0], vecs[:, 1]
        X_rot    = X @ vecs                        # data in PC coordinates

        # ── Grid helper ───────────────────────────────────────────────────────
        def _build_grid(tfn):
            """Return a VGroup of grid lines; each point mapped through tfn(x,y)."""
            lines = []
            for xi in range(-4, 5):
                w, op = (1.8, 0.75) if xi == 0 else (0.8, 0.38)
                lines.append(Line(tfn(xi, -4.), tfn(xi, 4.),
                                  color=_AX, stroke_width=w, stroke_opacity=op))
            for yi in range(-4, 5):
                w, op = (1.8, 0.75) if yi == 0 else (0.8, 0.38)
                lines.append(Line(tfn(-5., yi), tfn(5., yi),
                                  color=_AX, stroke_width=w, stroke_opacity=op))
            return VGroup(*lines)

        # Original (Cartesian) grid
        orig_grid = _build_grid(sp)

        # Rotated grid: same grid lines but endpoints projected through vecs
        def sp_rot(x, y):
            p = np.array([x, y], dtype=float) @ vecs
            return sp(p[0], p[1])

        rotated_grid = _build_grid(sp_rot)

        # PC-aligned grid (axis-aligned, fades in after rotation)
        pc_grid = _build_grid(sp)

        # Keep the axis-endpoint array for label position math below
        carteesian_axes_arr    = np.array([[[-5., 0.], [5., 0.]],
                                           [[ 0., -4.], [0., 4.]]])
        tgt_carteesian_axes_arr = carteesian_axes_arr @ vecs
        ax_lbl_x = Tex("$x_1$", font_size=24, color=GREY_D).move_to(sp(4.7, 0.35))
        ax_lbl_y = Tex("$x_2$", font_size=24, color=GREY_D).move_to(sp(0.35, 3.8))

        # ── Data dots (original coords) ───────────────────────────────────────
        dots = VGroup(*[
            Dot(sp(x, y), radius=_DR, color=_DOT, fill_opacity=0.75)
            for x, y in X
        ])

        # ── Phase 1: show scatter ─────────────────────────────────────────────
        title = Tex("2D data with covariance", font_size=38, color=WHITE
                     ).to_edge(UP, buff=0.30)
        self.play(FadeIn(orig_grid), FadeIn(ax_lbl_x), FadeIn(ax_lbl_y),
                  Write(title), run_time=0.8)
        self.play(FadeIn(dots), run_time=1.0)
        self.wait(1.2)

        # ── Phase 2: principal-component arrows ───────────────────────────────
        origin_s = sp(0, 0)
        a1 = _ARROW_SCALE * std1   # arrow lengths in data units
        a2 = _ARROW_SCALE * std2
        arr1 = Arrow(origin_s, sp(a1 * pc1[0], a1 * pc1[1]),
                     color=_PC1, stroke_width=5, buff=0,
                     tip_length=0.22,
                     max_stroke_width_to_length_ratio=100)
        arr2 = Arrow(origin_s, sp(a2 * pc2[0], a2 * pc2[1]),
                     color=_PC2, stroke_width=5, buff=0,
                     tip_length=0.22,
                     max_stroke_width_to_length_ratio=100)

        lbl_pc1 = Tex("PC1", font_size=28, color=_PC1
                       ).next_to(arr1.get_tip(), RIGHT, buff=0.15)
        lbl_pc2 = Tex("PC2", font_size=28, color=_PC2
                       ).next_to(arr2.get_tip(), UP,    buff=0.10)

        new_t1 = Tex("Principal components in data space", font_size=38, color=WHITE
                      ).to_edge(UP, buff=0.30)
        pc_banner = Tex("Principal Components", font_size=34, color=YELLOW
                         ).move_to(np.array([0.0, -4.5, 0.0]))

        self.play(
            GrowArrow(arr1), GrowArrow(arr2),
            FadeIn(lbl_pc1), FadeIn(lbl_pc2),
            ReplacementTransform(title, new_t1),
            run_time=1.0,
        )
        self.play(Write(pc_banner), run_time=0.5)
        self.wait(1.5)

        # ── Phase 3: rotate data to PC space ─────────────────────────────────
        new_t2 = Tex("Aligned to principal components", font_size=38, color=WHITE
                      ).to_edge(UP, buff=0.30)

        # Target arrows: PC1 → x-axis, PC2 → y-axis (same scaled lengths)
        tgt_arr1 = Arrow(origin_s, sp(a1, 0),
                         color=_PC1, stroke_width=5, buff=0,
                         tip_length=0.22,
                         max_stroke_width_to_length_ratio=100)
        tgt_arr2 = Arrow(origin_s, sp(0, a2),
                         color=_PC2, stroke_width=5, buff=0,
                         tip_length=0.22,
                         max_stroke_width_to_length_ratio=100)
        tgt_lbl1 = Tex("PC1", font_size=28, color=_PC1
                        ).move_to(sp(a1 + 0.55, 0.0))
        tgt_lbl2 = Tex("PC2", font_size=28, color=_PC2
                        ).move_to(sp(0.0, a2 + 0.45))

        # Cartesian axis labels follow the axes as they rotate, keeping $x_1$/$x_2$ text
        x1_rot_end = tgt_carteesian_axes_arr[0, 1]   # rotated positive tip of x₁ axis
        x2_rot_end = tgt_carteesian_axes_arr[1, 1]   # rotated positive tip of x₂ axis
        tgt_ax_lbl_x_rot = Tex(r"$x_1$", font_size=24, color=GREY_D
                                ).move_to(sp(x1_rot_end[0] * 0.88, x1_rot_end[1] * 0.88))
        tgt_ax_lbl_y_rot = Tex(r"$x_2$", font_size=24, color=GREY_D
                                ).move_to(sp(x2_rot_end[0] * 0.88, x2_rot_end[1] * 0.88))

        # PC axis labels appear at the end (horizontal/vertical final state)
        pc_ax_lbl_x = Tex(r"$\mathrm{PC}_1$", font_size=24, color=GREY_D
                           ).move_to(sp(4.7, 0.35))
        pc_ax_lbl_y = Tex(r"$\mathrm{PC}_2$", font_size=24, color=GREY_D
                           ).move_to(sp(0.35, 3.8))

        # Timing constants (tune to taste)
        _ROT_T     = 2.0   # full rotation duration
        _CART_ROT  = 0.9   # cartesian axes rotate for this long before fading
        _CART_FADE = 0.5   # cartesian fade-out duration
        _PC_DELAY  = 1.4   # delay before PC coordinate system appears
        _PC_FADE   = 0.5   # PC axes/labels fade-in duration

        self.play(
            # ── Dots + arrows + original grid: all rotate together ─────────────
            AnimationGroup(
                *[dot.animate.move_to(sp(xr, yr)) for dot, (xr, yr) in zip(dots, X_rot)],
                Transform(arr1,    tgt_arr1),
                Transform(arr2,    tgt_arr2),
                Transform(lbl_pc1, tgt_lbl1),
                Transform(lbl_pc2, tgt_lbl2),
                ReplacementTransform(new_t1, new_t2),
                FadeOut(pc_banner),
                Transform(orig_grid, rotated_grid),
                Transform(ax_lbl_x, tgt_ax_lbl_x_rot),
                Transform(ax_lbl_y, tgt_ax_lbl_y_rot),
                run_time=_ROT_T,
                rate_func=smooth,
            ),
            # ── Old grid fades out; PC-aligned grid fades in ──────────────────
            Succession(
                Wait(_PC_DELAY),
                AnimationGroup(
                    FadeOut(orig_grid),
                    FadeOut(ax_lbl_x),
                    FadeOut(ax_lbl_y),
                    run_time=_CART_FADE,
                ),
                AnimationGroup(
                    FadeIn(pc_grid),
                    FadeIn(pc_ax_lbl_x),
                    FadeIn(pc_ax_lbl_y),
                    run_time=_PC_FADE,
                ),
            ),
        )
        self.wait(1.0)

        # ── Phase 4a: project onto PC1 (collapse to x-axis) ──────────────────
        strip1_ax = Line(
            np.array([-_STRIP_HX, _STRIP1_Y, 0.0]),
            np.array([ _STRIP_HX, _STRIP1_Y, 0.0]),
            color=_PC1, stroke_width=2.5,
        )
        strip1_lbl = Tex("PC1 projection", font_size=26, color=_PC1
                          ).move_to(np.array([0.0, _STRIP1_Y + 0.45, 0.0]))

        self.play(FadeIn(strip1_ax), FadeIn(strip1_lbl), run_time=0.6)

        # Shadow dots: copy of main dots that collapse to the PC1 axis (y → 0)
        shadows1 = VGroup(*[
            Dot(sp(xr, yr), radius=_DR, color=_PC1, fill_opacity=0.55)
            for xr, yr in X_rot
        ])
        # Projection dots in the strip (x-position = PC1 value)
        proj1_dots = VGroup(*[
            Dot(np.array([xr * _SCL, _STRIP1_Y, 0.0]),
                radius=_DR, color=_PC1, fill_opacity=0.80)
            for xr, _ in X_rot
        ])

        self.play(FadeIn(shadows1), run_time=0.35)
        self.play(
            *[d.animate.move_to(sp(xr, 0.0)) for d, (xr, _) in zip(shadows1, X_rot)],
            FadeIn(proj1_dots),
            run_time=1.5, rate_func=smooth,
        )
        self.wait(0.6)
        self.play(FadeOut(shadows1), run_time=0.5)
        self.wait(0.9)

        # ── Phase 4b: project onto PC2 (collapse to y-axis) ──────────────────
        strip2_ax = Line(
            np.array([-_STRIP_HX, _STRIP2_Y, 0.0]),
            np.array([ _STRIP_HX, _STRIP2_Y, 0.0]),
            color=_PC2, stroke_width=2.5,
        )
        strip2_lbl = Tex("PC2 projection", font_size=26, color=_PC2
                          ).move_to(np.array([0.0, _STRIP2_Y + 0.45, 0.0]))

        self.play(FadeIn(strip2_ax), FadeIn(strip2_lbl), run_time=0.6)

        shadows2 = VGroup(*[
            Dot(sp(xr, yr), radius=_DR, color=_PC2, fill_opacity=0.55)
            for xr, yr in X_rot
        ])
        proj2_dots = VGroup(*[
            Dot(np.array([yr * _SCL, _STRIP2_Y, 0.0]),
                radius=_DR, color=_PC2, fill_opacity=0.80)
            for _, yr in X_rot
        ])

        self.play(FadeIn(shadows2), run_time=0.35)
        self.play(
            *[d.animate.move_to(sp(0.0, yr)) for d, (_, yr) in zip(shadows2, X_rot)],
            FadeIn(proj2_dots),
            run_time=1.5, rate_func=smooth,
        )
        self.wait(0.6)
        self.play(FadeOut(shadows2), run_time=0.5)
        self.wait(2.5)


# ── Embed helper (Jupyter / Quarto) ──────────────────────────────────────────
_MIME_TYPES = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".gif":  "image/gif",
}


def pca_visualization(width: int = PIXEL_WIDTH) -> None:
    """Embed the pre-rendered PCAScene inline.

    The scene must have been rendered first via ``make pca``.
    """
    from IPython.display import HTML, display as _display

    media_root = Path(__file__).parent.parent / "media"
    matches = [
        p for p in media_root.rglob("PCAScene.*")
        if p.suffix in _MIME_TYPES
    ]
    if not matches:
        raise FileNotFoundError(
            "No PCAScene render found under media/. "
            "Run 'make pca' before rendering the slides."
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
