"""
Triplet contrastive — Manim animation.

Shows how 2-D CNN embeddings self-organise angularly under pairwise
cosine contrastive loss (no softmax / no weight vectors).

Render (after caching):
    python -m manim -ql viz/triplet_viz.py TripletScene

Cache training data once:
    pixi run -e local-amd python viz/triplet_train.py
"""
from __future__ import annotations

import numpy as np

import manim
from manim import (
    BLACK, WHITE, GREY_A, GREY_B,
    UP, DOWN, LEFT, RIGHT,
    Scene, VGroup,
    Arrow, Dot, Text,
    FadeIn, GrowArrow, Transform, UpdateFromAlphaFunc,
    smooth, ManimColor,
)

try:
    from .softmax_train import CLASS_COLORS, N_PER_CLASS, ORBIT_RADII, norm_rows
    from .triplet_train  import TEMPERATURE, load_cache, compute_centroids
except ImportError:
    from softmax_train import CLASS_COLORS, N_PER_CLASS, ORBIT_RADII, norm_rows
    from triplet_train  import TEMPERATURE, load_cache, compute_centroids

# ── Render config ─────────────────────────────────────────────────────────────

PIXEL_WIDTH  = 1280
PIXEL_HEIGHT = 1280
FRAME_WIDTH  = 10

manim.config.pixel_width  = PIXEL_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.frame_width  = FRAME_WIDTH



Text.set_default(font="Ubuntu")

# ── Manim scene config ────────────────────────────────────────────────────────

RADIUS = 3.0
SHIFT  = LEFT * 0

MANIM_ORBIT_RADII = ORBIT_RADII * RADIUS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mp(xy: np.ndarray, r: float = RADIUS) -> np.ndarray:
    return np.array([xy[0] * r, xy[1] * r, 0.0]) + SHIFT


# ── Manim scene ───────────────────────────────────────────────────────────────

class TripletScene(Scene):
    def construct(self) -> None:
        snapshots = load_cache()
        epochs    = [s[0] for s in snapshots]
        n_snaps   = len(snapshots)
        N         = N_PER_CLASS * 10

        labels   = snapshots[0][2]           # fixed across all snapshots
        sample_r = MANIM_ORBIT_RADII[labels] # per-sample Manim orbit radius

        # Pre-compute dot positions and per-class centroids per snapshot
        all_pts:  list[np.ndarray] = []   # each: (N, 3)
        all_cen:  list[np.ndarray] = []   # each: (10, 2) normalised centroid dirs

        for ep, embs, lbls in snapshots:
            pts = np.column_stack(
                [embs[:, 0] * sample_r, embs[:, 1] * sample_r, np.zeros(N)]
            ) + SHIFT                          # embs already unit-norm
            all_pts.append(pts)
            all_cen.append(compute_centroids(embs, lbls))

        # ── Background ────────────────────────────────────────────────────────
        self.camera.background_color = BLACK

        # ── Orbit rings ───────────────────────────────────────────────────────
        from manim import Circle
        orbit_rings = VGroup(*[
            Circle(radius=float(MANIM_ORBIT_RADII[c]),
                   stroke_width=1.0, stroke_opacity=0.25,
                   color=ManimColor(CLASS_COLORS[c])).shift(SHIFT)
            for c in range(10)
        ])
        self.play(FadeIn(orbit_rings), run_time=0.5)

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text(
            f"Supervised contrastive  (temperature = {TEMPERATURE})",
            font_size=24, color=WHITE,
        ).to_edge(UP, buff=0.)
        self.play(FadeIn(title), run_time=0.5)

        # ── Epoch label ───────────────────────────────────────────────────────
        def _epoch_text(ep: int) -> Text:
            return Text(f"Epoch {ep:2d}", font_size=30, color=GREY_A).to_corner(
                DOWN + RIGHT, buff=0.5
            )

        epoch_lbl = _epoch_text(epochs[0])
        self.add(epoch_lbl)

        # ── Legend ────────────────────────────────────────────────────────────
        legend_items = VGroup()
        for c in range(10):
            dot = Dot(radius=0.09, color=ManimColor(CLASS_COLORS[c]))
            lbl = Text(str(c), font_size=20, color=ManimColor(CLASS_COLORS[c]))
            lbl.next_to(dot, RIGHT, buff=0.15)
            legend_items.add(VGroup(dot, lbl))
        legend_items.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        legend_items.to_edge(RIGHT, buff=0.45).shift(DOWN * 0.3)
        self.play(FadeIn(legend_items), run_time=0.5)

        # ── Centroid arrows ───────────────────────────────────────────────────
        def _make_centroid_arrow(cen_dir: np.ndarray, c: int) -> Arrow:
            tip = _mp(cen_dir, r=RADIUS * 1.06)
            return Arrow(
                SHIFT, tip, buff=0,
                color=ManimColor(CLASS_COLORS[c]),
                stroke_width=3.0,
                max_tip_length_to_length_ratio=0.10,
            )

        # centroid_arrows = [_make_centroid_arrow(all_cen[0][c], c) for c in range(10)]
        # self.play(*(GrowArrow(a) for a in centroid_arrows), run_time=1.0)

        # ── Digit labels near centroid tips ───────────────────────────────────
        def _tip_pos(cen_dir: np.ndarray) -> np.ndarray:
            return _mp(cen_dir, r=RADIUS * 1.20)

        digit_labels = [
            Text(str(c), font_size=22, color=ManimColor(CLASS_COLORS[c])).move_to(
                _tip_pos(all_cen[0][c])
            )
            for c in range(10)
        ]
        self.play(*(FadeIn(lbl) for lbl in digit_labels), run_time=0.5)

        # ── Dots (epoch 0) ────────────────────────────────────────────────────
        dots = [
            Dot(all_pts[0][i], radius=0.065,
                color=ManimColor(CLASS_COLORS[labels[i]]), fill_opacity=0.75)
            for i in range(N)
        ]
        self.play(FadeIn(VGroup(*dots)), run_time=0.8)
        self.wait(1.8)

        # Track angles for arc-sweeping dots, centroid arrows, and digit labels
        cur_dot_angles = np.arctan2(all_pts[0][:, 1] - SHIFT[1],
                                    all_pts[0][:, 0] - SHIFT[0])
        cur_cen_angles = np.arctan2(all_cen[0][:, 1], all_cen[0][:, 0])
        dots_group     = VGroup(*dots)

        def _arc_dots(sa, da, radii):
            def _fn(mob, alpha):
                a  = sa + alpha * da
                xs = np.cos(a) * radii + SHIFT[0]
                ys = np.sin(a) * radii + SHIFT[1]
                for i, d in enumerate(mob):
                    d.move_to(np.array([xs[i], ys[i], 0.0]))
            return _fn

        def _arc_arrow(sa, da, c):
            tip_r = RADIUS * 1.06
            color = ManimColor(CLASS_COLORS[c])
            def _fn(mob, alpha):
                angle = sa + alpha * da
                tip   = np.array([np.cos(angle) * tip_r, np.sin(angle) * tip_r, 0.0]) + SHIFT
                mob.become(Arrow(SHIFT, tip, buff=0, color=color,
                                 stroke_width=3.0, max_tip_length_to_length_ratio=0.10))
            return _fn

        def _arc_label(sa, da):
            label_r = RADIUS * 1.20
            def _fn(mob, alpha):
                angle = sa + alpha * da
                mob.move_to(np.array([np.cos(angle) * label_r,
                                      np.sin(angle) * label_r, 0.0]) + SHIFT)
            return _fn

        # ── Animate through training snapshots ────────────────────────────────
        for snap_i in range(1, n_snaps):
            ep      = epochs[snap_i]
            pts_new = all_pts[snap_i]
            cen_new = all_cen[snap_i]

            end_dot_angles = np.arctan2(pts_new[:, 1] - SHIFT[1], pts_new[:, 0] - SHIFT[0])
            delta_dot      = (end_dot_angles - cur_dot_angles + np.pi) % (2*np.pi) - np.pi
            end_cen_angles = np.arctan2(cen_new[:, 1], cen_new[:, 0])
            delta_cen      = (end_cen_angles - cur_cen_angles + np.pi) % (2*np.pi) - np.pi

            new_epoch_lbl = _epoch_text(ep)

            self.play(
                UpdateFromAlphaFunc(dots_group,
                    _arc_dots(cur_dot_angles.copy(), delta_dot, sample_r),
                    rate_func=smooth),
                # *(UpdateFromAlphaFunc(centroid_arrows[c],
                #     _arc_arrow(cur_cen_angles[c], delta_cen[c], c),
                #     rate_func=smooth)
                #   for c in range(10)),
                *(UpdateFromAlphaFunc(digit_labels[c],
                    _arc_label(cur_cen_angles[c], delta_cen[c]),
                    rate_func=smooth)
                  for c in range(10)),
                Transform(epoch_lbl, new_epoch_lbl),
                run_time=2.2,
            )
            cur_dot_angles = end_dot_angles
            cur_cen_angles = end_cen_angles
            self.wait(2.0)

        # ── Final annotation ──────────────────────────────────────────────────
        note = Text(
            "Classes self-organise angularly — no labels used at inference time",
            font_size=20, color=GREY_B,
        ).to_edge(DOWN, buff=0.0)
        self.play(FadeIn(note), run_time=0.8)
        self.wait(4.0)


# ── Display helper ────────────────────────────────────────────────────────────

_MIME = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".gif":  "image/gif",
}


def show_triplet(width: int = 800) -> None:
    """Embed the pre-rendered TripletScene inline (Jupyter / Quarto)."""
    from pathlib import Path
    from IPython.display import HTML, display as _display

    media_root = Path(__file__).parent.parent / "media"
    matches = [p for p in media_root.rglob("TripletScene.*") if p.suffix in _MIME]
    if not matches:
        raise FileNotFoundError(
            "No TripletScene render found under media/. "
            "Run 'make triplet' (or cache + render) first."
        )
    path = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        rel = path.relative_to(Path.cwd())
    except ValueError:
        rel = path

    mime = _MIME[path.suffix]
    if path.suffix == ".gif":
        html = f'<img src="{rel}" width="{width}" style="display:block;margin:auto">'
    else:
        html = (
            f'<video width="{width}" autoplay loop muted playsinline controls '
            f'style="display:block;margin:auto">'
            f'<source src="{rel}" type="{mime}">'
            f"</video>"
        )
    _display(HTML(html))
