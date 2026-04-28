"""
Softmax weight anchor — Manim animation.

Shows how 2-D CNN embeddings cluster angularly around their softmax weight
vectors as the network trains on MNIST.

Render (after caching):
    python -m manim -ql viz/softmax_viz.py SoftmaxScene

Cache training data once:
    pixi run -e local-amd python viz/softmax_train.py
"""
from __future__ import annotations

import numpy as np

import manim
from manim import (
    BLACK, WHITE, GREY_A, GREY_B,
    UP, DOWN, LEFT, RIGHT,
    Scene, VGroup,
    Circle, Arrow, Dot, Text,
    FadeIn, GrowArrow, Transform, UpdateFromAlphaFunc,
    smooth, ManimColor,
)

try:
    from .softmax_train import CLASS_COLORS, N_PER_CLASS, ORBIT_RADII, load_cache, norm_rows
except ImportError:  # Manim loads the file as a top-level script
    from softmax_train import CLASS_COLORS, N_PER_CLASS, ORBIT_RADII, load_cache, norm_rows

# ── Render config ─────────────────────────────────────────────────────────────

PIXEL_WIDTH  = 1280
PIXEL_HEIGHT = 1280
FRAME_WIDTH  = 10.

manim.config.pixel_width  = PIXEL_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.frame_width  = FRAME_WIDTH

Text.set_default(font="Ubuntu")

# ── Manim scene config ────────────────────────────────────────────────────────

RADIUS = 3.0            # reference radius in scene units
SHIFT  = [0,0,0]      # shift circle left to leave room for legend

# Manim orbit radii mirror the matplotlib ORBIT_RADII (0.8–1.2), scaled up
MANIM_ORBIT_RADII = ORBIT_RADII * RADIUS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mp(xy: np.ndarray, r: float = RADIUS) -> np.ndarray:
    """2-D unit direction → 3-D Manim point, scaled and shifted."""
    return np.array([xy[0] * r, xy[1] * r, 0.0]) + SHIFT


# ── Manim scene ───────────────────────────────────────────────────────────────

class SoftmaxScene(Scene):
    def construct(self) -> None:
        snapshots = load_cache()
        epochs    = [s[0] for s in snapshots]
        n_snaps   = len(snapshots)
        N         = N_PER_CLASS * 10

        # Labels are identical across all snapshots (fixed vis subset)
        labels = snapshots[0][2]

        # Per-sample Manim orbit radius (class 0 innermost, class 9 outermost)
        sample_r = MANIM_ORBIT_RADII[labels]

        # Pre-compute normalized dot positions and weight directions per snapshot
        all_pts: list[np.ndarray] = []   # each: (N, 3)
        all_w:   list[np.ndarray] = []   # each: (10, 2) normalized

        for ep, embs, lbls, weights in snapshots:
            ne = norm_rows(embs)     # (N, 2) — unit direction of each embedding
            nw = norm_rows(weights)  # (10, 2) — unit direction of each weight vector
            pts = np.column_stack([ne[:, 0] * sample_r, ne[:, 1] * sample_r, np.zeros(N)])
            pts += SHIFT
            all_pts.append(pts)
            all_w.append(nw)

        # ── Background ────────────────────────────────────────────────────────
        self.camera.background_color = BLACK

        # ── Orbit rings (one per class, faint) ───────────────────────────────
        orbit_rings = VGroup(*[
            Circle(radius=float(MANIM_ORBIT_RADII[c]), stroke_width=1.0, stroke_opacity=0.25,
                   color=ManimColor(CLASS_COLORS[c])).shift(SHIFT)
            for c in range(10)
        ])
        self.play(FadeIn(orbit_rings), run_time=0.5)

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text(
            "Softmax weights as angular anchors", font_size=26, color=WHITE
        ).to_edge(UP, buff=0.)
        self.play(FadeIn(title), run_time=0.5)

        # ── Epoch label ───────────────────────────────────────────────────────
        def _epoch_text(ep: int) -> Text:
            return Text(f"Epoch {ep:2d}", font_size=30, color=GREY_A).to_corner(
                DOWN + RIGHT, buff=0.5
            )

        epoch_lbl = _epoch_text(epochs[0])
        self.add(epoch_lbl)

        # ── Legend (right side) ───────────────────────────────────────────────
        legend_items = VGroup()
        for c in range(10):
            dot  = Dot(radius=0.09, color=ManimColor(CLASS_COLORS[c]))
            lbl  = Text(str(c), font_size=20, color=ManimColor(CLASS_COLORS[c]))
            lbl.next_to(dot, RIGHT, buff=0.15)
            legend_items.add(VGroup(dot, lbl))
        legend_items.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        legend_items.to_edge(RIGHT, buff=0.45).shift(DOWN * 0.3)
        self.play(FadeIn(legend_items), run_time=0.5)

        # ── Weight vector arrows ──────────────────────────────────────────────
        def _make_arrow(w_dir: np.ndarray, c: int) -> Arrow:
            tip = _mp(w_dir)
            return Arrow(
                SHIFT, tip, buff=0,
                color=ManimColor(CLASS_COLORS[c]),
                stroke_width=3.5,
                max_tip_length_to_length_ratio=0.10,
            )

        weight_arrows = [_make_arrow(all_w[0][c], c) for c in range(10)]
        self.play(*(GrowArrow(a) for a in weight_arrows), run_time=1.2)

        # ── Digit labels near arrow tips ──────────────────────────────────────
        def _tip_pos(w_dir: np.ndarray) -> np.ndarray:
            return _mp(w_dir, r=RADIUS * 1.14)

        digit_labels = [
            Text(str(c), font_size=22, color=ManimColor(CLASS_COLORS[c])).move_to(
                _tip_pos(all_w[0][c])
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

        # Track angles for arc-sweeping dots, arrows, and digit labels
        cur_dot_angles   = np.arctan2(all_pts[0][:, 1] - SHIFT[1],
                                      all_pts[0][:, 0] - SHIFT[0])
        cur_arrow_angles = np.arctan2(all_w[0][:, 1], all_w[0][:, 0])
        dots_group       = VGroup(*dots)

        def _arc_dots(sa, da, radii):
            def _fn(mob, alpha):
                a = sa + alpha * da
                xs = np.cos(a) * radii + SHIFT[0]
                ys = np.sin(a) * radii + SHIFT[1]
                for i, d in enumerate(mob):
                    d.move_to(np.array([xs[i], ys[i], 0.0]))
            return _fn

        def _arc_arrow(sa, da, c):
            tip_r = RADIUS
            color = ManimColor(CLASS_COLORS[c])
            def _fn(mob, alpha):
                angle = sa + alpha * da
                tip   = np.array([np.cos(angle) * tip_r, np.sin(angle) * tip_r, 0.0]) + SHIFT
                mob.become(Arrow(SHIFT, tip, buff=0, color=color,
                                 stroke_width=3.5, max_tip_length_to_length_ratio=0.10))
            return _fn

        def _arc_label(sa, da):
            label_r = RADIUS * 1.14
            def _fn(mob, alpha):
                angle = sa + alpha * da
                mob.move_to(np.array([np.cos(angle) * label_r,
                                      np.sin(angle) * label_r, 0.0]) + SHIFT)
            return _fn

        # ── Animate through training snapshots ────────────────────────────────
        for snap_i in range(1, n_snaps):
            ep      = epochs[snap_i]
            pts_new = all_pts[snap_i]
            w_new   = all_w[snap_i]

            end_dot_angles   = np.arctan2(pts_new[:, 1] - SHIFT[1], pts_new[:, 0] - SHIFT[0])
            delta_dot        = (end_dot_angles - cur_dot_angles + np.pi) % (2*np.pi) - np.pi
            end_arrow_angles = np.arctan2(w_new[:, 1], w_new[:, 0])
            delta_arrow      = (end_arrow_angles - cur_arrow_angles + np.pi) % (2*np.pi) - np.pi

            new_epoch_lbl = _epoch_text(ep)

            self.play(
                UpdateFromAlphaFunc(dots_group,
                    _arc_dots(cur_dot_angles.copy(), delta_dot, sample_r),
                    rate_func=smooth),
                *(UpdateFromAlphaFunc(weight_arrows[c],
                    _arc_arrow(cur_arrow_angles[c], delta_arrow[c], c),
                    rate_func=smooth)
                  for c in range(10)),
                *(UpdateFromAlphaFunc(digit_labels[c],
                    _arc_label(cur_arrow_angles[c], delta_arrow[c]),
                    rate_func=smooth)
                  for c in range(10)),
                Transform(epoch_lbl, new_epoch_lbl),
                run_time=2.2,
            )
            cur_dot_angles   = end_dot_angles
            cur_arrow_angles = end_arrow_angles
            self.wait(2.0)

        # ── Final annotation ──────────────────────────────────────────────────
        note = Text(
            "Each class clusters angularly around its softmax weight vector",
            font_size=20, color=GREY_B,
        ).to_edge(DOWN, buff=0.)
        self.play(FadeIn(note), run_time=0.8)
        self.wait(4.0)


# ── Display helper (Quarto / Jupyter) ─────────────────────────────────────────

_MIME = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".gif":  "image/gif",
}


def show_softmax(width: int = 800) -> None:
    """Embed the pre-rendered SoftmaxScene inline (Jupyter / Quarto)."""
    from pathlib import Path
    from IPython.display import HTML, display as _display

    media_root = Path(__file__).parent.parent / "media"
    matches = [p for p in media_root.rglob("SoftmaxScene.*") if p.suffix in _MIME]
    if not matches:
        raise FileNotFoundError(
            "No SoftmaxScene render found under media/. "
            "Run 'make softmax' (or cache + render) first."
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
