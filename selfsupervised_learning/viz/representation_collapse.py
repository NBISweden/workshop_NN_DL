"""
Representational collapse — Manim animation.

Shows what happens when only positive-pair (same-class) attraction is used
with no negative examples: embeddings first cluster by class (looks promising)
then all classes collapse to a single direction (constant prediction).

Pure numpy simulation — no training data or cache required.

Render:
    python -m manim -ql viz/representation_collapse.py RepresentationCollapseScene
"""
from __future__ import annotations

import numpy as np

import manim
from manim import (
    BLACK, WHITE, GREY_A, RED,
    UP, DOWN, LEFT, RIGHT,
    Scene, VGroup,
    Arrow, Circle, Dot, Text,
    FadeIn, GrowArrow, Transform, UpdateFromAlphaFunc,
    smooth, ManimColor,
)

try:
    from .softmax_train import CLASS_COLORS, N_PER_CLASS, ORBIT_RADII
except ImportError:
    from softmax_train import CLASS_COLORS, N_PER_CLASS, ORBIT_RADII

# ── Render config ──────────────────────────────────────────────────────────────

PIXEL_WIDTH  = 1280
PIXEL_HEIGHT = 1280
FRAME_WIDTH  = 10

manim.config.pixel_width  = PIXEL_WIDTH
manim.config.pixel_height = PIXEL_HEIGHT
manim.config.frame_width  = FRAME_WIDTH

Text.set_default(font="Ubuntu")

# ── Scene config ───────────────────────────────────────────────────────────────

RADIUS            = 3.0
SHIFT             = LEFT * 0
MANIM_ORBIT_RADII = ORBIT_RADII * RADIUS

# ── Simulation config ──────────────────────────────────────────────────────────

SAVE_AT      = [0, 5, 15, 35, 80]
INTRA_LR     = 0.18                       # per-step intra-class attraction
GLOBAL_LR    = 0.06                       # per-step global collapse pull
COLLAPSE_DIR = np.array([1.0, 0.0])       # target collapse direction

PHASE_LABELS = {
    0:  "Random initialization",
    5:  "Clusters begin to form…",
    15: "Classes cluster further…",
    35: "Classes start to merge",
    80: "Representational collapse",
}


# ── Simulation ─────────────────────────────────────────────────────────────────

def _compute_centroids(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Normalized per-class mean direction — shape (10, 2)."""
    cents = np.zeros((10, 2))
    for c in range(10):
        mask = labels == c
        if mask.any():
            mean = z[mask].mean(axis=0)
            norm = np.linalg.norm(mean)
            cents[c] = mean / norm if norm > 1e-8 else COLLAPSE_DIR
    return cents


def _simulate_collapse(seed: int = 42) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Simulate collapse under purely attractive loss (no negative examples).

    Two forces act on each unit-norm embedding:
      1. Intra-class: pull toward its class centroid  (INTRA_LR)
      2. Global:      pull toward COLLAPSE_DIR        (GLOBAL_LR)

    The global pull represents the missing repulsion between classes.
    Without negative examples nothing prevents all centroids from
    drifting toward the same direction.
    """
    rng    = np.random.default_rng(seed)
    N      = N_PER_CLASS * 10
    labels = np.repeat(np.arange(10), N_PER_CLASS)

    angles = rng.uniform(-np.pi, np.pi, N)
    z      = np.column_stack([np.cos(angles), np.sin(angles)])

    save_set   = set(SAVE_AT)
    snapshots: list[tuple[int, np.ndarray, np.ndarray]] = []

    if 0 in save_set:
        snapshots.append((0, z.copy(), labels.copy()))

    for step in range(1, max(SAVE_AT) + 1):
        cents  = _compute_centroids(z, labels)
        z     += INTRA_LR * (cents[labels] - z)
        z     += GLOBAL_LR * (COLLAPSE_DIR - z)
        norms  = np.linalg.norm(z, axis=1, keepdims=True)
        z     /= np.maximum(norms, 1e-8)

        if step in save_set:
            snapshots.append((step, z.copy(), labels.copy()))

    return snapshots


# ── Coordinate helper ──────────────────────────────────────────────────────────

def _mp(xy: np.ndarray, r: float = RADIUS) -> np.ndarray:
    return np.array([xy[0] * r, xy[1] * r, 0.0]) + SHIFT


# ── Manim scene ────────────────────────────────────────────────────────────────

class RepresentationCollapseScene(Scene):
    def construct(self) -> None:
        snapshots = _simulate_collapse()
        steps     = [s[0] for s in snapshots]
        labels    = snapshots[0][2]
        N         = N_PER_CLASS * 10

        sample_r = MANIM_ORBIT_RADII[labels]

        # Pre-compute dot positions and class centroids per snapshot
        all_pts: list[np.ndarray] = []
        all_cen: list[np.ndarray] = []
        for _step, embs, lbls in snapshots:
            pts = np.column_stack(
                [embs[:, 0] * sample_r, embs[:, 1] * sample_r, np.zeros(N)]
            ) + SHIFT
            all_pts.append(pts)
            all_cen.append(_compute_centroids(embs, lbls))

        # ── Background ────────────────────────────────────────────────────────
        self.camera.background_color = BLACK

        # ── Orbit rings ───────────────────────────────────────────────────────
        orbit_rings = VGroup(*[
            Circle(radius=float(MANIM_ORBIT_RADII[c]),
                   stroke_width=1.0, stroke_opacity=0.25,
                   color=ManimColor(CLASS_COLORS[c])).shift(SHIFT)
            for c in range(10)
        ])
        self.play(FadeIn(orbit_rings), run_time=0.5)

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text(
            "Positive-only loss — no negative examples",
            font_size=24, color=WHITE,
        ).to_edge(UP, buff=0.)
        self.play(FadeIn(title), run_time=0.5)

        # ── Phase label ───────────────────────────────────────────────────────
        def _phase_text(step: int) -> Text:
            msg   = PHASE_LABELS.get(step, f"Step {step}")
            color = RED if step == SAVE_AT[-1] else GREY_A
            return Text(msg, font_size=26, color=color).to_corner(
                DOWN + RIGHT, buff=0.5
            )

        phase_lbl = _phase_text(steps[0])
        self.add(phase_lbl)

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

        # ── Dots (step 0) ─────────────────────────────────────────────────────
        dots = [
            Dot(all_pts[0][i], radius=0.065,
                color=ManimColor(CLASS_COLORS[labels[i]]), fill_opacity=0.75)
            for i in range(N)
        ]
        self.play(FadeIn(VGroup(*dots)), run_time=0.8)
        self.wait(1.5)

        # Track per-dot angle so transitions sweep along the orbit, not straight lines
        cur_angles = np.arctan2(
            all_pts[0][:, 1] - SHIFT[1],
            all_pts[0][:, 0] - SHIFT[0],
        )
        dots_group = VGroup(*dots)

        def _make_arc_updater(start_a, delta_a, radii):
            def _fn(mob, alpha):
                angles_t = start_a + alpha * delta_a
                xs = np.cos(angles_t) * radii + SHIFT[0]
                ys = np.sin(angles_t) * radii + SHIFT[1]
                for i, d in enumerate(mob):
                    d.move_to(np.array([xs[i], ys[i], 0.0]))
            return _fn

        # ── Animate through snapshots ─────────────────────────────────────────
        for snap_i in range(1, len(snapshots)):
            step    = steps[snap_i]
            pts_new = all_pts[snap_i]
            cen_new = all_cen[snap_i]

            end_angles   = np.arctan2(pts_new[:, 1] - SHIFT[1], pts_new[:, 0] - SHIFT[0])
            delta_angles = (end_angles - cur_angles + np.pi) % (2 * np.pi) - np.pi

            new_phase_lbl  = _phase_text(step)
            # new_cen_arrows = [_make_centroid_arrow(cen_new[c], c) for c in range(10)]

            self.play(
                UpdateFromAlphaFunc(
                    dots_group,
                    _make_arc_updater(cur_angles.copy(), delta_angles, sample_r),
                    rate_func=smooth,
                ),
                # *(Transform(centroid_arrows[c], new_cen_arrows[c]) for c in range(10)),
                Transform(phase_lbl, new_phase_lbl),
                run_time=2.5,
            )
            cur_angles = end_angles
            self.wait(2.0 if step < SAVE_AT[-1] else 3.0)

        # ── Final annotation ──────────────────────────────────────────────────
        note = Text(
            "All classes share one representation — model predicts a constant",
            font_size=19, color=RED,
        ).to_edge(DOWN, buff=0.)
        self.play(FadeIn(note), run_time=0.8)
        self.wait(4.0)


# ── Display helper ─────────────────────────────────────────────────────────────

_MIME = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".gif":  "image/gif",
}


def show_representational_collapse(width: int = 800) -> None:
    """Embed the pre-rendered RepresentationCollapseScene inline (Jupyter / Quarto)."""
    from pathlib import Path
    from IPython.display import HTML, display as _display

    media_root = Path(__file__).parent.parent / "media"
    matches = [
        p for p in media_root.rglob("RepresentationCollapseScene.*")
        if p.suffix in _MIME
    ]
    if not matches:
        raise FileNotFoundError(
            "No RepresentationCollapseScene render found under media/. "
            "Run 'make rc' first."
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
