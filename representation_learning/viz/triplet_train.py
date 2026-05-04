"""
MNIST supervised-contrastive training — model, cache, and live matplotlib plot.

Loss: SupCon (Khosla et al. 2020) — for each anchor, maximise the log-prob of
drawing a same-class sample relative to all other samples in the batch.  This
is a competitive / relative loss, so it is never geometrically infeasible: it
finds the best angular arrangement possible within 2-D without needing a hard
absolute margin.  Temperature controls the slack: lower T = stricter separation
(analogous to a small margin), higher T = more permissive.

No classification head — purely nonparametric / geometry-driven.

Run to train with live visualization and save cache:
    pixi run -e local-amd python viz/triplet_train.py
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    from .softmax_train import CLASS_COLORS, EMBED_DIM, EPOCHS_SAVE, N_PER_CLASS, ORBIT_RADII, norm_rows
except ImportError:
    from softmax_train import CLASS_COLORS, EMBED_DIM, EPOCHS_SAVE, N_PER_CLASS, ORBIT_RADII, norm_rows

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_PATH  = Path(__file__).parent.parent / "media" / "triplet_cache.npz"
# A hard margin (push all negative pairs below a fixed cosine threshold) is
# geometrically infeasible in 2-D with 10 classes: 10 evenly-spaced classes sit
# 36° apart, so adjacent pairs have cos(36°)≈0.81.  Any positive margin causes
# the optimizer to keep pushing those pairs apart and collapse distinct classes
# to the same angle as a local-minimum escape.  SupCon avoids this entirely by
# using a competitive (softmax) formulation; temperature plays the same role as
# inverse margin — lower T = stricter separation.
TEMPERATURE = 0.1


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_centroids(embs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-class mean direction of (unit-norm) embeddings — shape (10, 2)."""
    cents = np.zeros((10, 2))
    for c in range(10):
        mask = labels == c
        if mask.any():
            cents[c] = norm_rows(embs[mask].mean(axis=0, keepdims=True))[0]
    return cents


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """CNN with 2-D bottleneck; forward returns unit-norm embeddings only."""
    import torch.nn as nn
    import torch.nn.functional as F

    class TripletCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.emb = nn.Linear(32 * 7 * 7, EMBED_DIM)
            # BN with no learnable affine: zero-centres and equalises the two
            # embedding dimensions before L2-norm.  Without this, random init
            # maps all MNIST images into a tiny cone (cos≈0.99 for all pairs)
            # because the 1568→2 projection captures "average digit" in one
            # direction; the overwhelming negative loss then tears embeddings
            # apart incoherently and collapses distinct classes to the same angle.
            self.bn = nn.BatchNorm1d(EMBED_DIM, affine=False)

        def forward(self, x):
            return F.normalize(self.bn(self.emb(self.conv(x).flatten(1))), dim=1)

    return TripletCNN()


# ── Loss ──────────────────────────────────────────────────────────────────────

def supcon_loss(z, y, temperature: float = TEMPERATURE):
    """
    Supervised contrastive (SupCon) loss.

    For each anchor i, maximise the average log-probability of sampling a
    same-class example j relative to all other examples in the batch:

        L_i = -(1/|P(i)|) * sum_{j in P(i)} [ s_ij/T - log sum_{k≠i} exp(s_ik/T) ]

    Unlike a margin-based loss this is never geometrically infeasible: it finds
    the best attainable arrangement and never collapses classes to share an angle
    just to escape an impossible hard constraint.  Lower T gives sharper
    separation (analogous to a smaller margin).
    """
    import torch

    B   = z.size(0)
    sim = (z @ z.T) / temperature

    eye  = torch.eye(B, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(eye, float("-inf"))        # exclude self-pairs

    same  = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    n_pos = same.float().sum(dim=1).clamp(min=1)

    log_denom = torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob  = sim - log_denom                       # (B, B) log-probabilities

    per_anchor = -(log_prob * same.float()).sum(dim=1) / n_pos
    return per_anchor[same.any(dim=1)].mean()


# ── Live matplotlib plot ──────────────────────────────────────────────────────

class TripletLivePlot:
    """
    Matplotlib figure that updates in-place during triplet training.

    Shows unit-norm embeddings on per-class orbits and per-class centroid
    arrows that emerge as classes cluster angularly.
    """

    def __init__(self, labels: np.ndarray) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec

        self._plt    = plt
        self._labels = labels
        self._sample_r = ORBIT_RADII[labels]

        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])
        ax     = fig.add_subplot(gs[0])
        leg_ax = fig.add_subplot(gs[1])
        leg_ax.set_axis_off()

        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Orbit rings
        theta = np.linspace(0, 2 * np.pi, 360)
        for c, r in enumerate(ORBIT_RADII):
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    color=CLASS_COLORS[c], alpha=0.18, lw=0.8)

        # Per-class centroid arrows (initially pointing right)
        self._arrows = []
        for c in range(10):
            arr = ax.annotate(
                "", xy=(1.0, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=CLASS_COLORS[c],
                    lw=1.8,
                    mutation_scale=12,
                ),
            )
            self._arrows.append(arr)

        # Dot scatter
        xy = np.zeros((len(labels), 2))
        colors = [CLASS_COLORS[lbl] for lbl in labels]
        self._scatter = ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=14, alpha=0.7, linewidths=0)

        self._title = ax.set_title("Epoch 0", color="white", fontsize=14)

        legend_handles = [mpatches.Patch(color=CLASS_COLORS[c], label=str(c)) for c in range(10)]
        leg_ax.legend(handles=legend_handles, loc="center", frameon=False,
                      labelcolor="white", fontsize=11, title="Digit", title_fontsize=11)
        leg_ax.set_facecolor("#1a1a1a")

        fig.tight_layout()
        self.fig = fig
        self._ax  = ax
        fig.canvas.draw()
        plt.pause(0.05)

    def update(self, embs: np.ndarray, labels: np.ndarray,
               epoch: int, batch: int | None = None) -> None:
        xy = embs * self._sample_r[:, np.newaxis]           # already unit-norm
        self._scatter.set_offsets(xy)

        cents = compute_centroids(embs, labels)             # (10, 2)
        for c, arr in enumerate(self._arrows):
            arr.set_position((0.0, 0.0))
            arr.xy = (float(cents[c, 0]), float(cents[c, 1]))

        label = f"Epoch {epoch}"
        if batch is not None:
            label += f"  batch {batch}"
        self._title.set_text(label)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def block(self) -> None:
        self._plt.ioff()
        self._plt.show()


# ── Training / caching ────────────────────────────────────────────────────────

def train_and_cache(
    live:         bool  = False,
    temperature:  float = TEMPERATURE,
    update_every: int   = 10,
) -> None:
    """
    Train a 2-D-bottleneck CNN with SupCon loss and cache snapshots.

    Parameters
    ----------
    live
        Show a live matplotlib plot that updates during training.
    temperature
        SupCon temperature (lower = stricter separation, like a smaller margin).
    update_every
        Refresh the live plot every this many training batches.
    """
    import torch
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  temperature={temperature}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("~/.cache/mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)

    # Fixed visualization subset: N_PER_CLASS per class from test set
    idx, counts = [], [0] * 10
    for i in range(len(test_ds)):
        _, lbl = test_ds[i]
        if counts[lbl] < N_PER_CLASS:
            idx.append(i)
            counts[lbl] += 1
        if min(counts) >= N_PER_CLASS:
            break

    vis_loader = DataLoader(Subset(test_ds, idx), batch_size=N_PER_CLASS * 10, shuffle=False)
    vis_imgs, vis_labels = next(iter(vis_loader))
    vis_imgs = vis_imgs.to(device)
    vis_labels_np = vis_labels.numpy()

    model = build_model().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    def snap() -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            z = model(vis_imgs)
        return z.cpu().numpy(), vis_labels_np

    plot = TripletLivePlot(vis_labels_np) if live else None

    saves: dict[int, tuple] = {}

    embs, lbls = snap()
    saves[0] = (embs, lbls)
    print(f"  Epoch  0 (init)")
    if plot:
        plot.update(embs, lbls, epoch=0)

    for ep in range(1, max(EPOCHS_SAVE) + 1):
        model.train()
        for batch_idx, (imgs, tgts) in enumerate(train_loader):
            imgs, tgts = imgs.to(device), tgts.to(device)
            opt.zero_grad()
            z    = model(imgs)
            loss = supcon_loss(z, tgts, temperature=temperature)
            loss.backward()
            opt.step()

            if plot and batch_idx % update_every == 0:
                embs, lbls = snap()
                plot.update(embs, lbls, epoch=ep, batch=batch_idx)
                model.train()

        if ep in EPOCHS_SAVE:
            embs, lbls = snap()
            saves[ep] = (embs, lbls)
            print(f"  Epoch {ep:2d}")
            if plot:
                plot.update(embs, lbls, epoch=ep)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    d = {"epochs": np.array(sorted(saves))}
    for ep, (z, lbls) in saves.items():
        d[f"emb_{ep}"] = z
        d[f"lbl_{ep}"] = lbls
    np.savez(CACHE_PATH, **d)
    print(f"Saved → {CACHE_PATH}")

    if plot:
        plot.block()


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def load_cache() -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return list of (epoch, embeddings, labels) tuples."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}\n"
            "Run:  pixi run -e local-amd python viz/triplet_train.py"
        )
    d = np.load(CACHE_PATH)
    return [
        (int(ep), d[f"emb_{ep}"], d[f"lbl_{ep}"])
        for ep in d["epochs"]
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_cache(live=True)
