"""
MNIST softmax anchor — model, training, cache, and live matplotlib plot.

Run to train with live visualization and save cache:
    pixi run -e local-amd python viz/softmax_train.py
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# ── Shared constants ──────────────────────────────────────────────────────────

CACHE_PATH  = Path(__file__).parent.parent / "media" / "softmax_cache.npz"
EPOCHS_SAVE = [0, 1, 3, 7, 15, 30]
N_PER_CLASS = 60
EMBED_DIM   = 2

# Digit 0 innermost orbit, digit 9 outermost (matplotlib unit-circle scale)
ORBIT_RADII: np.ndarray = np.linspace(0.8, 1.2, 10)

CLASS_COLORS = [        # Tableau-10 palette, one colour per digit
    "#4e79a7",  # 0  blue
    "#f28e2b",  # 1  orange
    "#59a14f",  # 2  green
    "#e15759",  # 3  red
    "#b07aa1",  # 4  purple
    "#76b7b2",  # 5  teal
    "#ff9da7",  # 6  pink
    "#9c755f",  # 7  brown
    "#9d9d9d",  # 8  grey
    "#edc948",  # 9  yellow
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def norm_rows(v: np.ndarray) -> np.ndarray:
    """Row-normalize array to unit L2 norm."""
    n = np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)
    return v / n


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """Return a fresh MNIST CNN with a 2-D bottleneck embedding layer.

    Both the embedding and the softmax weight rows are L2-normalised before
    computing logits, so the scores are pure cosine similarities scaled by a
    learnable temperature.  The second return value of forward() is the
    unit-norm embedding, ready to plot directly on the unit circle.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class MnistCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.emb  = nn.Linear(32 * 7 * 7, EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, 10, bias=False)

        def forward(self, x):
            z      = self.emb(self.conv(x).flatten(1))
            z_norm = F.normalize(z, dim=1)                  # unit-norm embeddings
            w_norm = F.normalize(self.head.weight, dim=1)   # unit-norm weight rows
            logits = z_norm @ w_norm.T                      # cosine similarity scores
            return logits, z_norm

    return MnistCNN()


# ── Live matplotlib plot ──────────────────────────────────────────────────────

class LivePlot:
    """
    Matplotlib figure that updates in-place during training.

    Shows normalized embeddings on per-class orbits and softmax weight
    vectors as arrows — mirrors the Manim SoftmaxScene visualization.
    """

    def __init__(self, labels: np.ndarray) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.legend_handler import HandlerPatch
        from matplotlib.gridspec import GridSpec

        self._plt = plt
        self._labels = labels
        # Per-sample orbit radius, fixed for the lifetime of the plot
        self._sample_r = ORBIT_RADII[labels]

        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])
        ax       = fig.add_subplot(gs[0])
        leg_ax   = fig.add_subplot(gs[1])
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

        # Softmax weight arrows (start pointing right; updated each call)
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

        # Dot scatter, initial positions at origin
        xy = np.zeros((len(labels), 2))
        colors = [CLASS_COLORS[lbl] for lbl in labels]
        self._scatter = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=colors, s=14, alpha=0.7, linewidths=0,
        )

        self._title = ax.set_title("Epoch 0", color="white", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Legend
        legend_handles = [
            mpatches.Patch(color=CLASS_COLORS[c], label=str(c))
            for c in range(10)
        ]
        leg_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=False,
            labelcolor="white",
            fontsize=11,
            title="Digit",
            title_fontsize=11,
        )
        leg_ax.set_facecolor("#1a1a1a")

        fig.tight_layout()
        self.fig = fig
        self._ax  = ax
        fig.canvas.draw()
        plt.pause(0.05)

    def update(
        self,
        embs: np.ndarray,
        weights: np.ndarray,
        epoch: int,
        batch: int | None = None,
    ) -> None:
        ne = norm_rows(embs)                          # (N, 2) unit directions
        xy = ne * self._sample_r[:, np.newaxis]       # place on orbits
        self._scatter.set_offsets(xy)

        nw = norm_rows(weights)                       # (10, 2) unit directions
        for c, arr in enumerate(self._arrows):
            arr.set_position((0.0, 0.0))
            arr.xy = (float(nw[c, 0]), float(nw[c, 1]))

        label = f"Epoch {epoch}"
        if batch is not None:
            label += f"  batch {batch}"
        self._title.set_text(label)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def block(self) -> None:
        """Keep window open after training finishes."""
        self._plt.ioff()
        self._plt.show()


# ── Training / caching ────────────────────────────────────────────────────────

def train_and_cache(live: bool = False, update_every: int = 10) -> None:
    """
    Train a 2-D-bottleneck MNIST CNN and cache embedding snapshots.

    Parameters
    ----------
    live
        Show a live matplotlib plot that updates during training.
    update_every
        Update the live plot every this many training batches.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("~/.cache/mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)

    # Fixed visualization subset: N_PER_CLASS per class from test set
    idx, counts = [], [0] * 10
    for i in range(len(test_ds)):
        _, lbl = test_ds[i]
        if counts[lbl] < N_PER_CLASS:
            idx.append(i)
            counts[lbl] += 1
        if min(counts) >= N_PER_CLASS:
            break
    vis_loader = DataLoader(
        Subset(test_ds, idx), batch_size=N_PER_CLASS * 10, shuffle=False
    )

    # Pre-load visualization images once to avoid re-creating the iterator
    vis_imgs, vis_labels = next(iter(vis_loader))
    vis_imgs = vis_imgs.to(device)

    model = build_model().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    def snap() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            _, z = model(vis_imgs)
            w    = model.head.weight.detach().cpu().numpy()
        return z.cpu().numpy(), vis_labels.numpy(), w

    def test_acc() -> float:
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for imgs, lbls in DataLoader(test_ds, batch_size=512, num_workers=0):
                ok  += (model(imgs.to(device))[0].argmax(1).cpu() == lbls).sum().item()
                tot += len(lbls)
        return ok / tot

    plot = LivePlot(vis_labels.numpy()) if live else None

    saves: dict[int, tuple] = {}

    # Epoch 0 = before any training
    embs, lbls, weights = snap()
    saves[0] = (embs, lbls, weights)
    print(f"  Epoch  0 (init):  acc={test_acc():.1%}")
    if plot:
        plot.update(embs, weights, epoch=0)

    for ep in range(1, max(EPOCHS_SAVE) + 1):
        model.train()
        for batch_idx, (imgs, tgts) in enumerate(train_loader):
            imgs, tgts = imgs.to(device), tgts.to(device)
            opt.zero_grad()
            crit(model(imgs)[0], tgts).backward()
            opt.step()

            if plot and batch_idx % update_every == 0:
                embs, _, weights = snap()
                plot.update(embs, weights, epoch=ep, batch=batch_idx)
                model.train()

        if ep in EPOCHS_SAVE:
            embs, lbls, weights = snap()
            saves[ep] = (embs, lbls, weights)
            print(f"  Epoch {ep:2d}:          acc={test_acc():.1%}")
            if plot:
                plot.update(embs, weights, epoch=ep)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    d = {"epochs": np.array(sorted(saves))}
    for ep, (z, lbls, w) in saves.items():
        d[f"emb_{ep}"] = z
        d[f"lbl_{ep}"] = lbls
        d[f"w_{ep}"]   = w
    np.savez(CACHE_PATH, **d)
    print(f"Saved → {CACHE_PATH}")

    if plot:
        plot.block()


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def load_cache() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (epoch, embeddings, labels, weights) tuples."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}\n"
            "Run:  pixi run -e local-amd python viz/softmax_train.py"
        )
    d = np.load(CACHE_PATH)
    return [
        (int(ep), d[f"emb_{ep}"], d[f"lbl_{ep}"], d[f"w_{ep}"])
        for ep in d["epochs"]
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_cache(live=True)
