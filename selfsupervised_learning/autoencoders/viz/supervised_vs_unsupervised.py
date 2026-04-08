import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np

rng = np.random.default_rng(42)

n = 120
centers = [(-2, -1), (2, 1)]
X = np.vstack([rng.multivariate_normal(c, [[0.8, 0.3], [0.3, 0.6]], n // 2) for c in centers])
y = np.array([0] * (n // 2) + [1] * (n // 2))

COLORS = ["#4C72B0", "#DD8452"]
ELLIPSES = [(-2, -1, 17, 3, 2), (2, 1, 30, 4, 2.6)]


def _draw_supervised(ax):
    for cls, color, label in zip([0, 1], COLORS, ["Class A", "Class B"]):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=color, edgecolors="white",
                   linewidths=0.5, s=50, label=label)
    xx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    ax.plot(xx, -xx * 0.6 - 0.2, color="black", lw=1.5, linestyle="--", label="Decision boundary")
    ax.set_title("Supervised Learning", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")


def _draw_unsupervised(ax):
    ax.scatter(X[:, 0], X[:, 1], c="steelblue", edgecolors="white",
               linewidths=0.5, s=50)
    for cx, cy, angle, width, height in ELLIPSES:
        ax.add_patch(Ellipse((cx, cy), width=width, height=height, angle=angle,
                             edgecolor="tomato", facecolor="tomato", alpha=0.15, lw=2))
        ax.add_patch(Ellipse((cx, cy), width=width, height=height, angle=angle,
                             edgecolor="tomato", facecolor="none", lw=2, linestyle="--"))
    ax.set_title("Unsupervised Learning", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=8, label="Unlabeled data"),
        mpatches.Patch(facecolor="tomato", alpha=0.4, edgecolor="tomato", linestyle="--", label="Discovered clusters"),
    ], fontsize=8, loc="upper left")
    ax.set_aspect("equal")


def supervised():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    _draw_supervised(ax)
    plt.tight_layout()
    plt.show()


def unsupervised():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    _draw_unsupervised(ax)
    plt.tight_layout()
    plt.show()


def supervised_vs_unsupervised():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=300, sharex=True, sharey=True)
    _draw_supervised(axes[0])
    _draw_unsupervised(axes[1])
    plt.tight_layout()
    plt.show()
