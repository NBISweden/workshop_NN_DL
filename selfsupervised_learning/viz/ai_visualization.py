import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from typing import Literal
import numpy as np

_LEVELS = [
    # key, radius, offsets, aspects, text_offset, color, full name, example (not in contained subset)
    ("ai", 1.00, 0.0, 1.0, 0.2, "#AED6F1", "Artificial Intelligence", "Expert Systems"),
    ("ml", 0.75, -0.19, 1.15, 0.2, "#A9DFBF", "Machine Learning",        "Random Forest"),
    ("rl", 0.50, -0.4, 1.4, 0.2, "#F9E79F", "Representation Learning", "Word2Vec"),
    ("dl", 0.28, -0.6, 1.6, 0.2, "#F1948A", "Deep Learning",           "Conv. Neural Networks"),
]

def _draw_donut(ax, active: list[str]):
    active_set = set(active)
    keys = [l[0] for l in _LEVELS]

    # Draw from largest to smallest so smaller ellipses sit on top.
    for key, radius, cy, aspect, text_offset, color, name, example in _LEVELS:
        if key not in active_set:
            continue
        ellipse = Ellipse((0, cy), width=2 * radius * aspect, height=2 * radius,
                          color=color, alpha=0.85, zorder=keys.index(key))
        ax.add_patch(ellipse)

        # Place label at the top of the ellipse (just inside the upper edge).
        label_y = cy + radius - text_offset
        ax.text(0, label_y, name, ha="center", va="top",
                fontsize=8, fontweight="bold", color="#222222", zorder=10)
        ax.text(0, label_y - 0.11, f"e.g. {example}", ha="center", va="top",
                fontsize=6.5, color="#555555", style="italic", zorder=10)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")


def ai_donut(subsets: list[list[Literal['ai', 'ml', 'dl', 'rl']]]):
    # Create a figure with an "AI donut" for each subplot.
    # Each subplot shows nested circles (ai > ml > rl > dl).
    # Circles in `active` are filled; others are faded.
    # "rl" stands for "representation learning".
    n_subplots = len(subsets)
    fig, axes = plt.subplots(1, n_subplots,
                             dpi=300,
                             figsize=(4 * n_subplots, 4.5))
    if n_subplots == 1:
        axes = [axes]

    for ax, active in zip(axes, subsets):
        _draw_donut(ax, active)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ai_donut([
        #["ai"],
        ["ai", "ml", "dl"],
        #["ai", "ml", "rl"],
        ["ai", "ml", "rl", "dl"],
    ])

