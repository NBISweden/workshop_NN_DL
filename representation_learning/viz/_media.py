"""Shared helpers for embedding pre-rendered Manim scenes."""
from __future__ import annotations

import re
from pathlib import Path

_MIME_TYPES: dict[str, str] = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".gif":  "image/gif",
}


def _best_render(media_root: Path, scene_name: str) -> Path:
    """Return the highest-quality pre-rendered file for *scene_name*.

    Quality is ranked by (height, fps) parsed from the Manim output directory
    name (e.g. ``720p60``), so the result is deterministic regardless of file
    system modification times.
    """
    matches = [
        p for p in media_root.rglob(f"{scene_name}.*")
        if p.suffix in _MIME_TYPES and "partial_movie_files" not in p.parts
    ]
    if not matches:
        raise FileNotFoundError(
            f"No {scene_name} render found under {media_root}. "
            f"Run 'make media' before rendering the slides."
        )

    def _quality(p: Path) -> tuple[int, int]:
        m = re.search(r"(\d+)p(\d+)", p.parent.name)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return max(matches, key=_quality)


def _embed_render(scene_name: str, width: int, caller_file: str) -> None:
    """Locate and display the best available render of *scene_name*."""
    from IPython.display import HTML, display as _display

    media_root = Path(caller_file).parent.parent / "media"
    media_path = _best_render(media_root, scene_name)
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


def show_dot_product(width: int = 700) -> None:
    _embed_render("DotProductScene", width, __file__)


def show_softmax(width: int = 800) -> None:
    _embed_render("SoftmaxScene", width, __file__)


def show_triplet(width: int = 800) -> None:
    _embed_render("TripletScene", width, __file__)


def show_representational_collapse(width: int = 800) -> None:
    _embed_render("RepresentationCollapseScene", width, __file__)


def show_triplet_concept(width: int = 800) -> None:
    _embed_render("TripletConceptScene", width, __file__)
