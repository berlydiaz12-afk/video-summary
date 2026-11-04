"""Shared colors and figure style.

Every entity keeps one color in every figure and image (removal reasons, transitions,
window decisions, short-shot candidates, miniclips), so a color learned in one output
means the same thing in all the others.
"""

from __future__ import annotations

from typing import Any

# Chart chrome (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# One color per entity, from a colorblind-checked categorical palette.
SERIES = "#2a78d6"          # single data series (luma, probability, F_j)
REASON_COLORS = {           # frame filtering: why a frame was removed
    "black": "#6250d6",
    "uniform": "#eb6834",
    "static": "#1baf7a",
}
TRANSITION = "#eda100"      # shot detection: frames with p > threshold
KEPT = "#008300"            # redundancy removal: kept window (it becomes a miniclip)
DISCARDED = "#e34948"       # redundancy removal: discarded window
SHORT_SHOT = "#8c5a2b"      # short-shot candidate (T frames, r = 1; always kept)
MINICLIP = "#e87ba4"        # miniclip: the T frames given to Swin
CANDIDATE = MUTED           # candidate window (before redundancy removal)

# Sequential blue ramp (light → dark) for magnitudes, e.g. distance matrices.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
SHOT_TONES = ("#86b6ef", "#256abf")  # alternating shots on the timeline


def rgb(hex_color: str) -> tuple[int, int, int]:
    """``#rrggbb`` → ``(r, g, b)`` for OpenCV drawing on RGB arrays."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def style_axes(ax: Any) -> None:
    """Recessive axes: no top/right spines, hairline grid, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_color(INK)
    ax.title.set_size(10)


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] = (10.0, 4.0),
    sharex: bool = False,
    height_ratios: list[float] | None = None,
) -> tuple[Any, Any]:
    """Styled figure and a 2-D array of axes.

    The figure is closed in pyplot right away: it does not pile up in notebooks and
    still shows when it is the last expression of a cell or passed to ``display``.
    """
    import matplotlib.pyplot as plt

    gridspec = {"height_ratios": height_ratios} if height_ratios else None
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, squeeze=False,
                             gridspec_kw=gridspec, facecolor=SURFACE, layout="constrained")
    plt.close(fig)
    for ax in axes.flat:
        style_axes(ax)
    return fig, axes


def save_figure(fig: Any, path: Any, dpi: int = 150) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
