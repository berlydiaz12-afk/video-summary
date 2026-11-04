"""matplotlib figures: time series, histograms, distance matrices, θ_H sweep and timeline.

Every function returns a ``Figure`` (shown in a notebook when it is the last
expression of a cell, or with ``display``); :func:`style.save_figure` writes it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from preproc.structs import true_runs

from . import style
from .frame_filtering import REASON_NAMES, FrameFilteringAnalysis
from .miniclips import MiniclipAnalysis
from .redundancy_removal import RedundancyAnalysis
from .shot_detection import ShotDetectionAnalysis
from .timeline import FrameTimeline


def _spans(ax: Any, runs_s: Sequence[tuple[float, float]], color: str, alpha: float = 0.25) -> None:
    """Full-height shaded spans ``(start_s, end_s)`` that do not change the y limits."""
    if not runs_s:
        return
    from matplotlib.collections import PolyCollection

    verts = [[(a, 0.0), (a, 1.0), (b, 1.0), (b, 0.0)] for a, b in runs_s]
    ax.add_collection(PolyCollection(verts, transform=ax.get_xaxis_transform(), facecolors=color,
                                     alpha=alpha, linewidths=0), autolim=False)


def _mask_runs_s(mask: np.ndarray, fps: int) -> list[tuple[float, float]]:
    return [(a / fps, b / fps) for a, b in true_runs(mask)]


def _threshold_line(ax: Any, value: float, label: str) -> None:
    ax.axhline(value, color=style.INK, linestyle="--", linewidth=1)
    ax.annotate(label, xy=(1, value), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=style.INK)


def _reason_handles(codes: np.ndarray) -> list[Any]:
    from matplotlib.patches import Patch

    return [Patch(facecolor=style.REASON_COLORS[name], alpha=0.5, label=f"removed: {name}")
            for code, name in REASON_NAMES.items() if (codes == code).any()]


def plot_frame_filtering(filtering: FrameFilteringAnalysis, fps: int) -> Any:
    """Mean luma, luma std and d_k over time, with θ_L, θ_σ, θ_E and the removed frames shaded."""
    cfg = filtering.config
    t = np.arange(filtering.n_frames) / fps
    diff = filtering.frame_diff.copy()
    diff[~np.isfinite(diff)] = np.nan
    diff = np.maximum(diff, 1e-2)
    codes = filtering.reason_codes()

    fig, axes = style.new_figure(3, 1, figsize=(12, 7.5), sharex=True)
    series = (
        (axes[0, 0], filtering.luma_mean, cfg.luma_threshold, "mean luma", "θ_L"),
        (axes[1, 0], filtering.luma_std, cfg.std_threshold, "luma std", "θ_σ"),
        (axes[2, 0], diff, cfg.static_threshold, "d_k (mean |f_k − f_k−1|)", "θ_E"),
    )
    for ax, y, thr, label, symbol in series:
        for code, name in REASON_NAMES.items():
            _spans(ax, _mask_runs_s(codes == code, fps), style.REASON_COLORS[name])
        ax.plot(t, y, color=style.SERIES, linewidth=0.8)
        _threshold_line(ax, thr, f"{symbol} = {thr:g}")
        ax.set_ylabel(label)
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_xlabel("time (s)")
    handles = _reason_handles(codes)
    if handles:
        axes[0, 0].legend(handles=handles, loc="upper left", fontsize=8, frameon=False, ncol=len(handles))
    axes[0, 0].set_title(
        f"Frame filtering: {filtering.n_removed} of {filtering.n_frames} frames removed "
        f"({filtering.n_removed / max(filtering.n_frames, 1):.1%}), {len(filtering.segments)} segments")
    return fig


def plot_transition_probabilities(shots: ShotDetectionAnalysis, fps: int) -> Any:
    """TransNet V2 probability per frame, the threshold, segment limits and removed frames."""
    n = len(shots.probabilities)
    t = np.arange(n) / fps
    fig, axes = style.new_figure(1, 1, figsize=(12, 3.6))
    ax = axes[0, 0]
    _spans(ax, _mask_runs_s(np.isnan(shots.probabilities), fps), style.GRIDLINE, alpha=0.9)
    limits = sorted({s.start / fps for s in shots.segments} | {s.end / fps for s in shots.segments})
    if limits:
        ax.vlines(limits, 0, 1, transform=ax.get_xaxis_transform(), colors=style.MUTED,
                  linestyles=":", linewidth=0.8)
    ax.plot(t, shots.probabilities, color=style.SERIES, linewidth=0.8)
    _threshold_line(ax, shots.threshold, f"threshold = {shots.threshold:g}")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, n / fps)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("p(transition)")
    ax.set_title(f"Shot detection: {len(shots.transitions)} transitions, {len(shots.shots)} shots "
                 "(gray: removed frames; dotted: segment limits)")
    return fig


def plot_shot_lengths(
    shots: ShotDetectionAnalysis,
    fps: int,
    strides: Sequence[int] = (8, 4, 2),
) -> Any:
    """Histogram of shot durations with the window length L = T·r for several strides.

    With short-shot candidates, a line at T marks the shortest shot that still gives a candidate.
    """
    fig, axes = style.new_figure(1, 1, figsize=(9, 4))
    ax = axes[0, 0]
    durations = np.array([s.length for s in shots.shots], dtype=np.float64) / fps
    if durations.size == 0:
        ax.text(0.5, 0.5, "no shots", ha="center", va="center", transform=ax.transAxes, color=style.INK_SECONDARY)
        return fig
    lo, hi = max(durations.min(), 1.0 / fps), durations.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 40) if hi > lo * 1.01 else 10
    ax.hist(durations, bins=bins, color=style.SEQUENTIAL_BLUE[3], edgecolor=style.SURFACE, linewidth=1)
    if np.ndim(bins):
        ax.set_xscale("log")
    T = shots.summary.frames_per_clip
    for r, dash in zip(strides, ("-", "--", ":", "-.")):
        length = T * r
        current = " (current)" if r == shots.summary.stride else ""
        ax.axvline(length / fps, color=style.INK, linestyle=dash, linewidth=1,
                   label=f"L = {length} frames, r = {r} ({length / fps:.2f} s){current}")
    if shots.summary.short_shot_windows:
        ax.axvline(T / fps, color=style.SHORT_SHOT, linestyle="-", linewidth=2,
                   label=f"T = {T} frames ({T / fps:.2f} s): shortest shot with a short-shot candidate")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("shot duration (s, log scale)")
    ax.set_ylabel("shots")
    c = shots.counts()
    ax.set_title(f"{len(shots.shots)} shots: {c['shots_with_window']} reach L = {shots.window_length} frames, "
                 f"{c['shots_with_short_candidate']} give a short-shot candidate, "
                 f"{c['shots_without_candidate']} give none")
    return fig


def plot_distance_matrix(redundancy: RedundancyAnalysis, shot: int) -> Any:
    """Pairwise mean Hamming distance between the candidate windows of one shot."""
    from matplotlib.colors import LinearSegmentedColormap

    ids, matrix = redundancy.shot_distance_matrix(shot)
    k = len(ids)
    fig, axes = style.new_figure(1, 1, figsize=(min(3 + 0.45 * k, 12), min(2.6 + 0.4 * k, 11)))
    ax = axes[0, 0]
    ax.grid(False)
    vmax = max(32.0, float(matrix.max()) if matrix.size else 32.0)
    cmap = LinearSegmentedColormap.from_list("sequential_blue", style.SEQUENTIAL_BLUE)
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.set_label("mean Hamming distance", fontsize=8, color=style.INK_SECONDARY)
    bar.ax.axhline(redundancy.threshold, color=style.INK, linewidth=1.5)
    bar.ax.tick_params(labelsize=7, colors=style.MUTED, labelcolor=style.INK_SECONDARY)
    kept = set(redundancy.kept)
    if k <= 40:
        labels = [f"{i}{'' if i in kept else ' ×'}" for i in ids]
        ax.set_xticks(range(k), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(k), labels, fontsize=7)
    if k <= 15:
        for a in range(k):
            for b in range(k):
                v = matrix[a, b]
                ax.text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.55 * vmax else style.INK)
    ax.set_title(f"Shot {shot}: {k} windows (× = discarded), θ_H = {redundancy.threshold:g} (line on the scale)")
    return fig


def plot_hamming_threshold_sweep(rows: Sequence[Mapping[str, Any]], current: float | None = None) -> Any:
    """Miniclips the video gives for each θ_H (rows of ``sweep_hamming_threshold``).

    Every kept candidate is a miniclip, so this curve is the only control of how many
    miniclips a video gives; the dashed line marks the θ_H of the configuration.
    """
    fig, axes = style.new_figure(1, 1, figsize=(9, 4))
    ax = axes[0, 0]
    if not rows:
        ax.text(0.5, 0.5, "no sweep values", ha="center", va="center", transform=ax.transAxes,
                color=style.INK_SECONDARY)
        return fig
    theta = np.array([float(r["hamming_threshold"]) for r in rows])
    clips = np.array([int(r["miniclips"]) for r in rows])
    ax.plot(theta, clips, color=style.MINICLIP, linewidth=2, marker="o", markersize=6,
            markeredgecolor=style.SURFACE, markeredgewidth=1.5)
    ax.set_ylim(0, max(1, int(clips.max())) * 1.12)
    ax.set_xlabel("θ_H (mean Hamming distance, a window is discarded when d ≤ θ_H)")
    ax.set_ylabel("miniclips")
    short = int(rows[0].get("short_shot_candidates", 0))
    if short:
        ax.axhline(short, color=style.SHORT_SHOT, linestyle=":", linewidth=1)
        ax.annotate(f"{short} short-shot candidates (always kept)", xy=(0, short),
                    xycoords=("axes fraction", "data"), xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=style.INK_SECONDARY)
    title = f"θ_H sweep: {int(rows[0]['windows'])} windows"
    if current is not None:
        ax.axvline(current, color=style.INK, linestyle="--", linewidth=1)
        nearest = int(np.argmin(np.abs(theta - current)))
        if np.isclose(theta[nearest], current):
            ax.annotate(f"θ_H = {current:g}: {clips[nearest]} miniclips", xy=(theta[nearest], clips[nearest]),
                        xytext=(6, 8), textcoords="offset points", fontsize=8, color=style.INK)
        title += f" (dashed: θ_H = {current:g} of the configuration)"
    ax.set_title(title)
    return fig


def plot_timeline(timeline: FrameTimeline, shots: ShotDetectionAnalysis, miniclips: MiniclipAnalysis) -> Any:
    """Whole video: probability, removed frames, shots, candidates, redundancy decisions and miniclips."""
    from matplotlib.patches import Patch

    fps, n = timeline.fps, timeline.n_frames
    fig, axes = style.new_figure(2, 1, figsize=(14, 5.6), sharex=True, height_ratios=[1.0, 2.4])
    ax_p, ax = axes[0, 0], axes[1, 0]

    t = np.arange(n) / fps
    ax_p.plot(t, timeline.probability, color=style.SERIES, linewidth=0.7)
    ax_p.axhline(timeline.threshold, color=style.INK, linestyle="--", linewidth=1)
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("TransNet p")

    rows = ["removed frames", "shots", "candidates", "redundancy", "miniclips"]
    y = {name: len(rows) - 1 - i for i, name in enumerate(rows)}
    height = 0.7

    def bars(row: str, spans: list[tuple[float, float]], colors: Any) -> None:
        if spans:
            ax.broken_barh(spans, (y[row] - height / 2, height), facecolors=colors, linewidth=0)

    def spans(windows: Sequence[Any]) -> list[tuple[float, float]]:
        return [(w.start / fps, w.length / fps) for w in windows]

    for code, name in REASON_NAMES.items():
        bars("removed frames", [(a / fps, (b - a) / fps) for a, b in true_runs(timeline.reason == code)],
             style.REASON_COLORS[name])
    bars("shots", [(s.start / fps, s.length / fps) for s in shots.shots],
         [style.SHOT_TONES[i % 2] for i in range(len(shots.shots))])
    bars("candidates", spans(shots.windows), style.CANDIDATE)
    bars("candidates", spans(shots.short_windows), style.SHORT_SHOT)
    kept = set(miniclips.kept)
    bars("redundancy", spans([w for i, w in enumerate(shots.windows) if i in kept]), style.KEPT)
    bars("redundancy", spans([w for i, w in enumerate(shots.windows) if i not in kept]), style.DISCARDED)
    bars("redundancy", spans(shots.short_windows), style.SHORT_SHOT)
    clip_spans = [(min(c.frame_indices) / fps, (max(c.frame_indices) + 1 - min(c.frame_indices)) / fps)
                  for c in miniclips.miniclips]
    bars("miniclips", clip_spans, style.MINICLIP)

    ax.set_yticks([y[r] for r in rows], rows)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, n / fps)
    ax.set_xlabel("time (s)")
    handles = _reason_handles(timeline.reason) + [
        Patch(facecolor=style.CANDIDATE, label="window"),
        Patch(facecolor=style.KEPT, label="kept window"),
        Patch(facecolor=style.DISCARDED, label="discarded window"),
        Patch(facecolor=style.SHORT_SHOT, label="short-shot candidate"),
        Patch(facecolor=style.MINICLIP, label="miniclip"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(handles),
              fontsize=8, frameon=False)
    ax_p.set_title(f"Timeline: {n} frames ({n / fps:.1f} s), {len(shots.shots)} shots, "
                   f"{len(shots.windows)} windows + {len(shots.short_windows)} short-shot candidates, "
                   f"{len(miniclips.miniclips)} miniclips (case {miniclips.case})")
    return fig
