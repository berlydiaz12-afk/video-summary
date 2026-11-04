"""Writes the data (CSV/JSON) and images (PNG/MP4) of each procedure.

Folder layout inside ``out``::

    01_decoding/          metadata.json, frame_mapping.csv
    02_frame_filtering/   frames.csv, series.png, removed_runs.csv, removed_runs/, segments.csv
    03_shot_detection/    probabilities.csv/.png, shots.csv, transitions.csv, transitions/,
                          shot_lengths.png, stride_sweep.csv, candidates.csv
    04_redundancy_removal/ decisions.csv, signatures.csv, matrices/, discarded/,
                          hamming_threshold_sweep.csv/.png
    05_miniclips/         miniclips.csv, miniclips.json
    06_swin_input/        miniclip_<k>.mp4, miniclip_<k>.png, tensors.json

When a video has more removed runs, transitions, discarded windows or shots than
``max_images``, images are drawn for an evenly spaced subset; the CSV files always
list everything.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

import numpy as np

from preproc.config import Config
from preproc.ffmpeg_io import FfmpegInfo

from . import plots, style
from .decoding import DecodedVideo
from .drawing import (
    Thumbnails,
    discarded_window_frames,
    discarded_window_strip,
    removed_run_frames,
    removed_run_strip,
    transition_frames,
    transition_strip,
    write_png,
)
from .frame_filtering import REASON_NAMES, FrameFilteringAnalysis
from .miniclips import MiniclipAnalysis
from .redundancy_removal import RedundancyAnalysis
from .shot_detection import ShotDetectionAnalysis
from .swin_input import SwinInputAnalysis, process_swin_input

DECODING = "01_decoding"
FRAME_FILTERING = "02_frame_filtering"
SHOT_DETECTION = "03_shot_detection"
REDUNDANCY_REMOVAL = "04_redundancy_removal"
MINICLIPS = "05_miniclips"
SWIN_INPUT = "06_swin_input"

T = TypeVar("T")


# --------------------------------------------------------------------------- helpers

def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([_cell(v) for v in row] for row in rows)
    return path


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6) if np.isfinite(value) else ""
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def pick_evenly(items: Sequence[T], limit: int | None) -> list[T]:
    """All items, or ``limit`` items evenly spaced along the video."""
    if limit is None or len(items) <= limit:
        return list(items)
    if limit <= 0:
        return []
    idx = np.unique(np.linspace(0, len(items) - 1, limit).round().astype(int))
    return [items[i] for i in idx]


def shots_for_matrices(redundancy: RedundancyAnalysis, limit: int | None) -> list[int]:
    """Shots with at least two windows, those with the most windows first."""
    counts: dict[int, int] = {}
    for w in redundancy.windows:
        counts[w.shot] = counts.get(w.shot, 0) + 1
    shots = sorted((s for s, c in counts.items() if c >= 2), key=lambda s: (-counts[s], s))
    return shots if limit is None else shots[:limit]


def required_thumbnails(
    filtering: FrameFilteringAnalysis,
    shots: ShotDetectionAnalysis,
    redundancy: RedundancyAnalysis,
    max_images: int | None,
) -> set[int]:
    """Frames the strips will show (same subsets as the export functions)."""
    frames: set[int] = set()
    for run in pick_evenly(filtering.runs, max_images):
        frames.update(n for _, n in removed_run_frames(run))
    for tr in pick_evenly(shots.transitions, max_images):
        frames.update(n for _, n in transition_frames(tr))
    for decision in pick_evenly(redundancy.discarded, max_images):
        ref, win = discarded_window_frames(decision, redundancy)
        frames.update(ref)
        frames.update(win)
    return {n for n in frames if 0 <= n < filtering.n_frames}


# --------------------------------------------------------------------------- decoding

def export_decoding(out: Path, decoded: DecodedVideo) -> dict[str, Path]:
    folder = out / DECODING
    files = {"metadata": write_json(folder / "metadata.json", decoded.metadata())}
    n = decoded.n_frames
    mapping = decoded.mapping
    rows = (
        (i, i / decoded.fps,
         int(mapping.source_index[i]) if mapping is not None else None,
         float(mapping.source_time[i]) if mapping is not None else None)
        for i in range(n)
    )
    files["frame_mapping"] = write_csv(folder / "frame_mapping.csv",
                                       ["frame", "time_s", "source_frame", "source_time_s"], rows)
    return files


# --------------------------------------------------------------------------- frame filtering

def export_frame_filtering(
    out: Path,
    filtering: FrameFilteringAnalysis,
    thumbs: Thumbnails,
    thumb_size: tuple[int, int],
    fps: int,
    max_images: int | None,
) -> dict[str, Path]:
    folder = out / FRAME_FILTERING
    codes = filtering.reason_codes()
    black, uniform, static, removed = filtering.black, filtering.uniform, filtering.static, filtering.removed
    rows = (
        (i, i / fps, filtering.luma_mean[i], filtering.luma_std[i], filtering.frame_diff[i],
         black[i], uniform[i], static[i], removed[i], REASON_NAMES.get(int(codes[i]), ""))
        for i in range(filtering.n_frames)
    )
    files = {
        "frames": write_csv(folder / "frames.csv",
                            ["frame", "time_s", "luma_mean", "luma_std", "frame_diff",
                             "black", "uniform", "static", "removed", "reason"], rows),
    }
    fig = plots.plot_frame_filtering(filtering, fps)
    style.save_figure(fig, folder / "series.png")
    files["series"] = folder / "series.png"

    files["removed_runs"] = write_csv(
        folder / "removed_runs.csv",
        ["run", "start", "end", "frames", "start_s", "duration_s", "reason",
         "black_frames", "uniform_frames", "static_frames"],
        ((r.index, r.start, r.end, r.length, r.start / fps, r.length / fps, r.reason,
          r.n_black, r.n_uniform, r.n_static) for r in filtering.runs),
    )
    for run in pick_evenly(filtering.runs, max_images):
        write_png(folder / "removed_runs" / f"run_{run.index:04d}.png",
                  removed_run_strip(run, filtering, thumbs, thumb_size, fps))
    files["segments"] = write_csv(
        folder / "segments.csv", ["segment", "start", "end", "frames", "start_s", "duration_s"],
        ((i, s.start, s.end, s.length, s.start / fps, s.length / fps) for i, s in enumerate(filtering.segments)),
    )
    return files


# --------------------------------------------------------------------------- shot detection

def export_shot_detection(
    out: Path,
    shots: ShotDetectionAnalysis,
    thumbs: Thumbnails,
    thumb_size: tuple[int, int],
    fps: int,
    stride_rows: Sequence[dict[str, Any]],
    max_images: int | None,
) -> dict[str, Path]:
    folder = out / SHOT_DETECTION
    segment_of = np.full(len(shots.probabilities), -1, dtype=np.int64)
    for i, s in enumerate(shots.segments):
        segment_of[s.start:s.end] = i
    p = shots.probabilities
    rows = (
        (i, i / fps, p[i], segment_of[i] if segment_of[i] >= 0 else None,
         bool(np.isfinite(p[i]) and p[i] > shots.threshold))
        for i in range(len(p))
    )
    files = {"probabilities": write_csv(folder / "probabilities.csv",
                                        ["frame", "time_s", "probability", "segment", "transition"], rows)}
    style.save_figure(plots.plot_transition_probabilities(shots, fps), folder / "probabilities.png")
    files["probabilities_plot"] = folder / "probabilities.png"

    per_shot = shots.windows_per_shot()
    short_shots = shots.short_candidate_shots()
    files["shots"] = write_csv(
        folder / "shots.csv",
        ["shot", "segment", "start", "end", "frames", "start_s", "duration_s", "windows", "short_shot_candidate"],
        ((s.index, s.segment, s.start, s.end, s.length, s.start / fps, s.length / fps, per_shot[s.index],
          s.index in short_shots) for s in shots.shots),
    )
    files["transitions"] = write_csv(
        folder / "transitions.csv", ["transition", "segment", "start", "end", "frames", "start_s", "peak"],
        ((t.index, t.segment, t.start, t.end, t.length, t.start / fps, t.peak) for t in shots.transitions),
    )
    for tr in pick_evenly(shots.transitions, max_images):
        write_png(folder / "transitions" / f"transition_{tr.index:04d}.png",
                  transition_strip(tr, shots, thumbs, thumb_size, fps))
    style.save_figure(plots.plot_shot_lengths(shots, fps), folder / "shot_lengths.png")
    files["shot_lengths"] = folder / "shot_lengths.png"
    if stride_rows:
        files["stride_sweep"] = write_csv(folder / "stride_sweep.csv", list(stride_rows[0]),
                                          (list(r.values()) for r in stride_rows))
    stride = shots.summary.stride
    files["candidates"] = write_csv(
        folder / "candidates.csv",
        ["candidate", "kind", "shot", "start", "end", "frames", "stride", "start_s", "end_s"],
        ((i, "short_shot" if shots.is_short(i) else "window", w.shot, w.start, w.end, w.length,
          w.stride or stride, w.start / fps, w.end / fps)
         for i, w in enumerate(shots.candidates)),
    )
    return files


# --------------------------------------------------------------------------- redundancy removal

def export_redundancy_removal(
    out: Path,
    redundancy: RedundancyAnalysis,
    thumbs: Thumbnails,
    thumb_size: tuple[int, int],
    fps: int,
    threshold_rows: Sequence[dict[str, Any]],
    max_images: int | None,
) -> dict[str, Path]:
    folder = out / REDUNDANCY_REMOVAL
    files = {
        "decisions": write_csv(
            folder / "decisions.csv",
            ["window", "shot", "start", "start_s", "reference", "reference_distance", "decision",
             "previous", "previous_distance"],
            ((d.window, d.shot, d.start, d.start / fps, d.reference, d.reference_distance,
              "kept" if d.kept else "discarded", d.previous, d.previous_distance)
             for d in redundancy.decisions),
        ),
    }
    offsets = [int(o) for o in redundancy.offsets]
    files["signatures"] = write_csv(
        folder / "signatures.csv",
        ["window", "shot", "start", *[f"phash_+{o}" for o in offsets]],
        ((i, w.shot, w.start, *[f"{int(h):016x}" for h in redundancy.signatures[i]])
         for i, w in enumerate(redundancy.windows)),
    )
    for shot in shots_for_matrices(redundancy, max_images):
        style.save_figure(plots.plot_distance_matrix(redundancy, shot), folder / "matrices" / f"shot_{shot:04d}.png")
    for decision in pick_evenly(redundancy.discarded, max_images):
        write_png(folder / "discarded" / f"window_{decision.window:04d}.png",
                  discarded_window_strip(decision, redundancy, thumbs, thumb_size, fps))
    if threshold_rows:
        files["hamming_threshold_sweep"] = write_csv(folder / "hamming_threshold_sweep.csv",
                                                     list(threshold_rows[0]),
                                                     (list(r.values()) for r in threshold_rows))
        style.save_figure(plots.plot_hamming_threshold_sweep(threshold_rows, redundancy.threshold),
                          folder / "hamming_threshold_sweep.png")
        files["hamming_threshold_sweep_plot"] = folder / "hamming_threshold_sweep.png"
    return files


# --------------------------------------------------------------------------- miniclips

def export_miniclips(out: Path, miniclips: MiniclipAnalysis) -> dict[str, Path]:
    """One row per miniclip and a JSON with the counts and the T frame indices of each miniclip."""
    folder = out / MINICLIPS
    rows = miniclips.rows()
    header = ["order", "kind", "candidate", "shot", "start_frame", "start_s", "stride", "span_s",
              "playback_fps", "motion", "fallback"]
    files = {"miniclips": write_csv(folder / "miniclips.csv", header, ([r[h] for h in header] for r in rows))}
    data = {
        "rule": "every kept candidate is a miniclip, in temporal order; N = 0: one miniclip of the longest shot",
        **miniclips.counts(),
        "miniclip_list": [{**r, "frames": list(c.frame_indices)} for r, c in zip(rows, miniclips.miniclips)],
    }
    files["miniclips_json"] = write_json(folder / "miniclips.json", data)
    return files


# --------------------------------------------------------------------------- Swin input

def export_swin_input(
    out: Path,
    decoded: DecodedVideo,
    miniclips: MiniclipAnalysis,
    cfg: Config,
    ff: FfmpegInfo,
    *,
    sheets: bool = True,
    measure_storage: bool = True,
    threads: int = 0,
) -> tuple[SwinInputAnalysis, list[str]]:
    """Stores every miniclip as ``preproc`` does (one .mp4 each), its contact sheet and ``tensors.json``.

    Works miniclip by miniclip (see :func:`swin_input.process_swin_input`). Returns the
    analysis and the warnings.
    """
    folder = out / SWIN_INPUT
    analysis = process_swin_input(decoded, miniclips.miniclips, cfg, ff, folder=folder, sheets=sheets,
                                  measure_storage=measure_storage, threads=threads)
    warnings: list[str] = []
    if analysis.alignment_error > cfg.analysis.alignment_tolerance:
        warnings.append(f"possible misalignment between passes: error {analysis.alignment_error:.2f} > "
                        f"tolerance {cfg.analysis.alignment_tolerance:.2f}")
    data = analysis.summary()
    data["miniclip_files"] = [
        {"order": m.order, "file": m.path.name if m.path is not None else None, "playback_fps": m.playback_fps,
         "bytes": m.bytes, "psnr_db": m.psnr if m.psnr is None or np.isfinite(m.psnr) else "inf"}
        for m in analysis.miniclips
    ]
    write_json(folder / "tensors.json", data)
    return analysis, warnings
