"""Diagnosis of the videos rejected with "ningún cuadro útil tras los pasos 2 y 3".

That error means no shot survived: frame filtering removed every frame, or it only left
frames that TransNet V2 marks as transitions (p > threshold). This module repeats both
procedures with the functions of ``preproc`` and reports which criterion removed the
frames, the distribution of each measure against its threshold, and how many frames a
looser threshold would keep. Nothing is written to disk.

Usage from Jupyter / Colab::

    import pandas as pd
    from preproc import load_config
    from rejected_diagnosis import diagnose_rejected_videos

    cfg = load_config("config.yaml")
    rows = diagnose_rejected_videos("D:/datos/pornography2k", "pornography2k", "salida", cfg, gpu=0)
    pd.DataFrame(rows).T
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from preproc.config import Config
from preproc.ffmpeg_io import FfmpegInfo, check_ffmpeg, decode_analysis, probe
from preproc.filtering import blank_mask, filter_frames, frame_differences, frame_stats, static_mask
from preproc.reporting import read_errors
from preproc.storage import OutputLayout

REJECTION_REASON = "ningún cuadro útil tras los pasos 2 y 3"
PERCENTILES = (5, 50, 95)


def diagnose_rejected_videos(
    input_dir: str | Path,
    dataset: str,
    output_dir: str | Path,
    cfg: Config,
    *,
    gpu: int = 0,
    luma_sweep: Sequence[float] = (4, 8, 12, 16),
    std_sweep: Sequence[float] = (1, 2, 3, 5),
    static_sweep: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> dict[str, dict[str, Any]]:
    """Diagnoses every video whose last error is ``REJECTION_REASON`` and still has no meta. Returns ``{id: row}``."""
    layout = OutputLayout(Path(output_dir).resolve() / dataset, dataset)
    ff = check_ffmpeg(cfg.ffmpeg.ffmpeg, cfg.ffmpeg.ffprobe)
    detector = None
    rows: dict[str, dict[str, Any]] = {}
    for vid, entry in sorted(read_errors(layout).items()):
        if REJECTION_REASON not in entry["motivo"] or layout.meta_path(vid).is_file():
            continue
        if detector is None:
            from preproc.transnet import TransNetDetector
            detector = TransNetDetector(cfg.transnet, device=f"cuda:{gpu}" if gpu >= 0 else "cpu")
        rows[vid] = diagnose_video(Path(input_dir) / entry["ruta"], cfg, ff, detector,
                                   luma_sweep, std_sweep, static_sweep)
    return rows


def diagnose_video(
    path: str | Path,
    cfg: Config,
    ff: FfmpegInfo,
    detector: Any,
    luma_sweep: Sequence[float],
    std_sweep: Sequence[float],
    static_sweep: Sequence[float],
) -> dict[str, Any]:
    """Frame filtering and shot detection of one video, measure by measure."""
    f = cfg.filter
    info = probe(path, ff)
    analysis = decode_analysis(path, ff, cfg, info)
    gray, rgb48 = analysis.gray, analysis.rgb48
    n = len(gray)
    mean, std = frame_stats(gray)
    diff = frame_differences(gray)[1:]  # d_0 = ∞

    result = filter_frames(gray, f)
    row: dict[str, Any] = {
        "fps_original": info.fps,
        "duracion_s": info.duration,
        "resolucion": f"{info.width}x{info.height}",
        "cuadros": n,
        # Frame filtering: share of frames under each threshold and removed by each mask.
        "pct_luma<θ_L": _pct(mean < f.luma_threshold),
        "pct_std<θ_σ": _pct(std < f.std_threshold),
        "pct_d<θ_E": _pct(diff < f.static_threshold),
        "pct_d==0": _pct(diff == 0),
        "pct_negro_uniforme": _pct(result.blank),
        "pct_pantalla_fija": _pct(result.static),
        "pct_eliminado": _pct(result.removed),
        **_percentiles("luma", mean),
        **_percentiles("std", std),
        **_percentiles("d", diff),
        "segmentos": len(result.segments),
        "cuadros_en_segmentos": result.n_kept,
    }

    # Shot detection on whatever frame filtering left.
    if result.segments:
        probs = detector.predict_segments([rgb48[s.start:s.end] for s in result.segments])
        p = np.concatenate(probs)
        row["pct_segmentos_p>θ"] = _pct(p > cfg.transnet.threshold)
        row["p_max"] = float(p.max())
        row["p_min"] = float(p.min())
    row["causa"] = _cause(result, row)

    # Frames that would survive frame filtering with a single threshold loosened.
    blank_ref = blank_mask(gray, f.luma_threshold, f.std_threshold)
    static_ref = static_mask(gray, f.static_threshold, f.static_min_frames)
    for t in luma_sweep:
        row[f"conserva_θ_L={t}"] = _pct(~(blank_mask(gray, t, f.std_threshold) | static_ref))
    for t in std_sweep:
        row[f"conserva_θ_σ={t}"] = _pct(~(blank_mask(gray, f.luma_threshold, t) | static_ref))
    for t in static_sweep:
        row[f"conserva_θ_E={t}"] = _pct(~(blank_ref | static_mask(gray, t, f.static_min_frames)))
    return row


def _cause(result: Any, row: dict[str, Any]) -> str:
    if not result.segments:
        if row["pct_negro_uniforme"] == 100.0:
            return "paso 2: todo negro/uniforme"
        if row["pct_pantalla_fija"] == 100.0:
            return "paso 2: todo pantalla fija"
        return "paso 2: negro/uniforme + pantalla fija cubren todo"
    return "paso 3: lo que dejó el paso 2 son transiciones (p > θ)"


def _pct(mask: np.ndarray) -> float:
    return round(100.0 * float(np.mean(mask)), 2) if np.size(mask) else 0.0


def _percentiles(name: str, values: np.ndarray) -> dict[str, float]:
    if not np.size(values):
        return {}
    return {f"{name}_p{q}": round(float(np.percentile(values, q)), 3) for q in PERCENTILES}
