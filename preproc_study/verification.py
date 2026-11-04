"""Consistency check against the bulk package (``preproc.pipeline.process_video``).

Both run the same decoding and the same functions, so every row should match. A
mismatch points to a configuration difference or to non-determinism (for example,
TransNet V2 probabilities very close to the threshold on a different device).

Pixels are compared miniclip by miniclip through ``process_video``'s ``clip_sink``
and a SHA-1 of the extracted frames, so neither side keeps every miniclip in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from preproc.config import Config
from preproc.detector import ShotDetector
from preproc.ffmpeg_io import FfmpegInfo
from preproc.pipeline import clip_stride, process_video
from preproc.structs import Miniclip

from .decoding import DecodedVideo
from .frame_filtering import FrameFilteringAnalysis
from .miniclips import MiniclipAnalysis
from .redundancy_removal import RedundancyAnalysis
from .shot_detection import ShotDetectionAnalysis
from .swin_input import SwinInputAnalysis, miniclip_digest


def verify_against_preproc(
    video: str | Path,
    cfg: Config,
    detector: ShotDetector,
    ff: FfmpegInfo,
    decoded: DecodedVideo,
    filtering: FrameFilteringAnalysis,
    shots: ShotDetectionAnalysis,
    redundancy: RedundancyAnalysis,
    miniclips: MiniclipAnalysis,
    swin_input: SwinInputAnalysis | None = None,
    threads: int = 0,
) -> list[dict[str, Any]]:
    """Runs ``process_video`` on the same video and compares its record with this study."""
    video = Path(video)
    digests: list[str] = []

    def sink(k: int, frames: np.ndarray, clip: Miniclip) -> None:
        digests.append(miniclip_digest(frames))

    _, meta = process_video(video, video.name, video.stem, cfg, detector, ff, threads=threads, clip_sink=sink)
    frame_counts, counts = meta["cuadros"], meta["conteos"]
    clips = miniclips.miniclips
    shot_counts = shots.counts()
    rows: list[tuple[str, Any, Any]] = [
        ("frames at fps", decoded.n_frames, frame_counts["paso1"]),
        ("black or uniform frames", int(filtering.blank.sum()), frame_counts["negros_uniformes"]),
        ("static frames", int(filtering.static.sum()), frame_counts["pantalla_fija"]),
        ("removed frames", filtering.n_removed, frame_counts["eliminados_paso2"]),
        ("segments", len(filtering.segments), counts["segmentos"]),
        ("shots", len(shots.shots), counts["tomas"]),
        ("frames inside shots", shots.useful_frames, frame_counts["paso3"]),
        ("shots with a window", shot_counts["shots_with_window"], counts["tomas_con_ventana"]),
        ("short-shot candidates", len(shots.short_windows), counts["tomas_cortas_candidatas"]),
        ("shots without candidate", shot_counts["shots_without_candidate"], counts["tomas_sin_candidata"]),
        ("candidates", len(shots.candidates), counts["ventanas_candidatas"]),
        ("kept candidates", len(miniclips.kept), counts["ventanas_conservadas"]),
        ("discarded windows", len(redundancy.discarded), counts["ventanas_descartadas"]),
        ("frames in candidates", shot_counts["frames_in_candidates"], frame_counts["en_ventanas"]),
        ("miniclips", len(clips), meta["seleccion"]["miniclips"]),
        ("fallback case", miniclips.case, meta["respaldo"]["caso"]),
        ("miniclip kinds", [c.kind for c in clips], [m["tipo"] for m in meta["miniclips"]]),
        ("miniclip start frames", [c.start for c in clips], [m["cuadro_inicial"] for m in meta["miniclips"]]),
        ("miniclip strides", [clip_stride(c) for c in clips], [m["r"] for m in meta["miniclips"]]),
        ("miniclip frame indices", [list(c.frame_indices) for c in clips], [m["cuadros"] for m in meta["miniclips"]]),
    ]
    result = [{"item": item, "study": ours, "preproc": theirs, "match": ours == theirs}
              for item, ours, theirs in rows]
    if swin_input is not None:
        ours = swin_input.digests
        same = sum(a == b for a, b in zip(ours, digests))
        result.append({"item": "miniclip pixels (SHA-1 per miniclip, before encoding)",
                       "study": f"{len(ours)} miniclips", "preproc": f"{len(digests)} miniclips, {same} identical",
                       "match": len(ours) == len(digests) == same})
    return result


def verification_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Verification against preproc",
        "",
        "Same video and configuration processed by `preproc.pipeline.process_video`.",
        "",
        "| Item | Study | preproc | Match |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['item']} | {_short(r['study'])} | {_short(r['preproc'])} | "
                     f"{'yes' if r['match'] else '**no**'} |")
    ok = all(r["match"] for r in rows)
    lines += ["", "All items match." if ok else "**Some items differ.** See the rows marked **no**."]
    return "\n".join(lines) + "\n"


def _short(value: Any, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."
