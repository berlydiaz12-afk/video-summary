from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import SummaryConfig
from .structs import Miniclip, Shot, VideoFailed, Window

FALLBACK_NONE = "ninguno"
FALLBACK_EMPTY = "N=0"
KIND_WINDOW = "conservada"
KIND_SHORT = "toma_corta"
KIND_LONGEST = "toma_mas_larga"


@dataclass
class SelectionResult:
    miniclips: list[Miniclip]  # en orden temporal
    case: str                  # "ninguno" o "N=0"


def miniclip_indices(start: int, frames_per_clip: int, stride: int) -> np.ndarray:
    return start + stride * np.arange(frames_per_clip, dtype=np.int64)


def clip_motion(gray: np.ndarray, indices: Sequence[int]) -> float:
    frames = gray[np.asarray(indices, dtype=np.int64)].astype(np.int16)
    if len(frames) < 2:
        return 0.0
    return float(np.abs(np.diff(frames, axis=0)).mean())


def uniform_indices(start: int, length: int, frames_per_clip: int) -> np.ndarray:
    """T índices centrados y uniformes en ``[start, start + length)``; si length < T se repiten en orden."""
    i = np.arange(frames_per_clip, dtype=np.int64)
    return start + ((2 * i + 1) * length) // (2 * frames_per_clip)


def plan_miniclips(
    candidates: Sequence[Window],
    kept: Sequence[int],
    shots: Sequence[Shot],
    gray: np.ndarray,
    cfg: SummaryConfig,
) -> SelectionResult:
    """Convierte las candidatas conservadas en miniclips (índices de cuadro sobre la línea de 25 fps)."""
    T, r = cfg.frames_per_clip, cfg.stride
    clips: list[Miniclip] = []
    for i in sorted(kept, key=lambda j: candidates[j].start):
        w = candidates[i]
        idx = miniclip_indices(w.start, T, w.stride or r)
        kind = KIND_SHORT if w.stride == 1 and r != 1 else KIND_WINDOW
        clips.append(Miniclip(tuple(idx.tolist()), w.shot, kind, i, clip_motion(gray, idx), False))
    if clips:
        return SelectionResult(clips, FALLBACK_NONE)
    return SelectionResult([_longest_shot_clip(shots, gray, T)], FALLBACK_EMPTY)


def _longest_shot_clip(shots: Sequence[Shot], gray: np.ndarray, frames_per_clip: int) -> Miniclip:
    """N = 0: T cuadros espaciados uniformemente en la toma más larga (empate: la primera)."""
    if not shots:
        raise VideoFailed("ningún cuadro útil tras los pasos 2 y 3")
    longest = max(shots, key=lambda s: s.length)
    idx = uniform_indices(longest.start, longest.length, frames_per_clip)
    return Miniclip(tuple(idx.tolist()), longest.index, KIND_LONGEST, None, clip_motion(gray, idx), True)
