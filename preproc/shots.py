from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .structs import Segment, Shot, Window, true_runs


def probs_to_shots(segments: Sequence[Segment], probs: Sequence[np.ndarray], threshold: float) -> list[Shot]:
    """Convierte las probabilidades por segmento en tomas (en orden temporal)."""
    if len(segments) != len(probs):
        raise ValueError("debe haber un vector de probabilidades por segmento")
    shots: list[Shot] = []
    for seg_idx, (seg, p) in enumerate(zip(segments, probs)):
        p = np.asarray(p)
        if len(p) != seg.length:
            raise ValueError(f"el segmento {seg_idx} tiene {seg.length} cuadros y {len(p)} probabilidades")
        for a, b in true_runs(p <= threshold):
            shots.append(Shot(len(shots), seg_idx, seg.start + a, seg.start + b))
    return shots


def candidate_windows(shots: Iterable[Shot], window_length: int) -> list[Window]:
    """Ventanas consecutivas sin solapamiento desde el inicio de cada toma; el sobrante se descarta."""
    windows: list[Window] = []
    for shot in shots:
        for i in range(shot.length // window_length):
            windows.append(Window(shot.index, shot.start + i * window_length, window_length))
    return windows


def short_shot_windows(shots: Iterable[Shot], frames_per_clip: int, window_length: int) -> list[Window]:
    """Una candidata por toma de T a L−1 cuadros: T cuadros consecutivos (r = 1) centrados en la toma."""
    windows: list[Window] = []
    for shot in shots:
        if frames_per_clip <= shot.length < window_length:
            start = shot.start + (shot.length - frames_per_clip) // 2
            windows.append(Window(shot.index, start, frames_per_clip, stride=1))
    return windows
