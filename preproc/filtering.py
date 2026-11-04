from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FilterConfig
from .structs import Segment, true_runs

_CHUNK = 2048


@dataclass
class FilterResult:
    blank: np.ndarray    # bool[n]: negro o uniforme
    static: np.ndarray   # bool[n]: dentro de una pantalla fija
    segments: list[Segment]

    @property
    def removed(self) -> np.ndarray:
        return self.blank | self.static

    @property
    def n_blank(self) -> int:
        return int(self.blank.sum())

    @property
    def n_static(self) -> int:
        return int(self.static.sum())

    @property
    def n_removed(self) -> int:
        return int(self.removed.sum())

    @property
    def n_kept(self) -> int:
        return len(self.blank) - self.n_removed


def frame_stats(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviación estándar de intensidad por cuadro."""
    n = len(gray)
    mean = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)
    for a in range(0, n, _CHUNK):
        chunk = gray[a:a + _CHUNK]
        x = chunk.reshape(len(chunk), -1).astype(np.float64)
        mean[a:a + len(chunk)] = x.mean(axis=1)
        std[a:a + len(chunk)] = x.std(axis=1)
    return mean, std


def frame_differences(gray: np.ndarray) -> np.ndarray:
    """d_k = diferencia absoluta media entre el cuadro k y el k−1 (d_0 = ∞)."""
    n = len(gray)
    d = np.full(n, np.inf, dtype=np.float64)
    for a in range(1, n, _CHUNK):
        b = min(a + _CHUNK, n)
        x = gray[a - 1:b].astype(np.int16)
        d[a:b] = np.abs(np.diff(x, axis=0)).reshape(b - a, -1).mean(axis=1)
    return d


def blank_mask(gray: np.ndarray, luma_threshold: float, std_threshold: float) -> np.ndarray:
    mean, std = frame_stats(gray)
    return (mean < luma_threshold) | (std < std_threshold)


def static_mask(gray: np.ndarray, diff_threshold: float, min_frames: int) -> np.ndarray:
    mask = np.zeros(len(gray), dtype=bool)
    below = frame_differences(gray) < diff_threshold
    for a, b in true_runs(below):        # d_k < θ_E para k en [a, b) → cuadros [a−1, b)
        if b - (a - 1) >= min_frames:
            mask[a - 1:b] = True
    return mask


def mask_to_segments(removed: np.ndarray) -> list[Segment]:
    return [Segment(a, b) for a, b in true_runs(~np.asarray(removed, dtype=bool))]


def filter_frames(gray: np.ndarray, cfg: FilterConfig) -> FilterResult:
    """Aplica el paso 2 a los cuadros de análisis en gris ``(n, H, W)``."""
    if cfg.luma_threshold is None or cfg.std_threshold is None or cfg.static_threshold is None:
        raise ValueError("faltan θ_L, θ_σ o θ_E en la configuración")
    blank = blank_mask(gray, cfg.luma_threshold, cfg.std_threshold)
    static = static_mask(gray, cfg.static_threshold, cfg.static_min_frames)
    return FilterResult(blank, static, mask_to_segments(blank | static))
