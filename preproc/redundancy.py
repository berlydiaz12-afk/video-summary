from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import SummaryConfig
from .phash import hamming_array, phash_batch
from .structs import Window


@dataclass(frozen=True)
class Discarded:
    candidate: int   # ventana descartada (índice entre las candidatas)
    reference: int   # última ventana conservada de la misma toma
    distance: float


@dataclass
class RedundancyResult:
    kept: list[int] = field(default_factory=list)  # índices entre las candidatas, en orden temporal
    discarded: list[Discarded] = field(default_factory=list)


def signature_offsets(window_length: int, step: int) -> np.ndarray:
    return np.arange(0, window_length, step, dtype=np.int64)


def window_signatures(gray: np.ndarray, windows: Sequence[Window], window_length: int, step: int) -> np.ndarray:
    """Firmas ``uint64[n_ventanas, n_posiciones]``."""
    offsets = signature_offsets(window_length, step)
    if not windows:
        return np.empty((0, len(offsets)), dtype=np.uint64)
    starts = np.fromiter((w.start for w in windows), dtype=np.int64, count=len(windows))
    idx = starts[:, None] + offsets[None, :]
    return phash_batch(gray[idx.ravel()]).reshape(idx.shape)


def window_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(hamming_array(sig_a, sig_b).mean())


def filter_redundant(candidates: Sequence[Window], gray: np.ndarray, cfg: SummaryConfig) -> RedundancyResult:
    if cfg.hamming_threshold is None:
        raise ValueError("falta θ_H en la configuración")
    sigs = window_signatures(gray, candidates, cfg.window_length, cfg.phash_step)
    result = RedundancyResult()
    last_kept: dict[int, int] = {}
    for i, window in enumerate(candidates):
        ref = last_kept.get(window.shot)
        if ref is not None:
            d = window_distance(sigs[i], sigs[ref])
            if d <= cfg.hamming_threshold:
                result.discarded.append(Discarded(i, ref, d))
                continue
        result.kept.append(i)
        last_kept[window.shot] = i
    return result
