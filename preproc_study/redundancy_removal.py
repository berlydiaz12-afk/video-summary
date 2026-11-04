"""Redundancy removal by perceptual hashing (pHash), with every decision explained.

Kept and discarded windows come from ``preproc.redundancy.filter_redundant``. This
module adds, for each window, its reference (the last kept window of the same shot),
the distance to it, the distance to the previous window of the same shot (kept or
not) and the full distance matrix of each shot.

Only the windows of L frames go through this procedure. Short-shot candidates are
the only candidate of their shot, so ``preproc`` keeps them without comparing them.

Every kept candidate becomes a miniclip, so θ_H is the only parameter that controls
how many miniclips a video gives: :func:`sweep_hamming_threshold` reports that number.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from preproc.config import SummaryConfig
from preproc.phash import hamming_array
from preproc.redundancy import RedundancyResult, filter_redundant, signature_offsets, window_distance, window_signatures
from preproc.selection import FALLBACK_EMPTY, FALLBACK_NONE
from preproc.structs import Window


@dataclass(frozen=True)
class WindowDecision:
    window: int                       # index among the candidate windows
    shot: int
    start: int
    reference: int | None             # last kept window of the same shot (None for the first)
    reference_distance: float | None  # mean Hamming distance to the reference
    previous: int | None              # previous candidate of the same shot
    previous_distance: float | None
    kept: bool


@dataclass
class RedundancyAnalysis:
    summary: SummaryConfig
    windows: list[Window]
    offsets: np.ndarray        # signature positions inside a window (0, 25, 50, …)
    signatures: np.ndarray     # uint64[n_windows, n_positions]
    result: RedundancyResult   # from preproc
    decisions: list[WindowDecision]

    @property
    def threshold(self) -> float:
        return float(self.summary.hamming_threshold)

    @property
    def kept(self) -> list[int]:
        return self.result.kept

    @property
    def discarded(self) -> list[WindowDecision]:
        return [d for d in self.decisions if not d.kept]

    def position_distances(self, a: int, b: int) -> np.ndarray:
        """Hamming distance between windows ``a`` and ``b`` at each signature position."""
        return hamming_array(self.signatures[a], self.signatures[b])

    def shot_windows(self, shot: int) -> list[int]:
        return [i for i, w in enumerate(self.windows) if w.shot == shot]

    def shot_distance_matrix(self, shot: int) -> tuple[list[int], np.ndarray]:
        """Candidate windows of a shot and their pairwise mean Hamming distances."""
        ids = self.shot_windows(shot)
        sig = self.signatures[ids]
        matrix = hamming_array(sig[:, None, :], sig[None, :, :]).mean(axis=-1)
        return ids, matrix

    def counts(self) -> dict[str, int]:
        return {
            "candidate_windows": len(self.windows),
            "kept_windows": len(self.kept),
            "discarded_windows": len(self.windows) - len(self.kept),
        }


def analyze_redundancy(gray: np.ndarray, windows: Sequence[Window], summary: SummaryConfig) -> RedundancyAnalysis:
    """Applies ``preproc`` redundancy removal and records every comparison."""
    result = filter_redundant(windows, gray, summary)
    length, step = summary.window_length, summary.phash_step
    signatures = window_signatures(gray, windows, length, step)
    kept = set(result.kept)
    last_kept: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    decisions: list[WindowDecision] = []
    for i, w in enumerate(windows):
        ref, prev = last_kept.get(w.shot), last_seen.get(w.shot)
        d_ref = window_distance(signatures[i], signatures[ref]) if ref is not None else None
        d_prev = window_distance(signatures[i], signatures[prev]) if prev is not None else None
        decisions.append(WindowDecision(i, w.shot, w.start, ref, d_ref, prev, d_prev, i in kept))
        if i in kept:
            last_kept[w.shot] = i
        last_seen[w.shot] = i
    return RedundancyAnalysis(
        summary=summary,
        windows=list(windows),
        offsets=signature_offsets(length, step),
        signatures=signatures,
        result=result,
        decisions=decisions,
    )


def sweep_hamming_threshold(
    gray: np.ndarray,
    windows: Sequence[Window],
    summary: SummaryConfig,
    thresholds: Iterable[float],
    *,
    short_candidates: int = 0,
    fps: int = 25,
) -> list[dict[str, float | int | str]]:
    """Windows kept and discarded, and resulting miniclips, for each θ_H (same rule as ``preproc``).

    ``short_candidates`` short-shot candidates (always kept) are added to the miniclips.
    With no candidate at all, ``preproc`` gives one miniclip of the longest shot (case ``N=0``).
    ``miniclip_s`` is the video time spanned by the miniclips (L / fps each window,
    T / fps each short-shot candidate).
    """
    T, L = summary.frames_per_clip, summary.window_length
    rows: list[dict[str, float | int | str]] = []
    for theta in thresholds:
        res = filter_redundant(windows, gray, dataclasses.replace(summary, hamming_threshold=float(theta)))
        n = len(windows)
        miniclips = len(res.kept) + short_candidates
        case = FALLBACK_NONE if miniclips else FALLBACK_EMPTY
        rows.append({
            "hamming_threshold": float(theta),
            "windows": n,
            "kept_windows": len(res.kept),
            "discarded_windows": len(res.discarded),
            "discarded_fraction": round(len(res.discarded) / n, 4) if n else 0.0,
            "short_shot_candidates": short_candidates,
            "miniclips": miniclips if miniclips else 1,
            "miniclip_s": round((len(res.kept) * L + short_candidates * T) / fps, 3),
            "fallback_case": case,
        })
    return rows
