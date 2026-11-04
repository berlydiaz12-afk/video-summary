"""Miniclip planning: every kept candidate becomes a miniclip.

The miniclips come from ``preproc.selection.plan_miniclips``, the same rule as the
bulk package (there is no motion-guided selection and no fixed or variable K):

* every kept window (r = 2 with the default configuration) and every short-shot
  candidate (r = 1) is a miniclip, in temporal order;
* with no candidate at all (case ``N=0``), one miniclip of T frames evenly spaced
  over the longest shot (on a tie, the first one).

This module adds what the procedure leaves implicit: the stride r and the playback
rate of each miniclip, the time it spans, the counts by kind and the motion m_j
(recorded only as a descriptor, it no longer decides anything).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from preproc.config import SummaryConfig
from preproc.pipeline import clip_stride
from preproc.selection import FALLBACK_EMPTY, KIND_LONGEST, KIND_SHORT, KIND_WINDOW, SelectionResult, plan_miniclips
from preproc.structs import Miniclip, Shot, Window


@dataclass
class MiniclipAnalysis:
    summary: SummaryConfig
    fps: int
    result: SelectionResult    # from preproc
    candidates: list[Window]   # windows, then short-shot candidates (preproc's indexing)
    kept: list[int]            # kept candidates: kept windows, then every short-shot candidate
    useful_frames: int

    @property
    def miniclips(self) -> list[Miniclip]:
        return self.result.miniclips

    @property
    def case(self) -> str:
        """``ninguno`` (miniclips from the candidates) or ``N=0`` (longest-shot fallback)."""
        return self.result.case

    @property
    def is_fallback(self) -> bool:
        return self.case == FALLBACK_EMPTY

    def stride(self, k: int) -> int | None:
        """Constant spacing r of miniclip ``k`` (None for the longest-shot fallback)."""
        return clip_stride(self.miniclips[k])

    def playback_fps(self, k: int) -> float:
        """Rate at which ``preproc`` stores miniclip ``k``: fps / r (the real duration of the frames)."""
        return self.fps / (self.stride(k) or self.summary.stride)

    def span_frames(self, k: int) -> int:
        """Frames of the 25 fps timeline covered by miniclip ``k``, first to last sampled frame."""
        idx = self.miniclips[k].frame_indices
        return max(idx) - min(idx) + 1

    @property
    def motion(self) -> np.ndarray:
        """m_j of every miniclip (descriptor only)."""
        return np.asarray([c.motion for c in self.miniclips], dtype=np.float64)

    def counts(self) -> dict[str, Any]:
        kinds = Counter(c.kind for c in self.miniclips)
        strides = Counter(self.stride(k) for k in range(len(self.miniclips)))
        span = sum(self.span_frames(k) for k in range(len(self.miniclips)))
        return {
            "candidates": len(self.candidates),
            "kept_candidates": len(self.kept),
            "miniclips": len(self.miniclips),
            "from_windows": kinds.get(KIND_WINDOW, 0),
            "from_short_shots": kinds.get(KIND_SHORT, 0),
            "from_longest_shot": kinds.get(KIND_LONGEST, 0),
            "by_stride": {("none" if r is None else str(r)): n for r, n in sorted(strides.items(), key=_stride_key)},
            "frames_per_miniclip": self.summary.frames_per_clip,
            "frames_given_to_swin": len(self.miniclips) * self.summary.frames_per_clip,
            "unique_frames": len({i for c in self.miniclips for i in c.frame_indices}),
            "spanned_s": round(span / self.fps, 3),
            "useful_duration_s": round(self.useful_frames / self.fps, 3),
            "fallback_case": self.case,
        }

    def rows(self) -> list[dict[str, Any]]:
        """One row per miniclip (same fields as the ``miniclips`` list of ``preproc``'s JSON, plus rates)."""
        fps = self.fps
        return [
            {
                "order": k,
                "kind": c.kind,
                "candidate": c.candidate_index,
                "shot": c.shot,
                "start_frame": c.start,
                "start_s": round(c.start / fps, 3),
                "stride": self.stride(k),
                "span_s": round(self.span_frames(k) / fps, 3),
                "playback_fps": round(self.playback_fps(k), 4),
                "motion": c.motion,
                "fallback": c.fallback,
            }
            for k, c in enumerate(self.miniclips)
        ]


def kept_candidates(n_windows: int, kept_windows: Sequence[int], n_short: int) -> list[int]:
    """Kept candidate indices as ``preproc`` builds them: kept windows, then every short-shot candidate."""
    return list(kept_windows) + list(range(n_windows, n_windows + n_short))


def analyze_miniclips(
    gray: np.ndarray,
    shots: Sequence[Shot],
    candidates: Sequence[Window],
    kept: Sequence[int],
    summary: SummaryConfig,
    fps: int,
) -> MiniclipAnalysis:
    """Applies ``preproc`` miniclip planning (raises ``VideoFailed`` when there are no shots)."""
    result = plan_miniclips(candidates, kept, shots, gray, summary)
    return MiniclipAnalysis(summary, fps, result, list(candidates), list(kept), sum(s.length for s in shots))


def _stride_key(item: tuple[int | None, int]) -> tuple[int, int]:
    r = item[0]
    return (1, 0) if r is None else (0, r)
