"""Blank and static frame filtering, analysed frame by frame.

The masks and segments come from ``preproc.filtering.filter_frames``, so they are
the bulk package's. This module only adds the per-frame series (mean luma, luma
standard deviation, mean absolute difference d_k) and splits the removed frames
into runs with the reason for each.

Reasons (a frame may meet more than one criterion):

* ``black``: mean luma < θ_L.
* ``uniform``: luma standard deviation < θ_σ (and not black).
* ``static``: inside a run of at least ``static_min_frames`` frames with d_k < θ_E.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from preproc.config import FilterConfig
from preproc.filtering import filter_frames, frame_differences, frame_stats
from preproc.structs import Segment, true_runs

# Reason codes, in priority order when a frame meets several criteria.
KEPT, BLACK, UNIFORM, STATIC = 0, 1, 2, 3
REASON_NAMES = {BLACK: "black", UNIFORM: "uniform", STATIC: "static"}


@dataclass(frozen=True)
class RemovedRun:
    """Run of consecutive removed frames ``[start, end)``."""

    index: int
    start: int
    end: int
    n_black: int
    n_uniform: int
    n_static: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def reason(self) -> str:
        """Criteria met inside the run, e.g. ``black`` or ``black+static``."""
        counts = (("black", self.n_black), ("uniform", self.n_uniform), ("static", self.n_static))
        return "+".join(name for name, count in counts if count > 0)


@dataclass
class FrameFilteringAnalysis:
    config: FilterConfig
    luma_mean: np.ndarray    # float64[n]
    luma_std: np.ndarray     # float64[n]
    frame_diff: np.ndarray   # float64[n], d_0 = inf
    blank: np.ndarray        # bool[n]: black or uniform (from preproc)
    static: np.ndarray       # bool[n]: inside a static screen (from preproc)
    segments: list[Segment]
    runs: list[RemovedRun] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.blank)

    @property
    def black(self) -> np.ndarray:
        return self.luma_mean < self.config.luma_threshold

    @property
    def uniform(self) -> np.ndarray:
        return (self.luma_std < self.config.std_threshold) & ~self.black

    @property
    def removed(self) -> np.ndarray:
        return self.blank | self.static

    @property
    def n_removed(self) -> int:
        return int(self.removed.sum())

    @property
    def n_kept(self) -> int:
        return self.n_frames - self.n_removed

    def reason_codes(self) -> np.ndarray:
        """int8[n]: 0 kept, 1 black, 2 uniform, 3 static (first criterion that applies)."""
        codes = np.zeros(self.n_frames, dtype=np.int8)
        codes[self.static] = STATIC
        codes[self.uniform] = UNIFORM
        codes[self.black] = BLACK
        return codes

    def counts(self) -> dict[str, int]:
        return {
            "frames": self.n_frames,
            "black": int(self.black.sum()),
            "uniform": int(self.uniform.sum()),
            "blank": int(self.blank.sum()),
            "static": int(self.static.sum()),
            "removed": self.n_removed,
            "kept": self.n_kept,
            "segments": len(self.segments),
            "removed_runs": len(self.runs),
        }


def analyze_frame_filtering(gray: np.ndarray, cfg: FilterConfig) -> FrameFilteringAnalysis:
    """Applies ``preproc`` frame filtering and keeps every intermediate series."""
    result = filter_frames(gray, cfg)
    mean, std = frame_stats(gray)
    analysis = FrameFilteringAnalysis(
        config=dataclasses.replace(cfg),
        luma_mean=mean,
        luma_std=std,
        frame_diff=frame_differences(gray),
        blank=result.blank,
        static=result.static,
        segments=result.segments,
    )
    black, uniform, static = analysis.black, analysis.uniform, analysis.static
    analysis.runs = [
        RemovedRun(i, a, b, int(black[a:b].sum()), int(uniform[a:b].sum()), int(static[a:b].sum()))
        for i, (a, b) in enumerate(true_runs(analysis.removed))
    ]
    return analysis
