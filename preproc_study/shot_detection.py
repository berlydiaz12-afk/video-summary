"""Shot boundary detection (TransNet V2) and candidate windows.

TransNet V2 runs once per kept segment through ``TransNetDetector.predict_segments``.
Shots and candidates come from ``preproc.shots``. A transition is a run of frames
with p > threshold inside a segment; those frames separate the shots.

There are two kinds of candidates, as in ``preproc.pipeline.process_video``:

* **windows**: consecutive, non-overlapping windows of L = T·r frames in every shot of
  at least L frames (``candidate_windows``). They go through redundancy removal.
* **short-shot candidates** (``summary.short_shot_windows``): one candidate of T
  consecutive frames (r = 1), centred in every shot of T to L − 1 frames
  (``short_shot_windows``). It is the only candidate of its shot, so it is always kept.

Candidate indices follow ``preproc``: windows first, then short-shot candidates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from preproc.config import SummaryConfig
from preproc.detector import ShotDetector
from preproc.shots import candidate_windows, probs_to_shots, short_shot_windows
from preproc.structs import Segment, Shot, Window, true_runs


@dataclass(frozen=True)
class Transition:
    """Run of frames ``[start, end)`` with p > threshold inside one segment."""

    index: int
    segment: int
    start: int
    end: int
    peak: float

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ShotDetectionAnalysis:
    threshold: float
    summary: SummaryConfig                 # T, r and short_shot_windows used to build the candidates
    segments: list[Segment]
    segment_probabilities: list[np.ndarray]
    probabilities: np.ndarray              # float32[n], NaN on removed frames
    shots: list[Shot]
    transitions: list[Transition]
    windows: list[Window]                  # windows of L frames (go through redundancy removal)
    short_windows: list[Window]            # short-shot candidates of T frames, r = 1 (always kept)

    @property
    def window_length(self) -> int:
        return self.summary.window_length

    @property
    def candidates(self) -> list[Window]:
        """All candidates with ``preproc``'s indexing: windows, then short-shot candidates."""
        return self.windows + self.short_windows

    def is_short(self, candidate: int) -> bool:
        return candidate >= len(self.windows)

    @property
    def useful_frames(self) -> int:
        """Frames inside shots (after frame filtering and transitions)."""
        return sum(s.length for s in self.shots)

    @property
    def transition_frames(self) -> int:
        return sum(s.length for s in self.segments) - self.useful_frames

    def windows_per_shot(self) -> np.ndarray:
        """Windows of L frames of every shot (short-shot candidates not included)."""
        counts = np.zeros(len(self.shots), dtype=np.int64)
        for w in self.windows:
            counts[w.shot] += 1
        return counts

    def short_candidate_shots(self) -> set[int]:
        return {w.shot for w in self.short_windows}

    def shots_without_candidate(self) -> int:
        """Shots too short to give any candidate (< T frames, or < L without short-shot candidates)."""
        with_candidate = {w.shot for w in self.candidates}
        return sum(1 for s in self.shots if s.index not in with_candidate)

    def counts(self) -> dict[str, int | float]:
        per_shot = self.windows_per_shot()
        return {
            "segments": len(self.segments),
            "transitions": len(self.transitions),
            "transition_frames": self.transition_frames,
            "shots": len(self.shots),
            "useful_frames": self.useful_frames,
            "shots_with_window": int((per_shot > 0).sum()),
            "shots_with_short_candidate": len(self.short_candidate_shots()),
            "shots_without_candidate": self.shots_without_candidate(),
            "windows": len(self.windows),
            "short_shot_candidates": len(self.short_windows),
            "candidates": len(self.candidates),
            "frames_in_candidates": sum(w.length for w in self.candidates),
        }


def predict_transitions(rgb48: np.ndarray, segments: Sequence[Segment], detector: ShotDetector) -> list[np.ndarray]:
    """TransNet V2 probability of every frame, one vector per segment."""
    if not segments:
        return []
    return detector.predict_segments([rgb48[s.start:s.end] for s in segments])


def build_shot_analysis(
    n_frames: int,
    segments: Sequence[Segment],
    segment_probabilities: Sequence[np.ndarray],
    threshold: float,
    summary: SummaryConfig,
) -> ShotDetectionAnalysis:
    """Shots, transitions and candidates from precomputed probabilities (no model call)."""
    shots = probs_to_shots(segments, segment_probabilities, threshold)
    windows = candidate_windows(shots, summary.window_length)
    short = (short_shot_windows(shots, summary.frames_per_clip, summary.window_length)
             if summary.short_shot_windows else [])
    probabilities = np.full(n_frames, np.nan, dtype=np.float32)
    transitions: list[Transition] = []
    for seg_idx, (seg, p) in enumerate(zip(segments, segment_probabilities)):
        p = np.asarray(p, dtype=np.float32)
        probabilities[seg.start:seg.end] = p
        for a, b in true_runs(p > threshold):
            transitions.append(Transition(len(transitions), seg_idx, seg.start + a, seg.start + b,
                                          float(p[a:b].max())))
    return ShotDetectionAnalysis(
        threshold=threshold,
        summary=summary,
        segments=list(segments),
        segment_probabilities=[np.asarray(p, dtype=np.float32) for p in segment_probabilities],
        probabilities=probabilities,
        shots=shots,
        transitions=transitions,
        windows=windows,
        short_windows=short,
    )


def sweep_stride(
    shots: Sequence[Shot],
    frames_per_clip: int,
    strides: Iterable[int],
    fps: int,
    short_shot_candidates: bool = True,
) -> list[dict[str, float | int]]:
    """For each stride r: windows, short-shot candidates, shots without candidate and covered useful time.

    ``covered_frames`` counts the frames spanned by the candidates (L per window, T per
    short-shot candidate), before redundancy removal.
    """
    useful = sum(s.length for s in shots)
    T = frames_per_clip
    rows: list[dict[str, float | int]] = []
    for r in strides:
        length = T * int(r)
        windows = candidate_windows(shots, length)
        short = short_shot_windows(shots, T, length) if short_shot_candidates else []
        per_shot = Counter(w.shot for w in windows)
        with_candidate = set(per_shot) | {w.shot for w in short}
        covered = len(windows) * length + len(short) * T
        rows.append({
            "stride": int(r),
            "window_length": length,
            "window_s": round(length / fps, 3),
            "shots": len(shots),
            "shots_with_window": len(per_shot),
            "windows": len(windows),
            "short_shot_candidates": len(short),
            "shots_without_candidate": len(shots) - len(with_candidate),
            "covered_frames": covered,
            "coverage": round(covered / useful, 4) if useful else 0.0,
        })
    return rows
