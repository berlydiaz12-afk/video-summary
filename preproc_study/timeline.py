"""Per-frame status of the whole procedure (used by the annotated video and the timeline figure)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame_filtering import FrameFilteringAnalysis
from .miniclips import MiniclipAnalysis
from .redundancy_removal import RedundancyAnalysis
from .shot_detection import ShotDetectionAnalysis

# Candidate states. Every kept candidate is a miniclip, so "kept" already means "miniclip".
OUTSIDE, DISCARDED, KEPT, SHORT = 0, 1, 2, 3
STATE_NAMES = {DISCARDED: "discarded", KEPT: "kept", SHORT: "short shot"}


@dataclass
class FrameTimeline:
    fps: int
    threshold: float          # transition threshold
    reason: np.ndarray        # int8[n]: 0 kept, 1 black, 2 uniform, 3 static
    probability: np.ndarray   # float32[n]: TransNet V2 p (NaN on removed frames)
    shot: np.ndarray          # int32[n]: shot index (-1 outside shots)
    candidate: np.ndarray     # int32[n]: candidate index (-1 outside candidates)
    candidate_state: np.ndarray  # int8[n]: OUTSIDE, DISCARDED, KEPT or SHORT
    miniclip: np.ndarray      # int32[n]: miniclip order whose span covers the frame (-1 otherwise)
    sampled: np.ndarray       # bool[n]: one of the T frames given to Swin

    @property
    def n_frames(self) -> int:
        return len(self.reason)

    def is_transition(self, frame: int) -> bool:
        p = self.probability[frame]
        return bool(np.isfinite(p) and p > self.threshold)


def build_timeline(
    filtering: FrameFilteringAnalysis,
    shots: ShotDetectionAnalysis,
    redundancy: RedundancyAnalysis,
    miniclips: MiniclipAnalysis,
    fps: int,
) -> FrameTimeline:
    n = filtering.n_frames
    shot = np.full(n, -1, dtype=np.int32)
    for s in shots.shots:
        shot[s.start:s.end] = s.index

    candidate = np.full(n, -1, dtype=np.int32)
    state = np.zeros(n, dtype=np.int8)
    kept = set(redundancy.kept)
    for i, w in enumerate(shots.windows):
        candidate[w.start:w.end] = i
        state[w.start:w.end] = KEPT if i in kept else DISCARDED
    offset = len(shots.windows)
    for j, w in enumerate(shots.short_windows):
        candidate[w.start:w.end] = offset + j
        state[w.start:w.end] = SHORT

    miniclip = np.full(n, -1, dtype=np.int32)
    sampled = np.zeros(n, dtype=bool)
    for order, clip in enumerate(miniclips.miniclips):
        idx = np.asarray(clip.frame_indices, dtype=np.int64)
        miniclip[idx.min():idx.max() + 1] = order
        sampled[idx] = True

    return FrameTimeline(
        fps=fps,
        threshold=shots.threshold,
        reason=filtering.reason_codes(),
        probability=shots.probabilities,
        shot=shot,
        candidate=candidate,
        candidate_state=state,
        miniclip=miniclip,
        sampled=sampled,
    )
