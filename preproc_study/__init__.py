"""Study of the video summarization procedure (Ch. 4) on a single video, for notebooks.

Reuses ``preproc`` for every decision (decoding, frame filtering, TransNet V2,
shots and candidates, redundancy removal, miniclip planning, extraction and storage)
and exposes every intermediate result: per-frame series, removed runs, transitions,
window decisions, the θ_H sweep and the miniclips given to Swin. ffmpeg reads and
writes video; OpenCV draws the images.

Public API:

* ``VideoStudy``: one procedure per method, results kept as attributes; ``run`` writes everything.
* ``study_video``: one-call study.
* Procedures without the class: ``decode_video``, ``analyze_frame_filtering``,
  ``predict_transitions`` + ``build_shot_analysis``, ``analyze_redundancy``,
  ``analyze_miniclips``, ``process_swin_input``, ``read_miniclip``, ``build_timeline``.
* Sweeps: ``sweep_stride``, ``sweep_hamming_threshold``.
* Figures (``plots``) and images (``drawing``) return objects for inline display.
"""

from . import drawing, plots
from ._version import __version__
from .decoding import DecodedVideo, FrameMapping, decode_video, estimate_frame_mapping
from .frame_filtering import FrameFilteringAnalysis, RemovedRun, analyze_frame_filtering
from .frame_rendering import StreamingEncoder, render_pass
from .miniclips import MiniclipAnalysis, analyze_miniclips, kept_candidates
from .redundancy_removal import RedundancyAnalysis, WindowDecision, analyze_redundancy, sweep_hamming_threshold
from .shot_detection import ShotDetectionAnalysis, Transition, build_shot_analysis, predict_transitions, sweep_stride
from .study import VideoStudy, study_video
from .swin_input import (
    StoredMiniclip,
    SwinInputAnalysis,
    TensorStatistics,
    process_swin_input,
    read_miniclip,
)
from .timeline import FrameTimeline, build_timeline

__all__ = [
    "__version__",
    "DecodedVideo",
    "FrameFilteringAnalysis",
    "FrameMapping",
    "FrameTimeline",
    "MiniclipAnalysis",
    "RedundancyAnalysis",
    "RemovedRun",
    "ShotDetectionAnalysis",
    "StoredMiniclip",
    "StreamingEncoder",
    "SwinInputAnalysis",
    "TensorStatistics",
    "Transition",
    "VideoStudy",
    "WindowDecision",
    "analyze_frame_filtering",
    "analyze_miniclips",
    "analyze_redundancy",
    "build_shot_analysis",
    "build_timeline",
    "decode_video",
    "drawing",
    "estimate_frame_mapping",
    "kept_candidates",
    "plots",
    "predict_transitions",
    "process_swin_input",
    "read_miniclip",
    "render_pass",
    "study_video",
    "sweep_hamming_threshold",
    "sweep_stride",
]
