"""Study of one video, procedure by procedure, for notebooks.

``VideoStudy`` loads the configuration, ffmpeg and TransNet V2 once. Each method
runs one procedure, stores its result as an attribute and returns it, so a cell can
re-run a procedure with other thresholds without decoding again::

    study = VideoStudy("video.mp4", "config.yaml", gpu=0)
    decoded = study.decode()
    filtering = study.filter_frames(luma_threshold=20)
    shots = study.detect_shots()                  # TransNet V2 output is cached per segment set
    redundancy = study.remove_redundancy(hamming_threshold=12)
    miniclips = study.plan_miniclips()            # every kept candidate is a miniclip
    frames = study.read_miniclip(0)               # (32, 224, 224, 3) uint8

Keyword overrides replace fields of the matching configuration section
(``FilterConfig``, ``SummaryConfig``) for that call only. :meth:`VideoStudy.run`
recomputes everything from ``study.config`` and writes all the outputs.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Sequence

import numpy as np

from preproc.config import Config, load_config
from preproc.detector import ShotDetector
from preproc.ffmpeg_io import FfmpegInfo, check_ffmpeg
from preproc.transnet import TransNetDetector

from . import plots, style
from .decoding import DecodedVideo, decode_video
from .export import (
    SWIN_INPUT,
    export_decoding,
    export_frame_filtering,
    export_miniclips,
    export_redundancy_removal,
    export_shot_detection,
    export_swin_input,
    required_thumbnails,
    write_json,
)
from .frame_filtering import FrameFilteringAnalysis, analyze_frame_filtering
from .frame_rendering import RenderResult, render_pass
from .miniclips import MiniclipAnalysis, analyze_miniclips, kept_candidates
from .redundancy_removal import RedundancyAnalysis, analyze_redundancy, sweep_hamming_threshold
from .report import build_summary, report_markdown
from .shot_detection import ShotDetectionAnalysis, build_shot_analysis, predict_transitions, sweep_stride
from .swin_input import SwinInputAnalysis, process_swin_input, read_miniclip
from .timeline import FrameTimeline, build_timeline
from .verification import verification_markdown, verify_against_preproc

DEFAULT_STRIDES = (1, 2, 4, 8)
DEFAULT_HAMMING_THRESHOLDS = tuple(range(0, 33, 2))


class VideoStudy:
    """Runs the summarization procedure on one video and keeps every intermediate result."""

    def __init__(
        self,
        video: str | Path,
        config: str | Path | Config,
        *,
        gpu: int = 0,
        threads: int = 0,
        detector: ShotDetector | None = None,
        ff: FfmpegInfo | None = None,
    ) -> None:
        self.video = Path(video)
        if not self.video.is_file():
            raise FileNotFoundError(f"video not found: {self.video}")
        self.config: Config = config if isinstance(config, Config) else load_config(config)
        self.ff = ff or check_ffmpeg(self.config.ffmpeg.ffmpeg, self.config.ffmpeg.ffprobe)
        self.gpu = gpu
        self.threads = threads
        self._detector = detector
        self._probability_cache: dict[tuple[Any, ...], list[np.ndarray]] = {}
        self.timings: dict[str, float] = {}
        self.decoded: DecodedVideo | None = None
        self.filtering: FrameFilteringAnalysis | None = None
        self.shots: ShotDetectionAnalysis | None = None
        self.redundancy: RedundancyAnalysis | None = None
        self.miniclips: MiniclipAnalysis | None = None
        self.swin_input: SwinInputAnalysis | None = None

    # ------------------------------------------------------------------ resources

    @property
    def device(self) -> str:
        return f"cuda:{self.gpu}" if self.gpu >= 0 else "cpu"

    @property
    def detector(self) -> ShotDetector:
        """TransNet V2, loaded on first use."""
        if self._detector is None:
            self._detector = TransNetDetector(self.config.transnet, device=self.device)
        return self._detector

    @contextmanager
    def _timer(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.timings[name] = perf_counter() - start

    @staticmethod
    def _need(value: Any, name: str, method: str) -> Any:
        if value is None:
            raise RuntimeError(f"no {name} yet: call {method}() first or pass it as an argument")
        return value

    # ------------------------------------------------------------------ procedures

    def decode(self, *, mapping: bool = True) -> DecodedVideo:
        """Constant-rate decoding with ffmpeg (low-resolution luma and TransNet input)."""
        with self._timer("decoding"):
            self.decoded = decode_video(self.video, self.config, self.ff, threads=self.threads, mapping=mapping)
        return self.decoded

    def filter_frames(self, decoded: DecodedVideo | None = None, **overrides: Any) -> FrameFilteringAnalysis:
        """Blank and static frame filtering. Overrides: ``luma_threshold``, ``std_threshold``,
        ``static_threshold``, ``static_min_frames``."""
        decoded = decoded or self._need(self.decoded, "decoded video", "decode")
        cfg = dataclasses.replace(self.config.filter, **overrides)
        with self._timer("frame_filtering"):
            self.filtering = analyze_frame_filtering(decoded.gray, cfg)
        return self.filtering

    def detect_shots(
        self,
        filtering: FrameFilteringAnalysis | None = None,
        decoded: DecodedVideo | None = None,
        *,
        threshold: float | None = None,
        **summary_overrides: Any,
    ) -> ShotDetectionAnalysis:
        """TransNet V2, shots and candidates. Overrides: ``threshold`` and ``frames_per_clip`` /
        ``stride`` / ``short_shot_windows`` (the candidates). The model output is reused
        while the segments do not change."""
        decoded = decoded or self._need(self.decoded, "decoded video", "decode")
        filtering = filtering or self._need(self.filtering, "frame filtering", "filter_frames")
        summary = dataclasses.replace(self.config.summary, **summary_overrides)
        threshold = self.config.transnet.threshold if threshold is None else float(threshold)
        key = (id(decoded), tuple((s.start, s.end) for s in filtering.segments))
        with self._timer("shot_detection"):
            if key not in self._probability_cache:
                self._probability_cache = {key: predict_transitions(decoded.rgb48, filtering.segments, self.detector)}
            probabilities = self._probability_cache[key]
            self.shots = build_shot_analysis(decoded.n_frames, filtering.segments, probabilities, threshold, summary)
        return self.shots

    def remove_redundancy(
        self,
        shots: ShotDetectionAnalysis | None = None,
        decoded: DecodedVideo | None = None,
        **summary_overrides: Any,
    ) -> RedundancyAnalysis:
        """pHash redundancy removal over the windows. Overrides: ``hamming_threshold``, ``phash_step``."""
        decoded = decoded or self._need(self.decoded, "decoded video", "decode")
        shots = shots or self._need(self.shots, "shot detection", "detect_shots")
        summary = dataclasses.replace(shots.summary, **summary_overrides)
        _same_candidates(summary, shots)
        with self._timer("redundancy_removal"):
            self.redundancy = analyze_redundancy(decoded.gray, shots.windows, summary)
        return self.redundancy

    def plan_miniclips(
        self,
        shots: ShotDetectionAnalysis | None = None,
        redundancy: RedundancyAnalysis | None = None,
        decoded: DecodedVideo | None = None,
    ) -> MiniclipAnalysis:
        """Every kept window and every short-shot candidate becomes a miniclip (N = 0: the longest shot)."""
        decoded = decoded or self._need(self.decoded, "decoded video", "decode")
        shots = shots or self._need(self.shots, "shot detection", "detect_shots")
        redundancy = redundancy or self._need(self.redundancy, "redundancy removal", "remove_redundancy")
        _same_candidates(redundancy.summary, shots)
        kept = kept_candidates(len(shots.windows), redundancy.kept, len(shots.short_windows))
        with self._timer("miniclips"):
            self.miniclips = analyze_miniclips(decoded.gray, shots.shots, shots.candidates, kept,
                                               redundancy.summary, self.config.fps)
        return self.miniclips

    def extract_miniclips(
        self,
        folder: str | Path | None = None,
        miniclips: MiniclipAnalysis | None = None,
        decoded: DecodedVideo | None = None,
        *,
        sheets: bool = True,
        measure_storage: bool = True,
    ) -> SwinInputAnalysis:
        """Second ffmpeg pass, miniclip by miniclip, at the Swin input size.

        With ``folder``, every miniclip is stored as ``preproc`` stores it (one .mp4 each) and,
        with ``measure_storage``, read back to measure the storage PSNR. Without ``folder``,
        nothing is written: only the alignment check, the digests and the tensor statistics.
        """
        decoded = decoded or self._need(self.decoded, "decoded video", "decode")
        miniclips = miniclips or self._need(self.miniclips, "miniclips", "plan_miniclips")
        with self._timer("swin_input"):
            self.swin_input = process_swin_input(decoded, miniclips.miniclips, self.config, self.ff,
                                                 folder=folder, sheets=sheets, measure_storage=measure_storage,
                                                 threads=self.threads)
        return self.swin_input

    def read_miniclip(self, k: int) -> np.ndarray:
        """The T frames of miniclip ``k`` at the Swin input size, ``(T, size, size, 3)`` uint8."""
        decoded = self._need(self.decoded, "decoded video", "decode")
        miniclips = self._need(self.miniclips, "miniclips", "plan_miniclips")
        return read_miniclip(decoded, miniclips.miniclips[k], self.config, self.ff, self.threads)

    def timeline(self) -> FrameTimeline:
        """Per-frame status from the last results of every procedure."""
        filtering = self._need(self.filtering, "frame filtering", "filter_frames")
        shots = self._need(self.shots, "shot detection", "detect_shots")
        redundancy = self._need(self.redundancy, "redundancy removal", "remove_redundancy")
        miniclips = self._need(self.miniclips, "miniclips", "plan_miniclips")
        return build_timeline(filtering, shots, redundancy, miniclips, self.config.fps)

    def read_frames(self, frames: Sequence[int], width: int = 320) -> dict[int, np.ndarray]:
        """RGB frames of the constant-rate timeline at ``width`` px (one streaming ffmpeg pass)."""
        decoded = self._need(self.decoded, "decoded video", "decode")
        result = render_pass(decoded, self.config, self.ff, thumbnail_frames=frames,
                             thumbnail_width=width, threads=self.threads)
        return result.thumbnails

    # ------------------------------------------------------------------ sweeps

    def sweep_stride(self, strides: Sequence[int] = DEFAULT_STRIDES) -> list[dict[str, Any]]:
        shots = self._need(self.shots, "shot detection", "detect_shots")
        return sweep_stride(shots.shots, shots.summary.frames_per_clip, strides, self.config.fps,
                            shots.summary.short_shot_windows)

    def sweep_hamming_threshold(self, thresholds: Sequence[float] = DEFAULT_HAMMING_THRESHOLDS) -> list[dict[str, Any]]:
        """Windows discarded and miniclips that result for each θ_H."""
        decoded = self._need(self.decoded, "decoded video", "decode")
        shots = self._need(self.shots, "shot detection", "detect_shots")
        redundancy = self._need(self.redundancy, "redundancy removal", "remove_redundancy")
        return sweep_hamming_threshold(decoded.gray, shots.windows, redundancy.summary, thresholds,
                                       short_candidates=len(shots.short_windows), fps=self.config.fps)

    # ------------------------------------------------------------------ whole study

    def run(
        self,
        out: str | Path,
        *,
        annotated_video: bool = True,
        video_width: int = 320,
        thumbnail_width: int = 160,
        max_images: int | None = 200,
        strides: Sequence[int] = DEFAULT_STRIDES,
        hamming_thresholds: Sequence[float] = DEFAULT_HAMMING_THRESHOLDS,
        sheets: bool = True,
        measure_storage: bool = True,
        verify: bool = False,
    ) -> dict[str, Any]:
        """Runs every procedure with ``self.config`` and writes all outputs to ``out``. Returns the summary."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.timings = {}
        total = perf_counter()
        fps = self.config.fps

        decoded = self.decode()
        filtering = self.filter_frames(decoded)
        shots = self.detect_shots(filtering, decoded)
        redundancy = self.remove_redundancy(shots, decoded)
        miniclips = self.plan_miniclips(shots, redundancy, decoded)
        warnings = list(decoded.warnings)

        with self._timer("sweeps"):
            stride_rows = self.sweep_stride(strides)
            threshold_rows = self.sweep_hamming_threshold(hamming_thresholds)

        with self._timer("swin_input"):
            swin_input, swin_warnings = export_swin_input(out, decoded, miniclips, self.config, self.ff,
                                                          sheets=sheets, measure_storage=measure_storage,
                                                          threads=self.threads)
        self.swin_input = swin_input
        warnings += swin_warnings

        timeline = self.timeline()
        with self._timer("rendering"):
            render: RenderResult = render_pass(
                decoded, self.config, self.ff,
                thumbnail_frames=required_thumbnails(filtering, shots, redundancy, max_images),
                thumbnail_width=thumbnail_width,
                timeline=timeline,
                video_path=out / "annotated_video.mp4" if annotated_video else None,
                video_width=video_width,
                threads=self.threads,
            )
        if render.frames_decoded and render.frames_decoded != decoded.n_frames:
            warnings.append(f"the render pass decoded {render.frames_decoded} frames, "
                            f"the analysis pass {decoded.n_frames}")
        if render.codec not in (None, "libx264"):
            warnings.append(f"annotated video encoded with {render.codec}: ffmpeg has no libx264")

        with self._timer("export"):
            thumbs, size = render.thumbnails, render.thumbnail_size
            export_decoding(out, decoded)
            export_frame_filtering(out, filtering, thumbs, size, fps, max_images)
            export_shot_detection(out, shots, thumbs, size, fps, stride_rows, max_images)
            export_redundancy_removal(out, redundancy, thumbs, size, fps, threshold_rows, max_images)
            export_miniclips(out, miniclips)
            style.save_figure(plots.plot_timeline(timeline, shots, miniclips), out / "timeline.png")

        verification = None
        if verify:
            with self._timer("verification"):
                verification = verify_against_preproc(self.video, self.config, self.detector, self.ff, decoded,
                                                      filtering, shots, redundancy, miniclips, swin_input,
                                                      self.threads)
            (out / "verification.md").write_text(verification_markdown(verification), encoding="utf-8")
            if not all(r["match"] for r in verification):
                warnings.append("the study differs from preproc.process_video (see verification.md)")

        self.timings["total"] = perf_counter() - total
        summary = build_summary(decoded, filtering, shots, redundancy, miniclips, swin_input.summary(),
                                warnings, self.timings, verification)
        summary["swin_input"]["folder"] = SWIN_INPUT
        write_json(out / "summary.json", summary)
        (out / "report.md").write_text(report_markdown(summary, out), encoding="utf-8")
        return summary


def study_video(
    video: str | Path,
    config: str | Path | Config,
    out: str | Path,
    *,
    gpu: int = 0,
    threads: int = 0,
    **run_options: Any,
) -> dict[str, Any]:
    """One-call study: ``VideoStudy(video, config).run(out, **run_options)``."""
    return VideoStudy(video, config, gpu=gpu, threads=threads).run(out, **run_options)


def _same_candidates(summary: Any, shots: ShotDetectionAnalysis) -> None:
    same = (summary.window_length == shots.window_length
            and summary.frames_per_clip == shots.summary.frames_per_clip
            and summary.short_shot_windows == shots.summary.short_shot_windows)
    if not same:
        raise ValueError("frames_per_clip, stride and short_shot_windows define the candidates: "
                         "change them in detect_shots()")
