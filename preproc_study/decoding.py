"""Constant frame rate decoding with ffmpeg.

The video is decoded once by ``preproc.ffmpeg_io.decode_analysis``, the same pass as
the bulk package: ``fps=25`` filter, area-scaled 64 × 36 BT.601 luma and bicubic
48 × 27 RGB for TransNet V2. Every frame index in this package is therefore an index
of the bulk package too.

ffmpeg does not report which source frame fed each output frame.
:func:`estimate_frame_mapping` rebuilds that table from the packet timestamps
reported by ffprobe, emulating the ``fps`` filter: each source timestamp is rounded
to the nearest output slot and output frame ``n`` shows the last source frame whose
slot is ``<= n``. The table is an estimate (edit lists, dropped packets and the
container start time can shift it by a frame).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from preproc import __version__ as preproc_version
from preproc.config import Config
from preproc.ffmpeg_io import TRANSNET_HEIGHT, TRANSNET_WIDTH, FfmpegInfo, decode_analysis, probe
from preproc.structs import VideoFailed, VideoInfo


@dataclass
class FrameMapping:
    """Estimated source frame of every frame on the constant-rate timeline."""

    source_index: np.ndarray  # int64[n]: source frame, in presentation order
    source_time: np.ndarray   # float64[n]: source timestamp (s) relative to the first frame
    n_source: int             # video packets reported by ffprobe (≈ source frames)

    @property
    def n_used(self) -> int:
        return int(np.unique(self.source_index).size)

    @property
    def repeated(self) -> int:
        """Output frames that repeat a source frame already shown (source below 25 fps)."""
        return len(self.source_index) - self.n_used

    @property
    def skipped(self) -> int:
        """Source frames never shown (source above 25 fps)."""
        return self.n_source - self.n_used


@dataclass
class DecodedVideo:
    """Low-resolution frames on the constant-rate timeline, plus metadata."""

    path: Path
    info: VideoInfo
    fps: int
    gray: np.ndarray          # (n, gray_height, gray_width) uint8 luma
    rgb48: np.ndarray         # (n, 27, 48, 3) uint8, TransNet V2 input
    mapping: FrameMapping | None
    ffmpeg_version: str
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.gray)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def time(self, frame: int) -> float:
        return frame / self.fps

    def metadata(self) -> dict[str, Any]:
        import cv2

        info = self.info
        width, height = info.display_size
        mapping = None
        if self.mapping is not None:
            mapping = {
                "estimated": True,
                "source_packets": self.mapping.n_source,
                "source_frames_used": self.mapping.n_used,
                "repeated_frames": self.mapping.repeated,
                "skipped_source_frames": self.mapping.skipped,
            }
        return {
            "path": str(self.path),
            "source": {
                "codec": info.codec,
                "fps": info.fps,
                "duration_s": info.duration,
                "frames": info.nb_frames,
                "width": width,
                "height": height,
                "coded_width": info.width,
                "coded_height": info.height,
                "rotation": info.rotation,
                "sar": info.sar,
                "display_aspect": round(info.display_aspect, 6),
            },
            "timeline": {
                "fps": self.fps,
                "frames": self.n_frames,
                "duration_s": round(self.duration, 3),
            },
            "mapping": mapping,
            "analysis_frames": {
                "gray": list(self.gray.shape[1:]),
                "transnet": [TRANSNET_HEIGHT, TRANSNET_WIDTH, 3],
            },
            "decoder": "ffmpeg (preproc.ffmpeg_io.decode_analysis)",
            "versions": {
                "ffmpeg": self.ffmpeg_version,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "preproc": preproc_version,
            },
            "warnings": list(self.warnings),
        }


def decode_video(
    path: str | Path,
    cfg: Config,
    ff: FfmpegInfo,
    *,
    threads: int = 0,
    mapping: bool = True,
) -> DecodedVideo:
    """Decodes the whole video at ``cfg.fps`` and low resolution (streaming, bounded memory)."""
    path = Path(path)
    info = probe(path, ff)
    frames = decode_analysis(path, ff, cfg, info, threads)
    if len(frames.gray) == 0:
        raise VideoFailed(f"ffmpeg returned no frames at {cfg.fps} fps")
    warnings: list[str] = []
    frame_mapping = None
    if mapping:
        try:
            frame_mapping = estimate_frame_mapping(probe_packet_times(path, ff), len(frames.gray), cfg.fps)
        except (VideoFailed, ValueError) as exc:
            warnings.append(f"frame mapping unavailable: {exc}")
    return DecodedVideo(path, info, cfg.fps, frames.gray, frames.rgb48, frame_mapping, ff.version, warnings)


def probe_packet_times(path: str | Path, ff: FfmpegInfo) -> np.ndarray:
    """Timestamps (s) of the video packets, without decoding (pts, or dts when pts is missing)."""
    cmd = [ff.ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=pts_time,dts_time", "-of", "csv=p=0", str(path)]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise VideoFailed(f"ffprobe could not list packets: {res.stderr.strip()[-500:]}")
    times: list[float] = []
    for line in res.stdout.splitlines():
        for value in line.strip().split(","):
            try:
                t = float(value)
            except ValueError:
                continue
            if np.isfinite(t):
                times.append(t)
                break
    if not times:
        raise ValueError("ffprobe reported no packet timestamps")
    return np.asarray(times, dtype=np.float64)


def estimate_frame_mapping(source_times: np.ndarray, n_frames: int, fps: int) -> FrameMapping:
    """Emulates the ``fps`` filter (nearest rounding) to map output frames to source frames."""
    t = np.sort(np.asarray(source_times, dtype=np.float64))
    if t.size == 0:
        raise ValueError("no source timestamps")
    t = t - t[0]
    slots = np.floor(t * fps + 0.5).astype(np.int64)
    source = np.searchsorted(slots, np.arange(n_frames, dtype=np.int64), side="right") - 1
    source = np.clip(source, 0, t.size - 1)
    return FrameMapping(source, t[source], int(t.size))
