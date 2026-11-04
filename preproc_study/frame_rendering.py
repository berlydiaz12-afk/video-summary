"""Streaming render pass: annotated video and thumbnails.

ffmpeg decodes the video with the same ``fps`` filter, at a small display size, and
pipes raw RGB frames to Python; OpenCV draws on each frame and a second ffmpeg
process encodes it. Thumbnails are kept only for the frames the strips need, so
memory does not grow with the video length.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from preproc.config import Config
from preproc.ffmpeg_io import FfmpegError, FfmpegInfo, _iter_raw_frames, _rate, _thread_args

from .decoding import DecodedVideo
from .drawing import annotate_frame
from .timeline import FrameTimeline


def display_size(width: int, aspect: float) -> tuple[int, int]:
    """``(width, height)`` with the display aspect ratio and an even height (required by yuv420p)."""
    height = max(2, int(round(width / aspect / 2)) * 2)
    return width - width % 2, height


class StreamingEncoder:
    """Encodes RGB frames one by one with ffmpeg (H.264 yuv420p, lossy: for viewing)."""

    def __init__(
        self,
        path: str | Path,
        size: tuple[int, int],
        fps: float,
        ff: FfmpegInfo,
        *,
        crf: int = 23,
        preset: str = "veryfast",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = size
        self.codec = "libx264" if ff.has_libx264 else "mpeg4"
        enc = (["-c:v", "libx264", "-preset", preset, "-crf", str(crf)] if self.codec == "libx264"
               else ["-c:v", "mpeg4", "-q:v", "3"])
        cmd = [ff.ffmpeg, "-hide_banner", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{self.width}x{self.height}",
               "-framerate", _rate(fps), "-i", "pipe:0",
               "-an", *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"expected a ({self.height}, {self.width}, 3) frame, got {frame.shape}")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            self._proc.wait()
            raise FfmpegError(f"ffmpeg stopped while writing {self.path}: {self._error()}") from None
        self.frames += 1

    def close(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        code = self._proc.wait()
        message = self._error()
        self._stderr.close()
        if code != 0:
            raise FfmpegError(f"ffmpeg could not encode {self.path}: {message}")

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()
        self._stderr.close()

    def _error(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", "replace").strip()[-800:]

    def __enter__(self) -> "StreamingEncoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


@dataclass
class RenderResult:
    thumbnails: dict[int, np.ndarray] = field(default_factory=dict)  # frame → RGB thumbnail
    thumbnail_size: tuple[int, int] = (160, 90)                      # (width, height)
    frames_decoded: int = 0
    video_path: Path | None = None
    codec: str | None = None


def render_pass(
    decoded: DecodedVideo,
    cfg: Config,
    ff: FfmpegInfo,
    *,
    thumbnail_frames: Iterable[int] = (),
    thumbnail_width: int = 160,
    timeline: FrameTimeline | None = None,
    video_path: str | Path | None = None,
    video_width: int = 320,
    threads: int = 0,
) -> RenderResult:
    """One streaming decode at display size: writes the annotated video and keeps the requested thumbnails.

    Without ``video_path`` the frames are decoded at thumbnail size only.
    """
    aspect = decoded.info.display_aspect
    thumb_size = display_size(thumbnail_width, aspect)
    wanted = {int(n) for n in thumbnail_frames if 0 <= int(n) < decoded.n_frames}
    result = RenderResult(thumbnail_size=thumb_size)
    if video_path is not None and timeline is None:
        raise ValueError("the annotated video needs the frame timeline")
    if video_path is None and not wanted:
        return result

    size = display_size(video_width, aspect) if video_path is not None else thumb_size
    w, h = size
    vf = f"fps={cfg.fps},scale={w}:{h}:flags=area,setsar=1,format=rgb24"
    cmd = [ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error",
           *_thread_args(threads, complex_graph=False),
           "-i", str(decoded.path), "-an", "-sn", "-dn", "-vf", vf,
           *ff.passthrough_args, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    encoder = StreamingEncoder(video_path, size, cfg.fps, ff) if video_path is not None else None
    n = 0
    try:
        for chunk in _iter_raw_frames(cmd, (h, w, 3), 32):
            for frame in chunk:
                if n in wanted:
                    result.thumbnails[n] = (frame.copy() if size == thumb_size
                                            else cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA))
                if encoder is not None and timeline is not None and n < timeline.n_frames:
                    encoder.write(annotate_frame(frame.copy(), n, timeline))
                n += 1
    except BaseException:
        if encoder is not None:
            encoder.abort()
        raise
    if encoder is not None:
        encoder.close()
        result.video_path = Path(video_path)  # type: ignore[arg-type]
        result.codec = encoder.codec
    result.frames_decoded = n
    return result
