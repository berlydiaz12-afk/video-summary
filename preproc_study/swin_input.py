"""Frame extraction at the Swin input size, storage as .mp4 and tensor statistics.

Works miniclip by miniclip, like the bulk package: ``preproc.pipeline.iter_miniclip_frames``
runs one ffmpeg pass (same ``fps``, ``select`` and scaler) and yields each miniclip as
soon as its T frames are decoded; the miniclip is checked, stored and released, so
memory does not grow with the number of miniclips.

Each miniclip is stored with ``preproc.ffmpeg_io.encode_clip`` (H.264 yuv420p,
``output.crf``, ``output.preset``, BT.601 tags) at fps / r, exactly like
``preproc``'s ``miniclips/<id>/<id>_<k>.mp4``. What Swin receives is that file
decoded, so with ``measure_storage`` every file is read back with
``preproc.storage.load_miniclip``: the PSNR against the extracted frames measures the
storage loss, and the tensor statistics are computed on the decoded frames.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from preproc.config import Config, OutputConfig
from preproc.ffmpeg_io import FfmpegInfo, encode_clip, extract_frames
from preproc.normalize import normalize, to_cthw, visible_region
from preproc.pipeline import alignment_error, clip_stride, iter_miniclip_frames
from preproc.storage import load_miniclip
from preproc.structs import Miniclip

from .decoding import DecodedVideo
from .drawing import miniclip_sheet, write_png


def playback_fps(clip: Miniclip, cfg: Config) -> float:
    """Rate at which ``preproc`` stores a miniclip: fps / r (r of the configuration for the N = 0 fallback)."""
    return cfg.fps / (clip_stride(clip) or cfg.summary.stride)


def miniclip_digest(frames: np.ndarray) -> str:
    """SHA-1 of the raw uint8 frames of a miniclip (compares pixels without keeping them)."""
    x = np.ascontiguousarray(frames, dtype=np.uint8)
    return hashlib.sha1(x.tobytes() + str(x.shape).encode()).hexdigest()


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """PSNR (dB) between two uint8 arrays of the same shape; ``inf`` when they are identical."""
    if reference.shape != test.shape:
        raise ValueError(f"shapes differ: {reference.shape} and {test.shape}")
    mse = float(np.mean(np.square(reference.astype(np.float64) - test.astype(np.float64))))
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


class TensorStatistics:
    """Per-channel mean/std before (x / 255) and after normalization, accumulated miniclip by miniclip."""

    def __init__(self, output: OutputConfig) -> None:
        self.output = output
        self._before = np.zeros((2, 3))
        self._after = np.zeros((2, 3))
        self.count = 0
        self.miniclips = 0
        self.shape: list[int] | None = None
        self.dtype: str | None = None

    def add(self, clip: np.ndarray) -> None:
        x = clip.reshape(-1, 3).astype(np.float64) / 255.0
        self._before += (x.sum(axis=0), np.square(x).sum(axis=0))
        normalized = normalize(clip, self.output.mean, self.output.std)
        tensor = to_cthw(normalized)
        self.shape, self.dtype = list(tensor.shape), str(tensor.dtype)
        y = normalized.reshape(-1, 3).astype(np.float64)
        self._after += (y.sum(axis=0), np.square(y).sum(axis=0))
        self.count += len(x)
        self.miniclips += 1

    def _stats(self, acc: np.ndarray) -> dict[str, list[float]]:
        if self.count == 0:
            return {"mean": [], "std": []}
        mean = acc[0] / self.count
        std = np.sqrt(np.maximum(acc[1] / self.count - mean ** 2, 0.0))
        return {"mean": [round(float(v), 6) for v in mean], "std": [round(float(v), 6) for v in std]}

    def as_dict(self) -> dict[str, Any]:
        return {
            "miniclips": self.miniclips,
            "tensor_per_miniclip": self.shape,
            "tensor_layout": "C, T, H, W (preproc.normalize.to_cthw)",
            "tensor_dtype": self.dtype,
            "normalization": {"mean": list(self.output.mean), "std": list(self.output.std)},
            "channels": ["R", "G", "B"],
            "before_normalization": self._stats(self._before),
            "after_normalization": self._stats(self._after),
        }


@dataclass
class StoredMiniclip:
    order: int
    path: Path | None          # None when nothing was written
    playback_fps: float
    bytes: int | None
    psnr: float | None         # storage loss (None without measure_storage)
    digest: str                # SHA-1 of the extracted frames (before encoding)


@dataclass
class SwinInputAnalysis:
    miniclips: list[StoredMiniclip] = field(default_factory=list)
    unique_frames: int = 0
    alignment_error: float = 0.0       # max error against the analysis luma (approximate check)
    statistics: dict[str, Any] = field(default_factory=dict)
    statistics_source: str = "extracted frames"
    crf: int = 18
    preset: str = "veryfast"

    @property
    def digests(self) -> list[str]:
        return [m.digest for m in self.miniclips]

    @property
    def total_bytes(self) -> int | None:
        sizes = [m.bytes for m in self.miniclips]
        return None if not sizes or any(s is None for s in sizes) else int(sum(sizes))  # type: ignore[arg-type]

    def psnr_summary(self) -> dict[str, float | None]:
        values = [m.psnr for m in self.miniclips if m.psnr is not None]
        if not values:
            return {"mean_db": None, "min_db": None}
        finite = [v for v in values if np.isfinite(v)]
        return {
            "mean_db": round(float(np.mean(finite)), 3) if finite else None,
            "min_db": round(float(min(values)), 3),
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.statistics,
            "statistics_of": self.statistics_source,
            "storage": {
                "format": f"mp4 H.264 yuv420p (preproc.ffmpeg_io.encode_clip, crf {self.crf}, preset {self.preset})",
                "files": len([m for m in self.miniclips if m.path is not None]),
                "bytes": self.total_bytes,
                "playback_fps": sorted({round(m.playback_fps, 4) for m in self.miniclips}),
                "psnr": self.psnr_summary(),
            },
            "unique_frames": self.unique_frames,
            "alignment_error": self.alignment_error,
        }


def process_swin_input(
    decoded: DecodedVideo,
    miniclips: Sequence[Miniclip],
    cfg: Config,
    ff: FfmpegInfo,
    *,
    folder: str | Path | None = None,
    sheets: bool = True,
    measure_storage: bool = True,
    threads: int = 0,
) -> SwinInputAnalysis:
    """Extracts, checks and (with ``folder``) stores every miniclip, one at a time.

    With ``folder``: ``miniclip_<k>.mp4`` (the file ``preproc`` would store) and, with
    ``sheets``, ``miniclip_<k>.png`` (the T frames in a grid). ``measure_storage``
    reads each .mp4 back to measure the storage PSNR and compute the tensor statistics
    on what Swin receives; it needs ``folder``.
    """
    out = Path(folder) if folder is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for old in (*out.glob("miniclip_*.mp4"), *out.glob("miniclip_*.png")):
            old.unlink()
    measure = measure_storage and out is not None
    region = visible_region(cfg.output, decoded.info.display_aspect)
    stats = TensorStatistics(cfg.output)
    result = SwinInputAnalysis(
        unique_frames=len({i for c in miniclips for i in c.frame_indices}),
        statistics_source="decoded .mp4 files (what Swin receives)" if measure else "extracted frames (before encoding)",
        crf=cfg.output.crf,
        preset=cfg.output.preset,
    )
    for k, frames in iter_miniclip_frames(decoded.path, ff, cfg, decoded.info, miniclips, threads):
        clip = miniclips[k]
        idx = np.asarray(clip.frame_indices, dtype=np.int64)
        result.alignment_error = max(result.alignment_error, alignment_error(frames, decoded.gray[idx], region))
        rate = playback_fps(clip, cfg)
        path, size, quality, received = None, None, None, frames
        if out is not None:
            path = out / f"miniclip_{k:03d}.mp4"
            encode_clip(frames, path, rate, ff, cfg.output.crf, cfg.output.preset)
            size = path.stat().st_size
            if sheets:
                write_png(out / f"miniclip_{k:03d}.png",
                          miniclip_sheet(frames, clip, k, cfg.fps, clip_stride(clip), rate))
            if measure:
                received = load_miniclip(path, ff)
                quality = psnr(frames, received) if received.shape == frames.shape else None
        stats.add(received)
        result.miniclips.append(StoredMiniclip(k, path, rate, size, quality, miniclip_digest(frames)))
    result.statistics = stats.as_dict()
    return result


def read_miniclip(
    decoded: DecodedVideo,
    clip: Miniclip,
    cfg: Config,
    ff: FfmpegInfo,
    threads: int = 0,
) -> np.ndarray:
    """The T frames of one miniclip at the Swin input size, ``(T, size, size, 3)`` uint8 (for notebooks)."""
    frames, unique = extract_frames(decoded.path, ff, cfg, decoded.info, [clip.frame_indices], threads)
    position = {f: i for i, f in enumerate(unique)}
    return frames[[position[f] for f in clip.frame_indices]]
