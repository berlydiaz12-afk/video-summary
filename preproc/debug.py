from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .config import Config
from .ffmpeg_io import FfmpegInfo, encode_video
from .structs import Miniclip

# Sin '<' ni '=' para que el nombre sea válido en Windows.
_CASE_SLUG = {"ninguno": "ninguno", "N=0": "N0"}


def miniclip_filename(order: int, clip: Miniclip, case: str, fps: int) -> str:
    mm, ss = divmod(int(clip.start // fps), 60)
    if clip.kind == "toma_corta":
        tag = f"corta{clip.candidate_index:03d}"
    elif clip.candidate_index is not None:
        tag = f"v{clip.candidate_index:03d}"
    else:
        tag = "larga"
    slug = _CASE_SLUG.get(case, "caso")
    return f"{order:03d}_{tag}_t{mm:02d}m{ss:02d}s_toma{clip.shot}_mov{clip.motion:.4f}_{slug}.mp4"


def write_miniclip_videos(
    clips: np.ndarray,
    miniclips: Sequence[Miniclip],
    case: str,
    out_dir: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
) -> tuple[list[Path], str]:
    """Escribe un .mp4 por miniclip en ``out_dir``. Devuelve las rutas y el códec usado."""
    prepare_debug_dir(out_dir)
    codec = cfg.debug.codec
    for order, (frames, clip) in enumerate(zip(clips, miniclips)):
        codec = write_debug_clip(frames, clip, order, case, out_dir, ff, cfg)
    return sorted(Path(out_dir).glob("*.mp4")), codec


def prepare_debug_dir(out_dir: str | Path) -> None:
    """Crea ``out_dir`` y borra los .mp4 de una corrida anterior."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.mp4"):
        old.unlink()


def write_debug_clip(
    frames: np.ndarray,
    clip: Miniclip,
    order: int,
    case: str,
    out_dir: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
) -> str:
    """Escribe un miniclip ``(T, H, W, 3)`` como .mp4 con nombre descriptivo. Devuelve el códec usado."""
    path = Path(out_dir) / miniclip_filename(order, clip, case, cfg.fps)
    steps = {b - a for a, b in zip(clip.frame_indices, clip.frame_indices[1:])}
    stride = steps.pop() if len(steps) == 1 and min(steps) > 0 else cfg.summary.stride
    fps = cfg.debug.playback_fps or cfg.fps / stride  # 25 / r: duración real del miniclip
    return encode_video(frames, path, fps, ff, codec=cfg.debug.codec, crf=cfg.debug.crf)
