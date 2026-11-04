from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import IMAGENET_MEAN, IMAGENET_STD, OutputConfig


def scale_filter(out: OutputConfig, display_aspect: float = 1.0) -> str:
    """Filtro de ffmpeg que lleva cada cuadro a ``size × size``.

    * ``stretch``: escala directa, sin conservar la proporción (opción del Cap. 4).
    * ``short_side_crop``: lado corto a ``size`` según la proporción de pantalla
      (SAR y rotación) y recorte central.
    """
    s, flags = out.size, out.sws_flags
    if out.resize_mode == "stretch":
        return f"scale={s}:{s}:flags={flags}"
    if out.resize_mode == "short_side_crop":
        w, h = _scaled_size(s, display_aspect)
        return f"scale={w}:{h}:flags={flags},crop={s}:{s}"
    raise ValueError(f"resize_mode desconocido: {out.resize_mode}")


def visible_region(out: OutputConfig, display_aspect: float = 1.0) -> tuple[float, float, float, float]:
    """Fracción ``(y0, y1, x0, x1)`` del cuadro original que queda en la salida."""
    if out.resize_mode == "stretch":
        return (0.0, 1.0, 0.0, 1.0)
    w, h = _scaled_size(out.size, display_aspect)
    fx, fy = out.size / w, out.size / h
    return ((1 - fy) / 2, (1 + fy) / 2, (1 - fx) / 2, (1 + fx) / 2)


def normalize(
    clip: np.ndarray,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """uint8 ``(..., H, W, 3)`` → float32 ``(x / 255 − media) / desv`` por canal RGB."""
    x = np.asarray(clip)
    if x.shape[-1] != 3:
        raise ValueError(f"se esperaba el canal RGB al final, forma recibida {x.shape}")
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    return (x.astype(np.float32) / np.float32(255.0) - m) / s


def to_cthw(clip: np.ndarray) -> np.ndarray:
    """``(..., T, H, W, C)`` → ``(..., C, T, H, W)``, el formato que espera Video Swin."""
    return np.ascontiguousarray(np.moveaxis(np.asarray(clip), -1, -4))


def _scaled_size(size: int, aspect: float) -> tuple[int, int]:
    if aspect >= 1.0:
        return max(size, int(round(size * aspect))), size
    return size, max(size, int(round(size / aspect)))
