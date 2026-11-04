from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dct

HASH_SIZE = 8
RESIZE = 32


def phash(image: np.ndarray) -> int:
    """pHash de 64 bits de una imagen ``(H, W)`` en gris o ``(H, W, 3|4)`` en RGB(A) uint8."""
    x = np.asarray(image)
    if x.ndim == 2:
        return int(phash_batch(x[None])[0])
    if x.ndim == 3 and x.shape[-1] in (3, 4):
        return int(phash_batch(x[None, ..., :3])[0])
    raise ValueError(f"forma de imagen no soportada: {x.shape}")


def phash_batch(images: np.ndarray) -> np.ndarray:
    """pHash de un lote: ``(N, H, W)`` en gris o ``(N, H, W, 3)`` en RGB → ``uint64[N]``."""
    x = np.asarray(images)
    if x.ndim == 4:
        if x.shape[-1] not in (3, 4):
            raise ValueError(f"se esperaba RGB al final, forma recibida {x.shape}")
        x = _luma(x[..., :3])
    elif x.ndim != 3:
        raise ValueError(f"forma de lote no soportada: {x.shape}")
    n, h, w = x.shape
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    if h == 0 or w == 0:
        raise ValueError("imagen vacía")
    x = x.astype(np.float64, copy=False)
    small = _area_matrix(h, RESIZE) @ x @ _area_matrix(w, RESIZE).T          # (N, 32, 32)
    coeffs = dct(dct(small, type=2, axis=1, norm="ortho"), type=2, axis=2, norm="ortho")
    block = coeffs[:, :HASH_SIZE, :HASH_SIZE].reshape(n, HASH_SIZE * HASH_SIZE)
    bits = block > np.median(block, axis=1, keepdims=True)
    packed = np.packbits(bits, axis=1)                                         # (N, 8), MSB primero
    return packed.view(">u8").reshape(n).astype(np.uint64)


def hamming(a: int, b: int) -> int:
    """Distancia de Hamming entre dos hashes de 64 bits."""
    return (int(a) ^ int(b)).bit_count()


def hamming_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancia de Hamming elemento a elemento entre arreglos de ``uint64`` (con *broadcasting*)."""
    x = np.atleast_1d(np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)))
    as_bytes = np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(as_bytes, axis=-1).sum(axis=-1, dtype=np.int64)


@lru_cache(maxsize=64)
def _area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Matriz ``(n_out, n_in)`` de promedio por área: cada salida cubre ``n_in / n_out`` entradas."""
    m = np.zeros((n_out, n_in), dtype=np.float64)
    scale = n_in / n_out
    for i in range(n_out):
        lo, hi = i * scale, (i + 1) * scale
        for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), n_in)):
            overlap = min(hi, j + 1) - max(lo, j)
            if overlap > 0:
                m[i, j] = overlap
        m[i] /= m[i].sum()
    m.setflags(write=False)
    return m


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Luminancia BT.601 en float64 (misma ponderación que la pasada de análisis)."""
    x = rgb.astype(np.float64)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
