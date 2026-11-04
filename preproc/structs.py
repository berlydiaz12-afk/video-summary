from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class VideoFailed(Exception):
    """El video no puede procesarse; el motivo se registra en ``errores_<conjunto>.log``."""


@dataclass(frozen=True)
class VideoInfo:
    """Metadatos originales del video según ffprobe."""

    path: str
    codec: str | None
    width: int
    height: int
    sar: float
    rotation: int
    fps: float | None
    duration: float | None
    nb_frames: int | None

    @property
    def rotated(self) -> bool:
        return abs(self.rotation) % 180 == 90

    @property
    def display_size(self) -> tuple[int, int]:
        """(ancho, alto) tras aplicar la rotación de los metadatos."""
        return (self.height, self.width) if self.rotated else (self.width, self.height)

    @property
    def display_aspect(self) -> float:
        """Proporción de pantalla (ancho / alto) considerando SAR y rotación."""
        aspect = self.width * self.sar / self.height
        return 1.0 / aspect if self.rotated else aspect


@dataclass(frozen=True)
class Segment:
    """Tramo de cuadros contiguos que sobrevive al paso 2."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Shot:
    """Toma: tramo continuo sin cuadros de transición dentro de un segmento."""

    index: int
    segment: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Window:
    """Ventana de ``length`` cuadros contenida en una toma."""

    shot: int
    start: int
    length: int
    stride: int | None = None  # separación entre cuadros del miniclip; None = r de la configuración

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class Miniclip:
    """Miniclip elegido en el paso 5."""

    frame_indices: tuple[int, ...]
    shot: int
    kind: str                    # "conservada" | "toma_corta" | "toma_mas_larga"
    candidate_index: int | None  # índice entre las ventanas candidatas (None si no es candidata)
    motion: float                # m_j
    fallback: bool               # proviene del caso de respaldo N = 0

    @property
    def start(self) -> int:
        return self.frame_indices[0]


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Tramos ``[start, end)`` donde ``mask`` es verdadero."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))
