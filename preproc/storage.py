from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np

from .ffmpeg_io import FfmpegInfo, check_ffmpeg, encode_clip, probe, read_video_frames

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_DATASET = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    dataset: str

    @classmethod
    def create(cls, output_dir: str | Path, dataset: str) -> "OutputLayout":
        if not _DATASET.fullmatch(dataset):
            raise ValueError(f"nombre de conjunto inválido: {dataset!r} (use letras, números, '.', '_' o '-')")
        layout = cls(Path(output_dir).resolve() / dataset, dataset)
        for d in (layout.miniclips_dir, layout.meta_dir, layout.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return layout

    @property
    def miniclips_dir(self) -> Path:
        return self.root / "miniclips"

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def debug_dir(self) -> Path:
        return self.root / "debug"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def summary_csv(self) -> Path:
        return self.root / f"resumen_{self.dataset}.csv"

    @property
    def summary_parquet(self) -> Path:
        return self.root / f"resumen_{self.dataset}.parquet"

    @property
    def stats_json(self) -> Path:
        return self.root / f"estadisticas_{self.dataset}.json"

    @property
    def errors_log(self) -> Path:
        return self.root / f"errores_{self.dataset}.log"

    def clip_dir(self, video_id: str) -> Path:
        return self.miniclips_dir / video_id

    def clip_path(self, video_id: str, index: int) -> Path:
        return self.clip_dir(video_id) / clip_filename(video_id, index)

    def meta_path(self, video_id: str) -> Path:
        return self.meta_dir / f"{video_id}.json"

    def debug_path(self, video_id: str) -> Path:
        return self.debug_dir / video_id


def discover_videos(
    input_dir: str | Path,
    extensions: Iterable[str],
    exclude: Iterable[str | Path] = (),
) -> list[tuple[Path, str]]:
    """Busca videos recursivamente. Devuelve ``(ruta absoluta, ruta relativa POSIX)`` ordenados."""
    root = Path(input_dir).resolve()
    exts = {e.lower() for e in extensions}
    excluded = [Path(p).resolve() for p in exclude]
    found: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not any(_is_within(here / d, ex) for ex in excluded))
        for name in filenames:
            if Path(name).suffix.lower() in exts:
                path = here / name
                found.append((path, path.relative_to(root).as_posix()))
    found.sort(key=lambda item: item[1])
    return found


def assign_ids(rel_paths: Sequence[str]) -> list[str]:
    """Id legible a partir de la ruta relativa (``sub/v1.mp4`` → ``sub__v1``); colisiones → sufijo hash."""
    base = [
        _UNSAFE.sub("_", PurePosixPath(rel).with_suffix("").as_posix().replace("/", "__")).strip("_") or "video"
        for rel in rel_paths
    ]
    counts = Counter(base)
    return [
        b if counts[b] == 1 else f"{b}__{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:8]}"
        for b, rel in zip(base, rel_paths)
    ]


def clip_filename(video_id: str, index: int) -> str:
    """Nombre del .mp4 del miniclip ``index`` (orden temporal): ``<id>_000.mp4``."""
    return f"{video_id}_{index:03d}.mp4"


def is_done(layout: OutputLayout, video_id: str, config_hash: str) -> bool:
    """El video ya tiene salida completa (JSON y todos sus miniclips) con la misma configuración."""
    meta_path = layout.meta_path(video_id)
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    files = (meta.get("salida") or {}).get("archivos") or []
    return (
        meta.get("config_hash") == config_hash
        and bool(files)
        and all((layout.root / f).is_file() for f in files)
    )


class ClipWriter:
    """Escribe los miniclips de un video uno a uno, sin tenerlos todos en memoria.

    Cada ``write`` codifica un miniclip en una carpeta temporal;
    ``commit`` reemplaza ``miniclips/<id>/`` por esa carpeta y escribe el .json.
    Si algo falla antes de ``commit`` (o al salir del ``with`` con una excepción),
    la carpeta temporal se borra y no queda salida a medias::

        with ClipWriter(layout, vid, ff, crf=18) as writer:
            writer.write(0, frames, fps)
            ...
            writer.commit(meta)
    """

    def __init__(self, layout: OutputLayout, video_id: str, ff: FfmpegInfo, *,
                 crf: int = 18, preset: str = "medium") -> None:
        self.layout, self.video_id, self.ff = layout, video_id, ff
        self.crf, self.preset = crf, preset
        self.final_dir = layout.clip_dir(video_id)
        self.tmp_dir = self.final_dir.with_name(self.final_dir.name + ".tmp")
        self.sizes: list[int] = []
        self.rates: list[float] = []
        self.shape: tuple[int, ...] | None = None
        layout.meta_path(video_id).unlink(missing_ok=True)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True)

    def __enter__(self) -> "ClipWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self.abort()

    def write(self, index: int, frames: np.ndarray, fps: float) -> None:
        """Codifica el miniclip ``index`` (en orden: 0, 1, 2, …) con ``fps`` de reproducción."""
        if index != len(self.sizes):
            raise ValueError(f"se esperaba el miniclip {len(self.sizes)}, llegó el {index}")
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        if self.shape is not None and frames.shape != self.shape:
            raise ValueError(f"forma {frames.shape} distinta de la de los miniclips anteriores {self.shape}")
        path = self.tmp_dir / clip_filename(self.video_id, index)
        encode_clip(frames, path, fps, self.ff, self.crf, self.preset)
        self.sizes.append(path.stat().st_size)
        self.rates.append(float(fps))
        self.shape = frames.shape

    def commit(self, meta: dict[str, Any]) -> None:
        """Publica la carpeta de miniclips y escribe ``meta/<id>.json`` (con la sección ``salida``)."""
        if not self.sizes or self.shape is None:
            raise ValueError(f"{self.video_id}: no se escribió ningún miniclip")
        shutil.rmtree(self.final_dir, ignore_errors=True)
        legacy = self.layout.miniclips_dir / f"{self.video_id}.mp4"  # formato anterior: un archivo por video
        legacy.unlink(missing_ok=True)
        os.replace(self.tmp_dir, self.final_dir)
        k, (t, h, w, _) = len(self.sizes), self.shape
        vid = self.video_id
        meta["salida"] = {
            "carpeta": f"miniclips/{vid}",
            "formato": f"mp4 H.264 yuv420p (libx264, crf {self.crf}, preset {self.preset}), un archivo por miniclip",
            "miniclips": k,
            "cuadros_por_miniclip": t,
            "ancho": w,
            "alto": h,
            "fps_reproduccion": sorted({round(f, 4) for f in self.rates}),
            "bytes": int(sum(self.sizes)),
            "archivos": [f"miniclips/{vid}/{clip_filename(vid, i)}" for i in range(k)],
        }
        write_json_atomic(self.layout.meta_path(vid), meta)

    def abort(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def save_result(
    layout: OutputLayout,
    video_id: str,
    clips: np.ndarray,
    meta: dict[str, Any],
    ff: FfmpegInfo,
    *,
    fps: float | Sequence[float],
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Guarda un arreglo ``(K, T, H, W, 3)`` ya en memoria (ver ``ClipWriter`` para hacerlo miniclip por miniclip).

    ``fps`` es la tasa de reproducción, única o una por miniclip (25 / r).
    """
    clips = np.asarray(clips)
    if clips.ndim != 5 or clips.shape[-1] != 3 or len(clips) == 0:
        raise ValueError(f"se esperaba (K, T, H, W, 3) con K > 0, forma recibida {clips.shape}")
    rates = [float(fps)] * len(clips) if isinstance(fps, (int, float)) else [float(f) for f in fps]
    if len(rates) != len(clips):
        raise ValueError(f"se recibieron {len(rates)} tasas de reproducción para {len(clips)} miniclips")
    with ClipWriter(layout, video_id, ff, crf=crf, preset=preset) as writer:
        for i, clip in enumerate(clips):
            writer.write(i, clip, rates[i])
        writer.commit(meta)


def load_miniclip(path: str | Path, ff: FfmpegInfo | None = None) -> np.ndarray:
    """Lee un miniclip ``.mp4`` como ``(T, H, W, 3)`` uint8 RGB.

    Para leer muchos archivos, pase un ``ff`` obtenido una sola vez con ``check_ffmpeg()``.
    """
    ff = ff or check_ffmpeg()
    info = probe(path, ff)
    frames = read_video_frames(path, ff, info.width, info.height)
    if len(frames) == 0:
        raise ValueError(f"{path}: no tiene cuadros")
    return frames


def load_miniclips(clip_dir: str | Path, ff: FfmpegInfo | None = None) -> np.ndarray:
    """Lee todos los miniclips de ``miniclips/<id>/`` (en orden) como ``(K, T, H, W, 3)`` uint8 RGB."""
    ff = ff or check_ffmpeg()
    paths = sorted(Path(clip_dir).glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"no hay miniclips .mp4 en {clip_dir}")
    return np.stack([load_miniclip(p, ff) for p in paths])


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
        return True
    except ValueError:
        return False
