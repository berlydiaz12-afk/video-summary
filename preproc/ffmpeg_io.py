from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np

from .config import Config
from .normalize import scale_filter
from .structs import VideoFailed, VideoInfo

TRANSNET_WIDTH = 48
TRANSNET_HEIGHT = 27
_BT601_TAGS = ["-colorspace", "smpte170m", "-color_primaries", "smpte170m",
               "-color_trc", "smpte170m", "-color_range", "tv"]


class FfmpegError(RuntimeError):
    """ffmpeg/ffprobe no está disponible o se comporta de forma inesperada."""


@dataclass(frozen=True)
class FfmpegInfo:
    """Binarios encontrados y capacidades relevantes de ffmpeg."""

    ffmpeg: str
    ffprobe: str
    version: str
    version_tuple: tuple[int, int] | None
    has_libx264: bool

    @property
    def passthrough_args(self) -> list[str]:
        """Evita que ffmpeg duplique o descarte cuadros a la salida (imprescindible tras ``select``)."""
        if self.version_tuple is None or self.version_tuple >= (5, 1):
            return ["-fps_mode", "passthrough"]
        return ["-vsync", "passthrough"]


@dataclass
class AnalysisFrames:
    """Cuadros de la pasada de análisis sobre la línea de tiempo a 25 fps."""

    gray: np.ndarray   # (n, gray_height, gray_width) uint8, luminancia BT.601
    rgb48: np.ndarray  # (n, 27, 48, 3) uint8, entrada de TransNet V2


def check_ffmpeg(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> FfmpegInfo:
    """Localiza ffmpeg/ffprobe, lee la versión y comprueba si está ``libx264``."""
    found: dict[str, str] = {}
    for key, exe in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        path = shutil.which(exe)
        if path is None:
            raise FfmpegError(
                f"No se encontró '{exe}'. Instale ffmpeg (https://ffmpeg.org/download.html), "
                f"agréguelo al PATH o indique la ruta en config.yaml (ffmpeg.{key})."
            )
        found[key] = path
    out = _run_text([found["ffmpeg"], "-hide_banner", "-version"])
    match = re.match(r"ffmpeg version (\S+)", out.strip())
    version = match.group(1) if match else "desconocida"
    nums = re.match(r"n?(\d+)\.(\d+)", version)
    version_tuple = (int(nums.group(1)), int(nums.group(2))) if nums else None
    if version_tuple is not None and version_tuple < (4, 0):
        raise FfmpegError(f"Se requiere ffmpeg >= 4.0 (se encontró {version}).")
    encoders = _run_text([found["ffmpeg"], "-hide_banner", "-encoders"])
    has_x264 = re.search(r"\blibx264\b", encoders) is not None
    return FfmpegInfo(found["ffmpeg"], found["ffprobe"], version, version_tuple, has_x264)


def probe(path: str | Path, ff: FfmpegInfo) -> VideoInfo:
    """Metadatos originales de la primera pista de video (tasa, duración, resolución, rotación)."""
    cmd = [ff.ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_streams", "-show_format", "-of", "json", str(path)]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise VideoFailed(f"ffprobe no pudo leer el archivo: {_tail(res.stderr)}")
    try:
        data = json.loads(res.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoFailed(f"salida de ffprobe inválida: {exc}") from None
    streams = data.get("streams") or []
    if not streams:
        raise VideoFailed("el archivo no tiene pista de video")
    st, fmt = streams[0], data.get("format") or {}
    width, height = int(st.get("width") or 0), int(st.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoFailed("resolución de video desconocida")
    nb = st.get("nb_frames")
    return VideoInfo(
        path=str(path),
        codec=st.get("codec_name"),
        width=width,
        height=height,
        sar=_parse_ratio(st.get("sample_aspect_ratio"), sep=":") or 1.0,
        rotation=_rotation(st),
        fps=_parse_ratio(st.get("avg_frame_rate")) or _parse_ratio(st.get("r_frame_rate")),
        duration=_parse_float(fmt.get("duration")) or _parse_float(st.get("duration")),
        nb_frames=int(nb) if str(nb).isdigit() else None,
    )


def analysis_filtergraph(cfg: Config) -> tuple[str, tuple[int, int, int]]:
    """Grafo de la pasada de análisis y forma de cada cuadro compuesto.

    Una sola decodificación produce, apiladas verticalmente, la imagen de
    ``gray_width × gray_height`` (pasos 2, 4 y 5) y la de 48 × 27 para TransNet V2.
    """
    a = cfg.analysis
    width = max(a.gray_width, TRANSNET_WIDTH)
    graph = (
        f"[0:v]fps={cfg.fps},split=2[g][t];"
        f"[g]scale={a.gray_width}:{a.gray_height}:flags={a.gray_sws_flags},format=rgb24,"
        f"pad={width}:{a.gray_height}:0:0[g2];"
        f"[t]scale={TRANSNET_WIDTH}:{TRANSNET_HEIGHT}:flags={a.transnet_sws_flags},format=rgb24,"
        f"pad={width}:{TRANSNET_HEIGHT}:0:0[t2];"
        "[g2][t2]vstack=inputs=2[out]"
    )
    return graph, (a.gray_height + TRANSNET_HEIGHT, width, 3)


def decode_analysis(
    path: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
    info: VideoInfo | None = None,
    threads: int = 0,
    chunk_frames: int = 512,
) -> AnalysisFrames:
    """Paso 1: decodifica todo el video a ``fps`` y baja resolución, en *streaming*."""
    graph, shape = analysis_filtergraph(cfg)
    gh, gw = cfg.analysis.gray_height, cfg.analysis.gray_width
    cmd = [ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error",
           *_thread_args(threads, complex_graph=True),
           "-i", str(path), "-an", "-sn", "-dn",
           "-filter_complex", graph, "-map", "[out]",
           *ff.passthrough_args, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    capacity = _estimate_frames(info, cfg.fps)
    gray = _GrowableArray((gh, gw), capacity)
    rgb = _GrowableArray((TRANSNET_HEIGHT, TRANSNET_WIDTH, 3), capacity)
    for chunk in _iter_raw_frames(cmd, shape, chunk_frames):
        gray.extend(luma(chunk[:, :gh, :gw]))
        rgb.extend(chunk[:, gh:, :TRANSNET_WIDTH])
    return AnalysisFrames(gray.array(), rgb.array())


def build_select_expr(groups: Sequence[Sequence[int]]) -> str:
    """Expresión del filtro ``select`` que deja pasar exactamente los índices pedidos.

    Una progresión aritmética (miniclip normal) usa un solo término
    ``between(n,a,b)*not(mod(n-a,r))``; cualquier otro grupo usa términos ``eq(n,i)``.
    """
    terms: list[str] = []
    for group in groups:
        for term in _group_terms([int(i) for i in group]):
            if term not in terms:
                terms.append(term)
    if not terms:
        raise ValueError("no hay cuadros que extraer")
    return f"gt({'+'.join(terms)},0)"


def extract_frames(
    path: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
    info: VideoInfo,
    groups: Sequence[Sequence[int]],
    threads: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Paso 6: decodifica solo los cuadros pedidos, ya redimensionados, en RGB uint8.

    Devuelve ``(cuadros, índices)``, donde ``cuadros[i]`` corresponde a ``índices[i]``
    (los índices únicos, ordenados).
    """
    unique = sorted({int(i) for g in groups for i in g})
    size = cfg.output.size
    frames = np.empty((len(unique), size, size, 3), dtype=np.uint8)
    for pos, (_, frame) in enumerate(iter_extracted_frames(path, ff, cfg, info, groups, threads)):
        frames[pos] = frame
    return frames, unique


def iter_extracted_frames(
    path: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
    info: VideoInfo,
    groups: Sequence[Sequence[int]],
    threads: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Paso 6 en *streaming*: entrega ``(índice, cuadro)`` de los cuadros pedidos, en orden creciente.

    Cada cuadro es RGB uint8 ``(size, size, 3)``; en memoria solo hay un bloque de
    64 cuadros a la vez. Lanza ``VideoFailed`` si ffmpeg devuelve otra cantidad.
    """
    unique = sorted({int(i) for g in groups for i in g})
    expr = build_select_expr(groups)
    size = cfg.output.size
    vf = f"fps={cfg.fps},select='{expr}',{scale_filter(cfg.output, info.display_aspect)},format=rgb24"
    cmd = [ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error",
           *_thread_args(threads, complex_graph=False),
           "-i", str(path), "-an", "-sn", "-dn", "-vf", vf,
           *ff.passthrough_args, "-frames:v", str(len(unique)),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    got = 0
    for chunk in _iter_raw_frames(cmd, (size, size, 3), 64):
        if got + len(chunk) > len(unique):
            raise VideoFailed(f"la extracción devolvió más cuadros de los {len(unique)} esperados")
        for frame in chunk:
            yield unique[got], frame
            got += 1
    if got != len(unique):
        raise VideoFailed(f"la extracción devolvió {got} cuadros de {len(unique)} esperados")


def encode_video(
    frames: np.ndarray,
    out_path: str | Path,
    fps: float,
    ff: FfmpegInfo,
    codec: str = "libx264",
    crf: int = 18,
) -> str:
    """Codifica cuadros RGB uint8 ``(n, H, W, 3)`` como .mp4 con pérdida (yuv420p, para verlos). Devuelve el códec."""
    frames = _as_rgb_frames(frames)
    _, h, w, _ = frames.shape
    if codec == "libx264" and not ff.has_libx264:
        codec = "mpeg4"
    if codec == "libx264":
        enc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
    else:
        enc = ["-c:v", codec, "-q:v", "2"]
    cmd = [ff.ffmpeg, "-hide_banner", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{w}x{h}",
           "-framerate", _rate(fps), "-i", "pipe:0",
           "-an", *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    _run_encoder(cmd, frames, out_path)
    return codec


def encode_clip(
    frames: np.ndarray,
    out_path: str | Path,
    fps: float,
    ff: FfmpegInfo,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Codifica un miniclip RGB uint8 ``(T, H, W, 3)`` como .mp4 H.264 ``yuv420p`` (reproducible en cualquier lado).

    La conversión RGB → YUV usa BT.601 en rango limitado, y el archivo lo declara
    para que reproductores y decodificadores la deshagan igual. El formato se fija
    con ``-f mp4`` para poder escribir a un archivo temporal.
    """
    if not ff.has_libx264:
        raise FfmpegError("el ffmpeg instalado no tiene el codificador libx264, necesario para guardar los miniclips")
    frames = _as_rgb_frames(frames)
    _, h, w, _ = frames.shape
    cmd = [ff.ffmpeg, "-hide_banner", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{w}x{h}",
           "-framerate", _rate(fps), "-i", "pipe:0",
           "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           *_BT601_TAGS, "-movflags", "+faststart", "-f", "mp4", str(out_path)]
    _run_encoder(cmd, frames, out_path)


def read_video_frames(path: str | Path, ff: FfmpegInfo, width: int, height: int) -> np.ndarray:
    """Decodifica todos los cuadros de un video como RGB uint8 ``(n, height, width, 3)``, sin remuestrear."""
    chunks = list(_iter_raw_frames(_raw_decode_cmd(path, ff), (height, width, 3), 256))
    if not chunks:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    return np.concatenate(chunks, axis=0)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Luminancia BT.601 (uint8, 0–255) de cuadros RGB uint8 ``(..., 3)``.

    Se calcula desde RGB en lugar de usar ``-pix_fmt gray``, que puede conservar
    el rango limitado (negro = 16) y desplazar θ_L.
    """
    x = rgb.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- internos

def _as_rgb_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"se esperaban cuadros (n, H, W, 3) con n > 0, forma recibida {frames.shape}")
    return frames


def _rate(fps: float) -> str:
    rate = Fraction(fps).limit_denominator(1001)
    return f"{rate.numerator}/{rate.denominator}"


def _run_encoder(cmd: list[str], frames: np.ndarray, out_path: str | Path) -> None:
    res = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise FfmpegError(f"no se pudo codificar {out_path}: {_tail(res.stderr.decode('utf-8', 'replace'))}")


def _raw_decode_cmd(path: str | Path, ff: FfmpegInfo) -> list[str]:
    return [ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
            "-an", "-sn", "-dn", *ff.passthrough_args, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]


def _iter_raw_frames(cmd: list[str], frame_shape: tuple[int, ...], chunk_frames: int) -> Iterator[np.ndarray]:
    """Ejecuta ffmpeg y entrega bloques de cuadros ``(m, *frame_shape)`` leídos de la tubería."""
    frame_bytes = int(np.prod(frame_shape))
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        try:
            assert proc.stdout is not None
            while True:
                data = _read_exact(proc.stdout, frame_bytes * chunk_frames)
                if not data:
                    break
                if len(data) % frame_bytes:
                    raise FfmpegError("la salida de ffmpeg terminó a mitad de un cuadro")
                yield np.frombuffer(data, dtype=np.uint8).reshape((-1, *frame_shape))
            proc.stdout.close()
            code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if code != 0:
            err.seek(0)
            raise VideoFailed(f"ffmpeg terminó con código {code}: {_tail(err.read().decode('utf-8', 'replace'))}")


def _read_exact(stream: BinaryIO, n: int) -> bytearray:
    """Lee ``n`` bytes, o menos solo si se llega al final del flujo."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = stream.readinto(view[got:])  # type: ignore[attr-defined]
        if not k:
            break
        got += k
    view.release()
    del buf[got:]
    return buf


class _GrowableArray:
    """Arreglo que crece por bloques (evita concatenar listas de cuadros)."""

    def __init__(self, item_shape: tuple[int, ...], capacity: int, dtype: type = np.uint8) -> None:
        self._data = np.empty((max(1, capacity), *item_shape), dtype=dtype)
        self._n = 0

    def extend(self, items: np.ndarray) -> None:
        need = self._n + len(items)
        if need > len(self._data):
            grown = np.empty((max(need, int(len(self._data) * 1.5) + 1), *self._data.shape[1:]),
                             dtype=self._data.dtype)
            grown[: self._n] = self._data[: self._n]
            self._data = grown
        self._data[self._n:need] = items
        self._n = need

    def array(self) -> np.ndarray:
        out = self._data[: self._n]
        return out.copy() if self._n < 0.9 * len(self._data) else out


def _thread_args(threads: int, complex_graph: bool) -> list[str]:
    if threads <= 0:
        return []
    option = "-filter_complex_threads" if complex_graph else "-filter_threads"
    return ["-threads", str(threads), option, str(threads)]


def _estimate_frames(info: VideoInfo | None, fps: int) -> int:
    if info is not None and info.duration:
        return int(info.duration * fps) + fps + 16
    return 4096


def _group_terms(idx: list[int]) -> list[str]:
    if not idx:
        return []
    if len(idx) > 1:
        steps = {b - a for a, b in zip(idx, idx[1:])}
        if len(steps) == 1:
            step = steps.pop()
            if step == 1:
                return [f"between(n,{idx[0]},{idx[-1]})"]
            if step > 1:
                return [f"between(n,{idx[0]},{idx[-1]})*not(mod(n-{idx[0]},{step}))"]
    return [f"eq(n,{i})" for i in sorted(set(idx))]


def _run_text(cmd: list[str]) -> str:
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise FfmpegError(f"falló '{' '.join(cmd)}': {_tail(res.stderr)}")
    return res.stdout


def _tail(text: str, lines: int = 12) -> str:
    out = " | ".join(line for line in (text or "").strip().splitlines()[-lines:] if line.strip())
    return out or "(sin mensaje)"


def _parse_ratio(value: object, sep: str = "/") -> float | None:
    if not isinstance(value, str) or sep not in value:
        return None
    num, den = value.split(sep, 1)
    try:
        num_f, den_f = float(num), float(den)
    except ValueError:
        return None
    if num_f <= 0 or den_f <= 0:
        return None
    return num_f / den_f


def _parse_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f > 0 and np.isfinite(f) else None


def _rotation(stream: dict) -> int:
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (TypeError, ValueError):
            pass
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    return 0
