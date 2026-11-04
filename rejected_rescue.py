"""Rescue of the videos rejected by the full pipeline (``preproc.run``).

A video is rejected when ``errores_<dataset>.log`` lists it and it has no ``meta/<id>.json``.
Typical reasons: every frame removed by frame filtering or shot detection
("ningún cuadro útil tras los pasos 2 y 3"), a corrupt stream that makes ffmpeg exit
with an error, or an extraction that returned fewer frames than expected.

The rescue skips frame filtering, shot detection and redundancy removal (they are what
rejected the video) and keeps everything else from the pipeline:

* decoding: same ``fps`` timeline and analysis gray (``config.yaml``), but ffmpeg ignores
  corrupt packets and the frames decoded before an ffmpeg error are kept;
* clip planning: consecutive non-overlapping windows of L = T·r frames over the whole
  timeline (T frames with stride r, like a regular candidate). A video of T to L−1 frames
  gives T consecutive centred frames (r = 1, like a short shot); a shorter one gives T
  uniformly spaced frames (repeated in order, like the N = 0 fallback);
* swin_input: same ``fps → select → scale`` filter, size, resize mode and sws flags;
* storage: same ``ClipWriter`` (H.264 yuv420p, ``output.crf`` / ``output.preset``), same
  ``miniclips/<id>/<id>_000.mp4`` layout and a ``meta/<id>.json`` with
  ``respaldo.caso = "rescate"``.

Frame counts of the rescue go under ``rescate`` (not ``cuadros``), so the reduction
percentages in ``estadisticas_<dataset>.json`` still describe only the full pipeline.

Usage from Jupyter / Colab (with ``preprocesamiento/`` as working directory or on ``sys.path``)::

    from preproc import load_config
    from rejected_rescue import list_rejected_videos, rescue_rejected_videos

    cfg = load_config("config.yaml")
    result = rescue_rejected_videos("D:/datos/pornography2k", "pornography2k", "salida", cfg)
    result["rescatados"], result["fallidos"]
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from tqdm import tqdm

from preproc import __version__
from preproc.config import Config, config_hash
from preproc.ffmpeg_io import (
    FfmpegError,
    FfmpegInfo,
    _iter_raw_frames,
    _thread_args,
    build_select_expr,
    check_ffmpeg,
    luma,
    probe,
)
from preproc.normalize import scale_filter
from preproc.pipeline import clip_stride
from preproc.reporting import LOGGER_NAME, append_error, build_summary, read_errors, setup_logging
from preproc.selection import clip_motion, miniclip_indices, uniform_indices
from preproc.storage import ClipWriter, OutputLayout
from preproc.structs import Miniclip, VideoFailed, VideoInfo

RESCUE_CASE = "rescate"
KIND_RESCUE_WINDOW = "rescate_ventana"
KIND_RESCUE_SHORT = "rescate_corto"
KIND_RESCUE_UNIFORM = "rescate_uniforme"

# Input options that keep decoding past corrupt packets and rebuild missing timestamps.
TOLERANT_INPUT_ARGS = ["-err_detect", "ignore_err", "-fflags", "+genpts+discardcorrupt"]


def list_rejected_videos(layout: OutputLayout) -> list[dict[str, str]]:
    """Videos in ``errores_<dataset>.log`` without ``meta/<id>.json`` (last reason per id)."""
    return [e for vid, e in sorted(read_errors(layout).items()) if not layout.meta_path(vid).is_file()]


def rescue_rejected_videos(
    input_dir: str | Path,
    dataset: str,
    output_dir: str | Path,
    cfg: Config,
    *,
    ids: Iterable[str] | None = None,
    max_clips: int | None = None,
    threads: int = 0,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Rescues every rejected video of ``output_dir/dataset`` (or only ``ids``) and rebuilds the summary.

    ``input_dir`` must be the same folder given to ``preproc.run`` (the log stores relative paths).
    ``max_clips`` limits the clips per video, spread evenly over the timeline (None = all windows).
    """
    ff = check_ffmpeg(cfg.ffmpeg.ffmpeg, cfg.ffmpeg.ffprobe)
    if not ff.has_libx264:
        raise FfmpegError(f"el ffmpeg {ff.version} ({ff.ffmpeg}) no tiene libx264, necesario para los miniclips")
    layout = OutputLayout.create(output_dir, dataset)
    log = setup_logging(layout, log_level)
    root = Path(input_dir).resolve()
    chash = config_hash(cfg)

    rejected = list_rejected_videos(layout)
    if ids is not None:
        wanted = {i.strip() for i in ids if i.strip()}
        rejected = [e for e in rejected if e["id"] in wanted]
    log.info("rescate: %d videos rechazados por el flujo completo", len(rejected))

    rescued: list[str] = []
    failed: dict[str, str] = {}
    for entry in tqdm(rejected, unit="video", dynamic_ncols=True):
        vid, rel = entry["id"], entry["ruta"]
        try:
            path = root / rel
            if not path.is_file():
                raise VideoFailed(f"no existe el archivo {path}")
            meta = rescue_video(path, rel, vid, entry["motivo"], cfg, ff, layout,
                                max_clips=max_clips, threads=threads, chash=chash)
            rescued.append(vid)
            log.info("rescatado %s: %d miniclips (motivo original: %s)",
                     vid, meta["seleccion"]["miniclips"], entry["motivo"])
            for warning in meta["avisos"]:
                log.warning("%s: %s", vid, warning)
        except Exception as exc:  # noqa: BLE001 - one failed rescue does not stop the rest
            reason = str(exc) if isinstance(exc, VideoFailed) else f"{type(exc).__name__}: {exc}"
            failed[vid] = reason
            append_error(layout, vid, rel, f"rescate fallido: {reason}")
            log.error("rescate fallido %s: %s", vid, reason)

    stats = build_summary(layout)
    log.info("rescate terminado: %d rescatados, %d fallidos", len(rescued), len(failed))
    return {"rescatados": rescued, "fallidos": failed, "estadisticas": stats}


def rescue_video(
    path: str | Path,
    rel_path: str,
    video_id: str,
    rejection_reason: str,
    cfg: Config,
    ff: FfmpegInfo,
    layout: OutputLayout,
    *,
    max_clips: int | None = None,
    threads: int = 0,
    chash: str | None = None,
) -> dict[str, Any]:
    """Decoding → clip planning → swin_input → storage for one video. Returns its ``meta``.

    Raises ``VideoFailed`` when ffprobe cannot read the file or ffmpeg returns no frame.
    """
    t_total = perf_counter()
    ffmpeg_notes: list[str] = []
    info = probe(path, ff)

    t = perf_counter()
    gray = decode_timeline(path, ff, cfg, threads, ffmpeg_notes)
    if len(gray) == 0:
        raise VideoFailed(f"ffmpeg no devolvió cuadros a {cfg.fps} fps ni en modo tolerante: {ffmpeg_notes}")
    t_decoding = perf_counter() - t

    clips, n_windows = plan_rescue_clips(gray, cfg, max_clips)

    t = perf_counter()
    written: list[Miniclip] = []
    with ClipWriter(layout, video_id, ff, crf=cfg.output.crf, preset=cfg.output.preset) as writer:
        for k, frames in iter_rescue_clips(path, ff, cfg, info, clips, threads, ffmpeg_notes):
            writer.write(k, frames, cfg.fps / (clip_stride(clips[k]) or cfg.summary.stride))
            written.append(clips[k])
        if not written:
            raise VideoFailed(f"la extracción no completó ningún miniclip: {ffmpeg_notes}")
        times = {"decodificacion": t_decoding, "extraccion": perf_counter() - t,
                 "total": perf_counter() - t_total}
        meta = rescue_meta(video_id, rel_path, rejection_reason, cfg, ff, info, len(gray),
                           written, len(clips), n_windows, max_clips, ffmpeg_notes, times, chash)
        writer.commit(meta)
    return meta


def decode_timeline(
    path: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
    threads: int = 0,
    notes: list[str] | None = None,
) -> np.ndarray:
    """Decoding: the whole video at ``cfg.fps`` as BT.601 gray of analysis size ``(n, h, w)`` uint8.

    Same ``fps`` and scaler as ``decode_analysis``, so indices match the extraction pass.
    """
    a = cfg.analysis
    vf = f"fps={cfg.fps},scale={a.gray_width}:{a.gray_height}:flags={a.gray_sws_flags},format=rgb24"
    cmd = [ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error",
           *_thread_args(threads, complex_graph=False), *TOLERANT_INPUT_ARGS,
           "-i", str(path), "-an", "-sn", "-dn", "-vf", vf,
           *ff.passthrough_args, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    shape = (a.gray_height, a.gray_width, 3)
    chunks = [luma(c) for c in _salvage_frames(cmd, shape, 512, notes)]
    if not chunks:
        return np.empty((0, a.gray_height, a.gray_width), dtype=np.uint8)
    return np.concatenate(chunks, axis=0)


def plan_rescue_clips(
    gray: np.ndarray,
    cfg: Config,
    max_clips: int | None = None,
) -> tuple[list[Miniclip], int]:
    """Clip planning over the whole timeline. Returns ``(clips, total windows before max_clips)``."""
    n = len(gray)
    T, r = cfg.summary.frames_per_clip, cfg.summary.stride
    L = T * r
    starts = list(range(0, n - L + 1, L))
    if not starts:
        if n >= T:
            idx = miniclip_indices((n - T) // 2, T, 1)
            kind = KIND_RESCUE_SHORT
        else:
            idx = uniform_indices(0, n, T)
            kind = KIND_RESCUE_UNIFORM
        return [Miniclip(tuple(idx.tolist()), 0, kind, None, clip_motion(gray, idx), True)], 0

    n_windows = len(starts)
    if max_clips is not None and max_clips > 0 and n_windows > max_clips:
        keep = np.unique(np.linspace(0, n_windows - 1, max_clips).round().astype(int))
        starts = [starts[i] for i in keep]
    clips = []
    for s in starts:
        idx = miniclip_indices(s, T, r)
        clips.append(Miniclip(tuple(idx.tolist()), 0, KIND_RESCUE_WINDOW, s // L, clip_motion(gray, idx), True))
    return clips, n_windows


def iter_rescue_clips(
    path: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
    info: VideoInfo,
    clips: Sequence[Miniclip],
    threads: int = 0,
    notes: list[str] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """swin_input: yields ``(k, frames (T, S, S, 3))`` in order; stops quietly if ffmpeg ends early.

    Same filter as ``iter_extracted_frames`` (``fps → select → scale``) with the tolerant input options.
    """
    groups = [tuple(c.frame_indices) for c in clips]
    unique = sorted({i for g in groups for i in g})
    size = cfg.output.size
    vf = (f"fps={cfg.fps},select='{build_select_expr(groups)}',"
          f"{scale_filter(cfg.output, info.display_aspect)},format=rgb24")
    cmd = [ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error",
           *_thread_args(threads, complex_graph=False), *TOLERANT_INPUT_ARGS,
           "-i", str(path), "-an", "-sn", "-dn", "-vf", vf,
           *ff.passthrough_args, "-frames:v", str(len(unique)),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    pending = Counter(i for g in groups for i in g)
    cache: dict[int, np.ndarray] = {}
    got = k = 0
    for chunk in _salvage_frames(cmd, (size, size, 3), 64, notes):
        for frame in chunk:
            if got == len(unique):
                break
            cache[unique[got]] = frame
            got += 1
            while k < len(groups) and all(i in cache for i in groups[k]):
                frames = np.stack([cache[i] for i in groups[k]])
                for i in groups[k]:
                    pending[i] -= 1
                    if pending[i] == 0:
                        del cache[i]
                yield k, frames
                k += 1
    if k < len(groups) and notes is not None:
        notes.append(f"la extracción completó {k} de {len(groups)} miniclips ({got} de {len(unique)} cuadros)")


def rescue_meta(
    video_id: str,
    rel_path: str,
    rejection_reason: str,
    cfg: Config,
    ff: FfmpegInfo,
    info: VideoInfo,
    n_frames: int,
    written: Sequence[Miniclip],
    n_planned: int,
    n_windows: int,
    max_clips: int | None,
    ffmpeg_notes: Sequence[str],
    times: dict[str, float],
    chash: str | None,
) -> dict[str, Any]:
    """``meta/<id>.json`` with the keys of the pipeline that apply, plus the ``rescate`` section."""
    T = cfg.summary.frames_per_clip
    width, height = info.display_size
    warnings = [f"video rescatado (rechazado por: {rejection_reason})", *ffmpeg_notes]
    return {
        "id": video_id,
        "ruta": rel_path,
        "original": {
            "fps": info.fps,
            "duracion_s": info.duration,
            "ancho": width,
            "alto": height,
            "rotacion": info.rotation,
            "sar": info.sar,
            "codec": info.codec,
        },
        "respaldo": {"caso": RESCUE_CASE},
        "seleccion": {
            "modo": "rescate_ventanas_fijas",
            "tomas_cortas": False,
            "duracion_util_s": round(n_frames / cfg.fps, 3),
            "miniclips": len(written),
        },
        "rescate": {
            "motivo_rechazo": rejection_reason,
            "cuadros_linea_temporal": n_frames,
            "ventanas_totales": n_windows,
            "miniclips_planeados": n_planned,
            "max_miniclips": max_clips,
            "cuadros_miniclips_unicos": len({i for c in written for i in c.frame_indices}),
            "entrada_tolerante": " ".join(TOLERANT_INPUT_ARGS),
        },
        "miniclips": [
            {
                "orden": k,
                "ventana": c.candidate_index,
                "tipo": c.kind,
                "cuadro_inicial": c.start,
                "tiempo_inicial_s": round(c.start / cfg.fps, 3),
                "r": clip_stride(c),
                "toma": None,
                "movimiento": c.motion,
                "respaldo": True,
                "cuadros": list(c.frame_indices),
            }
            for k, c in enumerate(written)
        ],
        "avisos": warnings,
        "tiempos_s": {key: round(v, 4) for key, v in times.items()},
        "forma": [len(written), T, cfg.output.size, cfg.output.size, 3],
        "version": __version__,
        "config_hash": chash or config_hash(cfg),
        "ffmpeg": ff.version,
    }


def _salvage_frames(
    cmd: list[str],
    frame_shape: tuple[int, ...],
    chunk_frames: int,
    notes: list[str] | None,
) -> Iterator[np.ndarray]:
    """``_iter_raw_frames`` that keeps the frames read before ffmpeg exits with an error."""
    try:
        yield from _iter_raw_frames(cmd, frame_shape, chunk_frames)
    except VideoFailed as exc:
        if notes is not None:
            notes.append(str(exc))
        logging.getLogger(LOGGER_NAME).debug("ffmpeg terminó con error, se conservan los cuadros leídos: %s", exc)
