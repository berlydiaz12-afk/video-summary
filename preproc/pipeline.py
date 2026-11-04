from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from ._version import __version__
from .config import Config
from .config import config_hash as compute_config_hash
from .debug import prepare_debug_dir, write_debug_clip
from .detector import ShotDetector
from .ffmpeg_io import FfmpegInfo, decode_analysis, iter_extracted_frames, luma, probe
from .filtering import filter_frames
from .normalize import visible_region
from .redundancy import filter_redundant
from .selection import plan_miniclips
from .shots import candidate_windows, probs_to_shots, short_shot_windows
from .structs import Miniclip, VideoFailed, VideoInfo


def process_video(
    path: str | Path,
    rel_path: str,
    video_id: str,
    cfg: Config,
    detector: ShotDetector,
    ff: FfmpegInfo,
    *,
    threads: int = 0,
    config_hash: str | None = None,
    debug_dir: str | Path | None = None,
    clip_sink: ClipSink | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Aplica el resumen y el preprocesamiento a un video. Lanza ``VideoFailed`` si no hay cuadros útiles.

    ``clip_sink(k, cuadros, miniclip)`` recibe cada miniclip ``(T, S, S, 3)`` en orden;
    en ese caso el arreglo devuelto es ``None``.
    """
    sc = cfg.summary
    if sc.hamming_threshold is None:
        raise ValueError("configuración incompleta: falta θ_H")
    L, T = sc.window_length, sc.frames_per_clip
    times: dict[str, float] = {}
    warnings: list[str] = []
    t_total = perf_counter()

    # Paso 1: 25 fps constantes, pasada de análisis a baja resolución.
    t = perf_counter()
    info = probe(path, ff)
    analysis = decode_analysis(path, ff, cfg, info, threads)
    gray, rgb48 = analysis.gray, analysis.rgb48
    del analysis
    n1 = len(gray)
    if n1 == 0:
        raise VideoFailed(f"ffmpeg no devolvió cuadros a {cfg.fps} fps")
    times["paso1"] = perf_counter() - t

    # Paso 2: cuadros negros/uniformes y pantallas fijas → segmentos.
    t = perf_counter()
    filt = filter_frames(gray, cfg.filter)
    times["paso2"] = perf_counter() - t

    # Paso 3: TransNet V2 por segmento → tomas → ventanas candidatas.
    t = perf_counter()
    segments = filt.segments
    probs = detector.predict_segments([rgb48[s.start:s.end] for s in segments]) if segments else []
    del rgb48
    shots = probs_to_shots(segments, probs, cfg.transnet.threshold)
    candidates = candidate_windows(shots, L)
    times["paso3"] = perf_counter() - t

    # Paso 4: redundancia por pHash dentro de cada toma.
    t = perf_counter()
    red = filter_redundant(candidates, gray, sc)
    # Una toma de T a L−1 cuadros aporta una sola candidata (r = 1); al ser la única de su
    # toma, el paso 4 la conservaría siempre, así que se agrega directamente como conservada.
    short = short_shot_windows(shots, T, L) if sc.short_shot_windows else []
    all_candidates = candidates + short
    kept = list(red.kept) + list(range(len(candidates), len(all_candidates)))
    times["paso4"] = perf_counter() - t

    # Paso 5: todas las candidatas conservadas son miniclips (sin selección por movimiento).
    t = perf_counter()
    sel = plan_miniclips(all_candidates, kept, shots, gray, sc)
    times["paso5"] = perf_counter() - t

    # Paso 6: extracción miniclip por miniclip a size × size RGB uint8 (y escritura, vía clip_sink).
    t = perf_counter()
    region = visible_region(cfg.output, info.display_aspect)
    collected: list[np.ndarray] = []
    align_err, debug_time, debug_codec = 0.0, 0.0, cfg.debug.codec
    if debug_dir is not None:
        prepare_debug_dir(debug_dir)
    n_unique = len({i for c in sel.miniclips for i in c.frame_indices})
    for k, frames in iter_miniclip_frames(path, ff, cfg, info, sel.miniclips, threads):
        clip = sel.miniclips[k]
        idx = np.asarray(clip.frame_indices, dtype=np.int64)
        align_err = max(align_err, alignment_error(frames, gray[idx], region))
        if debug_dir is not None:
            t_dbg = perf_counter()
            debug_codec = write_debug_clip(frames, clip, k, sel.case, debug_dir, ff, cfg)
            debug_time += perf_counter() - t_dbg
        if clip_sink is not None:
            clip_sink(k, frames, clip)
        else:
            collected.append(frames)
    clips = np.stack(collected) if clip_sink is None else None
    if align_err > cfg.analysis.alignment_tolerance:
        warnings.append(
            f"posible desalineación entre pasadas: error {align_err:.2f} > tolerancia "
            f"{cfg.analysis.alignment_tolerance:.2f}"
        )
    if debug_dir is not None and debug_codec != cfg.debug.codec:
        warnings.append(f"diagnóstico codificado con {debug_codec}: el ffmpeg instalado no tiene {cfg.debug.codec}")
    times["paso6"] = perf_counter() - t - debug_time
    if debug_dir is not None:
        times["diagnostico"] = debug_time
    times["total"] = perf_counter() - t_total

    n2 = filt.n_kept
    n3 = sum(s.length for s in shots)
    width, height = info.display_size
    meta: dict[str, Any] = {
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
        "cuadros": {
            "paso1": n1,
            "negros_uniformes": filt.n_blank,
            "pantalla_fija": filt.n_static,
            "eliminados_paso2": filt.n_removed,
            "paso2": n2,
            "transicion": n2 - n3,
            "paso3": n3,
            "en_ventanas": sum(w.length for w in all_candidates),
            "en_ventanas_conservadas": sum(all_candidates[i].length for i in kept),
            "miniclips": len(sel.miniclips) * T,
            "miniclips_unicos": n_unique,
        },
        "conteos": {
            "segmentos": len(segments),
            "tomas": len(shots),
            "tomas_con_ventana": sum(1 for s in shots if s.length >= L),
            "tomas_cortas_candidatas": len(short),
            "tomas_sin_candidata": sum(1 for s in shots if s.length < (T if sc.short_shot_windows else L)),
            "ventanas_candidatas": len(all_candidates),
            "ventanas_conservadas": len(kept),
            "ventanas_descartadas": len(red.discarded),
        },
        "respaldo": {"caso": sel.case},
        "seleccion": {
            "modo": "todas_las_candidatas",
            "tomas_cortas": sc.short_shot_windows,
            "duracion_util_s": round(n3 / cfg.fps, 3),
            "miniclips": len(sel.miniclips),
        },
        "miniclips": [_miniclip_meta(k, c, cfg.fps) for k, c in enumerate(sel.miniclips)],
        "ventanas_descartadas": [
            {"ventana": d.candidate, "referencia": d.reference, "distancia": d.distance} for d in red.discarded
        ],
        "alineacion": {"error_max": align_err, "tolerancia": cfg.analysis.alignment_tolerance},
        "avisos": warnings,
        "tiempos_s": {k: round(v, 4) for k, v in times.items()},
        "forma": [len(sel.miniclips), T, cfg.output.size, cfg.output.size, 3],
        "version": __version__,
        "config_hash": config_hash or compute_config_hash(cfg),
        "ffmpeg": ff.version,
    }
    return clips, meta


ClipSink = Callable[[int, np.ndarray, Miniclip], None]


def iter_miniclip_frames(
    path: str | Path,
    ff: FfmpegInfo,
    cfg: Config,
    info: VideoInfo,
    miniclips: Sequence[Miniclip],
    threads: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Entrega ``(k, cuadros (T, S, S, 3))`` en el orden de ``miniclips``, sin tener el video completo en RAM.

    Una sola pasada de ffmpeg entrega los cuadros en orden creciente; cada cuadro
    se guarda solo hasta que lo usaron todos los miniclips que lo necesitan.
    """
    groups = [tuple(int(i) for i in c.frame_indices) for c in miniclips]
    pending = Counter(i for g in groups for i in g)
    cache: dict[int, np.ndarray] = {}
    k = 0
    for index, frame in iter_extracted_frames(path, ff, cfg, info, groups, threads):
        cache[index] = frame
        while k < len(groups) and all(i in cache for i in groups[k]):
            frames = np.stack([cache[i] for i in groups[k]])
            for i in groups[k]:
                pending[i] -= 1
                if pending[i] == 0:
                    del cache[i]
            yield k, frames
            k += 1
    if k != len(groups):
        raise VideoFailed(f"la extracción completó {k} de {len(groups)} miniclips")


def alignment_error(
    frames: np.ndarray,
    reference: np.ndarray,
    region: tuple[float, float, float, float],
    grid: int = 4,
) -> float:
    """Error máximo por cuadro entre los extraídos y el gris de análisis, en una rejilla ``grid × grid``.

    Es una verificación aproximada (las dos pasadas usan escaladores distintos):
    detecta índices corridos en contenido con movimiento, no diferencias finas.
    Convierte ``frames`` a float32 (4 veces su tamaño): llamarla por miniclip, no por video.
    """
    if len(frames) == 0:
        return 0.0
    h, w = reference.shape[1:]
    y0, y1, x0, x1 = region
    r0, c0 = int(round(y0 * h)), int(round(x0 * w))
    r1, c1 = max(int(round(y1 * h)), r0 + 1), max(int(round(x1 * w)), c0 + 1)
    ref = reference[:, r0:r1, c0:c1].astype(np.float32)
    ext = luma(frames).astype(np.float32)
    g = max(1, min(grid, ref.shape[1], ref.shape[2]))
    diff = np.abs(_grid_means(ext, g) - _grid_means(ref, g)).mean(axis=(1, 2))
    return float(diff.max())


def _grid_means(x: np.ndarray, g: int) -> np.ndarray:
    rows = np.array_split(np.arange(x.shape[1]), g)
    cols = np.array_split(np.arange(x.shape[2]), g)
    return np.stack(
        [np.stack([x[:, r[0]:r[-1] + 1, c[0]:c[-1] + 1].mean(axis=(1, 2)) for c in cols], axis=-1) for r in rows],
        axis=-2,
    )


def clip_stride(clip: Miniclip) -> int | None:
    """Separación constante entre los cuadros del miniclip (r), o None si no es constante (caso N = 0)."""
    steps = {b - a for a, b in zip(clip.frame_indices, clip.frame_indices[1:])}
    return steps.pop() if len(steps) == 1 else None


def _miniclip_meta(order: int, clip: Miniclip, fps: int) -> dict[str, Any]:
    return {
        "orden": order,
        "ventana": clip.candidate_index,
        "tipo": clip.kind,
        "cuadro_inicial": clip.start,
        "tiempo_inicial_s": round(clip.start / fps, 3),
        "r": clip_stride(clip),
        "toma": clip.shot,
        "movimiento": clip.motion,
        "respaldo": clip.fallback,
        "cuadros": list(clip.frame_indices),
    }
