"""Full pipeline without frame filtering (step 2) for the videos rejected by ``preproc.run``.

Targets the videos whose last error is "ningún cuadro útil tras los pasos 2 y 3" and that
still have no ``meta/<id>.json``. They go through the same code as ``preproc.run``
(``runner._process_one`` → ``process_video`` → ``ClipWriter``), so decoding, TransNet V2,
candidate windows, redundancy removal, extraction, the ``.mp4`` files and the JSON keys
are exactly those of the original pipeline.

Step 2 is skipped by setting θ_L = θ_σ = θ_E = 0: frame means, standard deviations and
frame differences are never negative, so no frame is marked as black, uniform or static,
and TransNet V2 receives the whole video as a single segment.

The JSON keeps the ``config_hash`` of the original configuration, so a later
``preproc.run`` skips these videos instead of rejecting them again, and gets a
``paso2_omitido`` section with the original thresholds.

Usage from Jupyter / Colab (with ``preprocesamiento/`` as working directory or on ``sys.path``)::

    from preproc import load_config
    from rejected_without_filtering import run_without_frame_filtering

    cfg = load_config("config.yaml")
    # Rejected videos taken from errores_<dataset>.log of the original run:
    result = run_without_frame_filtering("D:/datos/pornography2k", "pornography2k", "salida", cfg, gpu=0)
    # Or every video of a folder that holds only the rejected ones (no error log needed):
    result = run_folder_without_frame_filtering("D:/datos/reject", "pornography2k", "salida", cfg, gpu=0)
"""

from __future__ import annotations

import copy
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from preproc.config import Config, config_hash, validate
from preproc.ffmpeg_io import FfmpegError, check_ffmpeg
from preproc.reporting import build_summary, read_errors, setup_logging
from preproc.runner import Job, WorkerSetup, _run_inline, _run_pool
from preproc.storage import OutputLayout, assign_ids, discover_videos, is_done, write_json_atomic

REJECTION_REASON = "ningún cuadro útil tras los pasos 2 y 3"
FOLDER_REASON = "carpeta de videos rechazados (motivo no registrado)"


def run_folder_without_frame_filtering(
    input_dir: str | Path,
    dataset: str,
    output_dir: str | Path,
    cfg: Config,
    *,
    workers: int = 0,
    gpu: int = 0,
    debug: bool = False,
    force: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Processes every video of ``input_dir`` without step 2, like ``preproc.run`` on that folder.

    Ids come from the paths relative to ``input_dir`` (``vPorn000123.mp4`` → ``vPorn000123``),
    as in ``preproc.run``: with a flat folder they match the ids of the original run. Outputs go
    to ``output_dir/dataset`` (created if missing); videos already done are skipped unless ``force``.
    """
    ff = check_ffmpeg(cfg.ffmpeg.ffmpeg, cfg.ffmpeg.ffprobe)
    if not ff.has_libx264:
        raise FfmpegError(f"el ffmpeg {ff.version} ({ff.ffmpeg}) no tiene libx264, necesario para los miniclips")
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de videos: {root}")
    layout = OutputLayout.create(output_dir, dataset)
    log = setup_logging(layout, log_level)
    chash = config_hash(cfg)

    videos = discover_videos(root, cfg.ffmpeg.extensions, exclude=[layout.root])
    if not videos:
        raise LookupError(f"no hay videos con extensiones {cfg.ffmpeg.extensions} en {root}")
    ids = assign_ids([rel for _, rel in videos])
    jobs = [Job(str(path), rel, vid, debug) for (path, rel), vid in zip(videos, ids)
            if force or debug or not is_done(layout, vid, chash)]
    log.info("sin paso 2: %d videos en %s, %d pendientes", len(videos), root, len(jobs))
    return _process_jobs(jobs, {j.video_id: FOLDER_REASON for j in jobs}, cfg, ff, layout, log, workers, gpu)


def without_frame_filtering(cfg: Config) -> Config:
    """Copy of ``cfg`` where frame filtering keeps every frame (θ_L = θ_σ = θ_E = 0)."""
    out = copy.deepcopy(cfg)
    out.filter.luma_threshold = 0.0
    out.filter.std_threshold = 0.0
    out.filter.static_threshold = 0.0
    validate(out)
    return out


def existing_layout(output_dir: str | Path, dataset: str) -> OutputLayout:
    """Layout of a previous ``preproc.run``; unlike ``OutputLayout.create`` it never creates folders."""
    layout = OutputLayout(Path(output_dir).resolve() / dataset, dataset)
    if not layout.root.is_dir():
        raise FileNotFoundError(f"no existe la salida de la corrida original: {layout.root} "
                                "(output_dir es la carpeta que contiene <dataset>/)")
    if not layout.errors_log.is_file():
        raise FileNotFoundError(f"no existe {layout.errors_log}: la corrida original no registró videos fallidos")
    layout.logs_dir.mkdir(exist_ok=True)
    return layout


def list_rejected_by_filtering(layout: OutputLayout) -> list[dict[str, str]]:
    """Videos whose last error is ``REJECTION_REASON`` and have no ``meta/<id>.json``."""
    return [
        e for vid, e in sorted(read_errors(layout).items())
        if _is_rejection(e["motivo"]) and not layout.meta_path(vid).is_file()
    ]


def explain_empty_selection(layout: OutputLayout, ids: Iterable[str] | None = None) -> str:
    """Why no video was selected: every id in the error log with its last reason and whether it has a JSON."""
    errors = read_errors(layout)
    lines = [f"ningún video para procesar en {layout.root} ({len(errors)} ids en {layout.errors_log.name}):"]
    for vid, e in sorted(errors.items()):
        state = "ya tiene meta/<id>.json" if layout.meta_path(vid).is_file() else "sin JSON"
        lines.append(f"  {vid}: {state}; último motivo: {e['motivo']}")
    if ids is not None:
        lines.append(f"  ids pedidos: {sorted(ids)}")
    return "\n".join(lines)


def _is_rejection(reason: str) -> bool:
    normalize = lambda s: unicodedata.normalize("NFC", s).casefold()  # noqa: E731
    return normalize(REJECTION_REASON) in normalize(reason)


def run_without_frame_filtering(
    input_dir: str | Path,
    dataset: str,
    output_dir: str | Path,
    cfg: Config,
    *,
    ids: Iterable[str] | None = None,
    workers: int = 0,
    gpu: int = 0,
    debug: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Processes the rejected videos of ``output_dir/dataset`` (or only ``ids``) without step 2.

    ``input_dir`` must be the folder given to ``preproc.run``. ``workers = 0`` processes them
    in this process (they are few); ``workers > 0`` uses the same process pool as ``preproc.run``.
    ``debug = True`` also writes ``debug/<id>/`` like ``--debug-ids``.
    """
    ff = check_ffmpeg(cfg.ffmpeg.ffmpeg, cfg.ffmpeg.ffprobe)
    if not ff.has_libx264:
        raise FfmpegError(f"el ffmpeg {ff.version} ({ff.ffmpeg}) no tiene libx264, necesario para los miniclips")
    layout = existing_layout(output_dir, dataset)
    log = setup_logging(layout, log_level)
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de videos: {root}")

    rejected = list_rejected_by_filtering(layout)
    if ids is not None:
        wanted = {i.strip() for i in ids if i.strip()}
        rejected = [e for e in rejected if e["id"] in wanted]
    if not rejected:
        raise LookupError(explain_empty_selection(layout, ids))
    jobs = [Job(str(root / e["ruta"]), e["ruta"], e["id"], debug) for e in rejected]
    reasons = {e["id"]: e["motivo"] for e in rejected}
    log.info("sin paso 2: %d videos rechazados por '%s'", len(jobs), REJECTION_REASON)
    return _process_jobs(jobs, reasons, cfg, ff, layout, log, workers, gpu)


def _process_jobs(
    jobs: list[Job],
    reasons: dict[str, str],
    cfg: Config,
    ff: Any,
    layout: OutputLayout,
    log: Any,
    workers: int,
    gpu: int,
) -> dict[str, Any]:
    """Runs ``jobs`` through ``preproc.runner`` without step 2, marks the JSONs and rebuilds the summary."""
    counts = {"ok": 0, "fallido": 0}
    if jobs:
        threads = cfg.ffmpeg.threads or max(1, (os.cpu_count() or 1) // max(1, workers))
        # Hash of the ORIGINAL configuration: preproc.run will treat these outputs as done.
        setup = WorkerSetup(without_frame_filtering(cfg), ff, layout, config_hash(cfg), threads, mode="cpu")
        if workers == 0:
            _run_inline(jobs, setup, gpu, log, counts)
        else:
            _run_pool(jobs, setup, workers, gpu, log, counts)

    processed = [j.video_id for j in jobs if layout.meta_path(j.video_id).is_file()]
    for vid in processed:
        mark_frame_filtering_skipped(layout, vid, reasons[vid], cfg)
    last_errors = read_errors(layout)
    failed = {j.video_id: last_errors.get(j.video_id, {}).get("motivo", "")
              for j in jobs if j.video_id not in processed}

    stats = build_summary(layout)
    log.info("sin paso 2 terminado: %d procesados, %d fallidos", len(processed), len(failed))
    return {"procesados": processed, "fallidos": failed, "estadisticas": stats}


def mark_frame_filtering_skipped(layout: OutputLayout, video_id: str, reason: str, cfg: Config) -> None:
    """Adds the ``paso2_omitido`` section and a warning to ``meta/<id>.json``."""
    path = layout.meta_path(video_id)
    meta = json.loads(path.read_text(encoding="utf-8"))
    f = cfg.filter
    meta["paso2_omitido"] = {
        "motivo_rechazo": reason,
        "umbrales_originales": {
            "luma_threshold": f.luma_threshold,
            "std_threshold": f.std_threshold,
            "static_threshold": f.static_threshold,
            "static_min_frames": f.static_min_frames,
        },
    }
    warning = "paso 2 omitido (θ_L = θ_σ = θ_E = 0): el flujo completo rechazó el video"
    if warning not in meta.setdefault("avisos", []):
        meta["avisos"].append(warning)
    write_json_atomic(path, meta)
