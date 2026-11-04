from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .config import Config, config_hash
from .detector import GpuHandles, GpuServer, RemoteDetector
from .ffmpeg_io import FfmpegError, FfmpegInfo, check_ffmpeg
from .pipeline import clip_stride, process_video
from .reporting import append_error, build_summary, setup_logging
from .storage import ClipWriter, OutputLayout, assign_ids, discover_videos, is_done
from .structs import VideoFailed


@dataclass(frozen=True)
class Job:
    path: str
    rel_path: str
    video_id: str
    debug: bool


@dataclass
class WorkerSetup:
    cfg: Config
    ff: FfmpegInfo
    layout: OutputLayout
    config_hash: str
    threads: int
    mode: str                        # "remote" (servidor GPU) | "cpu"
    handles: GpuHandles | None = None
    tmp_dir: str | None = None


_STATE: dict[str, Any] = {}


def run(
    input_dir: str | Path,
    dataset: str,
    output_dir: str | Path,
    cfg: Config,
    *,
    workers: int | None = None,
    gpu: int = 0,
    debug_samples: int = 0,
    debug_ids: Iterable[str] = (),
    force: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Procesa todos los videos de ``input_dir`` y escribe las salidas en ``output_dir/dataset``."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de entrada: {input_dir}")
    ff = check_ffmpeg(cfg.ffmpeg.ffmpeg, cfg.ffmpeg.ffprobe)
    if not ff.has_libx264:
        raise FfmpegError(
            f"el ffmpeg {ff.version} ({ff.ffmpeg}) no tiene libx264, necesario para guardar los "
            "miniclips en .mp4; instale un ffmpeg compilado con libx264"
        )
    layout = OutputLayout.create(output_dir, dataset)
    log = setup_logging(layout, log_level)
    chash = config_hash(cfg)
    log.info("ffmpeg %s (%s), libx264: %s", ff.version, ff.ffmpeg, ff.has_libx264)
    log.info("configuración %s, entrada %s, salida %s", chash, input_dir.resolve(), layout.root)

    videos = discover_videos(input_dir, cfg.ffmpeg.extensions, exclude=[layout.root])
    ids = assign_ids([rel for _, rel in videos])
    debug_ids = {d.strip() for d in debug_ids if d.strip()}
    unknown = debug_ids - set(ids)
    if unknown:
        log.warning("ids de diagnóstico inexistentes: %s", ", ".join(sorted(unknown)))
    debug_set = set(ids[:max(0, debug_samples)]) | debug_ids
    jobs = [Job(str(path), rel, vid, vid in debug_set) for (path, rel), vid in zip(videos, ids)]
    pending = [j for j in jobs if force or j.debug or not is_done(layout, j.video_id, chash)]
    skipped = len(jobs) - len(pending)
    log.info("%d videos encontrados, %d ya procesados, %d pendientes", len(jobs), skipped, len(pending))

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if os.name == "nt":
        workers = min(workers, 61)  # límite de ProcessPoolExecutor en Windows
    threads = cfg.ffmpeg.threads or max(1, (os.cpu_count() or 1) // max(1, workers))
    counts = {"ok": 0, "fallido": 0}

    if pending:
        setup = WorkerSetup(cfg, ff, layout, chash, threads, mode="cpu")
        if workers == 0:
            _run_inline(pending, setup, gpu, log, counts)
        else:
            _run_pool(pending, setup, workers, gpu, log, counts)

    stats = build_summary(layout)
    log.info("terminado: %d procesados, %d fallidos, %d omitidos", counts["ok"], counts["fallido"], skipped)
    return {"procesados": counts["ok"], "fallidos": counts["fallido"], "omitidos": skipped, "estadisticas": stats}


def _run_pool(pending: list[Job], setup: WorkerSetup, workers: int, gpu: int,
              log: Logger, counts: dict[str, int]) -> None:
    ctx = mp.get_context("spawn")
    server: GpuServer | None = None
    tmp_dir = setup.layout.root / "_tmp"
    try:
        if gpu >= 0:
            server = GpuServer(setup.cfg.transnet, f"cuda:{gpu}", workers, ctx)
            log.info("servidor GPU listo: %s", server.start())
            tmp_dir.mkdir(exist_ok=True)
            setup.mode, setup.handles, setup.tmp_dir = "remote", server.handles, str(tmp_dir)
        else:
            log.info("TransNet V2 en CPU dentro de cada trabajador (%d hilos)", setup.threads)

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker, initargs=(setup,)) as executor:
            futures = {executor.submit(_process_one, job): job for job in pending}
            try:
                with tqdm(total=len(futures), unit="video", dynamic_ncols=True) as bar:
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            result = future.result()
                        except BrokenProcessPool as exc:
                            raise RuntimeError(
                                "un proceso de trabajo terminó abruptamente (¿memoria insuficiente?)"
                            ) from exc
                        except Exception as exc:  # noqa: BLE001
                            result = _failure(job, f"{type(exc).__name__}: {exc}", traceback.format_exc())
                        _record(result, setup.layout, log, counts)
                        bar.update(1)
                        bar.set_postfix(ok=counts["ok"], fallidos=counts["fallido"])
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    finally:
        if server is not None:
            server.stop()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_inline(pending: list[Job], setup: WorkerSetup, gpu: int, log: Logger, counts: dict[str, int]) -> None:
    from .transnet import TransNetDetector

    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    detector = TransNetDetector(setup.cfg.transnet, device=device,
                                num_threads=None if gpu >= 0 else setup.threads)
    log.info("modo en proceso: %s", detector.describe())
    _STATE.clear()
    _STATE.update(setup=setup, detector=detector)
    with tqdm(total=len(pending), unit="video", dynamic_ncols=True) as bar:
        for job in pending:
            _record(_process_one(job), setup.layout, log, counts)
            bar.update(1)
            bar.set_postfix(ok=counts["ok"], fallidos=counts["fallido"])


def _init_worker(setup: WorkerSetup) -> None:
    _STATE.clear()
    _STATE["setup"] = setup
    _STATE["detector"] = None
    if setup.mode == "remote":
        assert setup.handles is not None and setup.tmp_dir is not None
        slot = setup.handles.slots.get(timeout=300)
        _STATE["detector"] = RemoteDetector(setup.handles, slot, Path(setup.tmp_dir), setup.cfg.transnet.timeout)


def _get_detector() -> Any:
    detector = _STATE.get("detector")
    if detector is None:
        from .transnet import TransNetDetector

        setup: WorkerSetup = _STATE["setup"]
        detector = TransNetDetector(setup.cfg.transnet, device="cpu", num_threads=setup.threads)
        _STATE["detector"] = detector
    return detector


def _process_one(job: Job) -> dict[str, Any]:
    setup: WorkerSetup = _STATE["setup"]
    layout = setup.layout
    start = time.perf_counter()
    cfg = setup.cfg
    try:
        # Cada miniclip se codifica apenas se extrae: la memoria no depende de cuántos tenga el video.
        with ClipWriter(layout, job.video_id, setup.ff, crf=cfg.output.crf, preset=cfg.output.preset) as writer:
            def sink(k: int, frames: Any, clip: Any) -> None:
                writer.write(k, frames, cfg.fps / (clip_stride(clip) or cfg.summary.stride))

            _, meta = process_video(
                job.path, job.rel_path, job.video_id, cfg, _get_detector(), setup.ff,
                threads=setup.threads,
                config_hash=setup.config_hash,
                debug_dir=layout.debug_path(job.video_id) if job.debug else None,
                clip_sink=sink,
            )
            writer.commit(meta)
        return {
            "estado": "ok",
            "id": job.video_id,
            "ruta": job.rel_path,
            "respaldo": meta["respaldo"]["caso"],
            "avisos": meta.get("avisos", []),
            "segundos": time.perf_counter() - start,
        }
    except VideoFailed as exc:
        return _failure(job, str(exc), "")
    except Exception as exc:  # noqa: BLE001 - un video fallido no detiene el resto
        return _failure(job, f"{type(exc).__name__}: {exc}", traceback.format_exc())


def _failure(job: Job, reason: str, detail: str) -> dict[str, Any]:
    return {"estado": "fallido", "id": job.video_id, "ruta": job.rel_path, "motivo": reason, "detalle": detail}


def _record(result: dict[str, Any], layout: OutputLayout, log: Logger, counts: dict[str, int]) -> None:
    vid = result["id"]
    if result["estado"] == "ok":
        counts["ok"] += 1
        log.info("ok %s (%.1f s, respaldo: %s)", vid, result["segundos"], result["respaldo"])
        for warning in result.get("avisos", []):
            log.warning("%s: %s", vid, warning)
        return
    counts["fallido"] += 1
    append_error(layout, vid, result["ruta"], result["motivo"])
    log.error("fallido %s: %s", vid, result["motivo"])
    if result.get("detalle"):
        log.debug("%s\n%s", vid, result["detalle"])
