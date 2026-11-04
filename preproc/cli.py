from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config, validate
from .ffmpeg_io import FfmpegError, check_ffmpeg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preproc",
        description="Resumen y preprocesamiento de video (Cap. 4) para Video Swin Transformer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="procesa una carpeta de videos")
    run.add_argument("--input", required=True, help="carpeta con los videos (se recorre recursivamente)")
    run.add_argument("--dataset", required=True, help="nombre del conjunto (solo para nombrar salidas)")
    run.add_argument("--config", required=True, help="archivo YAML de configuración")
    run.add_argument("--output", default="salida", help="carpeta raíz de salida (por defecto: salida)")
    run.add_argument("--workers", type=int, default=None,
                     help="procesos de CPU (por defecto: núcleos − 1; 0 = sin paralelismo)")
    run.add_argument("--gpu", type=int, default=0, help="índice de GPU para TransNet V2 (−1 = CPU)")
    run.add_argument("--debug-samples", type=int, default=0,
                     help="guarda los miniclips en .mp4 para los primeros N videos")
    run.add_argument("--debug-ids", default="", help="ids separados por comas para guardar en .mp4")
    run.add_argument("--force", action="store_true", help="reprocesa aunque la salida exista")
    run.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    report = sub.add_parser("report", help="regenera resumen CSV y estadísticas desde meta/*.json")
    report.add_argument("--output", default="salida")
    report.add_argument("--dataset", required=True)

    check = sub.add_parser("check", help="verifica ffmpeg, la configuración y la carga de TransNet V2")
    check.add_argument("--config", required=True)
    check.add_argument("--gpu", type=int, default=-1, help="índice de GPU a probar (−1 = CPU)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "report":
            return _cmd_report(args)
        return _cmd_check(args)
    except (ConfigError, FfmpegError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrumpido", file=sys.stderr)
        return 130


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run

    cfg = load_config(args.config)
    result = run(
        args.input, args.dataset, args.output, cfg,
        workers=args.workers,
        gpu=args.gpu,
        debug_samples=args.debug_samples,
        debug_ids=args.debug_ids.split(","),
        force=args.force,
        log_level=args.log_level,
    )
    print(f"procesados: {result['procesados']}  fallidos: {result['fallidos']}  omitidos: {result['omitidos']}")
    return 0 if result["fallidos"] == 0 else 1


def _cmd_report(args: argparse.Namespace) -> int:
    from .reporting import build_summary
    from .storage import OutputLayout

    layout = OutputLayout.create(args.output, args.dataset)
    stats = build_summary(layout)
    print(f"{stats['videos_procesados']} videos, {stats['videos_fallidos']} fallidos → {layout.stats_json}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    import numpy as np

    cfg = load_config(args.config, require_thresholds=False)
    ff = check_ffmpeg(cfg.ffmpeg.ffmpeg, cfg.ffmpeg.ffprobe)
    print(f"ffmpeg {ff.version} ({ff.ffmpeg}); libx264 (salida de miniclips): {'sí' if ff.has_libx264 else 'NO'}")
    try:
        validate(cfg, require_thresholds=True)
        print("configuración completa")
    except ConfigError as exc:
        print(f"aviso: {exc}")
    print(f"pesos: {cfg.transnet.weights}  (existe: {bool(cfg.transnet.weights) and Path(cfg.transnet.weights).is_file()})")

    from .transnet import TransNetDetector

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    detector = TransNetDetector(cfg.transnet, device=device)
    probs = detector.predict_segments([np.zeros((10, 27, 48, 3), dtype=np.uint8)])
    print(f"{detector.describe()}: inferencia de prueba correcta ({len(probs[0])} probabilidades)")
    return 0
