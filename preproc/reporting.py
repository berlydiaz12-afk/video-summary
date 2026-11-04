from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

from tqdm import tqdm

from .storage import OutputLayout, write_json_atomic

LOGGER_NAME = "preproc"
FALLBACK_CASES = ("ninguno", "N=0")
_FRAME_KEYS = (
    "paso1", "negros_uniformes", "pantalla_fija", "eliminados_paso2", "paso2", "transicion",
    "paso3", "en_ventanas", "en_ventanas_conservadas", "miniclips", "miniclips_unicos",
)
_COUNT_KEYS = (
    "segmentos", "tomas", "tomas_con_ventana", "tomas_cortas_candidatas", "tomas_sin_candidata",
    "ventanas_candidatas", "ventanas_conservadas", "ventanas_descartadas",
)


class _TqdmHandler(logging.Handler):
    """Escribe en consola sin romper la barra de progreso."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def setup_logging(layout: OutputLayout, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(layout.logs_dir / f"preproc_{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = _TqdmHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def append_error(layout: OutputLayout, video_id: str, rel_path: str, reason: str) -> None:
    """Agrega una línea ``fecha\\tid\\truta\\tmotivo`` a ``errores_<conjunto>.log``."""
    reason = " | ".join(line.strip() for line in (reason or "").strip().splitlines() if line.strip())
    line = "\t".join([datetime.now().isoformat(timespec="seconds"), video_id, rel_path, reason])
    with open(layout.errors_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_errors(layout: OutputLayout) -> dict[str, dict[str, str]]:
    """Último error registrado por id."""
    errors: dict[str, dict[str, str]] = {}
    if not layout.errors_log.is_file():
        return errors
    with open(layout.errors_log, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) == 4:
                errors[parts[1]] = {"fecha": parts[0], "id": parts[1], "ruta": parts[2], "motivo": parts[3]}
    return errors


def flatten(meta: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplana diccionarios anidados (``a.b`` → ``a_b``) y omite listas."""
    row: dict[str, Any] = {}
    for key, value in meta.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            row.update(flatten(value, f"{name}_"))
        elif isinstance(value, (list, tuple)):
            continue
        else:
            row[name] = value
    return row


def build_summary(layout: OutputLayout) -> dict[str, Any]:
    """Escribe ``resumen_<conjunto>.csv`` (y Parquet si hay pyarrow) y ``estadisticas_<conjunto>.json``."""
    log = logging.getLogger(LOGGER_NAME)
    metas: list[dict[str, Any]] = []
    for path in sorted(layout.meta_dir.glob("*.json")):
        try:
            metas.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            log.warning("no se pudo leer %s: %s", path.name, exc)

    rows = [flatten(m) for m in metas]
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    tmp = layout.summary_csv.with_name(layout.summary_csv.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, layout.summary_csv)
    _write_parquet(layout, rows, columns)

    done = {m.get("id") for m in metas}
    failed = [e for vid, e in sorted(read_errors(layout).items()) if vid not in done]
    stats = compute_stats(layout.dataset, metas, failed)
    write_json_atomic(layout.stats_json, stats)
    return stats


def compute_stats(dataset: str, metas: list[dict[str, Any]], failed: list[dict[str, str]]) -> dict[str, Any]:
    frames = {k: [m.get("cuadros", {}).get(k) for m in metas] for k in _FRAME_KEYS}
    totals = {k: sum(v for v in vals if v is not None) for k, vals in frames.items()}
    before = totals["paso1"]
    time_keys: list[str] = []
    for m in metas:
        time_keys.extend(k for k in m.get("tiempos_s", {}) if k not in time_keys)
    cases = Counter(m.get("respaldo", {}).get("caso") for m in metas)
    return {
        "conjunto": dataset,
        "videos_procesados": len(metas),
        "videos_fallidos": len(failed),
        "duracion_original_s": _summary(m.get("original", {}).get("duracion_s") for m in metas),
        "cuadros": {k: _summary(vals) for k, vals in frames.items()},
        "reduccion_pct": {
            "paso2": _reduction(totals["paso2"], before),
            "paso3": _reduction(totals["paso3"], before),
            "ventanas_candidatas": _reduction(totals["en_ventanas"], before),
            "ventanas_conservadas": _reduction(totals["en_ventanas_conservadas"], before),
            "miniclips_unicos": _reduction(totals["miniclips_unicos"], before),
        },
        "por_video": {k: _summary(m.get("conteos", {}).get(k) for m in metas) for k in _COUNT_KEYS},
        "miniclips_por_video": {
            "generados": _summary(m.get("seleccion", {}).get("miniclips") for m in metas),
            "duracion_util_s": _summary(m.get("seleccion", {}).get("duracion_util_s") for m in metas),
        },
        "respaldo": {case: cases.get(case, 0) for case in FALLBACK_CASES},
        "tiempo_s": {k: _summary(m.get("tiempos_s", {}).get(k) for m in metas) for k in time_keys},
        "config_hash": sorted({m.get("config_hash") for m in metas if m.get("config_hash")}),
        "version": sorted({m.get("version") for m in metas if m.get("version")}),
        "fallidos": failed,
    }


def _summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return {"total": 0, "media": None, "min": None, "max": None}
    return {"total": sum(vals), "media": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}


def _reduction(after: float, before: float) -> float | None:
    return round(100.0 * (1.0 - after / before), 4) if before else None


def _write_parquet(layout: OutputLayout, rows: list[dict[str, Any]], columns: list[str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return
    table = pa.table({c: [row.get(c) for row in rows] for c in columns})
    pq.write_table(table, layout.summary_parquet)
