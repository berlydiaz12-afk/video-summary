from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ._version import __version__

RESIZE_MODES = ("stretch", "short_side_crop")
X264_PRESETS = (
    "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_EXTENSIONS = (
    ".mp4", ".avi", ".flv", ".mkv", ".mov", ".wmv", ".webm", ".mpg", ".mpeg",
    ".m4v", ".3gp", ".ts", ".vob", ".ogv",
)


class ConfigError(ValueError):
    """Configuración inválida o incompleta."""


@dataclass
class FilterConfig:
    """Paso 2: filtrado de cuadros sin información."""

    luma_threshold: float | None = None    # θ_L
    std_threshold: float | None = None     # θ_σ
    static_threshold: float | None = None  # θ_E
    static_min_frames: int = 25            # 1 s a 25 fps


@dataclass
class TransNetConfig:
    """Paso 3: TransNet V2 (repo clonado como dependencia + pesos PyTorch convertidos)."""

    repo_dir: str | None = None
    weights: str | None = None
    threshold: float = 0.5   # transición si p > threshold (como el código oficial)
    batch_size: int = 64     # ventanas de 100 cuadros por lote
    timeout: float = 3600.0  # segundos que un trabajador espera al servidor GPU


@dataclass
class SummaryConfig:
    """Pasos 3–5: ventanas, redundancia y miniclips."""

    frames_per_clip: int = 32               # T
    stride: int = 8                         # r
    phash_step: int = 25                    # 1 cuadro por segundo
    hamming_threshold: float | None = None  # θ_H
    short_shot_windows: bool = False        # True: una toma de T a L−1 cuadros aporta una candidata con r = 1

    @property
    def window_length(self) -> int:
        """L = T · r."""
        return self.frames_per_clip * self.stride


@dataclass
class OutputConfig:
    """Paso 6: redimensionamiento y normalización."""

    size: int = 224
    resize_mode: str = "stretch"
    sws_flags: str = "bilinear+accurate_rnd+full_chroma_int"
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    crf: int = 18                  # calidad de x264 (yuv420p) de cada miniclip .mp4; 18 ≈ visualmente sin pérdida
    preset: str = "veryfast"       # preset de x264 (más lento = más pequeño)


@dataclass
class AnalysisConfig:
    """Pasada de análisis a baja resolución."""

    gray_width: int = 64
    gray_height: int = 36
    gray_sws_flags: str = "area"
    transnet_sws_flags: str = "bicubic"
    alignment_tolerance: float = 20.0


@dataclass
class FfmpegConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    threads: int = 0  # 0 = núcleos / trabajadores
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS


@dataclass
class DebugConfig:
    """Modo de diagnóstico: los miniclips finales en .mp4 con nombres descriptivos."""

    playback_fps: float | None = None  # None = fps / r (duración real de la ventana)
    codec: str = "libx264"
    crf: int = 18


@dataclass
class Config:
    fps: int = 25
    filter: FilterConfig = field(default_factory=FilterConfig)
    transnet: TransNetConfig = field(default_factory=TransNetConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ffmpeg: FfmpegConfig = field(default_factory=FfmpegConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @property
    def playback_fps(self) -> float:
        return self.debug.playback_fps or self.fps / self.summary.stride

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS: dict[str, type] = {
    "filter": FilterConfig,
    "transnet": TransNetConfig,
    "summary": SummaryConfig,
    "output": OutputConfig,
    "analysis": AnalysisConfig,
    "ffmpeg": FfmpegConfig,
    "debug": DebugConfig,
}

# Claves que no cambian el contenido de las salidas: no entran en el hash.
_NON_OUTPUT_KEYS = {
    "transnet": ("repo_dir", "weights", "batch_size", "timeout"),
    "ffmpeg": ("ffmpeg", "ffprobe", "threads", "extensions"),
}


def load_config(path: str | Path, *, require_thresholds: bool = True) -> Config:
    """Lee el YAML, resuelve rutas relativas al archivo y valida."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No existe el archivo de configuración: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return config_from_dict(data, base_dir=path.parent, require_thresholds=require_thresholds)


def config_from_dict(
    data: dict[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
    require_thresholds: bool = True,
) -> Config:
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ConfigError("La configuración debe ser un diccionario YAML.")
    unknown = sorted(set(data) - {"fps", *_SECTIONS})
    if unknown:
        raise ConfigError(f"Claves desconocidas en la configuración: {', '.join(unknown)}")
    sections = {name: _build_section(cls, data.get(name), name) for name, cls in _SECTIONS.items()}
    cfg = Config(fps=data.get("fps", 25), **sections)
    if base_dir is not None:
        _resolve_paths(cfg, Path(base_dir))
    validate(cfg, require_thresholds=require_thresholds)
    return cfg


def config_hash(cfg: Config) -> str:
    """Hash de los parámetros que afectan a las salidas, junto con la versión del código."""
    data = cfg.to_dict()
    data.pop("debug", None)
    for section, keys in _NON_OUTPUT_KEYS.items():
        for key in keys:
            data[section].pop(key, None)
    payload = json.dumps({"version": __version__, "config": data}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate(cfg: Config, *, require_thresholds: bool = True) -> None:
    """Normaliza tipos y comprueba rangos; con ``require_thresholds`` exige θ_L, θ_σ, θ_E y θ_H."""
    f, t, s, o, a, ff, d = (
        cfg.filter, cfg.transnet, cfg.summary, cfg.output, cfg.analysis, cfg.ffmpeg, cfg.debug
    )
    cfg.fps = _coerce(cfg.fps, int, "fps")
    f.luma_threshold = _coerce(f.luma_threshold, float, "filter.luma_threshold", optional=True)
    f.std_threshold = _coerce(f.std_threshold, float, "filter.std_threshold", optional=True)
    f.static_threshold = _coerce(f.static_threshold, float, "filter.static_threshold", optional=True)
    f.static_min_frames = _coerce(f.static_min_frames, int, "filter.static_min_frames")
    t.threshold = _coerce(t.threshold, float, "transnet.threshold")
    t.batch_size = _coerce(t.batch_size, int, "transnet.batch_size")
    t.timeout = _coerce(t.timeout, float, "transnet.timeout")
    s.frames_per_clip = _coerce(s.frames_per_clip, int, "summary.frames_per_clip")
    s.stride = _coerce(s.stride, int, "summary.stride")
    s.phash_step = _coerce(s.phash_step, int, "summary.phash_step")
    s.hamming_threshold = _coerce(s.hamming_threshold, float, "summary.hamming_threshold", optional=True)
    if not isinstance(s.short_shot_windows, bool):
        raise ConfigError("'summary.short_shot_windows' debe ser true o false")
    o.size = _coerce(o.size, int, "output.size")
    o.mean = _triple(o.mean, "output.mean")
    o.std = _triple(o.std, "output.std")
    o.crf = _coerce(o.crf, int, "output.crf")
    o.preset = str(o.preset)
    a.gray_width = _coerce(a.gray_width, int, "analysis.gray_width")
    a.gray_height = _coerce(a.gray_height, int, "analysis.gray_height")
    a.alignment_tolerance = _coerce(a.alignment_tolerance, float, "analysis.alignment_tolerance")
    ff.threads = _coerce(ff.threads, int, "ffmpeg.threads")
    ff.extensions = tuple(
        (e if str(e).startswith(".") else f".{e}").lower() for e in (ff.extensions or ())
    )
    d.playback_fps = _coerce(d.playback_fps, float, "debug.playback_fps", optional=True)
    d.crf = _coerce(d.crf, int, "debug.crf")

    if require_thresholds:
        missing = [
            label
            for label, value in (
                ("filter.luma_threshold (θ_L)", f.luma_threshold),
                ("filter.std_threshold (θ_σ)", f.std_threshold),
                ("filter.static_threshold (θ_E)", f.static_threshold),
                ("summary.hamming_threshold (θ_H)", s.hamming_threshold),
            )
            if value is None
        ]
        if missing:
            raise ConfigError("Faltan parámetros obligatorios en la configuración: " + ", ".join(missing))

    checks = [
        (cfg.fps > 0, "fps debe ser > 0"),
        (f.static_min_frames >= 2, "filter.static_min_frames debe ser >= 2"),
        (0.0 < t.threshold < 1.0, "transnet.threshold debe estar en (0, 1)"),
        (t.batch_size >= 1, "transnet.batch_size debe ser >= 1"),
        (t.timeout > 0, "transnet.timeout debe ser > 0"),
        (s.frames_per_clip >= 2, "summary.frames_per_clip debe ser >= 2"),
        (s.stride >= 1, "summary.stride debe ser >= 1"),
        (1 <= s.phash_step <= s.window_length, "summary.phash_step debe estar en [1, T·r]"),
        (s.hamming_threshold is None or s.hamming_threshold >= 0, "summary.hamming_threshold debe ser >= 0"),
        (o.resize_mode in RESIZE_MODES, f"output.resize_mode debe ser uno de {RESIZE_MODES}"),
        (o.size >= 16 and o.size % 2 == 0, "output.size debe ser par y >= 16"),
        (all(v > 0 for v in o.std), "output.std debe ser > 0"),
        (o.preset in X264_PRESETS, f"output.preset debe ser uno de {X264_PRESETS}"),
        (0 <= o.crf <= 51, "output.crf debe estar en [0, 51]"),
        (a.gray_width >= 8 and a.gray_height >= 8, "analysis.gray_width/gray_height deben ser >= 8"),
        (ff.threads >= 0, "ffmpeg.threads debe ser >= 0"),
        (len(ff.extensions) > 0, "ffmpeg.extensions no puede estar vacío"),
        (d.playback_fps is None or d.playback_fps > 0, "debug.playback_fps debe ser > 0"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ConfigError("; ".join(errors))


def _build_section(cls: type, data: Any, name: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"La sección '{name}' debe ser un diccionario.")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Claves desconocidas en '{name}': {', '.join(unknown)}")
    return cls(**data)


def _resolve_paths(cfg: Config, base: Path) -> None:
    t = cfg.transnet
    if t.repo_dir:
        t.repo_dir = str((base / Path(t.repo_dir).expanduser()).resolve())
    if t.weights:
        t.weights = str((base / Path(t.weights).expanduser()).resolve())


def _coerce(value: Any, kind: type, key: str, *, optional: bool = False) -> Any:
    if value is None:
        if optional:
            return None
        raise ConfigError(f"'{key}' no puede ser null")
    if isinstance(value, bool):
        raise ConfigError(f"'{key}' debe ser numérico, no booleano")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"'{key}' debe ser numérico: {value!r}") from None
    if kind is int:
        if not number.is_integer():
            raise ConfigError(f"'{key}' debe ser entero: {value!r}")
        return int(number)
    return number


def _triple(value: Any, key: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"'{key}' debe tener 3 valores (R, G, B)")
    return tuple(_coerce(v, float, key) for v in value)  # type: ignore[return-value]
