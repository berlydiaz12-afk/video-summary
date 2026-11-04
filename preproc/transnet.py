from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import TransNetConfig

WINDOW = 100
CONTEXT = 25
STEP = 50
INPUT_SHAPE = (27, 48, 3)
MODULE_NAME = "transnetv2_pytorch"


def load_transnet_class(repo_dir: str | None = None) -> type:
    """Clase ``TransNetV2`` del repo indicado o, si no se indica, del ``PYTHONPATH``."""
    if repo_dir:
        path = _module_path(Path(repo_dir))
        module = sys.modules.get(MODULE_NAME)
        if module is None or Path(getattr(module, "__file__", "")).resolve() != path.resolve():
            spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"no se pudo cargar {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[MODULE_NAME] = module
            spec.loader.exec_module(module)
        return module.TransNetV2
    try:
        return importlib.import_module(MODULE_NAME).TransNetV2
    except ModuleNotFoundError as exc:
        if exc.name != MODULE_NAME:
            raise
        raise ImportError(
            "No se encontró transnetv2_pytorch. Clone https://github.com/soCzech/TransNetV2 "
            "e indique su ruta en config.yaml (transnet.repo_dir)."
        ) from None


def pad_for_windows(frames: np.ndarray) -> np.ndarray:
    """Relleno de ``predict_frames``: la longitud resultante cubre ``len(frames)`` con ventanas de paso 50."""
    n = len(frames)
    rem = n % STEP
    pad_end = CONTEXT + STEP - (rem if rem else STEP)
    return np.concatenate([
        np.repeat(frames[:1], CONTEXT, axis=0),
        frames,
        np.repeat(frames[-1:], pad_end, axis=0),
    ])


class TransNetDetector:
    """Implementa ``ShotDetector``: probabilidad de transición (cabeza ``single_frame``) por cuadro."""

    def __init__(self, cfg: TransNetConfig, device: str = "cpu", num_threads: int | None = None) -> None:
        if not cfg.weights:
            raise ValueError("falta transnet.weights en la configuración")
        weights = Path(cfg.weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"No existen los pesos {weights}. Conviértalos una vez con "
                "TransNetV2/inference-pytorch/convert_weights.py (requiere TensorFlow)."
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if num_threads:
            torch.set_num_threads(num_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"se pidió {device} pero CUDA no está disponible en este PyTorch")

        model = load_transnet_class(cfg.repo_dir)()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self._torch = torch
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.batch_size = cfg.batch_size

    def describe(self) -> str:
        if self.device.type == "cuda":
            return f"TransNet V2 en {self.device} ({self._torch.cuda.get_device_name(self.device)})"
        return f"TransNet V2 en {self.device}"

    def predict_segments(self, segments: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Segmentos RGB uint8 ``(n, 27, 48, 3)`` → probabilidades ``float32[n]`` por segmento."""
        outputs: list[np.ndarray] = []
        padded: list[np.ndarray | None] = []
        for seg in segments:
            seg = np.asarray(seg, dtype=np.uint8)
            if seg.ndim != 4 or seg.shape[1:] != INPUT_SHAPE:
                raise ValueError(f"se esperaban cuadros (n, 27, 48, 3), forma recibida {seg.shape}")
            outputs.append(np.empty(len(seg), dtype=np.float32))
            padded.append(pad_for_windows(seg) if len(seg) else None)

        jobs = [(si, ptr) for si, p in enumerate(padded) if p is not None
                for ptr in range(0, len(p) - WINDOW + 1, STEP)]
        for b in range(0, len(jobs), self.batch_size):
            chunk = jobs[b:b + self.batch_size]
            batch = np.stack([padded[si][ptr:ptr + WINDOW] for si, ptr in chunk])  # type: ignore[index]
            probs = self._forward(batch)
            for (si, ptr), p in zip(chunk, probs):
                out = outputs[si]
                # la ventana que empieza en ptr (con relleno) predice los cuadros originales [ptr, ptr + 50)
                end = min(ptr + STEP, len(out))
                if ptr < end:
                    out[ptr:end] = p[CONTEXT:CONTEXT + end - ptr]
        return outputs

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            out = self.model(torch.from_numpy(batch).to(self.device))
            logits = out[0] if isinstance(out, tuple) else out
            return torch.sigmoid(logits[..., 0]).float().cpu().numpy()


def _module_path(repo: Path) -> Path:
    name = f"{MODULE_NAME}.py"
    for cand in (repo, repo / name, repo / "inference-pytorch" / name):
        if cand.is_file() and cand.name == name:
            return cand
    raise FileNotFoundError(f"no se encontró {name} en {repo} (se espera <repo>/inference-pytorch/{name})")
