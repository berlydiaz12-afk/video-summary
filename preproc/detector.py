from __future__ import annotations

import itertools
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .config import TransNetConfig


class ShotDetector(Protocol):
    def predict_segments(self, segments: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Segmentos ``(n, 27, 48, 3)`` uint8 → probabilidad de transición ``float32[n]`` por segmento."""
        ...


@dataclass
class GpuHandles:
    """Colas compartidas con los trabajadores (se heredan al crear los procesos)."""

    request_queue: Any
    response_queues: list[Any]
    slots: Any  # identificadores libres; cada trabajador toma uno al iniciar


class RemoteDetector:
    """Cliente del ``GpuServer`` usado dentro de cada trabajador."""

    def __init__(self, handles: GpuHandles, slot: int, tmp_dir: Path, timeout: float) -> None:
        self.request_queue = handles.request_queue
        self.response_queue = handles.response_queues[slot]
        self.slot = slot
        self.tmp_dir = Path(tmp_dir)
        self.timeout = timeout
        self._counter = itertools.count()

    def predict_segments(self, segments: Sequence[np.ndarray]) -> list[np.ndarray]:
        segments = [np.asarray(s, dtype=np.uint8) for s in segments]
        if not segments:
            return []
        lengths = [len(s) for s in segments]
        req_id = next(self._counter)
        path = self.tmp_dir / f"w{self.slot:03d}_{os.getpid()}_{req_id}.npy"
        np.save(path, np.concatenate(segments, axis=0), allow_pickle=False)
        try:
            self.request_queue.put((self.slot, req_id, str(path), lengths))
            while True:
                try:
                    status, rid, payload = self.response_queue.get(timeout=self.timeout)
                except queue.Empty:
                    raise RuntimeError(f"el servidor GPU no respondió en {self.timeout:.0f} s") from None
                if rid < req_id:  # respuesta tardía de una petición anterior que expiró
                    continue
                if rid != req_id:
                    raise RuntimeError("respuesta desincronizada del servidor GPU")
                if status != "ok":
                    raise RuntimeError(f"servidor GPU: {payload}")
                return payload
        finally:
            path.unlink(missing_ok=True)


class GpuServer:
    """Proceso único que ejecuta TransNet V2 en la GPU para todos los trabajadores."""

    def __init__(self, cfg: TransNetConfig, device: str, n_clients: int, ctx: Any) -> None:
        self.device = device
        self.handles = GpuHandles(
            request_queue=ctx.Queue(maxsize=max(2, 2 * n_clients)),
            response_queues=[ctx.Queue() for _ in range(n_clients)],
            slots=ctx.Queue(),
        )
        for i in range(n_clients):
            self.handles.slots.put(i)
        self._status = ctx.Queue()
        self._process = ctx.Process(
            target=_server_main,
            args=(cfg, device, self.handles.request_queue, self.handles.response_queues, self._status),
            name="preproc-gpu",
            daemon=True,
        )

    def start(self, timeout: float = 600.0) -> str:
        """Inicia el proceso y espera a que el modelo esté cargado. Devuelve su descripción."""
        self._process.start()
        try:
            status, message = self._status.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise RuntimeError("el servidor GPU no inició a tiempo") from None
        if status != "ok":
            self._process.join(5)
            raise RuntimeError(f"no se pudo cargar TransNet V2 en {self.device}: {message}")
        return message

    def stop(self, timeout: float = 30.0) -> None:
        if self._process.is_alive():
            try:
                self.handles.request_queue.put(None, timeout=5)
            except queue.Full:
                pass
            self._process.join(timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(5)


def _server_main(cfg: TransNetConfig, device: str, request_queue: Any,
                 response_queues: list[Any], status_queue: Any) -> None:
    try:
        from .transnet import TransNetDetector

        detector = TransNetDetector(cfg, device=device)
    except BaseException as exc:  # noqa: BLE001 - se informa al proceso principal
        status_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    status_queue.put(("ok", detector.describe()))

    while True:
        first = request_queue.get()
        if first is None:
            return
        group, stop = [first], False
        while len(group) < len(response_queues):
            try:
                item = request_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                stop = True
                break
            group.append(item)
        _serve_group(detector, group, response_queues)
        if stop:
            return


def _serve_group(detector: Any, group: list[tuple], response_queues: list[Any]) -> None:
    loaded: list[tuple[int, int, list[np.ndarray]]] = []
    for slot, req_id, path, lengths in group:
        try:
            frames = np.load(path, allow_pickle=False)
            bounds = np.cumsum([0, *lengths])
            if bounds[-1] != len(frames):
                raise ValueError("las longitudes no coinciden con el archivo")
            loaded.append((slot, req_id, [frames[a:b] for a, b in zip(bounds[:-1], bounds[1:])]))
        except Exception as exc:  # noqa: BLE001
            response_queues[slot].put(("error", req_id, f"{type(exc).__name__}: {exc}"))
    if not loaded:
        return
    try:
        probs = detector.predict_segments([seg for _, _, segs in loaded for seg in segs])
    except Exception as exc:  # noqa: BLE001
        for slot, req_id, _ in loaded:
            response_queues[slot].put(("error", req_id, f"{type(exc).__name__}: {exc}"))
        return
    pos = 0
    for slot, req_id, segs in loaded:
        response_queues[slot].put(("ok", req_id, probs[pos:pos + len(segs)]))
        pos += len(segs)
