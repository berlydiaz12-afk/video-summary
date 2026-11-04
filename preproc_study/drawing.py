"""OpenCV drawing: captions, frame strips, contact sheets and frame annotations.

All functions take and return RGB uint8 arrays (colors are RGB tuples too);
:func:`write_png` converts to BGR only when saving. OpenCV's Hershey fonts are
ASCII-only, so captions spell out symbols (``theta_H``, ``std``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import cv2
import numpy as np

from preproc.structs import Miniclip

from . import style
from .frame_filtering import REASON_NAMES, FrameFilteringAnalysis, RemovedRun
from .redundancy_removal import RedundancyAnalysis, WindowDecision
from .shot_detection import ShotDetectionAnalysis, Transition
from .timeline import DISCARDED, KEPT, SHORT, STATE_NAMES, FrameTimeline

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SURFACE = style.rgb(style.SURFACE)
INK = style.rgb(style.INK)
INK_SECONDARY = style.rgb(style.INK_SECONDARY)
PLACEHOLDER = style.rgb(style.GRIDLINE)
REASON_RGB = {code: style.rgb(style.REASON_COLORS[name]) for code, name in REASON_NAMES.items()}
STATE_RGB = {DISCARDED: style.rgb(style.DISCARDED), KEPT: style.rgb(style.KEPT), SHORT: style.rgb(style.SHORT_SHOT)}
MINICLIP_RGB = style.rgb(style.MINICLIP)
TRANSITION_RGB = style.rgb(style.TRANSITION)

Line = Union[str, tuple[str, tuple[int, int, int]]]
Thumbnails = Mapping[int, np.ndarray]


# --------------------------------------------------------------------------- basics

def write_png(path: str | Path, image: np.ndarray) -> None:
    """Saves an RGB image as PNG (``imencode`` also works with non-ASCII paths)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"OpenCV could not encode {path}")
    path.write_bytes(buf.tobytes())


def _line_text(line: Line) -> str:
    return line if isinstance(line, str) else line[0]


def _line_color(line: Line, default: tuple[int, int, int]) -> tuple[int, int, int]:
    return default if isinstance(line, str) else line[1]


def _line_height(scale: float) -> int:
    (_, h), base = cv2.getTextSize("Ag", FONT, scale, 1)
    return h + base + 2


def draw_text_box(
    img: np.ndarray,
    lines: Sequence[Line],
    origin: tuple[int, int] = (4, 4),
    *,
    scale: float = 0.4,
    color: tuple[int, int, int] = WHITE,
    background: tuple[int, int, int] = BLACK,
    alpha: float = 0.6,
    pad: int = 3,
) -> np.ndarray:
    """Draws text lines over a translucent box, in place."""
    if not lines:
        return img
    line_h = _line_height(scale)
    width = max(cv2.getTextSize(_line_text(l), FONT, scale, 1)[0][0] for l in lines) + 2 * pad
    x0, y0 = origin
    x1, y1 = min(img.shape[1], x0 + width), min(img.shape[0], y0 + line_h * len(lines) + 2 * pad)
    roi = img[y0:y1, x0:x1]
    if roi.size:
        overlay = np.empty_like(roi)
        overlay[:] = background
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    (_, text_h), _ = cv2.getTextSize("Ag", FONT, scale, 1)
    for i, line in enumerate(lines):
        y = y0 + pad + i * line_h + text_h
        cv2.putText(img, _line_text(line), (x0 + pad, y), FONT, scale, _line_color(line, color), 1, cv2.LINE_AA)
    return img


def placeholder(size: tuple[int, int], text: str = "n/a") -> np.ndarray:
    """Gray tile of ``size = (width, height)`` for frames outside the video."""
    w, h = size
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = PLACEHOLDER
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), FONT, 0.45, INK_SECONDARY, 1, cv2.LINE_AA)
    return img


def captioned(
    image: np.ndarray,
    lines: Sequence[Line],
    *,
    border: tuple[int, int, int] | None = None,
    border_px: int = 3,
    scale: float = 0.36,
    pad: int = 3,
) -> np.ndarray:
    """Image with an optional colored border and caption lines below it."""
    h, w = image.shape[:2]
    b = border_px
    line_h = _line_height(scale)
    cap_h = 2 * pad + line_h * len(lines)
    canvas = np.empty((h + 2 * b + cap_h, w + 2 * b, 3), dtype=np.uint8)
    canvas[:] = SURFACE
    if border is not None:
        canvas[:h + 2 * b] = border
    canvas[b:b + h, b:b + w] = image
    (_, text_h), _ = cv2.getTextSize("Ag", FONT, scale, 1)
    for i, line in enumerate(lines):
        y = h + 2 * b + pad + i * line_h + text_h
        default = INK if i == 0 else INK_SECONDARY
        cv2.putText(canvas, _line_text(line), (b, y), FONT, scale, _line_color(line, default), 1, cv2.LINE_AA)
    return canvas


def hstack(tiles: Sequence[np.ndarray], gap: int = 6) -> np.ndarray:
    height = max(t.shape[0] for t in tiles)
    parts: list[np.ndarray] = []
    for i, t in enumerate(tiles):
        if i:
            parts.append(_filled(height, gap))
        parts.append(_pad_to(t, height, t.shape[1]))
    return np.concatenate(parts, axis=1)


def vstack(rows: Sequence[np.ndarray], gap: int = 6) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    parts: list[np.ndarray] = []
    for i, r in enumerate(rows):
        if i:
            parts.append(_filled(gap, width))
        parts.append(_pad_to(r, r.shape[0], width))
    return np.concatenate(parts, axis=0)


def grid(tiles: Sequence[np.ndarray], cols: int, gap: int = 6) -> np.ndarray:
    rows = [hstack(tiles[i:i + cols], gap) for i in range(0, len(tiles), cols)]
    return vstack(rows, gap)


def with_title(image: np.ndarray, lines: Sequence[Line], *, scale: float = 0.5, margin: int = 8) -> np.ndarray:
    """Adds a title block above ``image`` and a margin around it."""
    line_h = _line_height(scale)
    header_h = 2 * margin + line_h * len(lines)
    width = max(image.shape[1] + 2 * margin,
                max(cv2.getTextSize(_line_text(l), FONT, scale, 1)[0][0] for l in lines) + 2 * margin)
    canvas = _filled(header_h + image.shape[0] + margin, width)
    (_, text_h), _ = cv2.getTextSize("Ag", FONT, scale, 1)
    for i, line in enumerate(lines):
        y = margin + i * line_h + text_h
        default = INK if i == 0 else INK_SECONDARY
        cv2.putText(canvas, _line_text(line), (margin, y), FONT, scale, _line_color(line, default), 1, cv2.LINE_AA)
    canvas[header_h:header_h + image.shape[0], margin:margin + image.shape[1]] = image
    return canvas


def _filled(height: int, width: int) -> np.ndarray:
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = SURFACE
    return img


def _pad_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    out = _filled(height, width)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def _thumb(thumbs: Thumbnails, frame: int, size: tuple[int, int]) -> np.ndarray:
    img = thumbs.get(frame)
    return img if img is not None else placeholder(size)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "-"


# --------------------------------------------------------------------------- frames each strip needs

def removed_run_frames(run: RemovedRun) -> list[tuple[str, int]]:
    """(role, frame) shown for a removed run: before, first, middle, last, after."""
    a, b = run.start, run.end
    items = [("before", a - 1), ("first", a), ("middle", (a + b - 1) // 2), ("last", b - 1), ("after", b)]
    return _dedupe(items)


def transition_frames(transition: Transition, context: int = 3) -> list[tuple[str, int]]:
    """(role, frame) shown for a transition: ``context`` frames before, the transition, ``context`` after."""
    a, b = transition.start, transition.end
    inside = list(range(a, b)) if b - a <= 3 else [a, (a + b - 1) // 2, b - 1]
    items = [(f"-{context - i}", a - context + i) for i in range(context)]
    items += [("T", n) for n in inside]
    items += [(f"+{i + 1}", b + i) for i in range(context)]
    return _dedupe(items)


def discarded_window_frames(decision: WindowDecision, redundancy: RedundancyAnalysis) -> tuple[list[int], list[int]]:
    """Signature frames of the reference window and of the discarded window."""
    ref = redundancy.windows[decision.reference]  # type: ignore[index]
    win = redundancy.windows[decision.window]
    return ([ref.start + int(o) for o in redundancy.offsets], [win.start + int(o) for o in redundancy.offsets])


def _dedupe(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    seen: set[int] = set()
    out: list[tuple[str, int]] = []
    for role, n in items:
        if n not in seen and n >= 0:
            seen.add(n)
            out.append((role, n))
    return out


# --------------------------------------------------------------------------- strips

def removed_run_strip(
    run: RemovedRun,
    filtering: FrameFilteringAnalysis,
    thumbs: Thumbnails,
    size: tuple[int, int],
    fps: int,
) -> np.ndarray:
    """Frames around a removed run, each with its luma, std and d_k; removed frames framed by reason color."""
    codes = filtering.reason_codes()
    cfg = filtering.config
    tiles = []
    for role, n in removed_run_frames(run):
        if n >= filtering.n_frames:
            tiles.append(captioned(placeholder(size, "end"), [role]))
            continue
        border = REASON_RGB.get(int(codes[n])) if run.start <= n < run.end else None
        lines: list[Line] = [
            f"{role}  n={n}",
            f"t={n / fps:.2f}s",
            f"luma={filtering.luma_mean[n]:.1f} std={filtering.luma_std[n]:.1f}",
            f"d={_fmt(filtering.frame_diff[n])}",
        ]
        tiles.append(captioned(_thumb(thumbs, n, size), lines, border=border))
    title: list[Line] = [
        f"Removed run {run.index}: frames {run.start}-{run.end - 1}, {run.length} frames "
        f"({run.length / fps:.2f} s from {run.start / fps:.2f} s)",
        f"reason: {run.reason}  (black {run.n_black}, uniform {run.n_uniform}, static {run.n_static})",
        f"thresholds: luma < {cfg.luma_threshold:g}, std < {cfg.std_threshold:g}, "
        f"d < {cfg.static_threshold:g} for >= {cfg.static_min_frames} frames",
    ]
    return with_title(hstack(tiles), title)


def transition_strip(
    transition: Transition,
    shots: ShotDetectionAnalysis,
    thumbs: Thumbnails,
    size: tuple[int, int],
    fps: int,
) -> np.ndarray:
    """Frames before, during and after a transition, each with its TransNet V2 probability."""
    n_frames = len(shots.probabilities)
    tiles = []
    for role, n in transition_frames(transition):
        if n >= n_frames:
            tiles.append(captioned(placeholder(size, "end"), [role]))
            continue
        p = float(shots.probabilities[n])
        inside = transition.start <= n < transition.end
        p_line: Line = (f"p={_fmt(p, 3)}", TRANSITION_RGB) if inside else f"p={_fmt(p, 3)}"
        lines: list[Line] = [f"{role}  n={n}", f"t={n / fps:.2f}s", p_line]
        tiles.append(captioned(_thumb(thumbs, n, size), lines, border=TRANSITION_RGB if inside else None))
    title: list[Line] = [
        f"Transition {transition.index}: frames {transition.start}-{transition.end - 1} "
        f"({transition.length} frames at {transition.start / fps:.2f} s, segment {transition.segment})",
        f"peak p = {transition.peak:.3f}; a frame is a transition when p > {shots.threshold:g} "
        "(p = '-' on removed frames)",
    ]
    return with_title(hstack(tiles), title)


def discarded_window_strip(
    decision: WindowDecision,
    redundancy: RedundancyAnalysis,
    thumbs: Thumbnails,
    size: tuple[int, int],
    fps: int,
) -> np.ndarray:
    """A discarded window under its reference, at the signature positions, with per-position distances."""
    ref_frames, win_frames = discarded_window_frames(decision, redundancy)
    per_position = redundancy.position_distances(decision.window, decision.reference)  # type: ignore[arg-type]
    ref_tiles = [captioned(_thumb(thumbs, n, size), [f"+{o}  n={n}", f"t={n / fps:.2f}s"],
                           border=STATE_RGB[KEPT])
                 for o, n in zip(redundancy.offsets, ref_frames)]
    win_tiles = [captioned(_thumb(thumbs, n, size), [f"+{o}  n={n}", f"t={n / fps:.2f}s", f"Hamming={int(h)}"],
                           border=STATE_RGB[DISCARDED])
                 for o, n, h in zip(redundancy.offsets, win_frames, per_position)]
    body = vstack([
        with_title(hstack(ref_tiles), [f"reference: window {decision.reference} (kept)"], scale=0.42, margin=2),
        with_title(hstack(win_tiles), [f"window {decision.window} (discarded)"], scale=0.42, margin=2),
    ])
    title: list[Line] = [
        f"Window {decision.window} (shot {decision.shot}, {decision.start / fps:.2f} s) discarded",
        f"mean Hamming distance {decision.reference_distance:.2f} <= theta_H {redundancy.threshold:g} "
        f"against window {decision.reference}",
    ]
    return with_title(body, title)


def miniclip_sheet(
    frames: np.ndarray,
    clip: Miniclip,
    order: int,
    fps: int,
    stride: int | None,
    playback_fps: float,
    *,
    cols: int = 8,
) -> np.ndarray:
    """The T frames given to Swin for one miniclip, in a grid, with frame index and time."""
    tiles = [captioned(f, [f"#{t}  n={n}", f"{n / fps:.2f}s"], border_px=1)
             for t, (f, n) in enumerate(zip(frames, clip.frame_indices))]
    if clip.candidate_index is None:
        origin = "longest shot (case N=0)"
    else:
        origin = f"candidate {clip.candidate_index} ({clip.kind})"
    spacing = f"r = {stride}" if stride is not None else "r not constant (evenly spaced over the shot)"
    title: list[Line] = [
        f"Miniclip {order}: {origin}, shot {clip.shot}, start frame {clip.start} ({clip.start / fps:.2f} s)",
        f"{spacing}   stored at {playback_fps:g} fps   motion m = {clip.motion:.4f} (descriptor only)",
    ]
    return with_title(grid(tiles, cols), title)


# --------------------------------------------------------------------------- annotated video

def annotate_frame(frame: np.ndarray, n: int, timeline: FrameTimeline) -> np.ndarray:
    """Writes the procedure status of frame ``n`` over it (in place) and returns it."""
    fps = timeline.fps
    lines: list[Line] = [f"n={n}  t={n / fps:.2f}s"]
    code = int(timeline.reason[n])
    bar: tuple[int, int, int] | None = None
    if code:
        lines.append((f"removed: {REASON_NAMES[code]}", REASON_RGB[code]))
        bar = REASON_RGB[code]
    else:
        p = float(timeline.probability[n])
        shot = int(timeline.shot[n])
        if shot >= 0:
            lines.append(f"shot {shot}  p={p:.3f}")
        else:
            lines.append((f"transition  p={p:.3f}", TRANSITION_RGB))
            bar = TRANSITION_RGB
        candidate, state = int(timeline.candidate[n]), int(timeline.candidate_state[n])
        clip = int(timeline.miniclip[n])
        if candidate >= 0:
            text = f"candidate {candidate}: {STATE_NAMES[state]}"
            if state != DISCARDED and clip >= 0:
                text += f" -> miniclip {clip}"
            lines.append((text, STATE_RGB[state]))
            bar = STATE_RGB[state]
        elif clip >= 0:
            lines.append((f"miniclip {clip} (longest shot)", MINICLIP_RGB))
            bar = MINICLIP_RGB
    scale = max(0.35, frame.shape[1] / 800)
    draw_text_box(frame, lines, (4, 4), scale=scale)
    h, w = frame.shape[:2]
    if bar is not None:
        frame[h - 6:h] = bar
    if timeline.sampled[n]:
        cv2.circle(frame, (w - 10, 10), 5, MINICLIP_RGB, -1, cv2.LINE_AA)
        cv2.circle(frame, (w - 10, 10), 5, WHITE, 1, cv2.LINE_AA)
    return frame
