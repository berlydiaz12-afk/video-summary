"""Summary (``summary.json``) and readable report (``report.md``) of one study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from preproc import __version__ as preproc_version

from ._version import __version__
from .decoding import DecodedVideo
from .export import (
    DECODING,
    FRAME_FILTERING,
    MINICLIPS,
    REDUNDANCY_REMOVAL,
    SHOT_DETECTION,
    SWIN_INPUT,
)
from .frame_filtering import FrameFilteringAnalysis
from .miniclips import MiniclipAnalysis
from .redundancy_removal import RedundancyAnalysis
from .shot_detection import ShotDetectionAnalysis


def build_summary(
    decoded: DecodedVideo,
    filtering: FrameFilteringAnalysis,
    shots: ShotDetectionAnalysis,
    redundancy: RedundancyAnalysis,
    miniclips: MiniclipAnalysis,
    swin_input: dict[str, Any] | None,
    warnings: list[str],
    timings: dict[str, float],
    verification: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fps = decoded.fps
    fc = filtering.config
    return {
        "video": str(decoded.path),
        "id": decoded.path.stem,
        "decoding": decoded.metadata(),
        "frame_filtering": {
            **filtering.counts(),
            "removed_fraction": round(filtering.n_removed / max(filtering.n_frames, 1), 4),
            "thresholds": {
                "luma_threshold": fc.luma_threshold,
                "std_threshold": fc.std_threshold,
                "static_threshold": fc.static_threshold,
                "static_min_frames": fc.static_min_frames,
            },
        },
        "shot_detection": {
            **shots.counts(),
            "useful_duration_s": round(shots.useful_frames / fps, 3),
            "threshold": shots.threshold,
            "frames_per_clip": shots.summary.frames_per_clip,
            "stride": shots.summary.stride,
            "window_length": shots.window_length,
            "short_shot_windows": shots.summary.short_shot_windows,
        },
        "redundancy_removal": {
            **redundancy.counts(),
            "hamming_threshold": redundancy.threshold,
            "phash_step": redundancy.summary.phash_step,
        },
        "miniclips": miniclips.counts(),
        "swin_input": swin_input,
        "verification": None if verification is None else {
            "all_match": all(r["match"] for r in verification),
            "rows": verification,
        },
        "warnings": list(warnings),
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "versions": {"preproc_study": __version__, "preproc": preproc_version,
                     **decoded.metadata()["versions"]},
    }


def report_markdown(summary: dict[str, Any], out: Path) -> str:
    """Readable report with the key numbers of every procedure and links to the images."""
    d = summary["decoding"]
    f = summary["frame_filtering"]
    s = summary["shot_detection"]
    r = summary["redundancy_removal"]
    m = summary["miniclips"]
    th = f["thresholds"]
    src, tl, mapping = d["source"], d["timeline"], d["mapping"]

    def link(rel: str) -> str:
        return f"[{rel}]({rel})" if (out / rel).exists() else f"{rel} (not generated)"

    def image(rel: str, alt: str) -> str:
        return f"![{alt}]({rel})" if (out / rel).exists() else f"_{rel} not generated_"

    short_note = (f"; shots of {s['frames_per_clip']} to {s['window_length'] - 1} frames give one short-shot "
                  f"candidate of {s['frames_per_clip']} consecutive frames (r = 1)"
                  if s["short_shot_windows"] else "; short-shot candidates disabled")
    by_stride = ", ".join(f"r = {k}: {v}" for k, v in m["by_stride"].items())

    lines = [
        f"# Study of `{summary['id']}`",
        "",
        f"Video: `{summary['video']}`",
        "",
        "## Decoding",
        "",
        f"- Source: {src['codec']}, {src['width']}×{src['height']} (rotation {src['rotation']}°, SAR {src['sar']:.3f}), "
        f"{_num(src['fps'])} fps, {_num(src['duration_s'])} s.",
        f"- Constant-rate timeline: **{tl['frames']} frames** at {tl['fps']} fps ({tl['duration_s']} s).",
    ]
    if mapping:
        lines.append(f"- Estimated mapping: {mapping['source_packets']} source frames, "
                     f"{mapping['repeated_frames']} repeated, {mapping['skipped_source_frames']} skipped.")
    lines += [
        f"- Data: {link(f'{DECODING}/metadata.json')}, {link(f'{DECODING}/frame_mapping.csv')}",
        "",
        "## Frame filtering",
        "",
        f"Thresholds: θ_L = {th['luma_threshold']}, θ_σ = {th['std_threshold']}, θ_E = {th['static_threshold']} "
        f"for ≥ {th['static_min_frames']} frames.",
        "",
        "| Frames | Black | Uniform | Static | Removed | Kept | Segments | Removed runs |",
        "|---|---|---|---|---|---|---|---|",
        f"| {f['frames']} | {f['black']} | {f['uniform']} | {f['static']} | {f['removed']} "
        f"({f['removed_fraction']:.1%}) | {f['kept']} | {f['segments']} | {f['removed_runs']} |",
        "",
        image(f"{FRAME_FILTERING}/series.png", "frame filtering series"),
        "",
        f"Strips of each removed run: `{FRAME_FILTERING}/removed_runs/`. "
        f"Data: {link(f'{FRAME_FILTERING}/frames.csv')}, {link(f'{FRAME_FILTERING}/removed_runs.csv')}, "
        f"{link(f'{FRAME_FILTERING}/segments.csv')}",
        "",
        "## Shot detection",
        "",
        f"TransNet V2 threshold p > {s['threshold']}; windows of L = T·r = {s['frames_per_clip']}·{s['stride']} "
        f"= {s['window_length']} frames{short_note}.",
        "",
        "| Segments | Transitions | Shots | Useful time | Shots with a window | Shots with a short-shot candidate "
        "| Shots without candidate | Windows | Short-shot candidates |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {s['segments']} | {s['transitions']} | {s['shots']} | {s['useful_duration_s']} s | "
        f"{s['shots_with_window']} | {s['shots_with_short_candidate']} | {s['shots_without_candidate']} | "
        f"{s['windows']} | {s['short_shot_candidates']} |",
        "",
        image(f"{SHOT_DETECTION}/probabilities.png", "transition probabilities"),
        "",
        image(f"{SHOT_DETECTION}/shot_lengths.png", "shot lengths"),
        "",
        f"Strips of each transition: `{SHOT_DETECTION}/transitions/`. Stride sweep: "
        f"{link(f'{SHOT_DETECTION}/stride_sweep.csv')}. Data: {link(f'{SHOT_DETECTION}/shots.csv')}, "
        f"{link(f'{SHOT_DETECTION}/candidates.csv')}, {link(f'{SHOT_DETECTION}/probabilities.csv')}",
        "",
        "## Redundancy removal",
        "",
        f"θ_H = {r['hamming_threshold']}, one pHash every {r['phash_step']} frames inside each window. "
        "Only windows are compared; short-shot candidates are always kept. "
        "θ_H is the only parameter that controls the number of miniclips.",
        "",
        "| Windows | Kept | Discarded |",
        "|---|---|---|",
        f"| {r['candidate_windows']} | {r['kept_windows']} | {r['discarded_windows']} |",
        "",
        image(f"{REDUNDANCY_REMOVAL}/hamming_threshold_sweep.png", "theta_H sweep"),
        "",
        f"Distance matrices: `{REDUNDANCY_REMOVAL}/matrices/`. Discarded windows next to their reference: "
        f"`{REDUNDANCY_REMOVAL}/discarded/`. θ_H sweep: {link(f'{REDUNDANCY_REMOVAL}/hamming_threshold_sweep.csv')}. "
        f"Data: {link(f'{REDUNDANCY_REMOVAL}/decisions.csv')}, {link(f'{REDUNDANCY_REMOVAL}/signatures.csv')}",
        "",
        "## Miniclips",
        "",
        "Every kept candidate is a miniclip, in temporal order (no selection by motion).",
        "",
        "| Kept candidates | Miniclips | From windows | From short shots | From the longest shot | By stride "
        "| Time spanned | Useful time | Case |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {m['kept_candidates']} | **{m['miniclips']}** | {m['from_windows']} | {m['from_short_shots']} | "
        f"{m['from_longest_shot']} | {by_stride} | {m['spanned_s']} s | {m['useful_duration_s']} s | "
        f"`{m['fallback_case']}` |",
        "",
        f"Data: {link(f'{MINICLIPS}/miniclips.csv')}, {link(f'{MINICLIPS}/miniclips.json')}",
        "",
        "## Swin input",
        "",
    ]
    t = summary.get("swin_input")
    if t:
        st = t["storage"]
        psnr = st["psnr"]
        size = f"{st['bytes'] / 2 ** 20:.2f} MB" if st["bytes"] is not None else "not measured"
        lines += [
            f"- {st['files']} files `{SWIN_INPUT}/miniclip_<k>.mp4`, the same as `preproc`'s "
            f"`miniclips/<id>/<id>_<k>.mp4`: {st['format']}, {size}, played at {st['playback_fps']} fps (fps / r).",
        ]
        if psnr["min_db"] is not None:
            lines.append(f"- Storage loss (PSNR of the decoded .mp4 against the extracted frames): "
                         f"mean {psnr['mean_db']} dB, minimum {psnr['min_db']} dB.")
        lines += [
            f"- Tensor per miniclip {t['tensor_per_miniclip']} ({t['tensor_layout']}, {t['tensor_dtype']}); "
            f"statistics of the {t['statistics_of']}.",
            f"- Per-channel mean before / after normalization: {t['before_normalization']['mean']} / "
            f"{t['after_normalization']['mean']}.",
            f"- Per-channel std before / after normalization: {t['before_normalization']['std']} / "
            f"{t['after_normalization']['std']}.",
            f"- Alignment check (max error against the analysis luma): {t['alignment_error']:.2f}.",
            f"- Contact sheets: `{SWIN_INPUT}/miniclip_<k>.png`. Data: {link(f'{SWIN_INPUT}/tensors.json')}",
        ]
    lines += [
        "",
        "## Overview",
        "",
        image("timeline.png", "timeline"),
        "",
        f"Annotated video: {link('annotated_video.mp4')}",
        "",
    ]
    ver = summary.get("verification")
    if ver is not None:
        lines += ["## Verification", "",
                  f"Against `preproc.pipeline.process_video`: {'all items match' if ver['all_match'] else '**differences found**'} "
                  f"({link('verification.md')}).", ""]
    lines += ["## Warnings", ""]
    lines += [f"- {w}" for w in summary["warnings"]] or ["None."]
    lines += ["", "## Timings (s)", "", "| Procedure | Seconds |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["timings_s"].items()]
    return "\n".join(lines) + "\n"


def _num(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)
