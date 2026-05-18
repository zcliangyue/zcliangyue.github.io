#!/usr/bin/env python3
"""
Compose multiple videos into a single frame using a row-major RxC grid (via FFmpeg xstack).

Requires FFmpeg (https://ffmpeg.org/). Optional: pip install imageio-ffmpeg to use a bundled binary.

Note: This file lives under tools/ (not Hexo's scripts/) so `hexo deploy` does not try to load it.

Examples (from repo root):
  python tools/video_matrix.py -r 1 -c 2 left.mp4 right.mp4 -o out.mp4
  python tools/video_matrix.py -r 2 -c 2 a.mp4 b.mp4 c.mp4 d.mp4 -o grid.mp4
  python tools/video_matrix.py -r 1 -c 2 a.mp4 b.mp4 -o out.mp4 --cell-width 960 --cell-height 540
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"--ffmpeg path not found: {explicit}")

    which = shutil.which("ffmpeg")
    if which:
        return which

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise RuntimeError(
        "FFmpeg not found. Install FFmpeg and add it to PATH, or run: pip install imageio-ffmpeg"
    )


def run_ffmpeg(args: list[str], ffmpeg: str) -> None:
    cmd = [ffmpeg, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout or "ffmpeg failed\n")
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)


def build_filter_complex(rows: int, cols: int, cell_w: int, cell_h: int) -> str:
    n = rows * cols
    parts: list[str] = []
    for i in range(n):
        parts.append(
            f"[{i}:v]scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease:"
            f"flags=lanczos,pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )
    layout_parts: list[str] = []
    for r in range(rows):
        for c in range(cols):
            layout_parts.append(f"{c * cell_w}_{r * cell_h}")
    layout = "|".join(layout_parts)
    labels = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{labels}xstack=inputs={n}:layout={layout}:fill=black[outv]")
    return ";".join(parts)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tile videos into an RxC matrix (row-major input order)."
    )
    p.add_argument("-r", "--rows", type=int, required=True, help="Number of rows.")
    p.add_argument("-c", "--cols", type=int, required=True, help="Number of columns.")
    p.add_argument(
        "videos",
        nargs="+",
        help="Input videos in row-major order (len = rows * cols).",
    )
    p.add_argument("-o", "--output", required=True, help="Output video path (.mp4).")
    p.add_argument(
        "--cell-width",
        type=int,
        default=640,
        help="Each cell width in pixels after letterbox/pad (default: 640).",
    )
    p.add_argument(
        "--cell-height",
        type=int,
        default=360,
        help="Each cell height in pixels (default: 360).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=20,
        help="libx264 CRF (lower is higher quality; default: 20).",
    )
    p.add_argument(
        "--preset",
        default="medium",
        help="libx264 preset (default: medium).",
    )
    p.add_argument(
        "--audio",
        choices=("first", "none"),
        default="first",
        help="Use audio from the first input only, or strip all audio (default: first).",
    )
    p.add_argument(
        "--ffmpeg",
        dest="ffmpeg_path",
        default=None,
        help="Path to ffmpeg executable (optional).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    rows, cols = args.rows, args.cols
    n_expected = rows * cols
    if len(args.videos) != n_expected:
        print(
            f"Expected {n_expected} video(s) (rows*cols), got {len(args.videos)}.",
            file=sys.stderr,
        )
        return 2

    ffmpeg = resolve_ffmpeg(args.ffmpeg_path)
    cell_w, cell_h = args.cell_width, args.cell_height
    graph = build_filter_complex(rows, cols, cell_w, cell_h)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["-y"]
    for v in args.videos:
        if not Path(v).is_file():
            print(f"Missing file: {v}", file=sys.stderr)
            return 2
        cmd.extend(["-i", str(Path(v).resolve())])
    cmd.extend(["-filter_complex", graph, "-map", "[outv]"])

    if args.audio == "first":
        cmd.extend(["-map", "0:a:0?", "-shortest"])
    else:
        cmd.extend(["-an"])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if args.audio == "first":
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd.append(str(out.resolve()))

    try:
        run_ffmpeg(cmd, ffmpeg)
    except subprocess.CalledProcessError:
        return 1

    print(f"Wrote {out} ({rows}x{cols} grid, cell {cell_w}x{cell_h}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
