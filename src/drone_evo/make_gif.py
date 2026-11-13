"""Recorta um trecho do vídeo e gera um GIF leve (para o README).

    uv run python -m drone_evo.make_gif out/demo.mp4 assets/demo.gif --start 53 --seconds 7
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .make_video import ffmpeg_exe


def make_gif(video: Path, out: Path, start: float, seconds: float, width: int = 640, fps: int = 12,
             colors: int = 96) -> None:
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds),
                    "-i", str(video), "-vf", vf, "-loop", "0", str(out)], check=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Gera um GIF a partir de um trecho do vídeo.")
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args(argv)
    make_gif(a.video, a.out, a.start, a.seconds, a.width, a.fps)
    print(f"{a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
