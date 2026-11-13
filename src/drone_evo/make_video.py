"""Gera o vídeo de demonstração (MP4 com trilha sonora e efeitos) a partir dos treinos.

    uv run python -m drone_evo.make_video --normal runs/normal --hard runs/hard --out out/demo.mp4

Os quadros são desenhados fora da tela e enviados direto ao ffmpeg (o binário que vem no
pacote imageio-ffmpeg), sem gravação de tela. O áudio é sintetizado em código e mixado nos
instantes exatos de cada evento (batidas, inspeções, troca de geração).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import build_soundtrack, write_wav
from .brain import forward
from .config import DT, HARD, N_RAYS, NORMAL, Difficulty
from .play import load_history
from .render import CYAN, GOLD, INK, MUTED, Renderer
from .select_best import generation_files, load_generation
from .world import Course, Sim

SEED_NORMAL = 4242       # mesmo percurso em todas as gerações do modo normal (dá para comparar)
SEED_HARD = 9237


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class FrameSink:
    """Recebe quadros RGB e grava um MP4 (sem áudio) via pipe para o ffmpeg."""

    def __init__(self, path: Path, w: int, h: int, fps: int):
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("o ffmpeg falhou ao codificar o vídeo")


@dataclass
class Scene:
    name: str
    kind: str                                   # "card" ou "flight"
    seconds: float = 0.0                        # card: duração; flight: mínimo (segura o último quadro)
    max_seconds: float = 0.0                    # flight: duração máxima
    lines: list = field(default_factory=list)   # card
    captions: list = field(default_factory=list)  # flight: (t0, t1, texto)
    genomes: np.ndarray | None = None
    diff: Difficulty = NORMAL
    seed: int = 0
    hist: np.ndarray | None = None
    gen_label: str = "1"
    title: str = "GERAÇÃO"


def build_scenes(a) -> list[Scene]:
    show = a.show
    n_hist = load_history(a.normal)
    n = len(generation_files(a.normal))
    picks = [1, max(2, round(n / 50)), max(3, round(n / 17)), max(4, round(n / 6)), n]
    ng = {g: load_generation(a.normal, g, show)[0] for g in dict.fromkeys(picks)}

    sc: list[Scene] = [Scene("intro", "card", 3.6, lines=[
        ("Ensinei drones a inspecionar", 50, INK, True),
        ("a rede elétrica", 50, GOLD, True),
        ("", 18, INK, False),
        ("sem programar nenhuma regra: só evolução", 26, MUTED, False),
        ("Python + NumPy, do zero", 22, CYAN, False)])]

    def fl(name, gen, caps, mn, mx):
        sc.append(Scene(name, "flight", mn, mx, captions=caps, genomes=ng[gen], diff=NORMAL, seed=SEED_NORMAL,
                        hist=None if n_hist is None else n_hist[:gen], gen_label=str(gen)))

    fl("gen_inicial", picks[0], [(0.3, 4.4, f"Geração {picks[0]}: {len(ng[picks[0]])} drones com cérebros aleatórios"),
                                 (4.7, 9.4, f"Cada um vê o mundo com {N_RAYS} raios de LiDAR\ne decide uma coisa só: o empuxo")], 9.4, 11.0)
    fl("gen_pais", picks[1], [(0.3, 4.6, f"Geração {picks[1]}: quem chega mais longe\nvira pai da próxima geração"),
                              (4.9, 8.6, "Cruzamento + mutação dos pesos da rede")], 8.6, 9.5)
    fl("gen_desvio", picks[2], [(0.3, 5.6, f"Geração {picks[2]}: já desviam de árvores e pássaros")], 6.0, 8.5)
    fl("gen_cabo", picks[3], [(0.3, 5.8, f"Geração {picks[3]}: voam rente ao cabo\ne inspecionam os isoladores")], 7.0, 8.5)
    fl("gen_final", picks[4], [(0.3, 5.4, f"Geração {picks[4]}: a rede domina o percurso"),
                               (6.0, 12.0, "Nenhuma regra foi programada.\nSó os pesos de uma rede neural, evoluídos.")], 12.0, 15.0)

    if a.hard is not None and generation_files(a.hard):
        h_hist = load_history(a.hard)
        nh = len(generation_files(a.hard))
        hg1 = load_generation(a.hard, 1, show)[0]
        hgn = load_generation(a.hard, nh, show)[0]
        sc.append(Scene("card_dificil", "card", 3.2, lines=[
            ("Agora, o modo difícil", 50, GOLD, True), ("", 18, INK, False),
            ("vento em rajadas · pássaros mais rápidos · mais velocidade", 26, INK, False)]))
        sc.append(Scene("dificil_inicial", "flight", 7.5, 9.0, captions=[
            (0.3, 5.2, "Os mesmos drones, agora com vento:\nde volta ao caos")], genomes=hg1, diff=HARD, seed=SEED_HARD,
            hist=None if h_hist is None else h_hist[:1], gen_label="1", title="GERAÇÃO · DIFÍCIL"))
        sc.append(Scene("dificil_final", "flight", 14.0, 17.0, captions=[
            (0.3, 5.4, f"Depois de {nh} gerações no modo difícil"),
            (6.0, 13.0, "Ele nunca tinha visto este percurso.\nSó aprendeu a voar.")], genomes=hgn, diff=HARD, seed=SEED_HARD,
            hist=h_hist, gen_label=str(nh), title="GERAÇÃO · DIFÍCIL"))

    sc.append(Scene("final", "card", 5.5, lines=[
        ("Tudo do zero em Python + NumPy", 44, INK, True), ("", 14, INK, False),
        ("rede neural + algoritmo genético, sem bibliotecas de IA", 24, MUTED, False), ("", 22, INK, False),
        (a.repo_url, 30, CYAN, True)]))
    return sc


def fade(t: float, t0: float, t1: float, edge: float = 0.3) -> int:
    return int(255 * max(0.0, min(1.0, (t - t0) / edge, (t1 - t) / edge)))


class Producer:
    def __init__(self, a):
        self.a = a
        w, h = (int(v) for v in a.size.lower().split("x"))
        self.r = Renderer(w, h, headless=True)
        self.fps = a.fps
        self.sink = None
        self.frames = 0
        self.events: list = []
        self.rotor_spans: list = []
        self.manifest: dict = {}
        self._last_crash_t = -1.0

    @property
    def t(self) -> float:
        return self.frames / self.fps

    def push(self) -> None:
        if self.sink is not None:
            self.sink.write(self.r.frame_bytes())
        self.frames += 1

    def card(self, sc: Scene) -> None:
        r = self.r
        t0 = self.t
        self.events.append((t0 + 0.05, "whoosh", 0.35))
        for f in range(int(sc.seconds * self.fps)):
            ts = f / self.fps
            r.background_only(120.0 * (t0 + ts))
            r.card(sc.lines, dim=130, alpha=fade(ts, 0.0, sc.seconds, 0.45))
            self.push()
        self.manifest[sc.name] = (round(t0, 2), round(self.t, 2))

    def flight(self, sc: Scene) -> None:
        r, fps = self.r, self.fps
        sim = Sim(Course(sc.seed, sc.diff), len(sc.genomes), seed=sc.seed, max_steps=int(sc.max_seconds / DT))
        r.reset()
        t0 = self.t
        self.events.append((t0 + 0.1, "genup", 0.5))
        obs = sim.observe()
        crashes = 0
        spf = round(1.0 / (fps * DT))          # passos de simulação por quadro
        f = 0
        while True:
            ts = f / fps
            if sim.done and ts >= sc.seconds:
                break
            for _ in range(spf):
                if sim.done:
                    break
                sim.step(forward(sc.genomes, obs))
                r.note_events(sim)
                obs = sim.observe()
                crashes += len(sim.crashed_now)
                if r.cur_leader in set(sim.inspected_now.tolist()):
                    self.events.append((t0 + ts, "ding", 0.5))
            cap, alpha = None, 255
            for c0, c1, txt in sc.captions:
                if c0 <= ts <= c1:
                    cap, alpha = txt, fade(ts, c0, c1)
            r.draw(sim, genomes=sc.genomes, obs=obs, hist=sc.hist, gen_label=sc.gen_label, caption=cap,
                   caption_alpha=alpha, dt=1.0 / fps, title=sc.title)
            self.push()
            f += 1
            if crashes and t0 + ts - self._last_crash_t > 0.12:
                self.events.append((t0 + ts, "crash", min(0.55, 0.12 + 0.07 * np.sqrt(crashes))))
                self._last_crash_t = t0 + ts
                crashes = 0
        self.rotor_spans.append((t0, self.t))
        self.manifest[sc.name] = (round(t0, 2), round(self.t, 2))

    def natural_length(self, sc: Scene) -> float:
        """Quanto dura o voo da cena antes de todos caírem (sem desenhar nada)."""
        sim = Sim(Course(sc.seed, sc.diff), len(sc.genomes), seed=sc.seed, max_steps=int(sc.max_seconds / DT))
        while not sim.done:
            sim.step(forward(sc.genomes, sim.observe()))
        return sim.step_i * DT


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Gera o vídeo de demonstração.")
    ap.add_argument("--normal", type=Path, required=True, help="execução do modo normal")
    ap.add_argument("--hard", type=Path, default=None, help="execução do modo difícil")
    ap.add_argument("--out", type=Path, default=Path("out/demo.mp4"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30, choices=(30, 60))
    ap.add_argument("--show", type=int, default=120, help="drones desenhados por cena")
    ap.add_argument("--repo-url", default="github.com/CaioNepomuceno2/drone-neuroevolution")
    ap.add_argument("--dry-run", action="store_true", help="só mede a duração natural de cada cena")
    a = ap.parse_args(argv)

    scenes = build_scenes(a)
    prod = Producer(a)
    if a.dry_run:
        for sc in scenes:
            if sc.kind == "flight":
                print(f"{sc.name:18s} voo natural {prod.natural_length(sc):5.1f}s  (mín {sc.seconds}s, máx {sc.max_seconds}s)")
        return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    silent = a.out.with_name(a.out.stem + "_sem_audio.mp4")
    prod.sink = FrameSink(silent, prod.r.w, prod.r.h, a.fps)
    for sc in scenes:
        print(f"cena {sc.name} ...", flush=True)
        prod.card(sc) if sc.kind == "card" else prod.flight(sc)
    prod.sink.close()

    duration = prod.frames / a.fps
    print(f"vídeo mudo pronto: {duration:.1f} s; sintetizando o áudio")
    audio = build_soundtrack(duration, prod.events, music_gain=0.5, rotor_gain=0.05, rotor_spans=prod.rotor_spans)
    wav = a.out.with_suffix(".wav")
    write_wav(wav, audio)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-shortest", str(a.out)], check=True)
    silent.unlink()
    wav.unlink()
    a.out.with_suffix(".json").write_text(json.dumps(
        {"duracao_s": round(duration, 2), "cenas": prod.manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"pronto: {a.out} ({duration:.1f} s)")


if __name__ == "__main__":
    main()
