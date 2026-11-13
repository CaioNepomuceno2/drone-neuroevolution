"""Visualizador ao vivo, com som.

    uv run python -m drone_evo.play --run runs/normal            # assiste à última geração
    uv run python -m drone_evo.play --run runs/normal --gen 1    # a geração 1 (o caos)
    uv run python -m drone_evo.play --model models/best_hard.npy --hard
    uv run python -m drone_evo.play --run runs/normal --human    # você contra a IA

Teclas: ESPAÇO/P pausa · N, B próxima/anterior geração · H dificuldade · R novo percurso ·
M som · ESC sai. No modo humano: ↑/W/ESPAÇO sobe, ↓/S desce, sem tecla = pairar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .audio import AudioPlayer
from .brain import GENOME_SIZE, forward
from .config import DT, HARD, NORMAL
from .render import MUTED, Renderer, fmt
from .select_best import generation_files, load_generation
from .world import Course, Sim


def load_history(run: Path | None) -> np.ndarray | None:
    f = None if run is None else Path(run) / "history.csv"
    if f is None or not f.exists():
        return None
    return np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)[:, 1:]


class Viewer:
    def __init__(self, a):
        self.a = a
        self.run = a.run
        self.diff = HARD if a.hard else NORMAL
        self.human = a.human
        self.hist = load_history(self.run)
        self.gen = a.gen
        self.n_gens = len(generation_files(self.run)) if self.run else 1
        self.seed = a.seed
        self.paused = False
        self.model = np.load(a.model) if a.model else None
        w, h = (int(v) for v in a.size.lower().split("x"))
        self.r = Renderer(w, h, headless=False)
        self.audio = AudioPlayer(enabled=not a.mute, music=not a.no_music)
        self.load()
        self.new_sim()

    # ------------------------------------------------------------------ estado
    def load(self) -> None:
        top = self.a.top if not self.human else min(self.a.top, 30)
        if self.model is not None:
            ai = self.model[None, :]
            self.gen_label = "modelo"
        else:
            ai, g = load_generation(self.run, self.gen, top)
            self.gen = g
            self.gen_label = str(g)
        self.genomes = np.concatenate([np.zeros((1, GENOME_SIZE)), ai]) if self.human else ai

    def new_sim(self) -> None:
        self.seed += 1
        self.sim = Sim(Course(self.seed, self.diff), len(self.genomes), seed=self.seed, max_steps=int(600 / DT))
        self.r.reset()
        self.obs = self.sim.observe()
        self.h_thrust = 0.5
        self.dead_frames = 0

    def change_gen(self, delta: int) -> None:
        if self.model is not None or not self.run:
            return
        new = int(np.clip(self.gen + delta, 1, self.n_gens))
        if new != self.gen:
            self.gen = new
            self.load()
            self.new_sim()
            self.audio.play("genup", 0.6)

    # ------------------------------------------------------------------ laço
    def step(self, keys) -> None:
        pg = self.r.pg
        thrust = forward(self.genomes, self.obs)
        if self.human:
            up = keys[pg.K_UP] or keys[pg.K_w] or keys[pg.K_SPACE]
            down = keys[pg.K_DOWN] or keys[pg.K_s]
            target = 0.92 if up else 0.08 if down else 0.5
            self.h_thrust += (target - self.h_thrust) * 0.35
            thrust[0] = self.h_thrust
        self.sim.step(thrust)
        self.r.note_events(self.sim)
        self.obs = self.sim.observe()
        sim = self.sim
        lead = self.r.cur_leader
        if len(sim.crashed_now):
            self.audio.crash(len(sim.crashed_now), sim.t)
        if lead in set(sim.inspected_now.tolist()):
            self.audio.play("ding", 0.55)

    def frame(self) -> None:
        sim, r = self.sim, self.r
        hist = None if self.hist is None else self.hist[: self.gen]
        if self.human:
            ai_best = float(sim.score()[1:].max())
            r.draw(sim, genomes=self.genomes, obs=self.obs, hist=hist, gen_label=self.gen_label, follow=0,
                   nn_index=1 + int(np.argmax(sim.score()[1:])), score_label="Seus pontos",
                   rival=("Melhor IA", fmt(ai_best)), marker="VOCÊ")
        else:
            r.draw(sim, genomes=self.genomes, obs=self.obs, hist=hist, gen_label=self.gen_label)
        hint = ("↑/W/ESPAÇO sobe · ↓/S desce · P pausa · H dificuldade · R reinicia · M som · ESC sai"
                if self.human else
                "ESPAÇO pausa · N/B geração · H dificuldade · R reinicia · M som · ESC sai")
        r.text(hint, 12, MUTED, (r.w / 2, r.h - 8 * r.u), anchor="midbottom")
        if self.paused:
            r.text("PAUSADO", 64, (255, 255, 255), (r.w / 2, r.h / 2), bold=True, anchor="center")
        r.present()

    def run_loop(self) -> None:
        pg = self.r.pg
        clock = pg.time.Clock()
        running = True
        frames = 0
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    if ev.key in (pg.K_ESCAPE, pg.K_q):
                        running = False
                    elif ev.key == pg.K_p or (ev.key == pg.K_SPACE and not self.human):
                        self.paused = not self.paused
                    elif ev.key == pg.K_n:
                        self.change_gen(+1)
                    elif ev.key == pg.K_b:
                        self.change_gen(-1)
                    elif ev.key == pg.K_h:
                        self.diff = NORMAL if self.diff is HARD else HARD
                        self.new_sim()
                    elif ev.key == pg.K_r:
                        self.new_sim()
                    elif ev.key == pg.K_m:
                        self.audio.toggle_mute()
            if not self.paused:
                self.step(pg.key.get_pressed())
                gone = (not self.sim.alive[0]) if self.human else self.sim.done
                self.dead_frames = self.dead_frames + 1 if gone or self.sim.done else 0
                if self.dead_frames > (150 if self.human else 100):
                    self.new_sim()
            self.frame()
            frames += 1
            if self.a.max_frames and frames >= self.a.max_frames:
                running = False
            clock.tick(0 if self.a.max_frames else 60)
        pg.quit()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Assiste (ou joga contra) os drones treinados.")
    ap.add_argument("--run", type=Path, default=None, help="pasta de uma execução de treino")
    ap.add_argument("--model", type=Path, default=None, help="um genoma salvo (.npy), em vez de uma geração")
    ap.add_argument("--gen", type=int, default=None, help="geração a mostrar (padrão: a última)")
    ap.add_argument("--top", type=int, default=120, help="quantos drones da geração desenhar")
    ap.add_argument("--hard", action="store_true", help="começar no modo difícil")
    ap.add_argument("--human", action="store_true", help="você controla um drone contra a IA")
    ap.add_argument("--seed", type=int, default=100, help="semente do primeiro percurso")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)   # só para testes
    a = ap.parse_args(argv)
    if a.run is None and a.model is None:
        ap.error("informe --run ou --model")
    Viewer(a).run_loop()


if __name__ == "__main__":
    main()
