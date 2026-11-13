"""Treino headless (sem janela) com checkpoint a cada geração, retomável.

    uv run python -m drone_evo.train --out runs/normal --gens 100
    uv run python -m drone_evo.train --out runs/hard --difficulty hard --init-from runs/normal

Se o processo for interrompido (ou o PC desligar), é só rodar o mesmo comando de novo:
o treino continua da última geração salva, com o mesmo estado do gerador aleatório.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .brain import GENOME_SIZE, random_population
from .config import DIFFICULTIES
from .evolution import (
    GAParams, evaluate, load_checkpoint, next_generation, save_npz_atomic,
)

HIST_COLS = ("best", "mean", "median", "survivors", "seconds")


@dataclass
class TrainConfig:
    out: Path
    gens: int = 100
    difficulty: str = "normal"
    seed: int = 1
    init_from: Path | None = None
    fresh: bool = False
    ga: GAParams = field(default_factory=GAParams)


def load_population(run_dir: Path) -> np.ndarray:
    """População final de uma execução anterior (para continuar em outra dificuldade)."""
    ck = load_checkpoint(Path(run_dir))
    if ck is None:
        raise FileNotFoundError(f"nenhum checkpoint em {run_dir}")
    return ck["pop"]


def write_history_csv(out: Path, hist: np.ndarray) -> None:
    lines = ["geracao," + ",".join(HIST_COLS)]
    for i, row in enumerate(hist, start=1):
        lines.append(f"{i}," + ",".join(f"{v:.4f}" for v in row))
    tmp = out / "history.tmp.csv"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(out / "history.csv")


def train(cfg: TrainConfig, log=print):
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    diff = DIFFICULTIES[cfg.difficulty]
    p = cfg.ga

    ck = None if cfg.fresh else load_checkpoint(out)
    rng = np.random.default_rng(cfg.seed)
    if ck is None:
        gen = 0
        hist = np.zeros((0, len(HIST_COLS)))
        pop = random_population(p.pop, rng)
        if cfg.init_from is not None:
            seed_pop = load_population(cfg.init_from)
            pop[: min(len(pop), len(seed_pop))] = seed_pop[: len(pop)]
            log(f"população inicial copiada de {cfg.init_from}")
        (out / "config.json").write_text(json.dumps({
            "difficulty": cfg.difficulty, "seed": cfg.seed, "genome_size": GENOME_SIZE,
            "ga": asdict(p), "init_from": str(cfg.init_from) if cfg.init_from else None,
        }, indent=2), encoding="utf-8")
    else:
        meta = ck["meta"]
        gen = int(meta["gen"])
        hist = ck["hist"]
        pop = ck["pop"]
        rng.bit_generator.state = meta["rng_state"]
        log(f"retomando da geração {gen} ({out})")

    pool = ProcessPoolExecutor(max_workers=min(p.workers, p.episodes)) if p.workers > 1 else None
    try:
        pop, hist = _loop(cfg, diff, pool, pop, hist, gen, rng, log)
    finally:
        if pool is not None:
            pool.shutdown()
    return pop, hist


def _loop(cfg, diff, pool, pop, hist, gen, rng, log):
    out, p = Path(cfg.out), cfg.ga
    while gen < cfg.gens:
        t0 = time.perf_counter()
        fit, stats = evaluate(pop, cfg.seed, gen, diff, p, pool)
        order = np.argsort(-fit)

        # população desta geração, do melhor para o pior (usada nos vídeos e no visualizador)
        save_npz_atomic(out / f"gen_{gen + 1:04d}.npz",
                        genomes=pop[order].astype(np.float32), fitness=fit[order].astype(np.float32))

        sigma = max(p.sigma_min, p.mut_sigma * p.sigma_decay ** gen)
        pop_next = next_generation(pop, fit, rng, sigma, p)
        secs = time.perf_counter() - t0
        hist = np.vstack([hist, [stats["best"], stats["mean"], stats["median"], stats["survivors"], secs]])
        gen += 1

        save_npz_atomic(out / "checkpoint.npz", pop=pop_next, hist=hist, meta=json.dumps({
            "gen": gen, "difficulty": cfg.difficulty, "seed": cfg.seed,
            "rng_state": rng.bit_generator.state,
        }))
        write_history_csv(out, hist)
        log(f"ger {gen:4d} | melhor {stats['best']:7.1f} | média {stats['mean']:6.1f} | "
            f"mediana {stats['median']:6.1f} | vivos no fim {stats['survivors']:5.1f} | {secs:4.1f}s")
        pop = pop_next
    return pop, hist


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Treina os drones com algoritmo genético.")
    ap.add_argument("--out", type=Path, required=True, help="pasta da execução (checkpoints e gerações)")
    ap.add_argument("--gens", type=int, default=100, help="total de gerações (vale o total, não o adicional)")
    ap.add_argument("--difficulty", choices=list(DIFFICULTIES), default="normal")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=2, help="percursos por geração")
    ap.add_argument("--max-seconds", type=float, default=45.0, help="duração máxima de cada voo")
    ap.add_argument("--workers", type=int, default=1, help="processos em paralelo (até o nº de percursos)")
    ap.add_argument("--init-from", type=Path, default=None, help="começar da população de outra execução")
    ap.add_argument("--fresh", action="store_true", help="ignora checkpoint existente e recomeça")
    a = ap.parse_args(argv)
    cfg = TrainConfig(out=a.out, gens=a.gens, difficulty=a.difficulty, seed=a.seed,
                      init_from=a.init_from, fresh=a.fresh,
                      ga=GAParams(pop=a.pop, episodes=a.episodes, max_seconds=a.max_seconds, workers=a.workers))
    train(cfg)


if __name__ == "__main__":
    main()
