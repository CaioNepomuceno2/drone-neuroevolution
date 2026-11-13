"""Algoritmo genético: avaliação, seleção, cruzamento, mutação e checkpoints."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .brain import forward, random_population
from .config import DT, Difficulty
from .world import Course, Sim


@dataclass
class GAParams:
    pop: int = 200
    elite: int = 8              # os melhores passam direto para a próxima geração
    tournament: int = 4         # tamanho do torneio de seleção
    mut_rate: float = 0.12      # fração dos pesos mutados em cada filho
    mut_sigma: float = 0.30     # tamanho inicial da mutação
    sigma_decay: float = 0.99   # a mutação diminui aos poucos
    sigma_min: float = 0.04
    immigrants: int = 4         # indivíduos aleatórios novos a cada geração
    episodes: int = 2           # percursos diferentes por geração (média)
    max_seconds: float = 45.0   # duração máxima de cada voo
    workers: int = 1            # processos em paralelo (um percurso por processo)


def course_seed(base: int, gen: int, episode: int) -> int:
    return int(base) * 1_000_003 + int(gen) * 101 + int(episode)


def run_episode(genomes: np.ndarray, course: Course, max_seconds: float, seed: int = 0) -> Sim:
    """Voa a população inteira no percurso até todos baterem ou acabar o tempo."""
    sim = Sim(course, len(genomes), seed=seed, max_steps=int(max_seconds / DT))
    while not sim.done:
        sim.step(forward(genomes, sim.observe()))
    return sim


def _episode_task(args):
    genomes, cs, diff, max_seconds = args
    sim = run_episode(genomes, Course(cs, diff), max_seconds, seed=cs)
    return sim.score(), float(sim.alive.sum())


def evaluate(genomes: np.ndarray, base_seed: int, gen: int, diff: Difficulty, p: GAParams, pool=None):
    """Pontuação média de cada genoma em `episodes` percursos. Devolve (fitness, stats).

    Com `pool` (ProcessPoolExecutor) os percursos rodam em paralelo; o resultado é idêntico.
    """
    tasks = [(genomes, course_seed(base_seed, gen, e), diff, p.max_seconds) for e in range(p.episodes)]
    results = list(pool.map(_episode_task, tasks)) if pool is not None else [_episode_task(t) for t in tasks]
    fit = sum(r[0] for r in results) / p.episodes
    survivors = sum(r[1] for r in results)
    stats = {
        "best": float(fit.max()),
        "mean": float(fit.mean()),
        "median": float(np.median(fit)),
        "survivors": survivors / p.episodes,
    }
    return fit, stats


def next_generation(genomes: np.ndarray, fitness: np.ndarray, rng: np.random.Generator,
                    sigma: float, p: GAParams) -> np.ndarray:
    """Elitismo + torneio + cruzamento uniforme + mutação gaussiana + imigrantes."""
    n, g = genomes.shape
    order = np.argsort(-fitness)
    elite = genomes[order[:p.elite]].copy()
    n_child = n - p.elite - p.immigrants

    cand = rng.integers(0, n, size=(n_child, 2, p.tournament))
    winner = cand[np.arange(n_child)[:, None], np.arange(2)[None, :], fitness[cand].argmax(axis=2)]
    mask = rng.random((n_child, g)) < 0.5
    child = np.where(mask, genomes[winner[:, 0]], genomes[winner[:, 1]])
    mutate = rng.random((n_child, g)) < p.mut_rate
    child = child + mutate * rng.normal(0.0, sigma, (n_child, g))

    return np.concatenate([elite, child, random_population(p.immigrants, rng)])


# -- checkpoints (gravação atômica: o PC já desligou sozinho no meio de escritas) ----------
def save_npz_atomic(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    if path.exists():
        os.replace(path, path.with_suffix(".prev.npz"))   # guarda a versão anterior
    os.replace(tmp, path)


def load_checkpoint(out: Path):
    """Lê o checkpoint; se o arquivo principal estiver corrompido, tenta o anterior."""
    for name in ("checkpoint.npz", "checkpoint.prev.npz"):
        f = out / name
        if not f.exists():
            continue
        try:
            with np.load(f, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                return {"pop": z["pop"], "hist": z["hist"], "meta": meta}
        except Exception as exc:   # arquivo truncado ou corrompido
            print(f"[aviso] {f.name} ilegível ({exc}); tentando o anterior")
    return None
