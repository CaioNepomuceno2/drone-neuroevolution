"""Escolhe o melhor drone de uma execução avaliando os candidatos em muitos percursos novos.

Os melhores de cada geração foram avaliados em apenas 2 percursos; aqui os finalistas das
últimas gerações disputam o mesmo conjunto de percursos inéditos e o de maior média vence.

    uv run python -m drone_evo.select_best runs/normal --difficulty normal --out models/best_normal.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import DIFFICULTIES, Difficulty
from .evolution import run_episode
from .world import Course

SELECT_SEED = 777_000      # percursos de seleção
REPORT_SEED = 888_000      # percursos separados só para o relatório final


def generation_files(run_dir: Path) -> list[Path]:
    return sorted(Path(run_dir).glob("gen_????.npz"))


def load_generation(run_dir: Path, gen: int | None = None, top: int | None = None) -> tuple[np.ndarray, int]:
    """Genomas de uma geração (do melhor ao pior). `gen=None` -> a última. Devolve (genomas, nº da geração)."""
    files = generation_files(run_dir)
    if not files:
        raise FileNotFoundError(f"nenhuma geração em {run_dir}")
    f = files[-1] if gen is None else Path(run_dir) / f"gen_{gen:04d}.npz"
    with np.load(f) as z:
        g = z["genomes"].astype(float)
    return (g[:top] if top else g), int(f.stem.split("_")[1])


def evaluate_on(genomes: np.ndarray, diff: Difficulty, seed0: int, courses: int, max_seconds: float):
    """Média por genoma em `courses` percursos: (pontos, isoladores, fração que chegou ao fim)."""
    pts = np.zeros(len(genomes))
    ins = np.zeros(len(genomes))
    fin = np.zeros(len(genomes))
    for e in range(courses):
        sim = run_episode(genomes, Course(seed0 + e, diff), max_seconds, seed=seed0 + e)
        pts += sim.score()
        ins += sim.inspections
        fin += sim.alive
    return pts / courses, ins / courses, fin / courses


def select_best(run_dir: Path, diff: Difficulty, last: int = 12, top: int = 10,
                courses: int = 24, max_seconds: float = 90.0):
    cands = []
    for f in generation_files(run_dir)[-last:]:
        with np.load(f) as z:
            cands.append(z["genomes"][:top].astype(float))
    cands = np.unique(np.concatenate(cands), axis=0)
    pts, _, _ = evaluate_on(cands, diff, SELECT_SEED, courses, max_seconds)
    best = cands[int(np.argmax(pts))]
    # relatório em percursos que NÃO participaram da seleção
    rp, ri, rf = evaluate_on(best[None, :], diff, REPORT_SEED, courses, max_seconds)
    report = {"candidatos": len(cands), "pontos": float(rp[0]), "isoladores": float(ri[0]),
              "chegou_ao_fim": float(rf[0]), "percursos": courses, "segundos_max": max_seconds}
    return best, report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Seleciona o melhor drone de uma execução.")
    ap.add_argument("run", type=Path)
    ap.add_argument("--difficulty", choices=list(DIFFICULTIES), default="normal")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--courses", type=int, default=24)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    a = ap.parse_args(argv)
    best, rep = select_best(a.run, DIFFICULTIES[a.difficulty], courses=a.courses, max_seconds=a.max_seconds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, best)
    print(f"salvo em {a.out}")
    print(f"  média {rep['pontos']:.0f} pontos e {rep['isoladores']:.1f} isoladores por voo; "
          f"chegou aos {rep['segundos_max']:.0f} s em {100 * rep['chegou_ao_fim']:.0f}% dos "
          f"{rep['percursos']} percursos novos")


if __name__ == "__main__":
    main()
