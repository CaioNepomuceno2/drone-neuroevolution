"""Tabela de resultados em Markdown, medida em percursos inéditos (os mesmos do relatório de `select_best`).

    uv run python -m drone_evo.report --normal runs/normal --models models
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import DT, HARD, NORMAL, Difficulty
from .select_best import REPORT_SEED, evaluate_on, load_generation
from .world import Course, Sim


def hover_baseline(diff: Difficulty, courses: int, max_seconds: float):
    """Sem cérebro nenhum: empuxo fixo em 0,5 (só pairar)."""
    pts, ins, tms, fin = [], [], [], []
    for e in range(courses):
        s = REPORT_SEED + e
        sim = Sim(Course(s, diff), 1, seed=s, max_steps=int(max_seconds / DT))
        while not sim.done:
            sim.step(np.array([0.5]))
        pts.append(sim.score()[0]); ins.append(sim.inspections[0]); tms.append(sim.step_i * DT); fin.append(sim.alive[0])
    return float(np.mean(pts)), float(np.mean(ins)), float(np.mean(fin)), float(np.median(tms))


def br(x: float, decimals: int = 0) -> str:
    """Número no formato brasileiro: 2.805 e 18,0."""
    return f"{x:,.{decimals}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def row(label: str, pts: float, ins: float, fin: float, extra: str = "") -> str:
    return f"| {label} | {br(pts)} | {br(ins, 1)} | {br(100 * fin)}%{extra} |"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Tabela de resultados em percursos inéditos.")
    ap.add_argument("--normal", type=Path, default=None, help="execução do modo normal (para a linha da geração 1)")
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--courses", type=int, default=24)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    a = ap.parse_args(argv)
    n, ms = a.courses, a.max_seconds

    print(f"Médias em {n} percursos inéditos, voos de até {ms:.0f} s.\n")
    for diff, name in ((NORMAL, "normal"), (HARD, "hard")):
        print(f"**Modo {diff.label.lower()}**\n")
        print("| Drone | Pontos | Isoladores | Chegam aos 90 s |\n|---|---|---|---|")
        hp, hi, hf, ht = hover_baseline(diff, n, ms)
        print(row("Só pairar (empuxo fixo em 0,5)", hp, hi, hf, f" (cai em {br(ht, 1)} s, mediana)"))
        if diff is NORMAL and a.normal is not None:
            g1, _ = load_generation(a.normal, 1, None)
            p, i, f = evaluate_on(g1, diff, REPORT_SEED, n, ms)
            k = int(np.argmax(p))
            print(row("Melhor dos 200 drones aleatórios da geração 1", p[k], i[k], f[k]))
        for tag in ("normal", "hard"):
            f = a.models / f"best_{tag}.npy"
            if not f.exists() or (diff is NORMAL and tag == "hard"):
                continue
            p, i, fin = evaluate_on(np.load(f)[None, :], diff, REPORT_SEED, n, ms)
            label = {"normal": "Melhor drone treinado no modo normal (250 gerações)",
                     "hard": "Melhor drone treinado no modo difícil (+250 gerações)"}[tag]
            print(row(label, p[0], i[0], fin[0]))
        print()


if __name__ == "__main__":
    main()
