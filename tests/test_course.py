"""Garantias do gerador de percursos: nada de situação impossível."""
from __future__ import annotations

import numpy as np
import pytest

from drone_evo.config import BIRD, GROUND_Y, HARD, NORMAL, RADIUS, TREE, X0
from drone_evo.world import Course, scroll_at, time_at


def arrival_time(c: Course, i: int) -> float:
    """Instante em que o obstáculo i chega ao x do drone (busca binária; vale para fixos e pássaros)."""
    lo, hi = 0.0, 300.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        xl = c.xw[i] - float(scroll_at(c.diff, mid)) - c.vb[i] * mid
        lo, hi = (mid, hi) if xl > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_scroll_and_time_are_inverse():
    for d in (NORMAL, HARD):
        t = np.linspace(0.0, 120.0, 25)
        assert np.allclose(time_at(d, scroll_at(d, t)), t, atol=1e-6)


@pytest.mark.parametrize("diff", [NORMAL, HARD])
@pytest.mark.parametrize("seed", range(8))
def test_no_bird_closes_the_only_free_corridor(diff, seed):
    c = Course(seed, diff)
    birds = np.nonzero(c.kind == BIRD)[0]
    statics = np.nonzero(c.kind != BIRD)[0]
    assert len(birds) > 20 and len(statics) > 20
    t_stat = np.array([arrival_time(c, i) for i in statics])
    for b in birds:
        tb = arrival_time(c, b)
        y0 = c.yt[b] + 10.0
        b_lo, b_hi = y0 - c.amp[b] - 10.0 - RADIUS, y0 + c.amp[b] + 10.0 + RADIUS
        for i, ts in zip(statics, t_stat):
            if abs(tb - ts) >= 1.4:
                continue
            if c.kind[i] == TREE:
                lo, hi = RADIUS + 8.0, c.yt[i] - RADIUS - 8.0
            else:
                lo, hi = c.h[i] + RADIUS + 8.0, GROUND_Y - RADIUS - 8.0
            free = max(max(0.0, b_lo - lo), max(0.0, hi - b_hi)) if (b_hi > lo and b_lo < hi) else hi - lo
            assert free >= 2.0 * RADIUS + 40.0, (seed, b, i)


@pytest.mark.parametrize("diff", [NORMAL, HARD])
def test_birds_keep_clear_of_the_poles(diff):
    c = Course(11, diff)
    t_pole = time_at(diff, c.pole_x)
    for b in np.nonzero(c.kind == BIRD)[0]:
        assert np.abs(t_pole - arrival_time(c, b)).min() >= 0.5


def test_bird_reaches_the_drone_column_at_its_arrival_time():
    c = Course(5, NORMAL)
    b = int(np.nonzero(c.kind == BIRD)[0][3])
    tb = arrival_time(c, b)
    xl = c.xw[b] - float(scroll_at(c.diff, tb)) - c.vb[b] * tb + X0
    assert abs(xl - X0) < 1e-6
