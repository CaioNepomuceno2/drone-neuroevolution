"""Simulação vetorizada: um percurso (Course) e N drones voando nele ao mesmo tempo.

Todos os drones de uma geração enfrentam o MESMO percurso, então os obstáculos
são calculados uma vez por passo e só o que depende do drone (altura, sensores,
colisão) é vetorizado em NumPy. Não há nenhuma dependência de pygame aqui, o
treino roda sem janela.
"""
from __future__ import annotations

import numpy as np

from .config import (
    ACC_EFF, SWITCH_SAFETY, BIRD, BRANCH, DRAG, DT, GRAVITY, GROUND_Y, INSPECT_BONUS, INSPECT_TOL,
    MAX_THRUST_ACC, N_RAYS, POLE_FIRST, POLE_SPACING, RADIUS, RAY_ANGLES_DEG,
    RAY_LEN, SCORE_PER_PX, TREE, VY_MAX, X0, Difficulty,
)


def scroll_at(diff: Difficulty, t):
    """Distância rolada (px) no instante t: a velocidade cresce linearmente com a distância e depois fica constante."""
    k = (diff.speed_max - diff.speed0) / diff.ramp_px
    t_r = np.log(diff.speed_max / diff.speed0) / k
    t = np.asarray(t, dtype=float)
    ramp = diff.speed0 / k * (np.exp(k * np.minimum(t, t_r)) - 1.0)
    return np.where(t <= t_r, ramp, diff.ramp_px + diff.speed_max * (t - t_r))


def time_at(diff: Difficulty, s):
    """Instante (s) em que a distância rolada chega a s (inversa de scroll_at)."""
    k = (diff.speed_max - diff.speed0) / diff.ramp_px
    t_r = np.log(diff.speed_max / diff.speed0) / k
    s = np.asarray(s, dtype=float)
    ramp = np.log1p(k * np.minimum(s, diff.ramp_px) / diff.speed0) / k
    return np.where(s <= diff.ramp_px, ramp, t_r + (s - diff.ramp_px) / diff.speed_max)


class Course:
    """Percurso gerado a partir de uma semente: postes, árvores, galhos e pássaros."""

    def __init__(self, seed: int, diff: Difficulty, length_px: float = 45000.0):
        self.seed = int(seed)
        self.diff = diff
        rng = np.random.default_rng(self.seed)

        # postes com o isolador a alturas diferentes
        n_poles = int(length_px // POLE_SPACING) + 2
        self.pole_x = POLE_FIRST + POLE_SPACING * np.arange(n_poles)
        self.pole_y = rng.uniform(150.0, 330.0, n_poles)

        # fases do vento (duas senoides somadas parecem rajadas)
        self.wind_phase = rng.uniform(0.0, 2.0 * np.pi, 2)

        # 1) obstáculos fixos (árvores e galhos), com espaço para trocar de faixa entre um e outro
        kind, xw, w, yt, h, vb, amp, om, ph = [], [], [], [], [], [], [], [], []
        lo_s, hi_s = [], []                  # faixa livre de cada obstáculo fixo
        x = 650.0
        prev = None   # último obstáculo fixo: (é_árvore, y_seguro_min, y_seguro_max, x_direita)
        stretch = 1.0 / max(1e-6, 1.0 - diff.bird_prob)
        while x < length_px:
            k = TREE if rng.random() < 0.5 else BRANCH
            if k == TREE:
                wk, hh = rng.uniform(40.0, 60.0), rng.uniform(170.0, 320.0)
                top = GROUND_Y - hh
                lo, hi = RADIUS + 8.0, top - RADIUS - 8.0            # faixa livre: acima da árvore
            else:
                wk, hh = rng.uniform(40.0, 60.0), rng.uniform(110.0, 300.0)
                top = 0.0
                lo, hi = hh + RADIUS + 8.0, GROUND_Y - RADIUS - 8.0  # faixa livre: abaixo do galho
            if prev is not None and prev[0] != (k == TREE):
                # árvore seguida de galho (ou o contrário): deixa tempo de trocar de faixa
                delta = max(0.0, lo - prev[2], prev[1] - hi)
                need = 2.0 * RADIUS + SWITCH_SAFETY * diff.speed_max * 2.0 * np.sqrt(delta / ACC_EFF)
                x = max(x, prev[3] + need)
            near = np.abs(self.pole_x - x)
            if near.min() < 110.0:                                   # deixa o poste livre
                x = float(self.pole_x[near.argmin()]) + 120.0
            if x >= length_px:
                break
            kind.append(k); xw.append(x); w.append(wk); yt.append(top); h.append(hh)
            vb.append(0.0); amp.append(0.0); om.append(0.0); ph.append(0.0)
            lo_s.append(lo); hi_s.append(hi)
            prev = (k == TREE, lo, hi, x + wk)
            x += rng.uniform(*diff.gap) * stretch

        # 2) pássaros, posicionados pelo INSTANTE em que chegam ao drone. Assim dá para garantir que
        #    nenhum pássaro feche o único corredor livre de uma árvore/galho alto (percurso impossível).
        t_static = time_at(diff, np.array(xw)) if xw else np.empty(0)
        t_pole = time_at(diff, self.pole_x)
        lo_a, hi_a = np.array(lo_s), np.array(hi_s)
        placed: list[tuple[float, float]] = []                       # (instante, altura média)
        t_end = float(time_at(diff, length_px - 3000.0))
        v_mid = 0.5 * (diff.speed0 + diff.speed_max)
        t = rng.uniform(2.0, 3.2)
        while t < t_end:
            bvx = rng.uniform(*diff.bird_vx)
            bamp = rng.uniform(*diff.bird_amp)
            bom = rng.uniform(1.0, 2.6)
            bph = rng.uniform(0.0, 2.0 * np.pi)
            y0 = rng.uniform(80.0, 440.0)
            b_lo, b_hi = y0 - bamp - 10.0 - RADIUS, y0 + bamp + 10.0 + RADIUS   # faixa que o pássaro ocupa
            tb = t
            for _ in range(8):
                ok = not (np.abs(t_pole - tb) < 0.5).any()
                if ok and len(t_static):
                    near_s = np.abs(t_static - tb) < 1.4
                    if near_s.any():
                        leftover = np.maximum(np.maximum(0.0, b_lo - lo_a[near_s]), np.maximum(0.0, hi_a[near_s] - b_hi))
                        blocks = (b_hi > lo_a[near_s]) & (b_lo < hi_a[near_s]) & (leftover < 2.0 * RADIUS + 40.0)
                        ok = not blocks.any()
                if ok:
                    ok = not any(abs(tb - pt) < 0.6 and abs(y0 - py) < 90.0 for pt, py in placed)
                if ok:
                    break
                tb += 0.7
            else:
                ok = False
            if ok:
                placed.append((tb, y0))
                kind.append(BIRD); xw.append(float(scroll_at(diff, tb)) + bvx * tb); w.append(30.0)
                yt.append(y0 - 10.0); h.append(20.0)
                vb.append(bvx); amp.append(bamp); om.append(bom); ph.append(bph)
            t += rng.uniform(*diff.gap) / v_mid / max(diff.bird_prob, 1e-6)

        self.set_obstacles(kind, xw, w, yt, h, vb, amp, om, ph)

    # -- construção manual (usada nos testes) --------------------------------
    def set_obstacles(self, kind, xw, w, yt, h, vb, amp, om, ph) -> None:
        self.kind = np.asarray(kind, dtype=np.int8)
        self.xw = np.asarray(xw, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.yt = np.asarray(yt, dtype=float)    # topo (pássaro: topo na posição média)
        self.h = np.asarray(h, dtype=float)
        self.vb = np.asarray(vb, dtype=float)    # velocidade própria (só pássaros)
        self.amp = np.asarray(amp, dtype=float)
        self.om = np.asarray(om, dtype=float)
        self.ph = np.asarray(ph, dtype=float)

    # -- consultas -----------------------------------------------------------
    def speed(self, scroll: float) -> float:
        d = self.diff
        return min(d.speed_max, d.speed0 + (d.speed_max - d.speed0) * scroll / d.ramp_px)

    def wind(self, t: float) -> float:
        if self.diff.wind == 0.0:
            return 0.0
        a, b = self.wind_phase
        return self.diff.wind * (0.6 * np.sin(1.3 * t + a) + 0.4 * np.sin(2.9 * t + b))

    def rects(self, scroll: float, t: float, left: float = 60.0, right: float = RAY_LEN + 40.0):
        """Retângulos dos obstáculos na tela: (idx, xl, xr, yt, yb), só os visíveis."""
        xl = self.xw - scroll + X0 - self.vb * t
        xr = xl + self.w
        idx = np.nonzero((xr > X0 - left) & (xl < X0 + right))[0]
        xl, xr = xl[idx], xr[idx]
        yt = self.yt[idx] + self.amp[idx] * np.sin(self.om[idx] * t + self.ph[idx])
        return idx, xl, xr, yt, yt + self.h[idx]


class Sim:
    """N drones, um percurso. Estado por drone em arrays NumPy."""

    def __init__(self, course: Course, n: int, seed: int = 0, max_steps: int = 2700,
                 y_jitter: float = 30.0):
        self.course = course
        self.n = int(n)
        self.max_steps = int(max_steps)
        rng = np.random.default_rng(int(seed) + 12345)
        self.y = 250.0 + rng.uniform(-y_jitter, y_jitter, self.n)
        self.vy = np.zeros(self.n)
        self.alive = np.ones(self.n, dtype=bool)
        self.inspections = np.zeros(self.n, dtype=np.int64)
        self.final_scroll = np.zeros(self.n)
        self.death_step = np.full(self.n, -1, dtype=np.int64)
        self.cause = np.zeros(self.n, dtype=np.int8)   # 0 vivo, 1 obstáculo, 2 chão/teto
        self.scroll = 0.0
        self.t = 0.0
        self.step_i = 0
        self.next_pole = 0
        self.crashed_now = np.empty(0, dtype=np.int64)
        self.inspected_now = np.empty(0, dtype=np.int64)
        self.last_thrust = np.full(self.n, 0.5)

        ang = np.deg2rad(RAY_ANGLES_DEG)
        self._dx, self._dy = np.cos(ang), np.sin(ang)
        self._inv_dx = 1.0 / np.where(np.abs(self._dx) < 1e-9, 1e-9, self._dx)
        self._inv_dy = 1.0 / np.where(np.abs(self._dy) < 1e-9, 1e-9, self._dy)
        self._rects = course.rects(self.scroll, self.t)

    # -- estado --------------------------------------------------------------
    @property
    def done(self) -> bool:
        return (not self.alive.any()) or self.step_i >= self.max_steps

    @property
    def distance_px(self) -> np.ndarray:
        return np.where(self.alive, self.scroll, self.final_scroll)

    def score(self) -> np.ndarray:
        return self.distance_px * SCORE_PER_PX + INSPECT_BONUS * self.inspections

    def leader(self) -> int:
        """Índice do melhor drone vivo (ou do melhor de todos se ninguém sobrou)."""
        sc = self.score()
        if self.alive.any():
            return int(np.argmax(np.where(self.alive, sc, -np.inf)))
        return int(np.argmax(sc))

    # -- sensores ------------------------------------------------------------
    def rays(self, y: np.ndarray | None = None) -> np.ndarray:
        """Distância até a colisão, normalizada: 0 = livre, 1 = encostando. (N, raios)

        Os obstáculos (e o chão/teto) são "engordados" pelo raio do drone: o raio mede quanto falta
        para o CORPO bater, não para o ponto central. Assim um pássaro pequeno não escapa entre dois raios.
        """
        y = self.y if y is None else y
        _, xl, xr, yt, yb = self._rects
        d = np.full((y.shape[0], N_RAYS), np.inf)
        if len(xl):
            oy = y[:, None, None]
            t1x = (xl - RADIUS - X0)[None, None, :] * self._inv_dx[None, :, None]
            t2x = (xr + RADIUS - X0)[None, None, :] * self._inv_dx[None, :, None]
            t1y = (yt - RADIUS)[None, None, :] - oy
            t2y = (yb + RADIUS)[None, None, :] - oy
            t1y = t1y * self._inv_dy[None, :, None]
            t2y = t2y * self._inv_dy[None, :, None]
            tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
            tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
            hit = (tmax >= 0.0) & (tmin <= tmax)
            d = np.where(hit, np.maximum(tmin, 0.0), np.inf).min(axis=2)
        down = self._dy > 1e-9
        up = self._dy < -1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = np.where(down[None, :], (GROUND_Y - RADIUS - y[:, None]) / np.where(down, self._dy, 1.0)[None, :], np.inf)
            tc = np.where(up[None, :], (RADIUS - y[:, None]) / np.where(up, self._dy, 1.0)[None, :], np.inf)
        d = np.minimum(d, np.minimum(tg, tc))
        return np.clip(1.0 - np.minimum(d, RAY_LEN) / RAY_LEN, 0.0, 1.0)

    def observe(self) -> np.ndarray:
        """Entradas da rede: 5 raios, altura, velocidade vertical, dy e dx até o próximo isolador."""
        c = self.course
        k = min(self.next_pole, len(c.pole_x) - 1)
        dy = np.clip((c.pole_y[k] - self.y) / 250.0, -1.0, 1.0)
        dx = float(np.clip((c.pole_x[k] - self.scroll) / 600.0, 0.0, 1.0))
        return np.column_stack([
            self.rays(),
            (self.y - 250.0) / 250.0,
            self.vy / VY_MAX,
            dy,
            np.full(self.n, dx),
        ])

    # -- um passo de simulação -----------------------------------------------
    def step(self, thrust: np.ndarray) -> None:
        c = self.course
        thrust = np.clip(np.asarray(thrust, dtype=float), 0.0, 1.0)
        self.last_thrust = thrust
        ay = GRAVITY - MAX_THRUST_ACC * thrust + c.wind(self.t)
        vy = np.clip((self.vy + ay * DT) * (1.0 - DRAG * DT), -VY_MAX, VY_MAX)
        alive = self.alive
        self.vy = np.where(alive, vy, 0.0)
        self.y = np.where(alive, self.y + vy * DT, self.y)

        self.scroll += c.speed(self.scroll) * DT
        self.t += DT
        self.step_i += 1
        self._rects = c.rects(self.scroll, self.t)

        # colisões: teto/chão e círculo contra retângulos
        _, xl, xr, yt, yb = self._rects
        bound = (self.y - RADIUS < 0.0) | (self.y + RADIUS > GROUND_Y)
        obst = np.zeros(self.n, dtype=bool)
        if len(xl):
            cx = np.clip(X0, xl, xr)[None, :]
            cy = np.clip(self.y[:, None], yt[None, :], yb[None, :])
            obst = (((cx - X0) ** 2 + (cy - self.y[:, None]) ** 2) <= RADIUS * RADIUS).any(axis=1)
        dead = alive & (bound | obst)
        self.crashed_now = np.nonzero(dead)[0]
        if dead.any():
            self.cause[dead] = np.where(obst[dead], 1, 2)
            self.final_scroll[dead] = self.scroll
            self.death_step[dead] = self.step_i
            self.alive = alive & ~dead

        # inspeção: o drone cruza o poste com o isolador na altura certa
        self.inspected_now = np.empty(0, dtype=np.int64)
        while self.next_pole < len(c.pole_x) and c.pole_x[self.next_pole] <= self.scroll:
            hit = self.alive & (np.abs(self.y - c.pole_y[self.next_pole]) <= INSPECT_TOL)
            self.inspections += hit
            self.inspected_now = np.concatenate([self.inspected_now, np.nonzero(hit)[0]])
            self.next_pole += 1

    # -- para o renderizador ---------------------------------------------------
    def view_rects(self):
        """Obstáculos de toda a tela (não só da janela dos sensores)."""
        return self.course.rects(self.scroll, self.t, left=140.0, right=900.0)

    def view_poles(self):
        """Postes visíveis: (x na tela, y do isolador, índice)."""
        c = self.course
        sx = c.pole_x - self.scroll + X0
        idx = np.nonzero((sx > -200.0) & (sx < 1200.0))[0]
        return sx[idx], c.pole_y[idx], idx
