"""Renderizador pygame: cenário, drones, LiDAR, HUD, gráfico de evolução e a rede neural.

Funciona em janela (modo ao vivo) ou fora da tela (`headless=True`) para gerar vídeo.
Tudo é desenhado em coordenadas lógicas do mundo (960x540) e escalado por `k`;
o HUD usa uma unidade `u` proporcional à largura (1 u = 1 px em 1280 de largura).
"""
from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

from .brain import forward_detail, split
from .config import (
    BRANCH, DT, GROUND_Y, INSPECT_TOL, RAY_ANGLES_DEG, RAY_LEN, TREE,
    VY_MAX, W, X0,
)

SKY_TOP, SKY_MID, SKY_BOT = (20, 30, 66), (92, 80, 138), (247, 168, 112)
CYAN, PINK, GOLD = (70, 214, 255), (255, 96, 170), (255, 205, 80)
INK = (236, 242, 255)
MUTED = (150, 164, 196)

LABELS_IN = [(f"LiDAR {int(a):+d}°" if a else "LiDAR 0°") for a in RAY_ANGLES_DEG] + \
            ["altura", "veloc. vertical", "alvo: Δ altura", "alvo: distância"]


def lerp(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


class Renderer:
    def __init__(self, width: int = 1280, height: int = 720, headless: bool = False, title: str = "Drone evolutivo"):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        pygame.init()
        self.w, self.h = int(width), int(height)
        self.k = self.w / W                       # mundo -> pixels
        self.u = self.w / 1280.0                  # unidade do HUD
        self.headless = headless
        self.screen = None
        if not headless:
            self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
            pygame.display.set_caption(title)
        self.surf = pygame.Surface((self.w, self.h))
        self._fonts: dict = {}
        self._build_background()
        self._build_sprites()
        self.reset()

    # ------------------------------------------------------------------ util
    def font(self, size: float, bold: bool = False):
        key = (round(size * self.u), bold)
        if key not in self._fonts:
            self._fonts[key] = self.pg.font.SysFont("segoeui,arial,helvetica,dejavusans", key[0], bold=bold)
        return self._fonts[key]

    def text(self, s: str, size: float, color=INK, pos=(0, 0), bold=False, anchor="topleft", alpha=255):
        img = self.font(size, bold).render(s, True, color)
        if alpha < 255:
            img.set_alpha(alpha)
        r = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        self.surf.blit(img, r)
        return r

    def reset(self) -> None:
        self.fx: list = []
        self.popups: list = []
        self._pending_crash: list = []
        self._pending_popup: list = []
        self.cur_leader = 0
        self.trail: deque = deque(maxlen=50)
        self._streaks = np.random.default_rng(3).uniform(0, 1, (18, 3))
        self._tick = 0

    # ------------------------------------------------------------ background
    def _build_background(self) -> None:
        pg, k, w, h = self.pg, self.k, self.w, self.h
        horizon = int(GROUND_Y * k)
        sky = pg.Surface((w, h))
        for y in range(h):
            t = min(1.0, y / horizon)
            col = lerp(SKY_TOP, SKY_MID, t / 0.55) if t < 0.55 else lerp(SKY_MID, SKY_BOT, (t - 0.55) / 0.45)
            pg.draw.line(sky, col, (0, y), (w, y))
        glow = pg.Surface((w, h), pg.SRCALPHA)
        cx, cy = int(520 * k), int(384 * k)
        for r in range(int(150 * k), 0, -3):
            a = int(70 * (1 - r / (150 * k)) ** 2)
            pg.draw.circle(glow, (255, 214, 150, a), (cx, cy), r)
        pg.draw.circle(glow, (255, 232, 190, 235), (cx, cy), int(34 * k))
        sky.blit(glow, (0, 0))
        self.sky = sky

        self.layers = []   # (superfície, período em px, fator de paralaxe)
        for base, col, amps, factor in ((372, (66, 58, 118), (34, 18, 8), 0.07),
                                        (428, (44, 44, 96), (26, 14, 6), 0.16)):
            period = 1600.0
            pw = int(period * k)
            layer = pg.Surface((pw, h), pg.SRCALPHA)
            pts = [(0, h)]
            for xi in range(0, pw + 1, 6):
                x = xi / k
                yy = base - (amps[0] * math.sin(2 * math.pi * x / period + 1.0)
                             + amps[1] * math.sin(2 * math.pi * 3 * x / period + 2.0)
                             + amps[2] * math.sin(2 * math.pi * 7 * x / period))
                pts.append((xi, yy * k))
            pts.append((pw, h))
            pg.draw.polygon(layer, col, pts)
            self.layers.append((layer, pw, factor))

        rng = np.random.default_rng(11)
        cl = pg.Surface((int(1400 * k), h), pg.SRCALPHA)
        for _ in range(7):
            x, y = rng.uniform(0, 1400), rng.uniform(40, 250)
            for _ in range(4):
                ex, ey = x + rng.uniform(-40, 40), y + rng.uniform(-8, 8)
                pg.draw.ellipse(cl, (255, 255, 255, 34),
                                (ex * k, ey * k, rng.uniform(70, 130) * k, rng.uniform(16, 28) * k))
        self.clouds = cl

    def _draw_background(self, scroll: float) -> None:
        s, k = self.surf, self.k
        s.blit(self.sky, (0, 0))
        off = (scroll * 0.03) % 1400.0
        for dx in (-off * k, (1400.0 - off) * k):
            s.blit(self.clouds, (dx, 0))
        for layer, pw, factor in self.layers:
            off = (scroll * factor * k) % pw
            for dx in (-off, pw - off):
                if dx < self.w:
                    s.blit(layer, (dx, 0))
        gy = int(GROUND_Y * k)
        self.pg.draw.rect(s, (24, 44, 46), (0, gy, self.w, self.h - gy))
        self.pg.draw.rect(s, (66, 118, 74), (0, gy, self.w, max(3, int(7 * k))))
        step = 38.0
        o = scroll % step
        for i in range(int(W / step) + 3):
            x = (i * step - o) * k
            self.pg.draw.line(s, (40, 80, 54), (x, gy + 7 * k), (x - 4 * k, gy + 15 * k), max(1, int(k * 1.4)))

    # --------------------------------------------------------------- sprites
    def _drone_sprite(self, body, accent, alpha: int = 255, rotor_alpha: int = 150):
        pg, S = self.pg, 4
        u = self.k * S
        wd, hd = 52, 30
        surf = pg.Surface((int(wd * u), int(hd * u)), pg.SRCALPHA)
        cx, cy = wd / 2 * u, hd / 2 * u
        dark = lerp(body, (10, 12, 24), 0.7)
        pg.draw.line(surf, dark, (cx - 17 * u, cy - 3 * u), (cx + 17 * u, cy - 3 * u), int(2.2 * u))
        for sx in (-17, 17):
            pg.draw.line(surf, dark, (cx + sx * u, cy - 3 * u), (cx + sx * u, cy - 7 * u), int(1.6 * u))
            pg.draw.ellipse(surf, (225, 240, 255, rotor_alpha),
                            (cx + (sx - 8.5) * u, cy - 8.6 * u, 17 * u, 3 * u))
        pg.draw.rect(surf, body, (cx - 12 * u, cy - 5 * u, 24 * u, 10.5 * u), border_radius=int(4.5 * u))
        pg.draw.rect(surf, lerp(body, (255, 255, 255), 0.35), (cx - 9 * u, cy - 4 * u, 14 * u, 2.4 * u),
                     border_radius=int(1.2 * u))
        pg.draw.circle(surf, dark, (cx + 1 * u, cy + 6.5 * u), 3.6 * u)
        pg.draw.circle(surf, accent, (cx + 2 * u, cy + 6.8 * u), 1.6 * u)
        pg.draw.circle(surf, accent, (cx + 10 * u, cy - 0.5 * u), 1.7 * u)
        out = pg.transform.smoothscale(surf, (int(wd * self.k), int(hd * self.k)))
        if alpha < 255:
            out.set_alpha(alpha)
        return out

    def _build_sprites(self) -> None:
        ramp = [(70, 110, 255), (60, 170, 255), (60, 220, 230), (110, 235, 170), (200, 235, 110), (255, 190, 80)]
        self.followers = [self._drone_sprite(c, lerp(c, (255, 255, 255), 0.5), alpha=120, rotor_alpha=90) for c in ramp]
        self.leader_sprite = self._drone_sprite((246, 248, 255), CYAN)

    # --------------------------------------------------------------- cenário
    def _draw_poles(self, sim) -> None:
        pg, s, k = self.pg, self.surf, self.k
        c = sim.course
        sx, sy, idx = sim.view_poles()
        nxt = sim.next_pole
        if len(idx) == 0:
            return
        # cabos entre postes consecutivos (curva de catenária), inclusive os vãos que saem da tela
        lo, hi = max(0, int(idx[0]) - 1), min(len(c.pole_x) - 1, int(idx[-1]) + 1)
        for j in range(lo, hi):
            x0, y0 = c.pole_x[j] - sim.scroll + X0, c.pole_y[j]
            x1, y1 = c.pole_x[j + 1] - sim.scroll + X0, c.pole_y[j + 1]
            for dy, wdt, col in ((0.0, 3.2, (18, 18, 30)), (-24.0, 1.8, (34, 34, 52))):
                pts = []
                for i in range(0, 26):
                    u = i / 25.0
                    pts.append(((x0 + (x1 - x0) * u) * k, (y0 + (y1 - y0) * u + 30 * 4 * u * (1 - u) + dy) * k))
                pg.draw.lines(s, col, False, pts, max(1, int(wdt * k)))
        pulse = 0.5 + 0.5 * math.sin(sim.t * 6.0)
        for j, i in enumerate(idx):
            x, y = sx[j], sy[j]
            col = (22, 20, 38)
            pg.draw.rect(s, col, ((x - 4) * k, (y - 18) * k, 8 * k, (GROUND_Y - y + 18) * k))
            pg.draw.rect(s, col, ((x - 44) * k, (y - 24) * k, 88 * k, 6 * k), border_radius=int(2 * k))
            for dx in (-34, 34):
                pg.draw.rect(s, (150, 170, 190), ((x + dx - 3) * k, (y - 24) * k, 6 * k, 10 * k), border_radius=int(2 * k))
            target = i == nxt
            r = (9 + 2.5 * pulse) * k if target else 6 * k
            if target:
                halo = pg.Surface((int(60 * k), int(60 * k)), pg.SRCALPHA)
                pg.draw.circle(halo, (70, 214, 255, 60), (int(30 * k), int(30 * k)), int(INSPECT_TOL * 0.6 * k))
                pg.draw.circle(halo, (70, 214, 255, 150), (int(30 * k), int(30 * k)), int(INSPECT_TOL * 0.6 * k), max(1, int(1.6 * k)))
                s.blit(halo, ((x - 30) * k, (y - 30) * k))
            pg.draw.circle(s, CYAN if target else (130, 150, 175), (x * k, y * k), r)
            pg.draw.circle(s, (230, 250, 255), (x * k, y * k), r * 0.4)

    def _draw_obstacles(self, sim) -> None:
        pg, s, k = self.pg, self.surf, self.k
        c = sim.course
        idx, xl, xr, yt, yb = sim.view_rects()
        for j, i in enumerate(idx):
            kind = c.kind[i]
            x, y, w, h = xl[j], yt[j], xr[j] - xl[j], yb[j] - yt[j]
            if kind == TREE:
                pg.draw.rect(s, (62, 42, 34), ((x + w * 0.32) * k, (yb[j] - 30) * k, w * 0.36 * k, 30 * k))
                pg.draw.rect(s, (24, 82, 62), (x * k, y * k, w * k, (h - 26) * k), border_radius=int(w * 0.5 * k))
                pg.draw.rect(s, (38, 116, 84), ((x + 3) * k, (y + 4) * k, w * 0.46 * k, (h - 40) * k), border_radius=int(w * 0.25 * k))
                pg.draw.rect(s, (60, 150, 104), ((x + 7) * k, (y + 10) * k, w * 0.16 * k, (h - 56) * k), border_radius=int(w * 0.08 * k))
            elif kind == BRANCH:
                pg.draw.rect(s, (88, 62, 44), (x * k, (y - 4) * k, w * k, (h + 4) * k), border_bottom_left_radius=int(w * 0.5 * k),
                             border_bottom_right_radius=int(w * 0.5 * k))
                pg.draw.rect(s, (118, 86, 58), ((x + 5) * k, (y - 4) * k, w * 0.22 * k, (h - 6) * k), border_radius=int(2 * k))
                for q in range(int(h // 46) + 1):
                    ly = y + 14 + q * 46
                    if ly > yb[j] - 6:
                        break
                    pg.draw.circle(s, (54, 128, 78), ((x + w * (0.28 if q % 2 else 0.72)) * k, ly * k), w * 0.3 * k)
                pg.draw.circle(s, (70, 150, 92), ((x + w * 0.5) * k, (yb[j] - w * 0.3) * k), w * 0.34 * k)
            else:
                flap = math.sin(sim.t * 17.0 + i * 1.7)
                cx, cy = (x + w * 0.5) * k, (y + h * 0.5) * k
                body = (24, 26, 46)
                pg.draw.polygon(s, body, [(cx + 8 * k, cy), (cx + 19 * k, cy - 3 * k), (cx + 19 * k, cy + 4 * k)])   # cauda
                pg.draw.ellipse(s, body, (cx - 14 * k, cy - 6.5 * k, 28 * k, 13 * k))
                pg.draw.circle(s, body, (cx - 12 * k, cy - 2 * k), 5.2 * k)
                pg.draw.polygon(s, (255, 168, 60), [(cx - 16 * k, cy - 3 * k), (cx - 23 * k, cy - 1 * k), (cx - 16 * k, cy)])
                pg.draw.circle(s, (255, 255, 255), (cx - 13 * k, cy - 3 * k), 1.3 * k)
                wing = [(cx - 5 * k, cy - 3 * k), (cx + 8 * k, cy - 3 * k), (cx + 4 * k, cy - 3 * k - 11 * k * flap)]
                pg.draw.polygon(s, (52, 56, 96), wing)

    def _draw_wind(self, sim, dt: float) -> None:
        wind = sim.course.wind(sim.t)
        if abs(wind) < 25.0:
            return
        s, k, pg = self.surf, self.k, self.pg
        mag = min(1.0, abs(wind) / sim.course.diff.wind) if sim.course.diff.wind else 0.0
        layer = pg.Surface((self.w, self.h), pg.SRCALPHA)
        for sx, sy, ln in self._streaks:
            x = ((sx * W - (sim.scroll * 1.3 + sim.t * 200.0)) % (W + 160.0)) - 80.0
            y = 40.0 + sy * (GROUND_Y - 80.0)
            length = (40.0 + 80.0 * ln) * (0.4 + mag)
            dy = math.copysign(0.22, wind) * length
            pg.draw.line(layer, (255, 255, 255, int(40 + 90 * mag)), (x * k, y * k), ((x + length) * k, (y - dy) * k), max(1, int(1.6 * k)))
        s.blit(layer, (0, 0))

    # ----------------------------------------------------------------- drones
    def _draw_drones(self, sim, leader: int, obs=None) -> None:
        pg, s, k = self.pg, self.surf, self.k
        score = sim.score()
        order = np.argsort(score)
        alive = sim.alive
        ranks = np.empty(sim.n, dtype=int)
        ranks[order] = np.arange(sim.n)
        nb = len(self.followers)
        bins = np.minimum((ranks * nb) // max(sim.n, 1), nb - 1)
        for i in order:
            if not alive[i] or i == leader:
                continue
            ang = -float(sim.vy[i]) / VY_MAX * 22.0
            img = pg.transform.rotate(self.followers[bins[i]], ang)
            s.blit(img, img.get_rect(center=(X0 * k, sim.y[i] * k)))
        if sim.alive[leader]:
            self._draw_leader(sim, leader, obs)

    def _draw_leader(self, sim, leader: int, obs=None) -> None:
        pg, s, k = self.pg, self.surf, self.k
        y = float(sim.y[leader])
        # rastro
        self.trail.append((sim.scroll, y))
        pts = [((X0 - (sim.scroll - sc)) * k, yy * k) for sc, yy in self.trail]
        if len(pts) > 2:
            trail = pg.Surface((self.w, self.h), pg.SRCALPHA)
            for a in range(1, len(pts)):
                pg.draw.line(trail, (255, 214, 120, int(150 * a / len(pts))), pts[a - 1], pts[a], max(1, int(2.4 * k)))
            s.blit(trail, (0, 0))
        # raios do LiDAR
        prox = sim.rays(np.array([y]))[0]
        rays = pg.Surface((self.w, self.h), pg.SRCALPHA)
        for ang, p in zip(RAY_ANGLES_DEG, prox):
            d = (1.0 - p) * RAY_LEN
            a = math.radians(ang)
            end = ((X0 + d * math.cos(a)) * k, (y + d * math.sin(a)) * k)
            col = lerp((90, 255, 170), (255, 80, 90), p * 1.4)
            pg.draw.line(rays, col + (170,), (X0 * k, y * k), end, max(1, int(1.6 * k)))
            if p > 0.001:
                pg.draw.circle(rays, col + (235,), end, 3.2 * k)
        s.blit(rays, (0, 0))
        ang = -float(sim.vy[leader]) / VY_MAX * 22.0
        img = pg.transform.rotate(self.leader_sprite, ang)
        s.blit(img, img.get_rect(center=(X0 * k, y * k)))
        spin = 0.55 + 0.45 * abs(math.sin(sim.t * 55.0))
        for sx in (-17, 17):
            pg.draw.ellipse(s, (255, 255, 255), ((X0 + sx - 9 * spin) * k, (y - 8.6) * k, 18 * spin * k, 2.6 * k))
        pg.draw.circle(s, GOLD, (X0 * k, y * k), 25 * k, max(1, int(1.6 * k)))

    # ---------------------------------------------------------------- efeitos
    def note_events(self, sim) -> None:
        """Anota batidas e inspeções do último passo. Chamar a cada passo da simulação, antes de `draw`
        (assim nada se perde quando se desenha só 1 passo a cada 2, como no vídeo de 30 fps)."""
        for i in sim.crashed_now[:12]:
            self._pending_crash.append((float(sim.y[i]), int(i) == self.cur_leader))
        for i in sim.inspected_now:
            if int(i) == self.cur_leader:
                self._pending_popup.append((float(sim.y[i]), int(sim.inspections[i])))

    def _update_fx(self, sim, leader: int, dt: float) -> None:
        rng = np.random.default_rng(sim.step_i)
        speed = sim.course.speed(sim.scroll)
        for y, is_leader in self._pending_crash[:24]:
            for _ in range(22 if is_leader else 7):
                a = rng.uniform(0, 2 * math.pi)
                v = rng.uniform(40, 200 if is_leader else 120)
                self.fx.append([X0, y, math.cos(a) * v, math.sin(a) * v, 0.55, 255 if is_leader else 150])
        for y, n_ins in self._pending_popup:
            self.popups.append([X0 + 30, y - 30, 1.2, f"+40  isolador {n_ins}"])
        self._pending_crash, self._pending_popup = [], []
        s, k = self.surf, self.k
        alive_fx = []
        for p in self.fx:
            p[0] += (p[2] - speed) * dt
            p[1] += p[3] * dt
            p[3] += 400.0 * dt
            p[4] -= dt
            if p[4] > 0:
                alive_fx.append(p)
                r = max(1.0, 3.5 * p[4] / 0.55) * k
                self.pg.draw.circle(s, lerp((255, 120, 60), (255, 230, 160), p[4] / 0.55), (p[0] * k, p[1] * k), r)
        self.fx = alive_fx[-400:]
        alive_pop = []
        for p in self.popups:
            p[0] -= speed * dt
            p[1] -= 28.0 * dt
            p[2] -= dt
            if p[2] > 0:
                alive_pop.append(p)
                self.text(p[3], 15, (150, 255, 200), (p[0] * k, p[1] * k), bold=True, anchor="midleft", alpha=int(255 * min(1, p[2] * 2)))
        self.popups = alive_pop

    # -------------------------------------------------------------------- HUD
    def _panel(self, x, y, w, h, alpha=150):
        pg, u = self.pg, self.u
        p = pg.Surface((int(w), int(h)), pg.SRCALPHA)
        pg.draw.rect(p, (8, 10, 24, alpha), (0, 0, int(w), int(h)), border_radius=int(12 * u))
        pg.draw.rect(p, (255, 255, 255, 38), (0, 0, int(w), int(h)), 1, border_radius=int(12 * u))
        self.surf.blit(p, (int(x), int(y)))

    def _hud_stats(self, sim, gen_label, leader: int, score_label: str = "Pontos do líder",
                   rival: tuple | None = None, title: str = "GERAÇÃO") -> None:
        u = self.u
        x, y, w = 372 * u, 24 * u, 310 * u     # à direita da coluna fixa do drone (x=293 px), para não cobri-lo
        rows = [("Vivos", f"{int(sim.alive.sum())} / {sim.n}"),
                (score_label, fmt(float(sim.score()[leader]))),
                ("Isoladores", str(int(sim.inspections[leader])))]
        if rival is not None:
            rows.append(rival)
        h = (146 + 22 * len(rows)) * u
        self._panel(x, y, w, h)
        self.text(title, 13, MUTED, (x + 18 * u, y + 12 * u), bold=True)
        self.text(str(gen_label), 54, GOLD, (x + 18 * u, y + 24 * u), bold=True)
        for n, (a, b) in enumerate(rows):
            yy = y + 92 * u + n * 22 * u
            self.text(a, 14, MUTED, (x + 18 * u, yy))
            self.text(b, 15, INK, (x + w - 18 * u, yy), bold=True, anchor="topright")
        tag = f"{sim.course.diff.label}" + ("  ·  vento" if sim.course.diff.wind else "")
        self.text(tag, 12, CYAN if not sim.course.diff.wind else PINK, (x + w - 18 * u, y + 16 * u), bold=True, anchor="topright")

    def _hud_chart(self, hist, gen_label_idx) -> None:
        pg, u = self.pg, self.u
        w, h = 420 * u, 190 * u
        x, y = self.w - 24 * u - w, 24 * u
        self._panel(x, y, w, h)
        self.text("Evolução da pontuação", 14, INK, (x + 18 * u, y + 12 * u), bold=True)
        gx, gy, gw, gh = x + 52 * u, y + 44 * u, w - 76 * u, h - 74 * u
        if hist is None or len(hist) == 0:
            return
        top = max(float(np.max(hist[:, 0])) * 1.08, 100.0)
        n = max(len(hist), 12)
        for frac in (0.0, 0.5, 1.0):
            yy = gy + gh * (1 - frac)
            pg.draw.line(self.surf, (255, 255, 255, 40) if False else (60, 68, 96), (gx, yy), (gx + gw, yy), 1)
            self.text(fmt(top * frac), 10, MUTED, (gx - 6 * u, yy), anchor="midright")
        self.text("gerações", 10, MUTED, (gx + gw, gy + gh + 6 * u), anchor="topright")
        for col_i, col in ((1, CYAN), (0, GOLD)):
            pts = [(gx + gw * i / max(n - 1, 1), gy + gh * (1 - float(v) / top)) for i, v in enumerate(hist[:, col_i])]
            if len(pts) > 1:
                pg.draw.lines(self.surf, col, False, pts, max(2, int(2.4 * u)))
            elif pts:
                pg.draw.circle(self.surf, col, pts[0], 3 * u)
        self.text("melhor", 11, GOLD, (gx + 6 * u, gy + 2 * u), bold=True)
        self.text("média", 11, CYAN, (gx + 62 * u, gy + 2 * u), bold=True)

    def _hud_nn(self, genome, obs) -> None:
        pg, u = self.pg, self.u
        w, h = 400 * u, 262 * u
        x, y = self.w - 24 * u - w, self.h - 24 * u - h
        self._panel(x, y, w, h)
        w1, b1, w2, b2 = split(genome)
        hid, out = forward_detail(genome, obs)
        n_in, n_h = len(obs), len(hid)
        self.text(f"Cérebro do líder ({n_in} → {n_h} → 1)", 14, INK, (x + 18 * u, y + 12 * u), bold=True)
        xi, xh, xo = x + 128 * u, x + w * 0.56, x + w - 52 * u
        top, bot = y + 46 * u, y + h - 22 * u
        yi = [top + (bot - top) * (i + 0.5) / n_in for i in range(n_in)]
        yh = [top + (bot - top) * (i + 0.5) / n_h for i in range(n_h)]
        yo = [(top + bot) / 2]
        layer = pg.Surface((int(w), int(h)), pg.SRCALPHA)
        mx = max(float(np.abs(w1).max()), float(np.abs(w2).max()), 1e-6)

        def link(a, b, wt):
            m = abs(wt) / mx
            if m < 0.18:
                return
            col = (CYAN if wt > 0 else PINK) + (int(40 + 200 * m),)
            pg.draw.line(layer, col, (a[0] - x, a[1] - y), (b[0] - x, b[1] - y), max(1, int((0.6 + 2.2 * m) * u)))

        for i in range(n_in):
            for j in range(n_h):
                link((xi, yi[i]), (xh, yh[j]), w1[i, j])
        for j in range(n_h):
            link((xh, yh[j]), (xo, yo[0]), w2[j, 0])
        self.surf.blit(layer, (int(x), int(y)))

        def node(cx, cy, v, r):
            v = float(np.clip(v, -1, 1))
            col = lerp((30, 34, 60), CYAN, v) if v >= 0 else lerp((30, 34, 60), PINK, -v)
            pg.draw.circle(self.surf, col, (cx, cy), r)
            pg.draw.circle(self.surf, (200, 210, 235), (cx, cy), r, max(1, int(u)))

        in_vals = [float(v) for v in obs]
        for i in range(n_in):
            node(xi, yi[i], in_vals[i], 5.0 * u)
            self.text(LABELS_IN[i], 9, MUTED, (xi - 11 * u, yi[i]), anchor="midright")
        for j in range(n_h):
            node(xh, yh[j], hid[j], 4.8 * u)
        thrust = 0.5 * (float(out[0]) + 1.0)
        node(xo, yo[0], float(out[0]), 9 * u)
        self.text("empuxo", 11, INK, (xo, yo[0] + 18 * u), anchor="midtop", bold=True)
        self.text("sobe" if thrust > 0.55 else "desce" if thrust < 0.45 else "paira", 11, GOLD, (xo, yo[0] - 20 * u), anchor="midbottom", bold=True)

    def _caption(self, text: str, alpha: int = 255) -> None:
        u = self.u
        font = self.font(30, True)
        lines = text.split("\n")
        imgs = [font.render(t, True, (255, 255, 255)) for t in lines]
        wmax = max(i.get_width() for i in imgs)
        hh = sum(i.get_height() for i in imgs)
        pad = 18 * u
        x_right = self.w - 24 * u - 400 * u - 16 * u
        cx = (24 * u + x_right) / 2
        bar = self.pg.Surface((int(wmax + 2 * pad), int(hh + 2 * pad)), self.pg.SRCALPHA)
        self.pg.draw.rect(bar, (6, 8, 20, int(190 * alpha / 255)), (0, 0, bar.get_width(), bar.get_height()), border_radius=int(14 * u))
        y = self.h - 24 * u - bar.get_height()
        self.surf.blit(bar, (int(cx - bar.get_width() / 2), int(y)))
        yy = y + pad
        for im in imgs:
            im.set_alpha(alpha)
            self.surf.blit(im, (int(cx - im.get_width() / 2), int(yy)))
            yy += im.get_height()

    # ---------------------------------------------------------------- interface
    def draw(self, sim, genomes=None, obs=None, hist=None, gen_label="1", caption=None,
             caption_alpha: int = 255, dt: float = DT, hud: bool = True, follow: int | None = None,
             nn_index: int | None = None, score_label: str = "Pontos do líder",
             rival: tuple | None = None, marker: str | None = None, title: str = "GERAÇÃO") -> int:
        """Desenha um quadro completo. Devolve o índice do drone em destaque.

        follow: drone em destaque (raios do LiDAR, rastro). nn_index: de quem é o cérebro mostrado.
        """
        leader = sim.leader() if follow is None else follow
        self.cur_leader = leader
        self._tick += 1
        self._draw_background(sim.scroll)
        self._draw_poles(sim)
        self._draw_obstacles(sim)
        self._draw_wind(sim, dt)
        self._draw_drones(sim, leader, obs)
        self._update_fx(sim, leader, dt)
        if marker and sim.alive[leader]:
            self.text(marker, 15, GOLD, (X0 * self.k, (sim.y[leader] - 36) * self.k), bold=True, anchor="midbottom")
        if hud:
            self._hud_stats(sim, gen_label, leader, score_label, rival, title)
            self._hud_chart(hist, None)
            if genomes is not None:
                ni = leader if nn_index is None else nn_index
                o = obs[ni] if obs is not None else sim.observe()[ni]
                self._hud_nn(genomes[ni], o)
        if caption:
            self._caption(caption, caption_alpha)
        return leader

    def card(self, lines: list[tuple[str, float, tuple, bool]], dim: int = 120, alpha: int = 255) -> None:
        """Tela de título/encerramento sobre o cenário (linhas = (texto, tamanho, cor, negrito))."""
        veil = self.pg.Surface((self.w, self.h), self.pg.SRCALPHA)
        veil.fill((6, 8, 20, dim))
        self.surf.blit(veil, (0, 0))
        total = sum(self.font(sz, b).get_height() * 1.25 for _, sz, _, b in lines)
        y = (self.h - total) / 2
        for txt, sz, col, bold in lines:
            self.text(txt, sz, col, (self.w / 2, y), bold=bold, anchor="midtop", alpha=alpha)
            y += self.font(sz, bold).get_height() * 1.25

    def background_only(self, scroll: float = 0.0) -> None:
        self._draw_background(scroll)

    def present(self) -> None:
        """Mostra o quadro na janela (modo ao vivo)."""
        if self.screen is None:
            return
        win = self.pg.display.get_surface()
        if win.get_size() != self.surf.get_size():
            self.pg.transform.smoothscale(self.surf, win.get_size(), win)
        else:
            win.blit(self.surf, (0, 0))
        self.pg.display.flip()

    def frame_bytes(self) -> bytes:
        """Quadro RGB bruto (para enviar ao ffmpeg)."""
        return self.pg.image.tobytes(self.surf, "RGB")

    def save(self, path: str) -> None:
        self.pg.image.save(self.surf, path)
