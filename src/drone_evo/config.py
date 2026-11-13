"""Constantes do mundo e níveis de dificuldade.

Convenções: o mundo tem 960x540 unidades "lógicas" (o renderizador só escala).
O eixo y cresce para baixo, como na tela. O drone fica sempre no mesmo x da
tela (X0): quem se move é o cenário, que rola para a esquerda.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- geometria -------------------------------------------------------------
W = 960.0            # largura lógica
H = 540.0            # altura lógica
GROUND_Y = 500.0     # y do chão
X0 = 220.0           # x fixo do drone na tela
RADIUS = 14.0        # raio do corpo do drone (colisão)

# --- física ----------------------------------------------------------------
DT = 1.0 / 60.0
GRAVITY = 900.0
MAX_THRUST_ACC = 1800.0   # com empuxo 0.5 a aceleração vertical é zero (pairar)
DRAG = 1.2
VY_MAX = 380.0
ACC_EFF = 700.0           # aceleração vertical "útil" (usada para garantir percursos possíveis)
SWITCH_SAFETY = 2.2       # folga extra de tempo para trocar de faixa entre árvore e galho (uma rede reativa não voa no limite)

# --- sensores (LiDAR) -------------------------------------------------------
RAY_ANGLES_DEG = tuple(float(a) for a in range(-60, 61, 15))   # 9 raios; positivo = para baixo
RAY_LEN = 380.0
N_RAYS = len(RAY_ANGLES_DEG)
N_INPUTS = N_RAYS + 4     # raios + altura + vel. vertical + dy/dx do isolador

# --- missão: inspecionar isoladores nos postes ------------------------------
POLE_SPACING = 1000.0
POLE_FIRST = 900.0
INSPECT_TOL = 48.0        # tolerância vertical para contar a inspeção
INSPECT_BONUS = 40.0      # pontos por isolador inspecionado
SCORE_PER_PX = 0.1        # 1 ponto a cada 10 px voados

# tipos de obstáculo
TREE, BRANCH, BIRD = 0, 1, 2


@dataclass(frozen=True)
class Difficulty:
    name: str
    label: str
    speed0: float                  # velocidade inicial de rolagem (px/s)
    speed_max: float               # velocidade máxima
    ramp_px: float                 # distância até chegar na velocidade máxima
    gap: tuple[float, float]       # espaçamento entre obstáculos (px)
    bird_prob: float               # chance de um obstáculo ser pássaro
    bird_vx: tuple[float, float]   # velocidade própria dos pássaros (px/s)
    bird_amp: tuple[float, float]  # amplitude do voo ondulado (px)
    wind: float                    # aceleração máxima das rajadas (px/s²)


NORMAL = Difficulty("normal", "Normal", 230.0, 300.0, 6000.0,
                    (320.0, 480.0), 0.45, (30.0, 90.0), (0.0, 50.0), 0.0)

HARD = Difficulty("hard", "Difícil", 280.0, 420.0, 6000.0,
                  (250.0, 380.0), 0.55, (80.0, 180.0), (20.0, 110.0), 380.0)

DIFFICULTIES = {d.name: d for d in (NORMAL, HARD)}
