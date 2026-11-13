"""Rede neural do drone, escrita do zero em NumPy (sem PyTorch, sem TensorFlow).

Arquitetura: 13 entradas -> 14 neurônios (tanh) -> 1 saída (tanh).
A saída vira o empuxo, de 0 a 1: 0,5 = pairar, acima sobe, abaixo desce.
Um "genoma" é só o vetor com todos os pesos e vieses da rede (211 números).
A população inteira é uma matriz (N, 133) e roda em uma única multiplicação.
"""
from __future__ import annotations

import numpy as np

from .config import N_INPUTS

N_IN = N_INPUTS
N_HID = 14
N_OUT = 1

_A = N_IN * N_HID
_B = _A + N_HID
_C = _B + N_HID * N_OUT
GENOME_SIZE = _C + N_OUT


def split(g: np.ndarray):
    """Separa o(s) genoma(s) em W1, b1, W2, b2 (aceita (G,) ou (N, G))."""
    lead = g.shape[:-1]
    w1 = g[..., :_A].reshape(lead + (N_IN, N_HID))
    b1 = g[..., _A:_B].reshape(lead + (N_HID,))
    w2 = g[..., _B:_C].reshape(lead + (N_HID, N_OUT))
    b2 = g[..., _C:].reshape(lead + (N_OUT,))
    return w1, b1, w2, b2


def forward(genomes: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Empuxo (N,) de cada drone, dado (N, G) genomas e (N, N_IN) entradas."""
    w1, b1, w2, b2 = split(genomes)
    h = np.tanh(np.matmul(obs[:, None, :], w1)[:, 0, :] + b1)
    o = np.tanh(np.matmul(h[:, None, :], w2)[:, 0, :] + b2)
    return 0.5 * (o[:, 0] + 1.0)


def forward_detail(genome: np.ndarray, obs: np.ndarray):
    """Um drone só, devolvendo as ativações (para desenhar a rede): (oculta, saída bruta)."""
    w1, b1, w2, b2 = split(genome)
    h = np.tanh(obs @ w1 + b1)
    o = np.tanh(h @ w2 + b2)
    return h, o


def random_population(n: int, rng: np.random.Generator, scale: float = 0.7) -> np.ndarray:
    return rng.normal(0.0, scale, (n, GENOME_SIZE))
