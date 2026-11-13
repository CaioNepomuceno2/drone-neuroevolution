"""Testes de renderização (fora da tela) e de áudio sintetizado."""
from __future__ import annotations

import numpy as np

from drone_evo import audio
from drone_evo.brain import forward, random_population
from drone_evo.config import HARD, NORMAL, N_INPUTS
from drone_evo.render import Renderer
from drone_evo.world import Course, Sim


def test_renderer_draws_a_full_frame_headless():
    r = Renderer(640, 360, headless=True)
    rng = np.random.default_rng(0)
    g = random_population(30, rng)
    for diff in (NORMAL, HARD):
        sim = Sim(Course(2, diff), 30, seed=2)
        r.reset()
        hist = np.column_stack([np.linspace(10, 500, 20), np.linspace(5, 200, 20)])
        for _ in range(90):
            obs = sim.observe()
            sim.step(forward(g, obs))
            r.note_events(sim)
        r.draw(sim, genomes=g, obs=obs, hist=hist, gen_label="7", caption="legenda de teste\nsegunda linha")
        px = np.frombuffer(r.frame_bytes(), dtype=np.uint8).reshape(360, 640, 3)
        assert px.std() > 10 and len(np.unique(px.reshape(-1, 3), axis=0)) > 200   # tem cenário de verdade


def test_audio_is_finite_and_has_expected_length():
    for name, fn in audio.SFX.items():
        x = fn()
        assert np.isfinite(x).all() and 0.0 < np.abs(x).max() <= 1.0, name
    m = audio.music_loop()
    assert m.shape == (int(8 * 4 * 60.0 / 96.0 * audio.SR), 2) and np.isfinite(m).all()
    assert abs(m[0] - m[-1]).max() < 0.1                      # o loop fecha sem estalo
    rotor = audio.sfx_rotor()
    assert abs(rotor[0] - rotor[-1]) < 0.1


def test_soundtrack_places_events_and_stays_below_clipping():
    track = audio.build_soundtrack(6.0, [(1.0, "ding", 0.5), (2.0, "crash", 0.5), (3.0, "genup", 0.5)])
    assert track.shape == (6 * audio.SR, 2)
    assert np.abs(track).max() <= 0.9
    quiet = np.abs(track[int(0.9 * audio.SR):int(1.0 * audio.SR)]).max()
    ding = np.abs(track[int(1.0 * audio.SR):int(1.1 * audio.SR)]).max()
    assert ding > quiet                                        # o "ding" aparece onde foi pedido


def test_input_size_matches_network():
    from drone_evo.brain import N_IN
    assert N_IN == N_INPUTS
