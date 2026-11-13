"""Testes do núcleo: física, sensores, colisão, inspeção, rede, algoritmo genético e checkpoint."""
from __future__ import annotations

import numpy as np

from drone_evo import brain
from drone_evo.config import (
    GROUND_Y, NORMAL, N_INPUTS, N_RAYS, RADIUS, RAY_ANGLES_DEG, RAY_LEN, X0,
)
from drone_evo.evolution import GAParams, load_checkpoint, next_generation
from drone_evo.train import TrainConfig, train
from drone_evo.world import Course, Sim


def empty_course(poles=True) -> Course:
    c = Course(1, NORMAL)
    c.set_obstacles([], [], [], [], [], [], [], [], [])
    return c


def one_wall(xl_at_start: float, top: float = 0.0, height: float = 500.0) -> Course:
    """Um paredão fixo cujo lado esquerdo começa em x = xl_at_start na tela."""
    c = empty_course()
    c.set_obstacles([0], [xl_at_start - X0], [40.0], [top], [height], [0.0], [0.0], [0.0], [0.0])
    return c


def sim_at(course: Course, ys) -> Sim:
    sim = Sim(course, len(ys), seed=0, y_jitter=0.0)
    sim.y = np.array(ys, dtype=float)
    return sim


def test_course_is_deterministic():
    a, b, c = Course(5, NORMAL), Course(5, NORMAL), Course(6, NORMAL)
    assert np.array_equal(a.xw, b.xw) and np.array_equal(a.yt, b.yt) and np.array_equal(a.pole_y, b.pole_y)
    assert len(a.xw) != len(c.xw) or not np.array_equal(a.xw, c.xw)


def test_hover_keeps_altitude():
    sim = sim_at(empty_course(), [250.0])
    for _ in range(120):
        sim.step(np.array([0.5]))
    assert abs(sim.y[0] - 250.0) < 1e-6 and abs(sim.vy[0]) < 1e-6 and sim.alive[0]


def test_no_thrust_falls_to_the_ground():
    sim = sim_at(empty_course(), [250.0])
    for _ in range(300):
        sim.step(np.array([0.0]))
    assert not sim.alive[0] and sim.cause[0] == 2


def test_full_thrust_hits_the_ceiling():
    sim = sim_at(empty_course(), [250.0])
    for _ in range(300):
        sim.step(np.array([1.0]))
    assert not sim.alive[0] and sim.cause[0] == 2


def test_rays_match_geometry():
    # os obstáculos são "engordados" pelo raio do drone: o raio mede a distância até o CORPO bater
    sim = sim_at(one_wall(X0 + 100.0), [250.0])
    prox = sim.rays()[0]
    for ang, p in zip(RAY_ANGLES_DEG, prox):
        expected = (100.0 - RADIUS) / np.cos(np.deg2rad(ang))
        assert abs(p - (1.0 - expected / RAY_LEN)) < 1e-6, (ang, p)


def test_rays_see_ground_and_ceiling():
    sim = sim_at(empty_course(), [450.0, 50.0])
    prox = sim.rays()
    steepest = np.sin(np.deg2rad(RAY_ANGLES_DEG[-1]))
    down = 1.0 - (GROUND_Y - RADIUS - 450.0) / steepest / RAY_LEN        # raio mais inclinado para baixo
    assert abs(prox[0, -1] - np.clip(down, 0.0, 1.0)) < 1e-6
    assert prox[1, 0] > 0.0                  # raio mais inclinado para cima chega ao teto quando está alto
    assert prox[0, N_RAYS // 2] == 0.0       # raio horizontal não vê nada


def test_small_bird_between_two_rays_is_still_seen():
    c = empty_course()
    # pássaro 30x20 a 200 px, centro 27 px abaixo do raio horizontal (o raio de 15° passaria 26 px mais abaixo)
    c.set_obstacles([2], [200.0], [30.0], [267.0], [20.0], [0.0], [0.0], [0.0], [0.0])
    sim = sim_at(c, [250.0])
    assert sim.rays()[0].max() > 0.0


def test_collision_kills():
    sim = sim_at(one_wall(X0 + 30.0), [250.0, 250.0])
    for _ in range(15):
        sim.step(np.array([0.5, 0.5]))
    assert not sim.alive.any() and (sim.cause == 1).all()
    assert (sim.score() > 0).all()


def test_flying_over_a_low_wall_is_safe():
    # paredão só do chão até y=400: o drone a y=200 passa por cima
    sim = sim_at(one_wall(X0 + 30.0, top=400.0, height=100.0), [200.0])
    for _ in range(60):
        sim.step(np.array([0.5]))
    assert sim.alive[0]


def test_inspection_requires_the_right_height():
    c = empty_course()
    c.pole_y[0] = 250.0
    sim = sim_at(c, [250.0, 400.0])
    while sim.scroll < c.pole_x[0] + 5:
        sim.step(np.array([0.5, 0.5]))
    assert sim.inspections.tolist() == [1, 0]
    assert sim.score()[0] > sim.score()[1]


def test_observation_shape_and_range():
    sim = Sim(Course(3, NORMAL), 10, seed=3)
    obs = sim.observe()
    assert obs.shape == (10, N_INPUTS)
    assert np.isfinite(obs).all() and obs.min() >= -1.0 and obs.max() <= 1.0


def test_population_forward_matches_single_genome():
    rng = np.random.default_rng(0)
    pop = brain.random_population(7, rng)
    obs = rng.uniform(-1, 1, (7, N_INPUTS))
    thrust = brain.forward(pop, obs)
    assert thrust.shape == (7,) and thrust.min() >= 0.0 and thrust.max() <= 1.0
    for i in range(7):
        _, o = brain.forward_detail(pop[i], obs[i])
        assert abs(thrust[i] - 0.5 * (o[0] + 1.0)) < 1e-12


def test_next_generation_keeps_elite_and_size():
    rng = np.random.default_rng(1)
    p = GAParams(pop=50, elite=5, immigrants=3)
    pop = brain.random_population(50, rng)
    fit = rng.uniform(0, 100, 50)
    new = next_generation(pop, fit, rng, 0.3, p)
    assert new.shape == pop.shape
    best5 = pop[np.argsort(-fit)[:5]]
    assert np.array_equal(new[:5], best5)


def small_cfg(out, gens):
    return TrainConfig(out=out, gens=gens, seed=7,
                       ga=GAParams(pop=60, episodes=1, max_seconds=15.0))


def test_evolution_improves(tmp_path):
    _, hist = train(small_cfg(tmp_path / "run", 14), log=lambda *_: None)
    # com voos de 15 s o melhor já satura na geração 1; o que prova a evolução é a média da população
    assert hist[-3:, 1].mean() > 1.5 * hist[:3, 1].mean()


def test_resume_gives_identical_result(tmp_path):
    a, _ = train(small_cfg(tmp_path / "a", 6), log=lambda *_: None)
    train(small_cfg(tmp_path / "b", 3), log=lambda *_: None)
    b, _ = train(small_cfg(tmp_path / "b", 6), log=lambda *_: None)     # retoma da geração 3
    assert np.array_equal(a, b)


def test_corrupted_checkpoint_falls_back_to_previous(tmp_path):
    train(small_cfg(tmp_path, 3), log=lambda *_: None)
    (tmp_path / "checkpoint.npz").write_bytes(b"lixo")                  # simula escrita interrompida
    ck = load_checkpoint(tmp_path)
    assert ck is not None and ck["meta"]["gen"] == 2


def test_step_cost_is_small():
    import time
    sim = Sim(Course(1, NORMAL), 200, seed=1)
    t = time.perf_counter()
    for _ in range(200):
        sim.step(sim.observe()[:, 0] * 0 + 0.5)
    assert (time.perf_counter() - t) / 200 < 0.005          # < 5 ms por passo com 200 drones
