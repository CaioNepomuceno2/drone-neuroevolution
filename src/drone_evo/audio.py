"""Áudio 100% sintetizado em código (NumPy): efeitos, ronco do rotor e trilha musical.

Nada aqui usa amostras de terceiros, então tudo pode ser publicado sem direitos autorais.
  * sfx_*  -> efeitos curtos (mono, float -1..1)
  * music_loop() -> trilha em loop perfeito (estéreo)
  * build_soundtrack() -> mixagem offline para o vídeo (trilha + efeitos nos instantes certos)
  * AudioPlayer -> reprodução ao vivo com pygame.mixer
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
_rng = np.random.default_rng(42)


# ------------------------------------------------------------------ blocos básicos
def mtof(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def tt(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR)) / SR


def saw(f, t):
    return 2.0 * ((f * t) % 1.0) - 1.0


def tri(f, t):
    return 2.0 * np.abs(saw(f, t)) - 1.0


def sq(f, t, duty=0.5):
    return np.where((f * t) % 1.0 < duty, 1.0, -1.0)


def pluck_env(t, attack=0.003, tau=0.15):
    return np.minimum(t / attack, 1.0) * np.exp(-t / tau)


def lowpass(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    """Filtro passa-baixa por FFT (fase zero, circular: serve para loops)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    return np.fft.irfft(spec / np.sqrt(1.0 + (f / fc) ** (2 * order)), n=len(x))


def highpass(x: np.ndarray, fc: float) -> np.ndarray:
    return x - lowpass(x, fc, 2)


def mix_into(buf: np.ndarray, start: int, sig: np.ndarray, gain: float = 1.0) -> None:
    """Soma `sig` em `buf` a partir de `start`, dando a volta no fim (loop sem emenda)."""
    n = len(buf)
    start %= n
    first = min(len(sig), n - start)
    buf[start:start + first] += gain * sig[:first]
    if first < len(sig):
        buf[: len(sig) - first] += gain * sig[first:]


# ------------------------------------------------------------------ efeitos
def sfx_rotor(dur: float = 1.0) -> np.ndarray:
    """Zumbido do rotor, em loop perfeito (ciclos inteiros + ruído circular)."""
    t = tt(dur)
    hum = np.zeros_like(t)
    for h, amp in zip(range(1, 7), (1.0, 0.7, 0.5, 0.35, 0.25, 0.15)):
        hum += amp * np.sin(2 * np.pi * 120.0 * h * t + h)
    hum *= 1.0 + 0.25 * np.sin(2 * np.pi * 12.0 * t)
    noise = lowpass(_rng.normal(0, 1, len(t)), 1400.0)
    out = 0.5 * hum / np.abs(hum).max() + 0.35 * noise / np.abs(noise).max()
    return 0.6 * out / np.abs(out).max()


def sfx_ding() -> np.ndarray:
    t = tt(0.9)
    out = np.zeros_like(t)
    for f, a, tau in ((1318.5, 1.0, 0.32), (1975.5, 0.5, 0.22), (2637.0, 0.25, 0.14)):
        out += a * np.sin(2 * np.pi * f * t) * pluck_env(t, 0.002, tau)
    return 0.7 * out / np.abs(out).max()


def sfx_crash() -> np.ndarray:
    t = tt(0.5)
    noise = lowpass(_rng.normal(0, 1, len(t)), 1800.0) * np.exp(-t / 0.10)
    f = 40.0 + 90.0 * np.exp(-t / 0.07)
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.14)
    out = 0.8 * noise / np.abs(noise).max() + 0.9 * thump
    return 0.8 * out / np.abs(out).max()


def sfx_genup() -> np.ndarray:
    out = np.zeros(int(0.7 * SR))
    for i, m in enumerate((72, 76, 79, 84)):
        t = tt(0.4)
        mix_into(out, int(i * 0.09 * SR), tri(mtof(m), t) * pluck_env(t, 0.003, 0.13), 0.6)
    return 0.7 * out / np.abs(out).max()


def sfx_whoosh() -> np.ndarray:
    t = tt(0.8)
    noise = highpass(lowpass(_rng.normal(0, 1, len(t)), 2500.0), 300.0)
    env = np.sin(np.pi * np.clip(t / 0.8, 0, 1)) ** 2
    return 0.5 * noise / np.abs(noise).max() * env


SFX = {"ding": sfx_ding, "crash": sfx_crash, "genup": sfx_genup, "whoosh": sfx_whoosh}


# ------------------------------------------------------------------ música
CHORDS = ([57, 60, 64], [53, 57, 60], [55, 60, 64], [55, 59, 62])     # Am  F  C  G
ROOTS = (45, 41, 48, 43)


def music_loop(bars: int = 8, bpm: float = 96.0) -> np.ndarray:
    """Trilha calma (estilo synthwave) em Lá menor. Estéreo (n, 2), loop sem emenda."""
    beat = 60.0 / bpm
    bar = 4 * beat
    lb = int(bar * SR)
    n = lb * bars
    pad_l, pad_r = np.zeros(n), np.zeros(n)
    arp_l, arp_r = np.zeros(n), np.zeros(n)
    bass, drums = np.zeros(n), np.zeros(n)

    for b in range(bars):
        ch = b % 4
        start = b * lb
        tp = tt(bar + 0.7)
        env = np.minimum(tp / 0.5, 1.0) * np.clip((tp[-1] - tp) / 0.7, 0.0, 1.0)
        for m in CHORDS[ch]:
            f = mtof(m)
            mix_into(pad_l, start, (saw(f * 1.004, tp) + saw(f * 0.996, tp)) * env, 0.12)
            mix_into(pad_r, start, (saw(f * 1.007, tp) + saw(f * 0.993, tp)) * env, 0.12)
        # baixo: raiz em quartas pontuadas
        for pos in (0.0, 0.75, 2.0, 2.75):
            tb = tt(0.6)
            mix_into(bass, start + int(pos * beat * SR),
                     (tri(mtof(ROOTS[ch]), tb) + 0.4 * sq(mtof(ROOTS[ch]), tb, 0.3)) * pluck_env(tb, 0.004, 0.22), 0.30)
        # arpejo em colcheias
        tones = [m + 12 for m in CHORDS[ch]] + [CHORDS[ch][0] + 24]
        for e, idx in enumerate((0, 1, 2, 1, 3, 2, 1, 2)):
            ta = tt(0.4)
            note = 0.55 * tri(mtof(tones[idx]), ta) + 0.45 * sq(mtof(tones[idx]), ta, 0.35)
            sig = note * pluck_env(ta, 0.003, 0.11)
            pos = start + int(e * beat * 0.5 * SR)
            mix_into(arp_l, pos, sig, 0.045)
            mix_into(arp_r, pos, sig, 0.075)
        # bumbo e chimbal
        for k in (0, 2):
            tk = tt(0.32)
            ph = np.cumsum(46.0 + 110.0 * np.exp(-tk / 0.035)) * 2 * np.pi / SR
            mix_into(drums, start + int(k * beat * SR), np.sin(ph) * np.exp(-tk / 0.13), 0.55)
        for e in range(8):
            th = tt(0.08)
            hat = highpass(_rng.normal(0, 1, len(th)), 6000.0) * np.exp(-th / 0.022)
            mix_into(drums, start + int(e * beat * 0.5 * SR), hat, 0.07 if e % 2 else 0.035)

    pad_l, pad_r = lowpass(pad_l, 900.0), lowpass(pad_r, 900.0)
    left = pad_l + arp_l + bass + drums
    right = pad_r + arp_r + bass + drums
    out = np.stack([left, right], axis=1)
    return 0.55 * out / np.abs(out).max()


# ------------------------------------------------------------------ mixagem offline
def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(path: Path, stereo_or_mono: np.ndarray) -> None:
    data = to_int16(stereo_or_mono)
    ch = 1 if data.ndim == 1 else data.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def build_soundtrack(duration: float, events, music_gain: float = 0.55, rotor_gain: float = 0.10,
                     rotor_spans=None, fade_in: float = 0.6, fade_out: float = 2.0) -> np.ndarray:
    """Trilha + ronco do rotor + efeitos em instantes dados. `events` = [(segundos, nome, ganho)].

    Devolve estéreo float (n, 2) com `duration` segundos.
    """
    n = int(duration * SR)
    out = np.zeros((n, 2))
    m = music_loop()
    reps = int(np.ceil(n / len(m)))
    out += music_gain * np.tile(m, (reps, 1))[:n]

    rotor = sfx_rotor()
    for t0, t1 in (rotor_spans or [(0.0, duration)]):
        a, b = int(t0 * SR), min(n, int(t1 * SR))
        if b <= a:
            continue
        seg = np.tile(rotor, int(np.ceil((b - a) / len(rotor))))[: b - a]
        fade = np.minimum(1.0, np.minimum(np.arange(b - a), np.arange(b - a)[::-1]) / (0.15 * SR))
        out[a:b] += (rotor_gain * seg * fade)[:, None]

    cache = {name: fn() for name, fn in SFX.items()}
    for t, name, gain in events:
        a = int(t * SR)
        if a >= n or a < 0:
            continue
        sig = cache[name]
        b = min(n, a + len(sig))
        out[a:b] += (gain * sig[: b - a])[:, None]

    ramp = np.ones(n)
    fi, fo = int(fade_in * SR), int(fade_out * SR)
    ramp[:fi] = np.linspace(0, 1, fi)
    ramp[n - fo:] = np.linspace(1, 0, fo)
    out *= ramp[:, None]
    peak = np.abs(out).max()
    return out / peak * 0.89 if peak > 0.89 else out


# ------------------------------------------------------------------ ao vivo
class AudioPlayer:
    """Toca música, rotor e efeitos durante o visualizador (pygame.mixer)."""

    def __init__(self, enabled: bool = True, music: bool = True):
        self.enabled = enabled
        self.sounds: dict = {}
        self._last_crash = -1.0
        if not enabled:
            return
        import pygame
        self.pg = pygame
        try:
            pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=512)
        except Exception as exc:   # sem placa de som
            print(f"[aviso] áudio desativado: {exc}")
            self.enabled = False
            return

        def snd(x):
            x = x if x.ndim == 2 else np.stack([x, x], axis=1)
            return pygame.sndarray.make_sound(np.ascontiguousarray(to_int16(x)))

        for name, fn in SFX.items():
            self.sounds[name] = snd(fn())
        self.sounds["rotor"] = snd(sfx_rotor())
        self.sounds["rotor"].set_volume(0.10)
        self.sounds["rotor"].play(loops=-1)
        self.music = snd(music_loop())
        self.music.set_volume(0.5 if music else 0.0)
        self.music.play(loops=-1)
        self.muted = False

    def play(self, name: str, volume: float = 1.0) -> None:
        if self.enabled and not self.muted:
            s = self.sounds[name]
            s.set_volume(volume)
            s.play()

    def crash(self, count: int, now: float) -> None:
        """Limita a rajada de batidas para não virar ruído quando 50 drones caem juntos."""
        if count and now - self._last_crash > 0.12:
            self._last_crash = now
            self.play("crash", min(0.6, 0.15 + 0.08 * np.sqrt(count)))

    def toggle_mute(self) -> None:
        if not self.enabled:
            return
        self.muted = not self.muted
        self.pg.mixer.pause() if self.muted else self.pg.mixer.unpause()


def export(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in SFX.items():
        write_wav(out_dir / f"{name}.wav", fn())
    write_wav(out_dir / "rotor_loop.wav", sfx_rotor())
    write_wav(out_dir / "musica_loop.wav", music_loop())
    print(f"áudios gravados em {out_dir}")


if __name__ == "__main__":
    import sys
    export(Path(sys.argv[1] if len(sys.argv) > 1 else "assets/audio"))
