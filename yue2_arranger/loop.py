"""
Seamless, bar-aligned loops from a rendered song (for menus, hubs, background music).

Takes the longest whole-bar span from the first beat that still leaves `xfade_bars` of tail, then equal-power
crossfades that tail into the loop's first bars. Played on repeat there is no seam, and the beat grid stays
continuous (beat k at k * 60 / bpm from the loop start). Tempo and the first beat come from variety.fit_tempo
unless given.
"""
from __future__ import annotations

import numpy as np

from . import variety


def make_loop(audio: np.ndarray, sample_rate: int, bpm: float | None = None, first_beat_s: float | None = None,
              beats_per_bar: int = 4, xfade_bars: int = 2, max_bars: int = 0) -> tuple[np.ndarray, dict]:
    """audio: (channels, samples) float. Returns (loop audio (channels, samples), info)."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or x.shape[0] > 16:
        raise ValueError(f"expected (channels, samples) audio, got shape {x.shape}")
    if not 1 <= beats_per_bar <= 16 or not 1 <= xfade_bars <= 16 or not 0 <= max_bars <= 4096:
        raise ValueError("beats_per_bar and xfade_bars must be 1..16, max_bars 0..4096")
    if bpm is not None and not 30 <= bpm <= 300:
        raise ValueError("bpm must be 30..300")
    if bpm is None or first_beat_s is None:
        mono = variety.to_mono(x, sample_rate)
        _, _, flux, _, dt = variety.frame_features(mono)
        fit_bpm, phase, _ = variety.fit_tempo(flux, dt, bpm)
        bpm = bpm or fit_bpm
        first_beat_s = phase if first_beat_s is None else first_beat_s
    bar = beats_per_bar * 60 / bpm
    s0 = int(round(first_beat_s * sample_rate))
    avail = (x.shape[1] - s0) / sample_rate
    nbars = int((avail - xfade_bars * bar) // bar)
    if max_bars:
        nbars = min(nbars, max_bars)
    if nbars < 1:
        raise ValueError(f"audio too short for a loop of whole bars plus a {xfade_bars}-bar crossfade at {bpm:.1f} bpm")
    n, xf = int(round(nbars * bar * sample_rate)), int(round(xfade_bars * bar * sample_rate))
    y = x[:, s0:s0 + n].copy()
    tail = x[:, s0 + n:s0 + n + xf]
    xf = min(xf, tail.shape[1], y.shape[1])
    t = np.linspace(0, np.pi / 2, xf, dtype=np.float32)[None]
    y[:, :xf] = y[:, :xf] * np.sin(t) + tail[:, :xf] * np.cos(t)
    return y, dict(bpm=round(float(bpm), 3), bars=nbars, loop_seconds=round(n / sample_rate, 3),
                   xfade_bars=xfade_bars, first_beat_s=round(float(first_beat_s), 3))
