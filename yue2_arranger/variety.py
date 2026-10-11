"""
How repetitive does a song sound? Deterministic audio measure (numpy only).

Per bar (on a fitted beat grid) it takes a 12-bin chroma (harmony/melody) and 24 log-band levels
(timbre/arrangement), then compares every bar with every EARLIER bar. A bar is "new" when no earlier bar
is close on BOTH (cosine below `new_sim`). From that:

  new_bars        share of bars that bring something not heard before
  last_new_s      time of the last new bar (a song that stops inventing at 1:10 loops after that)
  longest_flat_s  longest stretch without a new bar
  loop_share      share of bars that are near-copies of the bar 1, 2, 4 or 8 bars back
  sections        rough count of blocks that differ from their neighbours (novelty peaks)
  restarts        times after the opening that sound like the opening again (a splice that restarts the song)
  verdict         "varied" / "repetitive" against simple thresholds (tuned on a handful of YuE2 renders)
"""
from __future__ import annotations

import numpy as np

SR = 22050
NFFT, HOP = 4096, 512
MAX_SECONDS = 30 * 60
MAX_BARS = 4096


def to_mono(waveform, sample_rate: int) -> np.ndarray:
    """(channels, samples) or (samples,) float array at any rate -> mono float32 at SR."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(0)
    elif x.ndim != 1:
        raise ValueError(f"expected (channels, samples) audio, got shape {x.shape}")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    if len(x) / sample_rate > MAX_SECONDS:
        raise ValueError(f"audio longer than {MAX_SECONDS // 60} minutes")
    if sample_rate != SR:
        if sample_rate > SR:  # low-pass below the new Nyquist first, or high content folds into the features
            x = _lowpass(x, 0.45 * SR / sample_rate)
        n = int(round(len(x) * SR / sample_rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def _lowpass(x: np.ndarray, cutoff: float, taps: int = 63) -> np.ndarray:
    """Windowed-sinc FIR low-pass; cutoff as a fraction of the sample rate (0..0.5). Blockwise to bound memory."""
    k = np.arange(taps) - (taps - 1) / 2
    h = (2 * cutoff * np.sinc(2 * cutoff * k) * np.hamming(taps)).astype(np.float32)
    h /= h.sum()
    out = np.empty_like(x)
    step = 1 << 20
    pad = np.concatenate([np.zeros(taps // 2, np.float32), x, np.zeros(taps // 2, np.float32)])
    for a in range(0, len(x), step):
        b = min(len(x), a + step)
        out[a:b] = np.convolve(pad[a:b + taps - 1], h, mode="valid")
    return out


def frame_features(x: np.ndarray, block: int = 256):
    """Per-frame chroma (12), log bands (24), spectral flux and magnitude sum, computed block by block so a
    long song never holds its whole spectrogram in memory."""
    win = np.hanning(NFFT).astype(np.float32)
    xp = np.pad(x, (0, NFFT))
    n_frames = 1 + len(x) // HOP
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    sel = (f > 55) & (f < 4000)
    pcs = (np.round(12 * np.log2(f[sel] / 440.0)) % 12).astype(int)
    pc_masks = [pcs == k for k in range(12)]
    edges = np.geomspace(60, 10000, 25)
    band_masks = [(f >= edges[i]) & (f < edges[i + 1]) for i in range(24)]
    chroma = np.zeros((n_frames, 12), np.float32)
    bands = np.zeros((n_frames, 24), np.float32)
    flux = np.zeros(n_frames, np.float32)
    msum = np.zeros(n_frames, np.float32)
    prev = None
    for a in range(0, n_frames, block):
        idx = np.arange(a, min(n_frames, a + block))
        frames = np.lib.stride_tricks.sliding_window_view(xp, NFFT)[idx * HOP]
        mag = np.abs(np.fft.rfft(frames * win, axis=1)).astype(np.float32)
        sub = mag[:, sel]
        chroma[idx] = np.stack([sub[:, m].sum(1) for m in pc_masks], 1)
        bands[idx] = np.stack([np.log1p(mag[:, m].mean(1)) for m in band_masks], 1)
        lg = np.log1p(mag)
        d = np.diff(lg, axis=0, prepend=lg[:1] if prev is None else prev[None])
        flux[idx] = np.maximum(d, 0).sum(1)
        if prev is None:
            flux[idx[0]] = 0.0
        prev = lg[-1]
        msum[idx] = mag.sum(1)
    return chroma, bands, flux, msum, HOP / SR


def fit_tempo(flux: np.ndarray, dt: float, bpm_hint: float | None = None) -> tuple[float, float, bool]:
    """(bpm, phase seconds, halved): `halved` when a detected tempo above 160 was taken as half-time."""
    f = flux - flux.mean()
    halved = False
    if bpm_hint:
        bpm = float(bpm_hint)
    else:
        # flux autocorrelation (via FFT: linear in length) scored at the beat AND bar multiples, so 4/3 and 3/2
        # aliases lose
        nfft = 1 << int(np.ceil(np.log2(2 * len(f))))
        spec = np.fft.rfft(f, nfft)
        ac = np.fft.irfft(spec * np.conj(spec), nfft)[:len(f)]
        ac = ac / (ac[0] + 1e-12)
        grid = np.arange(len(ac))
        cands = np.arange(70, 180.01, 0.25)
        lag = 60 / cands / dt
        score = (np.interp(lag, grid, ac) + np.interp(2 * lag, grid, ac) + np.interp(4 * lag, grid, ac)
                 + 0.5 * np.interp(8 * lag, grid, ac))
        bpm = float(cands[int(np.argmax(score))])
        if bpm > 160:  # a 4-beat "bar" above 160 is usually half a bar of a song in the 80s
            bpm /= 2
            halved = True
    best = (-np.inf, bpm, 0.0)
    for b in np.arange(bpm - 0.5, bpm + 0.51, 0.05):  # refine tempo + phase with a comb on the flux
        period = 60 / b / dt
        for ph in np.linspace(0, period, 32, endpoint=False):
            i = np.round(np.arange(ph, len(f), period)).astype(int)
            s = f[i[i < len(f)]].mean()
            if s > best[0]:
                best = (s, b, ph * dt)
    return float(best[1]), float(best[2]), halved


def _opening_matches(chroma, bands, dt, win_s=4.0, skip_s=20.0):
    """Cosine similarity of every 4 s window to the song's opening window (log-spectral shape)."""
    F = np.concatenate([chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9), bands], 1)
    F = F - F.mean(1, keepdims=True)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    step = max(1, int(0.1 / dt))
    w = max(1, int(win_s / dt))
    start = int(0.25 / dt)
    if len(F) < start + 2 * w:
        return [], 0.0
    ref = F[start:start + w]
    sims = []
    for t in range(0, len(F) - w, step):
        sims.append((t * dt, float((F[t:t + w] * ref).sum(1).mean())))
    near = [s for t, s in sims if 1.0 < t < 15.0]
    self_sim = max(near) if near else 1.0
    if not np.isfinite(self_sim) or self_sim <= 0.05:  # opening too featureless to compare against
        return [], 1.0
    return [(t, s) for t, s in sims if t > skip_s], self_sim


def analyse(waveform, sample_rate: int, bpm_hint: float | None = None, beats_per_bar: int = 4,
            new_sim: float = 0.9, loop_sim: float = 0.97, restart_ratio: float = 0.95,
            min_new_share: float = 0.25, max_flat_s: float = 75.0) -> dict:
    if bpm_hint is not None and not 30 <= bpm_hint <= 300:
        raise ValueError("bpm_hint must be between 30 and 300 (or 0 / None to detect)")
    if not 1 <= int(beats_per_bar) <= 16:
        raise ValueError("beats_per_bar must be between 1 and 16")
    x = to_mono(waveform, sample_rate)
    if len(x) < SR * 5:
        raise ValueError("audio shorter than 5 seconds")
    if float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) < 1e-4:
        raise ValueError("audio is silent")
    chroma_f, bands_f, flux, msum, dt = frame_features(x)
    bpm, phase, halved = fit_tempo(flux, dt, bpm_hint)
    bar = beats_per_bar * 60 / bpm
    starts = np.arange(phase, len(flux) * dt - bar * 0.5, bar)
    C, B, loud = [], [], []
    for s in starts:
        a, b = int(s / dt), max(int(s / dt) + 1, int((s + bar) / dt))
        C.append(chroma_f[a:b].mean(0)); B.append(bands_f[a:b].mean(0)); loud.append(np.log1p(msum[a:b].mean()))
    C, B, loud = np.array(C), np.array(B), np.array(loud)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    B = B - B.mean(0)  # timbre relative to the song's average
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    if len(starts) < 8:
        raise ValueError(f"only {len(starts)} bars at {bpm:.0f} bpm; need at least 8")
    if len(starts) > MAX_BARS:  # the bar x bar similarity matrices grow with the square of this
        raise ValueError(f"{len(starts)} bars; at most {MAX_BARS}")
    keep = loud > np.median(loud) - 3.0  # drop near-silent bars (lead-in, tail, gaps): ~26 dB under the median
    n = len(starts)
    SC, SB = C @ C.T, B @ B.T
    both = np.minimum(SC, SB)
    new = np.zeros(n, bool)
    kept_idx = np.flatnonzero(keep)
    for pos, i in enumerate(kept_idx):
        prev = kept_idx[:pos]
        new[i] = True if not len(prev) else both[i, prev].max() < new_sim
    loop = np.zeros(n, bool)
    for i in range(n):
        for lag in (1, 2, 4, 8):
            if i - lag >= 0 and SC[i, i - lag] >= loop_sim and SB[i, i - lag] >= loop_sim:
                loop[i] = True
                break
    run = best = 0
    for i in kept_idx:  # silent bars neither count as nothing-new nor break a run
        run = 0 if new[i] else run + 1
        best = max(best, run)
    F = np.concatenate([C, B], 1)
    nov = np.array([np.linalg.norm(F[i - 4:i].mean(0) - F[i:i + 4].mean(0)) if 4 <= i <= n - 4 else 0.0 for i in range(n)])
    pos_nov = nov[nov > 0]
    thr = pos_nov.mean() + pos_nov.std() if len(pos_nov) else 1.0
    peaks = [i for i in range(1, n - 1) if nov[i] > thr and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    sims, self_sim = _opening_matches(chroma_f, bands_f, dt)
    restarts, last = [], -1e9
    for t, s in sims:
        if s >= restart_ratio * self_sim and t - last > 8.0:
            restarts.append(dict(t=round(t, 1), similarity=round(s / self_sim, 3)))
            last = t
    k = max(1, int(keep.sum()))
    idx_new = np.flatnonzero(new)
    new_share = float(new.sum() / k)
    flat_s = best * bar
    problems = []
    if new_share < min_new_share:
        problems.append(f"only {100 * new_share:.0f}% of bars are new (< {100 * min_new_share:.0f}%)")
    if flat_s > max_flat_s:
        problems.append(f"{flat_s:.0f} s without anything new (> {max_flat_s:.0f} s)")
    early = [r for r in restarts if r["t"] < 0.75 * len(x) / SR]  # a reprise near the end is normal
    if early:
        problems.append("sounds like it starts over at " + ", ".join(f"{r['t']:.0f}s" for r in early))
    return dict(duration_s=round(len(x) / SR, 1), bpm=round(bpm, 2), bars=k, new_bars=round(new_share, 3),
                last_new_s=round(float(starts[idx_new[-1]]) if len(idx_new) else 0.0, 1),
                longest_flat_s=round(flat_s, 1), loop_share=round(float(loop[keep].mean()) if keep.any() else 0.0, 3),
                sections=len(peaks) + 1, restarts=restarts, verdict="repetitive" if problems else "varied",
                problems=problems, tempo_note=(f"detected {2 * bpm:.0f} bpm, analysed as {bpm:.0f} (half-time); "
                                               "if the song really is that fast, pass bpm_hint") if halved else "", new_bar_times=[round(float(starts[i]), 1) for i in idx_new])


def summary(r: dict, name: str = "") -> str:
    head = f"{name}: " if name else ""
    m, s = divmod(int(round(r["duration_s"])), 60)
    lines = [f"{head}{r['verdict'].upper()}  ({m}:{s:02d}, {r['bpm']:.1f} bpm, {r['bars']} bars)",
             f"  new bars {100 * r['new_bars']:.0f}%   last new bar {r['last_new_s']:.0f}s   "
             f"longest stretch with nothing new {r['longest_flat_s']:.0f}s   near-copies {100 * r['loop_share']:.0f}%"]
    lines += [f"  - {p}" for p in r["problems"]]
    if r.get("tempo_note"):
        lines.append(f"  note: {r['tempo_note']}")
    return "\n".join(lines)
