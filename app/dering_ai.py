"""Metallic de-ringing: Demucs-residual-taught adaptive notch equalization.

Why this module exists: codec metallic/blechy ringing is COHERENT (follows the
music envelope) and often sits at stable narrow resonances, so envelope-
incoherence gating and flicker detectors cannot see it. The teacher is the
Demucs residual (input - model-rendered stems): it contains exactly what the
network does NOT model as music - codec ringing, aliasing combs, RVQ residue.

Pipeline per pass:
1. separate() -> stems; residual = mix - sum(stems)
2. find PROMINENT narrow peaks in the residual mean spectrum (>= spike_db above
   the +/-40-bin median) -> those are ringing frequencies
3. music guard: if the MUSIC (stems sum) has a comparable narrow peak at the
   same frequency, the peak is likely musical (flanger etc.) -> notch depth x0.35
4. apply smooth gaussian notch bank to the FULL mix, transient-protected
5. loudness compensation so the pass is loudness-neutral

Metrics for verification (metallic_metrics): spectral kurtosis (spikiness)
and notch-peak band energies before/after.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .analysis import stft, istft
from .loudness import integrated_loudness
from .enhance import separate, get_or_load_model
from .restoration import detect_transients


# ------------------------------------------------------------------ helpers

def _smooth_curve(v: np.ndarray, bins: int = 21) -> np.ndarray:
    """Moving average that keeps the input length."""
    bins = int(bins) | 1
    pad = bins // 2
    p = np.pad(v, (pad, bins - 1 - pad), mode="edge")
    return np.convolve(np.ravel(p), np.ones(bins) / bins, mode="same")[: len(v)]


def _narrow_peaks(db: np.ndarray, idx: np.ndarray, spike_db: float) -> list:
    """Local maxima of a dB curve, >= spike_db above local median, merged <5 bins."""
    peaks = []
    for i in range(3, len(db) - 3):
        if db[i] == np.max(db[i - 3:i + 4]):
            lo_w = max(0, i - 40)
            med = float(np.median(db[lo_w:i + 40]))
            prom = db[i] - med
            if prom >= spike_db:
                peaks.append((int(idx[i]), prom))
    peaks.sort()
    merged = []
    for b, prom in peaks:
        if merged and b - merged[-1][0] < 5:
            if prom > merged[-1][1]:
                merged[-1] = (b, prom)
        else:
            merged.append((b, prom))
    return merged


def _band_prom_db(avg_db: np.ndarray, idx: np.ndarray, bin_pos: int) -> float:
    """Prominence of bin bin_pos inside the band spectrum (dB over local median)."""
    j = int(np.searchsorted(idx, bin_pos))
    j = min(max(j, 0), len(idx) - 1)
    lo_w = max(0, j - 40)
    return float(avg_db[j] - np.median(avg_db[lo_w:j + 40]))


def residual_notch_curve(residual: np.ndarray, music: np.ndarray | None, sr: int,
                         n_fft: int = 4096, spike_db: float = 6.0,
                         f_lo: float = 2500.0, f_hi: float | None = None,
                         music_guard: bool = True) -> tuple[np.ndarray, dict]:
    """Per-bin notch gain (0..1) from prominent narrow peaks in the residual.

    music_guard: peaks that the MUSIC (stems sum) also exhibits with comparable
    prominence are treated as musical and their notch depth is reduced x0.35.
    """
    x = residual[:, 0] if residual.ndim > 1 else residual
    spec = np.abs(stft(x, n_fft, n_fft // 4)[0])
    avg = np.mean(spec, axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    f_hi_eff = min(f_hi, sr / 2 - 1000) if f_hi else (sr / 2 - 1000)

    band = (freqs >= f_lo) & (freqs <= f_hi_eff)
    idx = np.where(band)[0]
    db = 20.0 * np.log10(np.maximum(avg[idx], 1e-9))
    merged = _narrow_peaks(db, idx, spike_db)

    music_db = None
    if music_guard and music is not None and np.size(music) > 0:
        s = music[:, 0] if music.ndim > 1 else music
        mspec = np.abs(stft(s, n_fft, n_fft // 4)[0])
        mavg = np.mean(mspec, axis=0)
        music_db = 20.0 * np.log10(np.maximum(mavg[idx], 1e-9))

    w = np.zeros(len(freqs))
    peaks_out, n_guarded = [], 0
    for b, prom in merged:
        depth = float(np.clip((prom - spike_db) / 12.0, 0.25, 1.0))
        mp = 0.0
        if music_db is not None:
            mp = _band_prom_db(music_db, idx, b)
        if mp > 0.6 * prom:
            depth *= 0.35
            n_guarded += 1
        if depth < 0.1:
            continue
        fc = float(freqs[b])
        sigma = max(fc * 0.04, 25.0)
        g = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
        w = np.maximum(w, depth * g)
        peaks_out.append((b, prom, depth, mp))
    w = np.clip(_smooth_curve(w, 5), 0.0, 1.0)
    report = {
        "peaks": [(round(float(freqs[b]), 1), round(float(p), 1),
                   round(float(d), 2)) for b, p, d, _ in peaks_out],
        "n_notches": len(peaks_out),
        "n_guarded": n_guarded,
        "max_weight": float(np.max(w)),
    }
    return w, report


def apply_notch_curve(audio: np.ndarray, sr: int, gain_curve: np.ndarray,
                      n_fft: int = 4096, max_db: float = 7.0,
                      transient_guard: float = 0.55) -> np.ndarray:
    """Apply the per-bin notch attenuation to a signal (transient-protected)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    hop = n_fft // 4
    att = 10.0 ** (-max_db / 20.0)
    bin_gain = 1.0 - gain_curve * (1.0 - att)
    outs = []
    for c in range(x.shape[1]):
        spec, win, hop_, pad = stft(x[:, c], n_fft, hop)
        mag = np.abs(spec)
        phase = np.exp(1j * np.angle(spec))
        g = np.repeat(bin_gain[None, :], len(mag), axis=0)
        onsets = detect_transients(mag, hop, sr)
        if onsets.any():
            lift = np.zeros(len(mag))
            lift[onsets] = 1.0
            lift = lfilter([0.6], [1.0, -0.4], lift)
            g = np.maximum(g, np.clip(lift, 0, 1)[:, None])
        y = istft(mag * g * phase, n_fft, hop, win, len(x[:, c]), pad)
        outs.append(y)
    y = np.stack(outs, axis=1) if x.shape[1] > 1 else outs[0]
    return y


def _pass(mix: np.ndarray, sr: int, model, spike_db: float, max_db: float,
          music_guard: bool, progress_cb=None) -> tuple[np.ndarray, dict]:
    """One deringing pass: separate -> residual peaks -> notches on the mix."""
    stems, out_sr = separate(mix.astype(np.float32), sr, model=model)
    if progress_cb:
        progress_cb(0.4, "stems")
    if sr != out_sr:
        import torch
        import torchaudio.functional as AF
        m = AF.resample(torch.from_numpy(mix.T[None].astype("float32")),
                        sr, out_sr)[0].numpy().T.astype("float64")
    else:
        m = mix
    n = min(len(m), min(len(s) for s in stems.values()))
    m = m[:n]
    stems = {k: v[:n] for k, v in stems.items()}
    music = np.sum(list(stems.values()), axis=0)
    residual = m - music

    gain_curve, rep = residual_notch_curve(residual, music, out_sr,
                                           spike_db=spike_db,
                                           music_guard=music_guard)
    if progress_cb:
        progress_cb(0.7, "notches")
    if rep["n_notches"] == 0:
        return m.astype(np.float32), out_sr, {**rep, "out_sr": out_sr}
    y = apply_notch_curve(m, out_sr, gain_curve, max_db=max_db)

    lin = integrated_loudness(m, out_sr)
    lout = integrated_loudness(y, out_sr)
    if np.isfinite(lin) and np.isfinite(lout):
        d = lin - lout
        if 0.05 < abs(d) < 12:
            y = y * (10 ** (d / 20.0))
            rep["loudness_compensation_db"] = float(d)
    rep["out_sr"] = out_sr
    return y.astype(np.float32), out_sr, rep


def dering_ai(audio: np.ndarray, sr: int, model=None,
              spike_db: float = 6.0, max_db: float = 7.0,
              passes: int = 1, music_guard: bool = True,
              progress_cb=None) -> tuple[np.ndarray, int, dict]:
    """Model-guided metallic de-ringing (iterative-capable).

    spike_db: prominence threshold (lower = more notches)
    max_db:   max notch depth
    passes:   iterate (re-analyze the deringed result) up to N times
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    model = model or get_or_load_model()
    y = x.astype(np.float32)
    cur_sr = sr
    all_reports = []
    for p_i in range(max(1, int(passes))):
        if progress_cb:
            progress_cb(0.1 + 0.8 * p_i / max(passes, 1), "pass %d" % (p_i + 1))
        y, cur_sr, rep = _pass(y, cur_sr, model, spike_db, max_db,
                               music_guard, progress_cb=progress_cb)
        all_reports.append(rep)
        if rep.get("n_notches", 0) == 0:
            break
    lin = integrated_loudness(x, sr)
    lout = integrated_loudness(y, cur_sr)
    comp = None
    if np.isfinite(lin) and np.isfinite(lout):
        d = lin - lout
        if 0.05 < abs(d) < 12:
            y = y * (10 ** (d / 20.0))
            comp = float(d)
    return y.astype(np.float32), cur_sr, {
        "passes": all_reports, "loudness_compensation_db": comp, "out_sr": cur_sr}


def metallic_metrics(sig: np.ndarray, sr: int, n_fft: int = 4096) -> dict:
    """Metallic-content metrics: cepstral comb, temporal stability, kurtosis."""
    x = sig[:, 0] if sig.ndim > 1 else sig
    spec = np.abs(stft(x, n_fft, n_fft // 4)[0])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    logmag = np.log(np.maximum(spec, 1e-9))
    cep = np.fft.irfft(logmag.mean(axis=0))
    lo = max(2, int(sr / 8000))
    hi = min(int(sr / 500), len(cep) // 4)
    if hi <= lo:
        hi = lo + 1
    band = np.abs(cep[lo:hi])
    comb = float(band.max())
    peak_quef = int(lo + int(band.argmax()))
    band = (freqs >= 1000) & (freqs <= 12000)
    m = spec[:, band]
    stability = float(np.mean(np.std(m, axis=0) / (np.mean(m, axis=0) + 1e-9)))
    ms = np.mean(spec, axis=0)
    ms = ms / (np.mean(ms) + 1e-9)
    kurt = float(np.mean((ms - ms.mean()) ** 4) / (np.std(ms) ** 4 + 1e-12))
    return {"comb": comb, "comb_f0": sr / peak_quef if peak_quef > 0 else 0.0,
            "stability": stability, "kurtosis": kurt}