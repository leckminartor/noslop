"""Metallic/comb de-ringing: cepstrum-tracked adaptive notch system.

Metallic/blechy artifacts are spectrum-periodic (combs) created by codec MDCT
aliasing and RVQ codebook structure. They are COHERENT (follow the music
envelope) so envelope-incoherence gating cannot see them. This module tracks
the dominant comb period per frame-block via the real cepstrum of the
log-spectrum and applies adaptive, transient-protected notches.

Conservative by design: notches only fire where comb evidence is strong AND
stable across the analysis window, and they are shallow (max ~4.5 dB per
notch, 3 harmonics max) so they never maul real musical combs (flanger etc.).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .analysis import stft, istft
from .loudness import integrated_loudness


def _cepstral_comb_track(mag: np.ndarray, sr: int, n_fft: int,
                         f_lo: float = 500.0, f_hi: float = 8000.0) -> tuple[float, float, np.ndarray]:
    """Track dominant comb ripple over time. Returns (strength, f0, strength_time).

    strength: mean peak of |real cepstrum| in the ripple-quefrency band
    f0:       dominant ripple frequency (Hz) — comb period = sr/f0 samples
    """
    logmag = np.log(np.maximum(mag, 1e-9))
    mean_spec = logmag.mean(axis=0)
    cep = np.fft.irfft(mean_spec)
    lo = max(2, int(sr / f_hi))
    hi = min(int(sr / f_lo), len(cep) // 4)
    if hi <= lo:
        return 0.0, 0.0, np.zeros(mag.shape[0])
    band = np.abs(cep[lo:hi])
    peak = float(band.max())
    f0 = sr / (lo + int(band.argmax()))
    # per-frame strength: project each frame's log-mag deviation onto the comb
    # shape at f0 -> time curve of "how metallic is this frame"
    k = int(round(sr / f0))
    if k <= 2 or k >= mag.shape[1] - 2:
        return peak, f0, np.zeros(mag.shape[0])
    # comb template: cos ripple across bins with period k bins
    bins = np.arange(mag.shape[1])
    template = np.cos(2 * np.pi * bins / k)
    # per-frame ripple strength = |corr(logmag frame, template)|
    lm = logmag - logmag.mean(axis=1, keepdims=True)
    t = template - template.mean()
    num = (lm * t[None, :]).sum(axis=1)
    den = np.sqrt((lm ** 2).sum(axis=1)) * np.sqrt((t ** 2).sum()) + 1e-12
    strength_time = np.abs(num / den)
    return peak, f0, strength_time


def _notch_bank(sr: int, f0: float, n_fft: int, n_harmonics: int = 3,
                depth_db: float = 6.0, width_oct: float = 0.25):
    """Per-bin gain mask for a comb notch bank at f0, 2*f0, 3*f0."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    gain = np.ones_like(freqs)
    for h in range(1, n_harmonics + 1):
        fc = f0 * h
        if fc >= sr / 2 - 1000:
            break
        # gaussian notch in log-frequency
        sigma = fc * (width_oct * np.log(2) / 2)
        g = np.exp(-0.5 * ((freqs - fc) / max(sigma, 20.0)) ** 2)
        gain *= 1.0 - (1.0 - 10 ** (-depth_db / 20.0)) * g
    return gain[None, :]  # (1, bins)


def _stable_mask(strength_time: np.ndarray, thresh: float = 0.0,
                 attack: float = 0.3, release: float = 0.8) -> np.ndarray:
    """Frames where the comb is strong AND stable -> apply notches there."""
    smooth = lfilter([1.0 - attack], [1.0, -attack], strength_time)
    smooth = np.minimum(smooth, lfilter([1.0 - release], [1.0, -release], strength_time))
    med = np.median(smooth) + 1e-12
    rel = smooth / max(med, 1e-12)
    return np.clip((rel - 1.0) / max(np.max(rel) - 1.0, 1e-9), 0.0, 1.0)


def dering(audio: np.ndarray, sr: int,
           depth_db: float = 6.0,
           sensitivity: float = 0.5,
           f_lo: float = 500.0, f_hi: float = 8000.0,
           n_harmonics: int = 3,
           n_fft: int = 4096,
           min_comb_strength: float = 0.12) -> tuple[np.ndarray, dict]:
    """Adaptive cepstrum-tracked comb de-ringing.

    sensitivity 0..1: higher = more aggressive (more frames treated, deeper).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    hop = n_fft // 4

    outs, info = [], []
    for c in range(x.shape[1]):
        spec, win, hop_, pad = stft(x[:, c], n_fft, hop)
        mag = np.abs(spec)
        phase = np.exp(1j * np.angle(spec))
        peak, f0, strength_time = _cepstral_comb_track(mag, sr, n_fft, f_lo, f_hi)
        gate = _stable_mask(strength_time * sensitivity / max(peak, 1e-9) * 3.0,
                            attack=0.3, release=0.8)
        gate = np.clip(gate, 0, 1)

        if peak < min_comb_strength or f0 <= 0:
            outs.append(x[:, c])
            info.append({"comb": peak, "f0": f0, "applied": 0.0})
            continue

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        base = _notch_bank(sr, f0, n_fft, n_harmonics, depth_db)[0]
        # depth modulated by per-frame gate
        depth = 1.0 - base  # attenuation per bin
        g = (1.0 - depth[None, :] * gate[:, None])  # (frames, bins)
        # transient protection: no notching on attacks
        from .restoration import detect_transients
        onsets = detect_transients(mag, hop, sr)
        if onsets.any():
            lift = np.zeros(len(g))
            lift[onsets] = 1.0
            lift = lfilter([0.65], [1, -0.35], lift)
            g = np.maximum(g, (np.clip(lift, 0, 1)[:, None]))  # floor gain to 1 on attacks

        # temporal smoothing of gate (avoid pumping)
        g = lfilter([0.3], [1.0, -0.7], g, axis=0)
        spec_clean = mag * g * phase
        y = istft(spec_clean, n_fft, hop, win, len(x[:, c]), pad)
        outs.append(y)
        info.append({"comb": peak, "f0": f0,
                     "applied": float(np.mean(gate)),
                     "mean_gain_db": float(20 * np.log10(max(g.min(), 1e-6)))})

    y = np.stack(outs, axis=1) if x.shape[1] > 1 else outs[0]
    rep = info[0] if x.shape[1] == 1 else {k: [i[k] for i in info] for k in info[0]}
    # loudness compensation
    lin = integrated_loudness(x, sr)
    lout = integrated_loudness(y, sr)
    if np.isfinite(lin) and np.isfinite(lout):
        d = lin - lout
        if 0.05 < abs(d) < 12:
            y = y * (10 ** (d / 20.0))
            rep["loudness_compensation_db"] = float(d)
    return y.astype(np.float32), rep
