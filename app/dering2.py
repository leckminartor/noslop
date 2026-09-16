"""Metallic de-ringing v2: stable-resonance suppression + adaptive comb notches.

v1 lesson (diag_metal.py): cepstrum-tracked combs barely move the metric because
codec "metallic" sound is often NOT a clean comb but a set of NARROW, TIME-STABLE
resonances at arbitrary frequencies. New approach:

1. Estimate long-term average spectrum; find narrow peaks that are much louder
   than a smoothed version of themselves (spectral spikiness) AND stable in time
   (low temporal CV). Those are ringing bins.
2. NOT music: musical bins vary in time (CV high) — hats etc. are excluded.
3. Suppress ringing bins with a Wiener-style gain toward the local median
   spectrum (the "true" noise floor of the mix), transient-protected.
4. Additionally drive the existing comb-notch bank from the cepstrum when a
   strong periodic ripple is present (combed aliasing), still gated by stability.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .analysis import stft, istft
from .loudness import integrated_loudness
from .restoration import detect_transients


def _ringing_bins(mag: np.ndarray, n_fft: int, sr: int, hop: int,
                  smooth_bins: int = 31,
                  stab_max: float = 0.9) -> tuple[np.ndarray, dict]:
    """Per-bin weight 0..1: 1 = strongly ringing (stable + spiky)."""
    n_frames, n_bins = mag.shape
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # long-term average + smoothed background (median across time, then freq-smooth)
    avg = np.mean(mag, axis=0)
    k = smooth_bins | 1
    pad = k // 2
    avg_s = np.convolve(np.pad(avg, (pad, pad), mode="edge"), np.ones(k) / k, mode="same")[: len(avg)]

    # spikiness: avg above smoothed average (log ratio), only where meaningful
    spike_db = 20.0 * np.log10(np.maximum(avg, 1e-9)) - 20.0 * np.log10(np.maximum(avg_s, 1e-9))
    spike_w = np.clip(spike_db / 12.0, 0.0, 1.0)  # +12 dB spike = full weight

    # temporal stability: ringing bins are STABLE (low CV); musical bins vary
    cv = np.std(mag, axis=0) / (np.mean(mag, axis=0) + 1e-9)
    # music: cv ~ 1.2+; ringing: cv < 0.5. map: 0 at cv=0.9, 1 at cv=0.3
    cv = np.clip(np.std(mag, axis=0) / (np.mean(mag, axis=0) + 1e-9), 0, 4)
    stab_w = np.clip((0.9 - cv) / 0.6, 0.0, 1.0)

    # frequency focus: 2 kHz .. sr/2-2k (ringing lives high)
    f_w = np.clip((freqs - 2000.0) / 1500.0, 0.0, 1.0)
    w = spike_w * stab_w * f_w
    report = {"n_ringing_bins": int((w > 0.5).sum()), "total_bins": n_bins}
    return w[None, :], report


def dering_v2(audio: np.ndarray, sr: int,
              depth_db: float = 9.0,
              sensitivity: float = 0.5,
              n_fft: int = 4096,
              transient_guard: float = 0.6) -> tuple[np.ndarray, dict]:
    """Suppress time-stable narrow resonances (metallic ringing) in the mix."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    hop = n_fft // 4
    lufs_in = integrated_loudness(x, sr)

    outs, reports = [], []
    for c in range(x.shape[1]):
        spec, win, hop_, pad = stft(x[:, c], n_fft, hop)
        mag = np.abs(spec)
        phase = np.exp(1j * np.angle(spec))
        w_ring, rrep = _ringing_bins(mag, n_fft, sr, hop)

        # background: temporal median (the musical part lives here)
        bg = _median_smooth_time(mag, k=9)
        # ringing estimate = mag * ring weight (the STABLE excess over bg)
        excess = np.maximum(mag - bg, 0.0)
        art = excess * w_ring
        # scale by sensitivity
        art *= float(np.clip(sensitivity, 0, 2))

        psig = mag ** 2
        part = (1.8 * art) ** 2
        g = psig / (psig + part + 1e-24)
        g_min = 10 ** (-depth_db / 20.0)
        g = np.maximum(g, g_min)

        # transient guard
        onsets = detect_transients(mag, hop, sr)
        if onsets.any() and transient_guard > 0:
            lift = np.zeros(len(g))
            lift[onsets] = 1.0
            lift = lfilter([0.65], [1, -0.35], lift)
            g = np.maximum(g, (np.clip(lift, 0, 1) * transient_guard)[:, None])

        # asymmetric smoothing (fast attack, slow release)
        fast = lfilter([0.75], [1.0, -0.25], g, axis=0)
        slow = lfilter([0.3], [1.0, -0.7], g, axis=0)
        g = np.minimum(fast, slow)

        spec_clean = mag * g * phase
        y = istft(spec_clean, n_fft, hop, win, len(x[:, c]), pad)
        outs.append(y)
        reports.append({"gain_min_db": float(20 * np.log10(max(g.min(), 1e-6))),
                        "ringing_bins": rrep["n_ringing_bins"]})

    y = np.stack(outs, axis=1) if x.shape[1] > 1 else outs[0]
    rep = reports[0] if x.shape[1] == 1 else {k: [r[k] for r in reports] for k in reports[0]}
    lufs_out = integrated_loudness(y, sr)
    if np.isfinite(lufs_in) and np.isfinite(lufs_out):
        d = lufs_in - lufs_out
        if 0.05 < abs(d) < 12:
            y = y * (10 ** (d / 20.0))
            rep["loudness_compensation_db"] = float(d)
    return y.astype(np.float32), rep


def _median_smooth_time(mag: np.ndarray, k: int = 9) -> np.ndarray:
    if mag.shape[0] < k:
        k = max(3, (mag.shape[0] // 2) * 2 + 1)
    pad = k // 2
    padded = np.pad(mag, ((pad, pad), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k, axis=0)
    return np.median(win, axis=2)[: mag.shape[0]]