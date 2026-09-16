"""ITU-R BS.1770-4 integrated loudness (K-weighted, gated) implemented from scratch.

No pyloudnorm dependency: analog-prototype biquads + bilinear transform, so any sr works.
Validation: 997 Hz stereo sine at -20 dBFS must measure -23.0 LUFS (+/- 0.1).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, tf2sos, resample_poly


def _high_shelf_coeffs(fs: float):
    """Analog-matched high-shelf (+4 dB @ ~1.5 kHz), bilinear transform."""
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = np.tan(np.pi * f0 / fs)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.49966677
    a0 = 1 + K / Q + K ** 2
    b = [(Vh + Vb * K / Q + K ** 2) / a0,
         2 * (K ** 2 - Vh) / a0,
         (Vh - Vb * K / Q + K ** 2) / a0]
    a = [1.0, 2 * (K ** 2 - 1) / a0, (1 - K / Q + K ** 2) / a0]
    return np.asarray(b), np.asarray(a)


def _rlb_highpass_coeffs(fs: float):
    """RLB high-pass at 38.135 Hz, Q=0.5003, bilinear."""
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = np.tan(np.pi * f0 / fs)
    a0 = 1 + K / Q + K ** 2
    b = np.array([1.0, -2.0, 1.0]) / a0
    a = np.array([1.0, 2 * (K ** 2 - 1) / a0, (1 - K / Q + K ** 2) / a0])
    return b, a


def k_weighting_sos(fs: float):
    b1, a1 = _high_shelf_coeffs(fs)
    b2, a2 = _rlb_highpass_coeffs(fs)
    return np.vstack([tf2sos(b1, a1), tf2sos(b2, a2)])


def _mean_square_blocks(x: np.ndarray, fs: float):
    """400 ms blocks, 100 ms hop -> per-block mean squares per channel. (n_blocks, n_ch)"""
    block = int(round(0.4 * fs))
    hop = int(round(0.1 * fs))
    n = x.shape[0]
    if n < block:
        return None
    n_blocks = 1 + (n - block) // hop
    cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x ** 2]), axis=0)
    idx0 = np.arange(n_blocks) * hop
    sums = cs[idx0 + block] - cs[idx0]
    return sums / block


def integrated_loudness(audio: np.ndarray, fs: float) -> float:
    """Gated integrated loudness in LUFS. audio: (n,) or (n, ch), float, +-1 full scale."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 2 and x.shape[0] < x.shape[1] and x.shape[1] <= 5:
        x = x.T  # tolerate (ch, n)
    if x.ndim != 2:
        raise ValueError("audio must be (n,) or (n, ch)")
    sos = k_weighting_sos(fs)
    y = sosfilt(sos, x, axis=0)
    ms = _mean_square_blocks(y, fs)
    if ms is None:
        return -np.inf
    # channel weights: L/R/C = 1.0, surrounds 1.41 (power 1.5)
    w = np.ones(ms.shape[1])
    if ms.shape[1] >= 4:
        w[3:] = 1.41
    z = ms @ w
    loud = -0.691 + 10.0 * np.log10(np.maximum(z, 1e-30))
    keep = loud > -70.0  # absolute gate
    if not keep.any():
        return -np.inf
    mean_abs_loud = -0.691 + 10.0 * np.log10(max(np.mean(z[keep]), 1e-30))
    rel_thresh = mean_abs_loud - 10.0  # relative gate
    keep2 = keep & (loud > rel_thresh)
    if not keep2.any():
        keep2 = keep
    total = np.mean(z[keep2])
    return float(-0.691 + 10.0 * np.log10(max(total, 1e-30)))


def true_peak_dbtp(audio: np.ndarray, fs: float, oversample: int = 4) -> float:
    """Approximate true peak via 4x windowed-sinc interpolation (BS.1770-style)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    g = max(1, int(oversample))
    if g == 1:
        peak = np.max(np.abs(x))
        return float(20 * np.log10(max(peak, 1e-12)))
    from scipy.signal import upfirdn
    from scipy.special import i0

    def _kernel(g: int, Tw: float = 18.0, beta: float = 12.0) -> np.ndarray:
        """sinc kernel, Kaiser window applied on the CONTINUOUS time axis,
        per-branch DC normalization (exact amplitude at every sub-phase)."""
        n_t = int(Tw * g)
        tt = np.arange(-n_t, n_t + 1) / g
        h = np.sinc(tt)
        u = np.clip(tt / Tw, -1, 1)
        w = np.i0(beta * np.sqrt(np.maximum(0, 1 - u ** 2))) / np.i0(beta)
        h = h * w
        idx = np.arange(len(h))
        for p in range(g):
            m = ((idx - n_t) % g) == p
            s = h[m].sum()
            if s > 1e-9:
                h[m] /= s
        return h

    y = upfirdn(_kernel(g), x, up=g, down=1, axis=0)
    peak = np.max(np.abs(y))
    return float(20 * np.log10(max(peak, 1e-12)))


def match_loudness(audio: np.ndarray, fs: float, target_lufs: float,
                   max_gain_db: float = 12.0, ceiling_dbtp: float = -0.3):
    """Return (gain_applied_db, new_audio) applying gain to hit target LUFS with ceiling guard."""
    x = np.asarray(audio, dtype=np.float32)
    cur = integrated_loudness(x, fs)
    if not np.isfinite(cur):
        return 0.0, x
    gain = float(np.clip(target_lufs - cur, -max_gain_db, max_gain_db))
    y = x * (10.0 ** (gain / 20.0))
    tp = true_peak_dbtp(y, fs)
    if tp > ceiling_dbtp:
        adj = ceiling_dbtp - tp
        y = y * (10.0 ** (adj / 20.0))
        gain += adj
    return float(gain), y.astype(np.float32)