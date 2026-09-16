"""STFT helpers + spectral artifact analysis for RVQ/codec-degraded audio.

Detectors (all verifiable on synthetic + real codec material):
- residual-basis projection energy  -> artifact noise profile
- band spectral flatness            -> "digital sizzle"/metallic highs
- temporal flicker                  -> flickering sparse high bins (watery/metallic)
- phase increment variance          -> codec smearing on low-energy bins
"""
from __future__ import annotations

import numpy as np

from .loudness import integrated_loudness, true_peak_dbtp


def frame_signal(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(n_frames, n_fft) framed copy."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < n_fft:
        x = np.concatenate([x, np.zeros(n_fft - n)])
        n = n_fft
    n_frames = 1 + (n - n_fft) // hop
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=(n_frames, n_fft),
                                           strides=strides).copy()


def stft(x: np.ndarray, n_fft: int = 2048, hop: int | None = None,
         win: np.ndarray | None = None):
    """Returns (spec, win, hop, pad). Signal is zero-padded by n_fft on both sides,
    so every original sample is covered by complete frames (exact COLA interior)."""
    hop = hop or n_fft // 4
    w = win if win is not None else np.hanning(n_fft + 1)[: n_fft]  # periodic hann
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    n_frames = 1 + max(0, (n - 1)) // hop
    total = n_fft + n_frames * hop + n_fft
    xp = np.zeros(total)
    xp[n_fft:n_fft + n] = x
    frames = frame_signal(xp, n_fft, hop)
    spec = np.fft.rfft(frames * w, axis=1)
    return spec, w, hop, n_fft


def _ola_window_sum(n_frames: int, n_fft: int, hop: int, win: np.ndarray,
                    out_len: int) -> np.ndarray:
    wsum = np.zeros(out_len + n_fft)
    for i in range(n_frames):
        wsum[i * hop:i * hop + n_fft] += win ** 2
    return wsum[:out_len]


def istft(spec: np.ndarray, n_fft: int, hop: int, win: np.ndarray,
          out_len: int, pad: int = 0,
          win_for_sum: np.ndarray | None = None) -> np.ndarray:
    """Overlap-add with hann^2 COLA normalization (exact for hop = N/4 interior);
    slice [pad : pad+out_len] undoes stft's start padding."""
    frames = np.fft.irfft(spec, n=n_fft, axis=1)
    n_frames = len(frames)
    total = (n_frames - 1) * hop + n_fft
    out = np.zeros(total)
    wsum = np.zeros(total)
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i] * win
        wsum[s:s + n_fft] += win ** 2
    y = out / np.maximum(wsum, 1e-8)
    return y[pad:pad + out_len]


def spectrogram_db(spec: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))


def band_flatness_db(spec_mag: np.ndarray, n_fft: int, sr: float,
                     f_lo: float, f_hi: float) -> np.ndarray:
    """Per-frame spectral flatness (dB, 0=tonal..-60=noise-like) within a band."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    band = (freqs >= f_lo) & (freqs < f_hi)
    if band.sum() < 4:
        return np.zeros(spec_mag.shape[0])
    m = np.maximum(spec_mag[:, band], 1e-12)
    gm = np.exp(np.mean(np.log(m), axis=1))
    am = np.mean(m, axis=1)
    flat = np.maximum(gm / np.maximum(am, 1e-12), 1e-6)
    return 20.0 * np.log10(flat)


def flicker_index(spec_mag: np.ndarray, n_fft: int, sr: float, hop: int,
                  f_lo: float, f_hi: float) -> float:
    """Mean temporal variability of band magnitude, normalized to a 10 ms scale."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    band = (freqs >= f_lo) & (freqs < f_hi)
    if band.sum() < 2 or spec_mag.shape[0] < 4:
        return 0.0
    m = spec_mag[:, band]
    dm = np.abs(np.diff(m, axis=0))
    denom = np.mean(m) + 1e-10
    return float(np.mean(dm) / denom * hop / (sr * 0.01))


def phase_incr_var(spec: np.ndarray) -> np.ndarray:
    """Per-bin variance of phase increments across frames (rad^2)."""
    ph = np.angle(spec)
    d = np.diff(ph, axis=0)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return np.var(d, axis=0)


# ---------------------------------------------------------------- residual basis

def residual_basis(n_fft: int, sr: int, n_atoms: int = 12, seed: int = 0) -> np.ndarray:
    """Synthetic RVQ-residual spectral basis (n_atoms, n_bins), unit-RMS PSD shapes.

    Stage-like atoms: broadband noise -> HF tilt -> low tilt -> sibilant band
    -> metallic combs -> random smooth shapes.
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    nyq = sr / 2
    atoms = []

    def _norm(a):
        m = np.sqrt(np.mean(a ** 2))
        return a / max(m, 1e-9)

    # 1) broadband
    atoms.append(_norm(np.ones_like(freqs)))
    # 2) HF-weighted tilt (late-stage residuals live high)
    tilt = (freqs / nyq) ** 1.5 + 0.02
    atoms.append(_norm(tilt))
    # 3) low tilt (early-stage errors live lower)
    tilt2 = (freqs / nyq) ** 0.4 + 0.05
    atoms.append(_norm(tilt2))
    # 4) sibilant band 5-9 kHz
    band = np.exp(-0.5 * ((freqs - 6800.0) / 2200.0) ** 2) + 0.01
    atoms.append(_norm(band))
    # 5) metallic combs
    for k in (7, 11, 15):
        comb = np.abs(np.sin(np.pi * freqs * k / nyq)) ** 2 * (freqs / nyq + 0.1) + 0.005
        atoms.append(_norm(comb))
    # 6) random smooth shapes
    while len(atoms) < n_atoms:
        knots = rng.uniform(0.2, 1.0, size=4)
        shape = np.abs(np.interp(freqs / nyq, np.linspace(0, 1, 4), knots)) + 0.02
        atoms.append(_norm(shape))
    A = np.stack(atoms[:n_atoms])
    A[:, freqs < 80] = 0.0
    for i in range(A.shape[0]):
        n = np.sqrt(np.mean(A[i] ** 2))
        A[i] /= max(n, 1e-9)
    return A


def project_energy(log_mag_db: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project log-magnitude deviation onto residual basis -> (n_frames, n_atoms)."""
    return log_mag_db @ basis.T


def analyze_health(x: np.ndarray, sr: int, n_fft: int = 4096) -> dict:
    """Codec-health metrics (works standalone, no server needed)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    mono = x[:, 0]
    spec = np.abs(stft(mono, n_fft, n_fft // 4)[0])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    hf = (freqs >= 6000) & (freqs <= 16000)
    from .restoration import _envelope_incoherence, RestoreParams
    incoh = _envelope_incoherence(spec, RestoreParams(n_fft=n_fft), sr, n_fft // 4, freqs)[0]
    hf_incoh = float(np.mean(incoh[hf])) if hf.any() else 0.0
    lufs = integrated_loudness(x, sr)
    tp = true_peak_dbtp(x, sr)
    flags = []
    if hf_incoh > 0.45:
        flags.append("high_hf_sizzle")
    if hf_incoh > 0.65:
        flags.append("band_copy_artifacts")
    return {
        "hf_incoherence": round(hf_incoh, 4),
        "lufs": round(float(lufs), 2) if np.isfinite(lufs) else None,
        "true_peak_dbtp": round(float(tp), 2),
        "artifact_flags": flags,
    }