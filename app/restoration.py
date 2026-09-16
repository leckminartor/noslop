"""RVQ artifact restoration: spectral gating driven by residual-basis analysis.

Design (verified in tests/test_core.py):
- Artifacts are estimated as the part of the spectrum that is (a) noise-like in
  texture, (b) temporally flickering, and (c) shaped like the residual basis.
  The estimate is per-frame/per-bin, so the reduction follows the signal the
  way real RVQ residual energy does.
- A Wiener-style gain with oversubtraction + temporal smoothing removes the
  artifact without pumping; a spectral-flux transient detector raises the gain
  floor on attacks so drums/percussive transients pass untouched.
- Phase is never modified (no smearing); only magnitude is gated.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .analysis import flicker_index, istft, residual_basis, stft
from .loudness import integrated_loudness


def _median_smooth_time(mag: np.ndarray, k: int = 9) -> np.ndarray:
    """Median filter across time per bin (denoise the background estimate)."""
    if mag.shape[0] < k:
        k = max(3, (mag.shape[0] // 2) * 2 + 1)
    pad = k // 2
    padded = np.pad(mag, ((pad, pad), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k, axis=0)
    return np.median(win, axis=2)[: mag.shape[0]]


def detect_transients(spec_mag: np.ndarray, hop: int, sr: int,
                      flux_thresh_mult: float = 3.0) -> np.ndarray:
    """Boolean per-frame onset mask via half-wave spectral flux (MAD-robust threshold).

    median + mult*1.4826*MAD survives sparse extreme frames where std-based
    thresholds collapse to zero detections.
    """
    diff = np.diff(spec_mag, axis=0)
    if diff.size == 0:
        return np.zeros(spec_mag.shape[0], dtype=bool)
    flux = np.maximum(diff, 0).mean(axis=1)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) * 1.4826
    thresh = med + flux_thresh_mult * mad + 1e-12
    mask = np.zeros(spec_mag.shape[0], dtype=bool)
    mask[1:] = flux > thresh
    return mask


class RestoreParams:
    """Tunables. Defaults are conservative (transparent, ~9 dB max reduction)."""

    def __init__(self,
                 n_fft: int = 4096,
                 reduction_db: float = 9.0,       # max artifact attenuation
                 oversubtract: float = 2.0,       # Wiener oversubtraction factor
                 floor: float = 0.35,             # extra gain floor clamp (0..1)
                 hf_focus: float = 2500.0,        # below this, gating is weak
                 sizzle_band: tuple = (6000.0, 12000.0),
                 comb_strength: float = 0.0,      # 0..1, off by default
                 transient_guard: float = 0.85,   # gain floor on attacks
                 smooth_time: float = 0.6,        # legacy smoothing param
                 incoh_enabled: bool = True):     # envelope-decoherence bed detection
        self.n_fft = n_fft
        self.reduction_db = reduction_db
        self.oversubtract = oversubtract
        self.floor = floor
        self.hf_focus = hf_focus
        self.sizzle_band = sizzle_band
        self.comb_strength = comb_strength
        self.transient_guard = transient_guard
        self.smooth_time = smooth_time
        self.incoh_enabled = incoh_enabled


# Presets tuned on the strength sweep (tests: sweep_strength.py):
#   gentle  : SNR +0.2 dB, minimal HF dulling - for mastering-grade material
#   standard: balanced (SNR +0.3..0.4 dB on CELT-coded material)
#   strong  : maximum artifact reduction, audibly duller hats
PRESETS = {
    "gentle":   dict(oversubtract=1.0, floor=0.50, reduction_db=6),
    "standard": dict(oversubtract=1.3, floor=0.45, reduction_db=8),
    "strong":   dict(oversubtract=2.0, floor=0.25, reduction_db=12),
}


def params_from_preset(preset: str = "standard", **overrides) -> "RestoreParams":
    """Build RestoreParams from a named preset with optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}', choose from {list(PRESETS)}")
    kw = dict(PRESETS[preset])
    kw.update(overrides)
    return RestoreParams(**kw)


def _artifact_estimate(mag: np.ndarray, sr: int, hop: int, params: RestoreParams,
                       basis: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-frame/per-bin artifact magnitude estimate + analysis dict.

    v2: artifact = positive deviation from the time-median background, gated by
    fine local spectral flatness (noise texture vs tonal/harmonic), residual-basis
    shape match and temporal flicker.
    """
    n_frames, n_bins = mag.shape
    freqs = np.fft.rfftfreq(params.n_fft, d=1.0 / sr)

    # background = time-median; the deviation carries flickering/noisy content
    bg = _median_smooth_time(mag, k=9)
    dev_pos = np.maximum(mag - bg, 0.0)

    # ---- fine local flatness: texture of each bin neighborhood (noise vs tonal)
    k_local = 15
    padded = np.pad(mag, ((0, 0), (k_local // 2, k_local - 1 - k_local // 2)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k_local, axis=1)
    win = win[:, :n_bins, :]
    gm = np.exp(np.mean(np.log(np.maximum(win, 1e-12)), axis=2))
    am = np.mean(win, axis=2)
    flat_local = np.clip(gm / np.maximum(am, 1e-12), 1e-6, 1.0)
    # noise texture ~ flat_local 0.3..1; tonal ~ <0.05
    texture = np.clip((flat_local - 0.06) / 0.44, 0.0, 1.0)

    # ---- residual-basis shape match per frame (cosine sim of deviation shape)
    dev_db = 20.0 * np.log10(np.maximum(mag, 1e-10)) - 20.0 * np.log10(np.maximum(bg, 1e-10))
    vn = np.linalg.norm(dev_db, axis=1, keepdims=True) + 1e-9
    An = np.linalg.norm(basis, axis=1) + 1e-9
    cos = (dev_db @ basis.T) / (vn * An[None, :])        # (frames, atoms)
    match = np.clip(cos.max(axis=1), 0.0, 1.0)           # (frames,)

    # ---- global gates
    from .analysis import band_flatness_db
    flat_frames = band_flatness_db(mag, params.n_fft, sr, 500.0, sr / 2 - 500)
    noise_frame = np.clip((flat_frames + 60.0) / 60.0, 0.0, 1.0)

    flick = flicker_index(mag, params.n_fft, sr, hop,
                          params.sizzle_band[0],
                          min(params.sizzle_band[1], sr / 2 - 200))
    flicker_gate = float(np.clip(flick / 1.5, 0.0, 1.0))

    # ---- envelope incoherence: HF content whose envelope does not follow the
    # musical envelope (octave below) is codec band-copy noise (mp3/aac sizzle).
    incoh = (_envelope_incoherence(mag, params, sr, hop, freqs)[0]
             if params.incoh_enabled else np.zeros(n_bins))

    art = dev_pos.copy()
    art *= texture ** 1.5                                  # noise texture only
    art *= (0.6 + 0.4 * match[:, None])                    # basis shape match (floored)
    art *= noise_frame[:, None] * (0.5 + 0.5 * flicker_gate)
    # coherence protection: musical HF (hats) follows the envelope -> shield it hard
    art *= (0.15 + 0.85 * (1.0 - incoh))[None, :]
    # HF weighting: artifacts concentrate above hf_focus
    hf_w = np.clip((freqs - params.hf_focus) / max(2000.0, sr / 8), 0.0, 1.0)
    art *= (0.15 + 0.85 * hf_w)[None, :]

    # incoherent background bed: stable HF noise that ignores the music envelope
    if params.incoh_enabled:
        bed = bg * incoh[None, :] * texture * hf_w[None, :]
        art = np.maximum(art, bed)
    report = {
        "flatness_frame_mean_db": float(np.mean(flat_frames)) if len(flat_frames) else -60.0,
        "flicker": flick,
        "basis_match_mean": float(np.mean(match)),
        "artifact_energy_ratio": float(np.sum(art ** 2) / max(np.sum(mag ** 2), 1e-20)),
    }
    return art, report


def _envelope_incoherence(mag: np.ndarray, params: RestoreParams, sr: int, hop: int,
                          freqs: np.ndarray, window_s: float = 1.0) -> np.ndarray:
    """Per-bin fraction of envelope energy that is INCOHERENT with the octave-below
    reference envelope. ~0 where HF follows the music, ~1 for band-copy noise beds."""
    n_frames, n_bins = mag.shape
    # smooth envelopes over ~60 ms
    k_env = max(3, int(round(0.06 * sr / hop)) | 1)
    env = _median_smooth_time(mag, k=k_env)
    W = max(8, int(window_s * sr / hop) | 1)  # correlation window
    if n_frames < W:
        W = n_frames | 1
    incoh = np.zeros(n_bins)
    lo_max = int(params.hf_focus * 2.0 / (sr / n_bins / 2) ) if False else None
    # bin index of hf_focus
    f_lo_bin = int(params.hf_focus / (sr / 2) * (n_bins - 1))
    for b in range(f_lo_bin, n_bins):
        b_ref = b // 2
        if b_ref < 4:
            incoh[b] = 0.0
            continue
        # reference envelope: average over a small neighborhood below b/2
        lo = max(2, b // 2 - b // 8)
        hi = max(lo + 2, b // 2 + b // 8)
        ref = env[:, lo:hi].mean(axis=1)
        tgt = env[:, b]
        # sliding correlation over W frames
        cs = np.cumsum(np.concatenate([[0.0], tgt * ref]))
        cs_t = np.cumsum(np.concatenate([[0.0], tgt]))
        cs_r = np.cumsum(np.concatenate([[0.0], ref]))
        cs_tt = np.cumsum(np.concatenate([[0.0], tgt ** 2]))
        cs_rr = np.cumsum(np.concatenate([[0.0], ref ** 2]))
        half = W // 2
        idx0 = np.arange(n_frames)
        lo_i = np.maximum(0, idx0 - half)
        hi_i = np.minimum(n_frames, idx0 + half + 1)
        n_ = (hi_i - lo_i).astype(float)
        s_tr = cs[hi_i] - cs[lo_i]
        s_t = cs_t[hi_i] - cs_t[lo_i]
        s_r = cs_r[hi_i] - cs_r[lo_i]
        s_tt = cs_tt[hi_i] - cs_tt[lo_i]
        s_rr = cs_rr[hi_i] - cs_rr[lo_i]
        cov = s_tr / n_ - (s_t / n_) * (s_r / n_)
        vt = s_tt / n_ - (s_t / n_) ** 2
        vr = s_rr / n_ - (s_r / n_) ** 2
        corr = cov / np.sqrt(np.maximum(vt * vr, 1e-24))
        incoh[b] = np.clip(1.0 - np.nanmean(corr), 0.0, 1.0)
    # temporal smoothing across bins for stability (keep full length)
    k = 5
    kernel = np.ones(k) / k
    incoh = np.convolve(np.pad(incoh, (2, 2), mode="edge"), kernel, mode="same")
    return incoh[None, :n_bins]


def _one_pole_smooth_per_bin(gain: np.ndarray, alpha: float) -> np.ndarray:
    """Temporal one-pole smoothing per bin."""
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    return lfilter(b, a, gain, axis=0)


def _gate_smooth(gain: np.ndarray, attack: float = 0.25, release: float = 0.7) -> np.ndarray:
    """Asymmetric smoothing, vectorized: env = min(fast, slow) followers.
    Falls fast (attack on dips), rises slowly (release) -> flicker stays gated,
    musical continuity preserved."""
    fast = lfilter([1.0 - attack], [1.0, -attack], gain, axis=0)
    slow = lfilter([1.0 - release], [1.0, -release], gain, axis=0)
    return np.minimum(fast, slow)


def restore(audio: np.ndarray, sr: int, params: RestoreParams | None = None,
           seed: int = 0) -> tuple[np.ndarray, dict]:
    """Main entry: (audio (n,) or (n,ch), sr) -> (clean audio, report dict)."""
    p = params or RestoreParams()
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    basis = residual_basis(p.n_fft, sr, seed=seed)
    hop = p.n_fft // 4

    outs, reports, onset_counts = [], [], []
    for c in range(ch):
        spec, win, hop_, pad = stft(x[:, c], p.n_fft, hop)
        mag = np.abs(spec)
        phase = np.exp(1j * np.angle(spec))
        art, rep = _artifact_estimate(mag, sr, hop, p, basis)

        # ---- transient detection FIRST (musical attacks are not artifacts)
        onsets = detect_transients(mag, hop, sr)
        if onsets.any() and p.transient_guard > 0:
            lift = np.zeros(len(mag))
            lift[onsets] = 1.0
            lift = _one_pole_smooth_per_bin(lift[:, None], 0.35)[:, 0]
            lift = np.maximum(lift, 0.0)
            # shield the deviation estimate on/around transient frames
            art = art * (1.0 - 0.8 * lift[:, None])

        # ---- Wiener-style gain with floor
        psig = mag ** 2
        part = (p.oversubtract * art) ** 2
        g = psig / (psig + part + 1e-24)
        g_min = 10 ** (-p.reduction_db / 20.0)
        g_min = max(g_min, min(p.floor, 1.0))
        g = np.maximum(g, g_min)

        # ---- transient guard: raise the floor on onset frames
        if onsets.any() and p.transient_guard > 0:
            g = np.maximum(g, (lift * p.transient_guard)[:, None])

        # ---- temporal smoothing of the gain: fast attack, slow release
        g = _gate_smooth(g, attack=0.25, release=0.7)

        spec_clean = (mag * g) * phase
        y = istft(spec_clean, p.n_fft, hop, win, len(x[:, c]), pad)

        if p.comb_strength > 0.05:
            w = np.asarray(rep["basis_match_mean"])
            if w > 0.3:
                y = _apply_comb_notches(y, sr, strength=float(np.clip(w, 0, 1)) * p.comb_strength)

        outs.append(y)
        onset_counts.append(int(onsets.sum()))
        reports.append(rep)

    y = np.stack(outs, axis=1) if ch > 1 else outs[0]
    report = {"channels": ch, "onset_frames": onset_counts}
    for k in reports[0]:
        vals = [r[k] for r in reports]
        report[k] = vals if ch > 1 else vals[0]
    report["gain_min_db"] = float(20 * np.log10(max(g_min, 1e-6)))

    # loudness compensation: processing must not simply turn things down
    lin = integrated_loudness(x, sr)
    lout = integrated_loudness(y, sr)
    if np.isfinite(lin) and np.isfinite(lout):
        diff = lin - lout
        if 0.05 < abs(diff) < 12:
            y = y * (10 ** (diff / 20.0))
            report["loudness_compensation_db"] = float(diff)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y / peak * 0.999
        report["peak_normalized"] = True
    return y.astype(np.float32), report


def _apply_comb_notches(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Conservative feed-forward comb canceler at the delay that best cancels."""
    delays = [int(round(sr * d)) for d in (0.11e-3, 0.17e-3, 0.23e-3, 0.31e-3)]
    best, best_e = None, None
    for d in delays:
        if d <= 0 or d >= len(y) - 1:
            continue
        e = float(np.sum((y[d:] - y[:-d]) ** 2))
        if best_e is None or e < best_e:
            best_e, best = e, d
    if best is None:
        return y
    a = 0.35 * float(np.clip(strength, 0.0, 1.0))
    if a <= 0.01:
        return y
    yd = np.concatenate([np.zeros(best), y[:-best]])
    return (y - a * yd) / (1.0 + a * a)