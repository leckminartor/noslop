"""AI reverb/echo tail ('bright metallic hang'): detector + suppressor.

Signature of the artifact: after a musical onset, HF energy decays much slower
than the mix's own plausible decay. Measured on the bright-tail benchmark:
wet tail -25 dB/s / tail-peak -5.0 dB vs. dry -80 dB/s / -11.9 dB.

Self-calibrating approach (no magic floors):
- anchor band 1-2 kHz defines, per onset-segment, the decay behavior the mix
  itself allows ('the room');
- a causal limiter enforces a minimum HF decay rate of anchor_slope - margin
  (absolute backstop -35 dB/s): the allowed HF floor falls from each onset
  peak at that rate; wherever the observed tail hangs above the falling floor,
  gain < 1 pulls it down;
- asymmetric gain smoothing + onset guard keep attacks intact; loudness
  compensated per pass (loudness-neutral overall).

Iterating passes converges toward the dry reference (bench: -5.0 -> -9.9 dB
tail/peak after 3 passes vs. dry -11.9).

Verified in tests/bench_derverb.py.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .analysis import stft, istft
from .loudness import integrated_loudness
from .restoration import detect_transients


def _anchor_slope_db_s(seg_mag: np.ndarray, anchor_mask: np.ndarray,
                       t_frame: float) -> float:
    """Decay slope (dB/s) of the anchor band within one onset segment."""
    env = np.sqrt(np.mean(seg_mag[:, anchor_mask] ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(env, 1e-9))
    n_fit = min(len(db), max(4, int(0.25 / t_frame)))
    k = np.arange(n_fit)
    return float(np.polyfit(k, db[:n_fit], 1)[0] / t_frame)


def tail_metrics(sig: np.ndarray, sr: int, n_fft: int = 4096) -> dict:
    """Post-onset HF decay slope + tail/peak ratio - the metallic-tail indicators.

    Dry mixes: slope around -70..-90 dB/s, tail/peak around -12 dB. Bright
    'AI echo': slope -10..-30 dB/s, tail/peak near -5 dB (higher = more tail).
    """
    x = sig[:, 0] if sig.ndim > 1 else sig
    hop = n_fft // 4
    spec = np.abs(stft(x, n_fft, hop)[0])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    t_frame = hop / sr
    onsets = detect_transients(spec, hop, sr)
    hf = (freqs >= 2500) & (freqs <= 10000)
    env = np.sqrt(np.mean(spec[:, hf] ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(env, 1e-9))
    slopes, ratios = [], []
    for o in np.where(onsets)[0]:
        n_fit = min(len(db) - o, int(0.3 / t_frame))
        if n_fit >= 4:
            k = np.arange(n_fit)
            slopes.append(float(np.polyfit(k, db[o:o + n_fit], 1)[0] / t_frame))
        w0 = o + int(0.3 / t_frame)
        w1 = min(len(env), o + int(0.8 / t_frame))
        if w1 > w0:
            peak = float(np.max(env[o:w0]))
            tail = float(np.mean(env[w0:w1]))
            ratios.append(20.0 * np.log10(max(tail, 1e-9) / max(peak, 1e-9)))
    return {
        "hf_decay_slope_db_s": float(np.median(slopes)) if slopes else 0.0,
        "tail_peak_ratio_db": float(np.mean(ratios)) if ratios else 0.0,
        "n_onsets": int(len(slopes)),
    }


def _derverb_pass(audio: np.ndarray, sr: int, margin_db_s: float, max_db: float,
                  hf_start: float, n_fft: int, transient_guard: float
                  ) -> tuple[np.ndarray, dict]:
    """One suppression pass."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    hop = n_fft // 4
    lin = integrated_loudness(x, sr)

    outs = []
    for c in range(x.shape[1]):
        spec, win, hop_, pad = stft(x[:, c], n_fft, hop)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        t_frame = hop / sr

        anchor = (freqs >= 1000) & (freqs < 2000)
        hf_w = np.clip((freqs - hf_start) / 1500.0, 0.0, 1.0)[None, :]

        onsets = detect_transients(mag, hop, sr)
        seg_bounds = np.unique(np.concatenate([[0], np.where(onsets)[0], [len(mag)]]))

        g = np.ones_like(mag)
        for si in range(len(seg_bounds) - 1):
            s0, s1 = int(seg_bounds[si]), int(seg_bounds[si + 1])
            seg_len = s1 - s0
            if seg_len < 4:
                continue
            seg = mag[s0:s1]
            slope = _anchor_slope_db_s(seg, anchor, t_frame)
            required = min(slope - margin_db_s, -35.0)  # dB/s (negative) + backstop
            release = 10 ** (required * t_frame / 20.0)  # factor per frame (<1)
            floor = np.empty_like(seg)
            floor[0] = seg[0]
            for t_i in range(1, seg_len):
                floor[t_i] = floor[t_i - 1] * release
            g_seg = np.minimum(1.0, floor / np.maximum(seg, 1e-10))
            g_seg = 1.0 - hf_w * (1.0 - g_seg)
            g_seg = np.maximum(g_seg, 10 ** (-max_db / 20.0))
            g[s0:s1] = g_seg

        if onsets.any() and transient_guard > 0:
            lift = np.zeros(len(g))
            lift[onsets] = 1.0
            lift = lfilter([0.6], [1.0, -0.4], lift)
            g = np.maximum(g, np.clip(lift, 0, 1)[:, None] * transient_guard)

        fast = lfilter([0.7], [1.0, -0.3], g, axis=0)
        slow = lfilter([0.4], [1.0, -0.6], g, axis=0)
        g = np.minimum(fast, slow)

        phase = np.exp(1j * np.angle(spec))
        y = istft(mag * g * phase, n_fft, hop, win, len(x[:, c]), pad)
        outs.append(y)

    y = np.stack(outs, axis=1) if x.shape[1] > 1 else outs[0]
    lufs_out = integrated_loudness(y, sr)
    comp = None
    if np.isfinite(lin) and np.isfinite(lufs_out):
        d = lin - lufs_out
        if 0.05 < abs(d) < 12:
            y = y * (10 ** (d / 20.0))
            comp = float(d)
    return y.astype(np.float32), {"out_sr": sr, "loudness_compensation_db": comp}


def derverb(audio: np.ndarray, sr: int,
            margin_db_s: float = 45.0,
            max_db: float = 10.0,
            passes: int = 2,
            hf_start: float = 2500.0,
            n_fft: int = 4096,
            transient_guard: float = 0.7) -> tuple[np.ndarray, dict]:
    """Shorten HF tails that decay slower than the mix's own anchor band.

    margin_db_s: how much steeper than the anchor the HF must decay
    (lower = stronger). passes: iterate; each pass converges toward the dry
    tail reference. max_db caps the per-frame attenuation.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    lufs_in = integrated_loudness(x, sr)

    cur = x.astype(np.float32)
    cur_sr = sr
    passes_done = 0
    for _ in range(max(1, int(passes))):
        cur, _rep = _derverb_pass(cur, cur_sr, margin_db_s, max_db, hf_start,
                                  n_fft, transient_guard)
        cur_sr = _rep["out_sr"]
        passes_done += 1

    lufs_out = integrated_loudness(cur, cur_sr)
    comp = None
    if np.isfinite(lufs_in) and np.isfinite(lufs_out):
        d = lufs_in - lufs_out
        if 0.05 < abs(d) < 12:
            cur = cur * (10 ** (d / 20.0))
            comp = float(d)
    return cur.astype(np.float32), {"passes": passes_done,
                                    "loudness_compensation_db": comp,
                                    "out_sr": cur_sr}