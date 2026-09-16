"""AI enhancement stage: Demucs stem separation + per-stem restoration + SBR HF repair.

Why: magnitude gating can only REMOVE added artifacts. MP3/AAC/RVQ codecs often
REPLACE high-frequency content (band copying / band dropping) — that lost signal
needs reconstruction. Strategy:

1. Demucs separates drums/bass/other/vocals (trained on clean professional music,
   so each stem is an implicitly "clean" re-render of that part).
2. Per-stem DSP restoration with tuned parameters (drums get transient shielding,
   harmonic stems get the strongest gate, bass is skipped).
3. SBR-style band repair on the incoherent HF bed: bins whose envelope does not
   follow the octave below get their magnitude replaced by octave-replicated
   (coherent) energy — like HE-AAC's Spectral Band Replication, but inverted:
   we use it to UNDO what the codec destroyed.
4. Remix + loudness match + true-peak ceiling.

Everything is benchmarked in tests/bench_ai.py.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .analysis import stft
from .loudness import integrated_loudness, true_peak_dbtp
from .restoration import RestoreParams, restore, _median_smooth_time
from .io import read_audio, write_audio

TARGET_SR = 44100  # demucs native rate

# model cache so repeated enhance() calls reuse the loaded network
_MODEL_CACHE = {}


def get_or_load_model(model_name: str = "htdemucs", device: str = "cuda"):
    import torch
    if not torch.cuda.is_available():
        device = "cpu"
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_demucs(model_name, device)
    return _MODEL_CACHE[key]

STEM_PARAMS = {
    # transient_guard high: drums must survive; texture gate targets the
    # sizzle BETWEEN hits.
    "drums": dict(oversubtract=1.6, floor=0.30, reduction_db=10, hf_focus=3500.0),
    # bass has no meaningful HF content -> gating is pointless
    "bass": None,
    # harmonic/pad content: strongest treatment, most of the metallic ring lives here
    "other": dict(oversubtract=2.2, floor=0.25, reduction_db=12, hf_focus=2200.0),
    # vocals: sibilance needs care, artifacts concentrate 4-9 kHz
    "vocals": dict(oversubtract=1.8, floor=0.30, reduction_db=10, hf_focus=2800.0),
}


def _load_demucs(model_name: str = "htdemucs", device: str = "cuda"):
    from demucs.pretrained import get_model
    model = get_model(model_name)
    model.to(device)
    model.eval()
    return model


def separate(audio: np.ndarray, sr: int, model=None, device: str = "cuda",
             model_name: str = "htdemucs") -> tuple[dict, int]:
    """Return ({stem: (n,2) float32}, out_sr). Resamples to 44.1k if needed."""
    import torch
    from demucs.apply import apply_model
    import torchaudio.functional as AF

    if not torch.cuda.is_available():
        device = "cpu"
    model = model or _load_demucs(device=device)
    out_sr = model.samplerate

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if sr != out_sr:
        t = torch.from_numpy(x.T[None])           # (1, 2, n)
        t = AF.resample(t, sr, out_sr)
        x = t[0].numpy().T

    wav = torch.from_numpy(x.T[None]).to(device)  # (1, 2, n)
    with torch.inference_mode():
        stems = apply_model(model, wav, split=True, overlap=0.25, progress=False)[0]  # (n_stem, 2, n)
    names = model.sources
    out = {name: stems[i].cpu().numpy().T for i, name in enumerate(names)}  # (n,2) each
    return out, out_sr


def _incoherence_bins(mag: np.ndarray, n_fft: int, sr: int,
                      hop: int | None = None,
                      hf_focus: float = 6000.0) -> np.ndarray:
    """Per-bin incoherence (0=coherent music .. 1=noise bed), from restoration."""
    from .restoration import _envelope_incoherence, RestoreParams
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    p = RestoreParams(n_fft=n_fft)
    return _envelope_incoherence(mag, p, sr, n_fft // 4, freqs)[0]


def band_replicate_repair(x: np.ndarray, sr: int,
                          f_start: float = 8000.0,
                          incoh_thresh: float = 0.5,
                          copy_gain: float = 0.85,
                          n_fft: int = 4096) -> tuple[np.ndarray, dict]:
    """SBR-style HF repair: replace incoherent HF-bed bins with octave-below copies.

    Per bin b >= f_start: coherence c (music) vs (1-incoh). For incoherent bins,
    magnitude is blended toward mag[b/2] * copy_gain. Phase is kept (magnitudes
    carry the information; original phase avoids smearing).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    outs, info = [], []
    for c in range(x.shape[1]):
        spec, win, hop, pad = stft(x[:, c], n_fft, n_fft // 4)
        mag = np.abs(spec)
        phase = np.exp(1j * np.angle(spec))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        incoh = _incoherence_bins(mag, n_fft, sr)
        # repair weight per bin: 0 (keep) .. 1 (full replace)
        w = np.zeros_like(freqs)
        hi = (freqs >= f_start) & (freqs < sr / 2 - 1000)
        w[hi] = np.clip((incoh[hi] - incoh_thresh) / max(1e-6, (1.0 - incoh_thresh)), 0, 1)
        W = w[None, :]                                  # (1, bins)
        # octave-below copy: bin b <- bin b//2 (true octave mapping), smoothed over 3 bins
        nb = mag.shape[1]
        src_idx = np.clip(np.arange(nb) // 2, 0, nb - 1)
        rep = mag[:, src_idx]
        rep = np.apply_along_axis(lambda m: np.convolve(m, np.ones(3) / 3, "same"), 1, rep)
        mag_new = mag * (1 - W * (1 - copy_gain)) + W * copy_gain * rep
        spec_new = mag_new * phase
        from .analysis import istft
        y = istft(spec_new, n_fft, hop, win, len(x[:, c]), pad)
        # blend dry/wet by mean repair weight to avoid over-processing
        mean_w = float(np.mean(w[hi])) if hi.any() else 0.0
        y = x[:, c] * (1 - 0.0) + (y - x[:, c])  # full replacement already blended per-bin
        outs.append(y)
        info.append(mean_w)
    y = np.stack(outs, axis=1) if len(outs[0].shape) else outs[0]
    rep_info = {"sbr_mean_weight": info if len(info) > 1 else info[0]}
    return y.astype(np.float32), rep_info


def enhance(audio: np.ndarray, sr: int,
            use_ai: bool = True,
            sbr: bool = True,
            residual_alpha: float = 0.25,
            sbr_thresh: float = 0.35,
            model=None,
            progress_cb=None) -> tuple[np.ndarray, int, dict]:
    """Full pipeline. Returns (enhanced_audio, out_sr, report).

    use_ai=False -> pure DSP chain at original sr (no demucs).
    residual_alpha: how much of the demucs residual (the part demucs does NOT
    recognize as music - mostly the sizzle bed) survives. 1.0 = keep all
    (restored), 0.0 = full re-render (strongest de-slop).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    lufs_in = integrated_loudness(x, sr)

    if not use_ai:
        params = RestoreParams(oversubtract=1.6, floor=0.3, reduction_db=10)
        y, rep = restore(x.astype(np.float32), sr, params)
        y = y[:, None] if y.ndim == 1 else y
        y = y.astype(np.float64)
        out_sr = sr
        rep = {"stems": {"mix": rep}}
    else:
        model = model or get_or_load_model()
        stems, out_sr = separate(x, sr, model=model)
        remixed = []
        rep = {"stems": {}}
        names = list(stems.keys())
        for i, name in enumerate(names):
            if progress_cb:
                progress_cb(0.15 + 0.5 * i / max(len(names), 1), name)
            p = STEM_PARAMS.get(name)
            if p is None:
                remixed.append(stems[name])
                rep["stems"][name] = "skipped"
                continue
            params = RestoreParams(**p)
            ys, rsp = restore(stems[name].astype(np.float32), out_sr, params)
            if sbr and name in ("other", "vocals", "drums"):
                ys, ri = band_replicate_repair(ys.astype(np.float64), out_sr,
                                               f_start=8000.0, incoh_thresh=sbr_thresh)
                rsp["sbr"] = ri
            remixed.append(ys)
            rep["stems"][name] = rsp

        # residual = what demucs did NOT model as music (mostly the sizzle bed)
        import torch
        import torchaudio.functional as AF
        if sr != out_sr:
            xr = AF.resample(torch.from_numpy(x.T[None].astype(np.float32)),
                             sr, out_sr)[0].numpy().T.astype(np.float64)
        else:
            xr = x
        n = min(len(xr), min(len(s) for s in remixed))
        xr = xr[:n]
        remixed = [s[:n] for s in remixed]
        residual = xr[:n] - np.sum(remixed, axis=0)
        if progress_cb:
            progress_cb(0.75, "residual")
        yr, _ = restore(residual.astype(np.float32), out_sr,
                        RestoreParams(oversubtract=1.4, floor=0.4, reduction_db=8))
        if sbr:
            yr, ri = band_replicate_repair(yr.astype(np.float64), out_sr,
                                           f_start=8000.0, incoh_thresh=sbr_thresh)
        y = np.sum(remixed, axis=0) + float(np.clip(residual_alpha, 0.0, 1.0)) * yr
        rep["residual_alpha"] = residual_alpha
        if progress_cb:
            progress_cb(0.9, "mixdown")

    # loudness match to input + true-peak ceiling
    lufs_out = integrated_loudness(y, out_sr)
    if np.isfinite(lufs_in) and np.isfinite(lufs_out):
        diff = lufs_in - lufs_out
        if abs(diff) > 0.05 and abs(diff) < 12:
            y = y * (10 ** (diff / 20.0))
            rep["loudness_compensation_db"] = float(diff)
    tp = true_peak_dbtp(y, out_sr)
    if tp > -1.0:
        y = y * (10 ** ((-1.0 - tp) / 20.0))
    rep["lufs_in"] = None if not np.isfinite(lufs_in) else round(float(lufs_in), 2)
    rep["out_sr"] = out_sr
    return y.astype(np.float32), out_sr, rep