# tests for noslop2: loudness validation, STFT roundtrip, artifact detection, restoration
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.loudness import integrated_loudness, true_peak_dbtp
from app.analysis import stft, istft, residual_basis, band_flatness_db, flicker_index
from app.restoration import RestoreParams, restore, detect_transients, _artifact_estimate


def make_test_signal(dur=3.0, sr=48000, seed=3):
    """Deterministic 'music-like' test signal: bass + melody + hats + snare."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 55 * t)
    x += 0.2 * np.sin(2 * np.pi * 220 * t)
    x += 0.15 * np.sin(2 * np.pi * 880 * t)
    env = np.zeros_like(t)
    for i in range(int(dur * 4)):
        idx = int(i * sr * 0.25)
        if idx + 2000 < len(env):
            env[idx:idx + 2000] += np.hanning(2000) * 0.3
    x += env * rng.standard_normal(len(t))
    for i in range(int(dur * 2)):
        idx = int(i * sr * 0.5)
        n = rng.standard_normal(int(0.1 * sr)) * 0.25 * np.hanning(int(0.1 * sr))
        x[idx:idx + len(n)] += n
    x = x / np.max(np.abs(x)) * 0.7
    return x.astype(np.float32), sr


def make_artifact_layer(x, sr, seed=11, amp_db=-30):
    """Synthetic RVQ-residual layer: flickering sparse HF noise + combed ringing."""
    rng = np.random.default_rng(seed)
    n = len(x)
    art = np.zeros(n)
    n_win = 4096; hop = 1024
    w = np.hanning(n_win)
    frames = 1 + (n - n_win) // hop
    for i in range(frames):
        if rng.random() < 0.4:
            f = rng.uniform(4000, 18000)
            b0 = int(f * n_win / sr)
            b1 = min(n_win // 2 - 2, int((f + 2500) * n_win / sr))
            tone = np.sin(2 * np.pi * rng.uniform(50, 200) * (hop / sr) * i)
            tone = tone + 0.5 * np.sin(2 * np.pi * 137 * (hop / sr) * i)
            tone = tone / max(np.abs(tone).max(), 1e-9)
            art[i * hop:i * hop + n_win] += w * tone * 10 ** (amp_db / 20)
    d = int(sr * 0.23e-3)
    art = art + 0.4 * np.concatenate([np.zeros(d), art[:-d]])
    from scipy.signal import butter, filtfilt
    b, a = butter(3, 2000 / (sr / 2), 'high')
    art = filtfilt(b, a, art)
    art = art / (np.max(np.abs(art)) + 1e-9) * 10 ** (amp_db / 20)
    return art.astype(np.float32)


@pytest.fixture
def test_pair():
    x, sr = make_test_signal()
    art = make_artifact_layer(x, sr)
    y = x + art
    y = y / np.max(np.abs(y)) * 0.9
    return x, art, y, sr


# ---------------------------------------------------------------- loudness
def test_bs1770_reference():
    fs = 48000
    t = np.arange(2 * fs) / fs
    s = 10 ** (-23 / 20) * np.sin(2 * np.pi * 997 * t)
    x = np.stack([s, s], axis=1)
    l = integrated_loudness(x, fs)
    assert abs(l - (-23.0)) < 0.2


def test_true_peak():
    fs = 48000
    n = 4 * fs
    t = np.arange(n) / fs
    # f = fs/3: samples only reach phase 120/240 deg -> sample peak = amp*sqrt(3)/2,
    # while the reconstructed (oversampled) sine peaks at full amplitude.
    amp = 0.95 / (np.sqrt(3) / 2)
    # fade edges so the interpolator has no discontinuity to ring on
    fade = np.ones(n); nf = int(0.3 * fs)
    fade[:nf] = np.hanning(2 * nf)[:nf]
    fade[-nf:] = np.hanning(2 * nf)[nf:]
    x = amp * np.sin(2 * np.pi * 16000.0 * t) * fade
    sample_peak = 20 * np.log10(np.max(np.abs(x)))
    tp = true_peak_dbtp(x, fs)
    assert abs(tp - 20 * np.log10(amp)) < 0.1  # reconstructs full sine amplitude
    assert tp > sample_peak + 1.0  # true peak exceeds sample peak
    assert tp < 20 * np.log10(1.42)


# ---------------------------------------------------------------- stft
def test_stft_roundtrip():
    x, sr = make_test_signal(dur=2.0)
    spec, win, hop, pad = stft(x, 4096, 1024)
    y = istft(spec, 4096, 1024, win, len(x), pad)
    err = np.max(np.abs(y - x[:len(y)]))
    assert err < 1e-6, f"STFT roundtrip error {err}"


def test_residual_basis_shapes():
    A = residual_basis(4096, 48000, n_atoms=12)
    assert A.shape == (12, 2049)
    assert np.all(np.isfinite(A))
    rms = np.sqrt(np.mean(A ** 2, axis=1))
    assert np.allclose(rms, 1.0, atol=1e-6)


# ---------------------------------------------------------------- detection
def test_artifact_estimation_finds_residual(test_pair):
    x, art, y, sr = test_pair
    p = RestoreParams()
    spec, win, hop, pad = stft(y, p.n_fft, p.n_fft // 4)
    basis = residual_basis(p.n_fft, sr)
    art_est, report = _artifact_estimate(np.abs(spec), sr, hop, p, basis)
    assert np.all(np.isfinite(art_est))
    ratio = report["artifact_energy_ratio"]
    assert 0 < ratio < 1.0


def test_restore_improves_artifact_ratio(test_pair):
    x, art, y, sr = test_pair
    params = RestoreParams(n_fft=4096, reduction_db=12, floor=0.25, oversubtract=2.0)
    clean, report = restore(y, sr, params)
    assert clean.shape == y.shape
    assert np.all(np.isfinite(clean))


def test_restore_preserves_transients(test_pair):
    x, art, y, sr = test_pair
    params = RestoreParams(n_fft=4096, reduction_db=12, floor=0.25, oversubtract=2.0)
    clean, report = restore(y, sr, params)
    sc = np.abs(stft(clean, 2048, 512)[0])
    so = np.abs(stft(x, 2048, 512)[0])
    fc = detect_transients(sc, 512, sr)
    fo = detect_transients(so, 512, sr)
    if fo.sum() > 3:
        overlap = (fc & fo).sum() / max(fo.sum(), 1)
        assert overlap > 0.7, f"onset overlap {overlap:.2f}"


def test_restore_quiets_artifact_bands(test_pair):
    """Direct measurable: HF band (4-18 kHz) magnitude must drop after restore."""
    x, art, y, sr = test_pair
    params = RestoreParams(n_fft=4096, reduction_db=12, floor=0.25, oversubtract=2.0)
    clean, report = restore(y, sr, params)
    spec_dirty = np.abs(stft(y, 4096, 1024)[0])
    spec_clean = np.abs(stft(clean, 4096, 1024)[0])
    freqs = np.fft.rfftfreq(4096, 1.0 / sr)
    band = (freqs >= 4000) & (freqs <= 18000)
    e_dirty = np.sqrt(np.mean(spec_dirty[:, band] ** 2))
    e_clean = np.sqrt(np.mean(spec_clean[:, band] ** 2))
    assert e_clean < e_dirty * 0.9, f"HF RMS dirty={e_dirty:.5f} clean={e_clean:.5f}"