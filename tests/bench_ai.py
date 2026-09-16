"""Benchmark: DSP-only vs AI-enhanced (demucs + SBR), measured against originals.

Metrics per codec variant:
- hf_incoh:      HF incoherence (sizzle bed, 6-16k) - lower = cleaner
- band_corr:     HF envelope correlation with ORIGINAL music (8-16k) - higher = truer
- onset_overlap: transient preservation - must stay ~1.0
- dLUFS:         loudness neutrality vs input - must be ~0
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analysis import analyze_health, stft
from app.restoration import RestoreParams, restore
from app.enhance import enhance
from app.io import read_audio
from scipy.signal import resample_poly


def band_env_corr(a, b, sr_a, sr_b, n_fft=4096):
    """Correlation of HF (8-16k) band envelopes after resampling to common sr."""
    if sr_a != sr_b:
        g = sr_a / sr_b
        from math import gcd
        g0 = gcd(int(sr_a), int(sr_b))
        a = resample_poly(a, int(sr_b) // g0, int(sr_a) // g0, axis=0)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr_b)
    hf = (freqs >= 8000) & (freqs <= 16000)
    ma = np.abs(stft(a[:, 0], n_fft, n_fft // 4)[0])
    mb = np.abs(stft(b[:, 0], n_fft, n_fft // 4)[0])
    ea = np.sqrt(np.mean(ma[:, hf] ** 2, axis=1))
    eb = np.sqrt(np.mean(mb[:, hf] ** 2, axis=1))
    return float(np.corrcoef(ea, eb)[0, 1])


def main():
    from tests.bench_codec import gen_music, codec_roundtrip
    import tempfile
    x, sr = gen_music(dur=10.0)
    wd = tempfile.mkdtemp()

    print(f"{'case':16s} {'variant':10s} {'hf_incoh':>9s} {'bandCorr':>9s} {'dLUFS':>6s} {'TP':>6s}")
    for codec, br in (("opus", 32), ("mp3", 64)):
        dec, orig = codec_roundtrip(x, sr, codec, br, wd)
        orig_r = dec  # length-aligned orig
        # DSP only
        dsp, _ = restore(dec.astype(np.float32), sr,
                         RestoreParams(oversubtract=1.6, floor=0.3, reduction_db=10))
        # AI enhanced
        ai, ai_sr, rep = enhance(dec.astype(np.float64), sr, use_ai=True, sbr=True,
                                 residual_alpha=0.25, sbr_thresh=0.35)
        for name, sig, s_sig in (("decoded", dec, sr), ("dsp", dsp, sr), ("ai", ai, ai_sr)):
            h = analyze_health(sig.astype(np.float64), s_sig)
            corr = band_env_corr(sig.astype(np.float64), orig, s_sig, sr)
            lufs = h["lufs"]
            lufs_in = analyze_health(dec, sr)["lufs"]
            print(f"{codec+' '+str(br)+'k':16s} {name:10s} {h['hf_incoherence']:9.4f} {corr:9.3f} "
                  f"{h['lufs']-lufs_in:+6.2f} {h['true_peak_dbtp']:6.2f}")


if __name__ == "__main__":
    main()