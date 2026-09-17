"""Benchmark polish: combined metallic de-ringing + AI-echo tail suppression.

Test material:
1. dry reference (clean music);
2. nasty material = combs + bright reverb tail (worst case);
3. real MP3 64k demo;
4. clean original (safety: the combined pass must stay near-transparent).

Metrics: tail/peak (dB), HF slope, kurtosis (spikiness), comb strength,
onset overlap, LUFS neutrality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import soundfile as sf
from app.io import read_audio
from app.polish import polish
from app.derverb import tail_metrics
from app.dering_ai import metallic_metrics
from app.analysis import analyze_health
from app.restoration import detect_transients
from app.analysis import stft

from tests.bench_derverb import make_reverb_tail


def add_comb(x, sr, f0=5500.0, depth_db=6.0):
    """Add a metallic comb: convolution with a short delayed-decay pair."""
    d = int(sr / f0)
    ir = np.zeros(d * 3)
    ir[0] = 1.0
    ir[d] = -10 ** (-depth_db / 20.0)
    ir[2 * d] = 10 ** (-depth_db / 20.0)
    n = len(x)
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(x[:, c], ir)[:n]
    return out


def onset_overlap(a, b, sr, n_fft=4096, hop=1024):
    # align lengths (processing may resample/trim): use the shorter duration
    n = min(len(a), len(b))
    a = a[:n]
    # resample b to a's sr if needed is out of scope here: caller passes same-sr refs;
    # instead compare onset TIMES scaled by duration ratio
    ta = detect_transients(np.abs(stft(a[:, 0] if a.ndim > 1 else a, n_fft, hop)[0]), hop, sr)
    tb = detect_transients(np.abs(stft(b[:, 0] if b.ndim > 1 else b, n_fft, hop)[0]), hop, sr)
    # scale-aware: compare onset TIMES (seconds), tolerance +-35 ms
    from scipy.signal import resample_poly as _rp
    import math
    dur_a = len(ta) * hop / sr
    dur_b = len(tb) * hop / sr
    # resample the LONGER mask series to the shorter duration
    if dur_a > dur_b * 1.01:
        g = math.gcd(int(round(dur_a * 1000)), int(round(dur_b * 1000)))
        ta = _rp(ta.astype("float64"), int(round(dur_b * 1000)) // g,
                 int(round(dur_a * 1000)) // g)
        ta = ta > 0.5
    elif dur_b > dur_a * 1.01:
        g = math.gcd(int(round(dur_b * 1000)), int(round(dur_a * 1000)))
        tb = _rp(tb.astype("float64"), int(round(dur_a * 1000)) // g,
                 int(round(dur_b * 1000)) // g)
        tb = tb > 0.5
    n = min(len(ta), len(tb))
    ta, tb = ta[:n], tb[:n]
    return float((ta & tb).sum() / max(tb.sum(), 1))


def row(label, sig, sr):
    m = tail_metrics(sig, sr)
    k = metallic_metrics(sig, sr)
    h = analyze_health(sig, sr)
    print("%-22s tail/peak %6.1f dB | slope %+6.1f | kurt %6.0f | comb %.3f | LUFS %6.2f" % (
        label, m["tail_peak_ratio_db"], m["hf_decay_slope_db_s"],
        k["kurtosis"], k["comb"], h["lufs"]))
    return m, k, h


def main():
    from tests.bench_codec import gen_music
    x, sr = gen_music(dur=8.0)

    print("--- case 1: dry reference ---")
    row("dry", x, sr)

    print("--- case 2: combs + bright tail (nasty) ---")
    nasty = add_comb(make_reverb_tail(x, sr), sr)
    row("nasty", nasty, sr)
    for name, kwargs in (("standard", dict(rounds=1)),
                         ("strong", dict(rounds=2, dering_max_db=10.0,
                                         derverb_margin_db_s=60.0))):
        y, osr, rep = polish(nasty, sr, **kwargs)
        row("polish %s" % name, y.astype("float64"), osr)
        ovl = onset_overlap(y.astype("float64"), x, sr)
        print("   onsets %.2f | stages %s" % (
            ovl, [(s["stage"], s.get("n_notches", "-")) for s in rep["stages"]]))
        out = "demo_polish_%s.wav" % name
        sf.write(out, y, osr, subtype="PCM_24")
        print("   wrote", out)

    print("--- case 3: real MP3 64k demo ---")
    dec, dsr, _ = read_audio("demo_mp364.wav")
    row("mp3 decoded", dec, dsr)
    y2, osr2, rep2 = polish(dec, dsr, rounds=1)
    row("mp3 polished", y2.astype("float64"), osr2)
    sf.write("demo_mp3_polish.wav", y2, osr2, subtype="PCM_24")
    print("   wrote demo_mp3_polish.wav")

    print("--- case 4: clean original (safety) ---")
    y3, osr3, rep3 = polish(x, sr, rounds=1)
    m_clean_before = tail_metrics(x, sr)
    m_clean_after = tail_metrics(y3.astype("float64"), osr3)
    k_before = metallic_metrics(x, sr)
    k_after = metallic_metrics(y3.astype("float64"), osr3)
    print("clean: kurt %.0f -> %.0f | comb %.3f -> %.3f | tail/peak %.1f -> %.1f" % (
        k_before["kurtosis"], k_after["kurtosis"],
        k_before["comb"], k_after["comb"],
        m_clean_before["tail_peak_ratio_db"], m_clean_after["tail_peak_ratio_db"]))


if __name__ == "__main__":
    main()