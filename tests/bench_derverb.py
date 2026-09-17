"""Benchmark derverb: shortens the metallic 'AI echo' tail without hurting the mix.

Synthetic test: clean music + bright HF reverb tail (RT60 1.5s). Metrics:
tail/peak ratio (dB, higher = more hang), post-onset HF slope, LUFS neutrality,
onset preservation. Dry reference included.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import soundfile as sf
from app.io import read_audio
from app.derverb import derverb, tail_metrics
from app.analysis import analyze_health
from app.restoration import detect_transients
from app.analysis import stft


def make_reverb_tail(x, sr, rt60_s=1.5, hf_bias=0.8, amp=0.25, seed=7):
    """Bright synthetic reverb: exponentially decaying noise IR, HF-weighted."""
    rng = np.random.default_rng(seed)
    n_ir = int(rt60_s * sr)
    t = np.arange(n_ir) / sr
    decay = np.exp(-6.9 * t / rt60_s)
    ir = rng.standard_normal(n_ir) * decay
    from scipy.signal import butter, lfilter
    b, a = butter(1, 2000 / (sr / 2), "high")
    ir = (1 - hf_bias) * ir + hf_bias * lfilter(b, a, ir)
    ir[0] += 1.0
    ir = ir / np.max(np.abs(ir)) * amp
    n = len(x)
    wet = np.zeros_like(x)
    for c in range(x.shape[1]):
        wet[:, c] = np.convolve(x[:, c], ir)[:n]
    return x + wet


def main():
    from tests.bench_codec import gen_music
    x, sr = gen_music(dur=8.0)
    wet = make_reverb_tail(x, sr)

    m_dry = tail_metrics(x, sr)
    m_wet = tail_metrics(wet, sr)
    print("dry: slope %+.1f dB/s | tail/peak %.1f dB" % (
        m_dry["hf_decay_slope_db_s"], m_dry["tail_peak_ratio_db"]))
    print("wet: slope %+.1f dB/s | tail/peak %.1f dB" % (
        m_wet["hf_decay_slope_db_s"], m_wet["tail_peak_ratio_db"]))

    for name, kwargs in (("standard", dict(margin_db_s=45.0, passes=2)),
                         ("strong", dict(margin_db_s=75.0, passes=3))):
        y, rep = derverb(wet, sr, **kwargs)
        m = tail_metrics(y.astype("float64"), sr)
        a = analyze_health(y.astype("float64"), sr)
        o_wet = detect_transients(np.abs(stft(wet[:, 0], 4096, 1024)[0]), 1024, sr)
        o_y = detect_transients(np.abs(stft(y.astype("float64")[:, 0], 4096, 1024)[0]), 1024, sr)
        overlap = float((o_wet & o_y).sum() / max(o_wet.sum(), 1))
        print("derverb %s: tail/peak %.1f -> %.1f dB (dry %.1f) | slope %+.1f -> %+.1f | "
              "LUFS %.2f | onsets %.2f | passes %d" % (
                  name, m_wet["tail_peak_ratio_db"], m["tail_peak_ratio_db"],
                  m_dry["tail_peak_ratio_db"], m_wet["hf_decay_slope_db_s"],
                  m["hf_decay_slope_db_s"], a["lufs"], overlap, rep["passes"]))
        out = "demo_derverb_%s.wav" % name
        sf.write(out, y, sr, subtype="PCM_24")
        print("  wrote", out)

    # codec material: does the tail limiter help the real MP3 demo too?
    dec, dsr, _ = read_audio("demo_mp364.wav")
    m_dec = tail_metrics(dec, dsr)
    y2, rep2 = derverb(dec, dsr, margin_db_s=45.0, passes=2)
    m2 = tail_metrics(y2.astype("float64"), dsr)
    a2 = analyze_health(y2.astype("float64"), dsr)
    print("mp3 64k: tail/peak %.1f -> %.1f dB | slope %+.1f -> %+.1f | LUFS %.2f" % (
        m_dec["tail_peak_ratio_db"], m2["tail_peak_ratio_db"],
        m_dec["hf_decay_slope_db_s"], m2["hf_decay_slope_db_s"], a2["lufs"]))
    sf.write("demo_mp3_derverb.wav", y2, dsr, subtype="PCM_24")
    print("  wrote demo_mp3_derverb.wav")


if __name__ == "__main__":
    main()