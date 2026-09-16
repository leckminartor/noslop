"""End-to-end benchmark: real codec artifacts (Opus/MP3 low bitrate) -> restore -> metrics.

Measures against the ORIGINAL and against a naive lowpass (the 'dumb' approach):
- hf_noise_ratio:  flickering sparse energy 4-18kHz relative to total (lower=better)
- flatness_db:     spectral flatness in 4-18kHz (lower=noisier band, we want artifact gone
                   while keeping music -> compare band correlation with original instead)
- band_corr:       correlation of band energy profiles with ORIGINAL music (higher=better)
- onset_overlap:   preserved transients (higher=better)
- loudness delta:  LUFS change vs original (should be ~0)
"""
import numpy as np
import soundfile as sf
import subprocess, tempfile, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.analysis import stft, band_flatness_db, flicker_index
from app.restoration import RestoreParams, restore, detect_transients
from app.loudness import integrated_loudness


def gen_music(dur=10.0, sr=48000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    # bass line with pitch changes
    notes = [55, 55, 65.4, 49]
    bass = np.zeros_like(t)
    seg = int(dur * sr / 4)
    for i, f in enumerate(notes):
        s = slice(i * seg, (i + 1) * seg)
        bass[s] = 0.32 * np.sin(2 * np.pi * f * t[s]) * np.hanning(seg * 2)[:seg] if False else 0.32 * np.sin(2 * np.pi * f * t[s])
    # melody (detuned saw-ish via additive)
    mel = np.zeros_like(t)
    for i, f in enumerate([220, 261.6, 293.7, 246.9]):
        s = slice(i * seg, (i + 1) * seg)
        env = np.minimum(1, 8 * (t[s] - (i * dur / 4))) * np.exp(-2 * ((t[s] - i * dur / 4)))
        for h, a in ((1, 0.12), (2, 0.06), (3, 0.03)):
            mel[s] += a * np.sin(2 * np.pi * f * h * t[s]) * env
    # hats + snare
    perc = np.zeros_like(t)
    for i in range(int(dur * 8)):  # 8ths
        idx = int(i * sr * dur / (int(dur * 8)))
        if idx + 3000 < len(perc):
            n = rng.standard_normal(3000) * np.hanning(3000) * 0.12
            perc[idx:idx + 3000] += n
    for i in range(int(dur * 2)):
        idx = int(i * sr * 0.5)
        if idx + 6000 < len(perc):
            n = rng.standard_normal(6000) * np.hanning(6000) * 0.3
            perc[idx:idx + 6000] += n
    x = bass + mel + perc
    x = x / np.max(np.abs(x)) * 0.85
    st = np.stack([x, np.roll(x, 137) * 0.98], axis=1)
    return st.astype(np.float32), sr


def codec_roundtrip(x, sr, codec, bitrate, workdir):
    """Encode/decode through a real codec; returns decoded audio at original sr."""
    src = os.path.join(workdir, "in.wav")
    enc = os.path.join(workdir, f"out.{codec}")
    dec = os.path.join(workdir, f"dec_{codec}_{bitrate}.wav")
    sf.write(src, x, sr)
    if codec == "opus":
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "libopus",
               "-b:a", f"{bitrate}k", enc]
    elif codec == "mp3":
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "libmp3lame",
               "-b:a", f"{bitrate}k", enc]
    elif codec == "aac":
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "aac",
               "-b:a", f"{bitrate}k", enc]
    subprocess.run(cmd, check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-ar", str(sr),
                    "-c:a", "pcm_f32le", dec], check=True)
    y, dsr = sf.read(dec, always_2d=True)
    assert dsr == sr
    n = min(len(y), len(x))
    return y[:n].astype(np.float64), x[:n]


def band_profile(x, sr, n_fft=4096, hop=1024):
    """Per-band RMS profile (32 log bands 100Hz..20kHz)."""
    spec = np.abs(stft(x[:, 0] if x.ndim > 1 else x, n_fft, hop)[0])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(100, 20000, 33)
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi)
        prof.append(float(np.sqrt(np.mean(spec[:, band] ** 2))))
    return np.asarray(prof)


def onset_overlap(a, b, sr, n_fft=2048, hop=512):
    fa = detect_transients(np.abs(stft(a, n_fft, hop)[0]), hop, sr)
    fb = detect_transients(np.abs(stft(b, n_fft, hop)[0]), hop, sr)
    if fb.sum() < 3:
        return 1.0
    return float((fa & fb).sum() / max(fb.sum(), 1))


def main():
    workdir = tempfile.mkdtemp(prefix="noslop_bench_")
    x, sr = gen_music()
    prof_orig = band_profile(x, sr)

    results = []
    for codec, bitrate in (("opus", 16), ("opus", 32), ("mp3", 64), ("aac", 48)):
        dec, orig = codec_roundtrip(x, sr, codec, bitrate, workdir)
        # naive lowpass at 12k (the 'dumb' approach for comparison)
        from scipy.signal import butter, sosfilt
        low = sosfilt(butter(6, 12000 / (sr / 2), "low", output="sos"), dec, axis=0)
        # restore with defaults
        clean, rep = restore(dec.astype(np.float32), sr,
                             RestoreParams(n_fft=4096, reduction_db=12, floor=0.25, oversubtract=2.0))
        for name, sig in (("decoded", dec), ("lowpass", low), ("restored", clean)):
            prof = band_profile(sig, sr)
            corr = float(np.corrcoef(np.log10(prof_orig + 1e-9), np.log10(prof + 1e-9))[0, 1])
            spec = np.abs(stft(sig[:, 0] if sig.ndim > 1 else sig, 4096, 1024)[0])
            flat = float(np.mean(band_flatness_db(spec, 4096, sr, 6000, 16000)))
            flick = flicker_index(spec, 4096, sr, 1024, 6000, 16000)
            lufs = integrated_loudness(sig, sr)
            ovl = onset_overlap(sig[:, 0] if sig.ndim > 1 else sig, x[:, 0], sr)
            err = sig - orig
            snr = 10 * np.log10(np.sum(orig ** 2) / max(np.sum(err ** 2), 1e-20))
            # artifact-band error reduction: ||restored-orig|| / ||dec-orig|| in 6-16k
            spec_e = np.abs(stft(err[:, 0] if err.ndim > 1 else err, 4096, 1024)[0]) ** 2
            freqs = np.fft.rfftfreq(4096, 1.0 / sr)
            ab = (freqs >= 6000) & (freqs <= 16000)
            results.append((codec, bitrate, name, corr, flat, flick, lufs, ovl, snr,
                            float(np.sqrt(np.mean(spec_e[:, ab])))))

    print(f"{'codec':6s} {'br':4s} {'variant':9s} {'bandCorr':>8s} {'flatness':>9s} {'flicker':>8s} {'LUFS':>7s} {'onsets':>6s} {'SNR':>6s} {'hfErr':>7s}")
    for r in results:
        print(f"{r[0]:6s} {r[1]:4d} {r[2]:9s} {r[3]:8.3f} {r[4]:9.2f} {r[5]:8.4f} {r[6]:7.2f} {r[7]:6.2f} {r[8]:+6.1f} {r[9]:7.4f}")

    by = {(r[0], r[1], r[2]): r for r in results}
    for codec, br in (("opus", 16), ("opus", 32), ("mp3", 64), ("aac", 48)):
        d = by[(codec, br, "decoded")]; c = by[(codec, br, "restored")]; lp = by[(codec, br, "lowpass")]
        hf_red = (1 - c[9] / d[9]) * 100
        lp_red = (1 - lp[9] / d[9]) * 100
        print(f"\n[{codec} {br}k] HF-error reduction: restored {hf_red:+.1f}% | lowpass {lp_red:+.1f}% | "
              f"flicker {d[5]:.3f}->{c[5]:.3f} | onsets {c[7]:.2f} | dLUFS {c[6]-d[6]:+.2f}")
    return results


if __name__ == "__main__":
    main()