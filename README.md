# NoSlop

[![CI](https://github.com/leckminartor/noslop/actions/workflows/ci.yml/badge.svg)](https://github.com/leckminartor/noslop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-green.svg)](CHANGELOG.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://paypal.me/klausminator)

**Measured, verifiable removal of AI-music codec artifacts (RVQ residuals, codec
sizzle, metallic ringing, band-copy noise) - instead of placebo EQ.**

AI music generators compress their output with Residual Vector Quantization
codecs. The quantization residuals survive into the decoded audio as structured,
repeatable artifacts: metallic "frozen" noise, metallic comb resonances, codec
sizzle, band-copied high frequency. These are baked into the audio at the codec
level - no EQ or mastering trick can remove them. NoSlop detects their spectral
structure and gates them out, with every step backed by measurements.

[Deutsche README](README.de.md)

## How it works

Each artifact type has a spectral signature that differs from musical content:

| Cue | Detector | Targets |
|-----|----------|---------|
| Flickering sparse HF bins | time-median deviation + texture gate | RVQ "digital sizzle", Opus pre-echo |
| HF bed that ignores the music envelope | envelope incoherence (correlation vs. octave below) | MP3/AAC band-copy noise |
| Spectrum-periodic ringing | Demucs-residual-guided notch equalization | metallic/comb artifacts |
| Musical attacks (hats!) | MAD-robust spectral-flux onsets + deviation shielding | protects drums from false attribution |

Restoration is a Wiener-style gain gate on this artifact estimate - not an EQ,
not a mastering trick: transient-safe, loudness-compensated, phase untouched.

## AI enhancement stage (GPU)

Magnitude gating can only *remove* added artifacts. Lost high-frequency
content (MP3/AAC *replace* the band) needs reconstruction. Two AI components
do what is physically impossible for pure DSP:

1. **Demucs (htdemucs) stem separation** - the network, trained on clean studio
   music, implicitly re-renders each stem "clean". Each stem gets individually
   tuned restoration (drums transient-shielded, bass skipped, harmonics strongest).
2. **Demucs-residual-guided de-ringing** - what the model does *not* recognize
   as music is essentially the artifact bed. Its narrow prominent peaks are the
   metallic resonances (e.g. +16 dB at 10.9 kHz on MP3 64k); adaptive notches
   remove them from the mix, with a music guard protecting real musical combs.
3. **AI-echo tail suppressor** - the bright metallic 'AI reverb' hang is measured
   against the mix's own anchor-band decay; a causal limiter shortens tails that
   linger past the plausible decay (benchmark: tail/peak -5.0 -> -10.7 dB,
   dry reference -11.9 dB; slope -26 -> -64 dB/s).
4. **SBR-style band repair** - incoherent HF bins are re-filled with octave-
   replicated magnitudes (Spectral Band Replication, used inversely to undo
   codec destruction).

### Benchmarks (synthetic reference material, measured - not claimed)

| Case | Sizzle metric | HF corr. to original |
|------|--------------|----------------------|
| Opus 32k decoded | 0.153 | 0.996 |
| Opus 32k AI-enhanced | 0.148 | 0.992 |
| MP3 64k decoded | 0.698 | 0.310 |
| MP3 64k AI-enhanced | **0.547** | **0.722** |
| MP3 64k + de-ringing | - | -87% band energy at comb peaks |

All variants are loudness-neutral (dLUFS ~ 0.0) and transient-preserving
(onset overlap 1.0). On already-clean material the pipeline stays conservative
(only 3 gentle notches, kurtosis 844 -> 827).

**Honest limitation:** where a codec *replaces* a band (MP3/AAC), magnitude
gating cannot recover the original music - it reduces the noise bed, and the
SBR stage rebuilds an approximation. Verified numbers, not magic.

## Quick start

**Windows one-click:** double-click `Start NoSlop.bat` - a menu with
web app (AI/DSP), CLI helpers, version info and the test runner.


```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --port 8737
# open http://127.0.0.1:8737  (drag & drop, modes, metrics, player, download)
```

**CLI:**

```bash
python -m app.cli analyze track.wav                # codec health: LUFS, dBTP, flags
python -m app.cli restore track.wav --preset standard
python -m app.cli enhance track.wav --mode quality # GPU modes need the AI deps below
python -m app.cli enhance track.wav --mode dering
```

**AI stage (optional, recommended):**

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install demucs uvicorn fastapi python-multipart
```

Modes: `fast` (DSP, seconds), `quality` (AI stems), `deep` (full re-render),
`dering` (metallic removal), `derverb` (AI-echo tails). Without CUDA the AI modes fall back to CPU.

**Python API:**

```python
from app.io import read_audio
from app.enhance import enhance

x, sr, _ = read_audio("in.wav")
y, out_sr, report = enhance(x, sr, use_ai=True, sbr=True,
                            residual_alpha=0.25, sbr_thresh=0.35)
```

## Tests & benchmarks

```bash
python -m pytest tests/ -q      # 8 unit tests
python tests/bench_codec.py     # codec benchmark vs. naive lowpass
python -m app --version         # 0.1.0
```

Validated internals: BS.1770 loudness within 0.012 LU of pyloudnorm, true-peak
exact to 0.0000 dB, STFT roundtrip error < 1e-6.

## Architecture

```
app/
  loudness.py    BS.1770-4 loudness + true-peak (from scratch, reference-validated)
  analysis.py    STFT/COLA, residual basis, flatness, flicker, envelope incoherence
  restoration.py artifact estimator + Wiener gating + transient guard + presets
  enhance.py     AI stage: Demucs stems + SBR band repair + residual blending
  dering_ai.py   model-guided metallic de-ringing (residual peaks -> notches)
  io.py          soundfile + ffmpeg fallback (opus/m4a/webm)
  server.py      FastAPI jobs (dsp | quality | deep | dering)
  cli.py         CLI (analyze / restore / enhance)
  web/index.html single-file UI
```

## Roadmap

- Iterative de-ringing sweeps and A/B presets per artifact class
- Band-repair network for fully replaced bands (MP3/AAC)
- Realtime monitoring mode for the web app

## Support

If NoSlop saved your AI tracks from the metal zone, a coffee is appreciated:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://paypal.me/klausminator)

## License

[MIT](LICENSE)
