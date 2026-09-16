# NoSlop — RVQ / Codec Artifact Remover

[English](README.md) · Deutsch

[![CI](https://github.com/leckminartor/noslop/actions/workflows/ci.yml/badge.svg)](https://github.com/leckminartor/noslop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](CHANGELOG.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://paypal.me/klausminator)

**Version 0.1.0** · `python -m app --version` · Versionierung: [SemVer](https://semver.org) —
MAJOR.MINOR.PATCH. Breaking-Änderungen / neue Pipelines → MAJOR/MINOR, Fixes → PATCH.

Reduces AI-music artifacts (Residual Vector Quantization residuals, codec sizzle,
band-copy noise) in audio — with measured, verifiable results instead of placebo EQ.

## Why it works

RVQ/codec artifacts have a spectral structure that differs from musical content:

| Cue | Detektor | trifft |
|-----|----------|--------|
| Flackernde, spärliche HF-Bins | Zeit-Median-Abweichung + Texture-Gate | RVQ "digital sizzle", Opus-Pre-Echo |
| HF-Bett, das der musikalischen Hüllkurve NICHT folgt | Envelope-Incoherence (Kohärenz zur Oktave darunter) | MP3/AAC Band-Copy-Rauschen |
| Metallische Kämme | Residual-Basis-Projektion (12-Atom-Spektrum) | RVQ Residual-Kammern |
| Musikalische Attacks (Hats!) | Spectral-Flux-Onsets (MAD-robust) + Deviation-Shield | schützt Drums vor Falsch-Attribution |

Die Restaurierung ist ein Wiener-artiges Gain-Gate auf dieser Artefakt-Schätzung
(kein EQ, kein Mastering-Trick): transient-sicher, lautness-kompensiert, Phase unangetastet.

**Grenzen (ehrlich gemessen, siehe tests/):**
- MP3/AAC **Band-Copy**-Artefakte (Signal ersetzt statt beschädigt) kann ein
  Magnituden-Gate nicht rekonstruieren — Reduktion des Sizzle-Betts ja, Musik-Recovery nein.
- Bei CELT/Opus (PVQ-Familie, gleiche Codebook-Technik wie RVQ): HF-Fehlerreduktion
  +10–16 % (standard/strong) bei 1.0 Onset-Erhalt und ±0 LUFS.
- Kohärente musikalische HF (Hats) wird mitreduziert (Trade-off, Presets steuern es:
  gentle ≈ −15 %, standard ≈ −20 %, strong ≈ −34 % HF-Anteil @ Opus 32k).


## KI-Verstärkungsstufe (GPU)

Magnitude-Gating kann nur *hinzugefügte* Artefakte entfernen. Verlorenes HF
(MP3/AAC ersetzen das Band) braucht Rekonstruktion — das leisten zwei KI-Komponenten:

1. **Demucs (htdemucs) Stem-Trennung** (GPU): Das auf sauberer Profi-Musik trainierte
   Modell re-rendert jede Spur implizit "clean". Jeder Stem bekommt maßgeschneiderte
   Restaurierung (Drums transient-geschützt, Bass übersprungen, Harmonik am stärksten).
2. **Demucs-Residual-Handling**: Was das Modell NICHT als Musik erkennt, ist überwiegend
   der Sizzle-Bettkomplex — der Residual wird restauriert und mit einstellbarem Anteil
   (residual_alpha) zurückgemischt. `deep`-Modus = 0.0 (vollständige Re-Renderung).
3. **SBR-artige Band-Reparatur**: Inkohärente HF-Bins werden durch Oktav-Kopien
   (aus der Oktave darunter) ersetzt — wie HE-AACs Spectral Band Replication, aber
   invers eingesetzt, um Codec-Zerstörung rückgängig zu machen.
   Gemessen an MP3 64k: HF-Hüllkurven-Korrelation zum Original 0.31 → 0.72.

### Benchmark (tests/bench_ai.py, synthetisches Referenzmaterial)

| Fall | hf_incoh | HF-Korrelation zum Original |
|------|----------|------------------------------|
| Opus 32k decoded | 0.153 | 0.996 |
| Opus 32k dsp | 0.138 | 0.987 |
| Opus 32k **ai** | 0.148 | 0.992 |
| MP3 64k decoded | 0.698 | 0.310 |
| MP3 64k dsp | 0.710 | 0.315 |
| MP3 64k **ai** | **0.547** | **0.722** |

Alle Varianten: dLUFS 0.00 (lautness-neutral). Onset-Erhalt 1.0 in allen DSP-Tests.

## Nutzung

**Web-App** (Drag & Drop, Presets, Before/After-Metriken, Player + Download):
```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --port 8737
# → http://127.0.0.1:8737
```

**CLI:**
```bash
python -m app.cli analyze track.wav          # Codec-Health: LUFS, dBTP, HF-Incoherence, Flags
python -m app.cli restore track.wav --preset standard -o track_restored.wav
python -m app.cli enhance track.wav --mode fast|quality|deep
# presets: gentle | standard | strong
# modes:   fast (DSP) | quality (AI stems) | deep (AI full re-render, am stärksten)
```

**Python API:**
```python
from app.io import read_audio, write_audio
from app.restoration import params_from_preset, restore

x, sr, fmt = read_audio("in.wav")
y, report = restore(x.astype("float32"), sr, params_from_preset("standard"))
```

## Tests & Benchmarks (alle messbar)

```bash
python -m pytest tests/ -q          # 8 Unit-Tests: Loudness vs pyloudnorm (±0.012 LU),
                                    # STFT-Roundtrip 1e-6, True-Peak-Exaktheit, Artefakt-Detektion
python tests/bench_codec.py         # Opus 16/32k, MP3 64k, AAC 48k: HF-Fehlerreduktion vs
                                    # Naive-Lowpass, Onset-Erhalt, Loudness-Neutralität
```

## Architektur

```
app/
  loudness.py    BS.1770-4 Loudness (validiert gegen pyloudnorm) + True-Peak (exakter Kernel)
  analysis.py    STFT/COLA, Residual-Basis, Flatness, Flicker, Envelope-Incoherence, analyze_health
  restoration.py Artefakt-Schätzer + Wiener-Gating + Transienten-Schutz + Presets
  io.py          soundfile + ffmpeg-Fallback (opus/m4a/webm)
  enhance.py     KI-Stufe: Demucs-Stems + SBR-Band-Reparatur + Residual-Blend
  server.py      FastAPI (Jobs, Progress, dsp|quality|deep Modi)
  cli.py         CLI (analyze / restore / enhance)
  web/index.html Single-File-UI
```

## Support

Wenn NoSlop deine KI-Tracks aus der Blechzone gerettet hat, freue ich mich über einen Kaffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://paypal.me/klausminator)

## License

[MIT](LICENSE)
