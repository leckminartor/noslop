# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-16

### Added
- **DSP restoration chain** (`app/restoration.py`): spectral gating driven by
  residual-basis analysis - time-median deviation + texture gate, residual-basis
  shape match, MAD-robust transient detection with deviation shielding.
- **Envelope incoherence detector** (`app/analysis.py`): per-bin sliding
  correlation against the octave-below musical envelope; stable codec band-copy
  noise beds measure 0.05-0.12 coherence vs. 0.97-0.99 for musical HF.
- **AI enhancement stage** (`app/enhance.py`, GPU): Demucs (htdemucs) stem
  separation with per-stem tuned restoration, residual re-blending
  (`residual_alpha`), and SBR-style band repair (octave-replicated magnitudes
  for incoherent HF bins).
- **Metallic/comb de-ringing** (`app/dering_ai.py`, GPU): Demucs-residual-guided
  adaptive notch equalization with prominence-based peak detection and a music
  guard (musical combs are detected via the stems and protected at 35% depth).
- **BS.1770-4 loudness meter** (`app/loudness.py`): implemented from scratch
  (bilinear K-weighting + gated integration), validated against pyloudnorm
  within 0.012 LU; true-peak estimation exact to 0.0000 dB on test tones.
- **Web app** (`app/server.py`, `app/web/index.html`): FastAPI job pipeline with
  progress reporting, drag & drop UI, before/after codec-health metrics,
  audio player and download. Modes: `dsp` (fast), `quality` (AI stems),
  `deep` (full re-render), `dering` (metallic removal).
- **CLI** (`app/cli.py`): `analyze`, `restore`, `enhance` with presets
  (gentle/standard/strong) and modes (fast/quality/deep/dering).
- **Tests & benchmarks**: 8 unit tests (loudness vs. pyloudnorm, STFT
  roundtrip, true-peak exactness, artifact detection, transient preservation,
  STFT roundtrip 1e-6); codec benchmark (Opus 16/32k, MP3 64k, AAC 48k) with
  HF-error reduction vs. naive lowpass and onset-preservation checks.
