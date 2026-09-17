# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **AI-echo tail suppressor** (`app/derverb.py`): detects and shortens the bright
  metallic 'AI reverb' hang. Self-calibrating: the mix's own 1-2 kHz anchor band
  defines plausible decay; a causal limiter enforces anchor_slope - margin (backstop
  -35 dB/s) on HF tails. Iterative (2-3 passes converge toward the dry reference).
  Benchmark: synthetic bright tail -5.0 -> -10.7 dB tail/peak (dry -11.9), slope
  -26 -> -64 dB/s; MP3 demo -11.5 -> -13.9 dB. Onsets 1.00, LUFS neutral.
  Available as CLI `enhance --mode derverb` and server mode `derverb`.

## [0.1.1] - 2026-09-16

### Added
- UI shows version number and author byline ("by Klaus Perner (DJ LECK)") in the
  header; version is served from the API (single source of truth) and shown in
  the browser tab title.
- AI-mode availability is surfaced in the UI (disabled when torch/demucs missing).

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
