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

## [0.2.1] - 2026-09-17

### Changed
- **Presets now steer all modes**: the gentle/standard/strong dial scales the
  intensity of every stage (dering notch depth, derverb decay margin, quality/
  deep residual blend, polish rounds) via the central `PRESET_PROFILES` table -
  previously the presets only affected the dsp mode and the AI modes ran with
  hardcoded defaults.
- `enhance` CLI gains `--preset` for the AI modes.
- UI polish: author byline moved to a page footer; inline favicon (no 404s).

### Fixed
- Windows ProactorEventLoop reset traceback when clients abort audio streams
  (harmless, but it flooded the server log).
- release-script CITATION regex anchored (was clobbering `cff-version`).

## [0.2.0] - 2026-09-17## [0.2.0] - 2026-09-17

### Added
- **Combined polish pass** (`app/polish.py`, mode `polish`): metallic de-ringing
  and AI-echo tail suppression in one pipeline, alternating rounds. On the
  combined worst case (combs + bright tail): tail/peak -5.1 -> -9.7 dB (dry
  -11.9), HF slope -26 -> -54 dB/s, kurtosis 239 -> 623. On real MP3: tail
  -11.5 -> -12.7 dB, slope -68 -> -90 dB/s, kurtosis 799 -> 790, LUFS neutral.
- **Dry-skip safety gate**: polish skips the tail limiter on already-dry
  material (clean test: SNR +17 dB, HF within +7%, near-transparent).
- `polish` is the new default mode in the web UI.

### Fixed
- Favicon served inline (`/favicon.ico` + `<link rel=icon>`) - no more 404s.
- Windows launcher reads the version dynamically; `scripts/release.py` bumps
  the version everywhere (READMEs, UI title, CITATION, CHANGELOG) in one run.

## [0.1.1] - 2026-09-16## [0.1.1] - 2026-09-16

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
