# Contributing to NoSlop

Thank you for considering a contribution! This document explains how to set up
the project, run the tests, and submit changes.

## Development setup

```bash
git clone https://github.com/leckminartor/noslop.git
cd noslop

# base environment (DSP + web UI, CPU only)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS

# AI stage (GPU, optional but recommended)
.venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install demucs
```

The AI modes (`quality`, `deep`, `dering`) automatically fall back to CPU if no
CUDA device is available (much slower, but functional).

## Running the tests

```bash
python -m pytest tests/ -q          # unit tests (no GPU required)
python tests/bench_codec.py         # codec benchmark (needs ffmpeg on PATH)
```

All tests must pass before a PR is merged. Benchmarks are informational but
regressions in HF-error reduction or onset preservation should be discussed in
the PR.

## Code style

- Plain, typed Python (`from __future__ import annotations`), no heavy magic.
- Every DSP stage must be **verifiable**: add a metric or a test, not just
  "sounds better". Loudness neutrality (dLUFS ~ 0) and transient preservation
  (onset overlap ~ 1.0) are the two standing acceptance criteria.
- Keep processing loudness-neutral: any gain change must be compensated and
  reported in the processing report.

## Pull requests

1. Fork / branch from `main`.
2. Make the change, add tests if the change is measurable.
3. Run the test suite and the codec benchmark.
4. Describe **what changed and which metric proves it** in the PR description.

## Reporting issues

Please include: input format (sample rate, codec/bitrate if known), the
processing mode used, the processing report JSON (the app and CLI both emit
one), and an audio excerpt if possible.

## Releasing a new version

The version lives in **one place**: `app/__init__.py` (`__version__`).
Everything else is updated by the release script:

```bash
python scripts/release.py 0.2.0        # bump + update READMEs/UI/CITATION + CHANGELOG section
# -> then edit CHANGELOG.md and fill in the release notes
python scripts/release.py 0.2.0 --push  # alternative: commit + tag + push in one step
```

Pushing a `v*` tag triggers the release workflow, which runs the tests and
publishes a GitHub release automatically.

Files kept in sync by the script: `app/__init__.py`, `README.md`, `README.de.md`,
`app/web/index.html` (tab title), `CITATION.cff`, `CHANGELOG.md`. The Windows
launcher (`Start NoSlop.bat`) reads the version dynamically at runtime - nothing
to bump there.
