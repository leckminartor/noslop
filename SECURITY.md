# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |

## Reporting a vulnerability

NoSlop is a local audio processing tool: it reads audio files, processes them
in-process, and writes results to local disk or serves them from a local
FastAPI instance. It performs no telemetry and makes no network calls during
processing (model weights are downloaded from Hugging Face at first run of an
AI mode).

If you find a security-relevant issue (e.g. path traversal in file handling,
unsafe deserialization, remote code execution via crafted audio files), please
open a **private** report:

- Use GitHub's "Report a vulnerability" (Security Advisories) on this repo, or
- contact the maintainer directly.

Please do not open public issues for unpublished vulnerabilities.

Include: affected version, a minimal reproduction (audio sample + steps), and
the impact you observed. You can expect a response within 7 days.
