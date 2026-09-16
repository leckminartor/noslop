"""Audio I/O: soundfile-native formats + ffmpeg decode fallback (opus/m4a/webm/aac)."""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

FFMPEG = "ffmpeg"


def read_audio(path: str) -> tuple[np.ndarray, int, str]:
    """Read any audio file. Returns (audio (n,ch) float64, sr, format_name)."""
    info = sf.info(path)
    if info.format in ("WAV", "FLAC", "OGG", "WAVEX", "AIFF") or info.format is None:
        pass  # try soundfile first for everything, fall back on failure
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float64")
        return x, int(sr), info.format or "unknown"
    except Exception:
        pass
    # ffmpeg fallback: decode to 32-bit float wav, mono/stereo as-is
    tmp = tempfile.mktemp(suffix=".wav", dir=tempfile.gettempdir())
    try:
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", path,
                        "-c:a", "pcm_f32le", tmp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        x, sr = sf.read(tmp, always_2d=True, dtype="float64")
        return x, int(sr), "ffmpeg-decoded"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_audio(path: str, audio: np.ndarray, sr: int, subtype: str = "PCM_24"):
    """Write WAV/FLAC. audio: (n,) or (n,ch)."""
    x = np.asarray(audio)
    if x.ndim == 1:
        x = x[:, None]
    sf.write(path, x, sr, subtype=subtype)


def encode_opus(src: str, dst: str, bitrate: int = 96):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src,
                    "-c:a", "libopus", "-b:a", f"{bitrate}k", dst], check=True)


def encode_mp3(src: str, dst: str, bitrate: int = 192):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src,
                    "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", dst], check=True)