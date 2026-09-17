"""FastAPI server: upload -> analyze -> restore -> download, with job progress."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import __version__
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from .analysis import analyze_health

if sys.platform == "win32":
    # Windows ProactorEventLoop logs a spurious traceback when a client aborts a
    # stream (e.g. the audio player closing a range request after the job is
    # done). The data was delivered fine; silence the harmless reset.
    import asyncio
    from functools import wraps

    def _silence_connection_reset(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (ConnectionResetError, ConnectionAbortedError):
                pass
        return wrapper

    asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = \
        _silence_connection_reset(
            asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost)
from .io import read_audio, write_audio
from .restoration import PRESETS, params_from_preset, restore

app = FastAPI(title="NoSlop", version=__version__)

WORKDIR = os.path.join(tempfile.gettempdir(), "noslop_jobs")
os.makedirs(WORKDIR, exist_ok=True)

# optional AI stage: loaded lazily on first quality/deep job (needs .venv torch+demucs)
try:
    from .enhance import enhance as _enhance, get_or_load_model
    from .derverb import derverb as _derverb
    AI_AVAILABLE = True
except Exception:
    _enhance = None
    AI_AVAILABLE = False


@dataclass
class Job:
    id: str
    filename: str
    src_path: str
    status: str = "queued"       # queued|analyzing|processing|done|error
    progress: float = 0.0
    message: str = ""
    preset: str = "standard"
    mode: str = "dsp"
    analysis: dict = field(default_factory=dict)
    result: Optional[dict] = None
    out_path: Optional[str] = None
    error: Optional[str] = None


JOBS: dict[str, Job] = {}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "web", "index.html"),
              "r", encoding="utf-8") as f:
        return f.read()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Inline SVG favicon so the browser doesn't log a 404."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
           '<rect width="64" height="64" rx="12" fill="#0e1114"/>'
           '<path d="M18 46 L18 18 L22 18 L42 40 L42 18 L46 18 L46 46 L42 46 '
           'L22 24 L22 46 Z" fill="#37d4a0"/></svg>')
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/presets")
def presets():
    return {"presets": list(PRESETS.keys()), "default": "standard",
            "version": __version__,
            "ai_available": AI_AVAILABLE,
            "modes": ["dsp", "quality", "deep", "dering", "derverb", "polish"]}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...),
                     preset: str = Form("standard"),
                     mode: str = Form("dsp")):
    """mode: dsp (fast) | quality (AI stems, conservative) | deep (AI full re-render)."""
    if preset not in PRESETS:
        raise HTTPException(400, f"preset must be one of {list(PRESETS)}")
    if mode not in ("dsp", "quality", "deep", "dering", "derverb", "polish"):
        raise HTTPException(400, "mode must be dsp|quality|deep|dering|derverb|polish")
    if mode in ("quality", "deep") and not AI_AVAILABLE:
        raise HTTPException(400, "AI mode unavailable: install .venv with torch+demucs")
    jid = uuid.uuid4().hex[:12]
    src = os.path.join(WORKDIR, f"{jid}_src{os.path.splitext(file.filename)[1] or '.wav'}")
    with open(src, "wb") as f:
        f.write(await file.read())
    job = Job(id=jid, filename=file.filename, src_path=src, preset=preset)
    job.mode = mode
    JOBS[jid] = job
    job.status = "analyzing"; job.progress = 0.05; job.message = "reading audio"
    try:
        x, sr, fmt = read_audio(src)
    except Exception as e:
        job.status = "error"; job.error = f"cannot decode: {e}"
        return {"id": jid, "status": "error"}
    job.analysis = analyze_health(x, sr)
    job.message = f"decoded {fmt}, {len(x)/sr:.1f}s @ {sr} Hz"
    job.status = "processing"; job.progress = 0.2
    threading.Thread(target=_process, args=(job,), daemon=True).start()
    return {"id": jid, "status": job.status, "analysis": job.analysis}


def _process(job: Job):
    """Worker: DSP chunked restore, or AI enhance with progress callbacks."""
    try:
        x, sr, fmt = read_audio(job.src_path)
        job.message = f"decoded {fmt}, {len(x)/sr:.1f}s @ {sr} Hz"
        t0 = time.time()
        if job.mode == "dsp":
            params = params_from_preset(job.preset)
            n = len(x)
            chunk = 2 * sr                      # ~2 s chunks
            outs = []
            for i in range(0, n, chunk):
                seg = x[i:i + chunk]
                y, rep = restore(seg.astype(np.float32), sr, params)
                outs.append(y)
                job.progress = 0.2 + 0.7 * min(1.0, (i + len(seg)) / max(n, 1))
            y = np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
            if y.ndim == 1:
                y = y[:, None]
            y = y.astype(np.float64)
            out_sr = sr
            result_rep = {"mode": "dsp"}
        elif job.mode == "polish":
            from .polish import polish
            job.progress = 0.15
            y, out_sr, result_rep = polish(
                x, sr, rounds=1,
                progress_cb=lambda p, label: (setattr(job, "progress", 0.2 + 0.7 * p),
                                              setattr(job, "message", f"polish: {label}")))
            job.progress = 0.95
        elif job.mode == "derverb":
            from .derverb import derverb
            job.progress = 0.3
            y, result_rep = derverb(x, sr, margin_db_s=45.0, passes=2)
            out_sr = sr
            job.progress = 0.95
        elif job.mode == "dering":
            from .dering_ai import dering_ai
            job.progress = 0.15
            y, out_sr, result_rep = dering_ai(
                x, sr, spike_db=5.0, max_db=10.0, passes=2,
                progress_cb=lambda p, label: (setattr(job, "progress", 0.2 + 0.7 * p),
                                              setattr(job, "message", f"deringing: {label}")))
            job.progress = 0.95
        else:
            # AI path: quality/deep, single call (GPU), progress via callback
            if job.mode == "deep":
                kwargs = dict(use_ai=True, sbr=True, residual_alpha=0.0, sbr_thresh=0.3)
            else:
                kwargs = dict(use_ai=True, sbr=True, residual_alpha=0.25, sbr_thresh=0.35)
            job.progress = 0.15
            y, out_sr, result_rep = _enhance(
                x, sr, **kwargs,
                progress_cb=lambda p, label: (setattr(job, "progress", 0.2 + 0.7 * p),
                                              setattr(job, "message", f"AI: {label}")))
            job.progress = 0.95
        before = analyze_health(x, sr)
        after = analyze_health(y.astype(np.float64), out_sr)
        out_path = os.path.join(WORKDIR, f"{job.id}_restored.wav")
        write_audio(out_path, y, out_sr, subtype="PCM_24")
        job.out_path = out_path
        job.result = {"before": before, "after": after,
                      "duration_s": round(len(x) / sr, 2),
                      "processing_s": round(time.time() - t0, 2),
                      "out_sr": out_sr, "mode": job.mode}
        job.status = "done"; job.progress = 1.0; job.message = "ok"
    except Exception as e:
        job.status = "error"; job.error = f"{type(e).__name__}: {e}"


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return {"id": job.id, "status": job.status, "progress": job.progress,
            "message": job.message, "analysis": job.analysis,
            "result": job.result, "error": job.error}


@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    job = JOBS.get(jid)
    if not job or not job.out_path or not os.path.exists(job.out_path):
        raise HTTPException(404, "result not ready")
    name = os.path.splitext(job.filename)[0] + "_restored.wav"
    return FileResponse(job.out_path, filename=name, media_type="audio/wav")