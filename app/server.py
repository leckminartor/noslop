"""FastAPI server: upload -> analyze -> restore -> download, with job progress."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf

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
from .restoration import PRESETS, params_from_preset, profile_for, restore

app = FastAPI(title="NoSlop", version=__version__)

WORKDIR = os.path.join(tempfile.gettempdir(), "noslop_jobs")
os.makedirs(WORKDIR, exist_ok=True)

# persistent source library: uploaded files survive across jobs/restarts
LIBDIR = os.path.join(os.path.expanduser("~"), ".noslop", "library")
os.makedirs(LIBDIR, exist_ok=True)
LIB_INDEX = os.path.join(LIBDIR, "library.json")


def _library_load() -> dict:
    """Read the library index from disk."""
    try:
        with open(LIB_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _library_save(lib: dict) -> None:
    with open(LIB_INDEX, "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=1)


def library_add(path: str, orig_name: str, sid: str) -> dict:
    """Register a file (already copied into LIBDIR) in the library index."""
    info = sf.info(path)
    entry = {
        "id": sid,
        "filename": os.path.basename(orig_name) or f"source_{sid}",
        "path": path,
        "sr": info.samplerate,
        "duration_s": round(info.frames / info.samplerate, 2),
        "format": info.format,
        "added": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    lib = _library_load()
    lib[sid] = entry
    _library_save(lib)
    return entry


def library_get(sid: str) -> Optional[dict]:
    lib = _library_load()
    e = lib.get(sid)
    if e and os.path.exists(e.get("path", "")):
        return e
    return None

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


# ---------------- source library (reuse uploads across jobs) ----------------

@app.get("/api/library")
def library_list():
    lib = _library_load()
    items = [e for e in lib.values() if os.path.exists(e.get("path", ""))]
    items.sort(key=lambda e: e.get("added", ""), reverse=True)
    return {"items": items}


@app.post("/api/library")
async def library_upload(file: UploadFile = File(...)):
    """Store a source file in the persistent library (no processing)."""
    sid = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1] or ".wav"
    path = os.path.join(LIBDIR, f"{sid}{ext}")
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        entry = library_add(path, file.filename, sid)
    except Exception as e:
        os.remove(path)
        raise HTTPException(400, f"cannot decode: {e}")
    return entry


@app.delete("/api/library/{sid}")
def library_delete(sid: str):
    lib = _library_load()
    e = lib.pop(sid, None)
    if not e:
        raise HTTPException(404, "not found")
    _library_save(lib)
    try:
        os.remove(e["path"])
    except OSError:
        pass
    return {"ok": True}


@app.get("/api/library/{sid}/file")
def library_file(sid: str):
    e = library_get(sid)
    if not e:
        raise HTTPException(404, "not found")
    return FileResponse(e["path"], filename=e["filename"],
                        media_type="application/octet-stream")


@app.post("/api/jobs/source/{sid}")
async def create_job_from_source(sid: str,
                                 preset: str = Form("standard"),
                                 mode: str = Form("dsp")):
    """Run a job on a library source WITHOUT re-uploading the file."""
    e = library_get(sid)
    if not e:
        raise HTTPException(404, "source not found in library")
    if preset not in PRESETS:
        raise HTTPException(400, f"preset must be one of {list(PRESETS)}")
    if mode not in ("dsp", "quality", "deep", "dering", "derverb", "polish"):
        raise HTTPException(400, "mode must be dsp|quality|deep|dering|derverb|polish")
    if mode in ("quality", "deep") and not AI_AVAILABLE:
        raise HTTPException(400, "AI mode unavailable: install .venv with torch+demucs")
    jid = uuid.uuid4().hex[:12]
    job = Job(id=jid, filename=e["filename"], src_path=e["path"], preset=preset)
    job.mode = mode
    JOBS[jid] = job
    return _start_job(job)


# ---------------- classic single-shot upload job ----------------

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...),
                     preset: str = Form("standard"),
                     mode: str = Form("dsp")):
    """mode: dsp (fast) | quality (AI stems, conservative) | deep (AI full re-render).
    Every upload is ALSO stored in the persistent library, so the same source can be
    re-processed with other modes/presets without re-uploading."""
    if preset not in PRESETS:
        raise HTTPException(400, f"preset must be one of {list(PRESETS)}")
    if mode not in ("dsp", "quality", "deep", "dering", "derverb", "polish"):
        raise HTTPException(400, "mode must be dsp|quality|deep|dering|derverb|polish")
    if mode in ("quality", "deep") and not AI_AVAILABLE:
        raise HTTPException(400, "AI mode unavailable: install .venv with torch+demucs")
    jid = uuid.uuid4().hex[:12]
    # store the upload straight into the persistent library and process from there
    sid = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1] or ".wav"
    src = os.path.join(LIBDIR, f"{sid}{ext}")
    with open(src, "wb") as f:
        f.write(await file.read())
    try:
        library_add(src, file.filename, sid)
    except Exception:
        pass  # library is best-effort; the job itself validates decodability below
    job = Job(id=jid, filename=file.filename, src_path=src, preset=preset)
    job.mode = mode
    JOBS[jid] = job
    return _start_job(job)


def _start_job(job: "Job") -> dict:
    """Shared job kick-off: analyze source, then process in a worker thread."""
    job.status = "analyzing"; job.progress = 0.05; job.message = "reading audio"
    try:
        x, sr, fmt = read_audio(job.src_path)
    except Exception as e:
        job.status = "error"; job.error = f"cannot decode: {e}"
        return {"id": job.id, "status": "error"}
    job.analysis = analyze_health(x, sr)
    job.message = f"decoded {fmt}, {len(x)/sr:.1f}s @ {sr} Hz"
    job.status = "processing"; job.progress = 0.2
    threading.Thread(target=_process, args=(job,), daemon=True).start()
    return {"id": job.id, "status": job.status, "analysis": job.analysis}


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
            prof = profile_for(job.preset, "polish")
            job.progress = 0.15
            y, out_sr, result_rep = polish(
                x, sr, rounds=prof.get("rounds", 1),
                dering_spike_db=prof.get("dering_spike_db", 5.0),
                dering_max_db=prof.get("dering_max_db", 8.0),
                derverb_margin_db_s=prof.get("derverb_margin_db_s", 45.0),
                progress_cb=lambda p, label: (setattr(job, "progress", 0.2 + 0.7 * p),
                                              setattr(job, "message", f"polish: {label}")))
            job.progress = 0.95
        elif job.mode == "derverb":
            from .derverb import derverb
            prof_v = profile_for(job.preset, "derverb")
            job.progress = 0.3
            y, result_rep = derverb(x, sr, margin_db_s=prof_v.get("margin_db_s", 45.0),
                                    passes=prof_v.get("passes", 2))
            out_sr = sr
            job.progress = 0.95
        elif job.mode == "dering":
            from .dering_ai import dering_ai
            prof_d = profile_for(job.preset, "dering")
            job.progress = 0.15
            y, out_sr, result_rep = dering_ai(
                x, sr, spike_db=prof_d["spike_db"],
                max_db=prof_d["max_db"], passes=prof_d["passes"],
                progress_cb=lambda p, label: (setattr(job, "progress", 0.2 + 0.7 * p),
                                              setattr(job, "message", f"deringing: {label}")))
            job.progress = 0.95
        else:
            # AI path: quality/deep, single call (GPU), progress via callback
            prof_q = profile_for(job.preset, job.mode if job.mode in ("quality", "deep") else "quality")
            kwargs = dict(use_ai=True, sbr=True,
                          residual_alpha=prof_q.get("residual_alpha", 0.25),
                          sbr_thresh=prof_q.get("sbr_thresh", 0.35))
            if job.mode == "deep":
                kwargs["residual_alpha"] = 0.0
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


def _wait_and_open_browser(host: str, port: int, timeout_s: float = 30.0) -> None:
    """Open the web UI only AFTER the server answers its first HTTP request.

    Runs in a daemon thread; polled from a uvicorn startup handler so the
    .bat launcher never shows a connection error from a too-early `start URL`.
    """
    import threading
    import urllib.request

    url = f"http://{host}:{port}/api/presets"

    def _poll():
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.3)
        else:
            return  # server never became ready - don't open a dead tab
        try:
            import webbrowser
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    threading.Thread(target=_poll, daemon=True).start()


@app.on_event("startup")
def _open_browser_when_ready():
    # Set NOSLOP_AUTOOPEN=0 to disable (e.g. when started by scripts/tests).
    if os.environ.get("NOSLOP_AUTOOPEN", "1") != "0":
        _wait_and_open_browser("127.0.0.1", 8737)