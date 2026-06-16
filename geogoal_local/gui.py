"""Geo-Goal Local GUI — web-based dashboard for offline football tactical analysis.

Launch with:
    python -m geogoal_local.gui
    geogoal gui [--port 8888] [--output-dir ./output]
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.abspath("./output")

log_queue: queue.Queue[str] = queue.Queue()
process_state = {
    "running": False,
    "progress": 0,
    "step": "",
    "error": None,
    "frames": 0,
    "total": 0,
    "elapsed_s": 0.0,
}

_current_video_path: Optional[str] = None

# Live tracking frame buffer
_latest_frame_lock = threading.Lock()
_latest_frame_b64: Optional[str] = None
_latest_frame_info: dict = {}

app = FastAPI(title="Geo-Goal Local GUI")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class VideoSelectRequest(BaseModel):
    path: str


class ProcessRequest(BaseModel):
    video_path: str
    calib_path: str = "output/calib.json"
    model: str = "yolov8n.pt"
    device: str = "cpu"
    dlt_method: str = "cv2"
    frame_skip: int = 2
    max_frames: int = -1
    output_dir: str = "./output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video_meta(path: str) -> dict:
    """Extract video metadata and thumbnail."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps else 0

    thumbnail = ""
    ret, frame = cap.read()
    if ret:
        small = cv2.resize(frame, (320, int(320 * h / w)))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        thumbnail = base64.b64encode(buf).decode("ascii")
    cap.release()

    return {
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "frames": total,
        "duration_s": round(duration, 2),
        "thumbnail": thumbnail,
    }


def _open_file(path: str) -> None:
    """Open a file with the system default application."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return
    if path.endswith(".html"):
        webbrowser.open(f"file://{path}")
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    elif platform.system() == "Linux":
        subprocess.Popen(["xdg-open", path])
    else:
        os.startfile(path)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------


def _run_processing(
    video_path: str,
    calib_path: str,
    model: str,
    device: str,
    method: str,
    frame_skip: int,
    max_frames: int,
    output_dir: str,
) -> None:
    """Run the full pipeline in a background thread."""
    import builtins

    start_time = time.time()
    try:
        process_state.update(
            running=True, progress=0, step="init", error=None, frames=0, total=0
        )
        log_queue.put(f"[process] Starting processing: {video_path}")

        from geogoal_local.video_processor import VideoProcessor

        original_print = builtins.print

        def log_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            log_queue.put(msg)
            if "%" in msg:
                try:
                    for token in msg.replace("%", " ").split():
                        if token.isdigit():
                            pct = int(token)
                            if 0 <= pct <= 100:
                                process_state["progress"] = pct
                                break
                except Exception:
                    pass
            original_print(*args, **kwargs)

        builtins.print = log_print
        try:
            vp = VideoProcessor(
                model_name=model, device=device, output_dir=output_dir
            )
            vp.load_video(video_path)
            process_state["total"] = vp._total_frames

            if calib_path and os.path.exists(calib_path):
                log_queue.put(f"[calibration] Loading from {calib_path}")
                vp.load_calibration(calib_path)
            else:
                log_queue.put("[calibration] Using default homography")
                vp.set_homography(method=method)

            process_state["step"] = "processing"

            # Set up live frame callback
            def _on_frame(annotated_frame, fd, frame_idx):
                global _latest_frame_b64, _latest_frame_info
                h_f, w_f = annotated_frame.shape[:2]
                if w_f > 640:
                    scale = 640 / w_f
                    annotated_frame = cv2.resize(annotated_frame, (640, int(h_f * scale)))
                _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(buf).decode("ascii")
                with _latest_frame_lock:
                    _latest_frame_b64 = b64
                    _latest_frame_info = {
                        "frame_idx": fd.frame_idx,
                        "timestamp_ms": fd.timestamp_ms,
                        "players": len(fd.players),
                        "ball": fd.ball is not None,
                    }
                process_state["frames"] = frame_idx

            vp.frame_callback = _on_frame

            vp.process(
                max_frames=max_frames,
                frame_skip=frame_skip,
                export_json="match_data.json",
            )

            # Analytics
            process_state["step"] = "analytics"
            process_state["progress"] = 90
            log_queue.put("[analytics] Computing statistics...")
            try:
                from geogoal_local.analytics import load_and_compute

                data_file = os.path.join(output_dir, "match_data.json")
                stats_file = os.path.join(output_dir, "stats.json")
                if os.path.exists(data_file):
                    load_and_compute(data_file, stats_file)
            except ImportError:
                log_queue.put("[analytics] Module not available — skipping")

            # Report
            process_state["step"] = "report"
            process_state["progress"] = 95
            log_queue.put("[report] Generating HTML report...")
            try:
                from geogoal_local.report import generate_report

                data_file = os.path.join(output_dir, "match_data.json")
                stats_file = os.path.join(output_dir, "stats.json")
                generate_report(
                    data_path=data_file,
                    stats_path=stats_file,
                    output_path=os.path.join(output_dir, "report.html"),
                )
            except ImportError:
                log_queue.put("[report] Module not available — skipping")

            process_state["progress"] = 100
            process_state["step"] = "done"
            log_queue.put("✅ Processing complete!")
        finally:
            builtins.print = original_print

    except Exception as exc:
        process_state["error"] = str(exc)
        log_queue.put(f"❌ Error: {exc}")
        log_queue.put(traceback.format_exc())
    finally:
        process_state["running"] = False
        process_state["elapsed_s"] = round(time.time() - start_time, 1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML_PAGE


@app.post("/api/video/select")
async def video_select(req: VideoSelectRequest):
    global _current_video_path
    path = os.path.abspath(req.path)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": f"File not found: {path}"}, 400)
    try:
        meta = _video_meta(path)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, 400)
    _current_video_path = path
    return {"ok": True, "path": path, "meta": meta}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    global _current_video_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, file.filename or "uploaded_video.mp4")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    try:
        meta = _video_meta(save_path)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, 400)
    _current_video_path = save_path
    return {"ok": True, "path": save_path, "meta": meta}


@app.get("/api/calibration/status")
async def calibration_status():
    calib_path = os.path.join(OUTPUT_DIR, "calib.json")
    if not os.path.isfile(calib_path):
        return {"exists": False}
    try:
        with open(calib_path) as f:
            data = json.load(f)
        points = len(data.get("src_points", []))
        method = data.get("method", "cv2")
        landmarks = data.get("landmarks", [])
        return {
            "exists": True,
            "points": points,
            "method": method,
            "landmarks": landmarks,
        }
    except Exception:
        return {"exists": False}


@app.post("/api/calibration/open")
async def calibration_open():
    if not _current_video_path:
        return JSONResponse(
            {"ok": False, "error": "Load a video first"}, 400
        )

    def _start_selector():
        try:
            from geogoal_local.selector import run_selector

            run_selector(
                video_path=_current_video_path,
                output_dir=OUTPUT_DIR,
                port=8989,
            )
        except Exception as exc:
            log_queue.put(f"[selector] Error: {exc}")

    t = threading.Thread(target=_start_selector, daemon=True)
    t.start()
    return {"ok": True, "url": "http://localhost:8989"}


@app.post("/api/process/start")
async def process_start(req: ProcessRequest):
    if process_state["running"]:
        return JSONResponse(
            {"ok": False, "error": "Processing already running"}, 409
        )
    process_state["error"] = None
    t = threading.Thread(
        target=_run_processing,
        args=(
            req.video_path,
            req.calib_path,
            req.model,
            req.device,
            req.dlt_method,
            req.frame_skip,
            req.max_frames,
            req.output_dir,
        ),
        daemon=True,
    )
    t.start()
    return {"ok": True, "message": "Processing started"}


@app.get("/api/process/status")
async def process_status():
    return {
        "running": process_state["running"],
        "progress": process_state["progress"],
        "step": process_state["step"],
        "frames_processed": process_state["frames"],
        "total_frames": process_state["total"],
        "elapsed_s": process_state["elapsed_s"],
        "error": process_state["error"],
    }


@app.get("/api/process/events")
async def process_events():
    async def event_stream():
        last_progress = -1
        while True:
            while not log_queue.empty():
                try:
                    msg = log_queue.get_nowait()
                    yield f"data: {json.dumps({'type': 'log', 'message': msg})}\n\n"
                except Exception:
                    break

            if process_state["progress"] != last_progress:
                last_progress = process_state["progress"]
                yield (
                    f"data: {json.dumps({'type': 'progress', 'progress': process_state['progress'], 'step': process_state['step'], 'frames': process_state['frames'], 'total': process_state['total']})}\n\n"
                )

            if process_state.get("error"):
                yield f"data: {json.dumps({'type': 'error', 'message': process_state['error']})}\n\n"
                break

            if not process_state["running"] and process_state["progress"] >= 100:
                yield f"data: {json.dumps({'type': 'complete', 'message': 'Processing complete!'})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/tracking/frame")
async def tracking_frame():
    """Return the latest annotated frame as base64 JPEG + metadata."""
    with _latest_frame_lock:
        if _latest_frame_b64 is None:
            return {"ok": False, "frame": None}
        return {
            "ok": True,
            "frame": _latest_frame_b64,
            "info": _latest_frame_info,
        }


@app.get("/api/tracking/stream")
async def tracking_stream():
    """SSE stream of annotated frames during processing."""
    async def frame_stream():
        last_idx = -1
        while process_state["running"] or process_state["progress"] < 100:
            with _latest_frame_lock:
                if _latest_frame_b64 and _latest_frame_info.get("frame_idx", -1) != last_idx:
                    last_idx = _latest_frame_info.get("frame_idx", -1)
                    yield f"data: {json.dumps({'frame': _latest_frame_b64, 'info': _latest_frame_info})}\n\n"
            await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(frame_stream(), media_type="text/event-stream")


@app.post("/api/report/generate")
async def report_generate():
    try:
        from geogoal_local.report import generate_report

        data_path = os.path.join(OUTPUT_DIR, "match_data.json")
        stats_path = os.path.join(OUTPUT_DIR, "stats.json")
        out_path = os.path.join(OUTPUT_DIR, "report.html")
        generate_report(
            data_path=data_path,
            stats_path=stats_path,
            output_path=out_path,
        )
        return {"ok": True, "path": out_path}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, 500)


@app.post("/api/blender/render")
async def blender_render():
    blender_bin = shutil.which("blender")
    if blender_bin is None:
        return JSONResponse(
            {"ok": False, "error": "Blender not found on PATH"}, 404
        )
    script = Path("geogoal_local/blender/build_scene.py")
    if not script.exists():
        return JSONResponse(
            {"ok": False, "error": f"Blender script not found: {script}"}, 404
        )

    def _render():
        try:
            log_queue.put("[blender] Starting render...")
            cmd = [
                blender_bin,
                "--background",
                "--python",
                str(script),
                "--",
                "--data",
                os.path.join(OUTPUT_DIR, "match_data.json"),
                "--out",
                os.path.join(OUTPUT_DIR, "render.mp4"),
                "--blend",
                os.path.join(OUTPUT_DIR, "scene.blend"),
            ]
            subprocess.run(cmd, check=True)
            log_queue.put("✅ Blender render complete!")
        except Exception as exc:
            log_queue.put(f"❌ Blender error: {exc}")

    t = threading.Thread(target=_render, daemon=True)
    t.start()
    return {"ok": True}


@app.get("/api/outputs/status")
async def outputs_status():
    def _exists(name: str) -> bool:
        return os.path.isfile(os.path.join(OUTPUT_DIR, name))

    return {
        "match_data": _exists("match_data.json"),
        "stats": _exists("stats.json"),
        "report": _exists("report.html"),
        "render": _exists("render.mp4"),
        "scene_blend": _exists("scene.blend"),
        "calib": _exists("calib.json"),
    }


@app.get("/api/outputs/open/{filename:path}")
async def outputs_open(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "File not found"}, 404)
    _open_file(path)
    return {"ok": True}


@app.post("/api/shutdown")
async def shutdown():
    log_queue.put("[server] Shutting down...")
    threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
    return {"ok": True}


# ---------------------------------------------------------------------------
# HTML Page (self-contained — no external resources)
# ---------------------------------------------------------------------------

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Geo-Goal Local — Análisis Táctico</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--card:#161b22;--border:#30363d;
  --green:#3fb950;--green-dark:#2ea043;--blue:#58a6ff;
  --red:#f85149;--yellow:#d29922;
  --text:#e6edf3;--subtle:#8b949e;
  --radius:12px;
}
body{
  font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--text);
  min-height:100vh;padding:0;
}
/* Header */
.header{
  background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
  border-bottom:1px solid var(--border);
  padding:20px 32px;display:flex;align-items:center;justify-content:space-between;
}
.header h1{font-size:1.5rem;font-weight:700;display:flex;align-items:center;gap:10px}
.header h1 .icon{font-size:1.6rem}
.header .subtitle{color:var(--subtle);font-size:.85rem;margin-top:2px}
.header .version{color:var(--subtle);font-size:.8rem;background:var(--card);padding:4px 10px;border-radius:6px;border:1px solid var(--border)}

/* Layout */
.main{max-width:1200px;margin:0 auto;padding:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}

/* Cards */
.card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:24px;transition:border-color .3s,box-shadow .3s;position:relative;overflow:hidden;
}
.card.active{border-color:var(--green);box-shadow:0 0 20px rgba(63,185,80,.1)}
.card.locked{opacity:.5;pointer-events:none}
.card .step-label{
  font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--subtle);margin-bottom:8px;
}
.card h2{font-size:1.1rem;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.card h2 .emoji{font-size:1.3rem}

/* Buttons */
.btn{
  display:inline-flex;align-items:center;gap:6px;
  padding:8px 16px;border-radius:8px;border:1px solid var(--border);
  background:var(--card);color:var(--text);cursor:pointer;
  font-size:.85rem;transition:all .2s;
}
.btn:hover{border-color:var(--blue);color:var(--blue)}
.btn.primary{background:var(--green);border-color:var(--green);color:#fff;font-weight:600}
.btn.primary:hover{background:var(--green-dark);border-color:var(--green-dark)}
.btn.danger{border-color:var(--red);color:var(--red)}
.btn.danger:hover{background:var(--red);color:#fff}
.btn:disabled{opacity:.4;cursor:not-allowed;pointer-events:none}

/* Inputs */
.input-group{display:flex;gap:8px;margin-bottom:12px}
.input-group input[type="text"]{
  flex:1;padding:8px 12px;border-radius:8px;border:1px solid var(--border);
  background:var(--bg);color:var(--text);font-size:.85rem;
}
.input-group input:focus{outline:none;border-color:var(--blue)}
select{
  padding:6px 10px;border-radius:6px;border:1px solid var(--border);
  background:var(--bg);color:var(--text);font-size:.85rem;
}
input[type="number"]{
  width:70px;padding:6px 8px;border-radius:6px;border:1px solid var(--border);
  background:var(--bg);color:var(--text);font-size:.85rem;text-align:center;
}

/* Meta info */
.meta{font-size:.8rem;color:var(--subtle);margin-top:8px;line-height:1.6}
.meta .ok{color:var(--green)}
.meta .err{color:var(--red)}

/* Status badges */
.badge{
  display:inline-flex;align-items:center;gap:4px;
  padding:2px 8px;border-radius:4px;font-size:.75rem;
}
.badge.ok{background:rgba(63,185,80,.15);color:var(--green)}
.badge.no{background:rgba(248,81,73,.15);color:var(--red)}

/* Progress bar */
.progress-wrap{
  margin:16px 0 8px;height:10px;background:var(--bg);border-radius:5px;overflow:hidden;
}
.progress-bar{
  height:100%;width:0%;border-radius:5px;
  background:linear-gradient(90deg,var(--green),var(--green-dark));
  transition:width .5s ease;
}
.progress-text{font-size:.8rem;color:var(--subtle)}

/* Pulse dot */
.pulse{
  display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--green);margin-right:6px;
  animation:pulse-anim 1.5s infinite;
}
@keyframes pulse-anim{
  0%,100%{opacity:1;transform:scale(1)}
  50%{opacity:.4;transform:scale(1.3)}
}

/* Options grid */
.opts{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;margin-bottom:16px}
.opts label{font-size:.8rem;color:var(--subtle);display:flex;flex-direction:column;gap:4px}

/* Upload zone */
.upload-zone{
  border:2px dashed var(--border);border-radius:var(--radius);
  padding:20px;text-align:center;color:var(--subtle);font-size:.85rem;
  transition:border-color .3s,background .3s;cursor:pointer;margin-top:8px;
}
.upload-zone:hover,.upload-zone.dragover{
  border-color:var(--blue);background:rgba(88,166,255,.05);
}

/* Thumbnail */
.thumbnail{max-width:100%;border-radius:8px;margin-top:8px;border:1px solid var(--border)}

/* Mini stat card (tracking panel) */
.mini-stat-card{
  background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;
}

/* Log panel */
.log-panel{
  margin-top:20px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
}
.log-panel .log-header{
  padding:12px 16px;border-bottom:1px solid var(--border);font-size:.85rem;
  display:flex;align-items:center;gap:6px;color:var(--subtle);
}
.log-body{
  height:180px;overflow-y:auto;padding:12px 16px;
  font-family:'SF Mono',SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;
  font-size:.78rem;line-height:1.7;color:var(--subtle);
}
.log-body .entry{white-space:pre-wrap;word-break:break-all}
.log-body .entry.error{color:var(--red)}

/* Results list */
.output-list{list-style:none;margin-top:12px}
.output-list li{
  display:flex;align-items:center;gap:8px;padding:4px 0;font-size:.85rem;
}

/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <h1><span class="icon">⚽</span> GEO-GOAL LOCAL</h1>
    <div class="subtitle">Análisis Táctico de Fútbol — 100% Offline</div>
  </div>
  <div class="version">v1.0</div>
</div>

<div class="main">
  <div class="grid">

    <!-- STEP 1: VIDEO -->
    <div class="card active" id="card-video">
      <div class="step-label">Paso 1</div>
      <h2><span class="emoji"></span> Seleccionar Video</h2>
      <div class="input-group">
        <input type="text" id="video-path" placeholder="Ruta al video (ej: /Users/.../match.mp4)">
        <button class="btn" onclick="loadVideo()">Cargar</button>
      </div>
      <div class="upload-zone" id="upload-zone">
        Arrastra un video aquí o <strong>haz clic</strong> para subir
        <input type="file" id="file-input" accept="video/*" style="display:none">
      </div>
      <div id="video-meta" class="meta" style="display:none"></div>
      <img id="video-thumb" class="thumbnail" style="display:none">
    </div>

    <!-- STEP 2: CALIBRACIÓN -->
    <div class="card locked" id="card-calib">
      <div class="step-label">Paso 2</div>
      <h2><span class="emoji"></span> Calibrar Cancha</h2>
      <p style="font-size:.85rem;color:var(--subtle);margin-bottom:12px">
        Selecciona puntos de referencia en el campo para la homografía.
      </p>
      <button class="btn" id="btn-open-selector" onclick="openSelector()">Abrir Selector</button>
      <button class="btn" onclick="checkCalib()" style="margin-left:8px">🔄 Verificar</button>
      <div id="calib-meta" class="meta" style="margin-top:12px"></div>
    </div>

    <!-- STEP 3: PROCESAMIENTO -->
    <div class="card locked" id="card-process">
      <div class="step-label">Paso 3</div>
      <h2><span class="emoji">⚙</span> Procesar Video</h2>
      <div class="opts">
        <label title="Modelo YOLO para detección">Modelo
          <input type="text" id="opt-model" value="yolov8n.pt" style="width:100%">
        </label>
        <label title="Método de cálculo DLT">Método DLT
          <select id="opt-dlt">
            <option value="cv2" selected>cv2</option>
            <option value="manual">manual</option>
          </select>
        </label>
        <label title="Procesar cada N frames (0=todos)">Frame skip
          <input type="number" id="opt-skip" value="2" min="0" max="30">
        </label>
        <label title="-1 para todos los frames">Max frames
          <input type="number" id="opt-max" value="-1" min="-1">
        </label>
      </div>
      <button class="btn primary" id="btn-process" onclick="startProcess()">▶ PROCESAR</button>
      <div id="process-status" style="margin-top:12px;display:none">
        <div style="display:flex;align-items:center">
          <span class="pulse" id="pulse-dot"></span>
          <span class="progress-text" id="progress-label">Iniciando...</span>
        </div>
        <div class="progress-wrap"><div class="progress-bar" id="progress-bar"></div></div>
      </div>
    </div>

    <!-- STEP 4: RESULTADOS -->
    <div class="card locked" id="card-results">
      <div class="step-label">Paso 4</div>
      <h2><span class="emoji"></span> Resultados</h2>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
        <button class="btn" onclick="openOutput('report.html')">📄 Abrir Report</button>
        <button class="btn" onclick="renderBlender()">🎬 Render 3D</button>
        <button class="btn" onclick="openOutput('render.mp4')">▶ Ver Render</button>
      </div>
      <ul class="output-list" id="output-list"></ul>
    </div>
  </div>

  <!-- Live Tracking Viewer -->
  <div id="tracking-panel" class="card" style="display:none;margin-top:20px;">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border);">
      <span style="font-weight:600;">&#127909; Seguimiento en Vivo</span>
      <div style="display:flex;gap:8px;align-items:center;">
        <span id="tracking-fps" class="badge" style="background:var(--blue);color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;">-- fps</span>
        <span id="tracking-frame-count" style="color:var(--subtle);font-size:13px;">Frame: --</span>
      </div>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;padding:16px;">
      <div style="flex:1;min-width:400px;">
        <div style="position:relative;background:#000;border-radius:8px;overflow:hidden;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;">
          <img id="tracking-img" style="max-width:100%;max-height:100%;object-fit:contain;" alt="Tracking">
          <div id="tracking-waiting" style="position:absolute;color:var(--subtle);font-size:14px;">
            Esperando frames de procesamiento...
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:8px;color:var(--subtle);font-size:12px;">
          <span id="tracking-time">Tiempo: --</span>
          <span id="tracking-players">Jugadores: --</span>
          <span id="tracking-ball">Balon: --</span>
        </div>
      </div>
      <div style="width:220px;display:flex;flex-direction:column;gap:8px;">
        <div class="mini-stat-card">
          <div style="font-size:11px;color:var(--subtle);text-transform:uppercase;letter-spacing:1px;">Equipo Local</div>
          <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
            <span style="width:12px;height:12px;border-radius:50%;background:#dc3545;display:inline-block;"></span>
            <span id="tracking-home-count" style="font-size:20px;font-weight:700;">--</span>
            <span style="color:var(--subtle);font-size:12px;">jugadores</span>
          </div>
        </div>
        <div class="mini-stat-card">
          <div style="font-size:11px;color:var(--subtle);text-transform:uppercase;letter-spacing:1px;">Equipo Visitante</div>
          <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
            <span style="width:12px;height:12px;border-radius:50%;background:#0d6efd;display:inline-block;"></span>
            <span id="tracking-away-count" style="font-size:20px;font-weight:700;">--</span>
            <span style="color:var(--subtle);font-size:12px;">jugadores</span>
          </div>
        </div>
        <div class="mini-stat-card">
          <div style="font-size:11px;color:var(--subtle);text-transform:uppercase;letter-spacing:1px;">Progreso</div>
          <div style="margin-top:6px;">
            <div style="background:var(--border);border-radius:4px;height:8px;overflow:hidden;">
              <div id="tracking-progress-bar" style="height:100%;background:var(--green);width:0%;transition:width 0.3s;"></div>
            </div>
            <div id="tracking-progress-text" style="font-size:12px;color:var(--subtle);margin-top:4px;">0%</div>
          </div>
        </div>
        <div class="mini-stat-card">
          <div style="font-size:11px;color:var(--subtle);text-transform:uppercase;letter-spacing:1px;">Velocidad</div>
          <div id="tracking-speed" style="font-size:20px;font-weight:700;margin-top:4px;">-- fps</div>
        </div>
      </div>
    </div>
  </div>

  <!-- LOG -->
  <div class="log-panel">
    <div class="log-header">LOG</div>
    <div class="log-body" id="log-body"></div>
  </div>
</div>

<script>
// ---------- State ----------
const state = {
  videoPath: null,
  videoMeta: null,
  calibLoaded: false,
  processing: false,
  progress: 0,
  done: false,
  outputs: {}
};

// ---------- Log ----------
function addLog(msg) {
  const el = document.getElementById('log-body');
  const d = document.createElement('div');
  d.className = 'entry' + (msg.includes('❌') || msg.includes('Error') ? ' error' : '');
  const ts = new Date().toLocaleTimeString('es-ES', {hour12:false});
  d.textContent = `[${ts}] ${msg}`;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

// ---------- Step unlocking ----------
function updateSteps() {
  const c1 = document.getElementById('card-video');
  const c2 = document.getElementById('card-calib');
  const c3 = document.getElementById('card-process');
  const c4 = document.getElementById('card-results');
  c1.classList.add('active');
  c1.classList.remove('locked');

  if (state.videoPath) {
    c2.classList.remove('locked');
    c2.classList.add('active');
  } else {
    c2.classList.add('locked');
    c2.classList.remove('active');
  }

  if (state.videoPath && state.calibLoaded) {
    c3.classList.remove('locked');
    c3.classList.add('active');
  } else if (!state.processing) {
    c3.classList.add('locked');
    c3.classList.remove('active');
  }

  if (state.done) {
    c4.classList.remove('locked');
    c4.classList.add('active');
  } else {
    c4.classList.add('locked');
    c4.classList.remove('active');
  }
}

// ---------- Video ----------
async function loadVideo() {
  const p = document.getElementById('video-path').value.trim();
  if (!p) return;
  addLog('Loading video: ' + p);
  try {
    const r = await fetch('/api/video/select', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path: p})
    });
    const d = await r.json();
    if (!d.ok) { addLog('❌ ' + d.error); return; }
    onVideoLoaded(d.path, d.meta);
  } catch(e) { addLog('❌ ' + e.message); }
}

async function uploadVideo(file) {
  addLog('Uploading: ' + file.name);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/video/upload', {method:'POST', body:fd});
    const d = await r.json();
    if (!d.ok) { addLog('❌ ' + d.error); return; }
    onVideoLoaded(d.path, d.meta);
  } catch(e) { addLog('❌ ' + e.message); }
}

function onVideoLoaded(path, meta) {
  state.videoPath = path;
  state.videoMeta = meta;
  const mins = Math.floor(meta.duration_s / 60);
  const secs = Math.round(meta.duration_s % 60);
  document.getElementById('video-meta').style.display = 'block';
  document.getElementById('video-meta').innerHTML =
    `<span class="ok">✅ ${path.split('/').pop()}</span><br>` +
    `${meta.width}×${meta.height}, ${meta.fps}fps — ${mins}:${String(secs).padStart(2,'0')} — ${meta.frames} frames`;
  if (meta.thumbnail) {
    const img = document.getElementById('video-thumb');
    img.src = 'data:image/jpeg;base64,' + meta.thumbnail;
    img.style.display = 'block';
  }
  document.getElementById('video-path').value = path;
  addLog(`Video cargado: ${meta.width}×${meta.height}, ${meta.fps}fps`);
  checkCalib();
  updateSteps();
}

// Upload zone
const uz = document.getElementById('upload-zone');
const fi = document.getElementById('file-input');
uz.addEventListener('click', () => fi.click());
fi.addEventListener('change', (e) => { if(e.target.files[0]) uploadVideo(e.target.files[0]); });
uz.addEventListener('dragover', (e) => { e.preventDefault(); uz.classList.add('dragover'); });
uz.addEventListener('dragleave', () => uz.classList.remove('dragover'));
uz.addEventListener('drop', (e) => {
  e.preventDefault(); uz.classList.remove('dragover');
  if(e.dataTransfer.files[0]) uploadVideo(e.dataTransfer.files[0]);
});

// ---------- Calibration ----------
async function checkCalib() {
  try {
    const r = await fetch('/api/calibration/status');
    const d = await r.json();
    const el = document.getElementById('calib-meta');
    if (d.exists) {
      el.innerHTML = `<span class="ok">✅ calib.json</span> — ${d.points} puntos, método: ${d.method}`;
      state.calibLoaded = true;
      addLog(`Calibración encontrada: ${d.points} puntos`);
    } else {
      el.innerHTML = '<span class="err">❌ No se encontró calib.json</span>';
      state.calibLoaded = false;
    }
    updateSteps();
  } catch(e) { addLog('❌ ' + e.message); }
}

async function openSelector() {
  addLog('Opening calibration selector...');
  try {
    const r = await fetch('/api/calibration/open', {method:'POST'});
    const d = await r.json();
    if (d.ok) {
      addLog('Selector abierto en ' + d.url);
      setTimeout(() => window.open(d.url, '_blank'), 1000);
    } else {
      addLog('❌ ' + d.error);
    }
  } catch(e) { addLog('❌ ' + e.message); }
}

// ---------- Processing ----------
let evtSource = null;

// ---------- Live Tracking Viewer ----------
let trackingInterval = null;
let trackingFrameCount = 0;
let trackingStartTime = 0;

function startTrackingViewer() {
    const panel = document.getElementById('tracking-panel');
    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth' });
    trackingFrameCount = 0;
    trackingStartTime = Date.now();

    trackingInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/tracking/frame');
            const data = await res.json();
            if (data.ok && data.frame) {
                const img = document.getElementById('tracking-img');
                img.src = 'data:image/jpeg;base64,' + data.frame;
                document.getElementById('tracking-waiting').style.display = 'none';

                const info = data.info || {};
                document.getElementById('tracking-time').textContent =
                    'Tiempo: ' + (info.timestamp_ms / 1000).toFixed(1) + 's';
                document.getElementById('tracking-players').textContent =
                    'Jugadores: ' + (info.players || 0);
                document.getElementById('tracking-ball').textContent =
                    'Balon: ' + (info.ball ? 'Si' : 'No');
                document.getElementById('tracking-frame-count').textContent =
                    'Frame: ' + (info.frame_idx || 0);

                trackingFrameCount++;
                const elapsed = (Date.now() - trackingStartTime) / 1000;
                const fps = elapsed > 0 ? (trackingFrameCount / elapsed).toFixed(1) : '--';
                document.getElementById('tracking-fps').textContent = fps + ' fps';
                document.getElementById('tracking-speed').textContent = fps + ' fps';
            }

            document.getElementById('tracking-progress-bar').style.width =
                state.progress + '%';
            document.getElementById('tracking-progress-text').textContent =
                state.progress + '%';

        } catch (e) {
            // Ignore fetch errors during processing
        }

        if (!state.processing && state.progress >= 100) {
            clearInterval(trackingInterval);
            trackingInterval = null;
        }
    }, 150);
}

function stopTrackingViewer() {
    if (trackingInterval) {
        clearInterval(trackingInterval);
        trackingInterval = null;
    }
}

async function startProcess() {
  if (state.processing) return;
  state.processing = true;
  state.done = false;
  state.progress = 0;

  document.getElementById('btn-process').disabled = true;
  document.getElementById('process-status').style.display = 'block';
  updateProgressUI(0, 'Iniciando...');
  addLog('Starting processing pipeline...');

  const body = {
    video_path: state.videoPath,
    calib_path: 'output/calib.json',
    model: document.getElementById('opt-model').value,
    device: 'cpu',
    dlt_method: document.getElementById('opt-dlt').value,
    frame_skip: parseInt(document.getElementById('opt-skip').value) || 0,
    max_frames: parseInt(document.getElementById('opt-max').value) || -1,
    output_dir: './output'
  };

  try {
    const r = await fetch('/api/process/start', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
    });
    const d = await r.json();
    if (!d.ok) { addLog('❌ ' + d.error); resetProcess(); return; }
    connectSSE();
    startTrackingViewer();
  } catch(e) { addLog('❌ ' + e.message); resetProcess(); }
}

function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/process/events');
  evtSource.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'progress') {
      updateProgressUI(d.progress, d.step);
    } else if (d.type === 'log') {
      addLog(d.message);
    } else if (d.type === 'complete') {
      addLog('✅ Processing complete!');
      onProcessComplete();
      evtSource.close();
    } else if (d.type === 'error') {
      addLog('❌ ' + d.message);
      resetProcess();
      evtSource.close();
    }
  };
  evtSource.onerror = () => {
    if (state.processing) {
      setTimeout(connectSSE, 2000);
    }
  };
}

function updateProgressUI(pct, step) {
  state.progress = pct;
  document.getElementById('progress-bar').style.width = pct + '%';
  const labels = {init:'Inicializando',processing:'Procesando',analytics:'Calculando estadísticas',report:'Generando reporte',done:'Completado'};
  document.getElementById('progress-label').textContent = (labels[step]||step||'...') + ' — ' + pct + '%';
}

function onProcessComplete() {
  state.processing = false;
  state.done = true;
  document.getElementById('pulse-dot').style.display = 'none';
  document.getElementById('progress-label').textContent = '✅ Completado — 100%';
  document.getElementById('btn-process').disabled = false;
  stopTrackingViewer();
  refreshOutputs();
  updateSteps();
}

function resetProcess() {
  state.processing = false;
  document.getElementById('btn-process').disabled = false;
  document.getElementById('pulse-dot').style.display = 'none';
  stopTrackingViewer();
}

// ---------- Outputs ----------
async function refreshOutputs() {
  try {
    const r = await fetch('/api/outputs/status');
    const d = await r.json();
    state.outputs = d;
    const ul = document.getElementById('output-list');
    ul.innerHTML = '';
    const files = [
      ['match_data', 'match_data.json'],
      ['stats', 'stats.json'],
      ['report', 'report.html'],
      ['render', 'render.mp4'],
      ['calib', 'calib.json']
    ];
    files.forEach(([key, name]) => {
      const li = document.createElement('li');
      const ok = d[key];
      li.innerHTML = `<span class="badge ${ok?'ok':'no'}">${ok?'✅':'❌'}</span> ${name}`;
      if (ok) {
        const b = document.createElement('button');
        b.className = 'btn';
        b.textContent = 'Abrir';
        b.style.cssText = 'padding:2px 8px;font-size:.75rem;margin-left:auto';
        b.onclick = () => openOutput(name);
        li.appendChild(b);
      }
      ul.appendChild(li);
    });
  } catch(e) { /* ignore */ }
}

async function openOutput(name) {
  try {
    await fetch('/api/outputs/open/' + name);
    addLog('Opening: ' + name);
  } catch(e) { addLog('❌ ' + e.message); }
}

async function renderBlender() {
  addLog('Starting Blender render...');
  try {
    const r = await fetch('/api/blender/render', {method:'POST'});
    const d = await r.json();
    if (!d.ok) addLog('❌ ' + d.error);
    else addLog('Blender render started');
  } catch(e) { addLog('❌ ' + e.message); }
}

// ---------- Init ----------
checkCalib();
refreshOutputs();
addLog('Geo-Goal Local GUI ready — v1.0');
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_gui(port: int = 8888, output_dir: str = "./output") -> None:
    """Start the GUI server and open the browser."""
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.abspath(output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    url = f"http://localhost:{port}"
    print(f"[gui] Starting Geo-Goal Local GUI on {url}")
    print(f"[gui] Output directory: {OUTPUT_DIR}")

    # Open browser after a short delay
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    run_gui()
