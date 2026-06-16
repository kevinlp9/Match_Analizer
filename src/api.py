"""Geo-Goal AI Service API — FastAPI application with background worker."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional, List

import os

# Silenciar warnings de NNPACK en CPUs sin soporte (hardware de nube sin AVX2).
# PyTorch usa CPU estándar como fallback automáticamente — sin impacto en resultados.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Track process start time for uptime calculation
_START_TIME = time.time()

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from pydantic import BaseModel

from api_client import APIClient
from m2m_client import M2MClient
from state import (
    API_BASE,
    CLIENT_ID,
    CLIENT_SECRET,
    POLL_INTERVAL,
    MODEL_NAME,
    DEVICE,
)
from worker import AnalysisWorker

# ---------------------------------------------------------------------------
# Singleton detector — cargado UNA vez al arrancar, reutilizado en preview
# ---------------------------------------------------------------------------
_detector_singleton = None

def get_detector():
    """Retorna el ObjectDetector singleton (evita recargar YOLO en cada request)."""
    global _detector_singleton
    if _detector_singleton is None:
        from video_processor import ObjectDetector
        _detector_singleton = ObjectDetector(model_name=MODEL_NAME, device=DEVICE)
    return _detector_singleton


def get_api_client() -> APIClient:
    m2m = M2MClient(API_BASE, CLIENT_ID, CLIENT_SECRET)
    return APIClient(API_BASE, m2m)


worker_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_task
    import state
    print("[api] Starting AI service...")
    api_client = get_api_client()
    state.worker = AnalysisWorker(
        api_client,
        poll_interval=POLL_INTERVAL,
        model_name=MODEL_NAME,
        device=DEVICE,
    )
    worker_task = asyncio.create_task(state.worker.start())
    print(f"[api] Worker started (poll every {POLL_INTERVAL}s)")
    # Pre-cargar el detector YOLO para que el primer /analysis/preview no sufra el delay
    try:
        get_detector()
        print(f"[api] YOLO detector pre-loaded (model={MODEL_NAME}, device={DEVICE})")
    except Exception as e:
        print(f"[api] Warning: could not pre-load detector: {e}")
    yield
    print("[api] Shutting down...")
    if state.worker:
        state.worker.stop()
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    print("[api] Stopped.")


app = FastAPI(
    title="Geo-Goal AI Service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: permite que el frontend llame directo al AI service (para health checks
# y, en deploy, podría llamar también /analysis/preview si decides exponerlo).
# Orígenes configurables via env, default permisivo para dev local.
_cors_origins_env = os.environ.get("AI_CORS_ORIGINS", "")
if _cors_origins_env.strip():
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    _cors_origins = [
        "https://geo-goal.onrender.com",
        "https://geo-goal-1.onrender.com"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)

security = HTTPBearer()


async def require_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{API_BASE}/auth/user",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError:
            raise HTTPException(status_code=401, detail="Token inválido o expirado")
        except Exception:
            raise HTTPException(status_code=503, detail="No se puede conectar con el backend")

    user: dict = r.json()
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden acceder")

    return user


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SystemMetrics(BaseModel):
    # CPU
    cpu_percent: float
    cpu_count: int
    cpu_freq_mhz: Optional[float] = None
    # RAM
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    # Disk (root)
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    # GPU (if available)
    gpu_name: Optional[str] = None
    gpu_vram_used_mb: Optional[float] = None
    gpu_vram_total_mb: Optional[float] = None
    gpu_vram_percent: Optional[float] = None
    # This process
    proc_cpu_percent: float
    proc_ram_mb: float
    # Uptime
    uptime_seconds: float


def _collect_system_metrics() -> SystemMetrics:
    """Collect system performance metrics. Gracefully handles missing psutil/GPU."""
    try:
        import psutil
        proc = psutil.Process()

        cpu_pct  = psutil.cpu_percent(interval=0.1)
        cpu_cnt  = psutil.cpu_count(logical=True) or 1
        cpu_freq = None
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_freq = round(freq.current, 1)
        except Exception:
            pass

        mem  = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        ram_used  = round(mem.used  / 1e9, 2)
        ram_total = round(mem.total / 1e9, 2)
        disk_used  = round(disk.used  / 1e9, 2)
        disk_total = round(disk.total / 1e9, 2)

        proc_cpu = proc.cpu_percent(interval=0.05)
        proc_ram = round(proc.memory_info().rss / 1e6, 1)
    except Exception:
        cpu_pct = cpu_cnt = cpu_freq = 0
        ram_used = ram_total = disk_used = disk_total = 0.0
        proc_cpu = proc_ram = 0.0
        mem = type("_", (), {"percent": 0})()
        disk = type("_", (), {"percent": 0})()

    # GPU via torch.cuda (already loaded for YOLO)
    gpu_name = gpu_vram_used = gpu_vram_total = gpu_vram_pct = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name      = torch.cuda.get_device_name(0)
            vram_used     = torch.cuda.memory_allocated(0) / 1e6          # MB
            vram_reserved = torch.cuda.memory_reserved(0)  / 1e6          # MB
            vram_total    = torch.cuda.get_device_properties(0).total_memory / 1e6
            gpu_vram_used  = round(vram_reserved, 1)   # reserved is more representative
            gpu_vram_total = round(vram_total, 1)
            if vram_total > 0:
                gpu_vram_pct = round(vram_reserved / vram_total * 100, 1)
    except Exception:
        pass

    return SystemMetrics(
        cpu_percent=cpu_pct,
        cpu_count=cpu_cnt,
        cpu_freq_mhz=cpu_freq,
        ram_used_gb=ram_used,
        ram_total_gb=ram_total,
        ram_percent=getattr(mem, "percent", 0),
        disk_used_gb=disk_used,
        disk_total_gb=disk_total,
        disk_percent=getattr(disk, "percent", 0),
        gpu_name=gpu_name,
        gpu_vram_used_mb=gpu_vram_used,
        gpu_vram_total_mb=gpu_vram_total,
        gpu_vram_percent=gpu_vram_pct,
        proc_cpu_percent=proc_cpu,
        proc_ram_mb=proc_ram,
        uptime_seconds=round(time.time() - _START_TIME, 0),
    )


class HealthResponse(BaseModel):
    status: str
    worker_running: bool
    current_job: Optional[int] = None
    poll_interval: int
    device: str
    system: Optional[SystemMetrics] = None


class ProcessJobRequest(BaseModel):
    match_id: int
    job_id: int
    video_url: str
    src_pts: list


class JobStatus(BaseModel):
    worker_running: bool
    current_job: Optional[int] = None


# ---------------------------------------------------------------------------
# Preview models
# ---------------------------------------------------------------------------

class PreviewSrcPt(BaseModel):
    x: float
    y: float

class PreviewRequest(BaseModel):
    frame_base64: str                              # "data:image/jpeg;base64,..." o solo base64
    src_pts: Optional[List[PreviewSrcPt]] = None   # 4 esquinas en coords pixel
    detect_pitch: bool = False                     # auto-detectar (no implementado en v1)

class PreviewDetection(BaseModel):
    tracker_id: int
    x_m: float
    y_m: float
    px: float
    py: float
    team: str                                      # "home"|"away"|"referee"|"unknown"
    bbox: List[float]                              # [x1,y1,x2,y2]

class PreviewBall(BaseModel):
    x_m: float
    y_m: float
    px: float
    py: float

class PreviewResponse(BaseModel):
    homography_ok: bool
    src_pts: List[PreviewSrcPt]
    players: List[PreviewDetection]
    ball: Optional[PreviewBall] = None
    pitch: dict
    frame_dims: dict
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Dashboard routes (admin web UI)
# ---------------------------------------------------------------------------

from dashboard import router as dashboard_router

app.include_router(dashboard_router)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check público — no requiere auth.
    Devuelve estado del worker + métricas de sistema (CPU, RAM, GPU, disco, uptime).
    """
    import state
    w = state.worker
    return HealthResponse(
        status="ok",
        worker_running=w.is_running if w else False,
        current_job=w.current_job if w else None,
        poll_interval=POLL_INTERVAL,
        device=DEVICE,
        system=_collect_system_metrics(),
    )


@app.get("/jobs", response_model=JobStatus)
async def get_jobs_status(_: dict = Depends(require_admin)):
    import state
    w = state.worker
    return JobStatus(
        worker_running=w.is_running if w else False,
        current_job=w.current_job if w else None,
    )


@app.post("/jobs/process")
async def process_job_manually(req: ProcessJobRequest, _: dict = Depends(require_admin)):
    """Manually trigger processing of a specific job."""
    import state
    w = state.worker
    if not w:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    job = {
        "jobId": req.job_id,
        "matchId": req.match_id,
        "videoSupabaseUrl": req.video_url,
        "srcPts": req.src_pts,
    }
    asyncio.create_task(w.process_job(job))
    return {"message": "Job processing started", "jobId": req.job_id}


@app.post("/jobs/poll")
async def force_poll(_: dict = Depends(require_admin)):
    """Force an immediate poll for pending jobs."""
    import state
    w = state.worker
    if not w:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    pending = get_api_client().get_pending_analysis()
    if pending:
        asyncio.create_task(w.process_job(pending[0]))
        return {"message": "Job found and processing started", "jobId": pending[0]["jobId"]}

    return {"message": "No pending jobs found"}


# ---------------------------------------------------------------------------
# Preview endpoint (sync, sin tocar DB)
# ---------------------------------------------------------------------------

@app.post("/analysis/preview", response_model=PreviewResponse)
async def analysis_preview(req: PreviewRequest, _: dict = Depends(require_admin)):
    """Detecta jugadores en 1 frame y proyecta a cancha 2D. Sin tocar DB.

    Correcciones aplicadas vs plan original:
    - Usa el singleton get_detector() → YOLO no se recarga en cada request (~0ms vs ~2s).
    - El frame se reescala a max 1280px antes de inferencia para mayor velocidad.
    """
    import base64
    import cv2
    import numpy as np
    from video_processor import (
        TeamClassifier,
        HomographyCalculator,
        PerspectiveTransformer,
        DEFAULT_DST_PTS,
        PITCH_LENGTH_M,
        PITCH_WIDTH_M,
    )

    # 1. Decode base64 → np.ndarray (BGR)
    try:
        b64 = req.frame_base64.split(",")[-1]   # acepta data URL o raw base64
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode retornó None — formato no soportado")
    except Exception as e:
        return PreviewResponse(
            homography_ok=False, src_pts=[], players=[], ball=None,
            pitch={"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
            frame_dims={"width": 0, "height": 0},
            error=f"Frame inválido: {e}",
        )

    h, w = frame.shape[:2]

    # 2. Necesitamos exactamente 4 esquinas (auto-detect no implementado en v1)
    if not req.src_pts or len(req.src_pts) != 4:
        return PreviewResponse(
            homography_ok=False, src_pts=[], players=[], ball=None,
            pitch={"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
            frame_dims={"width": w, "height": h},
            error="Se requieren exactamente 4 puntos en src_pts",
        )

    src_pts_np = np.array([[p.x, p.y] for p in req.src_pts], dtype=np.float32)

    # 3. Homografía
    try:
        H = HomographyCalculator().compute(src_pts_np, DEFAULT_DST_PTS, method="cv2")
        transformer = PerspectiveTransformer(H)
    except Exception as e:
        return PreviewResponse(
            homography_ok=False, src_pts=req.src_pts, players=[], ball=None,
            pitch={"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
            frame_dims={"width": w, "height": h},
            error=f"Homografía falló: {e}",
        )

    # 4. Detección YOLO — SINGLETON (no recarga modelo)
    # Resize a max 1280px para acelerar inferencia en frames de alta resolución
    MAX_DIM = 1280
    scale = 1.0
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame_inf = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        frame_inf = frame

    detector = get_detector()
    detections = detector.detect(frame_inf, conf=0.3)

    if len(detections) == 0:
        return PreviewResponse(
            homography_ok=True, src_pts=req.src_pts, players=[], ball=None,
            pitch={"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
            frame_dims={"width": w, "height": h},
            error="No se detectaron jugadores en este frame. Prueba con un frame del minuto 1-5.",
        )

    # 5. K-means para clasificación de equipos (solo si hay ≥3 detecciones)
    teams = (
        TeamClassifier().fit_predict(frame_inf, detections)
        if len(detections) >= 3
        else ["unknown"] * len(detections)
    )

    # 6. Proyectar cada detección a coordenadas de cancha en metros
    players: List[PreviewDetection] = []
    ball: Optional[PreviewBall] = None

    for i in range(len(detections)):
        xyxy_inf = detections.xyxy[i]
        cls = int(detections.class_id[i]) if detections.class_id is not None else 0

        # Revertir escala → coordenadas del frame original para homografía correcta
        xyxy_orig = xyxy_inf / scale if scale != 1.0 else xyxy_inf

        bp = transformer.player_base_point(xyxy_orig)
        try:
            x_m, y_m = transformer.pixel_to_pitch(*bp)
        except Exception:
            continue

        if cls == 32:   # sports ball (COCO class 32)
            ball = PreviewBall(
                x_m=round(x_m, 2), y_m=round(y_m, 2),
                px=round(bp[0], 1), py=round(bp[1], 1),
            )
            continue

        players.append(PreviewDetection(
            tracker_id=i,
            x_m=round(x_m, 2), y_m=round(y_m, 2),
            px=round(bp[0], 1), py=round(bp[1], 1),
            team=teams[i] if i < len(teams) else "unknown",
            bbox=xyxy_orig.tolist(),
        ))

    return PreviewResponse(
        homography_ok=True,
        src_pts=req.src_pts,
        players=players,
        ball=ball,
        pitch={"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
        frame_dims={"width": w, "height": h},
    )

