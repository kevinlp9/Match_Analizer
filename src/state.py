"""Shared global state and configuration — avoids circular imports."""
from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from worker import AnalysisWorker

# ── Config ────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("GEO_API_URL", "")
CLIENT_ID = os.environ.get("M2M_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("M2M_CLIENT_SECRET", "")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "60"))
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")

# ── Worker singleton ──────────────────────────────────────────────────────

worker: Optional[AnalysisWorker] = None
