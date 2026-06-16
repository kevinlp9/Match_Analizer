"""API client that uses M2MClient for auth — interacts with Geo-Goal backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import requests

from m2m_client import M2MClient


class APIClient:
    def __init__(self, api_base: str, m2m: M2MClient):
        self.api_base = api_base.rstrip("/")
        self.m2m = m2m

    def _headers(self) -> Dict[str, str]:
        token = self.m2m.get_token()
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_match_analytics(self, match_id: int) -> Any:
        url = f"{self.api_base}/public/matches/{match_id}/analytics"
        r = requests.get(url, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def get_match_detail(self, match_id: int) -> Any:
        url = f"{self.api_base}/public/matches/{match_id}/detail"
        r = requests.get(url, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Analysis job endpoints (for AI service worker)
    # ------------------------------------------------------------------

    def get_pending_analysis(self) -> List[Dict[str, Any]]:
        """Fetch all queued analysis jobs ready for processing."""
        url = f"{self.api_base}/public/matches/pending-analysis"
        r = requests.get(url, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def claim_analysis_job(self, match_id: int) -> Dict[str, Any]:
        """Claim a queued job for processing. Returns 409 if already claimed."""
        url = f"{self.api_base}/public/matches/{match_id}/analysis/claim"
        r = requests.put(url, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def push_tracking_batch(self, match_id: int, payload: Dict[str, Any]) -> Any:
        """
        POST batch frame data al backend. Reintentos con backoff exponencial
        para que un error de red transitorio no descarte el resultado de varios
        minutos de procesamiento.

        - Reintenta hasta 3 veces ante errores de conexión o 5xx.
        - NO reintenta ante 4xx (config error, no se va a arreglar con retry).
        """
        import time as _time
        url = f"{self.api_base}/public/matches/{match_id}/tracking/batch"

        frames_count = len(payload.get("frames", [])) if isinstance(payload, dict) else 0
        payload_size_mb = len(str(payload)) / (1024 * 1024)
        print(f"[push] match {match_id}: enviando {frames_count} frames (~{payload_size_mb:.1f} MB)")

        last_err: Optional[Exception] = None
        for attempt in range(1, 4):                              # 3 intentos: t=0, t=5s, t=15s
            try:
                r = requests.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=180,                                  # 3 min para batches grandes
                )

                # 4xx no se arregla con retry, fallar rápido
                if 400 <= r.status_code < 500:
                    r.raise_for_status()

                # 5xx o conexión: reintenta
                r.raise_for_status()
                if attempt > 1:
                    print(f"[push] match {match_id}: éxito en intento {attempt}")
                return r.json()

            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                wait = 5 * attempt                                # 5s, 10s, 15s
                print(f"[push] match {match_id}: intento {attempt}/3 falló por red ({e}); reintentando en {wait}s")
                _time.sleep(wait)

            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status and 400 <= status < 500:
                    # Error del cliente: no reintentamos (payload inválido, auth, etc.)
                    body = ""
                    try:
                        body = e.response.text[:500]
                    except Exception:
                        pass
                    print(f"[push] match {match_id}: error {status} no recuperable. body: {body}")
                    raise
                last_err = e
                wait = 5 * attempt
                print(f"[push] match {match_id}: intento {attempt}/3 falló con {status}; reintentando en {wait}s")
                _time.sleep(wait)

        # Si llegamos aquí, los 3 intentos fallaron
        assert last_err is not None
        raise last_err

    def report_progress(
        self,
        match_id: int,
        status: str,
        progress: int = 0,
        current_step: str = "",
        frames_processed: Optional[int] = None,
        total_frames: Optional[int] = None,
        error_msg: str = "",
    ) -> Any:
        """PUT progress update to Geo-Goal backend."""
        url = f"{self.api_base}/public/matches/{match_id}/analysis/progress"
        payload: Dict[str, Any] = {"status": status, "progress": progress}
        if current_step:
            payload["currentStep"] = current_step
        if frames_processed is not None:
            payload["framesProcessed"] = frames_processed
        if total_frames is not None:
            payload["totalFrames"] = total_frames
        if error_msg:
            payload["error"] = error_msg
        r = requests.put(
            url,
            json=payload,
            headers=self._headers(),
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def download_video(self, url: str, dest_path: str) -> str:
        """Download a video from Supabase URL to a local path. Returns dest_path."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)

        print(f"[download] {url} -> {dest} ({dest.stat().st_size} bytes)")
        return str(dest)
