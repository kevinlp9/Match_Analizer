"""Background worker that polls for pending analysis jobs and processes them."""
from __future__ import annotations

import asyncio
import os
import traceback
from typing import Any, Optional

import numpy as np

from api_client import APIClient
from m2m_client import M2MClient
from video_processor import VideoProcessor
from event_detector import detect_all_events, detect_all_exhaustive, FrameSnapshot, PlayerPos


class AnalysisWorker:
    """Polls Geo-Goal backend for queued analysis jobs and processes them."""

    def __init__(
        self,
        api_client: APIClient,
        poll_interval: int = 30,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
    ) -> None:
        self.api = api_client
        self.poll_interval = poll_interval
        self.model_name = model_name
        self.device = device
        self._running = False
        self._current_job: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_job(self) -> Optional[int]:
        return self._current_job

    async def start(self) -> None:
        """Start the polling loop."""
        self._running = True
        print(f"[worker] Starting — poll interval: {self.poll_interval}s, device: {self.device}")
        while self._running:
            try:
                await self._poll_and_process()
            except Exception as e:
                print(f"[worker] Poll loop error: {e}")
                traceback.print_exc()
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        """Gracefully stop the worker."""
        self._running = False
        print("[worker] Stopping...")

    async def process_job(self, job: dict) -> None:
        """Process a single analysis job end-to-end."""
        job_id = job["jobId"]
        match_id = job["matchId"]
        video_url = job.get("videoSupabaseUrl")
        video_path_hint = job.get("videoPath")        # ruta local que conoce el backend
        src_pts_raw = job.get("srcPts")

        if not video_url and not video_path_hint:
            print(f"[worker] Job {job_id}: no videoSupabaseUrl ni videoPath, skipping")
            return

        if not src_pts_raw:
            print(f"[worker] Job {job_id}: no srcPts, skipping")
            return

        self._current_job = job_id
        output_dir = f"./output/{match_id}"
        video_path: Optional[str] = None
        downloaded = False                              # solo borrar si descargamos nosotros

        try:
            # 1. Claim the job
            print(f"[worker] Job {job_id}: claiming match {match_id}...")
            claimed = self.api.claim_analysis_job(match_id)
            print(f"[worker] Job {job_id}: claimed — status: {claimed.get('status')}")

            # 2. Resolver fuente del video, en orden de preferencia:
            #
            #    (a) DEV — videoPath local: backend y AI comparten FS, 0 transferencia.
            #    (b) PROD — streaming HTTP: cv2.VideoCapture(url) lee por HTTP range
            #        requests, sin descargar el archivo completo. Requiere que el MP4
            #        tenga el atom `moov` al inicio (fast-start). La mayoría de
            #        encoders modernos lo hacen, pero no todos los celulares.
            #    (c) Fallback — descarga completa: si el streaming no abre o falla
            #        rápido, descargamos el archivo entero y procesamos local
            #        (comportamiento histórico).
            if video_path_hint and os.path.exists(video_path_hint):
                # (a) Archivo local
                video_path = video_path_hint
                print(f"[worker] Job {job_id}: usando video LOCAL ({video_path}) — sin transferencia")
                self.api.report_progress(match_id, "processing", progress=5, current_step="local-file")

            elif video_url and self._streaming_works(video_url, job_id):
                # (b) Streaming directo desde Supabase — sin disco
                video_path = video_url
                print(f"[worker] Job {job_id}: STREAMING directo desde {video_url}")
                self.api.report_progress(match_id, "processing", progress=5, current_step="streaming")

            elif video_url:
                # (c) Fallback: descarga clásica
                self.api.report_progress(match_id, "processing", progress=5, current_step="downloading")
                print(f"[worker] Job {job_id}: streaming no funcionó, descargando {video_url}...")
                video_path = self.api.download_video(video_url, f"./uploads/{match_id}.mp4")
                downloaded = True

            else:
                raise RuntimeError(
                    f"No hay forma de obtener el video: videoPath '{video_path_hint}' no existe y no hay videoSupabaseUrl"
                )

            # 3. Build src_pts array
            src_pts = np.array([[pt["x"], pt["y"]] for pt in src_pts_raw], dtype=np.float32)

            # 4. Process video
            print(f"[worker] Job {job_id}: processing video...")
            player_tags = job.get("playerTags")
            print(f"[worker] Job {job_id}: {len(player_tags) if player_tags else 0} player tags from job")

            # JSON keys son strings → convertir a int para el VideoProcessor
            identity_map_raw = job.get("identityMap") or {}
            identity_map = {int(k): int(v) for k, v in identity_map_raw.items()}
            print(f"[worker] Job {job_id}: identity_map has {len(identity_map)} entries")

            frame_skip_val = int(os.environ.get("ANALYSIS_FRAME_SKIP", "4"))
            # Calcular fps efectivos para el modelo Kalman interno de ByteTrack
            base_fps = 25.0  # fps típico de broadcast de fútbol
            effective_fps = max(1, int(round(base_fps / (frame_skip_val + 1))))

            # lineupMode del partido: 11 (estándar) o 7 (cancha chica).
            # Si el backend no lo envía (compat. hacia atrás), default 11.
            lineup_mode = int(job.get("lineupMode") or 11)
            print(f"[worker] Job {job_id}: lineup_mode={lineup_mode} (jugadores por equipo)")

            vp = VideoProcessor(
                model_name=self.model_name,
                device=self.device,
                api_base=self.api.api_base,
                output_dir=output_dir,
                player_tags=player_tags,
                identity_map=identity_map,
                frame_rate=effective_fps,
                max_gap_frames=int(os.environ.get("INTERP_MAX_GAP_FRAMES", "15")),
                ball_predict_frames=int(os.environ.get("INTERP_BALL_PREDICT_FRAMES", "8")),
                lineup_mode=lineup_mode,
            )
            vp.load_video(video_path)
            vp.set_homography(src_pts=src_pts, method="cv2")

            # Override progress reporting to go through our API client
            self._instrument_progress(vp, match_id, job_id, effective_fps)

            await asyncio.to_thread(
                vp.process,
                job_id=job_id,
                export_json="match_data.json",
                push_match_id=match_id,
                frame_skip=frame_skip_val,  # 5 fps en lugar de 25 → 5× más rápido
            )

            # 5. Mark completed
            self.api.report_progress(
                match_id, "completed",
                progress=100, current_step="done",
                frames_processed=vp._total_frames,
                total_frames=vp._total_frames,
            )
            print(f"[worker] Job {job_id}: completed successfully")

        except Exception as e:
            error_msg = str(e)[:1000]
            print(f"[worker] Job {job_id}: FAILED — {error_msg}")
            traceback.print_exc()
            try:
                self.api.report_progress(
                    match_id, "failed",
                    progress=0, current_step="error",
                    error_msg=error_msg,
                )
            except Exception as report_err:
                print(f"[worker] Job {job_id}: failed to report failure: {report_err}")

        finally:
            self._current_job = None
            # SOLO borrar si nosotros descargamos.
            # Si usamos el archivo local del backend, NO tocar (lo creó él).
            if downloaded and video_path:
                try:
                    os.unlink(video_path)
                    print(f"[worker] Job {job_id}: cleaned up {video_path}")
                except Exception:
                    pass

    def _streaming_works(self, url: str, job_id: int, probe_timeout_s: float = 8.0) -> bool:
        """
        Verifica si podemos procesar el video por streaming HTTP en lugar de descargarlo.

        Abre cv2.VideoCapture(url) y mide cuánto tarda en leer el primer frame:
          - Rápido (<probe_timeout_s)  → el MP4 es fast-start, podemos hacer streaming.
          - Lento o falla              → moov al final / red mala / formato no soportado
                                          → mejor descargar el archivo completo.

        Se ejecuta en un thread para poder forzar timeout (cv2 no respeta señales).
        """
        # Permitir desactivar globalmente con env var (útil para debugging)
        if os.environ.get("DISABLE_VIDEO_STREAMING", "").lower() in ("1", "true", "yes"):
            print(f"[worker] Job {job_id}: streaming desactivado por DISABLE_VIDEO_STREAMING")
            return False

        import cv2
        import threading
        import time

        result = {"ok": False, "elapsed": 0.0, "error": None}

        def _probe():
            t0 = time.time()
            cap = None
            try:
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    result["error"] = "VideoCapture no abrió"
                    return
                ret, frame = cap.read()
                if not ret or frame is None:
                    result["error"] = "no se pudo leer el primer frame"
                    return
                result["ok"] = True
            except Exception as e:
                result["error"] = str(e)
            finally:
                if cap is not None:
                    cap.release()
                result["elapsed"] = time.time() - t0

        thread = threading.Thread(target=_probe, daemon=True)
        thread.start()
        thread.join(timeout=probe_timeout_s)

        if thread.is_alive():
            # El probe sigue corriendo tras el timeout → el video es muy lento de abrir,
            # casi seguro moov al final. Lo dejamos como daemon (morirá con el proceso).
            print(f"[worker] Job {job_id}: probe HTTP excedió {probe_timeout_s}s → fallback a descarga")
            return False

        if not result["ok"]:
            print(f"[worker] Job {job_id}: probe HTTP falló ({result['error']}) → fallback a descarga")
            return False

        print(f"[worker] Job {job_id}: probe HTTP OK en {result['elapsed']:.2f}s → streaming viable")
        return True

    def _instrument_progress(self, vp: VideoProcessor, match_id: int, job_id: int, effective_fps: float) -> None:
        """Wrap the exporter to use our API client for progress reporting."""

        def _report(
            _match_id: int,
            _m2m_token: str,
            status: str,
            progress: int = 0,
            current_step: str = "",
            frames_processed: Optional[int] = None,
            total_frames: Optional[int] = None,
            error_msg: str = "",
        ) -> None:
            try:
                self.api.report_progress(
                    match_id,
                    status,
                    progress=progress,
                    current_step=current_step,
                    frames_processed=frames_processed,
                    total_frames=total_frames,
                    error_msg=error_msg,
                )
            except Exception as e:
                print(f"[worker] Progress report failed: {e}")

        vp.exporter.report_progress = _report  # type: ignore

        def _push(_match_id: int, _m2m_token: str) -> Any:
            # Construir FrameSnapshots para el detector de eventos (Fase 7)
            raw_frames = vp.exporter._frames
            snapshots = []
            for f in raw_frames:
                ball = None
                if f.get("ballX") is not None and f.get("ballY") is not None:
                    # coordenadas del exporter están en 0-100 (normalizadas) → convertir a metros
                    ball = (f["ballX"] * 105.0 / 100.0, f["ballY"] * 68.0 / 100.0)
                players = []
                for p in f.get("players", []):
                    players.append(PlayerPos(
                        player_id=p.get("trackerId", p.get("id", 0)),
                        team=p.get("team", "unknown"),
                        x=p.get("x", 0.0) * 105.0 / 100.0,
                        y=p.get("y", 0.0) * 68.0 / 100.0,
                    ))
                snapshots.append(FrameSnapshot(
                    frame_idx=f.get("frameIdx", 0),
                    timestamp_ms=f.get("timestampMs", 0),
                    ball=ball,
                    players=players,
                ))

            # Detectar eventos automáticos.
            # Cambiamos al modo EXHAUSTIVO: además de ball_out/goal/pass detecta
            # shot, key_pass, cross, clearance, dribble, foul, yellow_card,
            # red_card, substitution, own_goal, penalty_scored.
            #
            # Para volver al modo legacy (solo 3 detectores), usar la env var:
            #   EVENTS_MODE=legacy
            inferred_events = []
            try:
                events_mode = os.environ.get("EVENTS_MODE", "exhaustive").lower()
                if events_mode == "exhaustive":
                    inferred_events = detect_all_exhaustive(snapshots, fps=effective_fps)
                else:
                    inferred_events = detect_all_events(
                        snapshots,
                        fps=effective_fps,
                        detect_out=True,
                        detect_goals=True,
                        detect_passes=True,
                    )
                print(f"[worker] Job {job_id}: {len(inferred_events)} eventos inferidos detectados (mode={events_mode})")
            except Exception as detect_err:
                print(f"[worker] Job {job_id}: error en detección de eventos: {detect_err}")
                import traceback; traceback.print_exc()

            payload = {
                "pitch": {"length_m": 105.0, "width_m": 68.0},
                "frames": raw_frames,
                "inferredEvents": inferred_events,
            }
            return self.api.push_tracking_batch(match_id, payload)

        vp.exporter.push_to_api = _push  # type: ignore

    async def _poll_and_process(self) -> None:
        """Check for pending jobs and process the first available one."""
        try:
            jobs = self.api.get_pending_analysis()
        except Exception as e:
            print(f"[worker] Failed to fetch pending jobs: {e}")
            return

        if not jobs:
            return

        print(f"[worker] Found {len(jobs)} pending job(s)")

        # Process only the first pending job per poll cycle
        job = jobs[0]
        await self.process_job(job)
