"""
Geo-Goal AI Video Processor — Local / Offline variant
Pipeline: Detection → Homography → Perspective Transform → Interpolation → Export

Core mathematical foundation (thesis OE1-OE4):
  - DLT (Direct Linear Transform) for homography estimation
  - Homogeneous coordinates [x, y, 1]^T
  - 3x3 homography matrix H mapping source (pixels) → destination (pitch coords)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard football pitch dimensions (metres) — FIFA ranges: 100-110 x 64-75
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Default source points (pixel coords) for a tactical-cam frame of
# Belgium vs Russia EURO 2020 — override with manual annotation.
# These are example values; you MUST run annotate_keypoints() on your frame.
DEFAULT_SRC_PTS = np.array(
    [
        [509, 183],   # top-left corner of penalty area (or pitch corner)
        [639, 307],  # top-right
        [0, 306],  # bottom-right
        [125, 184],   # bottom-left
    ],
    dtype=np.float32,
)

# Corresponding pitch-coordinate destination (metres from top-left origin)
DEFAULT_DST_PTS = np.array(
    [
        [0, 0],
        [PITCH_LENGTH_M, 0],
        [PITCH_LENGTH_M, PITCH_WIDTH_M],
        [0, PITCH_WIDTH_M],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Player2D:
    """A player projected onto the 2D pitch plane."""

    id: int
    x: float  # metres (length axis)
    y: float  # metres (width axis)
    team: str  # "home" | "away" | "referee" | "unknown"
    interpolated: bool = False  # True when position was estimated, not detected


@dataclass
class FrameData:
    """All extracted data for a single video frame."""

    frame_idx: int
    timestamp_ms: int
    players: List[Player2D] = field(default_factory=list)
    ball: Optional[Tuple[float, float]] = None  # (x, y) in metres
    ball_interpolated: bool = False              # True when ball pos was estimated
    confidence: float = 1.0  # mean YOLO detection confidence for this frame


# ---------------------------------------------------------------------------
# 1. Object Detector (YOLOv8)
# ---------------------------------------------------------------------------


class ObjectDetector:
    """Wraps a YOLO model for player/ball detection.

    COCO classes used:
      0  — person   → player
      32 — sports ball → football
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu") -> None:
        self.model = YOLO(model_name)
        self.device = device

    def detect(self, frame: np.ndarray, conf: float = 0.3) -> sv.Detections:
        """Run inference and return supervision Detections filtered to person+ball."""
        results = self.model(frame, device=self.device, conf=conf, verbose=False)
        if results[0].boxes is None:
            return sv.Detections.empty()
        detections = sv.Detections.from_ultralytics(results[0])
        # Keep only COCO classes we care about
        mask = np.isin(detections.class_id, [0, 32])
        return detections[mask]


# ---------------------------------------------------------------------------
# 2. Object Tracker (ByteTrack via supervision) — tuned parameters
# ---------------------------------------------------------------------------


class ObjectTracker:
    """Assigns persistent IDs to players across frames using ByteTrack.

    Parameters tuned for football tracking:
      - track_activation_threshold: lowered to 0.25 so detections confirmed sooner.
      - lost_track_buffer: extended to 60 frames (≈2 s at 5 fps skip) to keep tracks
        alive through brief occlusions / camera pans.
      - minimum_matching_threshold: lowered to 0.7 for more lenient IoU association.
      - frame_rate: set to 5 (matching default ANALYSIS_FRAME_SKIP=4 → 5 fps) so
        ByteTrack's internal Kalman velocity model has correct time scaling.
    """

    def __init__(self, frame_rate: int = 5) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
            frame_rate=frame_rate,
        )

    def track(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker and return detections with tracker_id field."""
        return self.tracker.update_with_detections(detections)


# ---------------------------------------------------------------------------
# 3. Team Classifier (K-Means on jersey colour)
# ---------------------------------------------------------------------------


class TeamClassifier:
    """Separates players into two teams + referee by clustering jersey colours.

    Algorithm:
      1. Extract a tight crop of each player's upper body from their bbox.
      2. Build a colour histogram in HSV space.
      3. Run K-Means (k=3) to partition into home / away / referee clusters.
      4. If player_tags are provided, override labels for matched detections.
    """

    def __init__(self, player_tags: Optional[List[Dict[str, Any]]] = None) -> None:
        self._centroids: Optional[np.ndarray] = None
        self._home_label: Optional[int] = None
        self._away_label: Optional[int] = None
        self._ref_label: Optional[int] = None
        self._player_tags = player_tags  # [{ x, y, label }, ...] in pixel coords

    def fit_predict(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> List[str]:
        """Fit K-Means on first call; return team labels for every detection."""
        from sklearn.cluster import KMeans

        features = self._extract_colour_features(frame, detections)

        if len(features) < 3:
            return ["unknown"] * len(detections)

        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        self._centroids = kmeans.cluster_centers_

        # Heuristic: referee wears black/dark -> lowest V channel mean
        v_means = [self._centroids[i][2] for i in range(3)]
        self._ref_label = int(np.argmin(v_means))

        # Remaining two clusters are the teams
        team_ids = [i for i in range(3) if i != self._ref_label]
        self._home_label = team_ids[0]
        self._away_label = team_ids[1]

        result = self._label(labels)

        # Override with player_tags (ground truth)
        if self._player_tags:
            result = self._apply_player_tags(detections, result)

        return result

    def predict(self, frame: np.ndarray, detections: sv.Detections) -> List[str]:
        """Re-use fitted clusters on subsequent frames."""
        if self._centroids is None or len(detections) == 0:
            return self.fit_predict(frame, detections)

        features = self._extract_colour_features(frame, detections)
        dists = np.linalg.norm(features[:, None, :] - self._centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return self._label(labels)

    def _apply_player_tags(
        self,
        detections: sv.Detections,
        current_labels: List[str],
    ) -> List[str]:
        """Override labels for detections closest to each player tag."""
        if not self._player_tags or len(detections) == 0:
            return current_labels

        labels = list(current_labels)
        used_detections: set = set()

        for tag in self._player_tags:
            tx, ty = tag.get("x", 0), tag.get("y", 0)
            tag_label = tag.get("label", "unknown")
            if tag_label == "referee":
                continue  # skip unlabeled

            # Find nearest detection bbox center
            best_idx = -1
            best_dist = float("inf")
            for i, xyxy in enumerate(detections.xyxy.astype(int)):
                if i in used_detections:
                    continue
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                dist = (cx - tx) ** 2 + (cy - ty) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # Only override if within reasonable distance (150px radius)
            if best_idx >= 0 and best_dist < 150 * 150:
                labels[best_idx] = tag_label
                used_detections.add(best_idx)

        return labels

    # ------------------------------------------------------------------
    def _extract_colour_features(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        feats = []
        for xyxy in detections.xyxy.astype(int):
            x1, y1, x2, y2 = xyxy
            # Upper third of the bounding box (jersey region)
            crop = hsv[y1 : y1 + (y2 - y1) // 3, x1:x2]
            if crop.size == 0:
                feats.append([0, 0, 0])
                continue
            # Mean H, S, V — robust to small variations
            h_mean = np.mean(crop[:, :, 0])
            s_mean = np.mean(crop[:, :, 1])
            v_mean = np.mean(crop[:, :, 2])
            feats.append([h_mean, s_mean, v_mean])
        return np.array(feats, dtype=np.float32)

    def _label(self, ids: np.ndarray) -> List[str]:
        result: List[str] = []
        for i in ids:
            if i == self._home_label:
                result.append("home")
            elif i == self._away_label:
                result.append("away")
            elif i == self._ref_label:
                result.append("referee")
            else:
                result.append("unknown")
        return result


# ---------------------------------------------------------------------------
# 4. Homography Calculator (DLT — thesis OE2 & OE3)
# ---------------------------------------------------------------------------


class HomographyCalculator:
    """Estimates the 3×3 homography matrix from ≥4 point correspondences.

    Two implementations provided:
      - _dlt_manual(): for the thesis — documents the linear system explicitly.
      - _dlt_cv2():    production path using OpenCV (RANSAC-robust).
    """

    def compute(
        self,
        src_pts: np.ndarray,  # shape (N, 2) — pixel coordinates
        dst_pts: np.ndarray,  # shape (N, 2) — pitch coordinates (metres)
        method: str = "cv2",
    ) -> np.ndarray:
        """Return 3×3 homography matrix H such that dst ≅ H ⋅ src."""
        if method == "manual":
            return self._dlt_manual(src_pts, dst_pts)
        return self._dlt_cv2(src_pts, dst_pts)

    # ------------------------------------------------------------------
    # Manual DLT (thesis — Objective 2 & 3)
    # ------------------------------------------------------------------
    @staticmethod
    def _dlt_manual(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Solve for H using the Direct Linear Transform.

        For each correspondence (x_i, y_i) → (X_i, Y_i) in homogeneous coords:

            [x_i, y_i, 1]^T  →  [X_i, Y_i, 1]^T  (up to scale)

        The cross-product [X_i, Y_i, 1]^T × H ⋅ [x_i, y_i, 1]^T = 0
        yields two linearly-independent equations per point:

            [ 0^T   , -w'_i·p_i^T ,  y'_i·p_i^T ] ⋅ h = 0
            [ w'_i·p_i^T , 0^T    , -x'_i·p_i^T ] ⋅ h = 0

        where p_i = [x_i, y_i, 1], h = vec(H) (9×1 column-major).

        Stacking 2N equations → A·h = 0, solved via SVD.
        """
        N = src.shape[0]
        A = np.zeros((2 * N, 9), dtype=np.float64)

        for i in range(N):
            x, y = src[i]
            X, Y = dst[i]
            # Row 2i:   [0,0,0, -x, -y, -1, Y·x, Y·y, Y]
            A[2 * i] = [0, 0, 0, -x, -y, -1, Y * x, Y * y, Y]
            # Row 2i+1: [x, y, 1,  0,  0,  0, -X·x, -X·y, -X]
            A[2 * i + 1] = [x, y, 1, 0, 0, 0, -X * x, -X * y, -X]

        # SVD: A = U·Σ·V^T  →  h = last column of V (minimum singular vector)
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]  # shape (9,)
        H = h.reshape(3, 3)

        # Normalise so H[2, 2] = 1
        return H / H[2, 2]

    # ------------------------------------------------------------------
    # OpenCV DLT with RANSAC (production)
    # ------------------------------------------------------------------
    @staticmethod
    def _dlt_cv2(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        H, _mask = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if H is None:
            raise RuntimeError("cv2.findHomography failed — try more/better keypoints")
        return H


# ---------------------------------------------------------------------------
# 5. Perspective Transformer (thesis OE4)
# ---------------------------------------------------------------------------


class PerspectiveTransformer:
    """Applies the homography H to project player pixel coords → pitch coords.

    The *base point* of a player is the midpoint of the bottom edge of their
    bounding box — this approximates where their feet touch the ground,
    which is the point that lies on the pitch plane.
    """

    def __init__(self, H: np.ndarray) -> None:
        self.H = H

    def player_base_point(self, xyxy: np.ndarray) -> Tuple[float, float]:
        """Given bbox [x1, y1, x2, y2], return (px, py) of centre-bottom."""
        x_centre = (xyxy[0] + xyxy[2]) / 2.0
        y_bottom = xyxy[3]  # bottom edge
        return float(x_centre), float(y_bottom)

    def pixel_to_pitch(self, px: float, py: float) -> Tuple[float, float]:
        """Map a single pixel coordinate to pitch metres via H."""
        vec = np.array([[px], [py], [1.0]], dtype=np.float64)
        projected = self.H @ vec
        # De-homogenise
        x = projected[0, 0] / projected[2, 0]
        y = projected[1, 0] / projected[2, 0]
        return float(x), float(y)


# ---------------------------------------------------------------------------
# 5b. Track Interpolator — fills gaps in player/ball tracks
# ---------------------------------------------------------------------------


class TrackInterpolator:
    """Post-processes a list of FrameData to fill missing positions.

    Strategy
    --------
    Players
      - For frames where a tracker_id is absent, linearly interpolate x/y
        between the last known and next known position.
      - Interpolated entries are flagged with `interpolated=True`.
      - Gaps longer than `max_gap_frames` are left unfilled (track truly lost).

    Ball
      - Same linear interpolation between known positions.
      - For the tail (no next known position), if the gap is ≤ `ball_predict_frames`
        apply dead reckoning using the last known velocity vector.
    """

    def __init__(
        self,
        max_gap_frames: int = 15,
        ball_predict_frames: int = 8,
        player_predict_frames: int = 8,
    ) -> None:
        self.max_gap_frames = max_gap_frames
        self.ball_predict_frames = ball_predict_frames
        # Cuántos frames extrapolar un jugador hacia adelante/atrás cuando su
        # track desaparece o aparece "tarde". A frame_skip=4 (5fps), 8 frames
        # ≈ 1.6 segundos de "memoria" antes de dar el track por perdido.
        self.player_predict_frames = player_predict_frames

    def interpolate(self, frames: List[FrameData]) -> List[FrameData]:
        """Return a new list of FrameData with gaps filled in-place (mutates frames)."""
        if not frames:
            return frames

        self._interpolate_players(frames)
        self._interpolate_ball(frames)
        return frames

    # ------------------------------------------------------------------
    def _interpolate_players(self, frames: List[FrameData]) -> None:
        """Linear interpolation + dead-reckoning of player positions over missing frames.

        Tres pasos:
          1. INTERPOLACIÓN INTERIOR: gap entre dos detecciones reales → lineal.
          2. EXTRAPOLACIÓN COLA: el track desaparece al final → predice usando
             velocidad de los dos últimos puntos conocidos (dead reckoning).
          3. EXTRAPOLACIÓN INICIO: el track aparece "tarde" → predice hacia atrás
             usando velocidad de los dos primeros puntos.

        El resultado: ningún jugador se queda sin coordenadas mientras su track
        estuvo "vivo" + 1.6 segundos de inercia antes y después.
        """
        n = len(frames)
        if n == 0:
            return

        # Solo consideramos detecciones REALES como ancla (no interpoladas
        # de pasadas anteriores, aunque en este flujo no debería haber).
        track_map: Dict[int, List[Tuple[int, Player2D]]] = {}
        for fi, fd in enumerate(frames):
            for p in fd.players:
                if not p.interpolated:
                    track_map.setdefault(p.id, []).append((fi, p))

        for pid, entries in track_map.items():
            if len(entries) < 2:
                # Track con una sola detección: no podemos estimar velocidad
                # ni dirección. Lo dejamos como está.
                continue

            # ---- 1. INTERPOLACIÓN INTERIOR ----
            for k in range(len(entries) - 1):
                i0, p0 = entries[k]
                i1, p1 = entries[k + 1]
                gap = i1 - i0 - 1
                if gap <= 0 or gap > self.max_gap_frames:
                    continue

                for step in range(1, gap + 1):
                    t = step / (gap + 1)
                    x_interp = p0.x + t * (p1.x - p0.x)
                    y_interp = p0.y + t * (p1.y - p0.y)
                    fi_fill = i0 + step
                    # No duplicar si ya tiene a este player_id (no debería pasar)
                    if any(pp.id == pid for pp in frames[fi_fill].players):
                        continue
                    frames[fi_fill].players.append(
                        Player2D(
                            id=pid,
                            x=round(x_interp, 3),
                            y=round(y_interp, 3),
                            team=p1.team if p1.team != "unknown" else p0.team,
                            interpolated=True,
                        )
                    )

            # ---- 2. EXTRAPOLACIÓN COLA (dead reckoning hacia adelante) ----
            i_last, p_last = entries[-1]
            i_prev, p_prev = entries[-2]
            dt_tail = i_last - i_prev
            if dt_tail > 0 and i_last < n - 1:
                vx = (p_last.x - p_prev.x) / dt_tail
                vy = (p_last.y - p_prev.y) / dt_tail
                max_extra = min(self.player_predict_frames, n - 1 - i_last)
                for step in range(1, max_extra + 1):
                    fi_fill = i_last + step
                    # Si ya hay detección en ese frame, paramos (track volvió)
                    if any(pp.id == pid for pp in frames[fi_fill].players):
                        break
                    x_pred = p_last.x + vx * step
                    y_pred = p_last.y + vy * step
                    # Clamping a un margen razonable del campo
                    x_pred = max(-5, min(PITCH_LENGTH_M + 5, x_pred))
                    y_pred = max(-5, min(PITCH_WIDTH_M + 5, y_pred))
                    frames[fi_fill].players.append(
                        Player2D(
                            id=pid,
                            x=round(x_pred, 3),
                            y=round(y_pred, 3),
                            team=p_last.team,
                            interpolated=True,
                        )
                    )

            # ---- 3. EXTRAPOLACIÓN INICIO (dead reckoning hacia atrás) ----
            i_first, p_first = entries[0]
            i_second, p_second = entries[1]
            dt_head = i_second - i_first
            if dt_head > 0 and i_first > 0:
                vx = (p_second.x - p_first.x) / dt_head
                vy = (p_second.y - p_first.y) / dt_head
                max_back = min(self.player_predict_frames, i_first)
                for step in range(1, max_back + 1):
                    fi_fill = i_first - step
                    if any(pp.id == pid for pp in frames[fi_fill].players):
                        break
                    x_pred = p_first.x - vx * step
                    y_pred = p_first.y - vy * step
                    x_pred = max(-5, min(PITCH_LENGTH_M + 5, x_pred))
                    y_pred = max(-5, min(PITCH_WIDTH_M + 5, y_pred))
                    frames[fi_fill].players.append(
                        Player2D(
                            id=pid,
                            x=round(x_pred, 3),
                            y=round(y_pred, 3),
                            team=p_first.team,
                            interpolated=True,
                        )
                    )

    # ------------------------------------------------------------------
    def _interpolate_ball(self, frames: List[FrameData]) -> None:
        """Linear interpolation + dead-reckoning for ball positions."""
        n = len(frames)

        # Collect known ball positions: frame_index → (x, y)
        known: Dict[int, Tuple[float, float]] = {}
        for fi, fd in enumerate(frames):
            if fd.ball is not None:
                known[fi] = fd.ball

        if len(known) < 1:
            return

        sorted_known = sorted(known.keys())

        # --- Interior gaps: linear interpolation ---
        for k in range(len(sorted_known) - 1):
            i0 = sorted_known[k]
            i1 = sorted_known[k + 1]
            gap = i1 - i0 - 1
            if gap <= 0 or gap > self.max_gap_frames:
                continue
            x0, y0 = known[i0]
            x1, y1 = known[i1]
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                fi_fill = i0 + step
                frames[fi_fill].ball = (
                    round(x0 + t * (x1 - x0), 3),
                    round(y0 + t * (y1 - y0), 3),
                )
                frames[fi_fill].ball_interpolated = True

        # --- Tail: dead reckoning using last velocity ---
        last_known_fi = sorted_known[-1]
        if last_known_fi < n - 1:
            # Estimate velocity from last two known positions
            if len(sorted_known) >= 2:
                fi_prev = sorted_known[-2]
                fi_last = sorted_known[-1]
                dt = fi_last - fi_prev
                if dt > 0:
                    vx = (known[fi_last][0] - known[fi_prev][0]) / dt
                    vy = (known[fi_last][1] - known[fi_prev][1]) / dt
                else:
                    vx = vy = 0.0
            else:
                vx = vy = 0.0

            x_last, y_last = known[last_known_fi]
            for step in range(1, self.ball_predict_frames + 1):
                fi_fill = last_known_fi + step
                if fi_fill >= n:
                    break
                if frames[fi_fill].ball is not None:
                    break  # already filled by interpolation
                x_pred = x_last + vx * step
                y_pred = y_last + vy * step
                # Clamping al área del campo + margen pequeño
                x_pred = max(-5, min(PITCH_LENGTH_M + 5, x_pred))
                y_pred = max(-5, min(PITCH_WIDTH_M + 5, y_pred))
                frames[fi_fill].ball = (round(x_pred, 3), round(y_pred, 3))
                frames[fi_fill].ball_interpolated = True

        # --- Head: dead reckoning backwards using initial velocity ---
        # Cuando el balón aparece "tarde" en el video (los primeros segundos no
        # se detecta). Extrapolamos hacia atrás con la velocidad de los dos
        # primeros frames conocidos.
        first_known_fi = sorted_known[0]
        if first_known_fi > 0 and len(sorted_known) >= 2:
            fi_first = sorted_known[0]
            fi_second = sorted_known[1]
            dt_head = fi_second - fi_first
            if dt_head > 0:
                vx_h = (known[fi_second][0] - known[fi_first][0]) / dt_head
                vy_h = (known[fi_second][1] - known[fi_first][1]) / dt_head
            else:
                vx_h = vy_h = 0.0

            x_first, y_first = known[first_known_fi]
            for step in range(1, self.ball_predict_frames + 1):
                fi_fill = first_known_fi - step
                if fi_fill < 0:
                    break
                if frames[fi_fill].ball is not None:
                    break
                x_pred = x_first - vx_h * step
                y_pred = y_first - vy_h * step
                x_pred = max(-5, min(PITCH_LENGTH_M + 5, x_pred))
                y_pred = max(-5, min(PITCH_WIDTH_M + 5, y_pred))
                frames[fi_fill].ball = (round(x_pred, 3), round(y_pred, 3))
                frames[fi_fill].ball_interpolated = True


# ---------------------------------------------------------------------------
# 6. Data Exporter (local-only — no cloud/API)
# ---------------------------------------------------------------------------


class DataExporter:
    """Serialises FrameData to JSON with pitch metadata and homography matrix."""

    # Map classifier string labels to numeric team IDs
    TEAM_MAP: Dict[str, int] = {"home": 1, "away": 2, "referee": 0, "unknown": -1}

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames: List[Dict[str, Any]] = []
        self._homography: Optional[List[List[float]]] = None

    def set_homography(self, H: np.ndarray) -> None:
        """Store the 3×3 homography matrix as a nested list for JSON export."""
        self._homography = H.tolist()

    def add_frame(self, fd: FrameData, confidence: float = 1.0) -> None:
        self._frames.append(
            {
                "timestampMs": fd.timestamp_ms,
                "ball": (
                    {
                        "x": round(fd.ball[0], 3),
                        "y": round(fd.ball[1], 3),
                        "interpolated": fd.ball_interpolated,
                    }
                    if fd.ball is not None
                    else None
                ),
                "players": [
                    {
                        "playerId": p.id,
                        "teamId": self.TEAM_MAP.get(p.team, -1),
                        "x": round(p.x, 3),
                        "y": round(p.y, 3),
                        "interpolated": p.interpolated,
                    }
                    for p in fd.players
                ],
                "source": "video",
                "confidence": round(confidence, 4),
                "coordSystem": "meters",
            }
        )

    def save(self, filename: str = "match_data.json") -> str:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pitch": {"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
                    "homography": self._homography,
                    "frames": self._frames,
                },
                f,
                indent=2,
            )
        return str(path)


# ---------------------------------------------------------------------------
# 7. Pitch Visualizer (matplotlib — for validation)
# ---------------------------------------------------------------------------


class PitchVisualizer:
    """Draws a top-down 2D pitch with player positions for visual validation."""

    @staticmethod
    def draw(
        frame_data: FrameData,
        save_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-5, PITCH_LENGTH_M + 5)
        ax.set_ylim(-5, PITCH_WIDTH_M + 5)
        ax.set_aspect("equal")
        ax.set_title(f"Frame {frame_data.frame_idx} — Tactical View")

        # Pitch outline
        ax.add_patch(
            mpatches.Rectangle((0, 0), PITCH_LENGTH_M, PITCH_WIDTH_M, fill=False, lw=2)
        )
        # Half-way line
        ax.axvline(PITCH_LENGTH_M / 2, color="black", ls="--", lw=1)
        # Centre circle
        centre = plt.Circle((PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2), 9.15, fill=False, lw=1)
        ax.add_patch(centre)

        # Draw players
        colours = {"home": "red", "away": "blue", "referee": "yellow", "unknown": "gray"}
        for p in frame_data.players:
            alpha = 0.4 if p.interpolated else 1.0
            ax.scatter(
                p.x, p.y,
                c=colours.get(p.team, "gray"),
                s=60,
                edgecolors="black",
                zorder=5,
                alpha=alpha,
            )
            ax.annotate(str(p.id), (p.x + 0.5, p.y + 0.5), fontsize=7)

        # Ball
        if frame_data.ball is not None:
            alpha = 0.4 if frame_data.ball_interpolated else 1.0
            ax.scatter(
                frame_data.ball[0], frame_data.ball[1],
                c="white",
                s=100,
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
                marker="o",
                alpha=alpha,
            )

        handles = [
            mpatches.Patch(color="red", label="Home"),
            mpatches.Patch(color="blue", label="Away"),
            mpatches.Patch(color="yellow", label="Referee"),
        ]
        ax.legend(handles=handles, loc="upper right")
        ax.set_xlabel("Pitch length (m)")
        ax.set_ylabel("Pitch width (m)")
        ax.invert_yaxis()  # pitch top = y=0 in broadcast view

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# 8. Video Processor (Orchestrator)
# ---------------------------------------------------------------------------


class VideoProcessor:
    """High-level pipeline orchestrator.

    Usage:
        vp = VideoProcessor("yolov8n.pt")
        vp.load_video("belgium_russia.mp4")
        vp.set_homography()  # uses default src/dst — call after annotating
        vp.process(export_json="output/match_data.json")
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        output_dir: str = "./output",
        player_tags: Optional[List[Dict[str, Any]]] = None,
        identity_map: Optional[Dict[int, int]] = None,
        frame_rate: int = 5,
        max_gap_frames: int = 15,
        ball_predict_frames: int = 8,
        lineup_mode: int = 11,
    ) -> None:
        self.detector = ObjectDetector(model_name, device)
        self.tracker = ObjectTracker(frame_rate=frame_rate)
        self.classifier = TeamClassifier(player_tags=player_tags)
        self.interpolator = TrackInterpolator(
            max_gap_frames=max_gap_frames,
            ball_predict_frames=ball_predict_frames,
        )
        self.transformer: Optional[PerspectiveTransformer] = None
        self.exporter = DataExporter(output_dir)
        self.H: Optional[np.ndarray] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 25.0
        self._total_frames: int = 0
        # ------------------------------------------------------------------
        # Límites físicos del partido — máximos teóricos en el campo:
        #   - lineup_mode == 11  → 11 vs 11 = 22 jugadores
        #   - lineup_mode == 7   → 7 vs 7   = 14 jugadores
        #   - +2 árbitros aprox. de margen
        # Sin este límite, YOLO detecta banca, cuerpo técnico, fotógrafos, etc
        # y se crean tracks fantasma que generan saltos en la visualización.
        # ------------------------------------------------------------------
        self.lineup_mode: int = lineup_mode
        self.max_per_team: int = lineup_mode  # 11 o 7
        self.max_refs: int = 3                # tolerancia árbitros / asistentes
        print(f"[processor] lineup_mode={lineup_mode} → max {self.max_per_team} por equipo + {self.max_refs} árbitros")
        self.identity_map: Dict[int, int] = identity_map or {}

    # ------------------------------------------------------------------
    # Load video
    # ------------------------------------------------------------------
    def load_video(self, video_path: str) -> "VideoProcessor":
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[video] {video_path}: {self._total_frames} frames @ {self._fps:.1f} fps")
        return self

    # ------------------------------------------------------------------
    # Homography setup
    # ------------------------------------------------------------------
    def set_homography(
        self,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        method: str = "cv2",
    ) -> "VideoProcessor":
        """Set source→destination points and compute H.

        If src_pts/dst_pts are None, DEFAULT_SRC_PTS / DEFAULT_DST_PTS are used.
        """
        src = src_pts if src_pts is not None else DEFAULT_SRC_PTS
        dst = dst_pts if dst_pts is not None else DEFAULT_DST_PTS
        calculator = HomographyCalculator()
        self.H = calculator.compute(src, dst, method=method)
        self.transformer = PerspectiveTransformer(self.H)
        self.exporter.set_homography(self.H)
        print(f"[homography] H computed ({method}):\n{self.H}")
        return self

    # ------------------------------------------------------------------
    # Load calibration from calib.json
    # ------------------------------------------------------------------
    def load_calibration(self, calib_path: str) -> "VideoProcessor":
        """Load calibration from calib.json (src/dst points + player tags)."""
        with open(calib_path, "r") as f:
            calib = json.load(f)
        src_pts = np.array(calib["src_points"], dtype=np.float32)
        dst_pts = np.array(calib["dst_points"], dtype=np.float32)
        method = calib.get("method", "cv2")
        self.set_homography(src_pts=src_pts, dst_pts=dst_pts, method=method)
        # Load player tags if present
        if calib.get("player_tags"):
            self.classifier = TeamClassifier(player_tags=calib["player_tags"])
        return self

    # ------------------------------------------------------------------
    # Frame-by-frame processing
    # ------------------------------------------------------------------
    def process(
        self,
        max_frames: int = -1,
        frame_skip: int = 0,
        visualize_every: int = 0,
        export_json: Optional[str] = None,
    ) -> List[FrameData]:
        """Run the full pipeline over the loaded video.

        Args:
            max_frames: cap total frames processed (-1 = all).
            frame_skip: process every Nth frame (0 = every frame).
            visualize_every: save a pitch plot every N frames (0 = off).
            export_json: if set, write JSON to this path (relative to output_dir).
        """
        if self._cap is None:
            raise RuntimeError("No video loaded — call load_video() first")
        if self.transformer is None:
            raise RuntimeError("No homography set — call set_homography() first")

        estimated_total = self._total_frames
        if frame_skip > 0:
            estimated_total = estimated_total // (frame_skip + 1)
        if max_frames > 0:
            estimated_total = min(estimated_total, max_frames)

        all_frames: List[FrameData] = []
        frame_idx = 0
        processed = 0
        last_report_pct = -1
        t0 = time.time()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if max_frames > 0 and processed >= max_frames:
                break

            if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                frame_idx += 1
                continue

            fd = self._process_frame(frame, frame_idx)
            all_frames.append(fd)

            if visualize_every > 0 and frame_idx % visualize_every == 0:
                PitchVisualizer.draw(
                    fd,
                    save_path=str(self.exporter.output_dir / f"frame_{frame_idx:06d}.png"),
                )
                print(f"[viz] saved frame {frame_idx} plot")

            processed += 1
            frame_idx += 1

            # Print-based progress every 10%
            if estimated_total > 0:
                pct = int(processed / estimated_total * 100)
                if pct >= last_report_pct + 10:
                    print(f"[process] {pct}% — {processed}/{estimated_total} frames")
                    last_report_pct = pct

        elapsed = time.time() - t0
        print(
            f"[process] {processed} frames in {elapsed:.1f}s "
            f"({processed / elapsed:.1f} fps)"
        )

        # ── Interpolation pass ────────────────────────────────────────────
        print("[interpolate] Filling gaps in player/ball tracks...")
        all_frames = self.interpolator.interpolate(all_frames)
        interp_players = sum(
            sum(1 for p in fd.players if p.interpolated) for fd in all_frames
        )
        interp_ball = sum(1 for fd in all_frames if fd.ball_interpolated)
        print(
            f"[interpolate] Added {interp_players} interpolated player positions, "
            f"{interp_ball} interpolated ball positions"
        )

        # ── Add all frames to exporter (after interpolation) ─────────────
        for fd in all_frames:
            self.exporter.add_frame(fd, confidence=fd.confidence)

        if export_json:
            path = self.exporter.save(export_json)
            print(f"[export] saved to {path}")

        return all_frames

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameData:
        timestamp_ms = int(frame_idx / self._fps * 1000)

        detections = self.detector.detect(frame)
        mean_conf = 1.0
        if len(detections) == 0:
            return FrameData(frame_idx=frame_idx, timestamp_ms=timestamp_ms)

        if detections.confidence is not None and len(detections.confidence) > 0:
            mean_conf = float(np.mean(detections.confidence))

        fd = FrameData(frame_idx=frame_idx, timestamp_ms=timestamp_ms, confidence=mean_conf)

        detections = self.tracker.track(detections)

        if frame_idx == 0:
            teams = self.classifier.fit_predict(frame, detections)
        else:
            teams = self.classifier.predict(frame, detections)

        # ------------------------------------------------------------------
        # 1) Recolectar candidatos en buffers separados por categoría.
        # 2) Aplicar FILTRO ESPACIAL (descartar fuera del campo).
        # 3) Aplicar LÍMITE POR EQUIPO (top-N por confianza).
        # 4) Aplicar BALÓN ÚNICO (top-1 por confianza).
        # ------------------------------------------------------------------
        # Márgenes en metros para considerar "dentro del campo".
        # 5m de gracia para tirador de banda, esquinas, etc.
        FIELD_MARGIN_M = 5.0
        x_min, x_max = -FIELD_MARGIN_M, PITCH_LENGTH_M + FIELD_MARGIN_M
        y_min, y_max = -FIELD_MARGIN_M, PITCH_WIDTH_M + FIELD_MARGIN_M
        # El balón puede salir más lejos (saques largos, tiros desviados)
        BALL_MARGIN_M = 12.0
        bx_min, bx_max = -BALL_MARGIN_M, PITCH_LENGTH_M + BALL_MARGIN_M
        by_min, by_max = -BALL_MARGIN_M, PITCH_WIDTH_M + BALL_MARGIN_M

        # Buffer: lista de candidatos (conf, tracker_id, x_m, y_m, team)
        home_candidates: List[Tuple[float, int, float, float]] = []
        away_candidates: List[Tuple[float, int, float, float]] = []
        ref_candidates: List[Tuple[float, int, float, float]] = []
        unknown_candidates: List[Tuple[float, int, float, float, str]] = []
        ball_candidates: List[Tuple[float, float, float]] = []  # (conf, x, y)

        confidences = (
            detections.confidence if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )

        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            class_id = detections.class_id[i] if detections.class_id is not None else -1
            tracker_id = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else i
            )
            conf = float(confidences[i]) if i < len(confidences) else 1.0

            bp = self.transformer.player_base_point(xyxy)
            x, y = self.transformer.pixel_to_pitch(*bp)

            # --- BALÓN ---
            if class_id == 32:
                # Filtro espacial: balón debe estar cerca del campo
                if bx_min <= x <= bx_max and by_min <= y <= by_max:
                    ball_candidates.append((conf, x, y))
                continue

            # --- PERSONAS ---
            # Filtro espacial DURO: si la proyección cae lejos del campo,
            # casi seguro es banca, fotógrafo, fan, asistente, etc.
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue

            team_label = teams[i] if i < len(teams) else "unknown"
            if team_label == "home":
                home_candidates.append((conf, tracker_id, x, y))
            elif team_label == "away":
                away_candidates.append((conf, tracker_id, x, y))
            elif team_label == "referee":
                ref_candidates.append((conf, tracker_id, x, y))
            else:
                unknown_candidates.append((conf, tracker_id, x, y, team_label))

        # ---- Limitar por equipo: top-N por confianza ----
        # Si hay más detecciones de las posibles físicamente, las extra son
        # casi seguro tracks fantasma o duplicados (banca/asistentes).
        def _take_top(buf, n: int):
            buf.sort(key=lambda t: t[0], reverse=True)
            return buf[:n]

        home_top = _take_top(home_candidates, self.max_per_team)
        away_top = _take_top(away_candidates, self.max_per_team)
        ref_top = _take_top(ref_candidates, self.max_refs)
        # Unknown: pequeño margen, podría ser jugador mal clasificado
        unknown_top = sorted(unknown_candidates, key=lambda t: t[0], reverse=True)[:3]

        # ---- Volcar a fd.players ----
        def _push(items, team: str):
            for tup in items:
                if len(tup) == 4:
                    conf_, trk, xx, yy = tup
                    team_lbl = team
                else:
                    conf_, trk, xx, yy, team_lbl = tup
                fd.players.append(Player2D(
                    id=self.identity_map.get(trk, trk),
                    x=round(xx, 3),
                    y=round(yy, 3),
                    team=team_lbl,
                    interpolated=False,
                ))

        _push(home_top, "home")
        _push(away_top, "away")
        _push(ref_top, "referee")
        _push(unknown_top, "unknown")

        # ---- Balón: solo el de mayor confidence ----
        if ball_candidates:
            ball_candidates.sort(key=lambda t: t[0], reverse=True)
            _, bx, by = ball_candidates[0]
            fd.ball = (round(bx, 3), round(by, 3))

        return fd


# ---------------------------------------------------------------------------
# Utility: Manual Keypoint Annotation Tool
# ---------------------------------------------------------------------------


def annotate_keypoints(
    video_path: str,
    frame_number: int = 0,
    num_points: int = 4,
) -> np.ndarray:
    """Open a video frame and let the user click N keypoints.

    Returns an (N, 2) float32 array of pixel coordinates.
    Click the points in order; press ESC to discard; ENTER to confirm.

    Typical keypoint order (football pitch):
      0: near touchline / goal-line intersection (left)
      1: near touchline / goal-line intersection (right)
      2: far  touchline / goal-line intersection (right)
      3: far  touchline / goal-line intersection (left)
    (adjust to whatever four coplanar points you can identify clearly.)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total:
        cap.release()
        raise IndexError(f"Frame {frame_number} out of range ({total} frames)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_number}")

    points: List[Tuple[int, int]] = []
    window = "Annotate Keypoints (click in order, ENTER to confirm, ESC to discard)"

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_click)

    print(f"Click {num_points} points on the frame in order. ENTER=confirm, ESC=discard")

    while True:
        disp = frame.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(disp, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(
                disp, str(i), (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            if len(points) >= num_points:
                break
            print(f"Need {num_points} points, only have {len(points)}")
        elif key == 27:  # ESC
            points.clear()
            break

    cv2.destroyAllWindows()

    if len(points) < num_points:
        raise RuntimeError("Annotation cancelled or incomplete")

    print("Annotated source points:")
    for i, (px, py) in enumerate(points):
        print(f"  {i}: ({px}, {py})")
    return np.array(points, dtype=np.float32)
