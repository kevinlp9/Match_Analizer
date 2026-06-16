"""
event_detector.py — Fase 7 (Sub-fases A, B, C)
Detección automática de eventos desde frames de tracking.

Detecta:
  - ball_out (7.A): balón fuera del campo ≥ N frames consecutivos
  - goal (7.B): heurística multi-señal (≥3 de 4 señales)
  - pass / interception (7.C): cambio de posesor del balón

Todos los eventos generados llevan:
  - confidence: float [0, 1]
  - requires_review: True  (el admin confirma o rechaza desde la UI)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de campo
# ──────────────────────────────────────────────────────────────────────────────
PITCH_W = 105.0      # metros
PITCH_H = 68.0       # metros
GOAL_Y_MIN = (PITCH_H / 2) - 3.66   # 30.34 m
GOAL_Y_MAX = (PITCH_H / 2) + 3.66   # 37.66 m
CENTER_X = PITCH_W / 2              # 52.5 m
CENTER_Y = PITCH_H / 2              # 34.0 m


@dataclass
class PlayerPos:
    """Posición de un jugador en un frame."""
    player_id: int
    team: str       # "home" | "away" | "referee" | "unknown"
    x: float
    y: float


@dataclass
class FrameSnapshot:
    """Representación mínima de un frame para el detector."""
    frame_idx: int
    timestamp_ms: int
    ball: Optional[Tuple[float, float]]   # (x, y) en metros, o None
    players: List[PlayerPos]


# ──────────────────────────────────────────────────────────────────────────────
# 7.A — Balón fuera de campo
# ──────────────────────────────────────────────────────────────────────────────

def detect_ball_out_events(frames: List[FrameSnapshot], out_buffer: int = 5) -> List[Dict[str, Any]]:
    """
    Detecta cuando el balón sale del campo durante ≥ out_buffer frames consecutivos.
    Clasifica como 'throw_in' (salió por lateral) o 'goal_kick_or_corner' (salió por línea de fondo).
    """
    events: List[Dict[str, Any]] = []
    out_streak = 0
    last_inside_idx = -1

    for i, fd in enumerate(frames):
        if fd.ball is None:
            continue
        x, y = fd.ball
        is_inside = (0.0 <= x <= PITCH_W) and (0.0 <= y <= PITCH_H)

        if is_inside:
            out_streak = 0
            last_inside_idx = i
        else:
            out_streak += 1
            if out_streak == out_buffer:
                # Clasificar por dónde salió
                if last_inside_idx >= 0 and frames[last_inside_idx].ball is not None:
                    last_x, _ = frames[last_inside_idx].ball  # type: ignore[misc]
                    out_type = "goal_kick_or_corner" if (last_x < 2 or last_x > PITCH_W - 2) else "throw_in"
                else:
                    out_type = "throw_in"

                start_idx = max(0, i - out_buffer)
                events.append({
                    "frame_idx": start_idx,
                    "timestamp_ms": frames[start_idx].timestamp_ms,
                    "event_type": "ball_out",
                    "subtype": out_type,
                    "ball_x": float(x),
                    "ball_y": float(y),
                    "confidence": 0.80,
                    "requires_review": True,
                })

    return events


# ──────────────────────────────────────────────────────────────────────────────
# 7.B — Gol (heurística multi-señal)
# ──────────────────────────────────────────────────────────────────────────────

def detect_goal_events(frames: List[FrameSnapshot], fps: float = 5.0) -> List[Dict[str, Any]]:
    """
    Detecta goles candidatos combinando ≥3 de 4 señales:
      1. Balón cruzó la línea de gol dentro del ancho de la portería
      2. Reanudación desde el centro (≤30s después)
      3. Aglomeración de jugadores cerca de la portería (≤16s después)
      4. Pausa de movimiento general (≤20s después)
    """
    events: List[Dict[str, Any]] = []
    i = 0
    skip_until = -1

    while i < len(frames):
        if i < skip_until:
            i += 1
            continue

        fd = frames[i]
        if fd.ball is None:
            i += 1
            continue

        x, y = fd.ball

        # Señal 1: cruzó la línea de gol
        crossed = (x < 0 or x > PITCH_W) and (GOAL_Y_MIN <= y <= GOAL_Y_MAX)
        if not crossed:
            i += 1
            continue

        side = "home_goal" if x > PITCH_W else "away_goal"
        signals = 1

        look_ahead_30s = int(fps * 30)
        look_ahead_16s = int(fps * 16)
        look_ahead_20s = int(fps * 20)
        window_end = min(len(frames), i + look_ahead_30s)

        # Señal 2: reanudación desde el centro
        for fd2 in frames[i + 1:window_end]:
            if fd2.ball is None:
                continue
            bx, by = fd2.ball
            if abs(bx - CENTER_X) < 5 and abs(by - CENTER_Y) < 5:
                signals += 1
                break

        # Señal 3: aglomeración ≥6 jugadores cerca de la portería (dentro de 20m)
        goal_x = PITCH_W if side == "home_goal" else 0.0
        for fd2 in frames[i + 1:min(len(frames), i + look_ahead_16s)]:
            near = sum(1 for p in fd2.players if abs(p.x - goal_x) < 20)
            if near >= 6:
                signals += 1
                break

        # Señal 4: pausa de movimiento (velocidad media de jugadores ≈ 0)
        speeds = []
        prev_positions: Dict[int, Tuple[float, float]] = {}
        for fd2 in frames[i + 1:min(len(frames), i + look_ahead_20s)]:
            for p in fd2.players:
                if p.player_id in prev_positions:
                    px, py = prev_positions[p.player_id]
                    spd = ((p.x - px) ** 2 + (p.y - py) ** 2) ** 0.5
                    speeds.append(spd)
                prev_positions[p.player_id] = (p.x, p.y)

        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            if avg_speed < 0.3:   # muy poca velocidad → jugadores parados
                signals += 1

        if signals >= 3:
            confidence = 0.60 + 0.10 * (signals - 3)   # 0.60, 0.70, 0.80 según señales
            events.append({
                "frame_idx": i,
                "timestamp_ms": fd.timestamp_ms,
                "event_type": "goal",
                "subtype": side,
                "ball_x": float(x),
                "ball_y": float(y),
                "confidence": round(min(0.90, confidence), 2),
                "signals": signals,
                "requires_review": True,
            })
            skip_until = i + look_ahead_30s   # evitar duplicados dentro de 30s

        i += 1

    return events


# ──────────────────────────────────────────────────────────────────────────────
# 7.C — Pases / Interceptaciones
# ──────────────────────────────────────────────────────────────────────────────

def detect_pass_events(
    frames: List[FrameSnapshot],
    max_ball_dist_m: float = 3.0,
    min_duration_frames: int = 3,
) -> List[Dict[str, Any]]:
    """
    Detecta pases rastreando el posesor más cercano al balón.
    - Cuando el posesor cambia entre jugadores del mismo equipo → pass
    - Cuando cambia de equipo → interception
    - Filtra cambios muy rápidos (< min_duration_frames) para reducir ruido.
    """
    events: List[Dict[str, Any]] = []

    @dataclass
    class Carrier:
        player_id: int
        team: str
        frame_idx: int
        x_ball: float
        y_ball: float

    last_carrier: Optional[Carrier] = None
    carrier_since: int = 0

    for i, fd in enumerate(frames):
        if fd.ball is None:
            continue
        bx, by = fd.ball

        # Jugador más cercano dentro de max_ball_dist_m
        nearest: Optional[PlayerPos] = None
        nearest_d = max_ball_dist_m
        for p in fd.players:
            if p.team in ("referee", "unknown"):
                continue
            d = ((p.x - bx) ** 2 + (p.y - by) ** 2) ** 0.5
            if d < nearest_d:
                nearest_d = d
                nearest = p

        if nearest is None:
            last_carrier = None
            continue

        if last_carrier is None:
            last_carrier = Carrier(nearest.player_id, nearest.team, i, bx, by)
            carrier_since = i
            continue

        if nearest.player_id != last_carrier.player_id:
            duration = i - carrier_since
            if duration >= min_duration_frames:
                same_team = last_carrier.team == nearest.team
                events.append({
                    "frame_idx": last_carrier.frame_idx,
                    "timestamp_ms": frames[last_carrier.frame_idx].timestamp_ms,
                    "event_type": "pass" if same_team else "interception",
                    "from_player_id": last_carrier.player_id,
                    "to_player_id": nearest.player_id,
                    "from_team": last_carrier.team,
                    "to_team": nearest.team,
                    "outcome": "complete" if same_team else "failed",
                    "x_start": last_carrier.x_ball,
                    "y_start": last_carrier.y_ball,
                    "x_end": float(bx),
                    "y_end": float(by),
                    "confidence": 0.70,
                    "requires_review": False,   # pases no requieren revisión por defecto
                })

            last_carrier = Carrier(nearest.player_id, nearest.team, i, bx, by)
            carrier_since = i

    return events


# ──────────────────────────────────────────────────────────────────────────────
# Función principal: detectar todos los eventos
# ──────────────────────────────────────────────────────────────────────────────

def detect_all_events(
    frames: List[FrameSnapshot],
    fps: float = 5.0,
    detect_out: bool = True,
    detect_goals: bool = True,
    detect_passes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Versión LEGACY: ejecuta solo 3 detectores (ball_out, goals, passes).
    Mantenida para retrocompatibilidad. Para versión exhaustiva usar
    detect_all_exhaustive().
    """
    all_events: List[Dict[str, Any]] = []

    if detect_out:
        all_events.extend(detect_ball_out_events(frames))

    if detect_goals:
        all_events.extend(detect_goal_events(frames, fps=fps))

    if detect_passes:
        all_events.extend(detect_pass_events(frames))

    all_events.sort(key=lambda e: e.get("timestamp_ms", 0))
    return all_events


# ──────────────────────────────────────────────────────────────────────────────
# 7.B EXHAUSTIVO — detectores adicionales
# ──────────────────────────────────────────────────────────────────────────────
#
# Cubre TODOS los eventType del enum MatchEvent excepto var_review/offside:
#   shot, clearance, dribble, foul, yellow_card, red_card, substitution,
#   cross, key_pass, own_goal, penalty_scored
#
# Filosofía:
#   - Marca cada evento con confidence (0.4 - 0.9)
#   - requires_review = True si confidence < 0.7 o si requiere interpretación
#   - El admin abre una timeline editable y CORRIGE el output del AI
# ──────────────────────────────────────────────────────────────────────────────

PENALTY_SPOT_HOME = (PITCH_W - 11.0, CENTER_Y)   # portería en x = PITCH_W
PENALTY_SPOT_AWAY = (11.0, CENTER_Y)             # portería en x = 0


def _nearest_player_pos(fd: FrameSnapshot, max_dist: float = 3.0) -> Optional[PlayerPos]:
    """Jugador más cercano al balón dentro de max_dist metros."""
    if fd.ball is None:
        return None
    bx, by = fd.ball
    best, best_d = None, max_dist
    for p in fd.players:
        d = ((p.x - bx) ** 2 + (p.y - by) ** 2) ** 0.5
        if d < best_d:
            best, best_d = p, d
    return best


def _ball_speed_ms(frames: List[FrameSnapshot], i: int) -> float:
    """Velocidad del balón en m/s en el frame i (vs i-1)."""
    if i == 0 or frames[i].ball is None or frames[i - 1].ball is None:
        return 0.0
    x1, y1 = frames[i].ball
    x0, y0 = frames[i - 1].ball
    dt = (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000.0
    if dt <= 0:
        return 0.0
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt


def _avg_player_speed(frames: List[FrameSnapshot], i: int, lookback: int = 3) -> float:
    """Velocidad promedio de TODOS los jugadores en los últimos lookback frames."""
    if i < lookback:
        return 0.0
    total = 0.0
    n = 0
    for j in range(i - lookback + 1, i + 1):
        prev = {p.player_id: (p.x, p.y) for p in frames[j - 1].players} if j > 0 else {}
        dt = (frames[j].timestamp_ms - frames[j - 1].timestamp_ms) / 1000.0 if j > 0 else 0
        if dt <= 0:
            continue
        for p in frames[j].players:
            if p.player_id in prev:
                px, py = prev[p.player_id]
                total += ((p.x - px) ** 2 + (p.y - py) ** 2) ** 0.5 / dt
                n += 1
    return total / n if n > 0 else 0.0


def _build_carrier_timeline(frames: List[FrameSnapshot]) -> List[Dict[str, Any]]:
    """Para cada frame: quién tiene el balón (None si está libre)."""
    timeline: List[Dict[str, Any]] = []
    for fd in frames:
        carrier = _nearest_player_pos(fd, max_dist=3.0)
        timeline.append({
            "frame_idx": fd.frame_idx,
            "ts": fd.timestamp_ms,
            "player_id": carrier.player_id if carrier else None,
            "team": carrier.team if carrier else None,
            "x": carrier.x if carrier else None,
            "y": carrier.y if carrier else None,
            "ball_x": fd.ball[0] if fd.ball else None,
            "ball_y": fd.ball[1] if fd.ball else None,
        })
    return timeline


# ── Shot ───────────────────────────────────────────────────────────────────

def detect_shot_events(frames: List[FrameSnapshot]) -> List[Dict[str, Any]]:
    """Tiros: balón acelera >12 m/s en dirección a portería desde un carrier."""
    events: List[Dict[str, Any]] = []
    last_shot_ts = 0

    for i in range(1, len(frames)):
        speed = _ball_speed_ms(frames, i)
        if speed < 12.0:
            continue
        if frames[i].ball is None or frames[i - 1].ball is None:
            continue

        x0, y0 = frames[i - 1].ball
        x1, y1 = frames[i].ball
        dx = x1 - x0
        if abs(dx) < 0.1:
            continue

        # ¿Va a alguna portería?
        if dx > 0 and x1 > 60:
            target_x = PITCH_W
            target_side = "away"  # apunta a portería del visitante
        elif dx < 0 and x1 < 45:
            target_x = 0.0
            target_side = "home"
        else:
            continue

        # Proyectar trayectoria al goal-line
        t_proj = (target_x - x1) / dx if dx != 0 else 0
        y_at_goal = y1 + (y1 - y0) * t_proj
        if not (GOAL_Y_MIN - 6 < y_at_goal < GOAL_Y_MAX + 6):
            continue

        # Buscar el shooter (carrier en frames anteriores)
        shooter: Optional[PlayerPos] = None
        for j in range(i - 1, max(i - 5, 0), -1):
            np_ = _nearest_player_pos(frames[j], max_dist=2.5)
            if np_:
                shooter = np_
                break
        if not shooter:
            continue

        # Anti-duplicado: no registrar otro tiro en <2 segundos
        if frames[i].timestamp_ms - last_shot_ts < 2000:
            continue
        last_shot_ts = frames[i].timestamp_ms

        events.append({
            "event_type": "shot",
            "timestamp_ms": frames[i].timestamp_ms,
            "frame_idx": i,
            "player_id_candidate": shooter.player_id,
            "team_side": shooter.team,
            "x_start": float(shooter.x),
            "y_start": float(shooter.y),
            "x_end": float(x1),
            "y_end": float(y1),
            "ball_speed_ms": round(speed, 1),
            "confidence": round(0.55 + min(0.25, (speed - 12) / 30), 3),
            "requires_review": True,
        })
    return events


# ── Pass / Interception / Cross (versión exhaustiva mejorada) ──────────────

def detect_passes_exhaustive(
    frames: List[FrameSnapshot],
    timeline: List[Dict[str, Any]],
    min_distance_m: float = 5.0,
    min_duration_s: float = 0.4,
) -> List[Dict[str, Any]]:
    """Pases, interceptaciones y cruces con filtros para reducir ruido."""
    events: List[Dict[str, Any]] = []
    last_solid: Optional[Dict[str, Any]] = None

    for t in timeline:
        if t["player_id"] is None:
            continue

        if last_solid and t["player_id"] == last_solid["player_id"]:
            continue

        if last_solid:
            duration_s = (t["ts"] - last_solid["ts"]) / 1000.0
            if duration_s < min_duration_s:
                last_solid = t
                continue

            same_team = last_solid["team"] == t["team"]
            bx0 = last_solid["ball_x"] or 0
            by0 = last_solid["ball_y"] or 0
            bx1 = t["ball_x"] or 0
            by1 = t["ball_y"] or 0
            distance = ((bx1 - bx0) ** 2 + (by1 - by0) ** 2) ** 0.5
            if distance < min_distance_m:
                last_solid = t
                continue

            is_cross = (
                same_team
                and (by0 < 12 or by0 > (PITCH_H - 12))
                and (bx1 < 18 or bx1 > (PITCH_W - 18))
            )

            event_type = "cross" if is_cross else ("pass" if same_team else "interception")
            events.append({
                "event_type": event_type,
                "timestamp_ms": last_solid["ts"],
                "frame_idx": last_solid["frame_idx"],
                "player_id_candidate": last_solid["player_id"],
                "related_player_id_candidate": t["player_id"],
                "team_side": last_solid["team"],
                "x_start": bx0,
                "y_start": by0,
                "x_end": bx1,
                "y_end": by1,
                "outcome": "complete" if same_team else "failed",
                "distance_m": round(distance, 1),
                "confidence": 0.55 if same_team else 0.50,
                "requires_review": False,
            })

        last_solid = t
    return events


def detect_key_pass_events(
    passes: List[Dict[str, Any]],
    shots: List[Dict[str, Any]],
    max_gap_s: float = 5.0,
) -> List[Dict[str, Any]]:
    """Promueve pases a key_pass si el receptor disparó <max_gap_s después."""
    out: List[Dict[str, Any]] = []
    for sh in shots:
        for pa in passes:
            if pa["event_type"] != "pass":
                continue
            if pa.get("related_player_id_candidate") != sh.get("player_id_candidate"):
                continue
            dt = (sh["timestamp_ms"] - pa["timestamp_ms"]) / 1000.0
            if 0 < dt < max_gap_s:
                kp = dict(pa)
                kp["event_type"] = "key_pass"
                kp["confidence"] = 0.65
                out.append(kp)
                break
    return out


# ── Clearance ─────────────────────────────────────────────────────────────

def detect_clearance_events(frames: List[FrameSnapshot]) -> List[Dict[str, Any]]:
    """Despeje: balón en zona defensiva acelera >18 m/s hacia campo contrario."""
    events: List[Dict[str, Any]] = []
    last_ts = 0
    for i in range(1, len(frames)):
        if frames[i].ball is None or frames[i - 1].ball is None:
            continue
        speed = _ball_speed_ms(frames, i)
        if speed < 18.0:
            continue
        bx_prev = frames[i - 1].ball[0]
        bx = frames[i].ball[0]
        in_def_home = bx_prev < 25
        in_def_away = bx_prev > (PITCH_W - 25)
        if not (in_def_home or in_def_away):
            continue
        outgoing = (in_def_home and bx > bx_prev) or (in_def_away and bx < bx_prev)
        if not outgoing:
            continue
        carrier = _nearest_player_pos(frames[i - 1], max_dist=3.0)
        if not carrier:
            continue
        if frames[i].timestamp_ms - last_ts < 1500:
            continue
        last_ts = frames[i].timestamp_ms

        events.append({
            "event_type": "clearance",
            "timestamp_ms": frames[i].timestamp_ms,
            "frame_idx": i,
            "player_id_candidate": carrier.player_id,
            "team_side": carrier.team,
            "x_start": float(carrier.x),
            "y_start": float(carrier.y),
            "ball_speed_ms": round(speed, 1),
            "confidence": 0.55,
            "requires_review": False,
        })
    return events


# ── Dribble ───────────────────────────────────────────────────────────────

def detect_dribble_events(
    frames: List[FrameSnapshot],
    timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Carrier mantiene el balón >2s pasando <2m de ≥2 rivales."""
    events: List[Dict[str, Any]] = []
    i = 0
    while i < len(timeline):
        t = timeline[i]
        if t["player_id"] is None:
            i += 1
            continue

        j = i
        rivals_close = 0
        while j < len(timeline) and timeline[j]["player_id"] == t["player_id"]:
            fd = frames[j]
            bx = timeline[j]["ball_x"] or 0
            by = timeline[j]["ball_y"] or 0
            for p in fd.players:
                if p.team and t["team"] and p.team != t["team"]:
                    d = ((p.x - bx) ** 2 + (p.y - by) ** 2) ** 0.5
                    if d < 2.0:
                        rivals_close += 1
                        break
            j += 1

        if j == i:
            i += 1
            continue

        duration_s = (timeline[j - 1]["ts"] - t["ts"]) / 1000.0
        if duration_s >= 2.0 and rivals_close >= 2:
            events.append({
                "event_type": "dribble",
                "timestamp_ms": t["ts"],
                "frame_idx": t["frame_idx"],
                "player_id_candidate": t["player_id"],
                "team_side": t["team"],
                "x_start": t["x"],
                "y_start": t["y"],
                "duration_s": round(duration_s, 1),
                "rivals_close_count": rivals_close,
                "confidence": 0.55,
                "requires_review": False,
            })
        i = j if j > i else i + 1
    return events


# ── Foul / Yellow / Red ───────────────────────────────────────────────────

def detect_foul_and_card_events(
    frames: List[FrameSnapshot],
    fps: float = 5.0,
) -> List[Dict[str, Any]]:
    """Falta: 2 jugadores de equipos contrarios <1.5m + pausa del juego.

    Diferencia foul / yellow_card / red_card por duración de la pausa:
        2-10 s  → foul
       10-25 s  → yellow_card
       >25 s    → red_card
    """
    events: List[Dict[str, Any]] = []
    i = 0
    while i < len(frames):
        fd = frames[i]
        close_pair: Optional[Tuple[PlayerPos, PlayerPos]] = None
        for a_idx, a in enumerate(fd.players):
            for b in fd.players[a_idx + 1:]:
                if a.team == b.team:
                    continue
                d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
                if d < 1.5:
                    close_pair = (a, b)
                    break
            if close_pair:
                break

        if not close_pair:
            i += 1
            continue

        # Mide la pausa siguiente
        pause_frames = 0
        max_lookahead = min(int(30 * fps), len(frames) - i)
        for j in range(i, i + max_lookahead):
            avg = _avg_player_speed(frames, j, lookback=2)
            if avg < 1.0:
                pause_frames += 1
            else:
                break

        pause_s = pause_frames / fps
        if pause_s >= 2.0:
            a, b = close_pair
            if pause_s >= 25:
                event_type = "red_card"
                conf = 0.40
            elif pause_s >= 10:
                event_type = "yellow_card"
                conf = 0.45
            else:
                event_type = "foul"
                conf = 0.50

            events.append({
                "event_type": event_type,
                "timestamp_ms": fd.timestamp_ms,
                "frame_idx": i,
                "player_id_candidate": b.player_id,            # asumido infractor
                "related_player_id_candidate": a.player_id,    # asumido víctima
                "team_side": b.team,
                "x_start": (a.x + b.x) / 2,
                "y_start": (a.y + b.y) / 2,
                "pause_duration_s": round(pause_s, 1),
                "confidence": conf,
                "requires_review": True,
            })
            i += pause_frames + 1
        else:
            i += 1
    return events


# ── Substitution ──────────────────────────────────────────────────────────

def detect_substitution_events(frames: List[FrameSnapshot]) -> List[Dict[str, Any]]:
    """Track muere antes del final + nuevo track del mismo equipo aparece desde banda."""
    events: List[Dict[str, Any]] = []
    if not frames:
        return events

    presence: Dict[int, List[int]] = {}
    team_by_player: Dict[int, str] = {}
    for i, fd in enumerate(frames):
        for p in fd.players:
            presence.setdefault(p.player_id, []).append(i)
            if p.player_id not in team_by_player and p.team:
                team_by_player[p.player_id] = p.team

    for pid, indices in presence.items():
        if len(indices) < 30:
            continue
        last_idx = indices[-1]
        if last_idx >= len(frames) - 150:
            continue   # quedó vivo hasta el final

        team = team_by_player.get(pid)
        if not team:
            continue

        for new_pid, new_indices in presence.items():
            if new_pid == pid:
                continue
            first_new = new_indices[0]
            if first_new < last_idx or first_new > last_idx + 300:
                continue
            if team_by_player.get(new_pid) != team:
                continue
            np_ = next((p for p in frames[first_new].players if p.player_id == new_pid), None)
            if not np_:
                continue
            if np_.y < 8 or np_.y > (PITCH_H - 8):
                events.append({
                    "event_type": "substitution",
                    "timestamp_ms": frames[last_idx].timestamp_ms,
                    "frame_idx": last_idx,
                    "player_id_candidate": pid,                    # sale
                    "related_player_id_candidate": new_pid,        # entra
                    "team_side": team,
                    "confidence": 0.50,
                    "requires_review": True,
                })
                break
    return events


# ── Refinamiento de goles: own_goal y penalty_scored ──────────────────────

def refine_goals_with_own_and_penalty(
    goal_events: List[Dict[str, Any]],
    frames: List[FrameSnapshot],
) -> List[Dict[str, Any]]:
    """Toma eventos 'goal' y los clasifica como own_goal o penalty_scored si aplica."""
    refined: List[Dict[str, Any]] = []
    for ge in goal_events:
        if ge.get("event_type") != "goal":
            refined.append(ge)
            continue

        i = ge["frame_idx"]
        if i >= len(frames) or frames[i].ball is None:
            refined.append(ge)
            continue

        bx, _ = frames[i].ball
        scoring_side = ge.get("team_side") or ("home" if bx > PITCH_W else "away")

        # ¿own_goal? El último carrier antes del cruce era del equipo defensor
        last_team = None
        last_player_id = None
        for j in range(i - 1, max(i - 30, 0), -1):
            np_ = _nearest_player_pos(frames[j])
            if np_:
                last_team = np_.team
                last_player_id = np_.player_id
                break

        own_goal = bool(last_team) and last_team != scoring_side

        # ¿penalty? Balón quieto en el penalty spot antes
        spot = PENALTY_SPOT_HOME if scoring_side == "home" else PENALTY_SPOT_AWAY
        penalty = False
        still = 0
        for j in range(max(i - 50, 0), i):
            if frames[j].ball is None:
                continue
            bxs, bys = frames[j].ball
            if abs(bxs - spot[0]) < 2.0 and abs(bys - spot[1]) < 2.0:
                still += 1
            else:
                still = 0
            if still >= 12:
                penalty = True
                break

        new_ev = dict(ge)
        if own_goal:
            new_ev["event_type"] = "own_goal"
            # El equipo que "anota" en sentido del marcador es el contrario
            new_ev["team_side"] = "away" if scoring_side == "home" else "home"
            new_ev["player_id_candidate"] = last_player_id
            new_ev["confidence"] = min(0.65, new_ev.get("confidence", 0.6))
        elif penalty:
            new_ev["event_type"] = "penalty_scored"
            new_ev["confidence"] = max(new_ev.get("confidence", 0.6), 0.70)
        refined.append(new_ev)
    return refined


# ──────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR EXHAUSTIVO — usar este desde el worker
# ──────────────────────────────────────────────────────────────────────────────

def detect_all_exhaustive(
    frames: List[FrameSnapshot],
    fps: float = 5.0,
) -> List[Dict[str, Any]]:
    """Ejecuta TODOS los detectores y devuelve lista plana ordenada por tiempo.

    Incluye:
      - ball_out (throw_in / corner_won / goal_kick implícitos en subtype)
      - goal / own_goal / penalty_scored
      - shot
      - pass / interception / cross
      - key_pass
      - clearance
      - dribble
      - foul / yellow_card / red_card
      - substitution
    """
    if not frames:
        return []

    timeline = _build_carrier_timeline(frames)

    ball_outs = detect_ball_out_events(frames)
    raw_goals = detect_goal_events(frames, fps=fps)
    refined_goals = refine_goals_with_own_and_penalty(raw_goals, frames)
    shots = detect_shot_events(frames)
    passes_and_int = detect_passes_exhaustive(frames, timeline)
    passes_only = [e for e in passes_and_int if e["event_type"] == "pass"]
    key_passes = detect_key_pass_events(passes_only, shots)
    clearances = detect_clearance_events(frames)
    dribbles = detect_dribble_events(frames, timeline)
    fouls_cards = detect_foul_and_card_events(frames, fps=fps)
    subs = detect_substitution_events(frames)

    all_events = (
        ball_outs + refined_goals + shots + passes_and_int + key_passes
        + clearances + dribbles + fouls_cards + subs
    )
    all_events.sort(key=lambda e: e.get("timestamp_ms", 0))

    # Log resumen
    by_type: Dict[str, int] = {}
    for ev in all_events:
        et = ev.get("event_type", "?")
        by_type[et] = by_type.get(et, 0) + 1

    print(f"[events] TOTAL detectados (exhaustivo): {len(all_events)}")
    for et in sorted(by_type.keys()):
        print(f"  - {et}: {by_type[et]}")

    return all_events

