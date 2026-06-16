"""Advanced tactical analytics for Geo-Goal Local.

Computes per-player, per-team, spatial, and possession statistics
from match_data.json tracking data.

Statistics computed:
  - Per-player: distance, speed, sprints, time-by-third, heatmap
  - Per-team: centroid, amplitude, depth, compaction, convex-hull area,
    defensive-line height
  - Voronoi spatial control percentage
  - Ball possession by closest-player rule
  - Pressure matrix (opponents within 5 m)
  - Formation detection via gap-based clustering
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Default pitch dimensions (metres)
PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0

# Physics / filtering constants
MAX_PLAYER_SPEED_MS: float = 10.0  # ~36 km/h – discard teleportations
SPRINT_THRESHOLD_MS: float = 7.0   # ~25 km/h
MIN_SPRINT_FRAMES: int = 3

# Heatmap grid resolution
HEATMAP_COLS: int = 21
HEATMAP_ROWS: int = 14

# Voronoi grid resolution (2 m)
VORONOI_STEP: float = 2.0

# Formation clustering gap threshold (metres along X)
FORMATION_GAP_THRESHOLD: float = 8.0

# Pressure radius (metres)
PRESSURE_RADIUS: float = 5.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone-chain convex-hull algorithm.

    Returns the vertices of the convex hull in counter-clockwise order.
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """Shoelace formula for the area of a simple polygon."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


# ---------------------------------------------------------------------------
# Track extraction
# ---------------------------------------------------------------------------

def _build_tracks(frames: List[dict]) -> Tuple[
    Dict[int, List[Tuple[int, float, float, int]]],
    List[Tuple[int, float, float]],
]:
    """Extract per-player tracks and ball track from raw frames.

    Returns
    -------
    player_tracks : {playerId: [(timestampMs, x, y, teamId), ...]}
    ball_track    : [(timestampMs, x, y), ...]
    """
    player_tracks: Dict[int, List[Tuple[int, float, float, int]]] = {}
    ball_track: List[Tuple[int, float, float]] = []

    for frame in frames:
        ts = frame["timestampMs"]

        ball = frame.get("ball")
        if ball is not None:
            ball_track.append((ts, ball["x"], ball["y"]))

        for p in frame.get("players", []):
            pid = p["playerId"]
            tid = p["teamId"]
            player_tracks.setdefault(pid, []).append((ts, p["x"], p["y"], tid))

    return player_tracks, ball_track


# ---------------------------------------------------------------------------
# Per-player statistics
# ---------------------------------------------------------------------------

def _player_stats(
    track: List[Tuple[int, float, float, int]],
    pitch_length: float,
    pitch_width: float,
) -> dict:
    """Compute all per-player stats from a single player's track.

    Parameters
    ----------
    track : list of (timestampMs, x, y, teamId)
        Sorted chronologically by the caller (frames are assumed ordered).
    pitch_length, pitch_width : field dimensions in metres.

    Returns total distance, avg/max speed, sprint count,
    time-by-third, and heatmap.
    """
    total_distance = 0.0
    max_speed_ms = 0.0
    speeds: List[float] = []  # per-pair instantaneous speed (m/s)
    third_boundary_1 = pitch_length / 3.0
    third_boundary_2 = 2.0 * pitch_length / 3.0
    time_thirds = {"defensive": 0.0, "middle": 0.0, "attacking": 0.0}

    # Heatmap accumulator
    heatmap = np.zeros((HEATMAP_ROWS, HEATMAP_COLS), dtype=np.float64)
    cell_w = pitch_length / HEATMAP_COLS
    cell_h = pitch_width / HEATMAP_ROWS

    total_time_s = 0.0

    for i in range(len(track)):
        ts_i, x_i, y_i, _ = track[i]

        # Heatmap bin
        col = min(int(x_i / cell_w), HEATMAP_COLS - 1)
        row = min(int(y_i / cell_h), HEATMAP_ROWS - 1)
        col = max(col, 0)
        row = max(row, 0)
        heatmap[row, col] += 1.0

        if i == 0:
            continue

        ts_prev, x_prev, y_prev, _ = track[i - 1]
        dt_s = (ts_i - ts_prev) / 1000.0
        if dt_s <= 0:
            continue

        dist = _euclidean(x_prev, y_prev, x_i, y_i)
        speed = dist / dt_s

        # Filter impossible jumps
        if speed > MAX_PLAYER_SPEED_MS:
            speeds.append(0.0)
            continue

        total_distance += dist
        total_time_s += dt_s
        if speed > max_speed_ms:
            max_speed_ms = speed
        speeds.append(speed)

        # Time by third (attribute dt to the current position)
        if x_i < third_boundary_1:
            time_thirds["defensive"] += dt_s
        elif x_i < third_boundary_2:
            time_thirds["middle"] += dt_s
        else:
            time_thirds["attacking"] += dt_s

    # Average speed
    avg_speed_ms = (total_distance / total_time_s) if total_time_s > 0 else 0.0

    # Sprint count: sustained speed > threshold for >= MIN_SPRINT_FRAMES frames
    sprint_count = 0
    consecutive = 0
    for spd in speeds:
        if spd >= SPRINT_THRESHOLD_MS:
            consecutive += 1
        else:
            if consecutive >= MIN_SPRINT_FRAMES:
                sprint_count += 1
            consecutive = 0
    if consecutive >= MIN_SPRINT_FRAMES:
        sprint_count += 1

    # Normalise heatmap to 0-1
    hm_max = heatmap.max()
    if hm_max > 0:
        heatmap /= hm_max

    return {
        "totalDistanceM": round(total_distance, 2),
        "avgSpeedKmh": round(avg_speed_ms * 3.6, 2),
        "maxSpeedKmh": round(max_speed_ms * 3.6, 2),
        "sprintCount": sprint_count,
        "timeByThird": {k: round(v, 2) for k, v in time_thirds.items()},
        "heatmap": np.round(heatmap, 4).tolist(),
    }


# ---------------------------------------------------------------------------
# Per-team statistics
# ---------------------------------------------------------------------------

def _team_frame_positions(
    frames: List[dict], team_id: int
) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Return [(timestampMs, [(x, y), ...]), ...] for outfield players of *team_id*.

    Referees (teamId 0) and unknowns (-1) are excluded.
    """
    result: list = []
    for frame in frames:
        ts = frame["timestampMs"]
        pts = [
            (p["x"], p["y"])
            for p in frame.get("players", [])
            if p["teamId"] == team_id
        ]
        if pts:
            result.append((ts, pts))
    return result


def _team_stats(
    team_frames: List[Tuple[int, List[Tuple[float, float]]]],
) -> dict:
    """Compute centroid, amplitude, depth, compaction, hull area,
    and defensive-line height for one team across all frames.
    """
    centroids: List[dict] = []
    amplitudes: List[float] = []
    depths: List[float] = []
    compactions: List[float] = []
    hull_areas: List[float] = []
    def_line_heights: List[float] = []

    for ts, positions in team_frames:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        n = len(positions)

        # Centroid
        cx = sum(xs) / n
        cy = sum(ys) / n
        centroids.append({"timestampMs": ts, "x": round(cx, 2), "y": round(cy, 2)})

        # Amplitude (spread in Y)
        amplitudes.append(max(ys) - min(ys))

        # Depth (spread in X)
        depths.append(max(xs) - min(xs))

        # Compaction: average distance to centroid
        avg_dist = sum(_euclidean(cx, cy, px, py) for px, py in positions) / n
        compactions.append(avg_dist)

        # Convex-hull area
        if n >= 3:
            hull = _convex_hull(positions)
            hull_areas.append(_polygon_area(hull))
        else:
            hull_areas.append(0.0)

        # Defensive-line height: average X of the 4 players with the lowest X
        sorted_x = sorted(xs)
        deepest = sorted_x[: min(4, len(sorted_x))]
        def_line_heights.append(sum(deepest) / len(deepest))

    return {
        "centroid": centroids,
        "avgAmplitude": round(np.mean(amplitudes).item(), 2) if amplitudes else 0.0,
        "avgDepth": round(np.mean(depths).item(), 2) if depths else 0.0,
        "avgCompaction": round(np.mean(compactions).item(), 2) if compactions else 0.0,
        "avgConvexHullArea": round(np.mean(hull_areas).item(), 2) if hull_areas else 0.0,
        "avgDefensiveLineHeight": round(np.mean(def_line_heights).item(), 2) if def_line_heights else 0.0,
    }


# ---------------------------------------------------------------------------
# Voronoi spatial control (brute-force grid sampling)
# ---------------------------------------------------------------------------

def _voronoi_control(
    frames: List[dict],
    pitch_length: float,
    pitch_width: float,
    step: float = VORONOI_STEP,
) -> dict:
    """Compute Voronoi-based spatial control for home (1) and away (2) teams.

    For each frame a regular grid (``step`` m resolution) is laid over the
    pitch.  Each grid point is assigned to the nearest player; the fraction of
    points owned by each team gives the control percentage.

    To keep output compact, ``perFrame`` stores only every 10th frame.
    """
    # Pre-compute grid points
    gx = np.arange(step / 2, pitch_length, step)
    gy = np.arange(step / 2, pitch_width, step)
    grid_x, grid_y = np.meshgrid(gx, gy)  # shape (rows, cols)
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    n_points = len(grid_x_flat)

    home_total = 0.0
    away_total = 0.0
    counted_frames = 0
    per_frame: List[dict] = []

    for idx, frame in enumerate(frames):
        players = [
            p for p in frame.get("players", []) if p["teamId"] in (1, 2)
        ]
        if not players:
            continue

        px = np.array([p["x"] for p in players])
        py = np.array([p["y"] for p in players])
        teams = np.array([p["teamId"] for p in players])

        # Distance from every grid point to every player – shape (n_points, n_players)
        dx = grid_x_flat[:, None] - px[None, :]
        dy = grid_y_flat[:, None] - py[None, :]
        dists = dx * dx + dy * dy  # squared is fine for argmin

        nearest = np.argmin(dists, axis=1)
        owner_teams = teams[nearest]

        home_count = int(np.sum(owner_teams == 1))
        away_count = int(np.sum(owner_teams == 2))

        home_frac = home_count / n_points
        away_frac = away_count / n_points
        home_total += home_frac
        away_total += away_frac
        counted_frames += 1

        # Store every 10th frame
        if idx % 10 == 0:
            per_frame.append({
                "timestampMs": frame["timestampMs"],
                "home": round(home_frac, 4),
                "away": round(away_frac, 4),
            })

    if counted_frames > 0:
        home_avg = home_total / counted_frames
        away_avg = away_total / counted_frames
    else:
        home_avg = 0.5
        away_avg = 0.5

    return {
        "home": round(home_avg, 4),
        "away": round(away_avg, 4),
        "perFrame": per_frame,
    }


# ---------------------------------------------------------------------------
# Possession (closest-player rule)
# ---------------------------------------------------------------------------

def _possession(
    frames: List[dict],
    ball_track: List[Tuple[int, float, float]],
) -> dict:
    """Compute possession % by assigning each ball observation to the nearest
    player's team.
    """
    # Index ball positions by timestampMs for fast lookup
    ball_by_ts: Dict[int, Tuple[float, float]] = {
        ts: (bx, by) for ts, bx, by in ball_track
    }

    home_frames = 0
    away_frames = 0

    for frame in frames:
        ts = frame["timestampMs"]
        bp = ball_by_ts.get(ts)
        if bp is None:
            continue
        bx, by = bp

        best_dist = float("inf")
        best_team = -1
        for p in frame.get("players", []):
            if p["teamId"] not in (1, 2):
                continue
            d = _euclidean(bx, by, p["x"], p["y"])
            if d < best_dist:
                best_dist = d
                best_team = p["teamId"]

        if best_team == 1:
            home_frames += 1
        elif best_team == 2:
            away_frames += 1

    total = home_frames + away_frames
    if total == 0:
        return {"home": 0.5, "away": 0.5}
    return {
        "home": round(home_frames / total, 4),
        "away": round(away_frames / total, 4),
    }


# ---------------------------------------------------------------------------
# Pressure matrix
# ---------------------------------------------------------------------------

def _pressure(frames: List[dict]) -> Dict[str, dict]:
    """Average number of opponents within ``PRESSURE_RADIUS`` m per player.

    Returns {playerId_str: {"avgOpponentsWithin5m": float}}.
    """
    accum: Dict[int, List[int]] = {}  # playerId -> [count_per_frame]

    for frame in frames:
        players = frame.get("players", [])
        for p in players:
            if p["teamId"] not in (1, 2):
                continue
            pid = p["playerId"]
            tid = p["teamId"]
            count = 0
            for q in players:
                if q["teamId"] not in (1, 2) or q["teamId"] == tid:
                    continue
                if _euclidean(p["x"], p["y"], q["x"], q["y"]) <= PRESSURE_RADIUS:
                    count += 1
            accum.setdefault(pid, []).append(count)

    result: Dict[str, dict] = {}
    for pid, counts in accum.items():
        avg = sum(counts) / len(counts) if counts else 0.0
        result[str(pid)] = {"avgOpponentsWithin5m": round(avg, 2)}
    return result


# ---------------------------------------------------------------------------
# Formation detection
# ---------------------------------------------------------------------------

def _detect_formation(
    player_tracks: Dict[int, List[Tuple[int, float, float, int]]],
    team_id: int,
    gap: float = FORMATION_GAP_THRESHOLD,
) -> str:
    """Detect formation string (e.g. "4-4-2") for a team using gap-based
    clustering on average X positions.

    The goalkeeper (the player with the lowest average X) is excluded from the
    formation string because formations traditionally describe only outfield
    players.
    """
    avg_positions: List[Tuple[float, float]] = []  # (avgX, avgY)
    for pid, track in player_tracks.items():
        team_points = [(x, y) for _, x, y, tid in track if tid == team_id]
        if not team_points:
            continue
        ax = sum(p[0] for p in team_points) / len(team_points)
        ay = sum(p[1] for p in team_points) / len(team_points)
        avg_positions.append((ax, ay))

    if len(avg_positions) < 2:
        return "unknown"

    # Sort by average X (defensive → attacking)
    avg_positions.sort(key=lambda p: p[0])

    # Remove goalkeeper (lowest X)
    outfield = avg_positions[1:]
    if not outfield:
        return "unknown"

    # Gap-based clustering on X
    lines: List[int] = [1]
    for i in range(1, len(outfield)):
        if outfield[i][0] - outfield[i - 1][0] > gap:
            lines.append(1)
        else:
            lines[-1] += 1

    return "-".join(str(c) for c in lines)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def compute_all(data: dict) -> dict:
    """Compute every tactical statistic from a ``match_data`` dictionary.

    Parameters
    ----------
    data : dict
        Parsed ``match_data.json`` containing ``pitch``, ``frames``, etc.

    Returns
    -------
    dict
        Full statistics bundle ready for JSON serialisation.
    """
    frames: List[dict] = data.get("frames", [])
    pitch = data.get("pitch", {"length_m": PITCH_LENGTH, "width_m": PITCH_WIDTH})
    pl = pitch.get("length_m", PITCH_LENGTH)
    pw = pitch.get("width_m", PITCH_WIDTH)

    player_tracks, ball_track = _build_tracks(frames)

    # --- Per-player stats ---
    players_out: Dict[str, dict] = {}
    for pid, track in player_tracks.items():
        tid = track[0][3] if track else -1
        # Skip referees / unknowns
        if tid not in (1, 2):
            continue
        ps = _player_stats(track, pl, pw)
        ps["playerId"] = pid
        ps["teamId"] = tid
        players_out[str(pid)] = ps

    # --- Per-team stats ---
    teams_out: Dict[str, dict] = {}
    for tid, name in ((1, "home"), (2, "away")):
        tf = _team_frame_positions(frames, tid)
        ts = _team_stats(tf)
        ts["teamId"] = tid
        ts["name"] = name
        teams_out[str(tid)] = ts

    # --- Voronoi control ---
    voronoi = _voronoi_control(frames, pl, pw)

    # --- Possession ---
    poss = _possession(frames, ball_track)

    # --- Pressure ---
    press = _pressure(frames)

    # --- Formations ---
    formations: Dict[str, str] = {}
    for tid, name in ((1, "home"), (2, "away")):
        formations[name] = _detect_formation(player_tracks, tid)

    return {
        "pitch": pitch,
        "players": players_out,
        "teams": teams_out,
        "voronoiControl": voronoi,
        "possession": poss,
        "pressure": press,
        "formations": formations,
    }


def load_and_compute(
    data_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """Load ``match_data.json``, compute all statistics, and optionally
    write the result to *output_path* as JSON.

    Parameters
    ----------
    data_path : str
        Path to ``match_data.json``.
    output_path : str, optional
        If given, the stats dict is written here as pretty-printed JSON.

    Returns
    -------
    dict
        The full statistics bundle.
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    stats = compute_all(data)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[analytics] Stats saved to {output_path}")

    return stats
