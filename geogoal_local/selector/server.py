"""Local FastAPI server for interactive pitch calibration."""

from __future__ import annotations

import base64
import json
import os
import signal
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


# ── Pydantic models ──────────────────────────────────────────────────────────

class PlayerTag(BaseModel):
    x: float
    y: float
    label: str  # "home" | "away" | "referee"


class CalibrationPayload(BaseModel):
    src_points: List[List[float]]
    dst_points: List[List[float]]
    method: str = "cv2"
    player_tags: List[PlayerTag] = []
    landmarks_used: List[str] = []


# ── Auto-detection of pitch line intersections ───────────────────────────────

import numpy as np


def _detect_pitch_points(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect candidate calibration points from pitch line intersections.

    Strategy:
      1. Convert to HSV → mask green (pitch) → invert to get non-green (lines).
      2. Canny edge detection on the green-masked region.
      3. Hough lines to find dominant straight lines.
      4. Compute intersection points of line pairs.
      5. Filter: keep only intersections on or near the pitch (green region).
      6. Cluster nearby intersections and return centroids.

    Returns a list of {"x": px, "y": py, "score": float} sorted by score.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Broad green mask for pitch detection
    lower_green = np.array([30, 25, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Dilate to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.dilate(green_mask, kernel, iterations=2)

    # White / bright line detection within the green area
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to find bright lines on grass
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Lines are bright on green pitch — threshold within the green mask
    line_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -8
    )
    line_mask = cv2.bitwise_and(line_mask, green_mask)

    # Clean up noise
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Canny edges
    edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)

    # Hough lines (parametric)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=max(60, min(w, h) // 5))
    if lines is None or len(lines) < 2:
        return []

    # Deduplicate similar lines (merge those with similar rho/theta)
    merged: List[tuple] = []
    rho_thresh = max(15, w // 40)
    theta_thresh = np.pi / 36  # 5 degrees
    for line in lines:
        rho, theta = line[0]
        is_dup = False
        for mr, mt in merged:
            if abs(rho - mr) < rho_thresh and abs(theta - mt) < theta_thresh:
                is_dup = True
                break
        if not is_dup:
            merged.append((rho, theta))

    # Compute pairwise intersections
    intersections: List[tuple] = []
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            r1, t1 = merged[i]
            r2, t2 = merged[j]
            # Skip near-parallel lines
            if abs(np.sin(t1 - t2)) < 0.15:
                continue
            # Solve: x*cos(t)+y*sin(t)=r  for both lines
            A = np.array([[np.cos(t1), np.sin(t1)],
                          [np.cos(t2), np.sin(t2)]])
            b = np.array([r1, r2])
            try:
                pt = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            ix, iy = pt[0], pt[1]
            # Must be within frame bounds (with small margin)
            margin = 10
            if -margin <= ix < w + margin and -margin <= iy < h + margin:
                intersections.append((float(ix), float(iy)))

    if not intersections:
        return []

    # Cluster nearby intersections (simple greedy merge within radius)
    cluster_radius = max(15, w // 40)
    clusters: List[List[tuple]] = []
    used = set()
    for i, p in enumerate(intersections):
        if i in used:
            continue
        cluster = [p]
        used.add(i)
        for j, q in enumerate(intersections):
            if j in used:
                continue
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < cluster_radius ** 2:
                cluster.append(q)
                used.add(j)
        clusters.append(cluster)

    # Compute centroids + score (more lines intersecting = higher score)
    candidates = []
    for cluster in clusters:
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        # Clamp to frame
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        # Score: number of lines intersecting here, and whether it's on green
        score = len(cluster)
        gx, gy = int(cx), int(cy)
        if 0 <= gx < w and 0 <= gy < h and green_mask[gy, gx] > 0:
            score += 2  # bonus for being on the pitch
        candidates.append({"x": round(cx, 1), "y": round(cy, 1), "score": score})

    # Sort by score descending, return top 20
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:20]


# ── Landmark definitions (shared with front-end JS) ─────────────────────────

PITCH_LANDMARKS = {
    "Top-left corner": (0, 0),
    "Top-right corner": (105, 0),
    "Bottom-right corner": (105, 68),
    "Bottom-left corner": (0, 68),
    "Center spot": (52.5, 34),
    "Halfway top": (52.5, 0),
    "Halfway bottom": (52.5, 68),
    "Left PA top": (0, 13.84),
    "Left PA bottom": (0, 54.16),
    "Left PA top-right": (16.5, 13.84),
    "Left PA bottom-right": (16.5, 54.16),
    "Left penalty spot": (11, 34),
    "Right PA top": (105, 13.84),
    "Right PA bottom": (105, 54.16),
    "Right PA top-left": (88.5, 13.84),
    "Right PA bottom-left": (88.5, 54.16),
    "Right penalty spot": (94, 34),
    "Left goal area top-right": (5.5, 24.84),
    "Left goal area bottom-right": (5.5, 43.16),
    "Right goal area top-left": (99.5, 24.84),
    "Right goal area bottom-left": (99.5, 43.16),
    "Center circle top": (52.5, 24.85),
    "Center circle bottom": (52.5, 43.15),
}


def _auto_assign_landmarks(
    candidates: List[Dict[str, Any]], frame_w: int, frame_h: int
) -> List[Dict[str, Any]]:
    """Assign landmark identities to detected candidate points using spatial heuristics.

    Strategy:
      1. Compute convex hull of candidates to approximate the pitch boundary.
      2. Find bounding box of the hull.
      3. Map the 4 hull corners closest to the bounding-box corners as field corners.
      4. Find candidates near the vertical midline for halfway-line points.
      5. Find candidates matching penalty-area proportions.
      6. Return only high-confidence assignments (>0.6).
    """
    if len(candidates) < 4:
        return []

    pts = np.array([[c["x"], c["y"]] for c in candidates], dtype=np.float32)

    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2)

    x_min, y_min = hull_pts.min(axis=0)
    x_max, y_max = hull_pts.max(axis=0)
    bw = x_max - x_min
    bh = y_max - y_min

    if bw < frame_w * 0.1 or bh < frame_h * 0.1:
        return []

    assignments: List[Dict[str, Any]] = []
    used_indices: set = set()

    def find_closest(target_x: float, target_y: float, exclude: set = set()):
        best_idx = -1
        best_dist = float("inf")
        for i, (px, py) in enumerate(pts):
            if i in exclude:
                continue
            d = ((px - target_x) ** 2 + (py - target_y) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

    max_d = ((bw ** 2 + bh ** 2) ** 0.5) * 0.15

    # --- 4 field corners ---
    corners = [
        ("Top-left corner", x_min, y_min),
        ("Top-right corner", x_max, y_min),
        ("Bottom-right corner", x_max, y_max),
        ("Bottom-left corner", x_min, y_max),
    ]

    for landmark, tx, ty in corners:
        idx, dist = find_closest(tx, ty, used_indices)
        if idx >= 0 and dist < max_d:
            conf = max(0.0, 1.0 - dist / max_d)
            if conf > 0.6:
                lm = PITCH_LANDMARKS[landmark]
                assignments.append({
                    "x": float(pts[idx][0]),
                    "y": float(pts[idx][1]),
                    "landmark": landmark,
                    "dstX": lm[0],
                    "dstY": lm[1],
                    "confidence": round(conf, 2),
                })
                used_indices.add(idx)

    # --- Halfway-line points ---
    mid_x = (x_min + x_max) / 2
    mid_x_tolerance = bw * 0.12

    halfway_targets = [
        ("Halfway top", mid_x, y_min),
        ("Halfway bottom", mid_x, y_max),
    ]
    for landmark, tx, ty in halfway_targets:
        idx, dist = find_closest(tx, ty, used_indices)
        if idx >= 0:
            px, py = pts[idx]
            if abs(px - mid_x) < mid_x_tolerance and dist < max_d:
                conf = max(0.0, 1.0 - dist / max_d) * 0.9
                if conf > 0.6:
                    lm = PITCH_LANDMARKS[landmark]
                    assignments.append({
                        "x": float(px), "y": float(py),
                        "landmark": landmark,
                        "dstX": lm[0], "dstY": lm[1],
                        "confidence": round(conf, 2),
                    })
                    used_indices.add(idx)

    # --- Penalty-area corners ---
    pa_x_left = x_min + bw * (16.5 / 105)
    pa_x_right = x_max - bw * (16.5 / 105)
    pa_y_top = y_min + bh * (13.84 / 68)
    pa_y_bottom = y_max - bh * (13.84 / 68)
    pa_tolerance = max_d * 0.8

    pa_targets = [
        ("Left PA top-right", pa_x_left, pa_y_top),
        ("Left PA bottom-right", pa_x_left, pa_y_bottom),
        ("Right PA top-left", pa_x_right, pa_y_top),
        ("Right PA bottom-left", pa_x_right, pa_y_bottom),
    ]
    for landmark, tx, ty in pa_targets:
        idx, dist = find_closest(tx, ty, used_indices)
        if idx >= 0 and dist < pa_tolerance:
            conf = max(0.0, 1.0 - dist / pa_tolerance) * 0.85
            if conf > 0.6:
                lm = PITCH_LANDMARKS[landmark]
                assignments.append({
                    "x": float(pts[idx][0]), "y": float(pts[idx][1]),
                    "landmark": landmark,
                    "dstX": lm[0], "dstY": lm[1],
                    "confidence": round(conf, 2),
                })
                used_indices.add(idx)

    assignments.sort(key=lambda a: a["confidence"], reverse=True)
    return assignments[:8]


# ── HTML page (inline) ──────────────────────────────────────────────────────

SELECTOR_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geo-Goal — Pitch Calibration</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#121217;--panel:#1a1a24;--border:#2a2a3a;--accent:#4f8cff;
--accent2:#38d9a9;--text:#e0e0e8;--muted:#888;--danger:#ff4f6d}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--text);height:100vh;overflow:hidden;display:flex;flex-direction:column}
header{background:var(--panel);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:14px}
header h1{font-size:16px;font-weight:600;color:var(--accent)}
header span{font-size:12px;color:var(--muted)}
.main{display:flex;flex:1;overflow:hidden}
.canvas-wrap{flex:1;position:relative;overflow:hidden;background:#0a0a10;display:flex;align-items:center;justify-content:center}
canvas{cursor:crosshair;max-width:100%;max-height:100%}
.sidebar{width:360px;min-width:320px;background:var(--panel);border-left:1px solid var(--border);
display:flex;flex-direction:column;overflow-y:auto}
.section{padding:14px 16px;border-bottom:1px solid var(--border)}
.section h2{font-size:13px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:10px}
select,input,button{font-family:inherit;font-size:13px;background:var(--bg);color:var(--text);
border:1px solid var(--border);border-radius:6px;padding:7px 10px;outline:none;width:100%}
select:focus,input:focus{border-color:var(--accent)}
button{cursor:pointer;text-align:center;font-weight:600;transition:background .15s}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn-primary:hover{background:#3a78e8}
.btn-primary:disabled{opacity:.4;cursor:not-allowed}
.btn-danger{background:var(--danger);border-color:var(--danger);color:#fff}
.btn-danger:hover{background:#e03e58}
.btn-sm{padding:4px 10px;font-size:12px;width:auto}
.point-list{list-style:none;max-height:220px;overflow-y:auto}
.point-list li{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px}
.point-list li .num{width:22px;height:22px;border-radius:50%;background:var(--accent);color:#fff;
display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0}
.point-list li .info{flex:1;line-height:1.4}
.point-list li .coords{color:var(--muted)}
.tag-list{list-style:none;max-height:160px;overflow-y:auto}
.tag-list li{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border);font-size:12px}
.tag-list li .dot{width:14px;height:14px;border-radius:50%;flex-shrink:0}
.dot-home{background:#4f8cff}.dot-away{background:#ff6b6b}.dot-referee{background:#ffd43b}
.custom-fields{display:flex;gap:6px;margin-top:6px}
.custom-fields input{width:50%}
.toggle{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:13px}
.toggle input{width:auto}
.status{padding:10px 16px;font-size:12px;color:var(--accent2);text-align:center}
#teamPopup{position:fixed;display:none;background:var(--panel);border:1px solid var(--border);
border-radius:8px;padding:8px;z-index:100;box-shadow:0 8px 24px rgba(0,0,0,.5)}
#teamPopup button{margin:3px 0;padding:6px 14px}
.btn-success{background:#2ea043;border-color:#2ea043;color:#fff;font-weight:600}
.btn-success:hover{background:#28903c}
.btn-secondary{background:var(--bg);color:var(--muted);font-size:11px;padding:5px 8px}
.proposal-item{background:rgba(240,136,62,0.15)}
#pitchDiagram{display:block;width:100%;border:1px solid var(--border);border-radius:6px;background:#1a3a1a}
.proposal-actions{display:flex;gap:6px}
</style>
</head>
<body>

<header>
  <h1>⚽ Geo-Goal Calibration</h1>
  <span>Click on the frame to place landmark points, then save calibration</span>
</header>

<div class="main">
  <div class="canvas-wrap">
    <canvas id="cvs"></canvas>
  </div>

  <div class="sidebar">
    <!-- Landmark selector -->
    <div class="section">
      <h2>Landmark para el siguiente punto</h2>
      <p style="font-size:11px;color:#8b949e;margin-bottom:6px;">1. Elige un landmark → 2. Haz clic en ese punto en la imagen</p>
      <select id="landmarkSel"></select>
      <div id="customFields" class="custom-fields" style="display:none">
        <input id="customX" type="number" step="0.01" placeholder="X (m)">
        <input id="customY" type="number" step="0.01" placeholder="Y (m)">
      </div>
      <div style="margin-top:10px">
        <button class="btn-primary" id="autoDetectBtn" style="background:#38d9a9;border-color:#38d9a9;">🔍 Auto-detectar puntos</button>
        <label class="toggle" style="margin-top:6px">
          <input type="checkbox" id="showCandidates" checked> Mostrar puntos detectados
        </label>
      </div>
    </div>

    <!-- Automatic detection -->
    <div class="section">
      <h2>Detección automática</h2>
      <button class="btn-success" id="preDetectBtn" style="font-size:14px;padding:10px;">Pre-detectar campo</button>
      <div id="proposalInfo" style="display:none;margin-top:10px;">
        <p id="proposalSummary" style="font-size:12px;color:#f0883e;margin-bottom:8px;"></p>
        <div class="proposal-actions">
          <button class="btn-sm" id="acceptProposals" style="background:#2ea043;border-color:#2ea043;color:#fff;flex:1;">✓ Usar estos puntos</button>
          <button class="btn-sm btn-danger" id="discardProposals" style="flex:1;">✗ Descartar</button>
        </div>
      </div>
      <div style="margin-top:10px;">
        <button class="btn-sm btn-secondary" id="showCandidatesBtn">Mostrar candidatos</button>
      </div>
    </div>

    <!-- Pitch mini-diagram -->
    <div class="section">
      <h2>Mapa del campo</h2>
      <canvas id="pitchDiagram" width="200" height="130"></canvas>
    </div>

    <!-- Proposal list -->
    <div class="section" id="proposalSection" style="display:none;">
      <h2>Puntos propuestos (<span id="propCount">0</span>)</h2>
      <ul class="point-list" id="proposalList"></ul>
    </div>

    <!-- Point list -->
    <div class="section" style="flex:1">
      <h2>Placed Points (<span id="ptCount">0</span>)</h2>
      <ul class="point-list" id="pointList"></ul>
    </div>

    <!-- Player tagging -->
    <div class="section">
      <h2>Player Tagging</h2>
      <label class="toggle">
        <input type="checkbox" id="tagToggle"> Enable player tagging mode
      </label>
      <ul class="tag-list" id="tagList"></ul>
    </div>

    <!-- Actions -->
    <div class="section">
      <button class="btn-primary" id="saveBtn" disabled>Save Calibration (≥4 pts)</button>
      <div style="height:6px"></div>
      <button class="btn-danger btn-sm" id="clearBtn" style="width:100%">Clear All Points</button>
    </div>

    <div class="status" id="statusMsg"></div>
  </div>
</div>

<!-- Team popup for player tagging -->
<div id="teamPopup">
  <button class="btn-sm" data-team="home" style="background:#4f8cff;border-color:#4f8cff;color:#fff">Home</button>
  <button class="btn-sm" data-team="away" style="background:#ff6b6b;border-color:#ff6b6b;color:#fff">Away</button>
  <button class="btn-sm" data-team="referee" style="background:#ffd43b;border-color:#ffd43b;color:#111">Referee</button>
</div>

<script>
const LANDMARKS = {
  "Top-left corner": [0, 0],
  "Top-right corner": [105, 0],
  "Bottom-right corner": [105, 68],
  "Bottom-left corner": [0, 68],
  "Center spot": [52.5, 34],
  "Halfway top": [52.5, 0],
  "Halfway bottom": [52.5, 68],
  "Left PA top": [0, 13.84],
  "Left PA bottom": [0, 54.16],
  "Left PA top-right": [16.5, 13.84],
  "Left PA bottom-right": [16.5, 54.16],
  "Left penalty spot": [11, 34],
  "Right PA top": [105, 13.84],
  "Right PA bottom": [105, 54.16],
  "Right PA top-left": [88.5, 13.84],
  "Right PA bottom-left": [88.5, 54.16],
  "Right penalty spot": [94, 34],
  "Left goal area top-right": [5.5, 24.84],
  "Left goal area bottom-right": [5.5, 43.16],
  "Right goal area top-left": [99.5, 24.84],
  "Right goal area bottom-left": [99.5, 43.16],
  "Center circle top": [52.5, 24.85],
  "Center circle bottom": [52.5, 43.15],
  "Custom (enter manually)": null
};

// State
let frameImg = null;
let imgW = 0, imgH = 0;
let points = [];       // {px, py, landmark, dstX, dstY}
let playerTags = [];    // {x, y, label}
let taggingMode = false;
let pendingTagClick = null;
let detectedCandidates = [];  // auto-detected points
let showCandidatesFlag = true;
let proposedPoints = [];  // {x, y, landmark, dstX, dstY, confidence}

const cvs = document.getElementById("cvs");
const ctx = cvs.getContext("2d");
const sel = document.getElementById("landmarkSel");
const customFields = document.getElementById("customFields");
const pointList = document.getElementById("pointList");
const ptCount = document.getElementById("ptCount");
const tagList = document.getElementById("tagList");
const saveBtn = document.getElementById("saveBtn");
const clearBtn = document.getElementById("clearBtn");
const tagToggle = document.getElementById("tagToggle");
const statusMsg = document.getElementById("statusMsg");
const teamPopup = document.getElementById("teamPopup");

// Populate landmark dropdown
Object.keys(LANDMARKS).forEach(name => {
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = LANDMARKS[name]
    ? `${name}  (${LANDMARKS[name][0]}, ${LANDMARKS[name][1]})`
    : name;
  sel.appendChild(opt);
});

sel.addEventListener("change", () => {
  customFields.style.display = sel.value === "Custom (enter manually)" ? "flex" : "none";
  // Show destination coords preview
  const lm = LANDMARKS[sel.value];
  if (lm) {
    statusMsg.textContent = "Siguiente: " + sel.value + " → (" + lm[0] + ", " + lm[1] + ") m";
    statusMsg.style.color = "#58a6ff";
  }
});

tagToggle.addEventListener("change", () => { taggingMode = tagToggle.checked; });

// Load frame
async function loadFrame() {
  const res = await fetch("/frame");
  const data = await res.json();
  imgW = data.width;
  imgH = data.height;
  frameImg = new Image();
  frameImg.onload = () => {
    fitCanvas();
    draw();
    fetchCandidates();
  };
  frameImg.src = "data:image/jpeg;base64," + data.frame;
}

async function fetchCandidates() {
  try {
    const res = await fetch("/detect");
    const data = await res.json();
    detectedCandidates = data.candidates || [];
    draw();
  } catch(e) {}
}

function fitCanvas() {
  const wrap = cvs.parentElement;
  const wW = wrap.clientWidth, wH = wrap.clientHeight;
  const scale = Math.min(wW / imgW, wH / imgH, 1);
  cvs.width = Math.round(imgW * scale);
  cvs.height = Math.round(imgH * scale);
}

function canvasToImg(cx, cy) {
  return [cx / cvs.width * imgW, cy / cvs.height * imgH];
}

function imgToCanvas(ix, iy) {
  return [ix / imgW * cvs.width, iy / imgH * cvs.height];
}

function draw() {
  if (!frameImg) return;
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.drawImage(frameImg, 0, 0, cvs.width, cvs.height);

  // Draw lines connecting points (convex hull outline)
  if (points.length >= 2) {
    ctx.beginPath();
    ctx.strokeStyle = "rgba(79,140,255,0.5)";
    ctx.lineWidth = 2;
    for (let i = 0; i < points.length; i++) {
      const [cx, cy] = imgToCanvas(points[i].px, points[i].py);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    if (points.length >= 3) ctx.closePath();
    ctx.stroke();
  }

  // Draw points
  points.forEach((p, i) => {
    const [cx, cy] = imgToCanvas(p.px, p.py);
    ctx.beginPath();
    ctx.arc(cx, cy, 12, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(79,140,255,0.85)";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), cx, cy);
  });

  // Draw player tags
  playerTags.forEach(t => {
    const [cx, cy] = imgToCanvas(t.x, t.y);
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    ctx.fillStyle = t.label === "home" ? "#4f8cff" : t.label === "away" ? "#ff6b6b" : "#ffd43b";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // Draw auto-detected candidates
  if (showCandidatesFlag && detectedCandidates.length > 0) {
    detectedCandidates.forEach((c, i) => {
      const [cx, cy] = imgToCanvas(c.x, c.y);
      const r = 6 + Math.min(c.score, 6);
      // Pulsing diamond shape
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(Math.PI / 4);
      ctx.beginPath();
      ctx.rect(-r/2, -r/2, r, r);
      ctx.fillStyle = "rgba(56, 217, 169, 0.4)";
      ctx.fill();
      ctx.strokeStyle = "#38d9a9";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
      // Score label
      if (i < 8) {
        ctx.fillStyle = "#38d9a9";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(String(i+1), cx + r/2 + 3, cy - 2);
      }
    });
  }

  // Draw proposed points (orange circles with dashed outline)
  if (proposedPoints.length > 0) {
    proposedPoints.forEach((pp, i) => {
      const [cx, cy] = imgToCanvas(pp.x, pp.y);
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(240,136,62,0.3)";
      ctx.fill();
      ctx.strokeStyle = "#f0883e";
      ctx.lineWidth = 2.5;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#f0883e";
      ctx.fill();
      ctx.fillStyle = "#f0883e";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(pp.landmark, cx + 16, cy - 2);
      ctx.restore();
    });
  }
}

// Canvas click — snap to nearest candidate if close enough
cvs.addEventListener("click", (e) => {
  const rect = cvs.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  let [ix, iy] = canvasToImg(cx, cy);

  if (taggingMode) {
    pendingTagClick = {x: ix, y: iy};
    teamPopup.style.display = "block";
    teamPopup.style.left = e.clientX + "px";
    teamPopup.style.top = e.clientY + "px";
    return;
  }

  // Snap to nearest auto-detected candidate (within 15px in image space)
  if (detectedCandidates.length > 0) {
    let bestDist = 15 * 15;
    let bestC = null;
    for (const c of detectedCandidates) {
      const d = (ix - c.x) ** 2 + (iy - c.y) ** 2;
      if (d < bestDist) { bestDist = d; bestC = c; }
    }
    if (bestC) {
      ix = bestC.x;
      iy = bestC.y;
      statusMsg.textContent = "Snap to detected point (" + ix.toFixed(0) + ", " + iy.toFixed(0) + ")";
      statusMsg.style.color = "#38d9a9";
    }
  }

  const landmark = sel.value;

  // Prevent reusing the same landmark
  if (landmark !== "Custom (enter manually)" && points.some(p => p.landmark === landmark)) {
    alert("This landmark is already used! Select a different one from the dropdown.");
    return;
  }

  let dstX, dstY;
  if (landmark === "Custom (enter manually)") {
    dstX = parseFloat(document.getElementById("customX").value);
    dstY = parseFloat(document.getElementById("customY").value);
    if (isNaN(dstX) || isNaN(dstY)) { alert("Enter valid X and Y for custom landmark"); return; }
  } else {
    [dstX, dstY] = LANDMARKS[landmark];
  }

  points.push({px: Math.round(ix * 100) / 100, py: Math.round(iy * 100) / 100,
               landmark, dstX, dstY});

  // Auto-advance dropdown to next UNUSED landmark
  const usedLandmarks = new Set(points.map(p => p.landmark));
  const allNames = Object.keys(LANDMARKS);
  const nextUnused = allNames.find(n => !usedLandmarks.has(n) && LANDMARKS[n] !== null);
  if (nextUnused) sel.value = nextUnused;
  customFields.style.display = sel.value === "Custom (enter manually)" ? "flex" : "none";

  updateUI();
});

// Team popup buttons
teamPopup.querySelectorAll("button").forEach(btn => {
  btn.addEventListener("click", () => {
    if (pendingTagClick) {
      playerTags.push({x: pendingTagClick.x, y: pendingTagClick.y, label: btn.dataset.team});
      pendingTagClick = null;
      teamPopup.style.display = "none";
      updateUI();
    }
  });
});

document.addEventListener("click", (e) => {
  if (!teamPopup.contains(e.target) && e.target !== cvs) {
    teamPopup.style.display = "none";
    pendingTagClick = null;
  }
});

function updateUI() {
  // Point list
  ptCount.textContent = points.length;
  pointList.innerHTML = "";
  points.forEach((p, i) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="num">${i + 1}</span>
      <span class="info">
        <div>${p.landmark}</div>
        <div class="coords">px(${p.px.toFixed(1)}, ${p.py.toFixed(1)}) → m(${p.dstX}, ${p.dstY})</div>
      </span>`;
    const del = document.createElement("button");
    del.className = "btn-danger btn-sm";
    del.textContent = "✕";
    del.onclick = () => { points.splice(i, 1); updateUI(); };
    li.appendChild(del);
    pointList.appendChild(li);
  });

  // Tag list
  tagList.innerHTML = "";
  playerTags.forEach((t, i) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="dot dot-${t.label}"></span>
      <span style="flex:1">${t.label} (${Math.round(t.x)}, ${Math.round(t.y)})</span>`;
    const del = document.createElement("button");
    del.className = "btn-danger btn-sm";
    del.textContent = "✕";
    del.onclick = () => { playerTags.splice(i, 1); updateUI(); };
    li.appendChild(del);
    tagList.appendChild(li);
  });

  saveBtn.disabled = points.length < 4;
  saveBtn.textContent = points.length < 4
    ? `Save Calibration (need ${4 - points.length} more)`
    : `Save Calibration (${points.length} pts)`;
  draw();
  drawPitchDiagram();
}

// Save
saveBtn.addEventListener("click", async () => {
  // Validate distinct dst points
  const dstSet = new Set(points.map(p => p.dstX + "," + p.dstY));
  if (dstSet.size < points.length) {
    alert("Error: Some points map to the same pitch coordinates!\\nEach point must use a DIFFERENT landmark.");
    return;
  }

  const payload = {
    src_points: points.map(p => [p.px, p.py]),
    dst_points: points.map(p => [p.dstX, p.dstY]),
    method: "cv2",
    player_tags: playerTags.map(t => ({x: t.x, y: t.y, label: t.label})),
    landmarks_used: points.map(p => p.landmark)
  };
  try {
    const res = await fetch("/save", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const data = await res.json();
    statusMsg.textContent = "✓ " + data.message;
    statusMsg.style.color = "#38d9a9";
    saveBtn.disabled = true;
    saveBtn.textContent = "Saved!";
    // Auto-shutdown after short delay
    setTimeout(async () => {
      try { await fetch("/shutdown", {method: "POST"}); } catch(_) {}
      document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#38d9a9;font-size:20px;font-family:sans-serif">✓ Calibration saved — you can close this tab</div>';
    }, 1500);
  } catch (err) {
    statusMsg.textContent = "✗ Save failed: " + err.message;
    statusMsg.style.color = "#ff4f6d";
  }
});

clearBtn.addEventListener("click", () => {
  if (points.length === 0 && playerTags.length === 0) return;
  if (!confirm("Clear all points and tags?")) return;
  points = [];
  playerTags = [];
  updateUI();
});

window.addEventListener("resize", () => { fitCanvas(); draw(); });

// Auto-detect button (show candidates)
const autoDetectBtn = document.getElementById("autoDetectBtn");
const showCandidatesChk = document.getElementById("showCandidates");

autoDetectBtn.addEventListener("click", async () => {
  autoDetectBtn.disabled = true;
  autoDetectBtn.textContent = "Detectando...";
  try {
    const res = await fetch("/detect");
    const data = await res.json();
    detectedCandidates = data.candidates || [];
    autoDetectBtn.textContent = detectedCandidates.length > 0
      ? "Se encontraron " + detectedCandidates.length + " puntos"
      : "No se detectaron puntos";
    statusMsg.textContent = detectedCandidates.length + " intersecciones de lineas detectadas. Haz clic cerca de uno para usarlo.";
    statusMsg.style.color = "#38d9a9";
    draw();
  } catch (err) {
    autoDetectBtn.textContent = "Error: " + err.message;
  }
  setTimeout(() => {
    autoDetectBtn.disabled = false;
    autoDetectBtn.textContent = "Auto-detectar puntos";
  }, 3000);
});

showCandidatesChk.addEventListener("change", () => {
  showCandidatesFlag = showCandidatesChk.checked;
  draw();
});

// Show candidates button (secondary)
document.getElementById("showCandidatesBtn").addEventListener("click", async () => {
  const btn = document.getElementById("showCandidatesBtn");
  btn.disabled = true;
  btn.textContent = "Detectando...";
  try {
    const res = await fetch("/detect");
    const data = await res.json();
    detectedCandidates = data.candidates || [];
    btn.textContent = detectedCandidates.length > 0
      ? detectedCandidates.length + " candidatos"
      : "Sin candidatos";
    showCandidatesFlag = true;
    draw();
  } catch (err) {
    btn.textContent = "Error";
  }
  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = "Mostrar candidatos";
  }, 3000);
});

// Pre-detect button
document.getElementById("preDetectBtn").addEventListener("click", async () => {
  const btn = document.getElementById("preDetectBtn");
  btn.disabled = true;
  btn.textContent = "Analizando campo...";
  try {
    const res = await fetch("/auto_calibrate");
    const data = await res.json();
    proposedPoints = data.assignments || [];
    if (proposedPoints.length > 0) {
      document.getElementById("proposalInfo").style.display = "block";
      document.getElementById("proposalSummary").textContent =
        "Se detectaron " + proposedPoints.length + " puntos de referencia";
      document.getElementById("proposalSection").style.display = "block";
      updateProposalList();
      drawPitchDiagram();
    } else {
      statusMsg.textContent = "No se pudieron detectar puntos automaticamente. Usa 'Mostrar candidatos'.";
      statusMsg.style.color = "#ff4f6d";
    }
    btn.textContent = "Pre-detectar campo";
    btn.disabled = false;
    draw();
  } catch(err) {
    btn.textContent = "Error: " + err.message;
    btn.disabled = false;
  }
});

// Accept/Discard proposals
document.getElementById("acceptProposals").addEventListener("click", () => {
  proposedPoints.forEach(pp => {
    if (!points.some(p => p.landmark === pp.landmark)) {
      points.push({px: pp.x, py: pp.y, landmark: pp.landmark, dstX: pp.dstX, dstY: pp.dstY});
    }
  });
  proposedPoints = [];
  document.getElementById("proposalInfo").style.display = "none";
  document.getElementById("proposalSection").style.display = "none";
  updateUI();
  statusMsg.textContent = "Puntos aceptados y agregados";
  statusMsg.style.color = "#38d9a9";
});

document.getElementById("discardProposals").addEventListener("click", () => {
  proposedPoints = [];
  document.getElementById("proposalInfo").style.display = "none";
  document.getElementById("proposalSection").style.display = "none";
  draw();
  drawPitchDiagram();
  statusMsg.textContent = "Propuestas descartadas";
  statusMsg.style.color = "#888";
});

function updateProposalList() {
  const list = document.getElementById("proposalList");
  const count = document.getElementById("propCount");
  count.textContent = proposedPoints.length;
  list.innerHTML = "";
  proposedPoints.forEach((pp, i) => {
    const li = document.createElement("li");
    li.style.background = "rgba(240,136,62,0.15)";
    li.innerHTML =
      '<span class="num" style="background:#f0883e">' + (i+1) + '</span>' +
      '<span class="info">' +
        '<div style="color:#f0883e">' + pp.landmark + '</div>' +
        '<div class="coords">px(' + pp.x.toFixed(1) + ', ' + pp.y.toFixed(1) + ') -> m(' + pp.dstX + ', ' + pp.dstY + ')</div>' +
        '<div style="font-size:10px;color:#888">conf: ' + (pp.confidence*100).toFixed(0) + '%</div>' +
      '</span>';
    const del = document.createElement("button");
    del.className = "btn-danger btn-sm";
    del.textContent = "x";
    del.onclick = () => { proposedPoints.splice(i, 1); updateProposalList(); draw(); drawPitchDiagram(); };
    li.appendChild(del);
    list.appendChild(li);
  });
}

function drawPitchDiagram() {
  const pc = document.getElementById("pitchDiagram");
  if (!pc) return;
  const pctx = pc.getContext("2d");
  const pw = pc.width, ph = pc.height;
  const pad = 10;
  const fw = pw - 2*pad, fh = ph - 2*pad;

  pctx.clearRect(0, 0, pw, ph);
  pctx.fillStyle = "#1a3a1a";
  pctx.fillRect(0, 0, pw, ph);

  pctx.strokeStyle = "rgba(255,255,255,0.4)";
  pctx.lineWidth = 1;
  pctx.strokeRect(pad, pad, fw, fh);

  // Halfway line
  pctx.beginPath();
  pctx.moveTo(pad + fw/2, pad);
  pctx.lineTo(pad + fw/2, pad + fh);
  pctx.stroke();

  // Center circle
  pctx.beginPath();
  pctx.arc(pad + fw/2, pad + fh/2, fh * 0.135, 0, Math.PI * 2);
  pctx.stroke();

  // Center spot
  pctx.beginPath();
  pctx.arc(pad + fw/2, pad + fh/2, 2, 0, Math.PI * 2);
  pctx.fillStyle = "rgba(255,255,255,0.4)";
  pctx.fill();

  // Penalty areas
  var paW = fw * 16.5/105;
  var paH = fh * 40.32/68;
  var paY = pad + (fh - paH)/2;
  pctx.strokeRect(pad, paY, paW, paH);
  pctx.strokeRect(pad + fw - paW, paY, paW, paH);

  // Goal areas
  var gaW = fw * 5.5/105;
  var gaH = fh * 18.32/68;
  var gaY = pad + (fh - gaH)/2;
  pctx.strokeRect(pad, gaY, gaW, gaH);
  pctx.strokeRect(pad + fw - gaW, gaY, gaW, gaH);

  // Accepted points (blue)
  points.forEach(function(p) {
    var dx = pad + (p.dstX / 105) * fw;
    var dy = pad + (p.dstY / 68) * fh;
    pctx.beginPath();
    pctx.arc(dx, dy, 4, 0, Math.PI * 2);
    pctx.fillStyle = "#4f8cff";
    pctx.fill();
    pctx.strokeStyle = "#fff";
    pctx.lineWidth = 1;
    pctx.stroke();
  });

  // Proposed points (orange)
  proposedPoints.forEach(function(pp) {
    var dx = pad + (pp.dstX / 105) * fw;
    var dy = pad + (pp.dstY / 68) * fh;
    pctx.beginPath();
    pctx.arc(dx, dy, 4, 0, Math.PI * 2);
    pctx.fillStyle = "#f0883e";
    pctx.fill();
    pctx.strokeStyle = "#fff";
    pctx.lineWidth = 1;
    pctx.stroke();
  });
}

loadFrame();
</script>
</body>
</html>"""


# ── Server ───────────────────────────────────────────────────────────────────

def run_selector(
    video_path: str,
    output_dir: str = "./output",
    port: int = 8989,
) -> None:
    """Start the calibration selector server and open the browser."""
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Extract frame 0
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame from: {video_path}")

    h, w = frame.shape[:2]
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_b64 = base64.b64encode(buf).decode("ascii")

    # Build app
    app = FastAPI(title="Geo-Goal Selector")
    server: Optional[uvicorn.Server] = None

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return SELECTOR_HTML

    @app.get("/frame")
    async def get_frame():
        return JSONResponse({"frame": frame_b64, "width": w, "height": h})

    @app.get("/detect")
    async def detect_points():
        """Auto-detect pitch line intersections as candidate calibration points."""
        candidates = _detect_pitch_points(frame)
        return JSONResponse({"ok": True, "candidates": candidates, "count": len(candidates)})

    @app.get("/auto_calibrate")
    async def auto_calibrate():
        """Auto-detect and assign landmark identities to candidate points."""
        candidates = _detect_pitch_points(frame)
        assignments = _auto_assign_landmarks(candidates, w, h)
        return JSONResponse({"ok": True, "assignments": assignments, "count": len(assignments)})

    @app.post("/save")
    async def save_calibration(payload: CalibrationPayload):
        # Validate: dst_points must not all be identical
        dst = payload.dst_points
        if len(dst) >= 2 and all(d == dst[0] for d in dst):
            return JSONResponse(
                {"ok": False, "message": "Error: All destination points are the same! "
                 "Select a DIFFERENT landmark for each point."},
                status_code=400,
            )

        calib_path = os.path.join(output_dir, "calib.json")
        data = payload.model_dump()
        with open(calib_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[selector] Calibration saved → {calib_path}")
        return {"ok": True, "message": f"Saved to {calib_path}"}

    @app.post("/shutdown")
    async def shutdown():
        print("[selector] Shutdown requested — stopping server")
        if server is not None:
            server.should_exit = True
        return {"ok": True}

    # Run
    print(f"[selector] Starting calibration server on http://localhost:{port}")
    print(f"[selector] Video: {video_path}")
    print(f"[selector] Output: {output_dir}")
    print(f"[selector] Open http://localhost:{port} in your browser\n")

    # Only auto-open browser when run standalone (not from GUI which opens its own tab)
    import inspect
    caller = inspect.stack()
    launched_from_gui = any("gui" in frame.filename for frame in caller if frame.filename)
    if not launched_from_gui:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
    print("[selector] Server stopped")
