"""
report.py — Generate a self-contained HTML tactical-analysis report.

The output is a single ``report.html`` file that works offline (no CDN,
no external resources).  All visualisations use vanilla HTML + CSS +
JavaScript with ``<canvas>`` for 2-D graphics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    data_path: str = "output/match_data.json",
    stats_path: str = "output/stats.json",
    output_dir: str = "output",
    **_kwargs,
) -> str:
    """CLI-friendly wrapper used by *geogoal report*."""
    output_path = os.path.join(output_dir, "report.html")
    render_path = os.path.join(output_dir, "render.mp4")
    return generate_report(
        data_path=data_path,
        stats_path=stats_path,
        output_path=output_path,
        render_path=render_path,
    )


def generate_report(
    data_path: str = "output/match_data.json",
    stats_path: str = "output/stats.json",
    output_path: str = "output/report.html",
    video_path: str | None = None,
    render_path: str = "output/render.mp4",
) -> str:
    """Generate self-contained HTML report.  Returns *output_path*."""

    # -- Load data ----------------------------------------------------------
    with open(data_path, "r", encoding="utf-8") as fh:
        match_data: dict = json.load(fh)

    with open(stats_path, "r", encoding="utf-8") as fh:
        stats: dict = json.load(fh)

    match_json = json.dumps(match_data, separators=(",", ":"))
    stats_json = json.dumps(stats, separators=(",", ":"))

    # -- Resolve render video path relative to output file ------------------
    render_rel = ""
    if render_path and os.path.isfile(render_path):
        render_rel = os.path.relpath(render_path, os.path.dirname(output_path))

    # -- Build HTML ---------------------------------------------------------
    html = _build_html(match_json, stats_json, render_rel)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"[report] ✓ Written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _build_html(match_json: str, stats_json: str, render_rel: str) -> str:
    """Return the complete HTML string."""

    # We use a raw-string for the HTML template so curly braces used by JS
    # don't collide with Python format strings.  The three dynamic parts are
    # injected via simple string replacement markers.
    MATCH_DATA_MARKER = "\"__MATCH_DATA__\""
    STATS_DATA_MARKER = "\"__STATS_DATA__\""
    RENDER_PATH_MARKER = "__RENDER_PATH__"

    html = _HTML_TEMPLATE
    html = html.replace(MATCH_DATA_MARKER, match_json, 1)
    html = html.replace(STATS_DATA_MARKER, stats_json, 1)
    html = html.replace(RENDER_PATH_MARKER, render_rel, 1)
    return html


# ---------------------------------------------------------------------------
# Monolithic HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEO-GOAL LOCAL — Tactical Analysis Report</title>
<style>
/* ── Reset & Base ─────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:14px;scroll-behavior:smooth}
body{
  background:#0d1117;color:#c9d1d9;
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif;
  line-height:1.6;padding:1rem;
}
a{color:#3fb950;text-decoration:none}
a:hover{text-decoration:underline}

/* ── Layout ───────────────────────────────────────────────────── */
.container{max-width:1200px;margin:0 auto}

header{
  text-align:center;padding:1.5rem 1rem;
  border-bottom:1px solid #30363d;margin-bottom:1.5rem;
}
header h1{font-size:1.6rem;font-weight:700;color:#3fb950;letter-spacing:.04em}
header p{font-size:.85rem;color:#8b949e;margin-top:.25rem}

.card{
  background:#161b22;border:1px solid #30363d;
  border-radius:10px;padding:1.25rem;margin-bottom:1.5rem;
  overflow:hidden;
}
.card h2{
  font-size:1.05rem;font-weight:600;color:#e6edf3;
  margin-bottom:.75rem;display:flex;align-items:center;gap:.5rem;
}
.card h2 .icon{font-size:1.1rem}

/* ── Canvas / Tactical board ──────────────────────────────────── */
.canvas-wrap{position:relative;width:100%;overflow:hidden;border-radius:8px}
#tacticalCanvas{display:block;width:100%;background:#1a7a3a;border-radius:8px;cursor:grab}
#tacticalCanvas:active{cursor:grabbing}

.controls{
  display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;
  margin-top:.65rem;
}
.controls button{
  background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  border-radius:6px;padding:.3rem .7rem;cursor:pointer;font-size:.8rem;
}
.controls button:hover{background:#30363d}
.controls input[type=range]{flex:1;min-width:120px;accent-color:#3fb950}
.controls .time{font-size:.78rem;color:#8b949e;min-width:64px;text-align:right}

.layer-toggles{
  display:flex;flex-wrap:wrap;gap:.75rem;margin-top:.55rem;
}
.layer-toggles label{
  font-size:.78rem;color:#8b949e;display:flex;align-items:center;gap:.3rem;
  cursor:pointer;user-select:none;
}
.layer-toggles input[type=checkbox]{accent-color:#3fb950}

/* ── Stats grid ───────────────────────────────────────────────── */
.stats-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;
}
@media(max-width:800px){.stats-grid{grid-template-columns:1fr}}

/* ── Sortable table ───────────────────────────────────────────── */
.tbl-wrap{overflow-x:auto}
table.stats{width:100%;border-collapse:collapse;font-size:.8rem}
table.stats th,table.stats td{
  padding:.45rem .6rem;text-align:left;border-bottom:1px solid #21262d;
}
table.stats th{
  cursor:pointer;color:#8b949e;user-select:none;position:sticky;top:0;
  background:#161b22;
}
table.stats th:hover{color:#e6edf3}
table.stats th .arrow{font-size:.65rem;margin-left:.2rem}
table.stats tr:hover td{background:#1c2128}
.team-home{color:#f85149}
.team-away{color:#58a6ff}
.team-ref{color:#d29922}

/* ── Bar chart ────────────────────────────────────────────────── */
.bar-row{display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem;font-size:.8rem}
.bar-label{min-width:130px;color:#8b949e}
.bar-track{flex:1;height:18px;background:#21262d;border-radius:4px;position:relative;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .4s ease}
.bar-value{min-width:52px;text-align:right;font-variant-numeric:tabular-nums}

/* ── Video ────────────────────────────────────────────────────── */
.video-wrap{text-align:center}
.video-wrap video{max-width:100%;border-radius:8px;border:1px solid #30363d}
.video-placeholder{
  padding:2rem;color:#484f58;font-style:italic;
  border:1px dashed #30363d;border-radius:8px;text-align:center;
}

/* ── Before/After ─────────────────────────────────────────────── */
.compare{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:700px){.compare{grid-template-columns:1fr}}
.compare-side{
  border:1px solid #30363d;border-radius:8px;overflow:hidden;
  display:flex;flex-direction:column;align-items:center;
  padding:1rem;background:#0d1117;
}
.compare-side h3{font-size:.82rem;color:#8b949e;margin-bottom:.5rem}
#miniCanvas{border-radius:6px;width:100%;max-width:480px;background:#1a7a3a}

/* ── Homography ───────────────────────────────────────────────── */
.matrix-grid{
  display:inline-grid;grid-template-columns:repeat(3,1fr);gap:4px;
  font-family:'Courier New',Courier,monospace;font-size:.82rem;
}
.matrix-grid .cell{
  background:#21262d;padding:.4rem .7rem;border-radius:4px;text-align:right;
  min-width:90px;
}
.matrix-bracket{
  display:flex;align-items:center;gap:.5rem;justify-content:center;
}
.bracket{font-size:3.5rem;font-weight:100;color:#484f58;line-height:1}

/* ── Footer ───────────────────────────────────────────────────── */
footer{text-align:center;padding:1.5rem;color:#484f58;font-size:.72rem}
</style>
</head>
<body>
<!-- ═══════════════ Embedded Data ═══════════════════════════════ -->
<script id="match-data" type="application/json">"__MATCH_DATA__"</script>
<script id="stats-data" type="application/json">"__STATS_DATA__"</script>

<div class="container">

<!-- ── Header ──────────────────────────────────────────────────── -->
<header>
  <h1>⚽ GEO-GOAL LOCAL — Tactical Analysis Report</h1>
  <p>Self-contained offline report &middot; Canvas 2D visualisations</p>
</header>

<!-- ── Tactical Board ──────────────────────────────────────────── -->
<div class="card">
  <h2><span class="icon">📐</span> Tactical Board</h2>
  <div class="canvas-wrap">
    <canvas id="tacticalCanvas" width="840" height="546"></canvas>
  </div>
  <div class="controls">
    <button id="btnPlay" title="Play">▶</button>
    <button id="btnPause" title="Pause">⏸</button>
    <button id="btnStop" title="Stop">■</button>
    <input id="scrub" type="range" min="0" max="1" value="0" step="1">
    <span class="time" id="timeLabel">0 / 0</span>
  </div>
  <div class="layer-toggles">
    <label><input type="checkbox" id="layerTrajectories"> Trajectories</label>
    <label><input type="checkbox" id="layerHeatmap"> Heatmap</label>
    <label><input type="checkbox" id="layerVoronoi"> Voronoi</label>
    <label><input type="checkbox" id="layerHull"> Convex Hull</label>
  </div>
  <div style="margin-top:.5rem">
    <label style="font-size:.78rem;color:#8b949e">Heatmap player:
      <select id="heatmapPlayer" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:2px 6px;font-size:.78rem">
        <option value="all">All</option>
      </select>
    </label>
  </div>
</div>

<!-- ── Stats ────────────────────────────────────────────────────── -->
<div class="stats-grid">

  <!-- Player Stats -->
  <div class="card">
    <h2><span class="icon">👤</span> Player Stats</h2>
    <div class="tbl-wrap">
      <table class="stats" id="playerTable">
        <thead><tr>
          <th data-key="playerId">ID <span class="arrow"></span></th>
          <th data-key="teamId">Team <span class="arrow"></span></th>
          <th data-key="totalDistanceM">Distance (m) <span class="arrow"></span></th>
          <th data-key="avgSpeedKmh">Avg Speed <span class="arrow"></span></th>
          <th data-key="maxSpeedKmh">Max Speed <span class="arrow"></span></th>
          <th data-key="sprintCount">Sprints <span class="arrow"></span></th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- Team Stats -->
  <div class="card">
    <h2><span class="icon">📊</span> Team Stats</h2>
    <div id="teamBars"></div>
  </div>

</div><!-- /stats-grid -->

<!-- ── Blender Render ──────────────────────────────────────────── -->
<div class="card">
  <h2><span class="icon">🎬</span> 3D Render (Blender)</h2>
  <div class="video-wrap" id="renderWrap"></div>
</div>

<!-- ── Before / After ──────────────────────────────────────────── -->
<div class="card">
  <h2><span class="icon">🔀</span> Before / After Comparison</h2>
  <div class="compare">
    <div class="compare-side">
      <h3>Original Perspective</h3>
      <div class="video-placeholder">
        Original video frame with perspective distortion.<br>
        <small>Run <code>geogoal process</code> to generate tracking data.</small>
      </div>
    </div>
    <div class="compare-side">
      <h3>Tactical View (Frame 0)</h3>
      <canvas id="miniCanvas" width="480" height="312"></canvas>
    </div>
  </div>
</div>

<!-- ── Homography Matrix ───────────────────────────────────────── -->
<div class="card">
  <h2><span class="icon">🧮</span> Homography Matrix <em>H</em></h2>
  <div class="matrix-bracket" id="matrixWrap"></div>
</div>

<footer>GEO-GOAL LOCAL &copy; 2025 — generated offline</footer>

</div><!-- /container -->

<!-- ═══════════════ JavaScript ═════════════════════════════════ -->
<script>
"use strict";
/* ================================================================
   0. Parse embedded data
   ================================================================ */
const MATCH = JSON.parse(document.getElementById("match-data").textContent);
const STATS = JSON.parse(document.getElementById("stats-data").textContent);
const RENDER_REL = "__RENDER_PATH__";

const FRAMES = MATCH.frames || [];
const PITCH_W = (MATCH.pitch && MATCH.pitch.length_m) || 105;
const PITCH_H = (MATCH.pitch && MATCH.pitch.width_m) || 68;
const MARGIN = 5; // metres of padding around pitch

/* ================================================================
   1. Tactical Canvas Setup
   ================================================================ */
const canvas = document.getElementById("tacticalCanvas");
const ctx = canvas.getContext("2d");

/* ── View transform (pan / zoom) ────────────────────────────── */
let viewTransform = { scale: 1, panX: 0, panY: 0 };
let isDragging = false, dragStart = { x: 0, y: 0 }, panStart = { x: 0, y: 0 };

canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    viewTransform.panX = mx - (mx - viewTransform.panX) * factor;
    viewTransform.panY = my - (my - viewTransform.panY) * factor;
    viewTransform.scale *= factor;
    draw();
}, { passive: false });

canvas.addEventListener("mousedown", e => {
    isDragging = true;
    const rect = canvas.getBoundingClientRect();
    dragStart.x = e.clientX; dragStart.y = e.clientY;
    panStart.x = viewTransform.panX; panStart.y = viewTransform.panY;
});
canvas.addEventListener("mousemove", e => {
    if (!isDragging) return;
    const dx = (e.clientX - dragStart.x) * (canvas.width / canvas.getBoundingClientRect().width);
    const dy = (e.clientY - dragStart.y) * (canvas.height / canvas.getBoundingClientRect().height);
    viewTransform.panX = panStart.x + dx;
    viewTransform.panY = panStart.y + dy;
    draw();
});
canvas.addEventListener("mouseup", () => isDragging = false);
canvas.addEventListener("mouseleave", () => isDragging = false);

/* ── Coordinate transforms ──────────────────────────────────── */
function baseScale() {
    return Math.min(
        canvas.width / (PITCH_W + MARGIN * 2),
        canvas.height / (PITCH_H + MARGIN * 2)
    );
}
function metersToCanvas(xm, ym) {
    const s = baseScale();
    const ox = (canvas.width - PITCH_W * s) / 2;
    const oy = (canvas.height - PITCH_H * s) / 2;
    let cx = ox + xm * s;
    let cy = oy + ym * s;
    // apply view transform
    cx = cx * viewTransform.scale + viewTransform.panX;
    cy = cy * viewTransform.scale + viewTransform.panY;
    return [cx, cy];
}

/* ================================================================
   2. Pitch Drawing
   ================================================================ */
function drawPitch(context, w, h, applyView) {
    const s = Math.min(w / (PITCH_W + MARGIN * 2), h / (PITCH_H + MARGIN * 2));
    const ox = (w - PITCH_W * s) / 2;
    const oy = (h - PITCH_H * s) / 2;

    function m2c(xm, ym) {
        let cx = ox + xm * s, cy = oy + ym * s;
        if (applyView) {
            cx = cx * viewTransform.scale + viewTransform.panX;
            cy = cy * viewTransform.scale + viewTransform.panY;
        }
        return [cx, cy];
    }
    function ms(v) { return v * s * (applyView ? viewTransform.scale : 1); }

    // Background
    context.fillStyle = "#1a7a3a";
    context.fillRect(0, 0, w, h);

    // Grass stripes
    const stripeW = PITCH_W / 12;
    for (let i = 0; i < 12; i++) {
        const [sx, sy] = m2c(i * stripeW, 0);
        const sw = ms(stripeW);
        const sh = ms(PITCH_H);
        context.fillStyle = i % 2 === 0 ? "#1e8c42" : "#1a7a3a";
        context.fillRect(sx, sy, sw + 1, sh + 1);
    }

    context.strokeStyle = "rgba(255,255,255,0.85)";
    context.lineWidth = Math.max(1, ms(0.15));

    // Pitch outline
    const [tlx, tly] = m2c(0, 0);
    context.strokeRect(tlx, tly, ms(PITCH_W), ms(PITCH_H));

    // Halfway line
    const [hx0, hy0] = m2c(PITCH_W / 2, 0);
    const [hx1, hy1] = m2c(PITCH_W / 2, PITCH_H);
    context.beginPath(); context.moveTo(hx0, hy0); context.lineTo(hx1, hy1); context.stroke();

    // Center circle (r = 9.15m)
    const [ccx, ccy] = m2c(PITCH_W / 2, PITCH_H / 2);
    context.beginPath(); context.arc(ccx, ccy, ms(9.15), 0, Math.PI * 2); context.stroke();

    // Center spot
    context.fillStyle = "rgba(255,255,255,0.85)";
    context.beginPath(); context.arc(ccx, ccy, ms(0.4), 0, Math.PI * 2); context.fill();

    // Penalty areas (16.5m deep, 40.32m wide)
    const paW = 16.5, paH = 40.32, paOff = (PITCH_H - paH) / 2;
    const [pa1x, pa1y] = m2c(0, paOff);
    context.strokeRect(pa1x, pa1y, ms(paW), ms(paH));
    const [pa2x, pa2y] = m2c(PITCH_W - paW, paOff);
    context.strokeRect(pa2x, pa2y, ms(paW), ms(paH));

    // Goal areas (5.5m deep, 18.32m wide)
    const gaW = 5.5, gaH = 18.32, gaOff = (PITCH_H - gaH) / 2;
    const [ga1x, ga1y] = m2c(0, gaOff);
    context.strokeRect(ga1x, ga1y, ms(gaW), ms(gaH));
    const [ga2x, ga2y] = m2c(PITCH_W - gaW, gaOff);
    context.strokeRect(ga2x, ga2y, ms(gaW), ms(gaH));

    // Penalty spots (11m from goal line)
    context.fillStyle = "rgba(255,255,255,0.85)";
    const [ps1x, ps1y] = m2c(11, PITCH_H / 2);
    context.beginPath(); context.arc(ps1x, ps1y, ms(0.35), 0, Math.PI * 2); context.fill();
    const [ps2x, ps2y] = m2c(PITCH_W - 11, PITCH_H / 2);
    context.beginPath(); context.arc(ps2x, ps2y, ms(0.35), 0, Math.PI * 2); context.fill();

    // Penalty arcs
    const arcR = 9.15;
    const arcAngle = Math.acos(5.5 / arcR);
    const [arc1x, arc1y] = m2c(11, PITCH_H / 2);
    context.beginPath();
    context.arc(arc1x, arc1y, ms(arcR), -arcAngle, arcAngle);
    context.stroke();
    const [arc2x, arc2y] = m2c(PITCH_W - 11, PITCH_H / 2);
    context.beginPath();
    context.arc(arc2x, arc2y, ms(arcR), Math.PI - arcAngle, Math.PI + arcAngle);
    context.stroke();

    // Corner arcs
    const cR = 1;
    for (const [cx2, cy2, sa, ea] of [
        [0, 0, 0, Math.PI / 2],
        [PITCH_W, 0, Math.PI / 2, Math.PI],
        [PITCH_W, PITCH_H, Math.PI, 3 * Math.PI / 2],
        [0, PITCH_H, 3 * Math.PI / 2, 2 * Math.PI],
    ]) {
        const [ax, ay] = m2c(cx2, cy2);
        context.beginPath();
        context.arc(ax, ay, ms(cR), sa, ea);
        context.stroke();
    }
}

/* ================================================================
   3. Algorithms
   ================================================================ */

/* ── Catmull-Rom spline ─────────────────────────────────────── */
function catmullRom(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    return [
        0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
        0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
    ];
}

/* ── Convex hull — Andrew's monotone chain ──────────────────── */
function convexHull(points) {
    if (points.length < 3) return points.slice();
    const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (O, A, B) => (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0]);
    const lower = [];
    for (const p of pts) {
        while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop();
        lower.push(p);
    }
    const upper = [];
    for (const p of pts.slice().reverse()) {
        while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop();
        upper.push(p);
    }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/* ── KDE heatmap ────────────────────────────────────────────── */
function computeKDE(positions, gridW, gridH, bandwidth) {
    const grid = Array.from({ length: gridH }, () => new Float32Array(gridW));
    const bw2 = bandwidth * bandwidth;
    for (const [px, py] of positions) {
        const gx = Math.floor(px / PITCH_W * gridW);
        const gy = Math.floor(py / PITCH_H * gridH);
        const r = Math.ceil(bandwidth / (PITCH_W / gridW) * 2);
        for (let dy = -r; dy <= r; dy++) {
            for (let dx = -r; dx <= r; dx++) {
                const ix = gx + dx, iy = gy + dy;
                if (ix < 0 || ix >= gridW || iy < 0 || iy >= gridH) continue;
                const wx = (ix / gridW * PITCH_W) - px;
                const wy = (iy / gridH * PITCH_H) - py;
                grid[iy][ix] += Math.exp(-(wx * wx + wy * wy) / (2 * bw2));
            }
        }
    }
    let max = 0;
    for (const row of grid) for (const v of row) if (v > max) max = v;
    if (max > 0) for (const row of grid) for (let i = 0; i < row.length; i++) row[i] /= max;
    return grid;
}

function heatmapColor(t) {
    // blue → cyan → green → yellow → red
    if (t < 0.25) { const s = t / 0.25; return [0, Math.round(s * 255), 255]; }
    if (t < 0.5)  { const s = (t - 0.25) / 0.25; return [0, 255, Math.round(255 * (1 - s))]; }
    if (t < 0.75) { const s = (t - 0.5) / 0.25; return [Math.round(255 * s), 255, 0]; }
    const s = (t - 0.75) / 0.25; return [255, Math.round(255 * (1 - s)), 0];
}

/* ================================================================
   4. Rendering layers
   ================================================================ */
let currentFrame = 0;
let playing = false;
let animId = null;

const layers = {
    trajectories: false,
    heatmap: false,
    voronoi: false,
    hull: false,
};

/* helper: classify team colors */
function teamColor(teamId, alpha) {
    alpha = alpha || 1;
    if (teamId === 1) return `rgba(248,81,73,${alpha})`;   // home red
    if (teamId === 2) return `rgba(88,166,255,${alpha})`;  // away blue
    if (teamId === 0) return `rgba(210,153,34,${alpha})`;  // referee yellow
    return `rgba(139,148,158,${alpha})`;                    // unknown
}
function teamClass(teamId) {
    if (teamId === 1) return "team-home";
    if (teamId === 2) return "team-away";
    if (teamId === 0) return "team-ref";
    return "";
}
function teamName(teamId) {
    if (teamId === 1) return "Home";
    if (teamId === 2) return "Away";
    if (teamId === 0) return "Ref";
    return "?";
}

/* ── Draw a single frame ────────────────────────────────────── */
function draw() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    drawPitch(ctx, W, H, true);

    if (FRAMES.length === 0) return;
    const frame = FRAMES[Math.min(currentFrame, FRAMES.length - 1)];
    const players = frame.players || [];
    const ball = frame.ball;

    /* ── Voronoi layer ─────────────────────────────────────── */
    if (layers.voronoi && players.length > 0) {
        drawVoronoi(players);
    }

    /* ── Heatmap layer ─────────────────────────────────────── */
    if (layers.heatmap) {
        drawHeatmap();
    }

    /* ── Convex hull layer ─────────────────────────────────── */
    if (layers.hull) {
        drawHulls(players);
    }

    /* ── Trajectory layer ──────────────────────────────────── */
    if (layers.trajectories) {
        drawTrajectories();
    }

    /* ── Players ───────────────────────────────────────────── */
    for (const p of players) {
        const [cx, cy] = metersToCanvas(p.x, p.y);
        const r = 6 * viewTransform.scale;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = teamColor(p.teamId);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.stroke();
        // label
        ctx.fillStyle = "#fff";
        ctx.font = `bold ${Math.max(9, 10 * viewTransform.scale)}px system-ui`;
        ctx.textAlign = "center";
        ctx.fillText(p.playerId, cx, cy - r - 3);
    }

    /* ── Ball ──────────────────────────────────────────────── */
    if (ball) {
        const [bx, by] = metersToCanvas(ball.x, ball.y);
        const br = 5 * viewTransform.scale;
        ctx.beginPath();
        ctx.arc(bx, by, br, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#000";
        ctx.stroke();
    }

    /* ── Vignette overlay ──────────────────────────────────── */
    const grd = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
    grd.addColorStop(0, "rgba(0,0,0,0)");
    grd.addColorStop(1, "rgba(0,0,0,0.35)");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, W, H);
}

/* ── Voronoi (grid sampling) ────────────────────────────────── */
function drawVoronoi(players) {
    const res = 3; // metres per sample
    const cols = Math.ceil(PITCH_W / res);
    const rows = Math.ceil(PITCH_H / res);
    const fieldPlayers = players.filter(p => p.teamId === 1 || p.teamId === 2);
    if (fieldPlayers.length === 0) return;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const sx = c * res + res / 2;
            const sy = r * res + res / 2;
            let minD = Infinity, closest = null;
            for (const p of fieldPlayers) {
                const d = (p.x - sx) ** 2 + (p.y - sy) ** 2;
                if (d < minD) { minD = d; closest = p; }
            }
            if (closest) {
                const [cx, cy] = metersToCanvas(c * res, r * res);
                const [cx2, cy2] = metersToCanvas(c * res + res, r * res + res);
                ctx.fillStyle = teamColor(closest.teamId, 0.15);
                ctx.fillRect(cx, cy, cx2 - cx, cy2 - cy);
            }
        }
    }
}

/* ── Heatmap ────────────────────────────────────────────────── */
function drawHeatmap() {
    const sel = document.getElementById("heatmapPlayer").value;
    let positions = [];

    const endIdx = Math.min(currentFrame + 1, FRAMES.length);
    for (let fi = 0; fi < endIdx; fi++) {
        const fr = FRAMES[fi];
        for (const p of (fr.players || [])) {
            if (sel === "all" || String(p.playerId) === sel) {
                if (p.teamId === 1 || p.teamId === 2) {
                    positions.push([p.x, p.y]);
                }
            }
        }
    }
    if (positions.length === 0) return;

    // Sample to limit computation
    if (positions.length > 3000) {
        const step = Math.ceil(positions.length / 3000);
        positions = positions.filter((_, i) => i % step === 0);
    }

    const gw = 53, gh = 34;
    const grid = computeKDE(positions, gw, gh, 5);
    const cellW = PITCH_W / gw, cellH = PITCH_H / gh;

    for (let r = 0; r < gh; r++) {
        for (let c = 0; c < gw; c++) {
            const v = grid[r][c];
            if (v < 0.02) continue;
            const [x1, y1] = metersToCanvas(c * cellW, r * cellH);
            const [x2, y2] = metersToCanvas((c + 1) * cellW, (r + 1) * cellH);
            const [cr2, cg, cb] = heatmapColor(v);
            ctx.fillStyle = `rgba(${cr2},${cg},${cb},${0.45 * v + 0.05})`;
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        }
    }
}

/* ── Convex hulls ───────────────────────────────────────────── */
function drawHulls(players) {
    for (const tid of [1, 2]) {
        const pts = players.filter(p => p.teamId === tid).map(p => [p.x, p.y]);
        if (pts.length < 3) continue;
        const hull = convexHull(pts);
        ctx.beginPath();
        const [hx0, hy0] = metersToCanvas(hull[0][0], hull[0][1]);
        ctx.moveTo(hx0, hy0);
        for (let i = 1; i < hull.length; i++) {
            const [hx, hy] = metersToCanvas(hull[i][0], hull[i][1]);
            ctx.lineTo(hx, hy);
        }
        ctx.closePath();
        ctx.fillStyle = teamColor(tid, 0.1);
        ctx.fill();
        ctx.strokeStyle = teamColor(tid, 0.6);
        ctx.lineWidth = 1.5 * viewTransform.scale;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

/* ── Trajectories (Catmull-Rom) ─────────────────────────────── */
function drawTrajectories() {
    const trailLen = 30;
    const startF = Math.max(0, currentFrame - trailLen);
    const endF = Math.min(currentFrame + 1, FRAMES.length);
    if (endF - startF < 2) return;

    // gather per-player position history
    const tracks = {};
    for (let fi = startF; fi < endF; fi++) {
        for (const p of (FRAMES[fi].players || [])) {
            if (p.teamId !== 1 && p.teamId !== 2) continue;
            const key = p.playerId;
            if (!tracks[key]) tracks[key] = { teamId: p.teamId, pts: [] };
            tracks[key].pts.push([p.x, p.y]);
        }
    }

    for (const pid of Object.keys(tracks)) {
        const tr = tracks[pid];
        const pts = tr.pts;
        if (pts.length < 2) continue;

        ctx.beginPath();
        const segments = 6; // interpolation steps between control points
        for (let i = 0; i < pts.length - 1; i++) {
            const p0 = pts[Math.max(i - 1, 0)];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            const p3 = pts[Math.min(i + 2, pts.length - 1)];
            for (let s = 0; s <= segments; s++) {
                const t = s / segments;
                const [mx, my] = catmullRom(p0, p1, p2, p3, t);
                const [cx, cy] = metersToCanvas(mx, my);
                if (i === 0 && s === 0) ctx.moveTo(cx, cy);
                else ctx.lineTo(cx, cy);
            }
        }
        const alpha = 0.5;
        ctx.strokeStyle = teamColor(tr.teamId, alpha);
        ctx.lineWidth = 1.5 * viewTransform.scale;
        ctx.stroke();
    }
}

/* ================================================================
   5. Animation Controls
   ================================================================ */
const scrub = document.getElementById("scrub");
const timeLabel = document.getElementById("timeLabel");
const btnPlay = document.getElementById("btnPlay");
const btnPause = document.getElementById("btnPause");
const btnStop = document.getElementById("btnStop");

scrub.max = Math.max(0, FRAMES.length - 1);
scrub.value = 0;

function updateTimeLabel() {
    const f = FRAMES[currentFrame];
    const ms = f ? f.timestampMs : 0;
    const sec = (ms / 1000).toFixed(1);
    timeLabel.textContent = `${currentFrame} / ${FRAMES.length - 1}  (${sec}s)`;
    scrub.value = currentFrame;
}

let lastTick = 0;
function animate(ts) {
    if (!playing) return;
    if (ts - lastTick >= 33) { // ~30 fps
        lastTick = ts;
        if (currentFrame < FRAMES.length - 1) {
            currentFrame++;
            updateTimeLabel();
            draw();
        } else {
            playing = false;
        }
    }
    animId = requestAnimationFrame(animate);
}

btnPlay.addEventListener("click", () => {
    if (currentFrame >= FRAMES.length - 1) currentFrame = 0;
    playing = true;
    lastTick = 0;
    animId = requestAnimationFrame(animate);
});
btnPause.addEventListener("click", () => { playing = false; });
btnStop.addEventListener("click", () => {
    playing = false;
    currentFrame = 0;
    updateTimeLabel();
    draw();
});
scrub.addEventListener("input", () => {
    currentFrame = parseInt(scrub.value, 10);
    updateTimeLabel();
    draw();
});

/* ── Layer toggles ──────────────────────────────────────────── */
for (const [id, key] of [
    ["layerTrajectories", "trajectories"],
    ["layerHeatmap", "heatmap"],
    ["layerVoronoi", "voronoi"],
    ["layerHull", "hull"],
]) {
    document.getElementById(id).addEventListener("change", function () {
        layers[key] = this.checked;
        draw();
    });
}

/* ── Heatmap player selector ────────────────────────────────── */
(function populateHeatmapSelect() {
    const sel = document.getElementById("heatmapPlayer");
    const seen = new Set();
    for (const fr of FRAMES) {
        for (const p of (fr.players || [])) {
            if (!seen.has(p.playerId) && (p.teamId === 1 || p.teamId === 2)) {
                seen.add(p.playerId);
                const opt = document.createElement("option");
                opt.value = String(p.playerId);
                opt.textContent = `#${p.playerId} (${teamName(p.teamId)})`;
                sel.appendChild(opt);
            }
        }
    }
    sel.addEventListener("change", () => { if (layers.heatmap) draw(); });
})();

/* ================================================================
   6. Player Stats Table (sortable)
   ================================================================ */
(function buildPlayerTable() {
    const tbody = document.querySelector("#playerTable tbody");
    const pStats = STATS.players || {};
    const rows = Object.values(pStats).map(p => ({
        playerId: p.playerId,
        teamId: p.teamId,
        totalDistanceM: p.totalDistanceM || 0,
        avgSpeedKmh: p.avgSpeedKmh || 0,
        maxSpeedKmh: p.maxSpeedKmh || 0,
        sprintCount: p.sprintCount || 0,
    }));

    let sortKey = "playerId", sortAsc = true;

    function render() {
        const sorted = rows.slice().sort((a, b) => {
            const va = a[sortKey], vb = b[sortKey];
            return sortAsc ? (va < vb ? -1 : va > vb ? 1 : 0)
                           : (va > vb ? -1 : va < vb ? 1 : 0);
        });
        tbody.innerHTML = "";
        for (const r of sorted) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="${teamClass(r.teamId)}">${r.playerId}</td>
                <td class="${teamClass(r.teamId)}">${teamName(r.teamId)}</td>
                <td>${r.totalDistanceM.toFixed(1)}</td>
                <td>${r.avgSpeedKmh.toFixed(2)} km/h</td>
                <td>${r.maxSpeedKmh.toFixed(2)} km/h</td>
                <td>${r.sprintCount}</td>`;
            tbody.appendChild(tr);
        }
        // Update arrows
        document.querySelectorAll("#playerTable th").forEach(th => {
            const k = th.dataset.key;
            th.querySelector(".arrow").textContent = k === sortKey ? (sortAsc ? "▲" : "▼") : "";
        });
    }

    document.querySelectorAll("#playerTable th").forEach(th => {
        th.addEventListener("click", () => {
            const k = th.dataset.key;
            if (sortKey === k) sortAsc = !sortAsc;
            else { sortKey = k; sortAsc = true; }
            render();
        });
    });

    render();
})();

/* ================================================================
   7. Team Stats Bars
   ================================================================ */
(function buildTeamBars() {
    const wrap = document.getElementById("teamBars");
    const possession = STATS.possession || {};
    const voronoi = STATS.voronoiControl || {};
    const teams = STATS.teams || {};
    const formations = STATS.formations || {};

    const bars = [
        { label: "Possession", home: ((possession.home || 0) * 100).toFixed(1) + "%", away: ((possession.away || 0) * 100).toFixed(1) + "%", homeVal: (possession.home || 0) * 100, awayVal: (possession.away || 0) * 100 },
        { label: "Voronoi Control", home: ((voronoi.home || 0) * 100).toFixed(1) + "%", away: ((voronoi.away || 0) * 100).toFixed(1) + "%", homeVal: (voronoi.home || 0) * 100, awayVal: (voronoi.away || 0) * 100 },
    ];

    for (const b of bars) {
        const total = b.homeVal + b.awayVal || 1;
        wrap.innerHTML += `
        <div class="bar-row">
            <span class="bar-label">${b.label}</span>
            <span class="bar-value team-home">${b.home}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:${b.homeVal / total * 100}%;background:#f85149"></div>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${b.awayVal / total * 100}%;background:#58a6ff;margin-left:auto"></div>
            </div>
            <span class="bar-value team-away">${b.away}</span>
        </div>`;
    }

    // Scalar stats
    const homeTeam = teams["1"] || {};
    const awayTeam = teams["2"] || {};
    const items = [
        { label: "Avg Compaction", home: (homeTeam.avgCompaction || 0).toFixed(2) + " m", away: (awayTeam.avgCompaction || 0).toFixed(2) + " m" },
        { label: "Avg Hull Area", home: (homeTeam.avgConvexHullArea || 0).toFixed(1) + " m²", away: (awayTeam.avgConvexHullArea || 0).toFixed(1) + " m²" },
        { label: "Formation", home: formations.home || "—", away: formations.away || "—" },
    ];
    for (const it of items) {
        wrap.innerHTML += `
        <div class="bar-row">
            <span class="bar-label">${it.label}</span>
            <span class="bar-value team-home" style="flex:1;text-align:center">${it.home}</span>
            <span class="bar-value team-away" style="flex:1;text-align:center">${it.away}</span>
        </div>`;
    }
})();

/* ================================================================
   8. Blender Render Embed
   ================================================================ */
(function embedRender() {
    const wrap = document.getElementById("renderWrap");
    if (RENDER_REL) {
        wrap.innerHTML = `<video controls width="100%" style="max-width:960px">
            <source src="${RENDER_REL}" type="video/mp4">
            <p>Blender render not available. Run: <code>geogoal blender</code></p>
        </video>`;
    } else {
        wrap.innerHTML = `<div class="video-placeholder">Blender render not available.<br><small>Run <code>geogoal blender</code> to generate.</small></div>`;
    }
})();

/* ================================================================
   9. Homography Matrix
   ================================================================ */
(function showMatrix() {
    const wrap = document.getElementById("matrixWrap");
    const H = MATCH.homography;
    if (!H || !Array.isArray(H)) {
        wrap.innerHTML = `<span style="color:#484f58">Homography matrix not available.</span>`;
        return;
    }
    let cells = "";
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            const v = (H[r] && H[r][c] != null) ? H[r][c].toExponential(4) : "—";
            cells += `<span class="cell">${v}</span>`;
        }
    }
    wrap.innerHTML = `<span class="bracket">[</span><div class="matrix-grid">${cells}</div><span class="bracket">]</span>`;
})();

/* ================================================================
   10. Mini canvas (Before/After — frame 0)
   ================================================================ */
(function drawMini() {
    const mc = document.getElementById("miniCanvas");
    const mctx = mc.getContext("2d");
    drawPitch(mctx, mc.width, mc.height, false);

    if (FRAMES.length === 0) return;
    const fr = FRAMES[0];
    const s = Math.min(mc.width / (PITCH_W + MARGIN * 2), mc.height / (PITCH_H + MARGIN * 2));
    const ox = (mc.width - PITCH_W * s) / 2;
    const oy = (mc.height - PITCH_H * s) / 2;
    function m2c(xm, ym) { return [ox + xm * s, oy + ym * s]; }

    for (const p of (fr.players || [])) {
        const [cx, cy] = m2c(p.x, p.y);
        mctx.beginPath();
        mctx.arc(cx, cy, 4, 0, Math.PI * 2);
        mctx.fillStyle = teamColor(p.teamId);
        mctx.fill();
        mctx.lineWidth = 1;
        mctx.strokeStyle = "rgba(255,255,255,0.6)";
        mctx.stroke();
    }
    if (fr.ball) {
        const [bx, by] = m2c(fr.ball.x, fr.ball.y);
        mctx.beginPath(); mctx.arc(bx, by, 3, 0, Math.PI * 2);
        mctx.fillStyle = "#fff"; mctx.fill();
    }
})();

/* ── Initial draw ──────────────────────────────────────────── */
updateTimeLabel();
draw();
</script>
</body>
</html>
'''

# ---------------------------------------------------------------------------
# CLI entry-point guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_report()
