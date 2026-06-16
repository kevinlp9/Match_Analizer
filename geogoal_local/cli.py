"""Geo-Goal Local CLI — offline football tactical analysis.

Usage:
    geogoal select <video>          Open interactive calibration selector
    geogoal process <video> [opts]  Run detection + tracking + export
    geogoal report                  Generate HTML report from match_data.json
    geogoal blender                 Build and render Blender 3D scene
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _cmd_select(args: argparse.Namespace) -> None:
    """Launch the interactive calibration selector."""
    from geogoal_local.selector import run_selector

    run_selector(
        video_path=args.video,
        output_dir=args.output_dir,
        port=args.port,
    )


def _cmd_process(args: argparse.Namespace) -> None:
    """Run the detection + tracking + export pipeline."""
    from geogoal_local.video_processor import VideoProcessor

    vp = VideoProcessor(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
    )
    vp.load_video(args.video)

    calib_path = Path(args.calib)
    if calib_path.exists():
        print(f"[cli] Loading calibration from {calib_path}")
        vp.load_calibration(str(calib_path))
    elif args.src_pts:
        src_pts = np.array(json.loads(args.src_pts), dtype=np.float32)
        vp.set_homography(src_pts=src_pts, method=args.dlt_method)
    else:
        print("[cli] No calibration found — using default source/destination points")
        vp.set_homography(method=args.dlt_method)

    export_path = "match_data.json"
    vp.process(
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        visualize_every=args.viz_every,
        export_json=export_path,
    )

    # Compute analytics if available
    try:
        from geogoal_local import analytics  # type: ignore[attr-defined]

        data_file = Path(args.output_dir) / "match_data.json"
        stats_file = Path(args.output_dir) / "stats.json"
        if data_file.exists():
            analytics.load_and_compute(str(data_file), str(stats_file))
    except ImportError:
        print("[cli] Analytics module not found — skipping stats computation")


def _cmd_report(args: argparse.Namespace) -> None:
    """Generate an HTML report from match data."""
    try:
        from geogoal_local import report  # type: ignore[attr-defined]

        data_path = Path(args.output_dir) / args.data
        stats_path = Path(args.output_dir) / args.stats
        report.generate(
            data_path=str(data_path),
            stats_path=str(stats_path),
            output_dir=args.output_dir,
        )
    except ImportError:
        print("[cli] Report module not yet implemented.")
        sys.exit(1)


def _cmd_blender(args: argparse.Namespace) -> None:
    """Build and render a Blender 3D scene."""
    blender_bin = shutil.which("blender")
    if blender_bin is None:
        print(
            "[cli] Blender not found on PATH.\n"
            "      Install Blender (https://www.blender.org/download/) and ensure\n"
            "      the 'blender' command is available in your terminal."
        )
        sys.exit(1)

    script = Path("geogoal_local/blender/build_scene.py")
    if not script.exists():
        print(f"[cli] Blender script not found: {script}")
        sys.exit(1)

    cmd = [
        blender_bin,
        "--background",
        "--python",
        str(script),
        "--",
        "--data",
        args.data,
        "--out",
        args.out,
        "--blend",
        args.blend,
        "--fps",
        str(args.fps),
    ]
    print(f"[cli] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _cmd_gui(args: argparse.Namespace) -> None:
    """Launch the web-based GUI dashboard."""
    from geogoal_local.gui import run_gui

    run_gui(port=args.port, output_dir=args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geogoal",
        description="Geo-Goal Local — offline football tactical analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── select ────────────────────────────────────────────────────────────
    p_sel = sub.add_parser("select", help="Open interactive calibration selector")
    p_sel.add_argument("video", help="Path to the video file")
    p_sel.add_argument("--port", type=int, default=8989, help="Server port (default: 8989)")
    p_sel.add_argument("--output-dir", default="./output", help="Output directory")
    p_sel.set_defaults(func=_cmd_select)

    # ── process ───────────────────────────────────────────────────────────
    p_proc = sub.add_parser("process", help="Run detection + tracking + export")
    p_proc.add_argument("video", help="Path to the video file")
    p_proc.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    p_proc.add_argument("--device", default="cpu", help="Inference device")
    p_proc.add_argument("--max-frames", type=int, default=-1, help="Max frames (-1=all)")
    p_proc.add_argument("--frame-skip", type=int, default=0, help="Process every Nth frame")
    p_proc.add_argument("--viz-every", type=int, default=0, help="Visualize every N frames")
    p_proc.add_argument("--output-dir", default="./output", help="Output directory")
    p_proc.add_argument("--calib", default="output/calib.json", help="Calibration file")
    p_proc.add_argument("--dlt-method", default="cv2", choices=["cv2", "manual"],
                        help="DLT method")
    p_proc.add_argument("--src-pts", default=None,
                        help="Source points as JSON string (overrides calib)")
    p_proc.set_defaults(func=_cmd_process)

    # ── report ────────────────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Generate HTML report from match data")
    p_rep.add_argument("--output-dir", default="./output", help="Output directory")
    p_rep.add_argument("--data", default="match_data.json", help="Match data filename")
    p_rep.add_argument("--stats", default="stats.json", help="Stats filename")
    p_rep.set_defaults(func=_cmd_report)

    # ── blender ───────────────────────────────────────────────────────────
    p_bl = sub.add_parser("blender", help="Build and render Blender 3D scene")
    p_bl.add_argument("--data", default="output/match_data.json", help="Match data path")
    p_bl.add_argument("--out", default="output/render.mp4", help="Output render path")
    p_bl.add_argument("--blend", default="output/scene.blend", help="Blender scene path")
    p_bl.add_argument("--fps", type=int, default=25, help="Render FPS")
    p_bl.set_defaults(func=_cmd_blender)

    # ── gui ───────────────────────────────────────────────────────────────
    p_gui = sub.add_parser("gui", help="Launch web-based GUI dashboard")
    p_gui.add_argument("--port", type=int, default=8888, help="Server port (default: 8888)")
    p_gui.add_argument("--output-dir", default="./output", help="Output directory")
    p_gui.set_defaults(func=_cmd_gui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
