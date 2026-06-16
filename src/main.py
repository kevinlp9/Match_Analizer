from __future__ import annotations

import argparse
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from m2m_client import M2MClient
from api_client import APIClient
from video_processor import VideoProcessor, annotate_keypoints


def main() -> None:
    parser = argparse.ArgumentParser(prog="geo-goal-ai")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- api ----
    api_p = sub.add_parser("api", help="Interact with Geo-Goal REST API")
    api_p.add_argument("--match-analytics", type=int, help="Match id to fetch analytics")
    api_p.add_argument("--match-detail", type=int, help="Match id to fetch detail")
    api_p.add_argument("--check-connection", action="store_true", help="Verify M2M token flow")

    # ---- annotate ----
    ann_p = sub.add_parser("annotate", help="Annotate keypoints on a video frame")
    ann_p.add_argument("video", help="Path to video file")
    ann_p.add_argument("--frame", type=int, default=0, help="Frame number to annotate (default: 0)")
    ann_p.add_argument("--num-points", type=int, default=4, help="Number of points (default: 4)")

    # ---- process ----
    proc_p = sub.add_parser("process", help="Run full video processing pipeline")
    proc_p.add_argument("video", help="Path to video file")
    proc_p.add_argument("--model", default="yolov8n.pt", help="YOLO model (default: yolov8n.pt)")
    proc_p.add_argument("--device", default="cpu", help="Inference device (default: cpu)")
    proc_p.add_argument("--max-frames", type=int, default=-1, help="Max frames to process (-1 = all)")
    proc_p.add_argument("--frame-skip", type=int, default=0, help="Process every Nth frame (0 = all)")
    proc_p.add_argument("--viz-every", type=int, default=0, help="Save pitch plot every N frames (0 = off)")
    proc_p.add_argument("--output-dir", default="./output", help="Output directory for JSON & plots")
    proc_p.add_argument("--push-match-id", type=int, help="Match id to push data to API")
    proc_p.add_argument("--dlt-method", default="cv2", choices=["cv2", "manual"], help="DLT method")
    proc_p.add_argument(
        "--src-pts", type=str, default=None,
        help="JSON array of 4 source points, e.g. '[[509,183],[639,307],[0,306],[125,184]]'"
    )
    proc_p.add_argument("--job-id", type=int, default=None, help="MatchAnalysisJob id for progress reporting")

    # ---- extract-frame ----
    ext_p = sub.add_parser("extract-frame", help="Extract a single frame as base64 JPEG")
    ext_p.add_argument("video", help="Path to video file")
    ext_p.add_argument("--frame", type=int, default=0, help="Frame number to extract (default: 0)")

    # ---- serve ----
    serve_p = sub.add_parser("serve", help="Start the AI service API (FastAPI + worker)")
    serve_p.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")

    args = parser.parse_args()

    # Fallback: if no subcommand, emulate old behaviour
    if args.command is None:
        _legacy_flat_args()
        return

    API_BASE = os.environ.get("GEO_API_URL")

    if args.command == "api":
        _cmd_api(args, API_BASE)
    elif args.command == "annotate":
        pts = annotate_keypoints(
            args.video,
            frame_number=args.frame,
            num_points=args.num_points,
        )
        print("Copy these into your code / config as SRC_PTS:")
        print(repr(pts))
    elif args.command == "process":
        _cmd_process(args, API_BASE)
    elif args.command == "extract-frame":
        import json as _json
        import base64 as _b64
        import cv2
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(_json.dumps({"error": f"Cannot open video: {args.video}"}))
            raise SystemExit(1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(_json.dumps({"error": f"Cannot read frame {args.frame}"}))
            raise SystemExit(1)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = _b64.b64encode(buf).decode("ascii")
        w = int(frame.shape[1])
        h = int(frame.shape[0])
        print(_json.dumps({"frame": b64, "width": w, "height": h}))

    elif args.command == "serve":
        import uvicorn
        from api import app
        uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_api(args: argparse.Namespace, api_base: str) -> None:
    CLIENT_ID = os.environ.get("M2M_CLIENT_ID")
    CLIENT_SECRET = os.environ.get("M2M_CLIENT_SECRET")
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Please set M2M_CLIENT_ID and M2M_CLIENT_SECRET in the environment.")
        raise SystemExit(2)

    m2m = M2MClient(api_base, CLIENT_ID, CLIENT_SECRET)
    api = APIClient(api_base, m2m)

    if args.check_connection:
        try:
            token = m2m.get_token()
            print("Connection OK — token fetched (length:", len(token), ")")
        except Exception as e:
            print("Connection failed:", str(e))
        return

    if args.match_analytics:
        print(api.get_match_analytics(args.match_analytics))
    if args.match_detail:
        print(api.get_match_detail(args.match_detail))


def _cmd_process(args: argparse.Namespace, api_base: str) -> None:
    import json as json_mod

    vp = VideoProcessor(
        model_name=args.model,
        device=args.device,
        api_base=api_base,
        output_dir=args.output_dir,
    )
    vp.load_video(args.video)

    src_pts = None
    if args.src_pts:
        try:
            pts_list = json_mod.loads(args.src_pts)
            # Accept two formats:
            #   [{"x":509,"y":183}, ...]   (frontend)
            #   [[509, 183], [639, 307], ...]  (manual CLI)
            if isinstance(pts_list[0], dict):
                src_pts = np.array([[pt["x"], pt["y"]] for pt in pts_list], dtype=np.float32)
            else:
                src_pts = np.array(pts_list, dtype=np.float32)
            print(f"[homography] Using custom src_pts: {src_pts.tolist()}")
        except Exception as e:
            print(f"[homography] Failed to parse --src-pts: {e}. Using defaults.")

    vp.set_homography(src_pts=src_pts, method=args.dlt_method)
    vp.process(
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        visualize_every=args.viz_every,
        export_json="match_data.json",
        push_match_id=args.push_match_id,
        job_id=args.job_id,
    )


def _legacy_flat_args() -> None:
    """Backwards-compatible flat-arg parser (no subcommand)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-analytics", type=int, help="match id to fetch analytics")
    parser.add_argument("--match-detail", type=int, help="match id to fetch detail")
    parser.add_argument("--check-connection", action="store_true")
    args = parser.parse_args()

    API_BASE = os.environ.get("GEO_API_URL")
    CLIENT_ID = os.environ.get("M2M_CLIENT_ID")
    CLIENT_SECRET = os.environ.get("M2M_CLIENT_SECRET")

    if not CLIENT_ID or not CLIENT_SECRET:
        print("Please set M2M_CLIENT_ID and M2M_CLIENT_SECRET in the environment.")
        raise SystemExit(2)

    m2m = M2MClient(API_BASE, CLIENT_ID, CLIENT_SECRET)
    api = APIClient(API_BASE, m2m)

    if args.check_connection:
        try:
            token = m2m.get_token()
            print("Connection OK — token fetched (length:", len(token), ")")
        except Exception as e:
            print("Connection failed:", str(e))
        return

    if args.match_analytics:
        print(api.get_match_analytics(args.match_analytics))
    if args.match_detail:
        print(api.get_match_detail(args.match_detail))


if __name__ == "__main__":
    main()
