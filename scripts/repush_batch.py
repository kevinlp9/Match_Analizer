"""
Recovery script — re-pushea un match_data.json ya generado al backend.

Útil cuando el procesamiento fue exitoso pero el push falló (por payload
limit, validador, red, etc.) y no quieres re-procesar el video.

Uso:
    python3 scripts/repush_batch.py <match_id> [path_to_json]

Por defecto busca el JSON en output/<match_id>/match_data.json.

Tras un push exitoso, marca el job más reciente como completado.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Asegura que podemos importar los módulos del paquete src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from m2m_client import M2MClient            # type: ignore
from api_client import APIClient            # type: ignore


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python3 scripts/repush_batch.py <match_id> [path_to_json]")
        return 2

    match_id = int(sys.argv[1])
    json_path = (
        Path(sys.argv[2]) if len(sys.argv) >= 3
        else ROOT / "output" / str(match_id) / "match_data.json"
    )

    if not json_path.exists():
        print(f"❌ JSON no encontrado: {json_path}")
        return 1

    api_base = os.environ.get("GEO_API_URL", "http://localhost:4000/api")
    client_id = os.environ.get("M2M_CLIENT_ID")
    client_secret = os.environ.get("M2M_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("❌ Falta M2M_CLIENT_ID o M2M_CLIENT_SECRET en el entorno")
        return 1

    m2m = M2MClient(api_base, client_id, client_secret)
    api = APIClient(api_base, m2m)

    with open(json_path) as f:
        payload = json.load(f)

    n = len(payload.get("frames", []))
    print(f"📤 Re-enviando {n} frames del match {match_id} desde {json_path}...")

    try:
        resp = api.push_tracking_batch(match_id, payload)
        print(f"✅ Push OK: {resp}")
    except Exception as e:
        print(f"❌ Push falló: {e}")
        return 1

    # Marcar el último job de este match como completed
    try:
        api.report_progress(
            match_id,
            "completed",
            progress=100,
            current_step="done",
            frames_processed=n,
            total_frames=n,
        )
        print(f"✅ Job marcado como completed para match {match_id}")
    except Exception as e:
        print(f"⚠️  Push OK pero no se pudo marcar el job como completed: {e}")
        print("    Hazlo manualmente:")
        print(f"    UPDATE match_analysis_jobs SET status='completed', progress=100 WHERE matchId={match_id};")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
