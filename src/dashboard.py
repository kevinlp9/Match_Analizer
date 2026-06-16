"""Dashboard routes for Geo-Goal AI Service — admin-only web UI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
API_BASE = os.environ.get("GEO_API_URL", "http://localhost:4000/api")

router = APIRouter()


# ── Simple template renderer (avoids jinja2 compatibility issues) ─────


def _render_template(name: str, **kwargs) -> HTMLResponse:
    """Read an HTML file from templates/ and inject kwargs with {{ key }} replacement."""
    path = TEMPLATES_DIR / name
    html = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        html = html.replace("{{ " + key + " }}", str(value))
        # Handle simple property access: {{ user.name }} → user dict
        if isinstance(value, dict):
            for sub in list(value.keys()):
                needle = "{{ " + key + "." + sub + " }}"
                html = html.replace(needle, str(value.get(sub, "")))
    return HTMLResponse(html)


# ── Helpers ──────────────────────────────────────────────────────────────


def _get_token(request: Request) -> str:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="No autenticado")
    return token


async def _verify_admin(request: Request) -> dict:
    """Verify the access token is valid and belongs to an admin user."""
    token = _get_token(request)
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{API_BASE}/auth/user",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=401, detail="Token inválido o expirado")
        except Exception:
            raise HTTPException(status_code=503, detail="No se puede conectar con el backend")

    user: dict = r.json()
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden acceder")

    return user


async def _call_backend(method: str, path: str, token: str, json_body: dict = None, timeout: int = 15) -> dict:
    """Make an authenticated call to the Geo-Goal backend."""
    async with httpx.AsyncClient() as client:
        r = await client.request(
            method,
            f"{API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            json=json_body,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()


# ── Auth helper for pages (redirects on failure) ────────────────────────


async def _require_admin_for_page(request: Request):
    """Return user dict if authenticated admin, else RedirectResponse to /login."""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{API_BASE}/auth/user",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            r.raise_for_status()
        except Exception:
            return RedirectResponse(url="/login")

    user: dict = r.json()
    if user.get("role") != "admin":
        return RedirectResponse(url="/login")

    return user


# ── Pages ────────────────────────────────────────────────────────────────


@router.get("/", response_class=RedirectResponse)
async def root_redirect():
    """Redirect root URL to the login page."""
    return RedirectResponse(url="/login")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page."""
    token = request.cookies.get("access_token")
    if token:
        # Check if already logged in as admin — if so, skip to dashboard
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(
                    f"{API_BASE}/auth/user",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                )
                r.raise_for_status()
                user = r.json()
                if user.get("role") == "admin":
                    return RedirectResponse(url="/dashboard")
            except Exception:
                pass
    return _render_template("login.html")


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the main dashboard page (admin only)."""
    result = await _require_admin_for_page(request)
    if isinstance(result, RedirectResponse):
        return result
    return _render_template("dashboard.html", user=result, userInitial=result.get("name", "A")[0].upper())


# ── Auth actions ─────────────────────────────────────────────────────────


@router.post("/login")
async def login_action(request: Request):
    """Authenticate user against Geo-Goal backend. Only allows admin users."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Cuerpo JSON requerido")

    email = body.get("email", "").strip()
    password = body.get("password", "")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email y contraseña son requeridos")

    # Validar que la URL del backend esté configurada
    if not API_BASE:
        print("[login] ❌ GEO_API_URL no está configurada en las env vars")
        raise HTTPException(
            status_code=503,
            detail="El servicio no está correctamente configurado (falta GEO_API_URL)",
        )

    print(f"[login] Intentando autenticar {email} contra {API_BASE}/auth/login")

    # 1. Get tokens from backend
    # Timeout amplio (45s) porque Render free tier puede tardar en despertar
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            r = await client.post(
                f"{API_BASE}/auth/login",
                json={"email": email, "password": password},
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"[login] ❌ Backend respondió {e.response.status_code}: {e.response.text[:200]}")
            if e.response.status_code == 401:
                raise HTTPException(status_code=401, detail="Credenciales inválidas")
            raise HTTPException(
                status_code=503,
                detail=f"Backend respondió con error {e.response.status_code}",
            )
        except httpx.TimeoutException:
            print(f"[login] ❌ Timeout al contactar {API_BASE}/auth/login")
            raise HTTPException(
                status_code=503,
                detail="El backend tardó demasiado en responder. Intenta de nuevo.",
            )
        except Exception as e:
            print(f"[login] ❌ Error inesperado: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"No se puede conectar con el backend: {type(e).__name__}",
            )

    tokens: dict = r.json()
    access_token = tokens.get("accessToken") or tokens.get("token")

    if not access_token:
        print(f"[login] ❌ Backend no devolvió token. Respuesta: {tokens}")
        raise HTTPException(status_code=500, detail="Token no recibido del backend")

    # 2. Verify user role
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.get(
                f"{API_BASE}/auth/user",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"[login] ❌ /auth/user respondió {e.response.status_code}")
            raise HTTPException(status_code=401, detail="No se pudo verificar el usuario")
        except Exception as e:
            print(f"[login] ❌ Error al verificar usuario: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"No se puede verificar el usuario: {type(e).__name__}",
            )

    user: dict = r.json()
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden acceder al dashboard de IA")

    response = JSONResponse({"ok": True, "user": user})
    response.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        max_age=86400,  # 24h
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
    )
    return response


@router.post("/logout")
async def logout_action():
    """Clear the access token cookie."""
    response = JSONResponse({"ok": True})
    response.delete_cookie("access_token")
    return response


# ── Dashboard data API ───────────────────────────────────────────────────


@router.get("/dashboard/api/health")
async def dashboard_health(request: Request):
    """Return local worker health status."""
    await _verify_admin(request)

    from state import worker as _w, POLL_INTERVAL, DEVICE

    return {
        "worker_running": _w.is_running if _w else False,
        "current_job": _w.current_job if _w else None,
        "poll_interval": POLL_INTERVAL,
        "device": DEVICE,
    }


@router.get("/dashboard/api/queue")
async def dashboard_queue(request: Request):
    """Return enriched job queue + live stats filtered by the admin's managed leagues."""
    token = _get_token(request)
    await _verify_admin(request)

    # 1. Get the admin's league IDs
    admin_league_ids: set[int] = set()
    try:
        dashboard_data = await _call_backend("GET", "/admin/dashboard", token, timeout=10)
        for league in dashboard_data.get("leagues", []):
            lid = league.get("id")
            if isinstance(lid, int):
                admin_league_ids.add(lid)
    except Exception:
        pass  # If we can't get leagues, fall through — no leagues = no jobs shown

    if not admin_league_ids:
        return {
            "jobs": [],
            "stats": {"queued": 0, "processing": 0, "failed": 0, "completed": 0},
            "message": "No gestionas ninguna liga",
        }

    # 2. Get aggregate counts per status (queued / processing / failed / etc.)
    stats = {"queued": 0, "processing": 0, "failed": 0, "annotating": 0, "uploaded": 0, "completed": 0}
    try:
        stats = await _call_backend("GET", "/admin/analysis/stats", token, timeout=8)
    except Exception:
        pass  # fall back to zeros

    # 3. Get the current queue list (queued + processing jobs so the admin sees active work)
    all_jobs: list[dict] = []
    try:
        # Pending (queued) jobs
        queued_jobs: list[dict] = await _call_backend("GET", "/public/matches/pending-analysis", token)
        all_jobs.extend(queued_jobs)
    except Exception:
        pass

    # Also fetch recent history to get processing + failed jobs visible in the list
    try:
        history_resp = await _call_backend("GET", "/admin/analysis/history", token, timeout=10)
        history_jobs: list[dict] = history_resp.get("jobs", [])
        # Include processing and failed jobs not already in the list
        existing_ids = {j.get("jobId") for j in all_jobs}
        for hj in history_jobs:
            if hj.get("status") in ("processing", "failed") and hj.get("jobId") not in existing_ids:
                all_jobs.append(hj)
                existing_ids.add(hj.get("jobId"))
    except Exception:
        pass

    # 4. Filter to admin's leagues only
    my_jobs = [j for j in all_jobs if j.get("leagueId") in admin_league_ids]

    # 5. Enrich with match name (batch, cached)
    enriched = []
    match_cache: dict[int, str | None] = {}

    for job in my_jobs:
        match_id = job.get("matchId")
        match_name: str | None = None

        if match_id and match_id not in match_cache:
            try:
                detail = await _call_backend("GET", f"/public/matches/{match_id}/detail", token, timeout=8)
                home = detail.get("match", {}).get("homeTeam", {}).get("name", "")
                away = detail.get("match", {}).get("awayTeam", {}).get("name", "")
                match_name = f"{home} vs {away}" if home or away else None
            except Exception:
                match_name = None
            match_cache[match_id] = match_name

        match_name = match_cache.get(match_id) if match_id else None
        enriched.append({**job, "matchName": match_name})

    return {"jobs": enriched, "stats": stats, "leagueIds": list(admin_league_ids)}


@router.post("/dashboard/api/poll")
async def dashboard_force_poll(request: Request):
    """Force the worker to poll for pending jobs (only from admin's leagues)."""
    token = _get_token(request)
    await _verify_admin(request)

    # Get admin's league IDs
    admin_league_ids: set[int] = set()
    try:
        dashboard_data = await _call_backend("GET", "/admin/dashboard", token, timeout=10)
        for league in dashboard_data.get("leagues", []):
            lid = league.get("id")
            if isinstance(lid, int):
                admin_league_ids.add(lid)
    except Exception:
        pass

    from state import worker as _w
    from api import get_api_client

    if not _w or not _w.is_running:
        raise HTTPException(status_code=503, detail="Worker no está corriendo")

    try:
        pending = get_api_client().get_pending_analysis()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar backend: {e}")

    # Filter to admin's leagues only
    if admin_league_ids:
        pending = [j for j in pending if j.get("leagueId") in admin_league_ids]

    if pending:
        import asyncio
        asyncio.create_task(_w.process_job(pending[0]))
        return {"message": f"Job encontrado y procesamiento iniciado — Job #{pending[0].get('jobId')}", "jobId": pending[0].get("jobId")}

    return {"message": "No hay trabajos pendientes en tus ligas"}


@router.get("/dashboard/api/history")
async def dashboard_history(request: Request):
    """Return analysis history for the admin's leagues."""
    token = _get_token(request)
    await _verify_admin(request)

    try:
        data = await _call_backend("GET", "/admin/analysis/history", token, timeout=12)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Error al consultar historial")
    except Exception:
        raise HTTPException(status_code=503, detail="Error al consultar historial")

    return data


@router.get("/dashboard/api/history/{job_id}")
async def dashboard_history_detail(request: Request, job_id: int):
    """Return analysis detail for a specific job in the admin's leagues."""
    token = _get_token(request)
    await _verify_admin(request)

    try:
        data = await _call_backend("GET", f"/admin/analysis/history/{job_id}", token, timeout=12)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Error al consultar detalle")
    except Exception:
        raise HTTPException(status_code=503, detail="Error al consultar detalle")

    return data
