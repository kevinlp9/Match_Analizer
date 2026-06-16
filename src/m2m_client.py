"""M2M client: obtain and cache client-credentials tokens."""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests


class M2MClient:
    def __init__(self, api_base: str, client_id: str, client_secret: str, cache_path: Optional[str] = None):
        self.api_base = api_base.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.cache_path = cache_path or os.environ.get("M2M_TOKEN_CACHE", ".m2m_token.json")
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def _load_cache(self) -> bool:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            token = data.get("accessToken")
            exp = float(data.get("expiresAt", 0))
            if token and exp > time.time() + 5:
                self._token = token
                self._expires_at = exp
                return True
        except Exception:
            return False
        return False

    def _save_cache(self, token: str, expires_in: int) -> None:
        payload = {"accessToken": token, "expiresAt": time.time() + int(expires_in)}
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass
        self._token = token
        self._expires_at = payload["expiresAt"]

    def get_token(self) -> str:
        if self._token and self._expires_at > time.time() + 5:
            return self._token
        if self._load_cache() and self._expires_at > time.time() + 5:
            return self._token  # type: ignore

        url = f"{self.api_base}/auth/oauth/token"
        body = {"grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret}
        r = requests.post(url, json=body, timeout=10)
        r.raise_for_status()
        data = r.json()
        token = data.get("accessToken") or data.get("access_token")
        expires_in = data.get("expiresIn") or data.get("expires_in") or 3600
        if not token:
            raise RuntimeError(f"token not present in response: {data}")
        self._save_cache(token, int(expires_in))
        return self._token  # type: ignore
