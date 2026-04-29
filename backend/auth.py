from __future__ import annotations

import hmac
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime

from fastapi import Cookie, HTTPException, Response, status

from config import settings


@dataclass
class AuthSession:
    token: str
    username: str
    created_at: datetime


class AuthManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, AuthSession] = {}

    def verify_credentials(self, username: str, password: str) -> bool:
        return hmac.compare_digest(username, settings.auth_username) and hmac.compare_digest(password, settings.auth_password)

    def create_session(self, username: str) -> AuthSession:
        session = AuthSession(
            token=secrets.token_urlsafe(32),
            username=username,
            created_at=datetime.utcnow(),
        )
        with self._lock:
            self._sessions[session.token] = session
        return session

    def get_session(self, token: str | None) -> AuthSession | None:
        if not token:
            return None
        with self._lock:
            return self._sessions.get(token)

    def clear_session(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)


auth_manager = AuthManager()


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 12,
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=settings.auth_cookie_name)


def require_auth(session_token: str | None = Cookie(default=None, alias=settings.auth_cookie_name)) -> str:
    session = auth_manager.get_session(session_token)
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return session.username
