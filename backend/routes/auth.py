from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response

from backend.auth import auth_manager, clear_auth_cookie, require_auth, set_auth_cookie
from config import settings


router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
def login(payload: LoginRequest, response: Response) -> dict[str, object]:
    if not auth_manager.verify_credentials(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    session = auth_manager.create_session(payload.username)
    set_auth_cookie(response, session.token)
    return {"authenticated": True, "username": session.username}


@router.post("/auth/logout")
def logout(response: Response, session_token: str | None = Cookie(default=None, alias=settings.auth_cookie_name)) -> dict[str, bool]:
    auth_manager.clear_session(session_token)
    clear_auth_cookie(response)
    return {"authenticated": False}


@router.get("/auth/me")
def auth_me(username: str = Depends(require_auth)) -> dict[str, object]:
    return {"authenticated": True, "username": username}
