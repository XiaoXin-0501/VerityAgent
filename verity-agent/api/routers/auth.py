"""用户登录与 Token 鉴权。

- ``POST /auth/login``：校验用户名 + 密码，签发 JWT；
- ``GET  /auth/me``：用 Bearer token 取当前用户与权限码；
- ``get_current_user`` / ``require_permission``：供其他路由作为依赖使用。

错误统一抛领域异常（``AuthenticationError`` / ``PermissionDeniedError``），
由全局异常处理器输出 ``ApiError`` 信封。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from config import settings
from exception import AuthenticationError, PermissionDeniedError
from store.pg.models.enums import UserStatus
from store.pg.models.user import User
from store.pg.repositories import PermissionRepository, UserRepository
from utils.time import utcnow

router = APIRouter(prefix="/auth", tags=["用户"])

_hasher = PasswordHasher()
_bearer = HTTPBearer(auto_error=False)


# ——————————————————— 请求 / 响应模型 ———————————————————

class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=128)


class UserInfo(BaseModel):
    """返回给前端的用户信息（不含密码哈希）。"""

    id: UUID
    username: str
    role: str
    email: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 秒
    user: UserInfo


class MeResponse(BaseModel):
    user: UserInfo
    permissions: list[str]


# ——————————————————— 密码哈希 ———————————————————

def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


# ——————————————————— JWT ———————————————————

def _auth_secret() -> str:
    return settings.auth.secret_key.get_secret_value()


def create_access_token(user: User) -> tuple[str, int]:
    now = utcnow()
    expires_in = settings.auth.access_token_expire_minutes * 60
    payload: dict[str, Any] = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role.value,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    if settings.auth.issuer:
        payload["iss"] = settings.auth.issuer
    if settings.auth.audience:
        payload["aud"] = settings.auth.audience
    token = jwt.encode(payload, _auth_secret(), algorithm=settings.auth.algorithm)
    return token, expires_in


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            _auth_secret(),
            algorithms=[settings.auth.algorithm],
            issuer=settings.auth.issuer,
            audience=settings.auth.audience,
            options={
                "require": ["exp", "sub"],
                "verify_iss": settings.auth.issuer is not None,
                "verify_aud": settings.auth.audience is not None,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("无效的登录凭证") from exc


def _user_info(user: User) -> UserInfo:
    return UserInfo(
        id=user.id,
        username=user.username,
        role=user.role.value,
        email=user.email,
    )


# ——————————————————— 鉴权依赖 ———————————————————

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("缺少 Bearer token")

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("无效的登录凭证")
    try:
        user_uuid = UUID(str(user_id))
    except (TypeError, ValueError) as exc:
        raise AuthenticationError("无效的登录凭证") from exc

    user = await UserRepository().get(user_uuid)
    if user is None:
        raise AuthenticationError("用户不存在")
    if user.status is not UserStatus.ACTIVE:
        raise PermissionDeniedError(
            f"账号状态为 {user.status.value}，无法访问",
            details={"status": user.status.value},
        )
    return user


def require_permission(permission_code: str):
    async def checker(user: User = Depends(get_current_user)) -> User:
        if not await PermissionRepository().has_permission(user.id, permission_code):
            raise PermissionDeniedError(
                f"缺少权限: {permission_code}",
                details={"permission_code": permission_code},
            )
        return user

    return checker


# ——————————————————— 接口 ———————————————————

@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    users = UserRepository()
    user = await users.get_by_username(payload.username)

    # 用户不存在与密码错误返回同一提示，避免暴露账号是否存在
    if user is None or not verify_password(payload.password, user.password_hash):
        raise AuthenticationError("用户名或密码错误")

    if user.status is not UserStatus.ACTIVE:
        raise PermissionDeniedError(
            f"账号状态为 {user.status.value}，无法登录",
            details={"status": user.status.value},
        )

    await users.mark_login(user.id)
    token, expires_in = create_access_token(user)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=_user_info(user),
    )


@router.get("/me", response_model=MeResponse)
async def me(user: User = Depends(get_current_user)) -> MeResponse:
    codes = await PermissionRepository().list_codes_for_user(user.id)
    return MeResponse(user=_user_info(user), permissions=sorted(codes))


__all__ = [
    "router",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "require_permission",
    "LoginRequest",
    "TokenResponse",
    "MeResponse",
    "UserInfo",
]
