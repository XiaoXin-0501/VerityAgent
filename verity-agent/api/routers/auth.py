"""用户登录与 Token 鉴权。

本模块只做最小可用版本：

- ``POST /auth/login``：校验用户名 + 密码，签发 JWT；
- ``GET  /auth/me``：用 Bearer token 取当前用户与权限码；
- ``get_current_user`` / ``require_permission``：供其他路由作为依赖使用。

前端拿到 token 后，请求时带上::

    Authorization: Bearer <access_token>

其他路由接入示例::

    from fastapi import Depends
    from routers.auth import get_current_user, require_permission
    from store.pg.models.user import User

    @router.get("/documents")
    async def list_documents(user: User = Depends(get_current_user)):
        ...

    @router.post("/kb/search")
    async def search(user: User = Depends(require_permission("kb:search"))):
        ...

密码哈希使用 argon2id；``users.password_hash`` 由注册/改密流程调用
:func:`hash_password` 生成。JWT 密钥与有效期见 ``setting.AuthConfig``
（环境变量 ``AUTH__SECRET_KEY`` / ``AUTH__ACCESS_TOKEN_EXPIRE_MINUTES``）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from setting import settings
from store.pg.models.enums import UserStatus
from store.pg.models.user import User
from store.pg.repositories import PermissionRepository, UserRepository

router = APIRouter(
    prefix="/auth",  # 该模块所有接口自动加统一前缀
    tags=["用户"],
)

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
    """生成 argon2id 密码哈希（注册 / 改密时调用）。"""
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """校验密码；哈希为空或格式非法一律视为不匹配。"""
    if not password_hash:
        return False
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


# ——————————————————— JWT ———————————————————

def create_access_token(user: User) -> tuple[str, int]:
    """签发访问 token，返回 ``(token, expires_in_seconds)``。"""
    auth = settings.auth
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=auth.access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role.value,
        "iat": now,
        "exp": expires_at,
    }
    token = jwt.encode(payload, auth.secret_key, algorithm=auth.algorithm)
    return token, auth.access_token_expire_minutes * 60


def decode_access_token(token: str) -> dict[str, Any]:
    """校验并解析 token；过期或非法时抛 401。"""
    try:
        return jwt.decode(
            token,
            settings.auth.secret_key,
            algorithms=[settings.auth.algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise _unauthorized("无效的登录凭证") from exc


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


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
    """从 ``Authorization: Bearer <token>`` 解析当前用户。

    每次请求都回库读取用户状态，保证停用账号立即失效；
    权限码不写进 token，敏感操作由 :func:`require_permission` 实时查库。
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("缺少 Bearer token")

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise _unauthorized("无效的登录凭证")
    try:
        user_uuid = UUID(str(user_id))
    except (TypeError, ValueError) as exc:
        raise _unauthorized("无效的登录凭证") from exc

    user = await UserRepository().get(user_uuid)
    if user is None:
        raise _unauthorized("用户不存在")
    if user.status is not UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"账号状态为 {user.status.value}，无法访问",
        )
    return user


def require_permission(permission_code: str):
    """生成权限校验依赖：当前用户必须真实拥有该权限码。"""

    async def checker(user: User = Depends(get_current_user)) -> User:
        allowed = await PermissionRepository().has_permission(user.id, permission_code)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少权限: {permission_code}",
            )
        return user

    return checker


# ——————————————————— 接口 ———————————————————

@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    """用户名 + 密码登录，成功返回 JWT。"""
    users = UserRepository()
    user = await users.get_by_username(payload.username)

    # 用户不存在与密码错误返回同一提示，避免暴露账号是否存在
    if user is None or not verify_password(payload.password, user.password_hash):
        raise _unauthorized("用户名或密码错误")

    if user.status is not UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"账号状态为 {user.status.value}，无法登录",
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
    """返回当前登录用户与权限码（前端可用于菜单/按钮级控制）。"""
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
