"""真实 PostgreSQL 登录集成测试。

测试对象是真实的 ``routers.auth`` 路由、真实的 ``users`` / ``user_permissions``
表与真实的连接池，不使用 mock。

覆盖流程：
1. 校验 ``init_schema`` 写入的默认用户 ``xiaoxin`` / ``xiaobai``（密码 123456）
   确实存在且都能登录；
2. 登录成功返回 JWT，错误密码 / 不存在用户返回 401；
3. 无 token / 伪造 token 访问 ``/auth/me`` 返回 401；
4. 携带 token 访问 ``/auth/me`` 返回用户与权限码；
5. ``require_permission`` 依赖放行已授权权限、拒绝未授权权限；
6. 登录成功后 ``users.last_login_at`` 被更新；
7. 回收测试临时授予的权限（默认用户本身保留）。

运行方式（需要可用的 PostgreSQL，连接参数来自 ``.env``）::

    uv run pytest tests/test_auth.py -v
    # 或直接运行
    uv run python tests/test_auth.py

测试会临时创建权限码 ``kb:search`` 并授予默认用户 ``xiaoxin``，结束时回收；
``kb:search`` 作为种子权限码保留，默认用户不会被删除。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 让测试既能被 pytest 收集，也能 `python tests/test_auth.py` 直接运行
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "verity-agent",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from exception import register_exception_handlers
from api.routers.auth import require_permission, verify_password
from api.routers.auth import router as auth_router
from config import settings
from store.pg import (
    PGConnectionManager,
    close_pg,
    init_pg,
    init_schema,
    set_pg,
)
from store.pg.models.user import User
from store.pg.repositories import PermissionRepository, UserRepository

TEST_USERNAME = "xiaoxin"
TEST_PASSWORD = "123456"
SEED_USERNAMES = ("xiaoxin", "xiaobai")
TEST_PERMISSION = "kb:search"
FORBIDDEN_PERMISSION = "kb:manage"

# 测试用 app：挂真实 auth 路由 + 两条受权限保护的验证路由
app = FastAPI(title="auth-integration-test")
register_exception_handlers(app)
app.include_router(auth_router)


@app.get("/test/protected")
async def protected(user: User = Depends(require_permission(TEST_PERMISSION))):
    return {"username": user.username}


@app.get("/test/forbidden")
async def forbidden(user: User = Depends(require_permission(FORBIDDEN_PERMISSION))):
    return {"username": user.username}


async def _verify_seed_users() -> User:
    """校验 init_schema 写入的默认用户存在且密码为 123456。"""
    users = UserRepository()
    found: dict[str, User] = {}
    for username in SEED_USERNAMES:
        user = await users.get_by_username(username)
        assert user is not None, f"init_schema 后未找到默认用户: {username}"
        assert verify_password(TEST_PASSWORD, user.password_hash), (
            f"默认用户 {username} 的密码不是 {TEST_PASSWORD}"
        )
        found[username] = user
        print(f"准备: 默认用户 {username} 存在且密码校验通过")
    return found[TEST_USERNAME]


async def _ensure_permission_code(code: str) -> None:
    permissions = PermissionRepository()
    if await permissions.get_by_code(code) is None:
        await permissions.create(code, "集成测试：知识库检索")


async def _cleanup(needs_revoke: bool) -> None:
    """测试只临时授予 kb:search；默认用户本身不删除。"""
    try:
        user = await UserRepository().get_by_username(TEST_USERNAME)
        if user is not None and needs_revoke:
            await PermissionRepository().revoke(user.id, TEST_PERMISSION)
            print(f"清理: 回收 {TEST_USERNAME} 的测试权限 {TEST_PERMISSION}")
    except Exception as exc:  # 清理失败不掩盖测试本身的失败
        print(f"清理失败: {exc!r}")


async def _run() -> None:
    needs_revoke = False
    try:
        # 集成测试用短超时，连不上时快速失败
        set_pg(
            PGConnectionManager(
                settings.pg.model_copy(
                    update={"pool_timeout": 10.0, "connect_timeout": 5}
                )
            )
        )
        try:
            await init_pg()
            await init_schema()
        except Exception as exc:
            raise AssertionError(
                "无法连接 PostgreSQL，请检查 .env 中的 "
                "PG__SERVER / PG__PORT / PG__USER / PG__PASSWORD / PG__DB；"
                "连接池日志会打印底层原因"
            ) from exc

        # 默认用户必须先由 init_schema 建好，测试不再自行创建
        user = await _verify_seed_users()
        previous_login_at = user.last_login_at

        permissions = PermissionRepository()
        await _ensure_permission_code(TEST_PERMISSION)
        had_permission = await permissions.has_permission(user.id, TEST_PERMISSION)
        needs_revoke = not had_permission
        await permissions.grant(user.id, TEST_PERMISSION)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 0) 两个默认用户都能用 123456 登录
            for username in SEED_USERNAMES:
                resp = await client.post(
                    "/auth/login",
                    json={"username": username, "password": TEST_PASSWORD},
                )
                assert resp.status_code == 200, resp.text
                assert resp.json()["user"]["username"] == username
            print("0. 默认用户 xiaoxin / xiaobai 登录成功")

            # 1) 正确密码登录
            resp = await client.post(
                "/auth/login",
                json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["token_type"] == "bearer"
            assert data["expires_in"] > 0
            assert data["user"]["username"] == TEST_USERNAME
            assert data["user"]["id"] == str(user.id)
            assert "password" not in resp.text
            token = data["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            print("1. 登录成功，token 长度:", len(token))

            # 2) 错误密码
            resp = await client.post(
                "/auth/login",
                json={"username": TEST_USERNAME, "password": "wrong-password"},
            )
            assert resp.status_code == 401, resp.text
            assert resp.json()["error"]["code"] == "UNAUTHENTICATED"
            assert resp.json()["error"]["message"] == "用户名或密码错误"
            print("2. 错误密码 -> 401")

            # 3) 不存在的用户（与错误密码同一提示）
            resp = await client.post(
                "/auth/login",
                json={"username": "not_exists_user", "password": TEST_PASSWORD},
            )
            assert resp.status_code == 401, resp.text
            assert resp.json()["error"]["code"] == "UNAUTHENTICATED"
            assert resp.json()["error"]["message"] == "用户名或密码错误"
            print("3. 不存在用户 -> 401")

            # 4) 缺少 token / 伪造 token
            resp = await client.get("/auth/me")
            assert resp.status_code == 401, resp.text
            resp = await client.get(
                "/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
            )
            assert resp.status_code == 401, resp.text
            print("4. 缺少/伪造 token -> 401")

            # 5) 携带 token 访问 /auth/me
            resp = await client.get("/auth/me", headers=headers)
            assert resp.status_code == 200, resp.text
            me = resp.json()
            assert me["user"]["id"] == str(user.id)
            assert TEST_PERMISSION in me["permissions"]
            print("5. /auth/me 通过，权限:", me["permissions"])

            # 6) require_permission 放行 / 拒绝
            resp = await client.get("/test/protected", headers=headers)
            assert resp.status_code == 200, resp.text
            assert resp.json()["username"] == TEST_USERNAME
            resp = await client.get("/test/forbidden", headers=headers)
            assert resp.status_code == 403, resp.text
            assert resp.json()["error"]["code"] == "PERMISSION_DENIED"
            assert "kb:manage" in resp.json()["error"]["message"]
            print("6. require_permission 放行已授权、拒绝未授权")

        # 7) last_login_at 已被更新
        refreshed = await UserRepository().get(user.id)
        assert refreshed is not None
        assert refreshed.last_login_at is not None
        if previous_login_at is not None:
            assert refreshed.last_login_at > previous_login_at
        print("7. last_login_at 已更新:", refreshed.last_login_at)

        print("\n真实登录测试全部通过")
    finally:
        await _cleanup(needs_revoke=needs_revoke)
        await close_pg()


def _new_event_loop():
    # Windows 上 psycopg 异步连接不支持 ProactorEventLoop
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


@pytest.mark.integration
def test_real_login_flow() -> None:
    """pytest 入口。"""
    asyncio.run(_run(), loop_factory=_new_event_loop)


if __name__ == "__main__":
    asyncio.run(_run(), loop_factory=_new_event_loop)
