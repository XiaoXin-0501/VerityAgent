"""Redis key 约定与连接器构造参数测试（不需要真实 Redis）。"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _ROOT / "verity-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from exception import ConfigurationError, RedisKeyError  # noqa: E402
from setting import RedisConfig  # noqa: E402
from store.redis import (  # noqa: E402
    RedisConnectionManager,
    RedisKeyPrefix,
    get_redis,
    make_key,
    set_redis,
)


def test_prefix_values_are_namespaced() -> None:
    for prefix in RedisKeyPrefix:
        assert prefix.value.startswith("verity:")
        assert not prefix.value.endswith(":")


def test_make_key_with_suffix() -> None:
    user_id = uuid4()
    assert make_key(RedisKeyPrefix.SESSION, "abc") == "verity:session:abc"
    assert make_key(RedisKeyPrefix.USER, user_id) == f"verity:user:{user_id}"
    assert make_key(RedisKeyPrefix.MESSAGE, 42) == "verity:message:42"


def test_make_key_without_suffix() -> None:
    assert make_key(RedisKeyPrefix.USER) == "verity:user"


def test_make_key_accepts_uuid_and_bytes() -> None:
    value = uuid4()
    assert make_key(RedisKeyPrefix.CONVERSATION, value) == f"verity:conversation:{value}"
    assert make_key(RedisKeyPrefix.CACHE, b"abc") == "verity:cache:abc"


def test_make_key_rejects_empty_suffix() -> None:
    with pytest.raises(RedisKeyError):
        make_key(RedisKeyPrefix.SESSION, "")
    with pytest.raises(RedisKeyError):
        make_key(RedisKeyPrefix.SESSION, "   ")


def test_make_key_rejects_none_suffix() -> None:
    with pytest.raises(RedisKeyError):
        make_key(RedisKeyPrefix.SESSION, None)


def test_make_key_rejects_invalid_prefix() -> None:
    with pytest.raises(RedisKeyError):
        make_key("verity:session")  # type: ignore[arg-type]


def test_manager_target_excludes_password() -> None:
    manager = RedisConnectionManager(
        RedisConfig(url="redis://:secret@localhost:6379/2")
    )
    stats = manager.get_stats()
    assert stats["target"] == "localhost:6379/2"
    assert "secret" not in stats["target"]


def test_manager_rejects_empty_url() -> None:
    with pytest.raises(ConfigurationError):
        RedisConnectionManager(RedisConfig(url=""))


def test_manager_validates_timeouts() -> None:
    with pytest.raises(ValueError):
        RedisConnectionManager(RedisConfig(), socket_timeout=0)
    with pytest.raises(ValueError):
        RedisConnectionManager(RedisConfig(), max_connections=0)


def test_singleton_set_and_get() -> None:
    manager = RedisConnectionManager(RedisConfig())
    try:
        set_redis(manager)
        assert get_redis() is manager
    finally:
        set_redis(None)
