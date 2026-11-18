"""Redis 存储模块。

结构：

- ``connection.py``：异步客户端/连接池封装与生命周期；
- ``keys.py``      ：固定前缀枚举 + ``make_key`` 后缀拼接；
- ``cache.py``     ：CacheScope/CacheKey/CachePolicy 与 JSON 缓存封装。

启动示例::

    from store.redis import init_redis, close_redis, get_redis

    await init_redis()
    ...
    await close_redis()
"""

from .cache import (
    CacheEntry,
    CacheKey,
    CachePolicy,
    CachePort,
    CacheScope,
    CacheStats,
    RedisCache,
    make_cache_key,
)
from .connection import (
    RedisConnectionManager,
    close_redis,
    get_redis,
    init_redis,
    set_redis,
)
from .keys import RedisKeyPrefix, make_key

__all__ = [
    "RedisConnectionManager",
    "get_redis",
    "set_redis",
    "init_redis",
    "close_redis",
    "RedisKeyPrefix",
    "make_key",
    "CacheScope",
    "CachePort",
    "CacheKey",
    "CachePolicy",
    "CacheEntry",
    "CacheStats",
    "RedisCache",
    "make_cache_key",
]
