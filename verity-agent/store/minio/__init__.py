"""MinIO 对象存储模块。

- ``connection.py``：客户端/连接池与生命周期；
- ``paths.py``：bucket key 约定；
- ``storage.py``：上传/下载/预签名/按前缀删除门面。
"""

from .connection import (
    MinioConnectionManager,
    close_minio,
    get_minio,
    init_minio,
    set_minio,
)
from .paths import (
    kb_artifact_key,
    kb_parse_prefix,
    kb_source_key,
    parse_prefix,
    sanitize_extension,
    session_artifact_key,
    session_overflow_key,
    session_parse_prefix,
    session_source_key,
    source_key,
)
from .storage import MinioStorage, ObjectInfo

__all__ = [
    "MinioConnectionManager",
    "get_minio",
    "set_minio",
    "init_minio",
    "close_minio",
    "MinioStorage",
    "ObjectInfo",
    "sanitize_extension",
    "source_key",
    "parse_prefix",
    "kb_source_key",
    "kb_parse_prefix",
    "kb_artifact_key",
    "session_source_key",
    "session_parse_prefix",
    "session_overflow_key",
    "session_artifact_key",
]
