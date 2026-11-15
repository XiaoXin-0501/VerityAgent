"""基础设施配置：PostgreSQL / Redis / MinIO / Weaviate / Neo4j。

所有字段都由 ``config.Settings`` 从环境变量映射；这里的默认值只是本地开发
占位，生产环境必须通过 env / secret 注入真实值。

环境变量写法：``PG__SERVER``、``MINIO__ENDPOINT``、``WEAVIATE__URL``、
``NEO4J__URI``；嵌套字段用双下划线，如 ``MINIO__BUCKETS__KB``。
"""

from __future__ import annotations

from pydantic import Field, SecretStr, field_validator

from .base import PROJECT_ROOT, ConfigModel

__all__ = [
    "PGConfig",
    "RedisConfig",
    "MinioBuckets",
    "MinioConfig",
    "WeaviateConfig",
    "Neo4jConfig",
]


class PGConfig(ConfigModel):
    """PostgreSQL 连接与连接池。"""

    # 数据库主机名 / IP
    server: str = "localhost"
    # 数据库端口
    port: int = Field(default=5432, ge=1, le=65535)
    # 登录用户名
    user: str = "agent"
    # 登录密码；SecretStr 防止日志/异常直接打印明文
    password: SecretStr = SecretStr("agent")
    # 数据库名；与 docker-compose 的 POSTGRES_DB 默认约定一致
    db: str = "agent_db"

    # 连接池最少保持的空闲连接数
    pool_min_size: int = Field(default=0, ge=0)
    # 连接池最大连接数
    pool_max_size: int = Field(default=10, ge=1)
    # 从连接池取连接的最长等待时间（秒）
    pool_timeout: float = Field(default=30.0, gt=0)
    # 单次建立 TCP 连接的超时时间（秒）
    connect_timeout: int = Field(default=10, gt=0)
    # 空闲连接最长保留时间（秒）；0 表示不限制
    pool_max_idle: float = Field(default=600.0, ge=0)
    # 单条连接最长存活时间（秒）；0 表示不限制
    pool_max_lifetime: float = Field(default=3600.0, ge=0)

    @property
    def url(self) -> str:
        """标准 ``postgresql://`` URL；psycopg 实际用 make_conninfo 拼参数。"""
        from urllib.parse import quote

        user = quote(self.user, safe="")
        password = quote(self.password.get_secret_value(), safe="")
        return f"postgresql://{user}:{password}@{self.server}:{self.port}/{self.db}"


class RedisConfig(ConfigModel):
    """Redis 连接与客户端连接池。"""

    # Redis URL；支持 redis://、rediss://、unix://，也兼容 host:port/db 简写
    url: str = "redis://localhost:6379/0"
    # Redis 6+ ACL 用户名；也可直接写在 URL 里
    username: str | None = None
    # Redis 密码；也可直接写在 URL 里；SecretStr 防止日志泄露
    # 兼容 docker-compose 的 REDIS_PASSWORD 变量名（未设置则保持空）
    password: SecretStr = SecretStr("")
    # 客户端连接池最大连接数
    max_connections: int = Field(default=20, ge=1)
    # 单条命令读写超时（秒）
    socket_timeout: float = Field(default=5.0, gt=0)
    # 建立连接超时（秒）
    socket_connect_timeout: float = Field(default=5.0, gt=0)
    # 连接健康检查间隔（秒）；0 表示关闭
    health_check_interval: int = Field(default=30, ge=0)
    # 是否把命令结果解码为 str；缓存 JSON 时保持 True
    decode_responses: bool = True
    # 逻辑 key 命名空间；所有 Redis key 统一为 <namespace>:<key_family>:...
    namespace: str = "verity"

    @field_validator("namespace")
    @classmethod
    def _normalize_namespace(cls, value: object) -> str:
        """命名空间不能为空、不能带首尾冒号，避免生成 `::cache` 这类 key。"""
        text = str(value or "").strip().strip(":")
        if not text:
            raise ValueError("redis namespace 不能为空")
        return text

    @field_validator("url", mode="before")
    @classmethod
    def _normalize_url(cls, value: object) -> str:
        """补全 redis:// scheme，兼容 host:port 写法。"""
        text = str(value).strip()
        if text and "://" not in text:
            # 兼容 ``host:port/db`` 写法，统一成 redis://
            return f"redis://{text}"
        return text

    @field_validator("password", mode="before")
    @classmethod
    def _fallback_compose_password(cls, value: object) -> object:
        """密码为空时回退读取 docker-compose 的 REDIS_PASSWORD。"""
        if value:
            return value
        import os

        # 兼容 docker-compose 中常见的 REDIS_PASSWORD 变量名
        return os.environ.get("REDIS_PASSWORD", "")


class MinioBuckets(ConfigModel):
    """MinIO 的两个 bucket 名（对应《MinIO 对象存储数据结构设计》）。"""

    # 知识库长期对象：source / parse / artifact 都在这个 bucket 下
    kb: str = "agent-kb"
    # 会话临时对象：source / parse / overflow / artifact；配 lifecycle/TTL
    session: str = "agent-session"


class MinioConfig(ConfigModel):
    """MinIO 对象存储配置。

    ``enabled=False`` 时应用仍可启动，上传/下载/删除接口应在业务层返回
    503；填写真实 endpoint / AK / SK 后再打开。
    """

    # 是否启用 MinIO 功能；占位阶段默认关闭，避免无服务时启动失败
    enabled: bool = False
    # MinIO API 地址，格式 host:port，不带 http:// 或 https://
    endpoint: str = "localhost:9000"
    # Access Key；建议通过 MINIO__ACCESS_KEY 环境变量注入
    access_key: str = ""
    # Secret Key；SecretStr 防止日志/异常泄露
    secret_key: SecretStr = SecretStr("")
    # 是否使用 HTTPS（true 时 endpoint 仍是 host:port）
    secure: bool = False
    # 区域；None 表示使用服务端默认区域
    region: str | None = None
    # 是否校验 TLS 证书；仅内网自签名证书时才建议关闭
    cert_check: bool = True
    # 建立连接超时（秒）
    connect_timeout_s: float = Field(default=10.0, gt=0)
    # 读取响应超时（秒）；大文件上传/下载可调大
    read_timeout_s: float = Field(default=60.0, gt=0)
    # 传输层最大重试次数（连接失败/5xx）
    max_retries: int = Field(default=3, ge=0, le=10)
    # 分片上传的分片大小（字节）；默认 8 MiB
    part_size: int = Field(default=8 * 1024 * 1024, gt=0)
    # 预签名 URL 有效期（秒）；仅用于响应期临时跳转，不入库
    presign_expiry_s: int = Field(default=300, ge=1)
    # dev/test 启动时是否自动创建缺失的 bucket；生产建议 False
    auto_create_buckets: bool = True
    # 启动时连接失败是否阻断应用；生产可设为 True
    required_for_startup: bool = False
    # 两个 bucket 名
    buckets: MinioBuckets = Field(default_factory=MinioBuckets)

    @property
    def is_configured(self) -> bool:
        """是否填入可用的连接信息（空 AK/SK 视为未配置）。"""
        return bool(self.endpoint and self.access_key and self.secret_key.get_secret_value())


class WeaviateConfig(ConfigModel):
    """Weaviate 向量库配置（D-97：import JSON 是唯一手写 schema 源）。

    ``auto_create_collections`` 只在 dev/test 打开；prod 只校验不自动改。
    """

    # 是否启用 Weaviate；占位阶段默认关闭
    enabled: bool = False
    # HTTP 地址；支持 http:// 或 https://，端口可从 URL 解析
    url: str = "http://localhost:8080"
    # gRPC 主机；为空时使用 HTTP URL 的 host
    grpc_host: str | None = None
    # gRPC 端口（Weaviate 4.x 默认 50051）
    grpc_port: int = Field(default=50051, ge=1, le=65535)
    # gRPC 是否使用 TLS
    grpc_secure: bool = False
    # API Key；Weaviate Cloud 或启用了认证的实例需要；SecretStr 防止泄露
    api_key: SecretStr = SecretStr("")
    # 普通查询/写入请求超时（秒）
    timeout_s: float = Field(default=30.0, gt=0)
    # 建立连接/初始化超时（秒）
    startup_timeout_s: float = Field(default=10.0, gt=0)
    # 是否跳过节流/连通性初始化检查（排障时可临时打开）
    skip_init_checks: bool = False
    # dev/test 是否自动创建缺失 collection；prod 必须 False
    auto_create_collections: bool = False
    # 期望 schema 版本；与 manifest/schema_import.json 对齐
    expected_schema_version: str = "2026-11-13"
    # 手写 schema 文件路径；为空时使用仓库 doc/weaviate_schema_import.json
    schema_import_path: str = str(PROJECT_ROOT / "doc" / "weaviate_schema_import.json")
    # 启动时是否执行 schema 校验（D-97）
    startup_schema_check: bool = True
    # DocChunk collection 名
    doc_chunk_collection: str = "DocChunk"
    # UserFact collection 名
    user_fact_collection: str = "UserFact"
    # 启动连接失败是否阻断应用
    required_for_startup: bool = False
    # 额外 HTTP header（非敏感）
    headers: dict[str, str] = Field(default_factory=dict)

    @property
    def http_parts(self) -> tuple[str, int, bool]:
        """从 ``url`` 解析 ``(host, port, secure)``，供 connect_to_custom 使用。"""
        from urllib.parse import urlsplit

        parts = urlsplit(self.url if "://" in self.url else f"http://{self.url}")
        secure = parts.scheme == "https"
        host = parts.hostname or "localhost"
        port = parts.port or (443 if secure else 8080)
        return host, port, secure

    @property
    def is_configured(self) -> bool:
        """MinIO 是否配置了地址。"""
        return bool(self.url)


class Neo4jConfig(ConfigModel):
    """Neo4j 图数据库配置。

    v1 入库始终建图；``graph_expansion_enabled`` 只控制检索扩展，不在本配置。
    """

    # 是否启用 Neo4j；占位阶段默认关闭
    enabled: bool = False
    # Bolt/Neo4j URI，如 bolt://localhost:7687 或 neo4j://host:7687
    uri: str = "bolt://localhost:7687"
    # 登录用户名
    user: str = "neo4j"
    # 登录密码；SecretStr 防止日志/异常泄露
    password: SecretStr = SecretStr("")
    # 目标数据库名（Neo4j 4+ 多库）
    database: str = "neo4j"
    # 驱动连接池最大连接数
    max_connection_pool_size: int = Field(default=50, ge=1)
    # 建立连接超时（秒）
    connection_timeout_s: float = Field(default=10.0, gt=0)
    # 事务级最大重试时间（秒）；驱动内部 transient error 重试
    max_transaction_retry_time_s: float = Field(default=30.0, gt=0)
    # 连接池中单条连接最长存活时间（秒）
    max_connection_lifetime_s: float = Field(default=3600.0, gt=0)
    # 启动时是否创建唯一约束/索引；语句幂等，建议保持 True
    init_schema_on_startup: bool = True
    # 启动连接失败是否阻断应用
    required_for_startup: bool = False

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        """校验 Neo4j URI 非空并补全 bolt:// scheme。"""
        text = value.strip()
        if not text:
            raise ValueError("Neo4j uri 不能为空")
        # 兼容 host:port 简写，统一补 bolt://
        if "://" not in text:
            return f"bolt://{text}"
        return text

    @property
    def is_configured(self) -> bool:
        """Neo4j 是否配置了地址与密码。"""
        return bool(self.uri and self.password.get_secret_value())
