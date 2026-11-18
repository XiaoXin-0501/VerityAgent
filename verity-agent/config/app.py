"""应用、服务与认证配置。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr, field_validator, model_validator

from .base import ConfigModel

__all__ = [
    "Environment",
    "AppConfig",
    "ServerConfig",
    "AuthConfig",
    "LoggingConfig",
]


class Environment(StrEnum):
    """运行环境；dev/test 允许自动建表/建 collection，prod 只校验。"""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class LoggingConfig(ConfigModel):
    """本地文件日志配置。

    - 默认落盘到 Windows D 盘；路径可被 ``APP__LOG__DIRECTORY`` 覆盖；
    - 使用标准库 ``RotatingFileHandler``，按用途分文件轮转（app/access/error/rag/flow）；
    - 多进程部署时标准库 handler 不保证跨进程安全，生产 worker=1 或改用
      外部日志采集。
    """

    # 是否写本地文件；false 时只保留控制台
    to_file: bool = True
    # 日志目录；相对路径按 verity-agent 源码根目录解析
    directory: str = "D:/verityAgent日志"
    # 主应用日志文件名（不含 HTTP access / RAG 组件日志）
    file_name: str = Field(default="verity-agent.log", min_length=1)
    # HTTP access 日志；与主应用日志分离，避免轮询请求淹没业务日志
    access_file_name: str = Field(default="access.log", min_length=1)
    # WARNING 及以上日志；跨组件集中排查问题
    error_file_name: str = Field(default="error.log", min_length=1)
    # RAG 入库/检索组件日志；按 logger 前缀分流
    rag_file_name: str = Field(default="rag.log", min_length=1)
    # 单次请求的流程详情日志（JSONL）：完整记录 prompt、工具调用、耗时、
    # token、成功/失败等；与人类可读的 app 日志分开，便于机器分析。
    flow_file_name: str = Field(default="flow.jsonl", min_length=1)
    # 是否写流程详情日志
    flow_enabled: bool = True
    # 流程日志是否记录每次 LLM 请求的完整 prompt（含 system/历史/工具）
    flow_include_prompts: bool = True
    # 单次 prompt/消息在流程日志中的最大字符数；超出截断并标记
    flow_max_prompt_chars: int = Field(default=20000, ge=500)
    # 单个工具结果在流程日志中的最大字符数；超出截断并标记
    flow_max_result_chars: int = Field(default=4000, ge=100)
    # 单文件大小上限（字节）；超过后轮转
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    # 保留的历史文件数量
    backup_count: int = Field(default=10, ge=1)
    # 文件编码；Windows 中文路径/内容统一 UTF-8
    encoding: str = "utf-8"
    # 是否同时输出到控制台
    console: bool = True

    @model_validator(mode="after")
    def _unique_file_names(self) -> "LoggingConfig":
        """各类日志必须写到不同文件，避免重复 handler 互相覆盖/重复。"""
        names = [
            self.file_name,
            self.access_file_name,
            self.error_file_name,
            self.rag_file_name,
            self.flow_file_name,
        ]
        if len(set(names)) != len(names):
            raise ValueError("log 文件名不能重复")
        return self


class AppConfig(ConfigModel):
    """应用级元信息与运行模式。"""

    # 运行环境：dev / test / prod；控制 schema 自动创建等行为
    env: Environment = Environment.DEV
    # 应用名；用于 OpenAPI 标题、日志、健康检查
    name: str = "VerityAgent"
    # 应用版本；用于 OpenAPI 与健康检查
    version: str = "0.1.0"
    # 是否开启 FastAPI debug（生产必须 False）
    debug: bool = False
    # HTTP API 统一前缀，如 /api/v1
    api_prefix: str = "/api/v1"
    # 请求 id header 名；缺失时服务端生成
    request_id_header: str = "X-Request-ID"
    # 日志级别：DEBUG / INFO / WARNING / ERROR
    log_level: str = "INFO"
    # 本地文件日志目录/轮转/编码
    log: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """日志级别必须是标准 logging 级别。"""
        text = value.strip().upper()
        if text not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("app.log_level 必须是 DEBUG/INFO/WARNING/ERROR/CRITICAL")
        return text

    @property
    def is_prod(self) -> bool:
        """是否为生产环境。"""
        return self.env is Environment.PROD


class ServerConfig(ConfigModel):
    """ASGI 服务启动参数。"""

    # 监听地址；0.0.0.0 表示所有网卡
    host: str = "0.0.0.0"
    # 监听端口
    port: int = Field(default=8000, ge=1, le=65535)
    # 开发热重载；与 workers > 1 互斥
    reload: bool = False
    # 生产 worker 进程数；Windows 上建议 1
    workers: int = Field(default=1, ge=1)
    # 反向代理挂载路径；如部署在 /verity 下
    root_path: str = ""
    # 允许跨域的前端源列表；空表示不启用 CORS
    cors_origins: list[str] = Field(default_factory=list)
    # CORS 是否允许携带 Cookie/Authorization
    cors_allow_credentials: bool = True
    # 普通请求体大小上限（字节）；JSON/表单等小请求
    max_request_body_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    # multipart 上传请求体上限（字节）；知识库文档可能较大，
    # 与单文件业务上限（RAG__INDEXER__MAX_SOURCE_BYTES）分开控制
    max_multipart_body_bytes: int = Field(
        default=1024 * 1024 * 1024,
        gt=0,
    )


class AuthConfig(ConfigModel):
    """JWT 认证配置。"""

    # JWT 签名密钥；生产必须用 AUTH__SECRET_KEY 覆盖为随机长字符串
    secret_key: SecretStr = SecretStr("verity-agent-dev-secret-change-me")
    # JWT 算法；只支持 HS256 / HS384 / HS512
    algorithm: str = "HS256"
    # access token 有效期（分钟）；默认 7 天
    access_token_expire_minutes: int = Field(default=60 * 24 * 7, ge=1)
    # 可选 issuer；配置后签发与校验都会带 iss
    issuer: str | None = None
    # 可选 audience；配置后签发与校验都会带 aud
    audience: str | None = None

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, value: str) -> str:
        """只允许 HMAC 系列 JWT 算法。"""
        if value not in {"HS256", "HS384", "HS512"}:
            raise ValueError("仅支持 HMAC JWT 算法")
        return value

    @field_validator("secret_key")
    @classmethod
    def _validate_secret(cls, value: SecretStr) -> SecretStr:
        """JWT 密钥不能为空。"""
        if not value.get_secret_value().strip():
            raise ValueError("auth.secret_key 不能为空")
        return value
