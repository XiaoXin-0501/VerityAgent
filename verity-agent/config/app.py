"""应用、服务与认证配置。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr, field_validator

from .base import ConfigModel

__all__ = ["Environment", "AppConfig", "ServerConfig", "AuthConfig"]


class Environment(StrEnum):
    """运行环境；dev/test 允许自动建表/建 collection，prod 只校验。"""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


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

    @property
    def is_prod(self) -> bool:
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
    # 请求体大小上限（字节）；防止超大上传拖垮进程
    max_request_body_bytes: int = Field(default=25 * 1024 * 1024, gt=0)


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
        if value not in {"HS256", "HS384", "HS512"}:
            raise ValueError("仅支持 HMAC JWT 算法")
        return value

    @field_validator("secret_key")
    @classmethod
    def _validate_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("auth.secret_key 不能为空")
        return value
