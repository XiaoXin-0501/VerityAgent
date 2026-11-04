# src/agentrag/settings.py
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 以 setting.py 所在位置为基准定位仓库根目录的 .env，
# 避免从 verity-agent/api 等子目录启动时读不到配置而回退到默认值。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


# —— 每一组配置一个类 ——

class PGConfig(BaseSettings):
    """PostgreSQL 连接与连接池配置"""
    server: str = "localhost"
    port: int = 5432
    user: str = "agent"
    password: str = "agent"
    db: str = "agent"

    # —— 连接池参数（对应 PG__POOL_* / PG__CONNECT_TIMEOUT）——
    pool_min_size: int = Field(default=0, ge=0)
    pool_max_size: int = Field(default=10, ge=1)
    # 从池里获取连接的最长等待时间（秒）
    pool_timeout: float = Field(default=30.0, gt=0)
    # 单次建立 TCP 连接的超时时间（秒）
    connect_timeout: int = Field(default=10, gt=0)
    # 空闲连接最长保留时间（秒），0 表示不限制
    pool_max_idle: float = Field(default=600.0, ge=0)
    # 单条连接最长存活时间（秒），0 表示不限制
    pool_max_lifetime: float = Field(default=3600.0, ge=0)

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.user}:{self.password}@{self.server}:{self.port}/{self.db}"
        )


class RedisConfig(BaseSettings):
    """Redis 连接配置（含 arq 队列）"""
    url: str = "redis://localhost:6379/0"


class AuthConfig(BaseModel):
    """登录 Token 配置（环境变量前缀 AUTH__）"""
    # 生产环境务必通过 AUTH__SECRET_KEY 覆盖
    secret_key: str = "verity-agent-dev-secret-change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 天


# —— 总入口：聚合所有子配置 ——

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # PG__SERVER → pg.server
        extra="ignore",  # .env 里多余的 key 不报错
    )

    pg: PGConfig = PGConfig()
    redis: RedisConfig = RedisConfig()
    auth: AuthConfig = AuthConfig()


# 全局单例，启动时读一次 .env
settings = Settings()
