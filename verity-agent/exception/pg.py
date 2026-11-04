"""PostgreSQL 存储层异常。

存储层捕获 ``psycopg.Error`` 后统一翻译成这里的类型（见
``store/pg/repositories/base.py::translate_pg_error``），上层不需要 import psycopg。

SQLSTATE 对照见 PostgreSQL 官方文档 Appendix A：
- 23505 unique_violation
- 23503 foreign_key_violation
- 23502 not_null_violation
- 23514 check_violation
- 40001 serialization_failure
- 40P01 deadlock_detected
"""

from __future__ import annotations

from .base import ConflictError, VerityError


class PGError(VerityError):
    """PostgreSQL 相关异常的基类。"""

    code = "pg_error"
    http_status = 500
    default_message = "数据库操作失败"


class PGConnectionError(PGError):
    """连接不上数据库 / 连接已断开。"""

    code = "pg_connection_error"
    http_status = 503
    default_message = "数据库连接失败"


class PGQueryError(PGError):
    """SQL 执行失败（语法、表不存在等）。"""

    code = "pg_query_error"
    default_message = "数据库查询失败"


class PGIntegrityError(PGError):
    """违反数据库完整性约束。"""

    code = "pg_integrity_error"
    http_status = 409
    default_message = "数据库约束冲突"


class PGUniqueViolationError(PGIntegrityError):
    """唯一约束冲突（23505）。"""

    code = "pg_unique_violation"
    default_message = "唯一约束冲突"


class PGForeignKeyViolationError(PGIntegrityError):
    """外键约束冲突（23503）。"""

    code = "pg_foreign_key_violation"
    default_message = "外键约束冲突"


class PGNotNullViolationError(PGIntegrityError):
    """非空约束冲突（23502）。"""

    code = "pg_not_null_violation"
    http_status = 400
    default_message = "必填字段为空"


class PGCheckViolationError(PGIntegrityError):
    """CHECK 约束冲突（23514）。"""

    code = "pg_check_violation"
    http_status = 400
    default_message = "字段取值不满足约束"


class PGSerializationError(PGError):
    """事务串行化 / 死锁冲突（40001 / 40P01），调用方可以重试。"""

    code = "pg_serialization_failure"
    http_status = 409
    default_message = "并发事务冲突，请重试"


class OptimisticLockError(ConflictError):
    """乐观锁版本号不匹配，说明数据已被其他事务修改。"""

    code = "optimistic_lock_conflict"
    default_message = "数据已被并发修改，请重试"


class MemoryFactLimitExceeded(ConflictError):
    """用户长期事实条数达到硬上限（默认 150 条）。"""

    code = "memory_fact_limit_exceeded"
    default_message = "用户长期事实已达上限，需要先淘汰旧事实"

    def __init__(self, fact_count: int, limit: int = 150) -> None:
        super().__init__(
            f"用户长期事实已达上限 {limit} 条（当前 {fact_count}），需先淘汰部分事实",
            detail={"fact_count": fact_count, "limit": limit},
        )
