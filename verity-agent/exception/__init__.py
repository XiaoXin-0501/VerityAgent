"""统一异常出口。

用法::

    from exception import NotFoundError, PermissionDeniedError
"""

from .base import (
    AuthenticationError,
    ConflictError,
    ConfigurationError,
    InvalidStateTransitionError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
    VerityError,
)
from .handlers import register_exception_handlers, verity_error_handler
from .message import (
    MessageDecodeError,
    MessageEncodeError,
    MessageError,
    MessageValidationError,
    UnknownMessageRoleError,
    UnsupportedMessageVersionError,
)
from .pg import (
    MemoryFactLimitExceeded,
    OptimisticLockError,
    PGCheckViolationError,
    PGConnectionError,
    PGError,
    PGForeignKeyViolationError,
    PGIntegrityError,
    PGNotNullViolationError,
    PGQueryError,
    PGSerializationError,
    PGUniqueViolationError,
)
from .redis import (
    RedisCommandError,
    RedisConnectionError,
    RedisError,
    RedisKeyError,
)

__all__ = [
    # base
    "VerityError",
    "ConfigurationError",
    "ValidationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "InvalidStateTransitionError",
    # handlers
    "register_exception_handlers",
    "verity_error_handler",
    # message
    "MessageError",
    "MessageValidationError",
    "MessageEncodeError",
    "MessageDecodeError",
    "UnsupportedMessageVersionError",
    "UnknownMessageRoleError",
    # pg
    "PGError",
    "PGConnectionError",
    "PGQueryError",
    "PGIntegrityError",
    "PGUniqueViolationError",
    "PGForeignKeyViolationError",
    "PGNotNullViolationError",
    "PGCheckViolationError",
    "PGSerializationError",
    "OptimisticLockError",
    "MemoryFactLimitExceeded",
    # redis
    "RedisError",
    "RedisConnectionError",
    "RedisCommandError",
    "RedisKeyError",
]
