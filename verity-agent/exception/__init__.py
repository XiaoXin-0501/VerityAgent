"""统一异常出口。

用法::

    from exception import NotFoundError, PermissionDeniedError
"""

from .base import (
    AuthenticationError,
    ConflictError,
    ConfigurationError,
    InternalError,
    InvalidStateTransitionError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitedError,
    UnavailableError,
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
from .llm import LLMError, LLMErrorCode
from .minio import (
    MinioConnectionError,
    MinioError,
    MinioObjectNotFoundError,
    MinioOperationError,
)
from .provider import ProviderError, ProviderErrorCode
from .neo4j import (
    Neo4jConnectionError,
    Neo4jError,
    Neo4jQueryError,
    Neo4jSchemaError,
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
from .weaviate import (
    WeaviateConnectionError,
    WeaviateError,
    WeaviateOperationError,
    WeaviateSchemaError,
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
    "RateLimitedError",
    "UnavailableError",
    "InternalError",
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
    # provider / llm
    "ProviderError",
    "ProviderErrorCode",
    "LLMError",
    "LLMErrorCode",
    # minio
    "MinioError",
    "MinioConnectionError",
    "MinioObjectNotFoundError",
    "MinioOperationError",
    # weaviate
    "WeaviateError",
    "WeaviateConnectionError",
    "WeaviateSchemaError",
    "WeaviateOperationError",
    # neo4j
    "Neo4jError",
    "Neo4jConnectionError",
    "Neo4jQueryError",
    "Neo4jSchemaError",
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
