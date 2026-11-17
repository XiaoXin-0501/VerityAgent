"""垂直领域画像：模型、加载器与只读索引。

一个部署可加载多个画像（按知识库绑定），但图抽取器实例仍按画像复用。
"""

from .loader import (
    BUILTIN_DOMAIN_DIR,
    load_domain_profile,
    resolve_domain_profile_path,
)
from .models import (
    DomainEntity,
    DomainEntityType,
    DomainMatch,
    DomainProfile,
    DomainPrompt,
    DomainRelationType,
)

__all__ = [
    "BUILTIN_DOMAIN_DIR",
    "DomainEntity",
    "DomainEntityType",
    "DomainMatch",
    "DomainProfile",
    "DomainPrompt",
    "DomainRelationType",
    "load_domain_profile",
    "resolve_domain_profile_path",
]
