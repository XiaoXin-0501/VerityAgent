"""垂直领域画像与领域抽取实现。

一个部署只加载一个 ``DomainProfile``；切换领域只替换 JSON 画像。
"""

from .extractor import (
    DomainGraphExtractor,
    canonicalize_relations,
    merge_keywords,
    merged_allowed_relations,
)
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
    "DomainGraphExtractor",
    "DomainMatch",
    "DomainProfile",
    "DomainPrompt",
    "DomainRelationType",
    "canonicalize_relations",
    "load_domain_profile",
    "merge_keywords",
    "merged_allowed_relations",
    "resolve_domain_profile_path",
]
