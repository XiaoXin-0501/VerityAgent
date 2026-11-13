"""表格/代码/Excel 的语义描述生成。

表格和代码正文的语义密度低，直接整段向量化会稀释检索信号。本模块提供
可注入的描述器：

- 优先使用上下文里已有的“描述”（相邻正文、表格 caption/summary、代码 docstring）；
- 缺失时由 ChatModel 根据 heading + 表头/前几行，或 heading + 代码片段生成；
- 生成结果只进入 embedding_text，不写 chunk 正文。

默认链路不实例化任何模型；由 ``IndexerConfig.table_summary_llm_enabled`` /
``code_summary_enabled`` 控制，bootstrap 在注入 chat model 时才构造。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from utils.text import truncate

__all__ = [
    "ChatCodeDescriber",
    "ChatTableDescriber",
    "CodeDescriptionRequest",
    "CodeDescriber",
    "TableDescriptionRequest",
    "TableDescriber",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TableDescriptionRequest:
    """表格语义描述输入；行只放前几行，避免模型输入过大。"""

    heading: str = ""
    caption: str = ""
    filename: str = ""
    sheet: str = ""
    doc_type: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    context: str = ""


@dataclass(frozen=True, slots=True)
class CodeDescriptionRequest:
    """代码语义描述输入；``code`` 是截断后的代码片段。"""

    heading: str = ""
    file_path: str = ""
    symbol_path: str = ""
    signature: str = ""
    docstring: str = ""
    context: str = ""
    code: str = ""


class TableDescriber(Protocol):
    """表格/Excel 描述器协议。"""

    async def describe(
        self, request: TableDescriptionRequest
    ) -> str | None:
        """生成 1~2 句检索描述；失败返回 None，由调用方降级。"""
        ...


class CodeDescriber(Protocol):
    """代码描述器协议。"""

    async def describe(
        self, request: CodeDescriptionRequest
    ) -> str | None:
        """生成 1~2 句检索描述；失败返回 None，由调用方降级。"""
        ...


class ChatTableDescriber:
    """用 ChatModel + PromptRegistry 生成表格/Excel 描述。"""

    def __init__(
        self,
        chat_model: Any,
        *,
        model_id: str,
        prompts: PromptRegistry | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 160,
        max_chars: int = 200,
    ) -> None:
        """保存模型句柄与生成参数。"""
        self.chat_model = chat_model
        self.model_id = model_id
        self.prompts = prompts or get_default_prompt_registry()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_chars = max_chars

    async def describe(
        self, request: TableDescriptionRequest
    ) -> str | None:
        """调用模型生成描述；异常只记 warning，不阻断入库。"""
        try:
            response = await self.chat_model.complete(
                self._build_request(request)
            )
            text = _response_text(response).strip()
        except Exception as exc:  # noqa: BLE001 - LLM 失败要降级而非中断
            logger.warning("表格语义描述生成失败: %s", exc)
            return None
        return truncate(text, self.max_chars) if text else None

    def _build_request(self, request: TableDescriptionRequest):
        """组装 ChatModelRequest（局部导入 provider 类型）。"""
        from provider.enums import ModelRole, ResponseFormatType
        from provider.llm.types import (
            ChatModelRequest,
            ModelGenerationOptions,
            ModelMessage,
            ModelResponseFormat,
            ModelTextPart,
        )

        rows_text = "\n".join(
            " | ".join(str(cell) for cell in row)
            for row in request.rows
        )
        user_text = self.prompts.render_text(
            PromptId.RAG_TABLE_DESCRIPTION_USER,
            doc_type=request.doc_type,
            heading=request.heading,
            filename=request.filename,
            sheet=request.sheet,
            caption=request.caption,
            context=request.context,
            headers=" | ".join(str(item) for item in request.headers),
            rows=rows_text,
        )
        return ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[
                        ModelTextPart(
                            text=self.prompts.render_text(
                                PromptId.RAG_TABLE_DESCRIPTION_SYSTEM,
                                max_chars=self.max_chars,
                            )
                        )
                    ],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[ModelTextPart(text=user_text)],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=self.temperature,
                top_p=1.0,
                max_output_tokens=self.max_output_tokens,
            ),
            response_format=ModelResponseFormat(
                type=ResponseFormatType.TEXT
            ),
            stream=False,
        )


class ChatCodeDescriber:
    """用 ChatModel + PromptRegistry 生成代码描述。"""

    def __init__(
        self,
        chat_model: Any,
        *,
        model_id: str,
        prompts: PromptRegistry | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 160,
        max_chars: int = 200,
    ) -> None:
        """保存模型句柄与生成参数。"""
        self.chat_model = chat_model
        self.model_id = model_id
        self.prompts = prompts or get_default_prompt_registry()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_chars = max_chars

    async def describe(
        self, request: CodeDescriptionRequest
    ) -> str | None:
        """调用模型生成描述；异常只记 warning，不阻断入库。"""
        try:
            response = await self.chat_model.complete(
                self._build_request(request)
            )
            text = _response_text(response).strip()
        except Exception as exc:  # noqa: BLE001 - LLM 失败要降级而非中断
            logger.warning("代码语义描述生成失败: %s", exc)
            return None
        return truncate(text, self.max_chars) if text else None

    def _build_request(self, request: CodeDescriptionRequest):
        """组装 ChatModelRequest（局部导入 provider 类型）。"""
        from provider.enums import ModelRole, ResponseFormatType
        from provider.llm.types import (
            ChatModelRequest,
            ModelGenerationOptions,
            ModelMessage,
            ModelResponseFormat,
            ModelTextPart,
        )

        user_text = self.prompts.render_text(
            PromptId.RAG_CODE_DESCRIPTION_USER,
            heading=request.heading,
            file_path=request.file_path,
            symbol_path=request.symbol_path,
            signature=request.signature,
            docstring=request.docstring,
            context=request.context,
            code=request.code,
        )
        return ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[
                        ModelTextPart(
                            text=self.prompts.render_text(
                                PromptId.RAG_CODE_DESCRIPTION_SYSTEM,
                                max_chars=self.max_chars,
                            )
                        )
                    ],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[ModelTextPart(text=user_text)],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=self.temperature,
                top_p=1.0,
                max_output_tokens=self.max_output_tokens,
            ),
            response_format=ModelResponseFormat(
                type=ResponseFormatType.TEXT
            ),
            stream=False,
        )


def _response_text(response: Any) -> str:
    """从 ChatModelResponse 取文本 part；忽略 thinking/tool parts。"""
    from provider.llm.types import ModelTextPart

    return "".join(
        part.text
        for part in response.message.content
        if isinstance(part, ModelTextPart)
    )
