"""轻量 ContextAssembler。

职责：把 system prompt、可选 memory/RAG sections 与工作窗口消息压到模型预算
以内，并保证最后一条用户消息不被裁掉。v1 不做复杂 section 去重/压缩；
MemoryView/ContextSnapshot 的完整契约仍留在上下文文档的目标章节。
"""

from __future__ import annotations

from agent.llm import FilePart, ImagePart, LLMMessage, ThinkingPart, TextPart
from utils.tokens import estimate_tokens, truncate_to_tokens

from .models import ContextPlan, ContextSection

__all__ = ["ContextAssembler"]


class ContextAssembler:
    """按 token 预算组装 Prompt 消息。"""

    def __init__(
        self,
        *,
        max_context_tokens: int = 8192,
        reserved_output_tokens: int = 2048,
        max_sections: int = 32,
    ) -> None:
        """保存模型窗口、输出预留与 section 上限。"""
        self.max_context_tokens = max(256, int(max_context_tokens))
        self.reserved_output_tokens = max(0, int(reserved_output_tokens))
        self.max_sections = max(1, int(max_sections))

    def assemble(
        self,
        *,
        system_prompt: str | None,
        sections: list[ContextSection],
        messages: list[LLMMessage],
        tool_tokens: int = 0,
    ) -> ContextPlan:
        """生成带预算裁剪的 ContextPlan；最后一条消息始终保留。"""
        available = max(
            64,
            self.max_context_tokens
            - self.reserved_output_tokens
            - max(0, int(tool_tokens)),
        )
        dropped: list[str] = []
        truncated = False

        # 1) system prompt 是硬上下文；单独超预算时先截断，而不是挤掉用户问题。
        base = (system_prompt or "").strip()
        base_tokens = estimate_tokens(base)
        if base_tokens > available:
            base = truncate_to_tokens(base, available)
            base_tokens = estimate_tokens(base)
            truncated = True

        # 2) sections 按优先级从高到低装入；超预算的低优先级 section 直接丢弃。
        ordered = sorted(
            sections[: self.max_sections],
            key=lambda item: item.priority,
            reverse=True,
        )
        included: list[ContextSection] = []
        section_budget = max(0, available - base_tokens)
        for section in ordered:
            content = section.content.strip()
            if not content:
                continue
            tokens = section.token_estimate or estimate_tokens(content)
            if tokens > section_budget:
                # 单段超预算时截断后保留，避免 memory/RAG 完全消失。
                if section_budget <= 32:
                    dropped.append(section.name)
                    truncated = True
                    continue
                content = truncate_to_tokens(content, section_budget)
                tokens = estimate_tokens(content)
                truncated = True
            included.append(
                section.model_copy(
                    update={"content": content, "token_estimate": tokens}
                )
            )
            section_budget -= tokens
        if len(sections) > self.max_sections:
            truncated = True

        # 3) 工作窗口从最近消息往前装；最后一条用户消息必须保留。
        message_budget = max(0, available - base_tokens - sum(
            item.token_estimate for item in included
        ))
        kept_reversed: list[LLMMessage] = []
        for index, message in enumerate(reversed(messages)):
            tokens = _message_tokens(message)
            if index == 0 and tokens > message_budget:
                # 最后一条通常就是当前用户问题；宁可截断历史也不能丢它。
                kept_reversed.append(message)
                message_budget = 0
                truncated = True
                continue
            if tokens > message_budget:
                truncated = True
                continue
            kept_reversed.append(message)
            message_budget -= tokens
        kept = list(reversed(kept_reversed))

        system_text = _render_system(base, included)
        result_messages: list[LLMMessage] = []
        if system_text:
            result_messages.append(
                LLMMessage(role="system", content=[TextPart(text=system_text)])
            )
        result_messages.extend(kept)
        total = _messages_tokens(result_messages)
        return ContextPlan(
            messages=result_messages,
            sections=included,
            token_estimate=total,
            dropped_sections=dropped,
            truncated=truncated,
        )


def _message_tokens(message: LLMMessage) -> int:
    """估算一条 LLMMessage 的文本/tool 参数 token（忽略二进制附件）。"""
    total = 0
    for part in message.content:
        if isinstance(part, (TextPart, ThinkingPart)):
            total += estimate_tokens(part.text)
        elif isinstance(part, (ImagePart, FilePart)):
            total += 1
    for call in message.tool_calls:
        total += estimate_tokens(call.name)
        total += estimate_tokens(str(call.arguments or {}))
    return total


def _messages_tokens(messages: list[LLMMessage]) -> int:
    """估算整组消息 token。"""
    return sum(_message_tokens(item) for item in messages)


def _render_system(base: str, sections: list[ContextSection]) -> str:
    """把 system prompt 与 sections 渲染成带来源边界的单条 system 文本。"""
    parts: list[str] = []
    if base:
        parts.append(base)
    for section in sections:
        parts.append(
            f"[{section.source.upper()}:{section.name}]\n{section.content}".strip()
        )
    return "\n\n".join(parts).strip()
