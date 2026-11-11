"""代码 parser（regex fallback；tree-sitter 接入前的可用实现）。"""

from __future__ import annotations

import re
from uuid import UUID

from rag.shared.enums import BlockType, CodeParseStatus, DocumentType, SymbolType
from rag.shared.models import (
    CodeBlock,
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
)

from ._common import decode_text, make_block, make_parsed_document

__all__ = ["CodeParser"]

_PYTHON_SYMBOL = re.compile(r"^(?P<indent>\s*)(?P<kind>class|def)\s+(?P<name>\w+)")
_JS_SYMBOL = re.compile(
    r"^(?P<indent>\s*)(?P<kind>class|function|const|let|var)\s+(?P<name>\w+)"
)
_JAVA_SYMBOL = re.compile(
    r"^(?P<indent>\s*)(?P<kind>class|interface|enum|public|private|protected)\s+"
)
_GO_SYMBOL = re.compile(r"^(?P<indent>\s*)func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)")

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


class CodeParser:
    parser_id = "code_regex"
    parser_version = "1"

    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
        extension = source.extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        language = _LANGUAGE_BY_EXT.get(extension)
        symbols = self._symbols(text, language)
        blocks = []
        if symbols:
            for order, symbol in enumerate(symbols):
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.CODE,
                        text=symbol["body"],
                        parser_id=self.parser_id,
                        code=CodeBlock(
                            language=language,
                            file_path=source.filename,
                            symbol_path=symbol["name"],
                            symbol_type=symbol["type"],
                            signature=symbol["signature"],
                            body=symbol["body"],
                            start_line=symbol["start_line"],
                            end_line=symbol["end_line"],
                            parse_status=CodeParseStatus.FALLBACK,
                        ),
                    )
                )
        else:
            blocks.append(
                make_block(
                    doc_id=doc_id,
                    page_no=0,
                    order=0,
                    block_type=BlockType.CODE,
                    text=text,
                    parser_id=self.parser_id,
                    code=CodeBlock(
                        language=language,
                        file_path=source.filename,
                        symbol_type=SymbolType.FILE,
                        body=text,
                        start_line=1,
                        end_line=max(1, len(text.splitlines())),
                        parse_status=CodeParseStatus.FALLBACK,
                    ),
                )
            )
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.CODE,
                    "has_text_layer": True,
                    "has_code": True,
                    "page_count": 1,
                    "text_coverage": 1.0,
                }
            ),
            blocks=blocks,
        )

    def _symbols(self, text: str, language: str | None) -> list[dict]:
        lines = text.splitlines()
        pattern = _PYTHON_SYMBOL if language == "python" else _GO_SYMBOL if language == "go" else _JS_SYMBOL
        starts: list[tuple[int, re.Match]] = []
        for index, line in enumerate(lines):
            match = pattern.match(line)
            if match:
                starts.append((index, match))
        symbols = []
        for position, (start, match) in enumerate(starts):
            end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
            kind = match.groupdict().get("kind", "")
            name = match.groupdict().get("name") or "<anonymous>"
            symbol_type = (
                SymbolType.CLASS
                if "class" in kind or "interface" in kind or "enum" in kind
                else SymbolType.FUNCTION
                if "func" in kind or "function" in kind or "def" in kind
                else SymbolType.BLOCK
            )
            body = "\n".join(lines[start:end]).strip()
            if not body:
                continue
            symbols.append(
                {
                    "name": name,
                    "type": symbol_type,
                    "signature": lines[start].strip(),
                    "body": body,
                    "start_line": start + 1,
                    "end_line": end,
                }
            )
        return symbols
