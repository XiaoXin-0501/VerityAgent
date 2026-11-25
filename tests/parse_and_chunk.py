"""把 tests/test-data 下的文档解析、切片，并导出为可检查的产物。

这是独立脚本（不是 pytest 用例），默认处理当前 parser 注册表支持的全部
格式：PDF / DOCX / Markdown / TXT / XLSX。

用法：

```bash
# 默认：解析 tests/test-data 下全部支持的文档
uv run python tests/parse_and_chunk.py

# 只处理指定文件/目录
uv run python tests/parse_and_chunk.py tests/test-data/某文档.pdf
uv run python tests/parse_and_chunk.py tests/test-data --limit 3

# 列出会处理哪些文件，不做解析
uv run python tests/parse_and_chunk.py --list

# 关闭图片 PDF 版式解析 / OCR（只看文本层）
uv run python tests/parse_and_chunk.py --image-engine none --ocr-provider none

# 关闭质量闸门/去重，观察原始切片
uv run python tests/parse_and_chunk.py --no-quality-gate --no-dedup

# 也可用环境变量指定输入
RAG_TEST_PATH=tests/test-data/某文档.md uv run python tests/parse_and_chunk.py
```

每个文档输出到 `tests/result-data/<样本名>/`：

```text
manifest.json   # 解析统计、profile、描述、warnings、耗时
parse.json      # ParsedDocument 完整投影
blocks.jsonl    # 每行一个 ContentBlock
chunks.jsonl    # 每行一个 ChunkDraft
chunks.md       # 人类可读切片
```

另外生成顶层汇总 `tests/result-data/parse_chunk_index.json`。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verity-agent"))

if sys.platform == "win32":
    # psycopg/OCR 依赖的异步循环在 Windows 上需要 Selector 策略
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import settings  # noqa: E402
from config.rag import IndexerConfig  # noqa: E402
from exception import RagError  # noqa: E402
from rag.indexer.text_pipeline import TextIngestionPipeline  # noqa: E402
from rag.shared.enums import DocumentType, SourceType  # noqa: E402
from rag.shared.models import ChunkDraft, DocumentSource, ParsedDocument  # noqa: E402

DEFAULT_INPUT = ROOT / "tests" / "test-data"
DEFAULT_OUTPUT = ROOT / "tests" / "result-data"
KB_ID = uuid5(NAMESPACE_URL, "verity-agent/parse-chunk-test-kb")
_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(slots=True)
class DocResult:
    """单个文档的处理结果，用于汇总。"""

    source: Path
    out_dir: Path | None
    status: str
    parser_id: str | None = None
    parser_version: str | None = None
    blocks: int = 0
    chunks: int = 0
    tokens: int = 0
    warnings: list[str] | None = None
    description: str | None = None
    duration_ms: int = 0
    error: str | None = None


def _safe_text(value: Any) -> str:
    """按控制台编码替换无法输出的字符，避免 Windows GBK 终端中断。"""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="replace").decode(
        encoding, errors="replace"
    )


def _safe_name(stem: str) -> str:
    """把文件名转成 Windows/Linux 都安全的目录名。"""
    name = _INVALID_NAME.sub("_", stem).strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    return (name[:160] or "document").strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, items: list[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(
                json.dumps(item, ensure_ascii=False, default=str) + "\n"
            )


def _build_config(args: argparse.Namespace) -> IndexerConfig:
    """在部署默认配置上按 CLI 参数覆盖解析/切片开关。"""
    config = settings.rag.indexer
    updates: dict[str, Any] = {}

    if args.image_engine:
        image_pdf = config.image_pdf.model_copy(
            update={"engine": args.image_engine}
        )
        if args.image_engine == "none":
            image_pdf = image_pdf.model_copy(update={"enabled": False})
        updates["image_pdf"] = image_pdf

    if args.ocr_provider:
        ocr = config.ocr.model_copy(update={"provider": args.ocr_provider})
        if args.ocr_provider == "none":
            ocr = ocr.model_copy(update={"enabled": False})
        updates["ocr"] = ocr

    if args.no_quality_gate:
        updates["quality_gate_enabled"] = False
    if args.no_dedup:
        updates["dedup"] = config.dedup.model_copy(
            update={"chunk_hash": False}
        )
    return config.model_copy(update=updates)


def _iter_candidates(paths: list[Path]) -> list[Path]:
    """展开目录，返回去重后的文件列表（保持稳定顺序）。"""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                item
                for item in sorted(path.rglob("*"))
                if item.is_file() and not item.name.startswith(".")
            )
        elif path.is_file():
            files.append(path)
        else:
            print(f"[skip] 路径不存在: {_safe_text(path)}")
    # 去重并保持顺序
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in files:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(item)
    return unique


def _detect(
    pipeline: TextIngestionPipeline, path: Path
) -> tuple[DocumentType, str, str]:
    """返回 (doc_type, extension, mime_type)。"""
    extension = path.suffix.lower()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    doc_type = pipeline.parsers.detect_doc_type(extension, mime_type)
    return doc_type, extension, mime_type


def _make_source(
    path: Path, data: bytes, extension: str, mime_type: str
) -> DocumentSource:
    """构造解析输入；doc_id 由相对路径 + 内容哈希确定，保证可复现。"""
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        relative = path.resolve().relative_to(DEFAULT_INPUT.resolve())
    except ValueError:
        relative = Path(path.name)
    return DocumentSource(
        object_key=relative.as_posix(),
        filename=path.name,
        extension=extension,
        mime_type=mime_type,
        size_bytes=len(data),
        sha256=sha256,
        source_type=SourceType.UPLOAD,
    )


def _doc_id(source: DocumentSource) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"verity-agent/parse-chunk/{source.object_key}:{source.sha256}",
    )


def _output_dir(out_root: Path, path: Path) -> Path:
    """按样本名建目录；不同源文件同名时才追加哈希后缀。

    同一个源文件重复运行时直接覆盖旧产物（内容变化也刷新），保证
    ``result-data/<样本名>/`` 始终是最新结果。
    """
    candidate = out_root / _safe_name(path.stem)
    manifest = candidate / "manifest.json"
    if manifest.is_file():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            previous_source = previous.get("source_path")
            if previous_source and (
                Path(previous_source).resolve() != path.resolve()
            ):
                digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
                candidate = out_root / f"{_safe_name(path.stem)}-{digest[:8]}"
        except (OSError, ValueError):
            pass
    candidate.mkdir(parents=True, exist_ok=True)
    stale_error = candidate / "error.txt"
    if stale_error.exists():
        stale_error.unlink()
    return candidate


def _render_chunks_md(
    parsed: ParsedDocument,
    chunks: list[ChunkDraft],
    source_name: str,
) -> str:
    """生成人类可读的切片预览。"""
    description = parsed.description.description if parsed.description else None
    lines = [
        f"# {source_name}",
        "",
        f"- parser: `{parsed.parser_id}` / `{parsed.parser_version}`",
        f"- doc_type: `{parsed.profile.doc_type.value}`",
        f"- pages: {len(parsed.pages)}",
        f"- blocks: {len(parsed.blocks)}",
        f"- chunks: {len(chunks)}",
        f"- tokens: {sum(item.token_count for item in chunks)}",
    ]
    if description:
        lines.append(f"- description: {description}")
    if parsed.warnings:
        lines.append(f"- warnings: {', '.join(parsed.warnings)}")
    lines.append("")
    for index, chunk in enumerate(chunks, 1):
        flags = ", ".join(sorted(flag.value for flag in chunk.quality_flags))
        lines.extend(
            [
                f"## {index}. {chunk.section_type.value} / "
                f"{chunk.block_type.value}",
                "",
                f"- chunk_id: `{chunk.chunk_id}`",
                f"- page: {chunk.page_no}",
                f"- tokens: {chunk.token_count}",
                f"- heading: {' > '.join(chunk.heading_path) or chunk.heading or '-'}",
            ]
        )
        if chunk.semantic_description:
            lines.append(f"- semantic_description: {chunk.semantic_description}")
        if flags:
            lines.append(f"- quality_flags: {flags}")
        lines.extend(["", "```text", chunk.text, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


async def _process_one(
    pipeline: TextIngestionPipeline,
    path: Path,
    out_root: Path,
) -> DocResult:
    """解析 + 切片一个文档并落盘。"""
    started = time.perf_counter()
    try:
        data = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        return DocResult(
            source=path,
            out_dir=None,
            status="failed",
            error=f"读取文件失败: {exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    doc_type, extension, mime_type = _detect(pipeline, path)
    if doc_type is DocumentType.UNKNOWN:
        return DocResult(
            source=path,
            out_dir=None,
            status="skipped",
            error=f"不支持的格式: {extension or path.name}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    source = _make_source(path, data, extension, mime_type)
    doc_id = _doc_id(source)
    out_dir = _output_dir(out_root, path)
    try:
        result = await pipeline.process(
            source,
            data,
            doc_id=doc_id,
            kb_id=KB_ID,
        )
    except RagError as exc:
        (out_dir / "error.txt").write_text(
            f"{exc.code}: {exc.message}\n{exc.details}\n",
            encoding="utf-8",
        )
        return DocResult(
            source=path,
            out_dir=out_dir,
            status="failed",
            error=f"{exc.code}: {exc.message}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 - 单文档失败不阻断整批
        (out_dir / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return DocResult(
            source=path,
            out_dir=out_dir,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    parsed = result.parsed
    chunks = result.chunks
    token_count = sum(item.token_count for item in chunks)
    description = (
        parsed.description.description if parsed.description is not None else None
    )

    manifest = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_path": str(path),
        "source_filename": path.name,
        "source_sha256": source.sha256,
        "size_bytes": source.size_bytes,
        "doc_id": str(doc_id),
        "kb_id": str(KB_ID),
        "parser_id": parsed.parser_id,
        "parser_version": parsed.parser_version,
        "profile": parsed.profile.model_dump(mode="json"),
        "page_count": len(parsed.pages),
        "block_count": len(parsed.blocks),
        "chunk_count": len(chunks),
        "token_count": token_count,
        "languages": parsed.profile.languages,
        "description": description,
        "description_source": (
            parsed.description.source if parsed.description is not None else None
        ),
        "warnings": result.warnings,
        "noise": result.noise.model_dump(mode="json"),
        "duration_ms": duration_ms,
        "files": [
            "parse.json",
            "blocks.jsonl",
            "chunks.jsonl",
            "chunks.md",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "parse.json", parsed.model_dump(mode="json"))
    _write_jsonl(
        out_dir / "blocks.jsonl",
        [item.model_dump(mode="json") for item in parsed.blocks],
    )
    _write_jsonl(
        out_dir / "chunks.jsonl",
        [item.model_dump(mode="json") for item in chunks],
    )
    (out_dir / "chunks.md").write_text(
        _render_chunks_md(parsed, chunks, path.name),
        encoding="utf-8",
    )

    return DocResult(
        source=path,
        out_dir=out_dir,
        status="ok",
        parser_id=parsed.parser_id,
        parser_version=parsed.parser_version,
        blocks=len(parsed.blocks),
        chunks=len(chunks),
        tokens=token_count,
        warnings=list(result.warnings),
        description=description,
        duration_ms=duration_ms,
    )


def _build_summary(
    results: list[DocResult],
    config: IndexerConfig,
    started_at: str,
) -> dict[str, Any]:
    ok = [item for item in results if item.status == "ok"]
    failed = [item for item in results if item.status == "failed"]
    skipped = [item for item in results if item.status == "skipped"]
    return {
        "started_at": started_at,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "chunk_size_tokens": config.chunk_size_tokens,
            "max_chunk_tokens": config.max_chunk_tokens,
            "min_chunk_tokens": config.min_chunk_tokens,
            "chunk_overlap_tokens": config.chunk_overlap_tokens,
            "chunking_version": config.chunking_version,
            "quality_gate_enabled": config.quality_gate_enabled,
            "dedup_chunk_hash": config.dedup.chunk_hash,
            "image_pdf_engine": config.image_pdf.engine,
            "image_pdf_enabled": config.image_pdf.enabled,
            "ocr_provider": config.ocr.provider,
            "ocr_enabled": config.ocr.enabled,
        },
        "totals": {
            "documents": len(results),
            "ok": len(ok),
            "failed": len(failed),
            "skipped": len(skipped),
            "blocks": sum(item.blocks for item in ok),
            "chunks": sum(item.chunks for item in ok),
            "tokens": sum(item.tokens for item in ok),
        },
        "documents": [
            {
                "source": str(item.source),
                "output_dir": (
                    str(item.out_dir) if item.out_dir is not None else None
                ),
                "status": item.status,
                "parser_id": item.parser_id,
                "parser_version": item.parser_version,
                "blocks": item.blocks,
                "chunks": item.chunks,
                "tokens": item.tokens,
                "warnings": item.warnings or [],
                "description": item.description,
                "duration_ms": item.duration_ms,
                "error": item.error,
            }
            for item in results
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_path = os.environ.get("RAG_TEST_PATH", "").strip()
    env_paths = [
        Path(item).expanduser()
        for item in env_path.split(os.pathsep)
        if item.strip()
    ]
    parser = argparse.ArgumentParser(
        description="解析并切片 tests/test-data，产物写入 tests/result-data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="输入文件或目录；默认 tests/test-data（或 RAG_TEST_PATH）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="产物根目录",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出会被处理的文件，不解析",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理前 N 个文件；0 表示全部",
    )
    parser.add_argument(
        "--image-engine",
        choices=["auto", "mineru", "rapid", "simple", "none"],
        default=None,
        help="覆盖图片/扫描 PDF 版式解析引擎",
    )
    parser.add_argument(
        "--ocr-provider",
        choices=["auto", "rapid", "paddle", "tesseract", "none"],
        default=None,
        help="覆盖 OCR provider",
    )
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="关闭质量闸门（保留低质量/短块）",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="关闭 chunk 哈希去重",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发处理的文档数；OCR/MinerU 建议保持 1",
    )
    args = parser.parse_args(argv)
    if not args.paths:
        args.paths = env_paths or [DEFAULT_INPUT]
    return args


async def _run(args: argparse.Namespace) -> int:
    config = _build_config(args)
    pipeline = TextIngestionPipeline(config)
    candidates = _iter_candidates(args.paths)

    supported: list[tuple[Path, DocumentType]] = []
    for path in candidates:
        doc_type, _ext, _mime = _detect(pipeline, path)
        if doc_type is DocumentType.UNKNOWN:
            continue
        supported.append((path, doc_type))
    if args.limit and args.limit > 0:
        supported = supported[: args.limit]

    if not supported:
        print("没有找到可解析的文档。")
        return 0

    if args.list:
        for path, doc_type in supported:
            print(
                f"{_safe_text(path)}  [{doc_type.value}]  "
                f"{path.stat().st_size / 1024:.1f} KiB"
            )
        print(f"共 {len(supported)} 个文件")
        return 0

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    total = len(supported)
    workers = max(1, args.workers)
    semaphore = asyncio.Semaphore(workers)
    print_lock = asyncio.Lock()

    async def run_one(index: int, path: Path, doc_type: DocumentType) -> DocResult:
        async with semaphore:
            result = await _process_one(pipeline, path, out_root)
        async with print_lock:
            status = result.status
            size = path.stat().st_size / 1024
            print(
                f"[{index}/{total}] {_safe_text(path.name)} "
                f"({size:.1f} KiB, {doc_type.value})"
            )
            if status == "ok":
                print(
                    f"  OK blocks={result.blocks} chunks={result.chunks} "
                    f"tokens={result.tokens} {result.duration_ms}ms "
                    f"parser={result.parser_id}"
                )
                if result.description:
                    print(f"  description: {_safe_text(result.description[:160])}")
                if result.warnings:
                    print(f"  warnings: {_safe_text(', '.join(result.warnings))}")
                print(f"  -> {_safe_text(result.out_dir)}")
            elif status == "skipped":
                print(f"  SKIP {_safe_text(result.error)}")
            else:
                print(f"  FAILED {_safe_text(result.error)}")
                if result.out_dir is not None:
                    print(f"  -> {_safe_text(result.out_dir)}/error.txt")
                else:
                    print("  （未能创建输出目录）")
            sys.stdout.flush()
        return result

    tasks = [
        run_one(index, path, doc_type)
        for index, (path, doc_type) in enumerate(supported, 1)
    ]
    results = await asyncio.gather(*tasks)

    summary = _build_summary(results, config, started_at)
    _write_json(out_root / "parse_chunk_index.json", summary)
    totals = summary["totals"]
    print(
        "\n完成: "
        f"documents={totals['documents']} ok={totals['ok']} "
        f"failed={totals['failed']} skipped={totals['skipped']} "
        f"blocks={totals['blocks']} chunks={totals['chunks']} "
        f"tokens={totals['tokens']}"
    )
    print(f"汇总: {_safe_text(out_root / 'parse_chunk_index.json')}")
    return 1 if totals["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
