"""分析 tests/result-data 下全部文档的切片质量。

用法：

```bash
# 先跑完 tests/parse_and_chunk.py，再执行：
uv run python tests/analyze_chunks.py

# 自定义目录/阈值
uv run python tests/analyze_chunks.py --result-dir tests/result-data
```

输出：

- `tests/result-data/chunk_analysis.md`：人类可读的分析报告；
- `tests/result-data/chunk_analysis.json`：机器可读的明细。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "tests" / "result-data"
_WHITESPACE = re.compile(r"\s+")
_BODY_PUNCT = re.compile(r"[。！；]")
_QUESTION_END = re.compile(r"[？?]\s*$")
_ZERO_WIDTH = "​‌‍﻿"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize(text: str) -> str:
    return _WHITESPACE.sub("", (text or "").strip(_ZERO_WIDTH))


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return ordered[index]


def _preview(value: str, limit: int = 80) -> str:
    text = (value or "").replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _analyze_document(
    doc: dict[str, Any],
    *,
    min_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    """统计单个文档的切片与标题情况。"""
    source = Path(doc["source"])
    out_dir = Path(doc["output_dir"]) if doc.get("output_dir") else None
    chunks = _read_jsonl(out_dir / "chunks.jsonl") if out_dir else []
    parse = _read_json(out_dir / "parse.json") if out_dir and (out_dir / "parse.json").is_file() else {}
    headings = [
        block
        for block in parse.get("blocks", [])
        if block.get("block_type") == "heading"
    ]
    title_text = next(
        (
            str(block.get("text") or "")
            for block in headings
            if int(block.get("heading_level") or 0) == 1
        ),
        None,
    )
    tokens = [int(chunk.get("token_count") or 0) for chunk in chunks]
    tiny_threshold = min_tokens
    tiny_half = max(1, min_tokens // 2)
    buckets = {
        "<20": 0,
        "20-40": 0,
        "40-100": 0,
        "100-200": 0,
        "200-300": 0,
        "300-450": 0,
        "450-600": 0,
        ">600": 0,
    }
    for token in tokens:
        if token < 20:
            buckets["<20"] += 1
        elif token < 40:
            buckets["20-40"] += 1
        elif token < 100:
            buckets["40-100"] += 1
        elif token < 200:
            buckets["100-200"] += 1
        elif token < 300:
            buckets["200-300"] += 1
        elif token < 450:
            buckets["300-450"] += 1
        elif token <= 600:
            buckets["450-600"] += 1
        else:
            buckets[">600"] += 1

    section_types = Counter(
        str(chunk.get("section_type") or "") for chunk in chunks
    )
    heading_levels = Counter(
        int(heading.get("heading_level") or 0) for heading in headings
    )
    # 可疑 heading：过长或带正文句末标点。
    suspicious_headings = []
    for heading in headings:
        text = str(heading.get("text") or "")
        # 文档标题、FAQ 问句本身可能是长句，不算异常。
        if title_text and text == title_text:
            continue
        if _QUESTION_END.search(text) and not _BODY_PUNCT.search(text):
            continue
        if _BODY_PUNCT.search(text) or len(text) > 100:
            suspicious_headings.append(heading)
    # 可疑 heading_path 叶子：过长/带句末标点/为空。
    suspicious_paths = []
    empty_paths = 0
    for chunk in chunks:
        path = chunk.get("heading_path") or []
        if not path:
            empty_paths += 1
            continue
        leaf = str(path[-1] or "")
        if title_text and leaf == title_text:
            continue
        if _QUESTION_END.search(leaf):
            continue
        if _BODY_PUNCT.search(leaf) or len(leaf) > 100:
            suspicious_paths.append(
                {
                    "chunk_index": chunk.get("chunk_index"),
                    "leaf": _preview(leaf, 120),
                }
            )

    # 重复正文：归一化后完全相同的 chunk 数。
    text_counter = Counter(
        _normalize(str(chunk.get("text") or "")) for chunk in chunks
    )
    duplicate_chunks = sum(
        count - 1 for text, count in text_counter.items() if text and count > 1
    )
    text_chunks: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        key = _normalize(str(chunk.get("text") or ""))
        text_chunks.setdefault(key, []).append(
            {
                "chunk_index": chunk.get("chunk_index"),
                "heading_path": " > ".join(chunk.get("heading_path") or []),
            }
        )
    duplicate_examples = [
        {
            "count": count,
            "text": _preview(text, 100),
            "chunks": text_chunks.get(text, [])[:3],
        }
        for text, count in text_counter.most_common(5)
        if text and count > 1
    ]

    tiny_examples = [
        {
            "chunk_index": chunk.get("chunk_index"),
            "tokens": chunk.get("token_count"),
            "heading_path": " > ".join(chunk.get("heading_path") or []),
            "text": _preview(str(chunk.get("text") or ""), 100),
        }
        for chunk in chunks
        if int(chunk.get("token_count") or 0) < tiny_half
    ][:5]

    return {
        "source": str(source),
        "output_dir": str(out_dir) if out_dir else None,
        "status": doc.get("status"),
        "blocks": int(doc.get("blocks") or 0),
        "chunks": len(chunks),
        "tokens": int(doc.get("tokens") or sum(tokens)),
        "headings": len(headings),
        "heading_levels": {str(key): value for key, value in sorted(heading_levels.items())},
        "size": {
            "min": min(tokens) if tokens else 0,
            "p10": _percentile(tokens, 0.1),
            "median": int(statistics.median(tokens)) if tokens else 0,
            "p90": _percentile(tokens, 0.9),
            "max": max(tokens) if tokens else 0,
            "mean": round(statistics.fmean(tokens), 1) if tokens else 0,
        },
        "tiny": {
            f"<{tiny_threshold}": sum(1 for token in tokens if token < tiny_threshold),
            f"<{tiny_half}": sum(1 for token in tokens if token < tiny_half),
            "examples": tiny_examples,
        },
        "over_max": sum(1 for token in tokens if token > max_tokens),
        "buckets": buckets,
        "section_types": dict(section_types),
        "suspicious_headings": [
            {
                "level": heading.get("heading_level"),
                "page": heading.get("page_no"),
                "text": _preview(str(heading.get("text") or ""), 140),
            }
            for heading in suspicious_headings[:10]
        ],
        "suspicious_heading_paths": suspicious_paths[:10],
        "empty_heading_paths": empty_paths,
        "duplicate_chunks": duplicate_chunks,
        "duplicate_examples": duplicate_examples,
    }


def _render_markdown(
    rows: list[dict[str, Any]],
    *,
    totals: dict[str, Any],
    config: dict[str, Any],
    min_tokens: int,
    max_tokens: int,
    generated_at: str,
) -> str:
    """生成 Markdown 报告。"""
    lines = [
        "# VerityAgent 切片质量分析",
        "",
        f"- 生成时间：{generated_at}",
        f"- 文档数：{totals['documents']}，成功：{totals['ok']}，失败：{totals['failed']}",
        f"- blocks：{totals['blocks']:,}，chunks：{totals['chunks']:,}，tokens：{totals['tokens']:,}",
        f"- 阈值：`min_chunk_tokens={min_tokens}`、`max_chunk_tokens={max_tokens}`",
        f"- 配置：`chunk_size={config.get('chunk_size_tokens')}`、"
        f"`max={config.get('max_chunk_tokens')}`、"
        f"`min={config.get('min_chunk_tokens')}`、"
        f"`quality_gate={config.get('quality_gate_enabled')}`、"
        f"`dedup={config.get('dedup_chunk_hash')}`",
        "",
        "## 1. 总览",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 文档 | {totals['documents']} |",
        f"| 失败 | {totals['failed']} |",
        f"| blocks | {totals['blocks']:,} |",
        f"| chunks | {totals['chunks']:,} |",
        f"| tokens | {totals['tokens']:,} |",
        f"| tiny<{min_tokens} | {totals['tiny']:,} "
        f"({totals['tiny'] / max(1, totals['chunks']):.1%}) |",
        f"| tiny<{max(1, min_tokens // 2)} | {totals['tiny_half']:,} "
        f"({totals['tiny_half'] / max(1, totals['chunks']):.1%}) |",
        f"| > {max_tokens} | {totals['over_max']:,} |",
        "",
        "### 尺寸分布",
        "",
        "| 区间 | chunks | 占比 |",
        "|---|---:|---:|",
    ]
    for name, count in totals["buckets"].items():
        lines.append(
            f"| {name} | {count:,} | {count / max(1, totals['chunks']):.1%} |"
        )
    lines.extend(
        [
            "",
            "### section_type 分布",
            "",
            "| section_type | chunks |",
            "|---|---:|",
        ]
    )
    for name, count in totals["section_types"].items():
        lines.append(f"| {name} | {count:,} |")

    lines.extend(
        [
            "",
            "## 2. 异常关注",
            "",
            f"### 2.1 极小 chunk（<{min_tokens}）比例最高",
            "",
            "| 文档 | chunks | tiny<40 | tiny<20 | 占比 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: item["tiny"][f"<{min_tokens}"]
        / max(1, item["chunks"]),
        reverse=True,
    )[:12]:
        if row["tiny"][f"<{min_tokens}"] == 0:
            break
        lines.append(
            f"| {_preview(Path(row['source']).name, 70)} | {row['chunks']} | "
            f"{row['tiny'][f'<{min_tokens}']} | {row['tiny'][f'<{max(1, min_tokens // 2)}']} | "
            f"{row['tiny'][f'<{min_tokens}'] / max(1, row['chunks']):.1%} |"
        )
    lines.extend(
        [
            "",
            f"### 2.2 超过 max_chunk_tokens（{max_tokens}）",
            "",
        ]
    )
    over_rows = [row for row in rows if row["over_max"]]
    if over_rows:
        lines.extend(
            [
                "| 文档 | chunks | >max | max token |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in over_rows:
            lines.append(
                f"| {_preview(Path(row['source']).name, 70)} | {row['chunks']} | "
                f"{row['over_max']} | {row['size']['max']} |"
            )
    else:
        lines.append("没有超过上限的 chunk。")

    lines.extend(
        [
            "",
            "### 2.3 可疑 heading（过长或含正文句末标点）",
            "",
        ]
    )
    suspicious = [row for row in rows if row["suspicious_headings"]]
    if suspicious:
        for row in suspicious[:15]:
            lines.append(f"**{_preview(Path(row['source']).name, 80)}**")
            for item in row["suspicious_headings"][:5]:
                lines.append(
                    f"- H{item['level']} p{item['page']}: {item['text']}"
                )
            lines.append("")
    else:
        lines.append("没有发现明显异常的 heading。")

    lines.extend(
        [
            "",
            "### 2.4 可疑 heading_path（过长或含正文标点）",
            "",
        ]
    )
    suspicious_paths = [row for row in rows if row["suspicious_heading_paths"]]
    if suspicious_paths:
        for row in suspicious_paths[:15]:
            lines.append(f"**{_preview(Path(row['source']).name, 80)}**")
            for item in row["suspicious_heading_paths"][:5]:
                lines.append(
                    f"- chunk {item['chunk_index']}: {item['leaf']}"
                )
            lines.append("")
    else:
        lines.append("没有发现明显异常的 heading_path。")

    lines.extend(
        [
            "",
            "### 2.5 重复正文与空 heading_path",
            "",
            "| 文档 | chunks | 重复 chunk | 空 heading_path |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: item["duplicate_chunks"], reverse=True)[:10]:
        if row["duplicate_chunks"] == 0 and row["empty_heading_paths"] == 0:
            continue
        lines.append(
            f"| {_preview(Path(row['source']).name, 70)} | {row['chunks']} | "
            f"{row['duplicate_chunks']} | {row['empty_heading_paths']} |"
        )

    lines.extend(
        [
            "",
            "## 3. 文档明细",
            "",
            "| 文档 | chunks | tokens | headings | min | p50 | p90 | max | tiny<40 | >600 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: item["chunks"], reverse=True):
        lines.append(
            f"| {_preview(Path(row['source']).name, 70)} | {row['chunks']} | "
            f"{row['tokens']:,} | {row['headings']} | {row['size']['min']} | "
            f"{row['size']['median']} | {row['size']['p90']} | {row['size']['max']} | "
            f"{row['tiny'][f'<{min_tokens}']} | {row['over_max']} |"
        )

    lines.extend(
        [
            "",
            "## 4. 结论与建议",
            "",
            "- `<20` token 的碎片应优先关注；`20-40` token 多为结构性短节"
            "（附录交叉引用、一级短节、front matter），是否合并取决于检索评估。",
            "- 超过 `max_chunk_tokens` 为硬错误，需要检查 chunk 合并/硬切逻辑。",
            "- 可疑 heading / heading_path 应抽查原文，确认是否把正文段落误判为标题。",
            "- 重复正文多来自多份重复文件或重复段落；确认是否需要在入库前去重。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="分析 tests/result-data 的切片质量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="parse_and_chunk.py 的输出目录",
    )
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    index_path = args.result_dir / "parse_chunk_index.json"
    if not index_path.is_file():
        print(f"缺少汇总文件：{index_path}")
        return 2
    index = _read_json(index_path)
    config = index.get("config") or {}
    min_tokens = args.min_tokens or int(config.get("min_chunk_tokens") or 40)
    max_tokens = args.max_tokens or int(config.get("max_chunk_tokens") or 600)

    rows = [
        _analyze_document(
            doc,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        for doc in index.get("documents", [])
    ]

    buckets = Counter()
    section_types: Counter[str] = Counter()
    for row in rows:
        buckets.update(row["buckets"])
        section_types.update(row["section_types"])
    totals = {
        "documents": len(rows),
        "ok": sum(1 for row in rows if row["status"] == "ok"),
        "failed": sum(1 for row in rows if row["status"] == "failed"),
        "blocks": sum(row["blocks"] for row in rows),
        "chunks": sum(row["chunks"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
        "tiny": sum(row["tiny"][f"<{min_tokens}"] for row in rows),
        "tiny_half": sum(
            row["tiny"][f"<{max(1, min_tokens // 2)}"] for row in rows
        ),
        "over_max": sum(row["over_max"] for row in rows),
        "buckets": dict(buckets),
        "section_types": dict(section_types.most_common()),
    }

    generated_at = datetime.now(UTC).isoformat()
    report = _render_markdown(
        rows,
        totals=totals,
        config=config,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        generated_at=generated_at,
    )
    (args.result_dir / "chunk_analysis.md").write_text(report, encoding="utf-8")
    (args.result_dir / "chunk_analysis.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "totals": totals,
                "documents": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "analysis done:",
        f"docs={totals['documents']}",
        f"chunks={totals['chunks']}",
        f"tiny<{min_tokens}={totals['tiny']}",
        f"over{max_tokens}={totals['over_max']}",
        "->",
        args.result_dir / "chunk_analysis.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
