"""HTML 表格 → TableBlock。

MinerU pipeline 的 ``table_body`` 是 HTML，可能包含 rowspan/colspan。
这里优先用 BeautifulSoup 展平合并单元格；未安装 bs4 时退回简单的
stdlib HTMLParser（不展开合并单元格，但至少保留行列文本）。
"""

from __future__ import annotations

from html.parser import HTMLParser

from rag.shared.models import TableBlock

from .._common import make_table

__all__ = ["html_table_to_table_block"]


def _normalize_rows(rows: list[list[str | None]]) -> list[list[str]]:
    """补齐列数并把 None 转成空串。"""
    width = max((len(row) for row in rows), default=0)
    return [
        [str(cell).strip() if cell is not None else "" for cell in row]
        + [""] * (width - len(row))
        for row in rows
    ]


def _parse_with_bs4(html: str) -> list[list[str]] | None:
    """用 BeautifulSoup 展开 rowspan/colspan。"""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return None

    rows: list[list[str | None]] = []
    for tr_index, tr in enumerate(table.find_all("tr")):
        while len(rows) <= tr_index:
            rows.append([])
        col = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while col < len(rows[tr_index]) and rows[tr_index][col] is not None:
                col += 1
            text = " ".join(cell.get_text(" ", strip=True).split())
            try:
                rowspan = max(1, int(str(cell.get("rowspan", "1"))))
            except (TypeError, ValueError):
                rowspan = 1
            try:
                colspan = max(1, int(str(cell.get("colspan", "1"))))
            except (TypeError, ValueError):
                colspan = 1
            for r in range(tr_index, tr_index + rowspan):
                while len(rows) <= r:
                    rows.append([])
                while len(rows[r]) < col + colspan:
                    rows[r].append(None)
                for c in range(col, col + colspan):
                    rows[r][c] = text
            col += colspan
    return _normalize_rows(rows)


class _SimpleTableParser(HTMLParser):
    """stdlib 兜底：只收集 tr/td/th 文本，不展开合并单元格。"""

    def __init__(self) -> None:
        """初始化行/单元格缓冲。"""
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        """遇到 tr 开新行，遇到 td/th 开新单元格。"""
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        """把文本写入当前单元格缓冲。"""
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        """闭合单元格时落文本，闭合行时落整行。"""
        if tag in {"td", "th"} and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None


def _parse_with_stdlib(html: str) -> list[list[str]]:
    """没有 bs4 时的简单解析。"""
    parser = _SimpleTableParser()
    parser.feed(html)
    parser.close()
    return parser.rows


def html_table_to_table_block(
    html: str,
    *,
    caption: str | None = None,
    confidence: float | None = None,
) -> TableBlock | None:
    """把 MinerU 的 HTML 表格转成 ``TableBlock``；失败返回 ``None``。"""
    if not html or "<table" not in html.lower():
        return None
    rows = _parse_with_bs4(html)
    if rows is None:
        rows = _parse_with_stdlib(html)
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return None
    headers = rows[0]
    data_rows = rows[1:]
    return make_table(
        headers,
        data_rows,
        caption=caption,
        confidence=confidence,
    )
