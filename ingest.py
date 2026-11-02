"""建索引：读取 data/ 下的文档 -> 切分 -> 向量化 -> 写入 index/

用法：
    python ingest.py                 # 自动使用 data/ 下第一个 .pdf/.txt/.md
    python ingest.py data/xxx.pdf    # 指定文件
"""
from __future__ import annotations

import sys
from pathlib import Path

from rag import INDEX_DIR, build_index, chunk_text

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUPPORTED = {".pdf", ".txt", ".md"}


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise SystemExit(f"不支持的文件类型：{suffix}（只支持 .pdf / .txt / .md）")


def pick_document() -> Path:
    candidates = sorted(
        p
        for p in DATA_DIR.glob("*")
        if p.suffix.lower() in SUPPORTED and not p.name.startswith(".")
    )
    if not candidates:
        raise SystemExit(f"{DATA_DIR} 下没有 .pdf/.txt/.md 文档，请先放一份进去。")
    return candidates[0]


def main() -> None:
    path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else pick_document()
    if not path.exists():
        raise SystemExit(f"文件不存在：{path}")

    print(f"读取：{path}")
    text = load_document(path)
    print(f"提取：{len(text)} 字")
    if len(text) < 300:
        raise SystemExit(
            "提取到的文字太少，可能是扫描版 PDF（整页是图片）。"
            "请换一份文本版 PDF，或先用 OCR 转成 txt。"
        )

    chunks = chunk_text(text)
    print(f"切分：{len(chunks)} 个片段（500 字 / 重叠 100 字）")

    print("向量化中，请稍候…")
    vecs = build_index(chunks, INDEX_DIR)
    print(f"完成：向量形状 {vecs.shape}，已写入 {INDEX_DIR}")
    print("下一步：streamlit run app.py")


if __name__ == "__main__":
    main()
