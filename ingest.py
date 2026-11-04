"""建索引：读取 data/ 下的文档 -> 切分 -> 向量化 -> 写入 index/

用法：
    python ingest.py                 # 自动使用 data/ 下第一个 .pdf/.txt/.md
    python ingest.py data/xxx.pdf    # 指定文件
"""
from __future__ import annotations

import sys # 读取命令行的参数
from pathlib import Path

from rag import INDEX_DIR, build_index, chunk_text # 直接调用，需要改时只用改一处

ROOT = Path(__file__).resolve().parent # 保证拿到的是绝对路径
DATA_DIR = ROOT / "data"
SUPPORTED = {".pdf", ".txt", ".md"}


def read_pdf(path: Path) -> str:
    """ 局限：不保留表格结构，不处理扫描件，不保留标题层级"""
    from pypdf import PdfReader # 延迟导入，减少出错点
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages) #"\n"换行防止内容粘在一起，加or“”在纯图片时返回none,不会在join时崩溃


def load_document(path: Path) -> str:
    suffix = path.suffix.lower() #使扩展名能够兼容大写
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="gbk", errors="ignore")
       # return path.read_text(encoding="utf-8", errors="ignore") 改进；先用utf-8，失败再试gbk，解决了只用utf-8会丢字的问题
    raise SystemExit(f"不支持的文件类型：{suffix}（只支持 .pdf / .txt / .md）") # 抛出异常并退出程序


def pick_document() -> Path:
    """ 局限：更改文件时不能保证读取的是新的文件，因为是按字母排序，可能会读取到旧文件"""
    candidates = sorted(# Path 对象可以比较大小（按路径字符串），所以能直接排序。排序保证了结果稳定
        p
        for p in DATA_DIR.glob("*") # 列出data下的所有条目，不递归
        if p.suffix.lower() in SUPPORTED and not p.name.startswith(".")
    )
    if not candidates: # 排除非三种文件以及以点开头的垃圾文件
        raise SystemExit(f"{DATA_DIR} 下没有 .pdf/.txt/.md 文档，请先放一份进去。")
    return candidates[0]


def main() -> None:
    path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else pick_document() #把用户参数展开成用户主目录，没有参数运行pick_document()
    # if not path.exists(): 改进：exist 对文件夹路径也生效，改进后只对文件生效
    if not path.is_file():
        raise SystemExit(f"文件不存在：{path}")

    print(f"读取：{path}")
    text = load_document(path)
    print(f"提取：{len(text)} 字")
    if len(text) < 300:
        raise SystemExit(
            "提取到的文字太少，可能是扫描版 PDF（整页是图片）。"
            "请换一份文本版 PDF，或先用 OCR 转成 txt。"
        )

    chunks = chunk_text(text) #调用rag.py的切分函数
    print(f"切分：{len(chunks)} 个片段（500 字 / 重叠 100 字）") #动了默认值需要修改

    print("向量化中，请稍候…")
    vecs = build_index(chunks, INDEX_DIR)
    print(f"完成：向量形状 {vecs.shape}，已写入 {INDEX_DIR}")
    print("下一步：streamlit run app.py")


if __name__ == "__main__":
    main()
