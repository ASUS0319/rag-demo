"""检索评测：统计 hit@k，输出一张可以直接贴进 README 的表格。

用法：
    python eval.py            # 默认 k=4
    python eval.py 1 4 8      # 一次试多个 k

questions.csv 需要两列：question,keyword
判定方式：召回的前 k 个片段里，只要有一个包含 keyword 就算命中。
（这是最粗糙但最容易上手的评测；进阶版见 README「下一步」。）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from rag import load_index, retrieve

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "eval" / "questions.csv"


def main() -> None:
    ks = [int(x) for x in sys.argv[1:]] or [4]

    if not QUESTIONS.exists():
        raise SystemExit(f"缺少评测集：{QUESTIONS}")

    chunks, vecs = load_index()
    rows = list(csv.DictReader(QUESTIONS.open(encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("评测集是空的。")

    print(f"问题数：{len(rows)}　片段数：{len(chunks)}\n")
    print("| k | 命中率 | 未命中的问题 |")
    print("|---|---|---|")

    for k in ks:
        missed = []
        valid = 0
        for row in rows:
            question = (row.get("question") or "").strip()
            keyword = (row.get("keyword") or "").strip()
            if not question or not keyword:
                continue
            valid += 1
            hits = retrieve(question, chunks, vecs, k=k)
            if not any(keyword in chunk for chunk, _ in hits):
                missed.append(question)

        hit = valid - len(missed)
        rate = hit / valid if valid else 0.0
        miss_text = "；".join(missed) if missed else "—"
        print(f"| {k} | {hit}/{valid}（{rate:.0%}） | {miss_text} |")


if __name__ == "__main__":
    main()
