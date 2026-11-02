"""RAG 核心逻辑：切分 -> 向量化 -> 检索 -> 生成。

设计原则：不依赖 LangChain，也不引入向量数据库。
数据量小于 1 万片段时，一次 numpy 矩阵乘法就够快，
而且出问题时你能看懂每一行代码。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
BASE_URL = os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")

SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。规则：\n"
    "1. 只能依据用户提供的资料回答，不得使用资料之外的知识；\n"
    "2. 资料中没有的信息，直接回答「资料中未提及」，不要猜测；\n"
    "3. 回答后另起一行，用 [1][2] 标出所依据的片段编号；\n"
    "4. 用简体中文，简洁直接，不要复述问题。"
)

_client = None


def get_client():
    """延迟创建客户端：没有 API Key 时也能 import 本模块，方便测试。"""
    global _client
    if _client is None:
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少 API_KEY：请把 .env.example 复制为 .env 并填入你的 key；"
                "线上部署则在 Streamlit Secrets 中配置。"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=60,      # 不设的话默认 600 秒，网络不通时会静默卡十分钟
            max_retries=2,
        )
    return _client


def _create_with_retry(fn, attempts: int = 4):
    """免费模型常有速率限制（429），遇到就指数退避重试。"""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1 or "429" not in str(exc):
                raise
            time.sleep(2 ** i)


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """滑窗切分，尽量在句末收尾。

    size=500 字（中文约 350 token），overlap=100 字用来防止答案被边界切断。
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]

        # 窗口填满时，若后半段有句号，就在句号处收尾，避免把一句话劈开
        if end < len(text):
            cut = max(
                piece.rfind("。"),
                piece.rfind("？"),
                piece.rfind("！"),
                piece.rfind("；"),
            )
            if cut > size * 0.6:
                piece = piece[:cut + 1]

        if piece.strip():
            chunks.append(piece.strip())

        if end >= len(text):
            break

        # 关键：下一步按「实际产出长度」前进，而不是固定 size，
        # 否则句子截断之后会留下覆盖不到的缝隙。
        step = len(piece) - overlap
        start = start + step if step > 0 else end

    return chunks


def embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """文本 -> L2 归一化向量。归一化后，点积就等于余弦相似度。"""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else " " for t in texts[i:i + batch_size]]
        resp = _create_with_retry(
            lambda b=batch: client.embeddings.create(model=EMBED_MODEL, input=b)
        )
        vectors.extend(item.embedding for item in resp.data)

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build_index(chunks: list[str], index_dir: Path = INDEX_DIR) -> np.ndarray:
    """向量化全部片段并落盘，避免每次启动都重新调用 API。"""
    vecs = embed(chunks)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vecs.npy", vecs)
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    return vecs


def load_index(index_dir: Path = INDEX_DIR):
    """读回索引，返回 (chunks, vecs)。"""
    chunks_path = index_dir / "chunks.json"
    vecs_path = index_dir / "vecs.npy"
    if not chunks_path.exists() or not vecs_path.exists():
        raise FileNotFoundError(f"索引不存在：{index_dir}，请先运行 python ingest.py")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return chunks, np.load(vecs_path)


def retrieve(query: str, chunks: list[str], vecs: np.ndarray, k: int = 4):
    """返回 [(片段, 相似度), ...]，按相似度从高到低。"""
    if not chunks:
        return []
    k = min(k, len(chunks))
    q = embed([query])[0]
    scores = vecs @ q
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


def answer(query: str, chunks: list[str], vecs: np.ndarray, k: int = 4):
    """检索 + 生成，返回 (回答文本, 命中的片段列表)。"""
    hits = retrieve(query, chunks, vecs, k)
    if not hits:
        return "资料库为空，请先运行 python ingest.py 建立索引。", []

    context = "\n\n".join(f"[{i}] {c}" for i, (c, _) in enumerate(hits, 1))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{query}"},
    ]
    client = get_client()
    resp = _create_with_retry(
        lambda: client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
    )
    return resp.choices[0].message.content, hits
