"""Streamlit 界面：文档问答 Demo。

运行：
    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="文档问答 Demo", page_icon="📄", layout="centered")

# 线上密钥放在 st.secrets，本地放在 .env。
# 必须在 import rag 之前写入环境变量，rag 读取时才拿得到。
try:
    for _key in ("API_KEY", "BASE_URL", "EMBED_MODEL", "CHAT_MODEL"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

from rag import CHAT_MODEL, EMBED_MODEL, answer, load_index  # noqa: E402

st.title("📄 文档问答 Demo")
st.caption("RAG 流程：检索文档片段 → 交给大模型 → 只基于片段作答")

try:
    chunks, vecs = load_index()
except FileNotFoundError:
    st.error("还没有建立索引。请在项目目录下先运行：")
    st.code("python ingest.py", language="bash")
    st.stop()

with st.sidebar:
    st.subheader("参数")
    k = st.slider("检索片段数 k", 1, 10, 4)
    show_score = st.toggle("显示相似度", value=True)
    st.divider()
    st.caption(f"文档片段：{len(chunks)}")
    st.caption(f"向量维度：{vecs.shape[1]}")
    st.caption(f"Embedding：{EMBED_MODEL}")
    st.caption(f"对话模型：{CHAT_MODEL}")

query = st.text_input("输入问题", placeholder="例如：实习补贴每天多少钱？")

if st.button("提问", type="primary"):
    if not query.strip():
        st.warning("请先输入问题")
        st.stop()

    try:
        with st.spinner("检索并生成中…"):
            text, hits = answer(query, chunks, vecs, k=k)
    except Exception as exc:
        st.error(f"调用失败：{exc}")
        st.stop()

    st.markdown("### 回答")
    st.markdown(text)

    with st.expander(f"查看检索到的 {len(hits)} 个片段"):
        for i, (chunk, score) in enumerate(hits, 1):
            head = f"**片段 {i}**"
            if show_score:
                head += f"　相似度 `{score:.3f}`"
            st.markdown(head)
            st.text(chunk[:500])
else:
    st.caption("提示：先在终端运行 `python ingest.py` 建索引，再回到这里提问。")
