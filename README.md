# 文档问答助手（RAG Demo）

基于 **bge-m3 向量检索 + Qwen 生成** 的文档问答系统：上传一份文档，用自然语言提问，回答只依据文档内容，并标注引用片段编号。

技术栈刻意保持最小：`numpy` 手写向量检索 + `streamlit` 做界面，**不依赖 LangChain、不使用向量数据库**。目标是让你能讲清每一行，而不是跑通一个黑盒。

---

## 30 秒跑起来

```bash
# 1. 建环境（Python 3.11）
conda create -n rag python=3.11 -y
conda activate rag
pip install -r requirements.txt

# 2. 配置密钥
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# 然后编辑 .env，填入你的 API_KEY

# 3. 建索引（仓库里已放了一份示例文档 data/sample.txt）
python ingest.py

# 4. 起服务
streamlit run app.py
```

浏览器打开 <http://localhost:8501>，问一句「实习补贴每天多少钱？」就能看到效果。

> 想换成自己的文档：把 PDF/TXT 放进 `data/`，重新跑一次 `python ingest.py` 即可。
> 也可以指定文件：`python ingest.py data/你的文件.pdf`

---

## 目录结构

```
rag-demo/
├── rag.py              # 核心：切分、向量化、检索、生成
├── ingest.py           # 建索引（读文档 → 切分 → 向量化 → 落盘）
├── app.py              # Streamlit 界面
├── eval.py             # 检索评测，输出 hit@k 表格
├── data/sample.txt     # 示例文档（换成你自己的）
├── eval/questions.csv  # 评测集（问题 + 期望关键词）
├── index/              # 索引产物：chunks.json + vecs.npy
├── requirements.txt
└── .env.example
```

---

## 四个文件分别在干什么

| 文件 | 职责 | 关键函数 |
|---|---|---|
| `rag.py` | 全部核心逻辑 | `chunk_text` / `embed` / `retrieve` / `answer` |
| `ingest.py` | 离线建索引，只跑一次 | `load_document` / `build_index` |
| `app.py` | 网页界面，只负责交互 | — |
| `eval.py` | 离线评测检索质量 | `retrieve` |

`ingest.py` 和 `app.py` 都调用 `rag.py`，核心逻辑只有一份。

---

## 可调参数（面试官一定会问）

| 参数 | 默认值 | 调大会怎样 | 调小会怎样 |
|---|---|---|---|
| `size`（片段长度） | 500 字 | 上下文更完整，但检索精度下降、token 变贵 | 检索更准，但答案容易被切断 |
| `overlap`（重叠） | 100 字 | 边界信息保留更多，索引变大 | 省空间，但跨段的答案容易丢 |
| `k`（召回数） | 4 | 召回率上升，噪声和成本也上升 | 便宜，但可能漏掉答案 |
| `temperature` | 0.3 | 更发散、更爱编 | 更稳定、更死板 |

**改完 `size` / `overlap` 必须重新跑 `python ingest.py`**，否则索引和切分对不上。

---

## 评测：别用「感觉还行」验收

```bash
python eval.py        # 默认 k=4
python eval.py 1 4 8  # 一次试三个 k
```

输出示例（示意，实际数字以你跑出来的为准）：

```
| k | 命中率 | 未命中的问题 |
|---|---|---|
| 1 | 9/13（69%） | 实习报告要求多少字？；…… |
| 2 | 11/13（85%） | 实习报告要求多少字？；…… |
| 4 | 13/13（100%） | — |
```

> ⚠️ 示例文档只有 6 个片段，k=4 已经覆盖三分之二，命中率必然接近 100%，这是正常的。
> 换成 30 页以上的真实 PDF（通常 100+ 片段）后，这张表才有区分度——那才是你简历上值得写的那组数字。

判定方式：召回的前 k 个片段里，只要有一个包含期望关键词就算命中。这是最粗糙的评测，但足以暴露「切分不对」「embedding 选错」这类问题。

**负样本要手工测**：问一个文档里根本没有的问题（例如「可以申请住宿补贴吗？」），正确答案应该是「资料中未提及」。如果模型开始编，说明你的 system prompt 没压住。

---

## 部署到 Streamlit Community Cloud（免费）

1. 把代码推到 **公开** GitHub 仓库。`index/` 也要提交（否则线上没有索引）。
   **确认 `.env` 没有被提交**——`.gitignore` 已经挡了，但推之前 `git status` 看一眼。
2. 打开 <https://share.streamlit.io>，用 GitHub 登录 → New app → 选仓库和分支 → 主文件填 `app.py`。
3. 点 **Advanced settings**，在 Secrets 里填：

```toml
API_KEY = "sk-你的key"
BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

4. Deploy，两分钟后拿到 `https://xxx.streamlit.app`。

国内访问 `streamlit.app` 偶尔慢，备选：**魔搭 ModelScope 创空间**（免费、国内直连）或阿里云轻量服务器（2 核 2G 学生机）+ Docker。

---

## 面试怎么讲这个项目

三个必答题，答案都在代码里：

**Q：为什么不用向量数据库？**
> 文档只有几千个片段，向量矩阵是 3000×1024 的 float32，约 12 MB；一次 `vecs @ q` 是毫秒级。引入 Chroma 或 Milvus 只会增加部署复杂度和故障点。当片段超过十万级、或者需要按元数据过滤、需要持久化增量更新时，再换向量库才划算。

**Q：切分为什么是 500 字 / 100 重叠？**
> 500 字中文约 350 token，塞 4 段进 prompt 大约 1400 token，成本和上下文都合适。100 字重叠是为了防止答案正好落在切口上。另外代码里会在句号处收尾，避免把一句话劈开——代价是实际片段长度会在 300~500 之间浮动。

**Q：模型编造（幻觉）怎么处理？**
> 三层：一是 system prompt 强制「资料中没有就说未提及」；二是要求回答标注 `[1][2]`，用户可以点开核对原文；三是把 `temperature` 压到 0.3。这三层都是工程约束，不是根治——根本的解法是提高召回率，召回不到就一定会编。

再加一句「局限」，立刻显得你做过：

> 现在的缺陷是表格和图片里的信息提取不到，扫描版 PDF 直接失效；另外只支持单文档、没有多轮对话，用户追问「那第二条呢」会丢上下文。下一步是接版面解析（MinerU）和 query 改写。

---

## 下一步（按性价比排序）

1. **重排（rerank）**：召回 top-10，再用 `bge-reranker` 精排取 4。通常能明显提升精度，成本是多一次 API 调用。
2. **混合检索**：BM25 关键词召回 + 向量召回合并。专有名词、编号类问题提升明显。
3. **引用溯源可视化**：答案里的 `[1]` 做成可点击，跳到原文片段。
4. **多轮对话**：拼接历史 + query 改写（把「那第二条呢」改写成完整问题）。
5. **流式输出**：`stream=True`，首字延迟从 2s 降到 0.3s。

做完第 1 项，把「纯向量 vs 向量+rerank」两组数字放进 README 的对比表——**这张表就是这个项目最值钱的部分**。

---

## 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `提取 0 字` / 内容很少 | 扫描版 PDF，整页是图片 | 换文本版 PDF，或先 OCR |
| `404 model not found` | 模型名写错 | 去平台「模型广场」抄准确名字 |
| `401 Unauthorized` | key 错误或没加载 | 检查 `.env` 文件名、是否重启过终端 |
| `ModuleNotFoundError: rag` | 不在项目目录下运行 | `cd rag-demo` 再执行 |
| 线上报「索引不存在」 | `index/` 没提交 | `git add -f index/` 后重新部署 |
| 端口被占用 | 8501 已被占 | `streamlit run app.py --server.port 8502` |
| 中文输出乱码 | Windows 控制台默认 GBK | 先执行 `chcp 65001`，或设置环境变量 `PYTHONUTF8=1` |

---

## 安全提醒

`.env` 里的 key 一旦推到公开仓库，会被爬虫在几分钟内扫到并刷爆额度。

- 提交前 `git status` 确认 `.env` 不在列表里
- 万一泄露：立刻去平台**吊销重发**，不要只删 commit（历史里还在）
- 线上密钥只放 Streamlit Secrets，永远不要写进代码
