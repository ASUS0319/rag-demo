# 文档问答助手（RAG）

基于向量检索的文档问答系统：把一份文档切片、向量化，用户提问时召回最相关的片段，交由大模型基于片段作答，并标注引用来源。

核心逻辑不使用 LangChain，也不引入向量数据库，检索部分用 NumPy 直接实现，全部代码控制在 200 行以内，便于阅读与修改。

## 效果

示例文档共 36 条校规，切分为 6 个片段。提问「实习补贴每天多少钱？」，系统返回「每人每天 120 元」，并标注引用片段 `[1]`，可展开核对原文。

<!-- 有截图后放到 docs/screenshot.png，取消下面这行注释 -->
<!-- ![界面截图](docs/screenshot.png) -->

## 功能

- 支持 PDF / TXT / Markdown 三种文档格式
- 检索 top-k 片段，回答附带 `[1][2]` 引用编号，可展开核对原文
- 文档中不存在的信息，模型明确拒答，不编造
- 提供检索质量评测脚本，输出 hit@k 指标
- 支持调整片段长度、重叠、召回数量等参数

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11+ |
| 向量化 | BAAI/bge-m3（1024 维） |
| 生成 | Qwen2.5-7B-Instruct |
| 检索 | NumPy 余弦相似度 |
| 界面 | Streamlit |
| 文档解析 | pypdf |

模型通过 OpenAI 兼容接口调用，可在 `.env` 中切换任意兼容平台。

## 架构

系统分为离线建索引和在线问答两个阶段。索引只需构建一次，之后的每次提问只处理问题本身。

```
离线（ingest.py，仅执行一次）
  文档 → 文本提取 → 切分 → 批量向量化 → 落盘 index/

在线（app.py，每次提问）
  问题 → 向量化 → 与索引矩阵做内积 → 取 top-k → 拼接 prompt → 生成回答
```

| 阶段 | 耗时 | 调用次数 |
|---|---|---|
| 建索引 | 10–30 秒 | 每片段一次（批量，每批 16 条） |
| 单次问答 | 2–5 秒 | 2 次（embedding + chat） |

## 快速开始

### 环境要求

- Python 3.11 及以上
- 无需 GPU，计算全部在模型 API 侧完成
- 一个可用的模型 API Key（支持 OpenAI 兼容接口）

### 安装

```bash
conda create -n rag python=3.11 -y
conda activate rag
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env        # Windows 使用 copy
```

编辑 `.env`：

```bash
API_KEY=sk-你的key
BASE_URL=https://api.siliconflow.cn/v1
EMBED_MODEL=BAAI/bge-m3
CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### 建索引

仓库内已附带示例文档 `data/sample.txt`，可直接运行：

```bash
python ingest.py
```

处理自己的文档：

```bash
python ingest.py data/your_file.pdf
```

### 启动

```bash
streamlit run app.py
```

浏览器访问 <http://localhost:8501>。

Windows 下若遇到 `.streamlit` 目录权限问题，可使用 `run.bat` 启动，脚本会将配置目录重定向到项目内。

## 项目结构

```
rag-demo/
├── rag.py              # 核心：切分、向量化、检索、生成
├── ingest.py           # 离线建索引
├── app.py              # Streamlit 界面
├── eval.py             # 检索评测，输出 hit@k
├── run.bat             # Windows 启动脚本
├── data/               # 原始文档
├── index/              # 索引产物：chunks.json + vecs.npy
├── eval/questions.csv  # 评测集
└── requirements.txt
```

| 模块 | 职责 | 关键函数 |
|---|---|---|
| `rag.py` | 全部核心逻辑 | `chunk_text` / `embed` / `retrieve` / `answer` |
| `ingest.py` | 文档解析与索引构建 | `load_document` / `build_index` |
| `app.py` | 交互界面 | — |
| `eval.py` | 检索质量评测 | `retrieve` |

`ingest.py`、`app.py`、`eval.py` 均复用 `rag.py`，核心逻辑只维护一份。

## 设计取舍

### 使用 NumPy 而非向量数据库

当前语料规模下，索引是一个 `n × 1024` 的 float32 矩阵。数千片段的场景下，一次 `vecs @ q` 为毫秒级，矩阵常驻内存仅数 MB。引入 Chroma、Milvus 会额外带来服务依赖、部署复杂度和故障点，收益不足以抵消成本。

切换条件：片段数达到十万级（矩阵约 400 MB）、需要按元数据过滤，或需要增量更新与持久化时，再引入向量数据库或近似最近邻索引。

### 切分策略

默认 500 字一段、重叠 100 字。中文 500 字约 350 token，取 4 段拼接后约 1400 token，在上下文长度与成本之间较为平衡。

切分在句末标点处收尾，避免把一句话截断；仅当标点位于窗口后 40% 时才截断，防止产生过短的片段。截断会导致实际片段长度在 300–500 之间浮动，因此下一段的起点按实际产出长度计算，而非固定的 `size - overlap`，否则句子截断后会在片段之间留下未被覆盖的文字。

### 向量归一化与打分

向量在写入索引前做 L2 归一化，模长为 1。此时两个向量的内积直接等于余弦相似度，检索阶段无需再做除法。

需要说明的是：归一化之后，余弦相似度与欧氏距离的排序结果完全等价（`‖a−b‖² = 2 − 2a·b`），选择余弦只是为了减少一次开方运算，并非精度更高。真正影响效果的是归一化本身。

### 幻觉控制

三层约束：

1. system prompt 限定只能依据给定资料作答，无相关信息时回答「资料中未提及」
2. 要求回答标注引用编号，用户可展开核对原始片段
3. `temperature` 设为 0.3，降低发散

这三层都是工程层面的约束，无法根治。根本手段是提高召回率——召回不到相关内容时，模型必然编造。

### 索引预计算

建索引与问答分离，是为了把不变的计算提前完成。若每次提问都重新向量化全文，单次响应会从 2 秒上升到 30 秒以上，API 调用量也成倍增加。

## 评测

检索质量用 hit@k 衡量：对每个测试问题，召回的前 k 个片段中只要有一个包含期望关键词，即记为命中。

```bash
python eval.py          # 默认 k=4
python eval.py 1 2 4 8  # 一次评估多个 k
```

输出为 Markdown 表格，可直接粘贴到下方：

<!-- 运行 eval.py 后，把真实结果粘贴到这里，并注明语料规模 -->

| k | 命中率 | 未命中的问题 |
|---|---|---|
| 1 | — | — |
| 4 | — | — |

评测集位于 `eval/questions.csv`，包含 `question` 与 `keyword` 两列。当前为 13 条基于示例文档的问题，建议扩展到 30 条以上，并覆盖两类问题：

- 文档中确实存在答案的（正样本）
- 文档中不存在答案的（负样本，期望模型拒答）

示例文档 `data/sample.txt` 仅约 2000 字、切分为 6 个片段，`k=4` 时召回已覆盖三分之二，指标接近饱和，仅可用于验证流程是否跑通。要得到有区分度的数字，需接入 30 页以上的真实文档（通常切分出 100 个以上片段）。

## 部署

以 Streamlit Community Cloud 为例：

1. 代码推送到公开 GitHub 仓库，`index/` 一并提交（线上文件系统只读，无法运行 `ingest.py`）
2. 在 <https://share.streamlit.io> 新建应用，选择仓库与分支，主文件填 `app.py`
3. 在 Advanced settings → Secrets 中写入：

```toml
API_KEY = "sk-你的key"
BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

4. 部署完成后获得公网地址

Secrets 使用 TOML 格式，与环境变量同名；应用启动时会读取并写入 `os.environ`。本地开发使用 `.env`，两者互不冲突。

`streamlit.app` 在国内访问可能不稳定，可替换为魔搭 ModelScope 创空间，或自建服务器 + Nginx 反向代理，代码无需改动。

## 路线图

1. **重排（rerank）**：召回 top-10 后用 `bge-reranker` 精排取 4，提升精度，代价是多一次 API 调用
2. **混合检索**：BM25 关键词召回与向量召回融合，改善编号、专有名词类问题的召回
3. **相似度阈值**：低于阈值时直接返回「未找到相关内容」，避免强行作答
4. **引用可视化**：答案中的 `[1]` 可点击跳转到对应原文片段
5. **多轮对话**：引入 query 改写，处理「那第二条呢」这类指代
6. **流式输出**：首字延迟从约 2 秒降至 0.3 秒

## 已知限制

- 表格与图片中的信息无法提取，扫描版 PDF 完全失效（需接入 OCR 或版面解析）
- 单文档、无多轮上下文
- 检索基于纯语义相似度，对精确编号类查询不敏感
- 索引与文档绑定，换文档需重新构建并重新部署

## 排错

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: rag` | 未在项目目录下运行 | `cd` 到项目目录后重试 |
| `缺少 API_KEY` | 未创建 `.env` 或未填写 | 从 `.env.example` 复制并填写 |
| `401 Unauthorized` | Key 错误，或文件被存为 `.env.txt` | 检查文件名与 Key |
| `404 model not found` | 模型名或 `BASE_URL` 错误 | 按平台文档核对模型 ID |
| `429 Too Many Requests` | 触发免费额度限流 | 已内置指数退避重试，仍失败则降低频率 |
| 提取字符数极少 | 扫描版 PDF，页面为图片 | 换文本版 PDF 或先做 OCR |
| 页面提示「索引不存在」 | 未运行 `ingest.py` | 先构建索引 |
| 线上提示「索引不存在」 | `index/` 未提交 | `git add index/` 后重新部署 |
| `Permission denied: .streamlit` | 用户主目录不可写 | 使用 `run.bat` 启动 |
| 端口被占用 | 8501 已被占用 | `--server.port 8502` |

## 密钥管理

- `.env` 已列入 `.gitignore`，不会进入版本库
- 提交前用 `git status` 确认 `.env` 不在待提交列表中
- 密钥一旦泄露，应立即在平台吊销并重新生成；仅删除提交记录无效，历史中仍可检索
- 线上部署的密钥只存放在平台 Secrets 中

## 参数说明

| 参数 | 默认值 | 位置 | 说明 |
|---|---|---|---|
| `size` | 500 | `chunk_text` | 片段长度（字符） |
| `overlap` | 100 | `chunk_text` | 相邻片段重叠（字符） |
| `k` | 4 | `retrieve` / 界面 | 召回片段数 |
| `temperature` | 0.3 | `answer` | 生成随机性 |
| `max_tokens` | 800 | `answer` | 回答长度上限 |

修改 `size` 或 `overlap` 后必须重新运行 `ingest.py`，否则索引与切分策略不一致。修改 `EMBED_MODEL` 同样需要重建索引——不同模型的向量维度与语义空间不同，旧索引无法复用。
