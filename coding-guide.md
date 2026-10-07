# Agent Memory Research Workspace

## 0. Goal

实现一个可长期使用的 **Agent Memory 研究工作区**。

它不是独立聊天应用，不实现自己的 Agent Runtime。直接使用 **Codex** 作为 Agent Host，由 Codex 负责：

- 推理
- 对话
- 规划
- Tool Use
- 当前会话上下文
- 本地文件操作
- 必要时的代码执行

项目只提供两个长期能力层：

```text
                    Codex
              Agent / Reasoning
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
     Paper RAG MCP         Hindsight MCP
          │                     │
          ▼                     ▼
     LlamaIndex            Research Memory
          │              retain / recall
          ▼                  / reflect
       Qdrant
```

核心语义必须始终保持：

```text
Paper RAG
= 论文里写了什么

Hindsight
= 我对 Agent Memory 这个研究方向形成了什么长期认知

Codex
= 当前应该查什么、怎么组合、如何回答
```

最终用户体验：

```bash
cd agent-memory-research
codex
```

然后直接：

```text
Hindsight 的核心 memory mechanism 是什么？

我之前怎么看 Hindsight？

我的判断和论文原文一致吗？

根据最近读过的论文，我目前怎么看 memory update？

哪些论文支持我对 lifecycle evaluation 的判断？

我现在还有哪些 unresolved research questions？
```

不要再开发第二套聊天 UI、Agent Loop 或 Conversation Manager。

---

# 1. 执行要求

直接完成整个项目。

除以下情况外，不要中途询问用户：

```text
必须获得用户私密凭据
必须执行不可逆外部操作
存在真正无法从代码、文档或合理默认值解决的阻塞
```

其他设计细节自行采用本文件中的默认方案。

完成标准不是“代码框架已经搭好”，而是：

```text
依赖可安装
Paper RAG 可 ingest
Paper RAG MCP 可运行
Hindsight 可连接
Codex 能看到两个 MCP
测试通过
README 可从零启动
示例研究问题可以工作
```

如果第三方库当前 API 与本指南示例存在差异：

1. 优先查当前官方文档；
2. 使用当前稳定 API；
3. 不为了保持示例语法而锁死旧版本；
4. 将最终实际版本写入 `uv.lock`；
5. 保持本指南定义的系统边界不变。

当前已确认的技术前提：

- Codex 支持项目级 `.codex/config.toml`，并可在其中配置 MCP；CLI 与 IDE 使用相同配置层。
- Codex 会自动读取 `AGENTS.md` 作为项目指令，因此研究策略应放在 `AGENTS.md`，而不是硬编码进 RAG 服务。
- Hindsight 提供本地 MCP Server，并支持固定到单个 bank；核心能力包括 `retain`、`recall`、`reflect`。
- 不使用 Hindsight 旧的 Codex 自动记忆集成。当前官方已将其迁移到 Coding Agents 插件，而该插件会自动注入/保留 coding session；这与本项目需要“选择性保存研究认知”的语义不同。
- MCP Python SDK 当前稳定线为 v2；不要复制旧版 `FastMCP` 示例，应使用当前稳定接口。
- LlamaIndex 已有 PyMuPDF Reader、Ingestion Pipeline 和 Qdrant Vector Store 集成，直接使用，不自行实现 PDF parser、embedding pipeline 或 vector database。

---

# 2. 产品原则

## 2.1 不造轮子

必须使用成熟组件：

```text
Agent Host        Codex

Memory            Hindsight

RAG Framework     LlamaIndex

Vector Store      Qdrant

PDF Parsing       LlamaIndex PyMuPDFReader / PyMuPDF

MCP               官方 MCP Python SDK v2

Python Env         uv

Tests              pytest
```

不要自行实现：

```text
Agent Loop
Memory Engine
Memory Extraction
Memory Dedup
Memory Consolidation
Vector Database
Embedding Index
PDF Parser
MCP Protocol
Conversation Store
LLM Provider Layer
```

---

## 2.2 Local-first

默认：

```text
论文           本地
Qdrant         本地
RAG metadata   本地
Hindsight      本地
Codex workspace 本地
```

只有模型调用可以经过外部 API。

以后允许把：

```text
Qdrant Local → Qdrant Server / Cloud
Hindsight Local → Self-host / Cloud
```

但业务接口不能因此改变。

---

## 2.3 极薄架构

属于本项目自己的代码应该尽可能少。

目标不是做 Framework。

目标是：

```text
把成熟组件以正确语义连接起来。
```

不要建立：

```text
repositories/
services/
managers/
controllers/
factories/
adapters/
registries/
```

这种没有现实需要的层级。

理想调用路径：

```text
MCP Tool
  ↓
RAG Service
  ↓
LlamaIndex
  ↓
Qdrant
```

最多三层。

---

## 2.4 RAG 和 Memory 永远分离

这是整个项目最重要的架构规则。

禁止：

```text
论文全文 → Hindsight
论文 chunk → Hindsight
RAG 搜索结果 → 自动 retain
论文摘要 → 自动 retain
```

论文属于：

```text
Paper RAG
```

研究者长期形成的认知属于：

```text
Hindsight
```

例如：

```text
“Mem0 的论文使用某种 update mechanism。”
```

属于 RAG。

而：

```text
“我认为 Mem0 对我的研究最重要的是它的 update abstraction。”
```

属于 Memory。

---

# 3. Memory 的定义

Hindsight bank 固定为：

```text
agent-memory-research
```

它保存的不是“聊天记录”，而是：

```text
研究判断
概念理解
taxonomy
hypothesis
research question
research decision
阅读后的评价
长期研究偏好
研究状态
```

典型 Memory：

```text
我认为 memory benchmark 不应该只评价最终回答。

我目前把 Agent Memory 工作分成 architecture、
formation、evolution、retrieval、evaluation 几类。

Hindsight 对我的工作最有价值的是 consolidation。

我还没有解决如何独立评价 memory update quality。

写 related work 时优先比较 mechanism，而不是简单总结论文。
```

不应保存：

```text
Mem0 第 4 页写了……
Hindsight 提出了……
论文 X 的摘要是……
刚才 search_papers 返回了……
用户问了一次……
Codex 临时推断……
```

---

# 4. 为什么不使用 Hindsight Coding Agents 自动插件

不要安装：

```text
@vectorize-io/hindsight-coding-agents
```

作为本项目的研究 Memory 机制。

原因不是该插件不好，而是语义不同。

Coding Agents 插件目标是：

```text
自动记住代码项目中的工作历史
```

本项目目标是：

```text
只保存值得长期保留的研究认知
```

因此使用：

```text
Hindsight MCP
```

并让 Codex 根据 `AGENTS.md` 明确决定什么时候调用：

```text
retain
recall
reflect
```

Memory formation 必须是 Agent 的显式行为，而不是所有 session 的自动副作用。

---

# 5. 最终目录结构

保持目录短而明确：

```text
agent-memory-research/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── .codex/
│   └── config.toml
│
├── papers/
│   └── .gitkeep
│
├── src/
│   └── paper_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       ├── ingest.py
│       ├── retrieval.py
│       ├── server.py
│       └── cli.py
│
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   ├── test_citations.py
│   └── test_mcp.py
│
└── evals/
    ├── rag.yaml
    └── README.md
```

不要增加目录，除非已经出现明确需要。

---

# 6. Python 与依赖

使用：

```text
Python >= 3.12
uv
```

使用当前兼容稳定版本，并让 `uv.lock` 锁定精确依赖。

需要的依赖类别：

```text
mcp
llama-index-core
llama-index-readers-file
llama-index-vector-stores-qdrant
对应 embedding provider 的 LlamaIndex integration
qdrant-client
pymupdf
pydantic-settings
typer
rich
pytest
pytest-asyncio
```

不要安装完整的大而全依赖，如果 LlamaIndex 已拆包则优先安装实际使用的 package。

不要添加 LangChain。

不要同时存在两个 RAG framework。

---

# 7. Embedding

Embedding 与 Codex 使用的模型完全解耦。

通过环境变量配置。

例如：

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

但代码必须允许以后替换。

不要自行建立复杂 Provider Framework。

只在：

```text
config.py
```

中根据配置创建 LlamaIndex 支持的 embedding implementation。

如果当前 LlamaIndex 已提供对应 integration，直接使用。

第一版至少支持一个稳定 embedding provider。

其他 provider 不作为完成条件。

---

# 8. Qdrant

默认使用 Qdrant Local Mode，持久化目录：

```text
.state/qdrant/
```

不要第一天要求 Docker。

配置：

```env
QDRANT_PATH=.state/qdrant
QDRANT_COLLECTION=agent_memory_papers
```

同时预留：

```env
QDRANT_URL=
QDRANT_API_KEY=
```

行为：

```text
如果 QDRANT_URL 存在
→ 连接 Qdrant Server

否则
→ 使用本地 persistent path
```

RAG 业务代码不能知道使用的是 local 还是 remote。

不要再包一层自制 VectorStore interface。

LlamaIndex 的 `QdrantVectorStore` 就是边界。

---

# 9. Papers

论文统一放：

```text
papers/
```

第一版只正式支持：

```text
.pdf
```

不要扫描整个 repository。

不要自动下载论文。

不要自动搜索 arXiv。

不要做文献管理器。

用户负责把 PDF 放进：

```text
papers/
```

系统负责：

```text
parse
chunk
embed
index
retrieve
cite
```

---

# 10. PDF Parsing

使用 LlamaIndex 的：

```text
PyMuPDFReader
```

目标不是完美理解 PDF layout。

第一阶段只要求：

```text
正文能够检索
页码能够保留
论文身份能够保留
```

每一页必须保留：

```text
paper_id
file_name
page_number
```

如果 Reader 已返回 page metadata，复用。

不要 OCR。

不要解析 Figure。

不要解析公式。

不要恢复复杂表格。

这些以后只有出现实际需求再处理。

---

# 11. Paper ID

Paper ID 必须稳定。

不要使用随机 UUID。

默认：

```text
SHA256(file bytes)[:16]
```

同时记录：

```text
file_name
file_hash
```

如果文件内容没有变化：

```text
paper_id
```

保持不变。

---

# 12. Chunking

使用 LlamaIndex 原生 Node Parser。

第一版：

```text
SentenceSplitter
```

默认：

```text
chunk_size = 700
chunk_overlap = 100
```

通过配置可修改。

不要自己写 tokenizer。

不要做 semantic chunking。

不要做 hierarchical chunking。

不要做 GraphRAG。

后续如果真实 retrieval eval 表明 section-aware chunking 有价值，再加入。

---

# 13. Chunk Metadata

每个 Node 至少必须包含：

```text
paper_id
file_name
page_number
content_hash
```

如果能够低成本提取：

```text
title
section
```

可以增加。

但：

```text
page_number
```

是硬要求。

用户必须能够从任何 retrieval result 回到 PDF 原始位置。

---

# 14. Paper Metadata

第一版不要为了 metadata extraction 引入复杂 pipeline。

默认：

```text
title = PDF metadata title
```

如果 PDF metadata 无效：

```text
title = filename stem
```

不要调用 LLM 自动猜 title、authors、venue。

以后需要再加。

---

# 15. Ingestion

提供 CLI：

```bash
uv run paper-rag ingest
```

流程：

```text
papers/*.pdf
      │
      ▼
检测新增 / 修改 / 删除
      │
      ▼
PyMuPDFReader
      │
      ▼
SentenceSplitter
      │
      ▼
Embedding
      │
      ▼
Qdrant
```

---

# 16. Incremental Ingestion

长期使用必须支持增量索引。

在：

```text
.state/index.json
```

维护极小 manifest：

```json
{
  "papers/example.pdf": {
    "sha256": "...",
    "paper_id": "...",
    "indexed_at": "..."
  }
}
```

不要为这件事引入 PostgreSQL 或另一个 ORM。

每次 ingest：

```text
不存在于 manifest
→ index

hash changed
→ 删除旧 paper_id nodes
→ re-index

hash unchanged
→ skip

manifest 有但文件不存在
→ 删除对应 nodes
→ remove manifest entry
```

操作必须幂等。

连续执行：

```bash
paper-rag ingest
paper-rag ingest
```

第二次不应重新 embedding 未改变论文。

---

# 17. 原子性

单篇论文 re-index 时：

```text
parse / chunk / embed 成功
↓
再替换旧 index
```

不要：

```text
先删除旧内容
↓
embedding 失败
↓
论文彻底消失
```

如果第三方 API 限制导致无法实现完全 transaction，至少保证失败时：

```text
manifest 不被错误更新
```

并打印明确错误。

---

# 18. RAG 不负责回答问题

这是关键设计。

Paper RAG MCP 不应该内部调用聊天 LLM 生成最终答案。

它只返回：

```text
retrieval results
```

让 Codex 自己综合。

错误：

```text
search_papers()
→ query engine
→ LLM answer
→ Codex
```

正确：

```text
search_papers()
→ relevant chunks
→ Codex
```

原因：

```text
Codex 才是 Agent 和 reasoning layer。
```

不要做 LLM-inside-LLM。

---

# 19. Retrieval

第一版使用纯 dense retrieval。

不要一开始加入：

```text
BM25
Hybrid Search
Reranker
Query Expansion
HyDE
RAG Fusion
Multi-query
Cross Encoder
```

默认：

```text
top_k = 8
```

允许工具参数覆盖。

检索结果按照 similarity 排序。

---

# 20. Retrieval 返回模型

定义清晰的 Pydantic model：

```python
class PaperHit(BaseModel):
    paper_id: str
    title: str
    file_name: str
    page: int
    score: float
    text: str
```

如有可靠 section：

```python
section: str | None = None
```

不要返回 LlamaIndex 内部 Node 对象。

MCP 边界必须是稳定的数据结构。

---

# 21. Citation

每条结果都必须提供可读 citation：

```text
Hindsight, p. 5
```

至少：

```text
文件名 + 页码
```

例如：

```text
hindsight.pdf, p. 5
```

若有 title：

```text
Hindsight: Learning from Experience, p. 5
```

不要只返回：

```text
chunk_9317
node_123
```

---

# 22. Paper RAG MCP

使用官方 MCP Python SDK 当前稳定版本。

不要复制 v1 的 `FastMCP` API。

实现一个 MCP Server：

```text
paper-rag
```

默认使用：

```text
stdio
```

因为：

```text
只有本地 Codex 使用
无需端口
无需额外 daemon
生命周期简单
```

未来如需多客户端再增加 Streamable HTTP。

---

# 23. MCP Tools

第一版严格只暴露三个工具。

## `search_papers`

接口语义：

```text
从论文语料中寻找与研究问题相关的证据。
```

参数：

```python
query: str
top_k: int = 8
paper: str | None = None
```

返回：

```text
PaperHit[]
```

如果 `paper` 不为空：

```text
通过 metadata filter 限定论文
```

不要先 retrieve 再 Python 手工过滤。

---

## `list_papers`

返回当前已经索引的论文：

```text
paper_id
title
file_name
```

用于：

```text
“我目前有哪些 Hindsight 相关论文？”
“比较这些论文。”
```

不要返回 chunk。

---

## `get_paper_pages`

参数：

```python
paper_id: str
start_page: int
end_page: int | None = None
```

用途：

```text
Codex 在 retrieval 后需要阅读连续上下文时，
直接获取指定论文连续页面。
```

限制：

```text
单次最多 10 页
```

避免整个 PDF 一次进入 context。

---

# 24. 不要暴露的 MCP Tools

不要暴露：

```text
embed
vector_search
query_engine
create_index
delete_node
rerank
get_embedding
chunk_pdf
```

这些是基础设施细节。

Agent 只应该看到研究语义。

最终 Paper MCP surface：

```text
search_papers
list_papers
get_paper_pages
```

---

# 25. Ingestion 不作为 Agent Tool

不要把：

```text
ingest
reindex
delete index
```

暴露给 Codex MCP。

Ingestion 是运维动作。

通过 CLI：

```bash
uv run paper-rag ingest
```

执行。

这样可以避免 Agent 在研究过程中意外重建 index。

---

# 26. Paper RAG CLI

只实现：

```bash
uv run paper-rag ingest
uv run paper-rag status
uv run paper-rag search "memory consolidation"
uv run paper-rag eval
```

这些命令用于：

```text
维护
debug
eval
```

不是另一套聊天界面。

---

# 27. `status`

输出保持极简：

```text
Paper RAG

Papers       83
Chunks     4217
Vector DB  local
Collection agent_memory_papers

Last ingest
2026-10-07 11:42
```

不要 ASCII Logo。

不要 emoji。

不要彩色 dashboard。

Rich 仅用于：

```text
alignment
dim text
errors
```

---

# 28. `search`

开发调试：

```bash
uv run paper-rag search "memory consolidation"
```

输出：

```text
1  hindsight.pdf  p.5   0.892
   Memory consolidation ...

2  mem0.pdf       p.7   0.847
   ...
```

用于直接观察 retrieval。

---

# 29. Hindsight

Hindsight 完全作为外部 Memory 系统使用。

不要：

```text
import hindsight internals
```

不要：

```text
自己访问 Hindsight PostgreSQL
```

不要：

```text
自己同步 Memory 数据
```

只通过官方：

```text
MCP
```

使用。

---

# 30. Hindsight Local MCP

开发默认启动官方 local MCP。

bank：

```text
agent-memory-research
```

优先使用 single-bank endpoint：

```text
http://localhost:8888/mcp/agent-memory-research/
```

这样 Codex 不需要每次决定：

```text
bank_id
```

降低 Tool 参数和误操作概率。

Hindsight 官方 local MCP 会自行处理其持久化存储。不要在本 repo 再维护 Memory DB。

---

# 31. Hindsight Tools

Codex 侧只启用研究需要的工具。

第一版允许：

```text
retain
recall
reflect
```

如果当前 Hindsight 的 Mental Model tools 稳定并且工具名与当前文档一致，可以额外允许：

```text
list_mental_models
get_mental_model
create_mental_model
refresh_mental_model
```

不要一次把 Hindsight 二十多个工具全部暴露给 Codex。

Codex 当前支持通过 `enabled_tools` 限制单个 MCP server 暴露的工具，应使用这个能力保持 tool surface 小。

---

# 32. `retain`

`retain` 用于：

```text
形成长期研究认知。
```

只在以下情况使用：

```text
用户明确提出长期观点
用户修正自己之前的观点
形成新的研究 hypothesis
形成 taxonomy
提出值得长期追踪的问题
作出研究决策
明确长期研究偏好
出现重要研究状态变化
```

例如：

```text
“我现在觉得 memory privacy 的关键问题不是泄露，
而是 lifecycle 中权限变化后的状态传播。”
```

应该 retain。

而：

```text
“Hindsight 第四节说了什么？”
```

不应 retain。

---

# 33. `recall`

用于：

```text
查找过去形成的相关研究认知。
```

典型问题：

```text
我之前怎么看 Mem0？

我是不是认为 retrieval 不是整个 memory lifecycle？

之前对 deletion 有什么判断？

我现在对 benchmark taxonomy 是怎么分的？
```

这些优先：

```text
recall
```

而不是 search_papers。

---

# 34. `reflect`

用于：

```text
跨多条长期 Memory 做综合。
```

典型问题：

```text
综合我最近几个月的阅读，
我目前对 Agent Memory 的整体 taxonomy 是什么？

我现在对 memory evaluation 的核心观点是什么？

我的研究方向最近发生了什么变化？

目前哪些 research questions 一直没有解决？
```

这些优先：

```text
reflect
```

不要把 `reflect` 当成更高级的 `recall`。

简单事实查找仍然使用：

```text
recall
```

---

# 35. Mental Models

Mental Model 不是第一天完成条件。

但架构应允许以后建立几个稳定的：

```text
Current Agent Memory Taxonomy

Current View of Memory Evaluation

Open Research Questions

Current View of Memory Lifecycle
```

Mental Model 用于：

```text
维护需要反复综合的研究认知。
```

不要为每篇论文创建 Mental Model。

---

# 36. Hindsight Mission

如果当前 Hindsight 版本支持通过官方配置设置 retain / reflect / observation mission，则配置如下语义。

核心 retain policy：

```text
这个 bank 保存用户在 Agent Memory 研究方向上的长期研究认知。

优先保存：
- 用户明确形成的概念理解
- 用户对论文的判断和评价
- 用户自己的 taxonomy
- hypothesis
- unresolved research questions
- research decisions
- 长期研究偏好
- 重要研究状态

不要保存：
- 论文原文
- 论文摘要
- Paper RAG retrieval result
- 可以再次从论文检索获得的事实
- 临时问题
- 一次性请求
- Agent 自己未经用户认可的猜测
```

如果当前 API 不支持 mission，不要自己实现 mission engine。

依靠：

```text
AGENTS.md
```

控制 Codex 的 retain 行为。

---

# 37. Codex 配置

创建：

```text
.codex/config.toml
```

项目配置至少包含两个 MCP：

```text
paper-rag
hindsight
```

Paper RAG：

```toml
[mcp_servers.paper-rag]
command = "uv"
args = ["run", "paper-rag-mcp"]
required = true
enabled_tools = ["search_papers", "list_papers", "get_paper_pages"]
```

Hindsight：

```toml
[mcp_servers.hindsight]
url = "http://localhost:8888/mcp/agent-memory-research/"
required = false
enabled_tools = ["retain", "recall", "reflect"]
```

如果当前 Codex config schema 对字段名有变化：

```text
查官方 Codex config reference，
调整到当前稳定语法。
```

不要退回手工要求用户每次：

```bash
codex mcp add ...
```

项目配置应该进入 repo。

---

# 38. `required`

Paper RAG：

```text
required = true
```

原因：

```text
这是本项目核心能力。
```

Hindsight：

```text
required = false
```

原因：

```text
即使 Hindsight daemon 暂时未启动，
Codex 仍应可以阅读论文和维护代码。
```

但 README 必须明确提示：

```text
进行长期研究前启动 Hindsight。
```

---

# 39. AGENTS.md

`AGENTS.md` 是整个系统最重要的行为策略文件。

不要写成几百行 prompt。

保持短、明确、稳定。

必须包含以下语义。

```markdown
# 角色

你是我的 Agent Memory 长期研究助手。

你的任务不是简单总结论文，而是协助我长期理解 Agent Memory 研究领域。

# 两类长期信息源

Paper RAG 表示外部论文证据。

Hindsight 表示我本人长期形成的研究认知。

严格区分两者。

论文中的事实不能因为被检索到，就自动成为我的 Memory。

# Tool 使用

当问题是“论文怎么说”时，优先使用 Paper RAG。

当问题是“我之前怎么看”时，优先使用 Hindsight recall。

当问题需要综合我过去大量研究认知时，使用 Hindsight reflect。

当问题要求比较“我的观点”和“论文证据”时，同时使用 Hindsight 和 Paper RAG。

如果当前上下文已经足以可靠回答，不要为了调用工具而调用工具。

# Paper RAG

不要凭记忆声称论文具体内容。

涉及某篇论文的 mechanism、实验、定义、结果或 claim 时，
如果当前上下文没有可靠证据，使用 search_papers。

重要结论尽可能指出论文和页码。

当单个 chunk 上下文不足时，使用 get_paper_pages 阅读连续上下文。

# Memory

只有具有跨 session 长期价值的信息才 retain。

适合 retain：
- 我的明确研究判断
- 概念理解
- taxonomy
- hypothesis
- unresolved research question
- research decision
- 稳定研究偏好
- 重要研究状态

不适合 retain：
- 论文正文
- Paper RAG 检索内容
- 普通问答
- 临时任务
- 一次性事实
- 你自己的未经确认推断

不要因为某句话“可能以后有用”就 retain。

宁可少记，不要乱记。

如果我的新观点明显修正旧观点，
不要同时把相互冲突的观点当作当前真值。
优先让 Hindsight 保持最新研究状态，并保留必要的历史演化。

# 证据与判断

回答时明确区分：
1. 论文证据；
2. 我的历史判断；
3. 你基于两者做出的当前分析。

不要把第三类伪装成第一类或第二类。

# 风格

默认简洁、直接、高信息密度。

研究问题优先解释：
problem
mechanism
memory representation
formation / update / retrieval / use
evaluation
limitations

不要写宣传式论文摘要。

如果用户只问一个具体问题，直接回答该问题，不机械输出固定模板。
```

实现时使用这份内容作为基础，不要自行膨胀成巨型 prompt。

Codex 会自动加载 `AGENTS.md`，无需在 Python 中重复这些规则。

---

# 40. 不重复 Policy

以下内容只存在于：

```text
AGENTS.md
```

不要复制到：

```text
Python prompt
MCP tool implementation
README
config
```

除非是用户文档需要简短解释。

目标：

```text
一个行为策略来源。
```

---

# 41. Tool Description

MCP Tool description 很重要。

保持准确而具有语义。

`search_papers`：

```text
Search the indexed Agent Memory paper corpus for external paper evidence.
Use this for claims about what papers say, propose, evaluate, or report.
Do not use it to retrieve the user's personal research judgments.
```

`list_papers`：

```text
List papers currently available in the indexed research corpus.
```

`get_paper_pages`：

```text
Read a continuous page range from one indexed paper when search snippets
are insufficient for interpretation or citation.
```

不要写含糊描述：

```text
Search stuff.
Get docs.
```

---

# 42. Config

项目自己的配置只保留必要项。

`.env.example`：

```env
# Embedding
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=

# Qdrant
QDRANT_PATH=.state/qdrant
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=agent_memory_papers

# RAG
RAG_TOP_K=8
RAG_CHUNK_SIZE=700
RAG_CHUNK_OVERLAP=100
```

Hindsight 的配置使用：

```text
Hindsight 官方环境变量
```

不要把所有 Hindsight 环境变量重新映射成自己的名字。

README 只说明如何传给 Hindsight。

---

# 43. Secrets

禁止：

```text
API key
token
credential
```

进入：

```text
Git
AGENTS.md
.codex/config.toml
index.json
Qdrant payload
tests
logs
```

`.gitignore` 至少：

```gitignore
.env
.state/
__pycache__/
.pytest_cache/
.venv/
```

是否忽略：

```text
papers/
```

默认不要整体忽略。

研究者可自行决定 PDF 是否进入 Git。

---

# 44. Logging

不要开发完整 observability system。

Python logging 即可。

默认：

```text
WARNING
```

CLI 加：

```text
--verbose
```

时：

```text
INFO
```

调试时允许：

```text
DEBUG
```

不要默认打印：

```text
embedding vector
完整 API response
完整 PDF text
secret
```

---

# 45. 可观察性

真正需要观察的是 retrieval。

`paper-rag search` 必须能看到：

```text
query
top_k
paper
score
title
page
text preview
```

这是理解 RAG 的主要入口。

Memory 的观察使用 Hindsight 自带能力，不另外复制一套 Memory trace。

Codex Tool Call 本身已经可以观察：

```text
何时 search_papers
何时 recall
何时 reflect
何时 retain
```

不要再实现自己的 Agent Trace 数据库。

---

# 46. Error Handling

错误必须简洁且可操作。

例如：

```text
Paper RAG 未找到索引。

请先运行：

uv run paper-rag ingest
```

而不是：

```text
RuntimeError: NoneType...
```

Embedding key 缺失：

```text
缺少 OPENAI_API_KEY。

请复制 .env.example 为 .env，
并配置 embedding provider。
```

Hindsight 不可用时由 Codex MCP 状态负责。

不要让 Paper RAG 因 Hindsight 不在线而失败。

两个系统必须独立。

---

# 47. Tests

不要测试第三方库内部行为。

只测试自己的边界。

必须有：

```text
test_ingest.py
```

验证：

```text
一个测试 PDF
→ 成功索引
→ manifest 正确
→ 再 ingest 不重复处理
```

---

```text
test_retrieval.py
```

验证：

```text
已知 query
→ 返回正确 paper
→ 返回 page
→ top_k 生效
```

---

```text
test_citations.py
```

验证：

```text
每个 PaperHit 都有：
paper_id
title/file_name
page
text
```

---

```text
test_mcp.py
```

验证三个工具：

```text
search_papers
list_papers
get_paper_pages
```

存在且返回结构正确。

---

# 48. 测试数据

不要把真实大量论文复制进 tests。

创建一个很小的 fixture PDF。

内容可以包含：

```text
Page 1:
Agent memory systems store information across interactions.

Page 2:
Memory retrieval selects relevant stored information.

Page 3:
Memory update modifies previously retained information.
```

然后测试：

```text
query = "How is stored information modified?"
```

应能 retrieve page 3。

---

# 49. Eval

测试解决：

```text
代码有没有坏。
```

Eval 解决：

```text
RAG 好不好用。
```

`evals/rag.yaml`：

```yaml
cases:
  - query: "How does Hindsight perform memory consolidation?"
    expected_papers:
      - hindsight

  - query: "How does Mem0 update memories?"
    expected_papers:
      - mem0

  - query: "Which work evaluates long-term memory retrieval?"
    expected_papers:
      - ...
```

Eval 第一版只做简单：

```text
Recall@K
```

判断 expected paper 是否出现在 top-k。

不要第一天：

```text
LLM judge
RAGAS
复杂 benchmark framework
```

---

# 50. Eval CLI

```bash
uv run paper-rag eval
```

输出：

```text
RAG Eval

Cases       12
Passed      10
Recall@8   0.83
```

失败项：

```text
FAIL
Query: ...
Expected: hindsight
Retrieved: mem0, memgpt, ...
```

这样以后修改：

```text
chunk size
embedding
top_k
hybrid search
```

可以真实比较，而不是凭感觉。

---

# 51. README

README 必须足够让一个完全不了解项目的人从零启动。

结构保持：

```text
# Agent Memory Research Workspace

一句话说明

## Architecture

## Setup

## Add Papers

## Start Hindsight

## Start Codex

## Usage

## Maintenance

## Tests

## Design Boundary
```

不要写长篇理念。

---

# 52. Setup UX

理想流程：

```bash
git clone ...
cd agent-memory-research

uv sync

cp .env.example .env
# 填 embedding API key

cp ~/Downloads/*.pdf papers/

uv run paper-rag ingest
```

启动 Hindsight：

```bash
HINDSIGHT_... uvx --from hindsight-api hindsight-local-mcp
```

这里必须根据当前 Hindsight 官方文档填写真正有效的当前启动方式和必要环境变量。

不要凭本指南猜测已经变化的 provider 配置。

Hindsight local MCP 官方当前入口是 `hindsight-local-mcp`，并在本地暴露 MCP endpoint。

然后：

```bash
codex
```

验证：

```bash
codex mcp list
```

能够看到：

```text
paper-rag
hindsight
```

---

# 53. First-run 验证

README 提供以下验证问题：

```text
列出当前论文库里的论文。
```

预期：

```text
Codex → list_papers
```

---

```text
Hindsight 的 memory consolidation 是怎么做的？
```

预期：

```text
Codex → search_papers
```

---

```text
记住：我认为 Hindsight 最值得关注的是 consolidation，
而不是普通 semantic retrieval。
```

预期：

```text
Codex → retain
```

---

关闭 Codex。

重新启动。

```text
我之前怎么看 Hindsight？
```

预期：

```text
Codex → recall
```

---

```text
我的看法和 Hindsight 论文自己的 framing 一致吗？
```

预期：

```text
Codex
├── recall
└── search_papers
```

然后明确区分：

```text
论文证据
用户历史观点
当前分析
```

---

# 54. 最重要的验收场景

必须手动验证以下六类行为。

## A. RAG only

```text
Mem0 的 memory update 怎么实现？
```

正确：

```text
search_papers
```

通常不需要：

```text
recall
```

---

## B. Memory only

```text
我之前怎么看 Mem0？
```

正确：

```text
recall
```

通常不需要：

```text
search_papers
```

---

## C. RAG + Memory

```text
我之前对 Mem0 的判断和论文一致吗？
```

正确：

```text
recall
search_papers
```

---

## D. Reflect

```text
综合之前的阅读，
我现在对 Agent Memory evaluation 的总体观点是什么？
```

正确：

```text
reflect
```

必要时再：

```text
search_papers
```

---

## E. Selective Retain

```text
我现在认为 memory evolution 应该和 retrieval
作为两个独立评价阶段。
```

正确：

```text
retain
```

---

## F. 不应 Retain

```text
Hindsight 的实验用了几个 dataset？
```

正确：

```text
search_papers
answer
```

不应该：

```text
retain
```

---

# 55. 研究回答的默认原则

涉及论文事实：

```text
优先 evidence。
```

涉及用户观点：

```text
优先 memory。
```

涉及综合分析：

```text
evidence + memory。
```

冲突时：

```text
不要偷偷消解。
```

例如：

```text
你的历史判断：
X 不支持 update。

论文证据：
X 明确提供 update。
```

应该指出：

```text
你的历史判断与论文证据不一致。
```

而不是擅自把旧 Memory 当成论文事实。

---

# 56. Context Discipline

不要因为能调用工具就把大量结果塞给 Codex。

`search_papers` 默认：

```text
top_k = 8
```

每个 hit 返回完整 chunk，但 chunk 本身控制在约：

```text
700 tokens
```

`get_paper_pages`：

```text
最多 10 页
```

不要增加：

```text
get_whole_paper
```

---

# 57. Tool Discipline

Codex 看到的核心长期工具应保持非常少：

```text
Paper
├── search_papers
├── list_papers
└── get_paper_pages

Memory
├── retain
├── recall
└── reflect
```

六个工具已经足够。

未来增加功能时首先问：

```text
这个能力是不是已有工具通过参数就能表达？
```

如果是：

```text
不要加新 tool。
```

---

# 58. 不实现自己的 Research Memory Schema

不要重新创建：

```text
concept
taxonomy
paper_judgment
research_question
research_state
preference
```

数据库表。

这些概念可以：

```text
自然存在于 Hindsight Memory
```

必要时通过：

```text
tags
metadata
mission
```

表达。

不要同时维护：

```text
Hindsight
+
自己的 memories.sqlite
```

两个真相源。

---

# 59. 不实现 Agent

禁止出现：

```text
agent.py
agent_loop.py
planner.py
tool_router.py
conversation.py
context_manager.py
```

Codex 就是 Agent Runtime。

如果发现自己正在写：

```python
while tool_call:
    ...
```

说明架构走错了。

删除它。

---

# 60. 不实现 Chat UI

禁止：

```text
Streamlit
Gradio
React
Next.js
Terminal Chat Loop
```

Codex 就是交互界面。

项目 CLI 只用于：

```text
ingest
status
search
eval
```

---

# 61. 不实现自己的 LLM 调用

Paper RAG 不需要 LLM。

它只需要：

```text
Embedding
```

因此不要添加：

```text
OpenAI chat completion
Gemini chat
DeepSeek chat
```

到 RAG codebase。

最终生成答案由 Codex 完成。

Memory 内部是否使用 LLM：

```text
由 Hindsight 自己负责。
```

---

# 62. 不做 Web Search

Paper RAG 第一版只搜索：

```text
papers/
```

如果 Codex 本身以后使用 Web：

```text
那属于 Agent 的其他能力。
```

不要把：

```text
Google Search
Semantic Scholar
arXiv Search
Crossref
```

混进 Paper MCP。

---

# 63. 不做 Citation Manager

不集成：

```text
Zotero
BibTeX manager
Semantic Scholar library
```

第一版。

如果长期使用后确实需要，可以再增加一个独立 source。

不要让第一版失焦。

---

# 64. 不做 Graph

即使研究主题本身是 Memory，也不要因此引入：

```text
Neo4j
Graphiti
GraphRAG
Knowledge Graph
```

Hindsight 内部如何组织 Memory：

```text
交给 Hindsight。
```

Paper evidence：

```text
先使用普通 RAG。
```

只有 Eval 明确证明 graph 能解决实际失败案例，再考虑。

---

# 65. 不做 Hybrid Retrieval

第一版建立：

```text
dense baseline
```

并保存 Eval。

以后真的出现：

```text
术语精确匹配失败
论文名检索失败
缩写检索失败
```

再引入：

```text
BM25 / hybrid
```

而不是因为“高级 RAG 一般都有”就提前加入。

---

# 66. 长期演进顺序

未来只按真实失败案例升级。

推荐顺序：

```text
V1
Dense RAG
+ Hindsight
+ Codex

↓

V1.1
更好的 paper metadata

↓

V1.2
section-aware chunking

↓

V1.3
hybrid retrieval

↓

V1.4
reranker

↓

V2
Zotero / Semantic Scholar 等论文来源

↓

V3
研究 synthesis / evidence mapping
```

不要提前实现后面的阶段。

---

# 67. 可替换性

Codex、Hindsight、Qdrant 都不是 Project Domain 本身。

但不要因此制造复杂 abstraction。

边界天然已经存在：

```text
Agent boundary   = MCP
Memory boundary  = Hindsight MCP
Vector boundary  = LlamaIndex QdrantVectorStore
```

这些边界已经足够。

不要再造：

```text
IMemoryProvider
IVectorProvider
IAgentProvider
```

---

# 68. 如果未来替换 Codex

架构应天然允许：

```text
Codex
Claude Code
Pi
其它 MCP Agent
```

调用同一个：

```text
Paper RAG MCP
Hindsight MCP
```

因此任何核心研究逻辑都不能写成：

```text
Codex-specific Python code
```

Codex-specific 内容只存在于：

```text
.codex/config.toml
AGENTS.md
```

---

# 69. 如果未来替换 Hindsight

Research Workspace 自己不直接依赖 Hindsight Python API。

所以替换只涉及：

```text
.codex/config.toml
AGENTS.md
```

和新的 Memory MCP。

Paper RAG 不受影响。

---

# 70. 如果未来替换 Qdrant

只有：

```text
retrieval.py
```

和配置可能变化。

MCP tools：

```text
search_papers
list_papers
get_paper_pages
```

返回结构不变。

---

# 71. 代码风格

代码要求：

```text
短
显式
类型完整
低魔法
低抽象
低依赖
```

优先：

```python
def search_papers(query: str, top_k: int = 8) -> list[PaperHit]:
    ...
```

不要：

```python
class RetrievalOrchestrationManagerFactory:
    ...
```

---

# 72. 函数大小

一般保持：

```text
< 50 行
```

如果明显超过：

```text
拆成具有明确业务意义的函数。
```

不要为了满足行数限制机械拆函数。

---

# 73. 类型

所有公共函数：

```text
完整 type hints
```

跨模块结构：

```text
Pydantic model
```

模块内部简单数据：

```text
普通 Python 类型
```

不要为每个三字段对象创建类。

---

# 74. 注释

不要写：

```python
# Increment i
i += 1
```

只解释：

```text
为什么这样设计
第三方 API 的特殊约束
不明显的生命周期要求
```

---

# 75. 文档字符串

MCP tools：

```text
必须有清晰 docstring
```

因为 Tool description 会直接影响 Agent 的工具选择。

普通内部函数：

```text
只有语义不明显时再写。
```

---

# 76. Dependency Discipline

增加依赖前先检查：

```text
标准库能不能直接完成？
现有依赖有没有同样能力？
```

不要因为几行代码添加一个 package。

尤其避免：

```text
ORM
DI container
workflow engine
logging framework
retry framework
```

除非真实需要。

---

# 77. Retry

对 embedding API 的短暂失败：

```text
最多简单指数退避重试 3 次。
```

优先使用 provider SDK 自带 retry。

不要自己实现复杂 queue。

---

# 78. Concurrency

第一版 ingestion 可以 batch。

但不要做复杂异步流水线。

优先使用：

```text
LlamaIndex 自带 batching / ingestion pipeline
```

项目自己的代码保持同步或简单 async。

---

# 79. State

本项目本地 runtime state 全部放：

```text
.state/
```

例如：

```text
.state/
├── qdrant/
└── index.json
```

不要散落：

```text
db/
cache/
storage/
tmp/
data/
```

多个目录。

---

# 80. Git

`.state/` 不入 Git。

代码、配置模板、eval 入 Git。

论文是否入 Git由用户决定。

---

# 81. Acceptance Criteria

实现完成后，必须满足以下全部条件。

### 安装

```bash
uv sync
```

成功。

---

### 测试

```bash
uv run pytest
```

全部通过。

---

### Ingest

将至少一个 PDF 放入：

```text
papers/
```

执行：

```bash
uv run paper-rag ingest
```

成功建立 index。

第二次执行不会重复 embedding 未变化论文。

---

### Status

```bash
uv run paper-rag status
```

正确显示：

```text
paper count
chunk count
vector backend
last ingest
```

---

### Search

```bash
uv run paper-rag search "memory consolidation"
```

返回：

```text
论文
页码
score
文本
```

---

### MCP

Paper MCP 正常启动。

Codex 能发现：

```text
search_papers
list_papers
get_paper_pages
```

---

### Hindsight

Hindsight local MCP 正常运行。

Codex 能发现：

```text
retain
recall
reflect
```

---

### Cross-session Memory

Session A：

```text
记住：我目前认为 Hindsight 最重要的是 memory consolidation。
```

关闭 Codex。

Session B：

```text
我之前怎么看 Hindsight？
```

能够通过 Hindsight recall 找回。

---

### RAG / Memory 分离

问：

```text
Hindsight 的论文如何定义 X？
```

系统通过：

```text
Paper RAG
```

得到答案。

这个论文事实不能仅仅因为回答过一次就自动变成 Hindsight Memory。

---

### Combined Reasoning

问：

```text
我之前对 Hindsight 的判断，
和论文自己的 framing 一致吗？
```

Codex 应同时使用：

```text
recall
search_papers
```

再回答。

---

# 82. Definition of Done

不要在以下状态停止：

```text
代码已经生成
但没有运行

测试已经写
但没有执行

配置已经写
但 Codex 没连接 MCP

MCP 可以启动
但 search 实际不可用

README 看起来正确
但没有按步骤验证
```

完成前必须实际执行：

```text
uv sync
pytest
ingest
status
search
MCP startup
```

能自动验证的全部验证。

对于必须依赖用户真实 API key 或真实 Hindsight LLM 配置的部分：

```text
提供清晰的配置位
使用 mock / fixture 完成自动测试
明确记录剩余的一步人工验证
```

不要伪造成功。

---

# 83. 实现过程中的决策优先级

出现不确定设计时，按以下顺序选择：

```text
正确语义
>
简单
>
可维护
>
可观察
>
性能
>
功能数量
```

这个项目不是展示复杂工程能力。

目标是：

> 用尽可能少的自有代码，把 Codex、Paper RAG 和 Hindsight 组合成一个真正可长期使用的 Agent Memory 研究工作区。

---

# 84. 最终架构

最终系统应保持：

```text
                         AGENTS.md
                    Research Policy
                          │
                          ▼
                        Codex
                   Agent Runtime
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
        Paper RAG MCP             Hindsight MCP
             │                         │
        LlamaIndex                 Long-term
             │                  Research Memory
          Qdrant                  │     │     │
             │                 retain recall reflect
             ▼
      Agent Memory Papers
```

三层语义：

```text
Codex
= 当前思考

Paper RAG
= 外部证据

Hindsight
= 长期研究认知
```

不要合并这三层。

---

# 85. 最终目标

系统长期使用后应该能够自然回答：

```text
论文怎么说？
```

来源：

```text
Paper RAG
```

---

```text
我以前怎么看？
```

来源：

```text
Hindsight recall
```

---

```text
综合我过去的研究，我现在怎么看？
```

来源：

```text
Hindsight reflect
```

---

```text
我的观点与现有论文证据是否一致？
```

来源：

```text
Hindsight
+
Paper RAG
+
Codex reasoning
```

这就是整个项目。

不要继续增加架构，直到真实使用暴露出新的问题。
