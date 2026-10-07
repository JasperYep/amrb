# AMRB · Agent Memory Research MCP

一个面向 Agent Memory 研究的 MCP 工作区：论文事实与个人研究认知分开保存。

```text
Agent ── AMRB MCP ──┬── Paper RAG / Qdrant：论文证据
                    └── Hindsight：研究者长期认知
```

同一个 MCP 提供 `search_papers`、`list_papers`、`get_paper_pages`、`retain`、`recall`、`reflect`。客户端只需注册一个 stdio MCP 命令。

## 部署

需要 Python 3.12+、[`uv`](https://docs.astral.sh/uv/) 和任意支持 MCP 的 Agent。Linux `systemd --user` 可自动管理 Hindsight API；其他环境可按 `amrb setup` 输出手动启动。

```bash
git clone git@github.com:JasperYep/amrb.git
cd amrb
uv sync --locked
cp .env.example .env
```

在 `.env` 配置 Embedding 模型、Key、Base URL，以及 Hindsight LLM 的模型、Key、Base URL。两者可共用同一 OpenAI 兼容服务；Qdrant 默认本地运行，使用 Cloud 时再填写 `QDRANT_URL` 和 `QDRANT_API_KEY`。

```bash
uv run amrb setup
```

初始化 Hindsight API、创建 AMRB 专属 `agent-memory-research` bank，并检查 Qdrant。把 PDF 放入 `papers/`，首次建索引：

```bash
uv run amrb paper sync
```

在 Agent 的 MCP 配置中注册以下 stdio 命令（把路径替换为本机仓库绝对路径）：

```text
uv --directory /path/to/amrb run amrb-mcp
```

## 隔离与数据

AMRB 的 MCP 只访问 `agent-memory-research` bank。Hindsight UI 会列出同一服务中的所有 banks；列表共用不代表记忆读写混用。新电脑首次部署会创建独立的本地 Hindsight 数据库。迁移记忆时只导出/导入 AMRB bank，不要复制包含其他项目数据的整个 Hindsight 数据目录。

`.env`、`.state/` 和 `papers/*.pdf` 不进 Git。换电脑时单独复制论文 PDF；本地 Qdrant 索引可用 `paper sync` 重建。Hindsight 默认数据位于 `~/.hindsight/data/`。

## 日常维护

```bash
uv run amrb status
uv run amrb doctor
uv run amrb paper add ~/Downloads/paper.pdf
uv run amrb paper sync
uv run amrb service status
```

Hindsight 官方 Web UI 是可选的独立进程：

```bash
npx --yes @vectorize-io/hindsight-control-plane --api-url http://127.0.0.1:8888
```

详细研究规则见 [AGENTS.md](AGENTS.md)；环境变量见 [.env.example](.env.example)。
