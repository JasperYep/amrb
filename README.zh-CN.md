# AMRB — Agent Memory & Retrieval Bridge

[English](README.md) | 简体中文

面向 Agent 的检索与长期记忆 MCP 桥接层。

AMRB 为 MCP client 提供 PDF 检索、长期记忆和 workspace 运维能力。Agent 负责对话与推理；AMRB 连接检索和记忆组件。

```text
MCP client ── AMRB ──┬── PDF 检索 / Qdrant
                     └── 长期记忆 / Hindsight
```

一个 workspace 对应一个项目或知识领域，拥有独立的检索 collection 和 Hindsight memory bank。Setup 会保存一份紧凑 profile：领域、用途、检索资料范围和长期记忆准入规则；当目录中还没有 `AGENTS.md` 时，会据此生成稳定的工作规则。

## 部署

安装一次 AMRB，然后初始化目标 workspace：

```bash
git clone https://github.com/JasperYep/amrb.git
cd amrb
uv sync --locked
uv run amrb setup --workspace /absolute/path/to/workspace
```

从其他目录运行：

```bash
uv --directory /absolute/path/to/amrb run amrb setup --workspace /absolute/path/to/workspace
```

可以直接对 Agent 说：“部署 AMRB 并接入当前 Agent。”

Agent 会先确认项目名称、领域、用途、检索资料范围和长期记忆准入规则，再将确认后的 profile 保存到 `amrb.toml` 和新生成的 `AGENTS.md`。普通 session 中 profile 和工作规则保持稳定，维持一致的上下文前缀；项目用途变化时再显式更新。Setup 还会收集 embedding 模型与 endpoint、API key 和 Qdrant 位置，并在需要时配置共享 Hindsight 服务。终端会显示 endpoint URL 和非敏感配置预览；API key 使用隐藏输入。Setup 会写入仅当前用户可读写的 `.env` 和所选 client 配置。目录中已有的 `AGENTS.md` 会保留。

保存的 workspace profile 也会放进 MCP server 初始化 instructions，让连接的 Agent 从当前 client 收到相同上下文。

默认配置 Codex。其他 MCP client 使用 `--skip-codex-config`；Setup 会打印需要添加到该 client 设置中的 `command` 和 `args`。所有兼容 stdio MCP 的 Agent 都可以连接同一个 AMRB server。

macOS 使用用户级 LaunchAgent 管理 Hindsight；Linux 使用 `systemd --user`。服务凭据保存在权限受限的 `~/.config/amrb/hindsight.env`。

脚本化 setup 可从进程环境变量或 workspace `.env` 读取必需设置：

```bash
uv --directory /absolute/path/to/amrb run amrb setup \
  --workspace /absolute/path/to/workspace \
  --name "Project name" --domain "Project domain" \
  --purpose "Workspace purpose" \
  --source-scope "PDF collection to retrieve from" \
  --memory-policy "Durable knowledge to retain" --non-interactive
```

## 检索与记忆

检索层使用 Qdrant 索引 PDF。MCP 工具可以搜索文档、查看索引清单、连续读取 PDF 页面。检索结果包含来源和 PDF 页序。

Hindsight 在独立 bank 中保存长期 workspace 知识。`retain` 记录用户明确表达的决策、假设、概念和偏好；`recall` 找回已有知识；`reflect` 综合其演化。Workspace 规则描述当前 Agent 如何使用这些工具。

每个项目或知识领域使用一个 workspace。Workspace 身份和 profile 保存在 `amrb.toml`；API key 和服务连接设置保存在 `.env`。生成的工作规则在普通 session 之间保持稳定；项目用途变化时，更新 `amrb.toml` 中的 profile 和 `AGENTS.md` 中的工作规则。

## 接入其他 MCP client

使用目标 workspace 启动 server：

```bash
uv --directory /absolute/path/to/amrb run amrb-mcp --workspace /absolute/path/to/workspace
```

Setup 结束时也会打印机器可读的启动配置。将其中的 `command` 和 `args` 写入当前 client 的 MCP 设置，然后重新连接 client。

## 检查与维护

```bash
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace status --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace doctor --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace paper list --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace paper add ~/Downloads/document.pdf
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace paper sync
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace service status --json
```

`paper list` 会显示 PDF 标题、页数、稳定 ID 和索引状态。`status` 与 `doctor` 会显示 workspace 身份、collection、memory bank、文档与 chunk 数量，以及服务状态。

Hindsight Control Plane 用于查看记忆条目；Qdrant Dashboard 用于查看远端向量基础设施。Setup 可以记录这些 URL 并检查可访问性。本地嵌入式 Qdrant 将向量保存在 workspace 中。
