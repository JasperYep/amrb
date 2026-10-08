# AMRB 研究工作区 MCP

[English](README.md) | 简体中文

AMRB 将一个研究项目接入同一个 MCP server，用于检索论文证据和管理长期研究记忆。MCP client 负责对话与推理；AMRB 不运行 Agent 循环，也不提供聊天界面。

```text
MCP client ── AMRB ──┬── Paper RAG / Qdrant：外部论文证据
                     └── Hindsight：研究者的长期判断
```

一个 workspace 对应一个研究项目。`amrb.toml` 保存非敏感的项目身份和数据目标；`.env` 保存 API 密钥和部署连接设置。不同 workspace 默认使用不同的 collection 和 Hindsight bank。

## 安装与初始化

安装一次 AMRB，然后在研究项目目录中初始化 workspace：

```bash
git clone https://github.com/JasperYep/amrb.git
cd amrb
uv sync --locked
uv run amrb setup --workspace /absolute/path/to/research-project
```

从源码运行时，也可以使用：

```bash
uv --directory /absolute/path/to/amrb run amrb setup --workspace /absolute/path/to/research-project
```

Setup 会询问项目名称、研究领域、可选的研究焦点、embedding endpoint 和密钥，以及使用本地还是远端 Qdrant。也可以填写 Hindsight Control Plane 和远端 Qdrant Dashboard 的 URL；setup 会检查已配置链接是否可访问。API 密钥和 URL 通过终端隐藏输入，不要放进 MCP 对话。Setup 会创建 `amrb.toml`、仅当前用户可读写的 `.env`、Codex MCP 配置，以及（仅当文件不存在时）通用的 `AGENTS.md` 研究规则。已有的 `AGENTS.md`、其他 MCP 配置项和未知 Codex 配置字段都会保留。

脚本化运行时，通过进程环境变量或 workspace 的 `.env` 提供必需设置：

```bash
uv --directory /absolute/path/to/amrb run amrb setup \
  --workspace /absolute/path/to/research-project \
  --name "Research project" --discipline "Your field" --non-interactive
```

如果 setup 发现尚未配置的论文或索引数据，请先检查目标资源，再显式传入 `--adopt-legacy`。缺少模型和切分配置的旧索引会保持“未验证”状态，检查配置并确认后再使用 `--accept-legacy-index`。Setup 会保留旧 paper ID、manifest、collection、bank 和共享 Hindsight 服务。

Workspace 配置会记录创建身份时所在的路径。如果配置被复制或移动，setup 会询问是继续使用原 collection 和 bank（`--copied-identity reuse`），还是创建新身份（`--copied-identity new`）。新身份会使用独立的 collection、bank 和本地 Qdrant 目录；它会归档复制来的 manifest，保留 PDF 和原有向量文件，不会重新建索引。对于归属不明且已有数据的 collection 或 Hindsight bank，必须通过 `--adopt-existing-resources` 明确确认。

## MCP 客户端

Codex 是首选引导客户端。Setup 会在 `.codex/config.toml` 中写入绑定当前 workspace 的 stdio 命令，并固定 AMRB 安装路径和 workspace 路径，因此 MCP 启动不依赖客户端的当前工作目录。其他 MCP 客户端可以这样启动同一 server：

```bash
uv --directory /absolute/path/to/amrb run amrb-mcp --workspace /absolute/path/to/research-project
```

MCP 提供 `search_papers`、`list_papers`、`get_paper_pages`、`retain`、`recall` 和 `reflect`。论文检索结果会标明论文、检索方式和 PDF 页序。Hindsight 工具在 server 启动时绑定 workspace 对应的 bank；每次工具调用都不需要传入项目或 bank 选择器。

## 检查与维护

```bash
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project status --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project doctor --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project paper list --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project paper add ~/Downloads/paper.pdf
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project paper sync
```

`paper list` 会显示每篇 PDF 的标题、页数、稳定 paper ID 和索引状态：已索引、未索引、源文件变更、文件缺失、索引失败或未验证。JSON 输出可供脚本读取。`status` 和 `doctor` 会显示 workspace、bank、collection、论文数、chunk 数和服务状态。

Hindsight Control Plane 是检查记忆的官方界面。Qdrant Dashboard 用于查看远端 Qdrant 基础设施，不能代替论文清单或检索结果。这些官方界面可能允许修改或删除数据；打开界面不代表它们是只读的。本地嵌入式 Qdrant 没有可打开的服务端 Dashboard。

AMRB 不会自动导入 PDF、保存全部对话记录，也不会把检索到的论文内容自动写入个人 Memory。Workspace 协作规则会说明何时使用 Paper RAG 和 Hindsight，但这些规则不会在 MCP server 端强制限制写入权限。

## 配置

`amrb.toml` 保存 workspace ID、项目说明、论文目录、embedding 模型与切分配置、Qdrant collection 和 Hindsight bank。Workspace ID 只在初始化时生成一次；重命名项目不会改变数据身份。相对数据路径以 workspace 根目录为基准解析。

`.env` 保存密钥和服务连接设置，并从版本控制中排除。生成的秘密文件只允许当前用户读写。Workspace 身份和已记录的 embedding / 切分配置以 `amrb.toml` 为准；部署设置按进程环境变量、workspace `.env`、默认值的顺序解析。AMRB 不会从安装目录加载 `.env`。

Hindsight 是共享的用户服务。新 workspace 会复用健康的服务，并使用独立 bank。Setup 不会改写服务凭据、重启健康的共享服务，也不会替换已有 bank 的 missions。

如果 bank 已存在，setup 会显示 mission 差异，并在更新前征求确认。脚本运行时，只有检查过拟更新内容后才使用 `--update-missions`。服务日志会对已配置的密钥和服务 URL 脱敏。Setup 失败时会说明已保存的内容和恢复步骤。
