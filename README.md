# ZotWatch

ZotWatch 是一个基于 Zotero 文库构建个人研究兴趣画像，并持续监测学术信息源的智能文献推荐系统。支持 AI 摘要生成、增量嵌入计算，可在本地手动执行或通过 GitHub Actions 自动运行。

## 功能概览

- **Zotero 同步**：通过 Zotero Web API 获取文库条目，支持增量更新
- **智能画像构建**：使用 Voyage AI 向量化条目，支持增量嵌入计算（仅处理新增/变更条目）
- **多源候选抓取**：支持 Crossref、arXiv、bioRxiv/medRxiv、OpenAlex 等数据源
- **智能评分排序**：结合语义相似度、时间衰减、引用指标、期刊质量及白名单加分
- **AI 摘要生成**：通过 OpenRouter API 调用 Claude 等模型，生成结构化论文摘要
- **多格式输出**：RSS 订阅、响应式 HTML 报告、推送回 Zotero

## 快速开始

### 1. 克隆仓库并安装依赖

```bash
git clone <your-repo-url>
cd ZotWatch
uv sync
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

必需的环境变量：
- `ZOTERO_API_KEY`：[Zotero API 密钥](https://www.zotero.org/settings/keys)
- `ZOTERO_USER_ID`：Zotero 用户 ID（在 API 密钥页面可见）
- `VOYAGE_API_KEY`：[Voyage AI API 密钥](https://dash.voyageai.com/)（用于文本嵌入）
- `OPENROUTER_API_KEY`：[OpenRouter API 密钥](https://openrouter.ai/keys)（用于 AI 摘要，可选）

可选的环境变量：
- `CROSSREF_MAILTO`：Crossref 礼貌池邮箱
- `OPENALEX_MAILTO`：OpenAlex 礼貌池邮箱

### 3. 运行

```bash
# 首次全量画像构建（计算所有条目的嵌入）
uv run zotwatch profile --full

# 增量更新画像（仅计算新增/变更条目的嵌入）
uv run zotwatch profile

# 日常监测（默认生成 RSS + HTML 报告 + AI 摘要，推荐 20 篇）
uv run zotwatch watch

# 只生成 RSS
uv run zotwatch watch --rss

# 只生成 HTML 报告
uv run zotwatch watch --report

# 自定义推荐数量
uv run zotwatch watch --top 50
```

## CLI 命令

### `zotwatch profile`

构建或更新用户研究画像。

```bash
zotwatch profile [OPTIONS]

Options:
  --full    全量重建（重新计算所有嵌入）
```

默认使用增量模式，仅对新增或内容变更的条目计算嵌入向量，大幅减少 API 调用。

### `zotwatch watch`

获取、评分并输出论文推荐。

```bash
zotwatch watch [OPTIONS]

Options:
  --rss        只生成 RSS 订阅源
  --report     只生成 HTML 报告
  --top N      保留前 N 条结果（默认 20）
  --push       推送推荐到 Zotero
```

默认行为：
- 同时生成 RSS 和 HTML 报告
- 自动为所有推荐论文生成 AI 摘要
- 推荐数量默认 20 篇

## 目录结构

```
ZotWatch/
├── src/zotwatch/           # 主包
│   ├── core/               # 核心模型和协议
│   ├── config/             # 配置管理
│   ├── infrastructure/     # 存储、嵌入、HTTP 客户端
│   │   ├── storage/        # SQLite 存储
│   │   └── embedding/      # Voyage AI + FAISS
│   ├── sources/            # 数据源（arXiv、Crossref 等）
│   ├── llm/                # LLM 集成（OpenRouter）
│   ├── pipeline/           # 处理管道
│   ├── output/             # 输出生成（RSS、HTML）
│   └── cli/                # Click CLI
├── config/
│   └── config.yaml         # 统一配置文件
├── data/                   # 画像/缓存（不纳入版本控制）
├── reports/                # 生成的 RSS/HTML 输出
└── .github/workflows/      # GitHub Actions 配置
```

## 配置说明

所有配置集中在 `config/config.yaml`：

```yaml
# Zotero API 设置
zotero:
  api:
    user_id: "${ZOTERO_USER_ID}"
    api_key: "${ZOTERO_API_KEY}"

# 数据源开关
sources:
  arxiv:
    enabled: true
    categories: ["cs.LG", "cs.CV", "cs.AI"]
  crossref:
    enabled: true
  # ...

# 评分权重
scoring:
  weights:
    similarity: 0.50
    recency: 0.15
    # ...

# 嵌入模型
embedding:
  provider: "voyage"
  model: "voyage-3.5"

# LLM 摘要
llm:
  enabled: true
  provider: "openrouter"
  model: "anthropic/claude-3.5-sonnet"
```

## GitHub Actions 自动运行

### 1. Fork 仓库

### 2. 配置 Secrets

在 **Settings → Secrets and variables → Actions** 添加：
- `ZOTERO_API_KEY`
- `ZOTERO_USER_ID`
- `VOYAGE_API_KEY`
- `OPENROUTER_API_KEY`（可选，用于 AI 摘要）
- `CROSSREF_MAILTO`（可选）

### 3. 启用 GitHub Pages

**Settings → Pages → Source** 设为 **GitHub Actions**。

### 4. 运行

- Workflow 默认每天自动运行
- RSS 地址：`https://[username].github.io/ZotWatch/feed.xml`

## 常见问题

**Q: 如何强制重新计算所有嵌入？**
```bash
uv run zotwatch profile --full
```

**Q: 缓存过旧怎么办？**

删除 `data/cache/candidate_cache.json` 强制刷新候选列表。

**Q: 推荐为空？**

检查是否所有候选都超出 7 天窗口或预印本比例被限制。可调节 `--top` 参数或修改 `config.yaml` 中的阈值。

**Q: AI 摘要不生成？**

确保 `OPENROUTER_API_KEY` 已配置，且 `config.yaml` 中 `llm.enabled: true`。

**Q: 如何禁用 AI 摘要？**

在 `config/config.yaml` 中设置 `llm.enabled: false`。

## License

MIT
