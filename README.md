# ZotWatch

ZotWatch 是一个基于 Zotero 文库构建个人研究兴趣画像，并持续监测学术信息源的智能文献推荐系统。支持 AI 摘要生成、增量嵌入计算，可在本地手动执行或通过 GitHub Actions 自动运行。

## 致谢

本项目受 [Yorks0n/ZotWatch](https://github.com/Yorks0n/ZotWatch) 启发，并在此基础上进行了修改和调整。

## 部署示例

- **在线演示**: [https://ehehe.cn/zotwatch/](https://ehehe.cn/zotwatch/)
- **RSS 订阅**: [https://ehehe.cn/zotwatch/feed.xml](https://ehehe.cn/zotwatch/feed.xml)

## 功能概览

- **Zotero 同步**：通过 Zotero Web API 获取文库条目，支持增量更新
- **智能画像构建**：使用 Voyage AI 向量化条目，支持增量嵌入计算（仅处理新增/变更条目）
- **多源候选抓取**：支持 Crossref、arXiv 数据源
- **智能评分排序**：结合语义相似度、时间衰减、引用指标、期刊质量及白名单加分
- **兴趣驱动推荐**：基于用户描述的研究兴趣，使用 Voyage Rerank 进行语义重排序
- **AI 摘要生成**：支持 Kimi (Moonshot AI) 和 OpenRouter (Claude 等) 两种 LLM 提供商
- **标题翻译**：自动将英文论文标题翻译为中文
- **研究画像分析**：自动分析文库，生成研究领域分类和洞察
- **多格式输出**：RSS 订阅、响应式 HTML 报告、推送回 Zotero

## 快速开始

### 1. 克隆仓库并安装依赖

```bash
git clone <your-repo-url>
cd ZotWatch
uv sync
```

### 2. 安装 Camoufox 浏览器

ZotWatch 使用 [Camoufox](https://github.com/nicholaswan/camoufox)（基于 Firefox 的反检测浏览器）从受 Cloudflare 保护的出版商网站抓取论文摘要。安装依赖后需要下载浏览器二进制文件：

```bash
uv run python -m camoufox fetch
```

> **注意**：首次下载约需 1-2 分钟，浏览器文件约 200MB。GitHub Actions 会自动处理此步骤并缓存。

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

| 变量名 | 必需 | 说明 | 获取地址 |
|--------|------|------|----------|
| `ZOTERO_API_KEY` | ✅ | Zotero API 密钥 | [Zotero API Keys](https://www.zotero.org/settings/keys) |
| `ZOTERO_USER_ID` | ✅ | Zotero 用户 ID（在 API 密钥页面可见） | 同上 |
| `VOYAGE_API_KEY` | ⚠️ | Voyage AI API 密钥（用于文本嵌入和重排序） | [Voyage AI](https://dash.voyageai.com/) |
| `DASHSCOPE_API_KEY` | ⚠️ | 阿里云 DashScope API 密钥（嵌入和重排序的备选） | [阿里云百炼平台](https://bailian.console.aliyun.com/?tab=model#/api-key) |
| `MOONSHOT_API_KEY` | ⚠️ | Kimi (Moonshot AI) API 密钥 | [Moonshot AI](https://platform.moonshot.cn/) |
| `OPENROUTER_API_KEY` | ⚠️ | OpenRouter API 密钥（支持 Claude 等模型） | [OpenRouter](https://openrouter.ai/keys) |
| `CROSSREF_MAILTO` | 推荐 | Crossref 礼貌池邮箱 | 你的邮箱地址 |

> **注意**：
> - **嵌入提供商**：`VOYAGE_API_KEY` 和 `DASHSCOPE_API_KEY` 二选一，用于文本嵌入和重排序。默认使用 Voyage AI，可在 `config/config.yaml` 中切换为 DashScope。
> - **LLM 提供商**：`MOONSHOT_API_KEY` 和 `OPENROUTER_API_KEY` 至少需要配置其中一个，用于 AI 摘要生成和标题翻译功能。默认使用 Kimi。

### 4. 运行

```bash
# 首次全量画像构建
uv run zotwatch profile --full

# 日常监测（生成 RSS + HTML 报告 + AI 摘要）
uv run zotwatch watch
```

详细命令参数请参考下方 [CLI 命令](#cli-命令) 章节。

## 配置指南

ZotWatch 的所有配置集中在 `config/config.yaml`，支持环境变量替换（`${VAR_NAME}` 语法）。本节提供常见配置场景的详细说明。

### 文本嵌入和重排序提供商配置

ZotWatch 支持两种文本嵌入和重排序提供商：**Voyage AI** 和 **DashScope（阿里云）**。

> **注意**：当兴趣驱动推荐（`scoring.interests.enabled`）开启时，嵌入提供商（`embedding.provider`）和重排序提供商（`scoring.rerank.provider`）必须使用相同的提供商。两者共享同一个 API 密钥，不支持混合使用。

#### 使用 Voyage AI（默认）

Voyage AI 提供高质量的英文文本嵌入，适合国际学术论文推荐。

**步骤 1：配置环境变量**

在 `.env` 文件中设置 Voyage AI API 密钥：

```bash
VOYAGE_API_KEY=your_voyage_api_key_here
```

获取 API 密钥：[Voyage AI Dashboard](https://dash.voyageai.com/)

**步骤 2：配置 config.yaml**

```yaml
# Embedding configuration
embedding:
  provider: "voyage"
  model: "voyage-3.5"
  api_key: "${VOYAGE_API_KEY}"
  batch_size: 128

# Scoring configuration
scoring:
  rerank:
    provider: "voyage"  # 必须与 embedding.provider 一致
    model: "rerank-2.5"
```

**可用模型**：
- 嵌入模型：`voyage-3.5`（1024 维，推荐）
- 重排序模型：`rerank-2`、`rerank-2.5`（推荐）

#### 使用 DashScope（阿里云）

DashScope 提供中文语义理解优化的嵌入模型，适合中文或中英文混合场景。

**步骤 1：配置环境变量**

在 `.env` 文件中设置 DashScope API 密钥：

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

获取 API 密钥：[阿里云百炼平台](https://bailian.console.aliyun.com/?tab=model#/api-key)

**步骤 2：配置 config.yaml**

```yaml
# Embedding configuration
embedding:
  provider: "dashscope"
  model: "text-embedding-v4"
  api_key: "${DASHSCOPE_API_KEY}"
  batch_size: 10  # DashScope 要求 batch_size ≤ 10

# Scoring configuration
scoring:
  rerank:
    provider: "dashscope"  # 必须与 embedding.provider 一致
    model: "qwen3-rerank"
```

**可用模型**：
- 嵌入模型：`text-embedding-v4`（1024 维）
- 重排序模型：`qwen3-rerank`

> [!IMPORTANT]
> - DashScope 的 `batch_size` 必须 ≤ 10（阿里云 API 限制）
> - Voyage AI 可使用更大的 batch_size（如 128）

**步骤 3：运行 watch 命令**

切换提供商后，系统会自动检测嵌入提供商/模型变更并重新构建画像：

```bash
uv run zotwatch watch
```

> **说明**：ZotWatch 会自动检测 `embedding.provider` 或 `embedding.model` 的变更，并在首次运行时自动触发全量画像重建，无需手动删除缓存文件。

### 大语言模型（LLM）提供商配置

ZotWatch 使用 LLM 生成论文摘要和翻译标题，支持两种提供商：**Kimi（Moonshot AI）** 和 **OpenRouter**。

#### 使用 Kimi（默认）

Kimi 是国内 Moonshot AI 提供的大语言模型，支持中文优化，响应速度快。

**步骤 1：配置环境变量**

```bash
MOONSHOT_API_KEY=your_moonshot_api_key_here
```

获取 API 密钥：[Moonshot AI 平台](https://platform.moonshot.cn/)

**步骤 2：配置 config.yaml**

```yaml
llm:
  enabled: true
  provider: "kimi"
  api_key: "${MOONSHOT_API_KEY}"
  model: "kimi-k2-turbo-preview"  # 标准模型
  max_tokens: 5120
  temperature: 0.3
  retry:
    max_attempts: 3
    backoff_factor: 2.0
    initial_delay: 1.0
  summarize:
    top_n: 20                # 为前 20 篇论文生成摘要
    cache_expiry_days: 30    # 摘要缓存有效期
  translation:
    enabled: true            # 启用标题翻译
```

**可用模型**：

| 模型 | 适用场景 | max_tokens | temperature |
|------|---------|------------|-------------|
| `kimi-k2-turbo-preview` | 摘要生成、翻译（推荐） | 5120 | 0.3 |
| `kimi-k2-thinking-turbo` | 复杂推理任务 | ≥16000 | 1.0 |

#### 使用 OpenRouter

OpenRouter 支持多种模型（如 Claude、GPT-4 等），适合需要特定模型的场景。

**步骤 1：配置环境变量**

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

获取 API 密钥：[OpenRouter Keys](https://openrouter.ai/keys)

**步骤 2：配置 config.yaml**

```yaml
llm:
  enabled: true
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3.5-sonnet"
  max_tokens: 8000
  temperature: 0.6
  retry:
    max_attempts: 3
    backoff_factor: 2.0
    initial_delay: 1.0
  summarize:
    top_n: 20
    cache_expiry_days: 30
  translation:
    enabled: true
```

**可用模型**：
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4-turbo`
- 更多模型请查看 [OpenRouter Models](https://openrouter.ai/models)

### 阈值模式配置

阈值模式控制如何将论文分类为 `must_read`（必读）、`consider`（可考虑）或 `ignore`（忽略）。

#### 固定阈值模式（Fixed）

使用静态阈值值，适合分数分布稳定的场景。

```yaml
scoring:
  thresholds:
    mode: "fixed"
    must_read: 0.75  # 分数 ≥ 0.75 的论文标记为 must_read
    consider: 0.55   # 分数 ≥ 0.55 但 < 0.75 的论文标记为 consider
```

#### 动态阈值模式（Dynamic，默认）

基于当前批次的分数分布动态计算阈值，适合分数波动较大的场景。

```yaml
scoring:
  thresholds:
    mode: "dynamic"
    must_read: 0.75  # 固定模式的备用值
    consider: 0.55   # 固定模式的备用值
    dynamic:
      must_read_percentile: 95  # 前 5% 的论文标记为 must_read
      consider_percentile: 70   # 70-95 百分位的论文标记为 consider
      min_must_read: 0.60       # 即使在前 5%，分数 < 0.60 也不标记为 must_read
      min_consider: 0.40        # 即使在 70 百分位，分数 < 0.40 也不标记为 consider
```

**动态模式优势**：
- 自动适应不同批次的分数分布
- 确保相对排名一致（始终保留前 5% 作为必读）
- 避免因绝对分数波动导致的推荐质量下降

### 摘要抓取器（Scraper）配置

摘要抓取器用于从出版商网站抓取缺失的论文摘要，使用 Camoufox 浏览器绕过 Cloudflare 防护。

```yaml
sources:
  scraper:
    enabled: true             # 启用摘要抓取
    rate_limit_delay: 1.0     # 请求间隔（秒）
    timeout: 60000            # 页面加载超时（毫秒）
    max_retries: 2            # 每个 URL 的重试次数
    max_html_chars: 15000     # 发送给 LLM 的最大 HTML 字符数
    llm_max_tokens: 1024      # LLM 提取响应的最大 token 数
    llm_temperature: 0.1      # LLM 温度（低温度确保准确提取）
    use_llm_fallback: true    # 规则提取失败时使用 LLM 后备
```

**调优建议**：
- **rate_limit_delay**：如遇到频繁封禁，可增加到 2.0-3.0 秒
- **timeout**：网络较慢时可增加到 90000 或 120000
- **use_llm_fallback**：禁用可节省 LLM API 调用，但可能导致部分摘要抓取失败

### 兴趣驱动推荐配置

基于用户描述的研究兴趣，使用语义重排序筛选最相关的论文。

```yaml
scoring:
  interests:
    enabled: true
    description: |
      我现在重点关注下面的研究方向：
      1) SAR 图像目标识别、检测、跟踪，尤其针对舰船目标
      2) 多源传感器融合，尤其是 SAR 与红外的融合
      3) 视觉基础模型预训练（例如：CLIP、MAE、DINO、JEPA 等相关方法）
      4) 大语言模型的最新进展
      5) 希望排除生物医学领域的研究
    top_k_recall: -1    # FAISS 召回数量，-1 表示跳过 FAISS 使用所有候选
    top_k_interest: 5   # 最终兴趣驱动推荐的论文数量
```

**参数说明**：
- **description**：用自然语言描述你的研究兴趣，支持中英文。LLM 会自动提炼关键词和排除关键词。
- **top_k_recall**：设为 `-1` 跳过 FAISS 召回，直接使用所有候选论文进行重排序（推荐）。设为正整数（如 `100`）时先通过 FAISS 召回前 N 篇再重排序。
- **top_k_interest**：最终返回的兴趣驱动推荐论文数量。

**禁用兴趣驱动推荐**：

```yaml
scoring:
  interests:
    enabled: false
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
- 自动翻译论文标题（如已启用）
- 推荐数量默认 20 篇

## 目录结构

```
ZotWatch/
├── src/zotwatch/           # 主包
│   ├── core/               # 核心模型和协议
│   ├── config/             # 配置管理
│   ├── infrastructure/     # 存储、嵌入、HTTP 客户端
│   │   ├── storage/        # SQLite 存储
│   │   ├── embedding/      # Voyage AI 嵌入 + FAISS 索引 + 重排序
│   │   └── enrichment/     # 摘要抓取器（Camoufox + LLM）
│   ├── sources/            # 数据源（arXiv、Crossref、Zotero）
│   ├── llm/                # LLM 集成（Kimi、OpenRouter）
│   ├── pipeline/           # 处理管道
│   ├── output/             # 输出生成（RSS、HTML、Zotero 推送）
│   └── cli/                # Click CLI
├── config/
│   └── config.yaml         # 统一配置文件
├── data/                   # 画像/缓存（不纳入版本控制）
├── reports/                # 生成的 RSS/HTML 输出
└── .github/workflows/      # GitHub Actions 配置
```

## 配置说明

所有配置集中在 `config/config.yaml`，支持环境变量替换（`${VAR_NAME}` 语法）。可配置项包括：

- **数据源**：arXiv/Crossref 的类别、时间范围、抓取数量
- **LLM 提供商**：Kimi 或 OpenRouter，模型选择、参数调整
- **评分阈值**：固定或动态阈值模式
- **兴趣驱动推荐**：自定义研究兴趣描述
- **输出设置**：时区、RSS 信息等

## 处理管道

ZotWatch 的 `watch` 命令执行以下处理流程：

1. **画像检查**：如果不存在画像，自动从 Zotero 构建
2. **增量同步**：从 Zotero 同步最新条目
3. **研究画像分析**：使用 LLM 分析文库，生成领域分类和洞察
4. **候选抓取**：从 arXiv、Crossref 获取候选论文
5. **摘要补全**：使用 Camoufox 从出版商网站抓取缺失摘要
6. **去重过滤**：过滤已在文库中的论文
7. **兴趣匹配**（可选）：基于用户兴趣描述进行语义重排序
8. **相似度排序**：使用 FAISS 索引计算与文库的语义相似度
9. **应用过滤**：时间窗口、预印本比例、数量限制
10. **AI 摘要**：为推荐论文生成结构化摘要
11. **标题翻译**：将英文标题翻译为中文
12. **输出生成**：生成 RSS、HTML 报告

## GitHub Actions 自动运行

通过 GitHub Actions 实现每日自动监测和推送，无需本地运行。

### 1. Fork 仓库

点击 GitHub 页面右上角的 **Fork** 按钮。

### 2. 配置 Secrets

在你的仓库中进入 **Settings → Secrets and variables → Actions → New repository secret**，添加以下密钥：

| Secret 名称 | 必需 | 说明 |
|------------|------|------|
| `ZOTERO_API_KEY` | ✅ | Zotero API 密钥 |
| `ZOTERO_USER_ID` | ✅ | Zotero 用户 ID |
| `VOYAGE_API_KEY` | ⚠️ | Voyage AI API 密钥（嵌入提供商，二选一） |
| `DASHSCOPE_API_KEY` | ⚠️ | 阿里云 DashScope API 密钥（嵌入提供商，二选一） |
| `MOONSHOT_API_KEY` | 推荐 | Kimi API 密钥（默认 LLM 提供商） |
| `OPENROUTER_API_KEY` | 可选 | OpenRouter API 密钥（备选 LLM 提供商） |
| `CROSSREF_MAILTO` | 推荐 | 你的邮箱，用于 Crossref 礼貌池 |
| `DEPLOY_KEY` | 可选 | SSH 私钥，用于部署到外部仓库 |

> **注意**：`VOYAGE_API_KEY` 和 `DASHSCOPE_API_KEY` 二选一配置即可，取决于你在 `config/config.yaml` 中选择的 `embedding.provider`。

### 3. 启用 GitHub Pages（可选）

如果部署到同一仓库：
1. 进入 **Settings → Pages**
2. **Source** 选择 **GitHub Actions**
3. 保存设置

### 4. 首次运行

1. 进入 **Actions** 标签页
2. 点击左侧 **Daily Watch & Deploy**
3. 点击 **Run workflow** 手动触发首次运行
4. 首次运行约需 5-10 分钟（构建画像 + 安装浏览器）

### 5. 访问结果

运行成功后，可通过以下地址访问：

- **RSS 订阅**：`https://[username].github.io/[repo]/feed.xml`
- **HTML 报告**：`https://[username].github.io/[repo]/report.html`

### 6. 自动运行

- Workflow 默认每天北京时间 **8:25** 自动运行
- 可在 `.github/workflows/daily_watch.yml` 中修改 cron 表达式调整时间
- 支持随时手动触发

### 运行时间说明

| 阶段 | 首次运行 | 后续运行 |
|------|---------|---------|
| 依赖安装 | ~1 分钟 | ~10 秒（有缓存） |
| Camoufox 安装 | ~2 分钟 | ~10 秒（有缓存） |
| Zotero 同步 | ~2 分钟 | ~10 秒（增量） |
| 画像构建 | ~30 秒 | 跳过（有缓存） |
| 候选抓取 | ~10 秒 | ~10 秒 |
| 摘要补全（Scraper） | ~15 分钟 | ~1-5 分钟（有缓存） |
| 评分排序 | ~30 秒 | ~30 秒 |
| AI 摘要生成 | ~4 分钟 | ~1 分钟（有缓存） |
| 标题翻译 | ~20 秒 | ~20 秒 |
| **总计** | **~25 分钟** | **~5-10 分钟** |

> **说明**：摘要补全阶段耗时取决于需要抓取的论文数量。上表基于约 100 篇需抓取摘要、25 篇生成 AI 摘要的典型场景。后续运行因缓存机制，实际耗时会显著减少。

## 数据文件

ZotWatch 在 `data/` 目录下存储以下文件：

| 文件 | 说明 | 版本控制 |
|------|------|----------|
| `journal_whitelist.csv` | Crossref 期刊白名单（ISSN、期刊名、类别、影响因子） | ✅ 已纳入 |
| `profile.sqlite` | Zotero 条目和元数据 | ❌ |
| `faiss.index` | FAISS 向量索引 | ❌ |
| `embeddings.sqlite` | 嵌入向量缓存 | ❌ |
| `metadata.sqlite` | 抓取的摘要缓存 | ❌ |

### 期刊白名单

`data/journal_whitelist.csv` 用于筛选 Crossref 数据源的期刊。只有在此白名单中的期刊才会被抓取。文件格式：

```csv
issn,title,category,impact_factor
0162-8828,IEEE Trans. Pattern Analysis and Machine Intelligence,AI/ML,18.60
0196-2892,IEEE Trans. Geoscience and Remote Sensing,RS,8.60
```

你可以根据自己的研究领域编辑此文件，添加或删除期刊。

## 常见问题

**Q: 如何强制重新计算所有嵌入？**
```bash
uv run zotwatch profile --full
```

**Q: 推荐为空？**

检查以下可能原因：
- 所有候选都超出时间窗口（默认 7 天）
- 预印本比例限制（默认 0.9）
- 没有摘要的论文被过滤

可调节 `--top` 参数或修改 `config/config.yaml` 中的相关阈值。

**Q: AI 摘要不生成？**

确保已配置 `MOONSHOT_API_KEY` 或 `OPENROUTER_API_KEY`，且 `config.yaml` 中 `llm.enabled: true`。

**Q: 如何添加或删除监测的期刊？**

编辑 `data/journal_whitelist.csv` 文件，添加或删除期刊的 ISSN。

## 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。
