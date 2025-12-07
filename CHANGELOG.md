# Changelog

本文档记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/spec/v2.0.0.html) 规范。

## [0.5.0] - 2025-12-04

### 新增

- 多数据源并行抓取：使用 ThreadPoolExecutor 实现并发获取，显著提升抓取效率
- 并行抓取配置常量：`DEFAULT_MAX_WORKERS`（默认 5）和 `DEFAULT_TIMEOUT_PER_SOURCE`（默认 300 秒）
- BaseEmbeddingProvider 新增 `encode_query()` 方法，支持查询专用编码（适配 Voyage 的 `input_type="query"`）
- 期刊白名单新增：International Journal of Digital Earth
- API 连接测试脚本新增 DashScope 支持，支持动态环境变量检测

### 变更

- **破坏性变更**：配置项 `scoring.interests.top_k_recall` 重命名为 `max_documents`，更清晰地表达其含义——重排序的最大文档数量，且不得超过 API 限制（Voyage: 1000，DashScope: 500）
- 重排序器新增文档数量校验，超出 API 限制时抛出 `ValueError`
- 重构 `BaseReranker`：引入 `_rerank_batch()` 内部方法，自动进行限制校验
- 简化重排序 API：移除 `RerankResult` 数据类和 `rerank_with_details()` 方法
- 改进并行抓取的超时处理：优化 `as_completed()` 的超时管理机制
- GitHub Actions 缓存策略优化：在缓存键中加入运行编号，提升缓存命中率

**迁移指南：**

更新 `config.yaml` 中的配置项名称：

```yaml
# 旧配置
scoring:
  interests:
    enabled: true
    top_k_recall: 50  # 需要重命名

# 新配置
scoring:
  interests:
    enabled: true
    max_documents: 500  # 已重命名，数值不得超过 API 限制
```

**API 限制说明：**
- Voyage 重排序：最多 1000 篇文档
- DashScope 重排序：最多 500 篇文档

## [0.4.0] - 2025-12-01

### 变更

- **破坏性变更**：移除 `scoring.rerank.enabled` 配置项。重排序功能现在自动跟随 `scoring.interests.enabled` 的状态启用或禁用，简化配置逻辑
- **性能优化**：InterestRanker 现使用 `CachingEmbeddingProvider` 与 ProfileRanker 共享嵌入向量缓存，启用兴趣推荐时 API 调用减少约 50%，执行时间缩短约 45%
- **文档更新**：README 中补充说明嵌入向量提供商/模型变更会被自动检测并触发配置重建，无需手动删除缓存

**迁移指南：**

从 `config.yaml` 中移除 `enabled` 字段：

```yaml
# 旧配置
scoring:
  rerank:
    enabled: true  # 移除此行
    provider: "voyage"
    model: "rerank-2.5"

# 新配置
scoring:
  rerank:
    provider: "voyage"
    model: "rerank-2.5"
```

**影响说明：**
- 提供商匹配校验仅在 `interests.enabled=true` 时执行
- `interests.enabled=false` 时不再要求提供商必须匹配
- 消除了未使用的重排序配置导致的校验错误

## [0.3.0] - 2025-11-29

### 新增

- RSS 输出支持 Dublin Core 和 PRISM 命名空间，提升 Zotero 兼容性
- HTML 报告新增发表年份分布图表
- 非英文论文标题翻译功能
- 动态阈值配置：`scoring.thresholds.mode: dynamic`，支持基于分数分布自动划分论文等级
- 研究者画像分析：使用 LLM 生成研究兴趣洞察
- 用户报告中显示文献库收录时间跨度
- 基于兴趣描述的论文推荐（InterestRanker）
- 期刊影响因子评分集成
- 主题式摘要分组（TopicSummary 模型）
- ScienceDirect 摘要提取：从嵌入式 JSON 中解析
- Zotero 数据导入进度回调

### 变更

- 重构 watch 命令为 WatchPipeline 架构，提升代码可维护性
- 类型注解升级为 Python 3.10+ 语法（`list`、`dict`、`X | None`）
- 优化 HTML 模板布局：图表改为双列显示，年份分布图改为全宽
- 改进兴趣精炼流程：增加 exclude_keywords 日志记录
- 配置生成时间戳支持时区转换

### 移除

- 废弃的测试脚本和 pytest 配置
- 遗留的测试工具和 fixtures

## [0.2.0] - 2025-11-26

### 新增

- OpenRouter API 集成：支持 AI 摘要生成
- 统一嵌入向量缓存系统（EmbeddingCache）
- Camoufox 浏览器自动化摘要抓取
- Kimi（Moonshot AI）API 支持：用于 LLM 摘要生成
- 元数据缓存：改进摘要存储机制
- 期刊白名单：支持 ISSN 过滤

### 变更

- 重构 CLI 命令结构，提升用户交互体验
- 简化评分逻辑：聚焦嵌入向量相似度
- 摘要抓取改为顺序执行，增加速率限制

### 移除

- 移除遗留的 OpenAlex 和 bioRxiv 数据源
- 移除可选的期刊指标评分
- 移除 Semantic Scholar 集成（由 Camoufox 抓取器替代）

## [0.1.0] - 初始版本

### 新增

- ZotWatch 初始实现（基于 [Yorks0n/ZotWatch](https://github.com/Yorks0n/ZotWatch) 修改）
- Zotero 文献库集成
- Crossref 和 arXiv 论文抓取
- 基于 FAISS 的语义相似度搜索
- 基础 RSS 和 HTML 输出生成
