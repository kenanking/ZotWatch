"""Prompt templates for LLM summarization."""

BULLET_SUMMARY_PROMPT = """请分析以下学术论文并提供简明扼要的摘要。

论文标题：{title}
摘要：{abstract}
作者：{authors}
期刊/会议：{venue}

请提供以下 5 个要点的总结：
1. **研究问题**：本文解决什么问题？
2. **研究方法**：采用了什么方法或技术？
3. **主要发现**：核心研究结果是什么？
4. **创新点**：本文的创新之处在哪里？
5. **相关性说明**：为什么这项研究对相关领域的研究者有价值？

请以 JSON 格式返回，包含以下键：research_question, methodology, key_findings, innovation, relevance_note
每个值应为简洁的单句话（不超过 50 字）。

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。"""

DETAILED_ANALYSIS_PROMPT = """请对以下学术论文进行详细分析。

论文标题：{title}
摘要：{abstract}
作者：{authors}
期刊/会议：{venue}

请撰写全面的分析，涵盖以下方面：

1. **研究背景**：本研究的背景和动机（2-3 句话）
2. **方法详情**：研究方法的详细说明（3-4 句话）
3. **研究结果**：主要发现及其意义（3-4 句话）
4. **局限性**：已知的局限性或潜在问题（2-3 句话）
5. **未来方向**：可能的后续研究方向（1-2 句话）
6. **研究相关性**：为什么这篇论文可能与研究类似课题的研究者相关

请以 JSON 格式返回，包含以下键：background, methodology_details, results, limitations, future_directions, relevance_to_interests

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。"""

INTEREST_REFINEMENT_PROMPT = """You are an academic research assistant. Based on the user's research interest description, generate an optimized search query for finding relevant academic papers.

User's research interests:
{user_interests}

Please output a JSON object with the following fields:
{{
  "refined_query": "A refined English query for retrieving relevant papers (50-100 words, covering key research topics, methods, and applications)",
  "include_keywords": ["keyword1", "keyword2", ...],
  "exclude_keywords": ["exclude1", "exclude2", ...]
}}

Guidelines:
- The refined_query should be a comprehensive English description suitable for semantic search
- include_keywords should contain 5-10 important technical terms
- exclude_keywords rules (VERY IMPORTANT - be conservative):
  * ONLY include keywords if the user EXPLICITLY mentions wanting to exclude a specific domain/topic
  * Return an empty array [] if no exclusions are explicitly mentioned

Important: Only return the JSON object, no additional text or markdown formatting."""

OVERALL_SUMMARY_PROMPT = """请根据以下学术论文列表，按研究主题进行分组并撰写总结。

论文数量：{paper_count}
论文类型：{section_type}

论文列表：
{papers_list}

请完成以下任务：
1. 根据论文内容，将论文分成若干个研究主题（数量由你根据内容自动判断）
2. 撰写一句概述性的开头，说明各主题的论文数量分布
3. 对每个主题，用 1-2 句话描述该主题下论文的关键研究要点。在提到具体论文时，必须使用双书名号（《》）括起论文标题或其方法简称来指代该论文，不要使用不明确的或泛指的表述

请以 JSON 格式返回：
{{
  "overview": "本期推荐涵盖了X篇关于主题A的论文、Y篇关于主题B的论文...",
  "topics": [
    {{
      "topic_name": "主题名称（2-6字）",
      "paper_count": 数量,
      "description": "1-2句话描述该主题的研究重点和关键方法，提到论文时使用《论文标题》或《方法简称》"
    }}
  ]
}}

注意：
- 主题分组应覆盖所有论文
- overview 应简洁说明各主题论文数量分布
- description 应具体描述研究内容，而非泛泛而谈
- 提到论文时必须使用《论文标题》或《方法简称》格式，例如：《BERT》、《Attention Is All You Need》

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。"""

DOMAIN_CLASSIFICATION_PROMPT = """请分析以下学术论文列表，将它们分类到不同的研究领域。

论文列表（共{paper_count}篇）：
{papers_list}

请将论文分类到最合适的研究领域，要求：
1. 领域名称应简洁（2-6个中文字符）
2. 每个领域至少包含2篇论文
3. 最多不超过{max_domains}个领域
4. 每个领域提供3个代表性论文标题

请以JSON格式返回：
{{
  "domains": [
    {{
      "domain": "领域名称",
      "paper_count": 数量,
      "sample_titles": ["标题1", "标题2", "标题3"]
    }}
  ]
}}

重要：只返回JSON对象，不要添加任何额外文字或markdown格式。"""

PROFILE_ANALYSIS_PROMPT = """请根据以下研究者的**文献阅读和收藏记录**统计数据，分析其研究兴趣画像。

**重要说明**：以下数据来自用户的 Zotero 文献库收藏，反映的是用户的**阅读偏好和研究兴趣**，而非用户本人发表的论文。请基于此进行分析，不要将这些论文误解为用户的研究成果。

## 基础统计
- 收藏文献总数：{total_papers}篇
- 收藏时长：{collection_duration}
- 论文发表年份：{year_range}

## 阅读兴趣领域分布（Top 5）
{top_domains}

## 高频关注作者（文献库中频繁出现的作者，Top 10）
{top_authors}

## 常关注的期刊/会议（Top 10）
{top_venues}

## 高频关键词（Top 20）
{top_keywords}

## 近期阅读趋势（最近3年季度统计）
{quarterly_trends}

## 近期新增收藏特征
{recent_analysis}

请生成以下分析内容：

1. **研究兴趣概述**：概括用户关注的核心研究方向（2-3句话）
2. **深度关注领域**：分析用户在哪些领域有持续深入的阅读积累（2-3句话）
3. **跨学科阅读倾向**：分析用户的跨学科阅读特点（1-2句话）
4. **兴趣演变趋势**：根据近期阅读数据，分析研究兴趣的变化趋势（2-3句话）
5. **延伸阅读建议**：基于现有阅读兴趣，推荐可能感兴趣的研究方向（1-2句话）

请以JSON格式返回：
{{
  "research_focus_summary": "研究兴趣概述",
  "strength_areas": "深度关注领域",
  "interdisciplinary_notes": "跨学科阅读倾向",
  "trend_observations": "兴趣演变趋势",
  "recommendations": "延伸阅读建议"
}}

注意：
- 分析应基于实际数据，不要臆测
- 这是用户的阅读收藏记录，不是发表成果，请使用"关注"、"阅读"、"收藏"等措辞
- 语言应专业但易懂
- 建议应具体可行

重要：只返回JSON对象，不要添加任何额外文字或markdown格式。"""


__all__ = [
    "BULLET_SUMMARY_PROMPT",
    "DETAILED_ANALYSIS_PROMPT",
    "INTEREST_REFINEMENT_PROMPT",
    "OVERALL_SUMMARY_PROMPT",
    "DOMAIN_CLASSIFICATION_PROMPT",
    "PROFILE_ANALYSIS_PROMPT",
]
