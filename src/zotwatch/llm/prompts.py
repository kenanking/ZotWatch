"""Prompt templates for LLM summarization."""

BULLET_SUMMARY_PROMPT = """请分析以下学术论文并提供简明扼要的摘要。

论文标题：{title}
摘要：{abstract}
作者：{authors}
发表场所：{venue}

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
发表场所：{venue}

请撰写全面的分析，涵盖以下方面：

1. **研究背景**：本研究的背景和动机（2-3 句话）
2. **方法详情**：研究方法的详细说明（3-4 句话）
3. **研究结果**：主要发现及其意义（3-4 句话）
4. **局限性**：已知的局限性或潜在问题（2-3 句话）
5. **未来方向**：可能的后续研究方向（1-2 句话）
6. **研究相关性**：为什么这篇论文可能与研究类似课题的研究者相关

请以 JSON 格式返回，包含以下键：background, methodology_details, results, limitations, future_directions, relevance_to_interests

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。"""


__all__ = ["BULLET_SUMMARY_PROMPT", "DETAILED_ANALYSIS_PROMPT"]
