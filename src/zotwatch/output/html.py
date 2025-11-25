"""HTML report generation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from zotwatch.core.models import RankedWork

logger = logging.getLogger(__name__)

# Fallback template when no external template is available
_FALLBACK_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZotWatch 文献推荐 - {{ generated_at.strftime('%Y-%m-%d') }}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            'bg-primary': '#f8fafc',
            'bg-card': '#ffffff',
            'bg-hover': '#f1f5f9',
            'text-primary': '#1e293b',
            'text-secondary': '#64748b',
            'border-color': '#e2e8f0',
            'accent': '#2563eb',
            'accent-hover': '#1d4ed8',
          }
        }
      }
    }
  </script>
  <style>
    .section-expand { transition: max-height 0.3s ease-out; overflow: hidden; }
    .section-expand.collapsed { max-height: 0; }
    .section-expand.expanded { max-height: 3000px; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
  </style>
</head>
<body class="bg-bg-primary min-h-screen text-text-primary">
  <header class="bg-bg-card border-b border-border-color">
    <div class="max-w-4xl mx-auto px-4 py-6">
      <h1 class="text-2xl font-bold text-text-primary">ZotWatch 文献推荐</h1>
      <p class="text-sm text-text-secondary mt-1">共 {{ works|length }} 篇论文 · 生成于 {{ generated_at.strftime('%Y年%m月%d日 %H:%M') }}</p>
    </div>
  </header>

  <main class="max-w-4xl mx-auto px-4 py-8">
    <div class="mb-6 flex items-center justify-between">
      <p class="text-sm text-text-secondary">按相关性评分排序，点击标题查看原文</p>
      <div class="flex gap-2">
        <button onclick="expandAll()" class="px-3 py-1.5 text-sm text-accent hover:text-accent-hover bg-bg-card border border-border-color rounded-md transition-colors">全部展开</button>
        <button onclick="collapseAll()" class="px-3 py-1.5 text-sm text-accent hover:text-accent-hover bg-bg-card border border-border-color rounded-md transition-colors">全部折叠</button>
      </div>
    </div>

    <div class="grid gap-5">
      {% for work in works %}
      <article class="bg-bg-card rounded-lg border border-border-color overflow-hidden hover:border-accent/50 transition-colors">
        <div class="p-5">
          <!-- 标题和元信息 -->
          <div class="flex items-start justify-between mb-3">
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-2 flex-wrap">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium
                  {% if work.label == 'must_read' %}bg-green-100 text-green-700 border border-green-300
                  {% elif work.label == 'consider' %}bg-amber-100 text-amber-700 border border-amber-300
                  {% else %}bg-bg-hover text-text-secondary border border-border-color{% endif %}">
                  {% if work.label == 'must_read' %}必读{% elif work.label == 'consider' %}推荐{% else %}参考{% endif %}
                </span>
                <span class="text-xs text-text-secondary">评分 {{ '%.2f'|format(work.score) }}</span>
                <span class="text-xs text-text-secondary">·</span>
                <span class="text-xs text-text-secondary">{{ work.source }}</span>
              </div>
              <h2 class="text-lg font-semibold text-text-primary leading-tight">
                <a href="{{ work.url or '#' }}" target="_blank" rel="noopener" class="hover:text-accent transition-colors">
                  {{ work.title }}
                </a>
              </h2>
            </div>
            <div class="ml-4 text-right text-sm text-text-secondary flex-shrink-0">
              <div class="font-medium">{{ work.published.strftime('%Y-%m-%d') if work.published else '未知' }}</div>
              <div class="text-xs text-text-secondary max-w-[150px] truncate" title="{{ work.venue or '' }}">{{ work.venue or '未知来源' }}</div>
            </div>
          </div>

          <!-- 作者 -->
          <p class="text-sm text-text-secondary mb-4">
            {{ work.authors[:5] | join('，') }}{% if work.authors|length > 5 %} 等{% endif %}
          </p>

          <!-- 原文摘要 -->
          {% if work.abstract %}
          <div class="bg-bg-hover/50 rounded-lg p-4 mb-4 border border-border-color">
            <button id="btn-abstract-{{ loop.index }}" onclick="toggleSection('abstract-{{ loop.index }}')"
                    class="w-full text-left text-sm font-medium text-text-primary flex items-center justify-between">
              <span class="flex items-center">
                <svg class="w-4 h-4 mr-1.5 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
                原文摘要
              </span>
              <svg class="w-4 h-4 text-text-secondary transform transition-transform" id="icon-abstract-{{ loop.index }}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
              </svg>
            </button>
            <div id="abstract-{{ loop.index }}" class="section-expand collapsed">
              <p class="text-sm text-text-secondary mt-3 leading-relaxed">{{ work.abstract }}</p>
            </div>
          </div>
          {% endif %}

          <!-- AI 总结 -->
          {% if work.summary %}
          <div class="bg-accent/10 rounded-lg p-4 mb-4 border border-accent/30">
            <h3 class="text-sm font-semibold text-accent mb-3 flex items-center">
              <svg class="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2zm-3 9a1 1 0 100 2 1 1 0 000-2zm6 0a1 1 0 100 2 1 1 0 000-2zm-3 4c-1.1 0-2.1.45-2.83 1.17l1.41 1.41A2 2 0 0012 17a2 2 0 001.42.59l1.41-1.42A4 4 0 0012 15z"/>
              </svg>
              AI 总结
            </h3>
            <div class="space-y-2 text-sm text-text-primary">
              <p><span class="font-medium text-accent">研究问题：</span>{{ work.summary.bullets.research_question }}</p>
              <p><span class="font-medium text-accent">研究方法：</span>{{ work.summary.bullets.methodology }}</p>
              <p><span class="font-medium text-accent">主要发现：</span>{{ work.summary.bullets.key_findings }}</p>
              <p><span class="font-medium text-accent">创新点：</span>{{ work.summary.bullets.innovation }}</p>
              {% if work.summary.bullets.relevance_note %}
              <p><span class="font-medium text-accent">相关性：</span>{{ work.summary.bullets.relevance_note }}</p>
              {% endif %}
            </div>

            <div class="mt-3 pt-3 border-t border-accent/30">
              <button id="btn-detail-{{ loop.index }}" onclick="toggleSection('detail-{{ loop.index }}')"
                      class="text-xs text-accent hover:text-accent-hover font-medium flex items-center">
                <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                </svg>
                查看详细分析
              </button>
              <div id="detail-{{ loop.index }}" class="section-expand collapsed mt-3">
                <div class="bg-bg-card rounded-lg p-4 text-sm text-text-primary space-y-3 border border-border-color">
                  <p><span class="font-medium text-text-primary">研究背景：</span>{{ work.summary.detailed.background }}</p>
                  <p><span class="font-medium text-text-primary">方法详情：</span>{{ work.summary.detailed.methodology_details }}</p>
                  <p><span class="font-medium text-text-primary">研究结果：</span>{{ work.summary.detailed.results }}</p>
                  <p><span class="font-medium text-text-primary">局限性：</span>{{ work.summary.detailed.limitations }}</p>
                  {% if work.summary.detailed.future_directions %}
                  <p><span class="font-medium text-text-primary">未来方向：</span>{{ work.summary.detailed.future_directions }}</p>
                  {% endif %}
                  <p class="text-accent"><span class="font-medium">研究相关性：</span>{{ work.summary.detailed.relevance_to_interests }}</p>
                </div>
              </div>
            </div>
          </div>
          {% endif %}

          <!-- 评分详情 -->
          <div class="pt-4 border-t border-border-color">
            <div class="flex flex-wrap gap-3 text-xs">
              <span class="px-2 py-1 bg-bg-hover rounded text-text-secondary">相似度 {{ '%.2f'|format(work.similarity) }}</span>
              <span class="px-2 py-1 bg-bg-hover rounded text-text-secondary">时效性 {{ '%.2f'|format(work.recency_score) }}</span>
              {% if work.journal_sjr %}
              <span class="px-2 py-1 bg-bg-hover rounded text-text-secondary">SJR {{ '%.2f'|format(work.journal_sjr) }}</span>
              {% endif %}
              {% if work.author_bonus > 0 %}
              <span class="px-2 py-1 bg-green-100 rounded text-green-700">关注作者</span>
              {% endif %}
              {% if work.venue_bonus > 0 %}
              <span class="px-2 py-1 bg-green-100 rounded text-green-700">关注期刊</span>
              {% endif %}
            </div>
          </div>
        </div>
      </article>
      {% endfor %}
    </div>
  </main>

  <footer class="bg-bg-card border-t border-border-color mt-12">
    <div class="max-w-4xl mx-auto px-4 py-6 text-center text-sm text-text-secondary">
      由 <a href="https://github.com/zotwatch/zotwatch" class="text-accent hover:text-accent-hover transition-colors">ZotWatch</a> 生成
    </div>
  </footer>

  <script>
    function toggleSection(id) {
      const el = document.getElementById(id);
      const btn = document.getElementById('btn-' + id);
      const icon = document.getElementById('icon-' + id);
      if (el.classList.contains('collapsed')) {
        el.classList.remove('collapsed');
        el.classList.add('expanded');
        if (btn.textContent.includes('详细分析')) {
          btn.innerHTML = '<svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/></svg>收起详细分析';
        }
        if (icon) icon.style.transform = 'rotate(180deg)';
      } else {
        el.classList.remove('expanded');
        el.classList.add('collapsed');
        if (btn.textContent.includes('详细分析')) {
          btn.innerHTML = '<svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>查看详细分析';
        }
        if (icon) icon.style.transform = 'rotate(0deg)';
      }
    }
    function expandAll() {
      document.querySelectorAll('.section-expand').forEach(el => {
        el.classList.remove('collapsed');
        el.classList.add('expanded');
      });
      document.querySelectorAll('[id^="btn-detail-"]').forEach(btn => {
        btn.innerHTML = '<svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/></svg>收起详细分析';
      });
      document.querySelectorAll('[id^="icon-abstract-"]').forEach(icon => {
        icon.style.transform = 'rotate(180deg)';
      });
    }
    function collapseAll() {
      document.querySelectorAll('.section-expand').forEach(el => {
        el.classList.remove('expanded');
        el.classList.add('collapsed');
      });
      document.querySelectorAll('[id^="btn-detail-"]').forEach(btn => {
        btn.innerHTML = '<svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>查看详细分析';
      });
      document.querySelectorAll('[id^="icon-abstract-"]').forEach(icon => {
        icon.style.transform = 'rotate(0deg)';
      });
    }
  </script>
</body>
</html>"""


def render_html(
    works: List[RankedWork],
    output_path: Path | str,
    *,
    template_dir: Optional[Path] = None,
    template_name: str = "report.html",
) -> Path:
    """Render HTML report from ranked works.

    Args:
        works: Ranked works to include
        output_path: Path to write HTML file
        template_dir: Directory containing templates
        template_name: Name of template file

    Returns:
        Path to written HTML file
    """
    generated_at = datetime.utcnow()

    # Try to load external template
    if template_dir and (template_dir / template_name).exists():
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template_name)
    else:
        # Use fallback template
        env = Environment(autoescape=select_autoescape(["html", "xml"]))
        template = env.from_string(_FALLBACK_TEMPLATE)

    rendered = template.render(
        works=works,
        generated_at=generated_at,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report with %d items to %s", len(works), path)
    return path


__all__ = ["render_html"]
