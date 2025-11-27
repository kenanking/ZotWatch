"""HTML report generation."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from zotwatch.core.models import FeaturedWork, OverallSummary, RankedWork

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
    .section-expand.expanded { max-height: 5000px; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
    .content-container { max-width: 48rem; margin: 0 auto; padding: 0 2rem; }
    @media (min-width: 1024px) { .content-container { padding: 0 4rem; } }
  </style>
</head>
<body class="bg-bg-primary min-h-screen text-text-primary">
  <!-- Header -->
  <header class="bg-bg-card border-b border-border-color">
    <div class="content-container py-8">
      <h1 class="text-2xl font-bold text-text-primary">ZotWatch 文献推荐</h1>
      <p class="text-sm text-text-secondary mt-1">
        共 {{ works|length + (featured_works|length if featured_works else 0) }} 篇论文 ·
        生成于 {{ generated_at.strftime('%Y-%m-%d %H:%M') }} UTC
      </p>
    </div>
  </header>

  <!-- Overall Summaries Section -->
  {% if overall_summaries %}
  <section class="py-8 border-b border-border-color">
    <div class="content-container">
      <h2 class="text-xl font-bold text-text-primary mb-6 flex items-center gap-2">
        <svg class="w-5 h-5 text-accent" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2z"/>
        </svg>
        本期研究趋势概览
      </h2>

      <div class="space-y-6">
        {% if overall_summaries.featured %}
        <div class="bg-bg-card rounded-lg border border-border-color p-5">
          <h3 class="text-base font-semibold text-text-primary mb-3 flex items-center gap-2">
            <svg class="w-4 h-4 text-accent" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
            </svg>
            精选推荐总结 ({{ overall_summaries.featured.paper_count }} 篇)
          </h3>
          <p class="text-sm text-text-primary leading-relaxed mb-3">{{ overall_summaries.featured.summary_text }}</p>
          {% if overall_summaries.featured.key_themes %}
          <div class="flex flex-wrap gap-2">
            {% for theme in overall_summaries.featured.key_themes %}
            <span class="px-2 py-1 bg-accent/10 text-accent text-xs rounded-full">{{ theme }}</span>
            {% endfor %}
          </div>
          {% endif %}
        </div>
        {% endif %}

        {% if overall_summaries.similarity %}
        <div class="bg-bg-card rounded-lg border border-border-color p-5">
          <h3 class="text-base font-semibold text-text-primary mb-3 flex items-center gap-2">
            <svg class="w-4 h-4 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
            相似度推荐总结 ({{ overall_summaries.similarity.paper_count }} 篇)
          </h3>
          <p class="text-sm text-text-primary leading-relaxed mb-3">{{ overall_summaries.similarity.summary_text }}</p>
          {% if overall_summaries.similarity.key_themes %}
          <div class="flex flex-wrap gap-2">
            {% for theme in overall_summaries.similarity.key_themes %}
            <span class="px-2 py-1 bg-bg-hover text-text-secondary text-xs rounded-full">{{ theme }}</span>
            {% endfor %}
          </div>
          {% endif %}
        </div>
        {% endif %}
      </div>
    </div>
  </section>
  {% endif %}

  <!-- Featured Recommendations Section -->
  {% if featured_works %}
  <section class="py-8 border-b border-border-color">
    <div class="content-container">
      <div class="mb-6 flex items-center justify-between">
        <div>
          <h2 class="text-lg font-semibold text-text-primary mb-1 flex items-center gap-2">
            <svg class="w-5 h-5 text-accent" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
            </svg>
            精选推荐
          </h2>
          <p class="text-sm text-text-secondary">基于研究兴趣匹配，共 {{ featured_works|length }} 篇</p>
        </div>
      </div>

      <div class="space-y-5">
        {% for work in featured_works %}
        <article class="bg-bg-card rounded-lg border border-border-color overflow-hidden hover:border-accent/50 transition-colors">
          <div class="p-5">
            <!-- Title and metadata -->
            <div class="mb-3">
              <!-- Tags row -->
              <div class="flex items-center gap-2 mb-2 flex-wrap">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium bg-accent/10 text-accent border border-accent/30">
                  匹配度 {{ '%.0f'|format(work.rerank_score * 100) }}%
                </span>
                <span class="text-xs text-text-secondary">{{ work.source }}</span>
                <!-- Date: visible on mobile, hidden on md+ -->
                <span class="text-xs text-text-secondary md:hidden">· {{ work.published.strftime('%Y-%m-%d') if work.published else '未知' }}</span>
              </div>
              <!-- Title and date row -->
              <div class="flex items-start justify-between gap-4">
                <h2 class="text-lg font-semibold text-text-primary leading-tight flex-1">
                  <a href="{{ work.url or '#' }}" target="_blank" rel="noopener" class="hover:text-accent transition-colors">
                    {{ work.title }}
                  </a>
                </h2>
                <!-- Date: hidden on mobile, visible on md+ -->
                <div class="hidden md:block text-right text-sm text-text-secondary flex-shrink-0">
                  <div class="font-medium">{{ work.published.strftime('%Y-%m-%d') if work.published else '未知' }}</div>
                </div>
              </div>
              <!-- Venue: below title on mobile -->
              <div class="mt-1 text-xs text-text-secondary" title="{{ work.venue or '' }}">{{ work.venue or '未知来源' }}</div>
            </div>

            <!-- Authors -->
            <p class="text-sm text-text-secondary mb-3">
              {{ work.authors[:5] | join('，') }}{% if work.authors|length > 5 %} 等{% endif %}
            </p>

            <!-- DOI -->
            <div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-secondary mb-4">
              {% if work.doi %}
              <span class="flex items-center">
                <span class="font-medium mr-1">DOI:</span>
                <a href="https://doi.org/{{ work.doi }}" target="_blank" rel="noopener" class="text-accent hover:text-accent-hover hover:underline">{{ work.doi }}</a>
              </span>
              {% elif work.identifier %}
              <span class="flex items-center">
                <span class="font-medium mr-1">ID:</span>
                <span>{{ work.identifier }}</span>
              </span>
              {% endif %}
            </div>

            <!-- Abstract (collapsible) -->
            {% if work.abstract %}
            <div class="bg-bg-hover/50 rounded-lg p-4 mb-4 border border-border-color">
              <button id="btn-featured-abstract-{{ loop.index }}" onclick="toggleSection('featured-abstract-{{ loop.index }}')"
                      class="w-full text-left text-sm font-medium text-text-primary flex items-center justify-between">
                <span class="flex items-center">
                  <svg class="w-4 h-4 mr-1.5 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                  </svg>
                  原文摘要
                </span>
                <svg class="w-4 h-4 text-text-secondary transform transition-transform" id="icon-featured-abstract-{{ loop.index }}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                </svg>
              </button>
              <div id="featured-abstract-{{ loop.index }}" class="section-expand collapsed">
                <p class="text-sm text-text-secondary mt-3 leading-relaxed">{{ work.abstract }}</p>
              </div>
            </div>
            {% endif %}

            <!-- AI Summary -->
            {% if work.summary %}
            <div class="bg-accent/10 rounded-lg p-4 mb-4 border border-accent/30">
              <h3 class="text-sm font-semibold text-accent mb-3 flex items-center">
                <svg class="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2z"/>
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
                <button id="btn-featured-detail-{{ loop.index }}" onclick="toggleSection('featured-detail-{{ loop.index }}')"
                        class="text-xs text-accent hover:text-accent-hover font-medium flex items-center">
                  <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                  </svg>
                  查看详细分析
                </button>
                <div id="featured-detail-{{ loop.index }}" class="section-expand collapsed mt-3">
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

            <!-- Score details -->
            <div class="pt-4 border-t border-border-color">
              <div class="flex flex-wrap gap-3 text-xs">
                <span class="px-2 py-1 bg-bg-hover rounded text-text-secondary">相似度 {{ '%.2f'|format(work.similarity) }}</span>
              </div>
            </div>
          </div>
        </article>
        {% endfor %}
      </div>
    </div>
  </section>
  {% endif %}

  <!-- Similarity Recommendations Section -->
  <main class="py-8">
    <div class="content-container">
      <div class="mb-6 flex items-center justify-between">
        <div>
          <h2 class="text-lg font-semibold text-text-primary mb-1">相似度推荐</h2>
          <p class="text-sm text-text-secondary">按相关性评分排序，点击标题查看原文</p>
        </div>
        <div class="flex gap-2">
          <button onclick="expandAll()" class="px-3 py-1.5 text-sm text-accent hover:text-accent-hover bg-bg-card border border-border-color rounded-md transition-colors">全部展开</button>
          <button onclick="collapseAll()" class="px-3 py-1.5 text-sm text-accent hover:text-accent-hover bg-bg-card border border-border-color rounded-md transition-colors">全部折叠</button>
        </div>
      </div>

      <div class="space-y-5">
        {% for work in works %}
        <article class="bg-bg-card rounded-lg border border-border-color overflow-hidden hover:border-accent/50 transition-colors">
          <div class="p-5">
            <!-- Title and metadata -->
            <div class="mb-3">
              <!-- Tags row -->
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
                <!-- Date: visible on mobile, hidden on md+ -->
                <span class="text-xs text-text-secondary md:hidden">· {{ work.published.strftime('%Y-%m-%d') if work.published else '未知' }}</span>
              </div>
              <!-- Title and date row -->
              <div class="flex items-start justify-between gap-4">
                <h2 class="text-lg font-semibold text-text-primary leading-tight flex-1">
                  <a href="{{ work.url or '#' }}" target="_blank" rel="noopener" class="hover:text-accent transition-colors">
                    {{ work.title }}
                  </a>
                </h2>
                <!-- Date: hidden on mobile, visible on md+ -->
                <div class="hidden md:block text-right text-sm text-text-secondary flex-shrink-0">
                  <div class="font-medium">{{ work.published.strftime('%Y-%m-%d') if work.published else '未知' }}</div>
                </div>
              </div>
              <!-- Venue: below title -->
              <div class="mt-1 text-xs text-text-secondary" title="{{ work.venue or '' }}">{{ work.venue or '未知来源' }}</div>
            </div>

            <!-- Authors -->
            <p class="text-sm text-text-secondary mb-3">
              {{ work.authors[:5] | join('，') }}{% if work.authors|length > 5 %} 等{% endif %}
            </p>

            <!-- DOI -->
            <div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-secondary mb-4">
              {% if work.doi %}
              <span class="flex items-center">
                <span class="font-medium mr-1">DOI:</span>
                <a href="https://doi.org/{{ work.doi }}" target="_blank" rel="noopener" class="text-accent hover:text-accent-hover hover:underline">{{ work.doi }}</a>
              </span>
              {% elif work.identifier %}
              <span class="flex items-center">
                <span class="font-medium mr-1">ID:</span>
                <span>{{ work.identifier }}</span>
              </span>
              {% endif %}
            </div>

            <!-- Abstract (collapsible) -->
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

            <!-- AI Summary -->
            {% if work.summary %}
            <div class="bg-accent/10 rounded-lg p-4 mb-4 border border-accent/30">
              <h3 class="text-sm font-semibold text-accent mb-3 flex items-center">
                <svg class="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2z"/>
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

            <!-- Score details -->
            <div class="pt-4 border-t border-border-color">
              <div class="flex flex-wrap gap-3 text-xs">
                <span class="px-2 py-1 bg-bg-hover rounded text-text-secondary">相似度 {{ '%.2f'|format(work.similarity) }}</span>
              </div>
            </div>
          </div>
        </article>
        {% endfor %}
      </div>
    </div>
  </main>

  <!-- Footer -->
  <footer class="bg-bg-card border-t border-border-color mt-12">
    <div class="content-container py-6 text-center text-sm text-text-secondary">
      灵感来自 <a href="https://github.com/Yorks0n/ZotWatch" class="text-accent hover:text-accent-hover transition-colors">ZotWatch</a>
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
        if (btn && btn.textContent.includes('详细分析')) {
          btn.innerHTML = '<svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/></svg>收起详细分析';
        }
        if (icon) icon.style.transform = 'rotate(180deg)';
      } else {
        el.classList.remove('expanded');
        el.classList.add('collapsed');
        if (btn && btn.textContent.includes('详细分析')) {
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
      document.querySelectorAll('[id^="btn-detail-"], [id^="btn-featured-detail-"]').forEach(btn => {
        btn.innerHTML = '<svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/></svg>收起详细分析';
      });
      document.querySelectorAll('[id^="icon-abstract-"], [id^="icon-featured-abstract-"]').forEach(icon => {
        icon.style.transform = 'rotate(180deg)';
      });
    }
    function collapseAll() {
      document.querySelectorAll('.section-expand').forEach(el => {
        el.classList.remove('expanded');
        el.classList.add('collapsed');
      });
      document.querySelectorAll('[id^="btn-detail-"], [id^="btn-featured-detail-"]').forEach(btn => {
        btn.innerHTML = '<svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>查看详细分析';
      });
      document.querySelectorAll('[id^="icon-abstract-"], [id^="icon-featured-abstract-"]').forEach(icon => {
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
    featured_works: Optional[List[FeaturedWork]] = None,
    overall_summaries: Optional[Dict[str, OverallSummary]] = None,
) -> Path:
    """Render HTML report from ranked works.

    Args:
        works: Ranked works to include
        output_path: Path to write HTML file
        template_dir: Directory containing templates
        template_name: Name of template file
        featured_works: Optional list of featured works based on user interests
        overall_summaries: Optional dict with "featured" and/or "similarity" OverallSummary

    Returns:
        Path to written HTML file
    """
    generated_at = datetime.now(timezone.utc)

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
        featured_works=featured_works or [],
        overall_summaries=overall_summaries or {},
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report with %d items to %s", len(works), path)
    return path


__all__ = ["render_html"]
