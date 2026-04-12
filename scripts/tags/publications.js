/* global hexo */

'use strict';

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/"/g, '&quot;');
}

function authorsToHtml(s) {
  if (!s) return '';
  const parts = String(s).split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part) => {
    const m = /^\*\*([^*]+)\*\*$/.exec(part);
    if (m) return `<b>${escapeHtml(m[1])}</b>`;
    return escapeHtml(part);
  }).join('');
}

const ICONS = {
  pdf: 'iconfont icon-file-pdf',
  earth: 'iconfont icon-earth',
  code: 'iconfont icon-codelibrary',
};

// 每条论文左侧缩略图占位框：宽高固定；图用 object-fit:fill 拉伸铺满（不裁切）
const PUB_IMG_FRAME_HEIGHT = '11rem';
// 左图 / 右文列宽占比（之和应为 100）
const PUB_IMG_COL_PCT = 35;
const PUB_TEXT_COL_PCT = 65;

function resolveImgUrl(ctx, src) {
  if (!src) return '';
  // 外链原样输出
  if (/^https?:\/\//i.test(src) || src.startsWith('//')) return src;
  // 与 about/index.md 一致：仅用资源目录下的文件名，交给 hexo-asset-image 在 after_post_render 里补全路径
  if (!src.includes('/') && !src.includes('\\')) return src;
  const path = src.startsWith('/') ? src : `/${src}`;
  if (typeof ctx.url_for === 'function') {
    return ctx.url_for(path);
  }
  const root = (hexo.config.root || '/').replace(/\/?$/, '/');
  return root + path.replace(/^\//, '');
}

function renderRow(pub, ctx) {
  const title = (pub.title || '').trim();
  const titleExtra = pub.title_style ? String(pub.title_style).trim() : '';
  const titleStyle = `font-size: 18px; font-weight: bold;${titleExtra ? ` ${titleExtra}` : ''}`;
  const titleHtml = `<span class="pub-title" style="${escapeHtml(titleStyle)}">${escapeHtml(title)}</span>`;

  let linksHtml = '';
  if (Array.isArray(pub.links)) {
    for (const link of pub.links) {
      if (!link || !link.url) continue;
      const iconClass = ICONS[link.icon] || ICONS.pdf;
      const text = (link.text || 'Link').trim();
      linksHtml += ` <i class="${iconClass}"></i> <a href="${escapeHtml(link.url)}" target="_blank" rel="noopener">${escapeHtml(text)}</a>`;
    }
  }

  const imgSrc = resolveImgUrl(ctx, pub.image);

  const textBlock = `${titleHtml}
        <br>
        ${authorsToHtml(pub.authors)}
        <br>
        <em>${escapeHtml(pub.venue || '')}</em>
        <br>${linksHtml}`;

  // 双列 flex + 顶对齐；左列固定画框，图片拉伸至与框同宽高（fill，非 cover 故不裁切）
  // 条目间距由外层 .pub-list 的 gap 统一控制，避免 margin 折叠导致「第一篇与第二篇」和后面不一致
  return `<div class="pub-item" style="display:flex;flex-flow:row nowrap;align-items:flex-start;width:100%;max-width:100%;border:none;margin:0;box-sizing:border-box;">
      <div class="pub-item__img-side" style="flex:0 0 ${PUB_IMG_COL_PCT}%;max-width:${PUB_IMG_COL_PCT}%;width:${PUB_IMG_COL_PCT}%;padding:0;border:none;box-sizing:border-box;min-width:0;">
        <div class="pub-item__img-frame" style="width:100%;height:${PUB_IMG_FRAME_HEIGHT};box-sizing:border-box;line-height:0;">
          <img src="${escapeHtml(imgSrc)}" alt="${escapeHtml(title)}" style="display:block;width:100%;height:100%;object-fit:fill;">
        </div>
      </div>
      <div class="pub-item__text" style="flex:1 1 ${PUB_TEXT_COL_PCT}%;max-width:${PUB_TEXT_COL_PCT}%;padding:0 0 0 20px;border:none;box-sizing:border-box;min-width:0;">
        ${textBlock}
      </div>
    </div>`;
}

// {% publications %} — 数据来自 source/_data/publications.yml（即 site.data.publications）
hexo.extend.tag.register('publications', function () {
  const ctx = this;
  const list = ctx.site && ctx.site.data && ctx.site.data.publications;
  if (!Array.isArray(list) || list.length === 0) {
    hexo.log.warn('[publications] 缺少或为空: source/_data/publications.yml（根级应为列表）');
    return '';
  }
  const rows = list.map((pub) => renderRow(pub, ctx)).join('\n');
  return `<div class="pub-list" style="display:flex;flex-direction:column;gap:2rem;width:100%;max-width:100%;box-sizing:border-box;">${rows}</div>`;
}, { ends: false });
