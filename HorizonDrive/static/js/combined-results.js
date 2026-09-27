'use strict';

document.addEventListener('DOMContentLoaded', function () {
  const stage = document.getElementById('combined-results-stage');
  const tabs = document.querySelector('.combined-results-tabs');
  const description = document.getElementById('combined-results-description');
  if (!stage || !tabs || !description) return;

  const nusc = Array.from(document.querySelectorAll('#sec-streaming-nusc .video-grid-item video source')).map(function (source) { return source.src; });
  const scaleGroups = Array.from(document.querySelectorAll('#sec-more-e2e .video-carousel-slide')).map(function (slide) {
    return Array.from(slide.querySelectorAll('.video-grid-item video source')).map(function (source) { return source.src; });
  });
  const groups = [nusc].concat(scaleGroups);
  const copy = [
    ['20s · nuScenes', 'Twenty-second rollouts on nuScenes Val Dataset.'],
    ['30s · Group 1', 'Thirty-second rollouts on self-collected scenes, group 1.'],
    ['30s · Group 2', 'Thirty-second rollouts on self-collected scenes, group 2.'],
    ['30s · Group 3', 'Thirty-second rollouts on self-collected scenes, group 3.'],
    ['30s · Group 4', 'Thirty-second rollouts on self-collected scenes, group 4.']
  ];
  groups.forEach(function (sources, groupIndex) {
    const group = document.createElement('div');
    group.className = 'combined-results-group' + (groupIndex === 0 ? ' is-active' : '');
    group.setAttribute('role', 'tabpanel');
    const grid = document.createElement('div');
    grid.className = 'video-grid';
    sources.forEach(function (src) {
      const item = document.createElement('div'); item.className = 'video-grid-item';
      const video = document.createElement('video');
      video.controls = false; video.muted = true; video.loop = true; video.playsInline = true; video.preload = 'metadata'; video.autoplay = false;
      const source = document.createElement('source'); source.src = src; source.type = 'video/mp4'; video.appendChild(source); item.appendChild(video); grid.appendChild(item);
    });
    group.appendChild(grid); stage.appendChild(group);
    const tab = document.createElement('button');
    tab.className = 'combined-results-tab' + (groupIndex === 0 ? ' is-active' : ''); tab.type = 'button'; tab.setAttribute('role', 'tab'); tab.setAttribute('aria-selected', String(groupIndex === 0)); tab.textContent = copy[groupIndex][0];
    tab.addEventListener('click', function () {
      document.querySelectorAll('.combined-results-tab').forEach(function (button, index) { button.classList.toggle('is-active', index === groupIndex); button.setAttribute('aria-selected', String(index === groupIndex)); });
      document.querySelectorAll('.combined-results-group').forEach(function (panel, index) { panel.classList.toggle('is-active', index === groupIndex); panel.querySelectorAll('video').forEach(function (video) { if (index !== groupIndex) video.pause(); }); });
      description.textContent = copy[groupIndex][1];
    });
    tabs.appendChild(tab);
  });

  // Reuse the site's hover-only center control for videos created after the
  // main page script has initialized, and autoplay only the visible group.
  if (typeof setupCustomVideoControls === 'function') setupCustomVideoControls();
  const observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      const video = entry.target;
      const active = video.closest('.combined-results-group.is-active');
      if (entry.isIntersecting && active && entry.intersectionRatio >= 0.35) video.play().catch(function () {});
      else video.pause();
    });
  }, { threshold: [0, 0.35, 0.6] });
  stage.querySelectorAll('video').forEach(function (video) { observer.observe(video); });

  // Keep the minute-level gallery immediately after the five-group showcase.
  const minuteGallery = document.getElementById('sec-minute-level-video');
  const abstract = document.getElementById('sec-abstract');
  if (minuteGallery && abstract) abstract.parentNode.insertBefore(minuteGallery, abstract);
});
