'use strict';

document.addEventListener('DOMContentLoaded', function () {
  const anchor = document.querySelector('.publication-resource-links') || document.querySelector('.publication-subtitle');
  if (!anchor) return;
  // Include every gallery and deduplicate sources reused by hidden galleries.
  const files = new Map();
  document.querySelectorAll('#main-content video').forEach(function (video) {
    const source = video.querySelector('source');
    const url = video.currentSrc || video.src || (source && source.src);
    if (!url || url.startsWith('blob:')) return;
    if (!files.has(url)) files.set(url, { url: url, videos: [], done: false });
    files.get(url).videos.push(video);
  });
  const jobs = Array.from(files.values());
  if (!jobs.length) return;
  const box = document.createElement('div');
  box.className = 'hd-video-loading';
  const label = document.createElement('div');
  label.className = 'hd-video-loading-label';
  const status = document.createElement('span');
  status.setAttribute('role', 'status');
  const count = document.createElement('span');
  count.setAttribute('aria-hidden', 'true');
  label.append(status, count);
  const progress = document.createElement('progress');
  progress.max = jobs.length;
  progress.setAttribute('aria-label', 'Full-page video downloads');
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'hd-video-loading-retry';
  retry.textContent = 'Retry';
  retry.hidden = true;
  box.append(label, progress, retry);
  anchor.after(box);
  let completed = 0;
  let running = false;
  function render(finished) {
    const allDone = completed === jobs.length;
    const failed = finished && !allDone;
    status.textContent = allDone ? 'All videos loaded' : failed ? 'Some downloads failed' : 'Loading all videos…';
    count.textContent = completed + ' / ' + jobs.length;
    progress.value = completed;
    progress.setAttribute('aria-valuetext', completed + ' of ' + jobs.length + ' videos fully downloaded');
    box.dataset.state = failed ? 'error' : allDone ? 'ready' : 'loading';
    retry.hidden = !failed;
    if (allDone && !finished) setTimeout(function () {
      box.classList.add('is-complete');
      box.setAttribute('aria-hidden', 'true');
    }, 600);
  }
  function attach(file, blob) {
    // Browser-managed Blobs keep fully downloaded files available for playback.
    const localUrl = URL.createObjectURL(blob);
    file.videos.forEach(function (video) {
      const position = video.currentTime;
      const playing = !video.paused;
      video.addEventListener('loadedmetadata', function () {
        if (position > 0 && Number.isFinite(video.duration)) video.currentTime = Math.min(position, video.duration);
        if (playing) video.play().catch(function () {});
      }, { once: true });
      video.src = localUrl;
      video.load();
    });
    // Object URLs are released when the document is unloaded.
  }
  async function run() {
    if (running) return;
    running = true;
    render(false);
    const pending = jobs.filter(function (job) { return !job.done; });
    let next = 0;
    async function worker() {
      while (next < pending.length) {
        const file = pending[next++];
        try {
          const response = await fetch(file.url, { cache: 'force-cache' });
          if (!response.ok) throw new Error('HTTP ' + response.status);
          const blob = await response.blob();
          if (!blob.size) throw new Error('Empty video');
          attach(file, blob);
          file.done = true;
          completed++;
          render(false);
        } catch (error) {
          // Failures never count toward completion; offer a retry below.
        }
      }
    }
    await Promise.all([worker(), worker()]);
    running = false;
    render(true);
  }
  retry.addEventListener('click', run);
  run();
});
