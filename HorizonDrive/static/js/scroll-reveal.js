'use strict';

// Native, reversible scroll motion. Layout and video playback remain independent.
document.addEventListener('DOMContentLoaded', function () {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const main = document.getElementById('main-content');
  if (!main) return;
  const grids = Array.from(main.querySelectorAll('.combined-results-stage .video-grid, #sec-minute-level-video .video-grid'));
  const headings = Array.from(main.querySelectorAll('section[id] h2.title'));
  const reveals = Array.from(main.querySelectorAll('.section-subtitle, .combined-results-description, #sec-closed-loop .demo-media, #sec-method-overview img, #sec-quantitative .table-container, #sec-abstract .content, #sec-citation pre'));
  const clamp = function (x) { return Math.max(0, Math.min(1, x)); };
  const smooth = function (x) { return x * x * (3 - 2 * x); };
  const titles = headings.filter(function (el) { return el.children.length === 0; }).map(function (el) {
    const text = el.textContent.trim();
    const visual = document.createElement('span');
    visual.setAttribute('aria-hidden', 'true');
    const accessible = document.createElement('span');
    accessible.className = 'hd-motion-sr';
    accessible.textContent = text;
    // Keep normal word wrapping and reserve every character's final space.
    const chars = Array.from(text).map(function (char) {
      const span = document.createElement('span');
      span.textContent = char;
      visual.appendChild(span);
      return span;
    });
    el.replaceChildren(accessible, visual);
    return { el: el, chars: chars };
  });
  grids.forEach(function (grid) { grid.classList.add('hd-scroll-grid'); });
  const gridStates = new Map();
  let geometryDirty = true;
  let inputDirty = true;
  let lastTime = 0;
  let copyTick = 0;
  let frame = 0;
  function draw(time) {
    frame = 0;
    const dt = lastTime ? Math.min(time - lastTime, 50) : 16.67;
    lastTime = time;
    // Time-based damping behaves consistently on 60 Hz and high-refresh screens.
    const blend = 1 - Math.exp(-dt / 125);
    const vh = window.innerHeight;
    const disabled = reduced.matches;
    const compact = window.innerWidth < 700;
    // Measure before writing to avoid repeated layout work during scrolling.
    const gridData = inputDirty ? grids.map(function (el) {
      const rect = el.getBoundingClientRect();
      let state = gridStates.get(el);
      if (!state || geometryDirty) {
        state = { current: state ? state.current : null, needsRender: true, items: Array.from(el.children).map(function (item) {
          return { el: item, dx: el.clientWidth / 2 - item.offsetLeft - item.offsetWidth / 2,
            dy: el.clientHeight / 2 - item.offsetTop - item.offsetHeight / 2 };
        }) };
        gridStates.set(el, state);
      }
      return { rect: rect, state: state };
    }) : [];
    // Copy fades are lower priority than the tile transforms. Sample their
    // geometry every fourth frame so wheel input does not force a full-page
    // layout read on every animation frame.
    copyTick = (copyTick + 1) % 4;
    const copyDue = inputDirty && copyTick === 0;
    const titleData = copyDue ? titles.map(function (title) { return { title: title, rect: title.el.getBoundingClientRect() }; }) : [];
    const revealData = copyDue ? reveals.map(function (el) { return { el: el, rect: el.getBoundingClientRect() }; }) : [];
    geometryDirty = false;
    inputDirty = false;
    gridData.forEach(function (data) {
      data.state.visible = data.rect.width > 0 && data.rect.bottom > 0 && data.rect.top < vh;
      // Unfold until the top reaches 42% of the viewport; start folding again
      // when the bottom crosses 48%, keeping the settled interval near center.
      const progress = disabled ? 1 : smooth(clamp(Math.min((vh - data.rect.top) / (vh * .58), data.rect.bottom / (vh * .48))));
      data.state.target = (1 - progress) * (compact ? .38 : .82);
    });
    let moving = false;
    gridStates.forEach(function (state) {
      const target = disabled ? 0 : state.target;
      const previous = state.current;
      if (disabled || !state.visible || previous === null) state.current = target;
      else state.current += (target - state.current) * blend;
      const unsettled = Math.abs(target - state.current) > .0005;
      if (!unsettled) state.current = target;
      moving = moving || (unsettled && state.visible);
      state.items.forEach(function (item, i) {
        if (previous === state.current && !disabled && !state.needsRender) return;
        const fold = state.current;
        const x = item.dx * fold;
        const y = item.dy * fold;
        const rotation = ((i * 7 % 11) - 5) * fold;
        item.el.style.transform = disabled ? '' : 'translate3d(' + x.toFixed(2) + 'px,' + y.toFixed(2) + 'px,0) rotate(' + rotation.toFixed(2) + 'deg) scale(' + (1 - fold * .12).toFixed(3) + ')';
      });
      state.needsRender = false;
    });
    titleData.forEach(function (data) {
      if (!data.rect.width) return;
      const progress = disabled ? 1 : clamp((vh * .96 - data.rect.top) / (vh * .24));
      const count = Math.ceil(progress * data.title.chars.length);
      if (data.title.count !== count) {
        data.title.chars.forEach(function (char, i) { char.style.opacity = i < count ? '1' : '.12'; });
        data.title.count = count;
      }
    });
    revealData.forEach(function (data) {
      if (!data.rect.width) return;
      const progress = disabled ? 1 : smooth(clamp((vh * .98 - data.rect.top) / (vh * .22)));
      const opacity = String(.25 + .75 * progress);
      if (data.el.style.opacity !== opacity) data.el.style.opacity = opacity;
    });
    if (moving) frame = requestAnimationFrame(draw);
    else lastTime = 0;
  }
  function schedule() { inputDirty = true; if (!frame) frame = requestAnimationFrame(draw); }
  function invalidate() { geometryDirty = true; schedule(); }
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', invalidate, { passive: true });
  reduced.addEventListener('change', schedule);
  // Tab switches and media metadata can change the visible grid's dimensions.
  const sizeObserver = new ResizeObserver(invalidate);
  grids.forEach(function (grid) { sizeObserver.observe(grid); });
  main.addEventListener('loadedmetadata', invalidate, true);
  main.addEventListener('click', invalidate);
  schedule();
});
