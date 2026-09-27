'use strict';
document.addEventListener('DOMContentLoaded', function () {
  const player = document.getElementById('rollout-player');
  if (!player) return;
  const buttons = Array.from(document.querySelectorAll('.rollout-scene'));
  const label = document.getElementById('rollout-current');
  const error = document.querySelector('.rollout-error');
  let selection = 0;
  buttons.forEach(function (button, index) {
    button.addEventListener('click', function () {
      if (button.getAttribute('aria-pressed') === 'true') return;
      const currentSelection = ++selection;
      player.pause();
      error.hidden = true;
      buttons.forEach(function (item) {
        item.classList.toggle('is-selected', item === button);
        item.setAttribute('aria-pressed', String(item === button));
      });
      const number = String(index + 1).padStart(2, '0');
      label.textContent = 'Scene ' + number + ' / 09';
      player.setAttribute('aria-label', 'Minute-level driving rollout, scene ' + number);
      player.poster = button.dataset.poster;
      player.src = button.dataset.src;
      player.load();
      player.play().catch(function (err) {
        if (currentSelection === selection && err.name !== 'AbortError' && err.name !== 'NotAllowedError') error.hidden = false;
      });
    });
  });
  player.addEventListener('error', function () { error.hidden = false; });
  player.addEventListener('loadeddata', function () { error.hidden = true; });
  // Keep native seeking and fullscreen controls; pause when leaving the showcase.
  const observer = new IntersectionObserver(function (entries) {
    if (!entries[0].isIntersecting) player.pause();
  }, { threshold: 0 });
  observer.observe(player);
});
