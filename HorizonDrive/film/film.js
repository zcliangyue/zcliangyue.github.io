(() => {
  const scenes = [...document.querySelectorAll("[data-scene]")];
  const toggleButton = document.querySelector("[data-toggle]");
  const toggleText = document.querySelector("[data-toggle-text]");
  const restartButton = document.querySelector("[data-restart]");
  const scrubber = document.querySelector("[data-scrubber]");
  const progress = document.querySelector("[data-progress]");
  const sceneLabel = document.querySelector("[data-scene-label]");
  const timecode = document.querySelector("[data-timecode]");

  if (!scenes.length || !toggleButton || !scrubber || !progress) {
    return;
  }

  const hydrateMethodDrawing = () => {
    const host = document.querySelector("[data-home-method]");
    if (!host) {
      return;
    }

    fetch("../", { credentials: "same-origin" })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Homepage Method Overview unavailable.");
        }
        return response.text();
      })
      .then((html) => {
        const methodSvg = new DOMParser()
          .parseFromString(html, "text/html")
          .querySelector(".method-wireframe-svg");

        if (!methodSvg) {
          throw new Error("Homepage Method Overview drawing missing.");
        }

        methodSvg.querySelectorAll("image[href]").forEach((image) => {
          const href = image.getAttribute("href");
          if (!href) {
            return;
          }

          image.setAttribute("href", new URL(href, new URL("../", window.location.href)).toString());
        });

        host.replaceChildren(methodSvg);
      })
      .catch(() => {
        host.innerHTML = '<span class="method-drawing-loading">Method Overview unavailable</span>';
      });
  };

  const durations = scenes.map((scene) => Number(scene.dataset.duration) || 6);
  const sceneStarts = durations.map((_, index) => durations
    .slice(0, index)
    .reduce((sum, duration) => sum + duration, 0));
  const totalDuration = durations.reduce((sum, duration) => sum + duration, 0);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SCENE_CROSSFADE_MS = 2000;
  const SCENE_PRELOAD_SECONDS = 2;
  let position = 0;
  let sceneIndex = -1;
  let playing = !reduceMotion;
  let lastFrame = performance.now();
  let animationFrame = 0;
  let leavingTimer = 0;

  const setSceneVideoLoaded = (video, shouldLoad) => {
    const source = video.querySelector("source");
    if (!source) {
      return;
    }

    if (!source.dataset.src && source.getAttribute("src")) {
      source.dataset.src = source.getAttribute("src");
    }

    const targetSrc = source.dataset.src;
    const currentSrc = source.getAttribute("src");

    if (shouldLoad) {
      if (targetSrc && currentSrc !== targetSrc) {
        source.setAttribute("src", targetSrc);
        video.load();
      }
      return;
    }

    if (currentSrc) {
      video.pause();
      source.removeAttribute("src");
      video.load();
    }
  };

  const syncSceneVideoSources = (activeIndex, options = {}) => {
    const preloadIndex = options.preloadIndex ?? activeIndex + 1;
    const keepLoadedIndex = options.keepLoadedIndex ?? null;

    scenes.forEach((scene, index) => {
      const shouldLoad = index === activeIndex
        || index === preloadIndex
        || index === keepLoadedIndex;

      scene.querySelectorAll("[data-scene-video]").forEach((video) => {
        setSceneVideoLoaded(video, shouldLoad);
      });
    });
  };

  const clearSceneTransitions = () => {
    window.clearTimeout(leavingTimer);
    leavingTimer = 0;
    scenes.forEach((scene) => {
      scene.classList.remove("is-leaving", "is-exiting");
    });
  };

  const preloadUpcomingSceneVideos = () => {
    const nextIndex = sceneIndex + 1;
    if (nextIndex >= scenes.length || nextIndex < 0) {
      return;
    }

    const nextSceneStart = sceneStarts[nextIndex] ?? totalDuration;
    const timeUntilNext = nextSceneStart - position;
    if (timeUntilNext > SCENE_PRELOAD_SECONDS || timeUntilNext <= 0) {
      return;
    }

    scenes[nextIndex].querySelectorAll("[data-scene-video]").forEach((video) => {
      setSceneVideoLoaded(video, true);
    });
  };

  const syncAbstractCarryoverToMethod = (previousIndex, nextIndex) => {
    const previousScene = scenes[previousIndex];
    const nextScene = scenes[nextIndex];
    if (!previousScene || !nextScene) {
      return;
    }
    if (!previousScene.classList.contains("scene-abstract") || !nextScene.classList.contains("scene-method")) {
      return;
    }

    const abstractVideo = previousScene.querySelector(".abstract-road");
    const methodVideo = nextScene.querySelector(".method-road");
    if (!(abstractVideo instanceof HTMLVideoElement) || !(methodVideo instanceof HTMLVideoElement)) {
      return;
    }

    const fromTime = Number.isFinite(abstractVideo.currentTime) ? abstractVideo.currentTime : 0;
    const applyCurrentTime = () => {
      const maxPlayable = Number.isFinite(methodVideo.duration) ? Math.max(0, methodVideo.duration - 0.08) : fromTime;
      methodVideo.currentTime = Math.min(fromTime, maxPlayable);
    };

    if (methodVideo.readyState >= 1) {
      applyCurrentTime();
      return;
    }

    methodVideo.addEventListener("loadedmetadata", applyCurrentTime, { once: true });
  };

  const formatTime = (seconds) => {
    const whole = Math.max(0, Math.round(seconds));
    const minutes = String(Math.floor(whole / 60)).padStart(2, "0");
    const remainder = String(whole % 60).padStart(2, "0");
    return `${minutes}:${remainder}`;
  };

  const sceneAt = (seconds) => {
    let elapsed = 0;
    for (let index = 0; index < durations.length; index += 1) {
      const nextElapsed = elapsed + durations[index];
      if (seconds < nextElapsed || index === durations.length - 1) {
        return index;
      }
      elapsed = nextElapsed;
    }
    return 0;
  };

  const sceneElapsed = (index) => Math.max(0, position - (sceneStarts[index] || 0));

  const resetTailVideo = (video) => {
    const tailSeconds = Number(video.dataset.tailSeconds);
    if (!tailSeconds) {
      return;
    }

    delete video.dataset.tailReady;

    const setTailStart = () => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        return;
      }

      video.currentTime = Math.max(0, video.duration - tailSeconds);
      video.dataset.tailReady = "true";
      if (playing) {
        video.play().catch(() => {});
      }
    };

    if (video.readyState >= 1) {
      setTailStart();
      return;
    }

    video.addEventListener("loadedmetadata", setTailStart, { once: true });
  };

  const syncSceneCarousels = () => {
    scenes.forEach((scene, index) => {
      scene.querySelectorAll("[data-film-carousel]").forEach((carousel) => {
        const pages = [...carousel.querySelectorAll("[data-film-page]")];
        const pageDuration = Number(carousel.dataset.pageDuration) || 8;
        const activePage = Math.min(
          Math.floor(sceneElapsed(index) / pageDuration),
          Math.max(pages.length - 1, 0)
        );

        pages.forEach((page, pageIndex) => {
          page.classList.toggle("is-active", pageIndex === activePage);
        });
      });
    });
  };

  const syncSceneReveals = () => {
    scenes.forEach((scene, index) => {
      scene.querySelectorAll("[data-scene-reveal]").forEach((element) => {
        const revealAt = Number(element.dataset.revealAt) || durations[index] / 2;
        element.classList.toggle("is-revealed", sceneElapsed(index) >= revealAt);
      });
    });
  };

  const canPlaySceneVideo = (video) => {
    const carouselPage = video.closest("[data-film-page]");
    if (carouselPage && !carouselPage.classList.contains("is-active")) {
      return false;
    }

    if (video.dataset.tailSeconds && video.dataset.tailReady !== "true") {
      return false;
    }

    return true;
  };

  const syncVideos = (activeIndex) => {
    scenes.forEach((scene, index) => {
      const isLeaving = scene.classList.contains("is-leaving");
      const shouldPlayScene = playing && (index === activeIndex || isLeaving);

      scene.querySelectorAll("[data-scene-video]").forEach((video) => {
        video.playbackRate = Number(scene.dataset.playbackRate) || 1;
        if (shouldPlayScene && canPlaySceneVideo(video)) {
          video.play().catch(() => {});
          return;
        }

        if (!isLeaving) {
          video.pause();
        }
      });
    });
  };

  const renderScene = (nextIndex) => {
    if (sceneIndex === nextIndex) {
      return;
    }

    const previousIndex = sceneIndex;
    clearSceneTransitions();
    sceneIndex = nextIndex;

    scenes.forEach((scene, index) => {
      if (index === sceneIndex) {
        scene.classList.add("is-active");
        scene.classList.remove("is-leaving", "is-exiting");
        return;
      }

      scene.classList.remove("is-active", "is-leaving", "is-exiting");
    });

    if (previousIndex >= 0 && previousIndex !== sceneIndex && !reduceMotion) {
      const leavingScene = scenes[previousIndex];
      leavingScene.classList.add("is-leaving");
      window.requestAnimationFrame(() => {
        leavingScene.classList.add("is-exiting");
      });
      leavingTimer = window.setTimeout(() => {
        leavingScene.classList.remove("is-leaving", "is-exiting");
        leavingScene.querySelectorAll("[data-scene-video]").forEach((video) => video.pause());
        leavingTimer = 0;
      }, SCENE_CROSSFADE_MS);
    }

    syncAbstractCarryoverToMethod(previousIndex, sceneIndex);
    syncSceneVideoSources(sceneIndex, {
      keepLoadedIndex: previousIndex >= 0 ? previousIndex : null,
      preloadIndex: sceneIndex + 1,
    });
    scenes[sceneIndex]?.querySelectorAll("[data-tail-seconds]").forEach(resetTailVideo);
    syncSceneCarousels();
    syncSceneReveals();
    syncVideos(sceneIndex);
  };

  const renderControls = () => {
    const ratio = Math.min(position / totalDuration, 1);
    scrubber.value = String(Math.round(ratio * Number(scrubber.max)));
    progress.style.width = `${ratio * 100}%`;
    toggleButton.classList.toggle("is-paused", !playing);
    toggleButton.setAttribute("aria-label", playing ? "Pause film preview" : "Play film preview");
    toggleText.textContent = playing ? "Pause" : "Play";
    sceneLabel.textContent = scenes[sceneIndex]?.dataset.label || "Opening";
    timecode.textContent = `${formatTime(position)} / ${formatTime(totalDuration)}`;
  };

  const seek = (seconds) => {
    position = Math.min(Math.max(seconds, 0), totalDuration);
    renderScene(sceneAt(Math.min(position, totalDuration - 0.001)));
    syncSceneCarousels();
    syncSceneReveals();
    syncVideos(sceneIndex);
    preloadUpcomingSceneVideos();
    renderControls();
  };

  const pause = () => {
    playing = false;
    syncVideos(sceneIndex);
    renderControls();
  };

  const play = () => {
    if (position >= totalDuration) {
      position = 0;
    }
    playing = true;
    lastFrame = performance.now();
    syncVideos(sceneIndex);
    renderControls();
  };

  const tick = (now) => {
    if (playing) {
      position += (now - lastFrame) / 1000;
      if (position >= totalDuration) {
        position = totalDuration;
        pause();
      }
      seek(position);
      preloadUpcomingSceneVideos();
    }

    lastFrame = now;
    animationFrame = window.requestAnimationFrame(tick);
  };

  toggleButton.addEventListener("click", () => {
    if (playing) {
      pause();
      return;
    }
    play();
  });

  restartButton?.addEventListener("click", () => {
    clearSceneTransitions();
    seek(0);
    play();
  });

  scrubber.addEventListener("input", () => {
    clearSceneTransitions();
    const nextPosition = (Number(scrubber.value) / Number(scrubber.max)) * totalDuration;
    seek(nextPosition);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) {
      return;
    }

    if (event.code === "Space") {
      event.preventDefault();
      toggleButton.click();
    }

    if (event.code === "ArrowRight") {
      clearSceneTransitions();
      seek(position + 2);
    }

    if (event.code === "ArrowLeft") {
      clearSceneTransitions();
      seek(position - 2);
    }
  });

  window.addEventListener("pagehide", () => {
    window.cancelAnimationFrame(animationFrame);
    clearSceneTransitions();
    scenes.forEach((scene) => {
      scene.querySelectorAll("video").forEach((video) => video.pause());
    });
  });

  hydrateMethodDrawing();
  seek(0);
  renderControls();
  syncVideos(sceneIndex);
  animationFrame = window.requestAnimationFrame(tick);
})();
