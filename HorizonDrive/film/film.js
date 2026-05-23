(() => {
  const scenes = [...document.querySelectorAll("[data-scene]")];
  const toggleButton = document.querySelector("[data-toggle]");
  const toggleText = document.querySelector("[data-toggle-text]");
  const restartButton = document.querySelector("[data-restart]");
  const scrubber = document.querySelector("[data-scrubber]");
  const progress = document.querySelector("[data-progress]");
  const sceneLabel = document.querySelector("[data-scene-label]");
  const timecode = document.querySelector("[data-timecode]");
  const loader = document.querySelector("[data-film-loader]");
  const loaderProgress = document.querySelector("[data-film-loader-progress]");
  const loaderStatus = document.querySelector("[data-film-loader-status]");
  const filmStage = document.querySelector(".film-stage");

  if (!scenes.length || !toggleButton || !scrubber || !progress) {
    return;
  }

  const durations = scenes.map((scene) => Number(scene.dataset.duration) || 6);
  const sceneStarts = durations.map((_, index) => durations
    .slice(0, index)
    .reduce((sum, duration) => sum + duration, 0));
  const totalDuration = durations.reduce((sum, duration) => sum + duration, 0);
  const params = new URLSearchParams(window.location.search);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const recordMode = params.has("record");
  const renderMode = params.has("render");
  const renderHeroFrames = params.has("heroFrames");
  const renderVideoFrames = params.has("videoFrames");
  const renderHeroFrameBase = params.get("heroFrameBase") || "../static/render/hero-frames";
  const renderVideoFrameBase = params.get("videoFrameBase") || "../static/render/video-frames";
  const FILM_DESIGN_STAGE_WIDTH = Number(params.get("designWidth")) || 1700;
  const FILM_DESIGN_STAGE_HEIGHT = Number(params.get("designHeight")) || 940;
  const SCENE_CROSSFADE_MS = 2000;
  const SCENE_PRELOAD_SECONDS = 2;
  const REVEAL_DURATION_S = 0.56;
  const ABSTRACT_INSIGHT_DURATION_S = 0.52;
  let position = 0;
  let sceneIndex = -1;
  let playing = false;
  let preloadComplete = false;
  let lastFrame = performance.now();
  let animationFrame = 0;
  let leavingTimer = 0;
  const renderHeroImages = new Map();
  const renderVideoImages = new Map();
  const renderVideoMaxFrame = new Map();
  const renderImageLoadGeneration = new WeakMap();
  const RENDER_FRAME_FPS = 30;
  const landscapeGate = document.querySelector("[data-landscape-gate]");
  const landscapeEnterButton = document.querySelector("[data-landscape-enter]");
  const filmPlayer = document.querySelector(".film-player");
  const landscapeGateEnabled = !renderMode && !recordMode;
  const mobileFilmQuery = window.matchMedia("(max-width: 900px)");
  const landscapeOrientationQuery = window.matchMedia("(orientation: landscape)");
  let playWhenLandscapeAllowed = false;
  let landscapeGateActive = false;

  if (renderMode) {
    document.documentElement.classList.add("film-render-mode");
    const renderUiScale = Math.max(window.innerWidth / FILM_DESIGN_STAGE_WIDTH, 1);
    document.documentElement.style.setProperty("--film-render-ui-scale", String(renderUiScale));
  }

  if (renderHeroFrames || renderVideoFrames) {
    document.documentElement.classList.toggle("film-render-hero-frames", renderHeroFrames);
    document.documentElement.classList.toggle("film-render-video-frames", renderVideoFrames);
    let renderVideoIndex = 0;
    scenes.forEach((scene) => {
      scene.querySelectorAll("[data-scene-video]").forEach((video) => {
        const isHero = video.dataset.continuousVideo === "hero";
        if ((isHero && !renderHeroFrames) || (!isHero && !renderVideoFrames)) {
          return;
        }

        const frame = document.createElement("img");
        frame.className = `${video.className} ${isHero ? "render-hero-frame" : "render-video-frame"}`.trim();
        frame.alt = "";
        frame.decoding = "sync";
        video.after(frame);
        if (isHero) {
          renderHeroImages.set(video, frame);
          return;
        }

        video.dataset.renderFrameIndex = String(renderVideoIndex);
        renderVideoImages.set(video, frame);
        renderVideoIndex += 1;
      });
    });
  }

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
    if (preloadComplete) {
      scenes.forEach((scene) => {
        scene.querySelectorAll("[data-scene-video]").forEach((video) => {
          setSceneVideoLoaded(video, true);
        });
      });
      return;
    }

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
      scene.style.removeProperty("opacity");
    });
  };

  const preloadFilmVideos = () => {
    const videos = [...document.querySelectorAll("[data-scene-video]")];
    if (!videos.length) {
      preloadComplete = true;
      return Promise.resolve();
    }

    let readyCount = 0;
    let errorCount = 0;

    const renderPreloadProgress = () => {
      const ratio = readyCount / videos.length;
      if (loaderProgress) {
        loaderProgress.style.width = `${ratio * 100}%`;
      }
      if (loaderStatus) {
        loaderStatus.textContent = errorCount
          ? `${errorCount} video${errorCount === 1 ? "" : "s"} could not be prepared`
          : "Loading video scenes";
      }
    };

    renderPreloadProgress();

    return Promise.all(videos.map((video) => new Promise((resolve) => {
      let settled = false;
      const settle = (hasError = false) => {
        if (settled) {
          return;
        }
        settled = true;
        readyCount += 1;
        if (hasError) {
          errorCount += 1;
        }
        renderPreloadProgress();
        resolve();
      };

      video.preload = "auto";
      video.addEventListener("canplay", () => settle(), { once: true });
      video.addEventListener("error", () => settle(true), { once: true });
      setSceneVideoLoaded(video, true);

      if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
        settle();
      }
    }))).then(() => {
      preloadComplete = true;
      if (loaderStatus) {
        loaderStatus.textContent = errorCount ? "Film ready with missing video scenes" : "Film ready";
      }
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

  const syncContinuousSceneVideo = (previousIndex, nextIndex) => {
    const previousScene = scenes[previousIndex];
    const nextScene = scenes[nextIndex];
    if (!previousScene || !nextScene) {
      return;
    }

    const previousVideo = previousScene.querySelector("[data-continuous-video]");
    if (!(previousVideo instanceof HTMLVideoElement)) {
      return;
    }

    const nextVideo = nextScene.querySelector(`[data-continuous-video="${previousVideo.dataset.continuousVideo}"]`);
    if (!(nextVideo instanceof HTMLVideoElement)) {
      return;
    }

    const fromTime = Number.isFinite(previousVideo.currentTime) ? previousVideo.currentTime : 0;
    const applyCurrentTime = () => {
      const maxPlayable = Number.isFinite(nextVideo.duration) ? Math.max(0, nextVideo.duration - 0.08) : fromTime;
      nextVideo.currentTime = Math.min(fromTime, maxPlayable);
    };

    if (nextVideo.readyState >= 1) {
      applyCurrentTime();
      return;
    }

    nextVideo.addEventListener("loadedmetadata", applyCurrentTime, { once: true });
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

  const smoothstep = (value) => {
    const t = Math.min(Math.max(value, 0), 1);
    return t * t * (3 - 2 * t);
  };

  const applyRenderRevealProgress = (element, progress) => {
    const eased = smoothstep(progress);
    element.classList.toggle("is-revealed", eased >= 1);

    if (element.classList.contains("feature-rail")) {
      element.style.opacity = String(eased);
      element.style.transform = `translateX(${-22 * (1 - eased)}px) translateY(${10 * (1 - eased)}px)`;
      return;
    }

    if (element.classList.contains("method-drawing-frame")) {
      element.style.opacity = String(eased);
      element.style.visibility = eased > 0 ? "visible" : "hidden";
      return;
    }

    element.style.opacity = String(eased);
  };

  const syncRenderReveals = () => {
    scenes.forEach((scene, index) => {
      const elapsed = sceneElapsed(index);

      scene.querySelectorAll("[data-scene-reveal]").forEach((element) => {
        const revealAt = Number(element.dataset.revealAt) || durations[index] / 2;
        applyRenderRevealProgress(element, (elapsed - revealAt) / REVEAL_DURATION_S);
      });

      scene.querySelectorAll(".abstract-key-insight").forEach((element) => {
        const eased = smoothstep(elapsed / ABSTRACT_INSIGHT_DURATION_S);
        element.classList.toggle("is-revealed", eased >= 1);
        element.style.opacity = String(eased);
        element.style.transform = `translateY(${10 * (1 - eased)}px)`;
      });
    });
  };

  const clearRenderPresentationStyles = () => {
    scenes.forEach((scene) => {
      scene.style.removeProperty("opacity");
      scene.style.removeProperty("transform");
    });

    document.querySelectorAll("[data-scene-reveal], .abstract-key-insight").forEach((element) => {
      element.style.removeProperty("opacity");
      element.style.removeProperty("transform");
      element.style.removeProperty("visibility");
    });
  };

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

  const resetFilmVideos = () => {
    scenes.forEach((scene) => {
      scene.querySelectorAll("[data-scene-video]").forEach((video) => {
        video.pause();

        if (video.dataset.tailSeconds) {
          resetTailVideo(video);
          return;
        }

        const resetCurrentTime = () => {
          video.currentTime = 0;
        };

        if (video.readyState >= 1) {
          resetCurrentTime();
          return;
        }

        video.addEventListener("loadedmetadata", resetCurrentTime, { once: true });
      });
    });
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
    if (renderMode) {
      syncRenderReveals();
      return;
    }

    scenes.forEach((scene, index) => {
      scene.querySelectorAll("[data-scene-reveal]").forEach((element) => {
        const revealAt = Number(element.dataset.revealAt) || durations[index] / 2;
        element.classList.toggle("is-revealed", sceneElapsed(index) >= revealAt);
      });
    });
  };

  const isMobileFilmDevice = () => {
    if (!mobileFilmQuery.matches) {
      return false;
    }

    return window.matchMedia("(hover: none) and (pointer: coarse)").matches
      || navigator.maxTouchPoints > 0;
  };

  const isLandscapeOrientation = () => landscapeOrientationQuery.matches;

  const canPlayFilm = () => {
    if (!landscapeGateEnabled || !isMobileFilmDevice()) {
      return true;
    }

    return isLandscapeOrientation();
  };

  const pauseAllSceneVideos = () => {
    scenes.forEach((scene) => {
      scene.querySelectorAll("[data-scene-video]").forEach((video) => video.pause());
    });
  };

  const syncFilmStageFit = () => {
    const active = landscapeGateEnabled
      && isMobileFilmDevice()
      && isLandscapeOrientation()
      && !landscapeGateActive;

    document.documentElement.classList.toggle("film-mobile-fit", active);

    if (!active) {
      document.documentElement.style.removeProperty("--film-fit-scale");
      document.documentElement.style.removeProperty("--film-fit-offset-x");
      document.documentElement.style.removeProperty("--film-fit-offset-y");
      return;
    }

    const viewport = window.visualViewport;
    const viewportWidth = viewport?.width ?? window.innerWidth;
    const viewportHeight = viewport?.height ?? window.innerHeight;
    const scaleX = viewportWidth / FILM_DESIGN_STAGE_WIDTH;
    const scaleY = viewportHeight / FILM_DESIGN_STAGE_HEIGHT;
    const fitScale = Math.min(scaleX, scaleY) * 0.94;

    document.documentElement.style.setProperty("--film-fit-scale", String(fitScale));
    document.documentElement.style.removeProperty("--film-fit-offset-x");
    document.documentElement.style.removeProperty("--film-fit-offset-y");
  };

  const updateLandscapeGate = () => {
    if (!landscapeGateEnabled) {
      return;
    }

    const mobile = isMobileFilmDevice();
    const landscape = isLandscapeOrientation();
    landscapeGateActive = mobile && !landscape;

    document.documentElement.classList.toggle("film-mobile-device", mobile);
    document.documentElement.classList.toggle("film-mobile-locked", landscapeGateActive);
    document.documentElement.classList.toggle("film-mobile-landscape", mobile && landscape);

    if (landscapeGate) {
      landscapeGate.hidden = !landscapeGateActive;
    }

    if (landscapeGateActive) {
      if (playing) {
        playWhenLandscapeAllowed = true;
        playing = false;
        pauseAllSceneVideos();
        renderControls();
      }
      syncFilmStageFit();
      return;
    }

    if (mobile && landscape && playWhenLandscapeAllowed && preloadComplete) {
      playWhenLandscapeAllowed = false;
      play({ skipLandscapeCheck: true });
    }

    syncFilmStageFit();
  };

  const enterLandscapeFullscreen = async () => {
    playWhenLandscapeAllowed = true;
    const target = filmPlayer || document.documentElement;

    try {
      if (!document.fullscreenElement && target.requestFullscreen) {
        await target.requestFullscreen();
      }
    } catch (_) {
      // Fullscreen may be blocked until the next explicit user gesture.
    }

    try {
      if (screen.orientation?.lock) {
        await screen.orientation.lock("landscape");
      }
    } catch (_) {
      // Orientation lock generally requires fullscreen and still may fail on iOS.
    }

    updateLandscapeGate();
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
    if (landscapeGateEnabled && !canPlayFilm()) {
      pauseAllSceneVideos();
      return;
    }

    scenes.forEach((scene, index) => {
      const isLeaving = scene.classList.contains("is-leaving");
      const shouldPlayScene = playing && (index === activeIndex || isLeaving);

      scene.querySelectorAll("[data-scene-video]").forEach((video) => {
        video.playbackRate = Number(scene.dataset.playbackRate) || 1;
        if (shouldPlayScene && canPlaySceneVideo(video)) {
          if (video.paused) {
            video.play().catch(() => {});
          }
          return;
        }

        if (!isLeaving && !video.paused) {
          video.pause();
        }
      });
    });
  };

  const setRenderSceneTransition = () => {
    if (!renderMode) {
      return;
    }

    scenes.forEach((scene, index) => {
      if (index !== sceneIndex && index !== sceneIndex - 1) {
        scene.style.removeProperty("opacity");
        scene.style.removeProperty("transform");
      }
    });

    if (sceneIndex <= 0) {
      return;
    }

    const transitionSeconds = SCENE_CROSSFADE_MS / 1000;
    const transitionElapsed = position - (sceneStarts[sceneIndex] || 0);
    const leavingScene = scenes[sceneIndex - 1];
    const activeScene = scenes[sceneIndex];

    if (transitionElapsed < 0 || transitionElapsed >= transitionSeconds) {
      leavingScene.classList.remove("is-leaving", "is-exiting");
      leavingScene.style.removeProperty("opacity");
      activeScene.style.removeProperty("opacity");
      activeScene.style.removeProperty("transform");
      return;
    }

    const fade = smoothstep(transitionElapsed / transitionSeconds);
    leavingScene.classList.add("is-leaving", "is-exiting");
    leavingScene.style.opacity = String(1 - fade);
    activeScene.style.opacity = String(fade);
    activeScene.style.transform = `scale(${1.02 - (0.02 * fade)})`;
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

    if (previousIndex >= 0 && previousIndex !== sceneIndex && !reduceMotion && !renderMode) {
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

    syncContinuousSceneVideo(previousIndex, sceneIndex);
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

  const seekVideoFrame = (video, targetTime) => new Promise((resolve) => {
    if (!Number.isFinite(targetTime) || video.readyState < 1) {
      resolve();
      return;
    }

    const maxTime = Number.isFinite(video.duration)
      ? Math.max(0, video.duration - 0.06)
      : Math.max(0, targetTime);
    const nextTime = Math.min(Math.max(targetTime, 0), maxTime);
    if (Math.abs(video.currentTime - nextTime) < 0.02 && video.readyState >= 2) {
      resolve();
      return;
    }

    let settled = false;
    const settle = () => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", settle);
      video.removeEventListener("error", settle);
      resolve();
    };
    const timeout = window.setTimeout(settle, 1800);

    video.addEventListener("seeked", settle);
    video.addEventListener("error", settle);
    video.currentTime = nextTime;
  });

  const buildRenderFrameSrc = (base, frameIndex) =>
    `${base}/frame-${String(frameIndex).padStart(5, "0")}.jpg`;

  const buildRenderVideoFrameSrc = (renderIndex, frameIndex) =>
    `${renderVideoFrameBase}/video-${renderIndex}/frame-${String(frameIndex).padStart(5, "0")}.jpg`;

  const waitForRenderPaint = () => new Promise((resolve) => {
    window.requestAnimationFrame(() => window.requestAnimationFrame(resolve));
  });

  const loadRenderImageFrame = (frame, nextSrc, fallbackSrc) => {
    if (!frame) {
      return Promise.resolve();
    }

    if (frame.getAttribute("src") === nextSrc && frame.complete && frame.naturalWidth > 0) {
      return Promise.resolve();
    }

    const generation = (renderImageLoadGeneration.get(frame) || 0) + 1;
    renderImageLoadGeneration.set(frame, generation);

    const waitForSrc = (src) => new Promise((resolve) => {
      if (renderImageLoadGeneration.get(frame) !== generation) {
        resolve(false);
        return;
      }

      const finish = (loaded) => {
        if (renderImageLoadGeneration.get(frame) !== generation) {
          return;
        }
        frame.removeEventListener("load", onLoad);
        frame.removeEventListener("error", onError);
        resolve(loaded);
      };

      const onLoad = async () => {
        if (typeof frame.decode === "function") {
          try {
            await frame.decode();
          } catch (_) {
            // Ignore decode failures; naturalWidth check below decides success.
          }
        }
        finish(frame.naturalWidth > 0);
      };

      const onError = () => finish(false);

      if (frame.getAttribute("src") === src && frame.complete) {
        onLoad();
        return;
      }

      frame.addEventListener("load", onLoad);
      frame.addEventListener("error", onError);
      frame.src = src;
    });

    return (async () => {
      let loaded = await waitForSrc(nextSrc);
      if (!loaded && fallbackSrc && fallbackSrc !== nextSrc) {
        loaded = await waitForSrc(fallbackSrc);
      }

      if (!loaded) {
        const lastGood = frame.dataset.renderLastGoodSrc;
        if (lastGood && lastGood !== frame.getAttribute("src")) {
          await waitForSrc(lastGood);
        }
        return;
      }

      frame.dataset.renderLastGoodSrc = frame.getAttribute("src");
    })();
  };

  const loadRenderHeroFrame = (video) => {
    const frame = renderHeroImages.get(video);
    if (!frame) {
      return Promise.resolve();
    }

    const heroFrame = Math.min(Math.max(Math.round(position * RENDER_FRAME_FPS), 0), 899);
    const nextSrc = buildRenderFrameSrc(renderHeroFrameBase, heroFrame);
    const fallbackSrc = buildRenderFrameSrc(renderHeroFrameBase, 0);
    return loadRenderImageFrame(frame, nextSrc, fallbackSrc);
  };

  const loadRenderVideoFrame = (video, targetTime) => {
    const frame = renderVideoImages.get(video);
    if (!frame || !Number.isFinite(targetTime)) {
      return Promise.resolve();
    }

    const renderIndex = String(Number(video.dataset.renderFrameIndex) || 0).padStart(3, "0");
    let targetFrame = Math.max(Math.round(targetTime * RENDER_FRAME_FPS), 0);
    const maxFrame = renderVideoMaxFrame.get(video);
    if (Number.isFinite(maxFrame)) {
      targetFrame = Math.min(targetFrame, maxFrame);
    }

    const nextSrc = buildRenderVideoFrameSrc(renderIndex, targetFrame);
    const fallbackSrc = buildRenderVideoFrameSrc(renderIndex, 0);
    return loadRenderImageFrame(frame, nextSrc, fallbackSrc);
  };

  const cacheRenderVideoMaxFrames = () => {
    renderVideoImages.forEach((_, video) => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        return;
      }
      renderVideoMaxFrame.set(
        video,
        Math.max(0, Math.floor(video.duration * RENDER_FRAME_FPS) - 1)
      );
    });
  };

  const prewarmRenderVideoFrames = () => Promise.all(
    [...renderVideoImages.keys()].map((video) => loadRenderVideoFrame(video, 0))
  );

  const renderVideoTime = (scene, index, video) => {
    if (video.dataset.continuousVideo) {
      return position;
    }

    let elapsed = sceneElapsed(index);
    const carousel = video.closest("[data-film-carousel]");
    const page = video.closest("[data-film-page]");
    if (carousel && page) {
      const pages = [...carousel.querySelectorAll("[data-film-page]")];
      const pageDuration = Number(carousel.dataset.pageDuration) || 8;
      elapsed -= Math.max(pages.indexOf(page), 0) * pageDuration;
    }

    const rate = Number(scene.dataset.playbackRate) || 1;
    if (video.dataset.tailSeconds && Number.isFinite(video.duration)) {
      return Math.max(0, video.duration - Number(video.dataset.tailSeconds)) + Math.max(elapsed, 0) * rate;
    }

    return Math.max(elapsed, 0) * rate;
  };

  const renderOfflineFrame = async (seconds) => {
    if (!preloadComplete) {
      return null;
    }

    playing = false;
    clearSceneTransitions();
    clearRenderPresentationStyles();
    seek(seconds);
    setRenderSceneTransition();
    syncSceneReveals();
    document.documentElement.style.setProperty("--film-render-time", `${position}s`);

    const frameSeeks = [];
    scenes.forEach((scene, index) => {
      const shouldRenderScene = index === sceneIndex || scene.classList.contains("is-leaving");
      scene.querySelectorAll("[data-scene-video]").forEach((video) => {
        video.pause();
        if (renderHeroFrames && video.dataset.continuousVideo === "hero") {
          if (shouldRenderScene) {
            frameSeeks.push(loadRenderHeroFrame(video));
          }
          return;
        }
        if (renderVideoFrames) {
          if (shouldRenderScene && canPlaySceneVideo(video)) {
            frameSeeks.push(loadRenderVideoFrame(video, renderVideoTime(scene, index, video)));
          }
          return;
        }
        if (shouldRenderScene && canPlaySceneVideo(video)) {
          frameSeeks.push(seekVideoFrame(video, renderVideoTime(scene, index, video)));
        }
      });
    });

    await Promise.all(frameSeeks);
    await waitForRenderPaint();
    renderControls();
    return {
      position,
      sceneIndex,
      sceneLabel: scenes[sceneIndex]?.dataset.label || "Opening",
    };
  };

  const pause = () => {
    playing = false;
    syncVideos(sceneIndex);
    renderControls();
  };

  const play = (options = {}) => {
    if (!preloadComplete) {
      return;
    }

    if (!options.skipLandscapeCheck && !canPlayFilm()) {
      playWhenLandscapeAllowed = true;
      updateLandscapeGate();
      return;
    }

    if (position >= totalDuration) {
      clearSceneTransitions();
      resetFilmVideos();
      seek(0);
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
    if (!preloadComplete) {
      return;
    }

    if (playing) {
      pause();
      return;
    }
    play();
  });

  restartButton?.addEventListener("click", () => {
    if (!preloadComplete) {
      return;
    }

    clearSceneTransitions();
    resetFilmVideos();
    seek(0);
    play();
  });

  scrubber.addEventListener("input", () => {
    if (!preloadComplete || !canPlayFilm()) {
      return;
    }

    clearSceneTransitions();
    const nextPosition = (Number(scrubber.value) / Number(scrubber.max)) * totalDuration;
    seek(nextPosition);
  });

  const bindMobileTimelineSwipe = () => {
    if (!landscapeGateEnabled || !filmStage) {
      return;
    }

    let touchScrubbing = false;
    let touchStartX = 0;
    let touchStartY = 0;
    let touchStartPosition = 0;
    let touchScrubWidth = 1;
    let touchWasPlaying = false;
    let touchAxisResolved = false;
    let touchIsHorizontal = false;

    const canScrub = () => preloadComplete && isMobileFilmDevice() && canPlayFilm();

    const seekFromTouch = (clientX) => {
      const deltaRatio = (clientX - touchStartX) / touchScrubWidth;
      const nextPosition = Math.min(
        Math.max(touchStartPosition + deltaRatio * totalDuration, 0),
        totalDuration
      );
      clearSceneTransitions();
      seek(nextPosition);
    };

    const endTouchScrub = () => {
      if (!touchScrubbing) {
        return;
      }

      touchScrubbing = false;
      touchAxisResolved = false;
      touchIsHorizontal = false;
      filmStage.classList.remove("is-touch-scrubbing");

      if (touchWasPlaying) {
        play();
      }
    };

    filmStage.addEventListener("touchstart", (event) => {
      if (!canScrub() || event.touches.length !== 1) {
        return;
      }

      touchScrubbing = true;
      touchAxisResolved = false;
      touchIsHorizontal = false;
      touchWasPlaying = false;
      touchStartX = event.touches[0].clientX;
      touchStartY = event.touches[0].clientY;
      touchStartPosition = position;
      touchScrubWidth = Math.max(filmStage.getBoundingClientRect().width, 1);
    }, { passive: true });

    filmStage.addEventListener("touchmove", (event) => {
      if (!touchScrubbing || event.touches.length !== 1) {
        return;
      }

      const deltaX = event.touches[0].clientX - touchStartX;
      const deltaY = event.touches[0].clientY - touchStartY;

      if (!touchAxisResolved) {
        if (Math.abs(deltaX) < 8 && Math.abs(deltaY) < 8) {
          return;
        }

        touchAxisResolved = true;
        touchIsHorizontal = Math.abs(deltaX) >= Math.abs(deltaY);
        if (!touchIsHorizontal) {
          touchScrubbing = false;
          return;
        }

        touchWasPlaying = playing;
        if (playing) {
          pause();
        }
        filmStage.classList.add("is-touch-scrubbing");
      }

      if (!touchIsHorizontal) {
        return;
      }

      seekFromTouch(event.touches[0].clientX);
      event.preventDefault();
    }, { passive: false });

    filmStage.addEventListener("touchend", endTouchScrub);
    filmStage.addEventListener("touchcancel", endTouchScrub);
  };

  bindMobileTimelineSwipe();

  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) {
      return;
    }

    if (!preloadComplete || !canPlayFilm()) {
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

  renderControls();
  if (landscapeGateEnabled) {
    landscapeEnterButton?.addEventListener("click", () => {
      enterLandscapeFullscreen();
    });
    mobileFilmQuery.addEventListener("change", updateLandscapeGate);
    landscapeOrientationQuery.addEventListener("change", updateLandscapeGate);
    document.addEventListener("fullscreenchange", updateLandscapeGate);
    window.addEventListener("resize", syncFilmStageFit);
    window.visualViewport?.addEventListener("resize", syncFilmStageFit);
    window.visualViewport?.addEventListener("scroll", syncFilmStageFit);
    updateLandscapeGate();
  } else {
    syncFilmStageFit();
  }

  const filmReady = preloadFilmVideos().then(async () => {
    loader?.classList.add("is-ready");
    if (renderMode && renderVideoFrames) {
      cacheRenderVideoMaxFrames();
      await prewarmRenderVideoFrames();
    }
    seek(0);
    if (!reduceMotion && !recordMode && !renderMode) {
      if (canPlayFilm()) {
        play();
      } else {
        playWhenLandscapeAllowed = true;
        updateLandscapeGate();
      }
      return;
    }
    renderControls();
  });

  if (renderMode) {
    window.horizonDriveFilmRender = {
      ready: filmReady,
      renderAt: renderOfflineFrame,
      totalDuration,
    };
    return;
  }

  animationFrame = window.requestAnimationFrame(tick);
})();
