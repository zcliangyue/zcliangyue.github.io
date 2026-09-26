(() => {
  const slides = [...document.querySelectorAll("[data-slide]")];
  const chapterButtons = [...document.querySelectorAll("[data-slide-target]")];
  const nextButtons = [...document.querySelectorAll("[data-next]")];
  const previousButton = document.querySelector("#previousButton");
  const nextButton = document.querySelector("#nextButton");
  const slideCurrent = document.querySelector("#slideCurrent");
  const notes = document.querySelector("#speakerNotes");
  const noteText = document.querySelector("#noteText");
  const closeNotes = document.querySelector("#closeNotes");
  const timerButton = document.querySelector("#timerButton");
  const themeButton = document.querySelector("#themeButton");
  const fullscreenButton = document.querySelector("#fullscreenButton");
  const root = document.documentElement;

  let currentSlide = 0;
  let timerSeconds = 180;
  let timerHandle = null;

  const colors = {
    bg: "#f3f6fb",
    surface: "#e9eef6",
    text: "#121a28",
    muted: "#718096",
    line: "#b7c3d2",
    accent: "#005aff",
    accentSoft: "#dce8ff"
  };

  function isDark() {
    return root.dataset.theme === "dark";
  }

  function palette() {
    return isDark()
      ? { ...colors, bg: "#0d131d", surface: "#151e2b", text: "#eef4fb", muted: "#8293a8", line: "#40516a", accent: "#4b88ff", accentSoft: "#152d59" }
      : colors;
  }

  function setSlide(index) {
    currentSlide = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, slideIndex) => slide.classList.toggle("is-active", slideIndex === currentSlide));
    chapterButtons.forEach((button, buttonIndex) => button.classList.toggle("is-active", buttonIndex === currentSlide));
    slideCurrent.textContent = String(currentSlide + 1);
    previousButton.disabled = currentSlide === 0;
    nextButton.disabled = currentSlide === slides.length - 1;
    noteText.textContent = slides[currentSlide].dataset.note || "";

    if (currentSlide === 0) drawOverviewBev();
    if (currentSlide === 1) drawFeatureCanvas(document.querySelector("[data-representation].is-active")?.dataset.representation || "pixel");
    if (currentSlide === 3) drawBev(document.querySelector("[data-memory-mode].is-active")?.dataset.memoryMode || "anchor");
  }

  chapterButtons.forEach((button) => button.addEventListener("click", () => setSlide(Number(button.dataset.slideTarget))));
  nextButtons.forEach((button) => button.addEventListener("click", () => setSlide(currentSlide + 1)));
  previousButton.addEventListener("click", () => setSlide(currentSlide - 1));
  nextButton.addEventListener("click", () => setSlide(currentSlide + 1));

  function toggleNotes(force) {
    const shouldOpen = typeof force === "boolean" ? force : notes.hidden;
    notes.hidden = !shouldOpen;
  }

  closeNotes.addEventListener("click", () => toggleNotes(false));

  function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }

  function updateTimer() {
    timerButton.textContent = formatTime(timerSeconds);
    timerButton.classList.toggle("is-finished", timerSeconds === 0);
  }

  function toggleTimer() {
    if (timerSeconds === 0) {
      timerSeconds = 180;
      updateTimer();
    }

    if (timerHandle) {
      clearInterval(timerHandle);
      timerHandle = null;
      timerButton.classList.remove("is-running");
      return;
    }

    timerButton.classList.add("is-running");
    timerHandle = setInterval(() => {
      timerSeconds -= 1;
      updateTimer();
      if (timerSeconds <= 0) {
        clearInterval(timerHandle);
        timerHandle = null;
        timerButton.classList.remove("is-running");
      }
    }, 1000);
  }

  timerButton.addEventListener("click", toggleTimer);

  function toggleTheme() {
    const nextTheme = isDark() ? "light" : "dark";
    root.dataset.theme = nextTheme;
    themeButton.textContent = nextTheme === "dark" ? "浅色" : "深色";
    localStorage.setItem("interview-theme", nextTheme);
    drawOverviewBev();
    drawFeatureCanvas(document.querySelector("[data-representation].is-active")?.dataset.representation || "pixel");
    drawBev(document.querySelector("[data-memory-mode].is-active")?.dataset.memoryMode || "anchor");
  }

  themeButton.addEventListener("click", toggleTheme);

  async function toggleFullscreen() {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen?.();
    } else {
      await document.exitFullscreen?.();
    }
  }

  fullscreenButton.addEventListener("click", toggleFullscreen);

  document.addEventListener("keydown", (event) => {
    if (["ArrowRight", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      setSlide(currentSlide + 1);
    }
    if (["ArrowLeft", "PageUp"].includes(event.key)) {
      event.preventDefault();
      setSlide(currentSlide - 1);
    }
    if (event.key.toLowerCase() === "n") toggleNotes();
    if (event.key.toLowerCase() === "f") toggleFullscreen();
    if (event.key.toLowerCase() === "t") toggleTheme();
    if (/^[1-4]$/.test(event.key)) setSlide(Number(event.key) - 1);
    if (event.key === "Escape") toggleNotes(false);
  });

  function sizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(rect.width * ratio));
    const height = Math.max(1, Math.floor(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function drawOverviewBev() {
    const canvas = document.querySelector("#overviewBev");
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 2 || bounds.height < 2) return;
    const { ctx, width, height } = sizeCanvas(canvas);
    const c = palette();
    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    for (let x = 14; x < width; x += 34) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 14; y < height; y += 34) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    ctx.fillStyle = c.accentSoft;
    roundedRect(ctx, width * 0.12, height * 0.18, width * 0.44, height * 0.56, 14);
    ctx.fill();

    ctx.fillStyle = c.accent;
    roundedRect(ctx, width * 0.43, height * 0.47, 18, 36, 6);
    ctx.fill();

    ctx.strokeStyle = c.accent;
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 7]);
    ctx.beginPath();
    ctx.moveTo(width * 0.18, height * 0.68);
    ctx.quadraticCurveTo(width * 0.55, height * 0.12, width * 0.86, height * 0.58);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const representationButtons = [...document.querySelectorAll("[data-representation]")];
  representationButtons.forEach((button) => button.addEventListener("click", () => {
    representationButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    const mode = button.dataset.representation;
    document.querySelector("#representationTitle").textContent = mode === "shared" ? "共同表征" : "域间距离";
    document.querySelector("#representationCopy").textContent = mode === "shared" ? "CG 与 Real 的结构语义高度重合" : "输入图像的风格差异显著";
    document.querySelector("#featureSummary").textContent = mode === "shared" ? "结构对齐" : "域间分离";
    drawFeatureCanvas(mode);
  }));

  function featurePoint(index, cluster, mode, domain) {
    const seed = index * 12.9898 + cluster * 41.137 + domain * 7.31;
    const jitterX = Math.sin(seed) * 0.5 + Math.sin(seed * 1.73) * 0.22;
    const jitterY = Math.cos(seed * 1.21) * 0.5 + Math.cos(seed * 2.17) * 0.22;
    const sharedCenters = [[0.28, 0.31], [0.67, 0.35], [0.48, 0.7]];
    const separatedCG = [[0.19, 0.27], [0.56, 0.25], [0.32, 0.68]];
    const separatedReal = [[0.42, 0.42], [0.78, 0.46], [0.64, 0.72]];
    const centers = mode === "shared" ? sharedCenters : domain === 0 ? separatedCG : separatedReal;
    const center = centers[cluster];
    const domainOffset = mode === "shared" ? (domain === 0 ? -0.012 : 0.012) : 0;
    return [center[0] + domainOffset + jitterX * 0.08, center[1] + jitterY * 0.1];
  }

  function drawFeatureCanvas(mode) {
    const canvas = document.querySelector("#featureCanvas");
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 2 || bounds.height < 2) return;
    const { ctx, width, height } = sizeCanvas(canvas);
    const c = palette();
    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 7]);
    for (let i = 1; i < 5; i += 1) {
      ctx.beginPath(); ctx.moveTo(width * i / 5, 24); ctx.lineTo(width * i / 5, height - 24); ctx.stroke();
    }
    for (let i = 1; i < 4; i += 1) {
      ctx.beginPath(); ctx.moveTo(20, height * i / 4); ctx.lineTo(width - 20, height * i / 4); ctx.stroke();
    }
    ctx.setLineDash([]);

    for (let domain = 0; domain < 2; domain += 1) {
      for (let cluster = 0; cluster < 3; cluster += 1) {
        for (let index = 0; index < 24; index += 1) {
          const [px, py] = featurePoint(index, cluster, mode, domain);
          const x = px * width;
          const y = py * height;
          ctx.beginPath();
          ctx.arc(x, y, domain === 0 ? 4.5 : 4, 0, Math.PI * 2);
          if (domain === 0) {
            ctx.strokeStyle = c.muted;
            ctx.lineWidth = 1.4;
            ctx.stroke();
          } else {
            ctx.fillStyle = c.accent;
            ctx.globalAlpha = 0.72;
            ctx.fill();
            ctx.globalAlpha = 1;
          }
        }
      }
    }
  }

  const methodButtons = [...document.querySelectorAll("[data-method-stage]")];
  const methodCrop = document.querySelector("#methodCrop");
  const horizonVideo = document.querySelector("#horizonVideo");
  const methodExplanation = document.querySelector("#methodExplanation");
  const methodCopy = {
    recovery: ["真实误差收集 + 平滑注入", "让 Teacher 从自己的 rollout 误差中恢复，缓解曝光偏差。"],
    dmd: ["Rollout Teacher 监督少步 Student", "用 Teacher 的长视频分布指导短窗口 DMD，保留长时序能力。"],
    result: ["分钟级交互生成", "Student 以少步推理持续生成稳定、可控的驾驶视频。"]
  };
  const methodDiagrams = {
    recovery: '<div class="flow-node"><small>Base Model</small><strong>Teacher Rollout</strong></div><div class="flow-link"><span>真实误差</span><b>→</b></div><div class="flow-node"><small>Error Replay</small><strong>平滑注入</strong></div><div class="flow-link"><span>恢复训练</span><b>→</b></div><div class="flow-node is-accent"><small>Enhanced Teacher</small><strong>Recovery Teacher</strong></div>',
    dmd: '<div class="flow-node"><small>Long Horizon</small><strong>Rollout Teacher</strong></div><div class="flow-link"><span>生成分布</span><b>→</b></div><div class="flow-node"><small>Distribution Matching</small><strong>DMD</strong></div><div class="flow-link"><span>少步蒸馏</span><b>→</b></div><div class="flow-node is-accent"><small>Short Window</small><strong>Few-step Student</strong></div>'
  };
  const methodDiagram = document.querySelector("#methodDiagram");

  methodButtons.forEach((button) => button.addEventListener("click", () => {
    methodButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    const stage = button.dataset.methodStage;
    const isResult = stage === "result";
    methodCrop.hidden = isResult;
    horizonVideo.hidden = !isResult;
    methodCrop.classList.toggle("method-crop-recovery", stage === "recovery");
    methodCrop.classList.toggle("method-crop-dmd", stage === "dmd");
    if (!isResult) methodDiagram.innerHTML = methodDiagrams[stage];
    methodExplanation.innerHTML = `<strong>${methodCopy[stage][0]}</strong><span>${methodCopy[stage][1]}</span>`;
    if (isResult) horizonVideo.play().catch(() => {});
    else horizonVideo.pause();
  }));

  const memoryButtons = [...document.querySelectorAll("[data-memory-mode]")];
  const distillStage = document.querySelector(".distill-stage");
  const bevCaption = document.querySelector("#bevCaption");
  const memoryCopy = {
    anchor: "从历史观测中检索几何对应区域，保证空间一致性。",
    open: "对从未观测的区域保持开放式生成，不受历史内容限制。",
    unified: "同一学生模型按区域调用两种能力，兼顾复现与生成。"
  };

  memoryButtons.forEach((button) => button.addEventListener("click", () => {
    memoryButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    const mode = button.dataset.memoryMode;
    distillStage.dataset.mode = mode;
    bevCaption.textContent = memoryCopy[mode];
    drawBev(mode);
  }));

  function drawRoad(ctx, width, height, c) {
    ctx.fillStyle = c.surface;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = isDark() ? "#222d3b" : "#d9e0e8";
    ctx.fillRect(width * 0.08, 0, width * 0.27, height);
    ctx.fillRect(width * 0.63, 0, width * 0.27, height);
    ctx.fillRect(0, height * 0.36, width, height * 0.28);

    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.setLineDash([10, 10]);
    [0.17, 0.26, 0.72, 0.81].forEach((x) => {
      ctx.beginPath(); ctx.moveTo(width * x, 0); ctx.lineTo(width * x, height); ctx.stroke();
    });
    [0.45, 0.55].forEach((y) => {
      ctx.beginPath(); ctx.moveTo(0, height * y); ctx.lineTo(width, height * y); ctx.stroke();
    });
    ctx.setLineDash([]);
  }

  function drawBev(mode) {
    const canvas = document.querySelector("#bevCanvas");
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 2 || bounds.height < 2) return;
    const { ctx, width, height } = sizeCanvas(canvas);
    const c = palette();
    ctx.clearRect(0, 0, width, height);
    drawRoad(ctx, width, height, c);

    const anchorAlpha = mode === "open" ? 0.12 : 0.28;
    const openAlpha = mode === "anchor" ? 0.08 : 0.18;

    ctx.fillStyle = colorWithAlpha(c.accent, anchorAlpha);
    roundedRect(ctx, width * 0.04, height * 0.08, width * 0.49, height * 0.7, 16);
    ctx.fill();
    ctx.strokeStyle = colorWithAlpha(c.accent, mode === "open" ? 0.2 : 0.85);
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 7]);
    ctx.stroke();
    ctx.setLineDash([]);

    const hatchX = width * 0.58;
    const hatchY = height * 0.08;
    const hatchW = width * 0.38;
    const hatchH = height * 0.7;
    ctx.save();
    roundedRect(ctx, hatchX, hatchY, hatchW, hatchH, 16);
    ctx.clip();
    ctx.strokeStyle = colorWithAlpha(c.muted, openAlpha + 0.24);
    ctx.lineWidth = 1;
    for (let x = hatchX - hatchH; x < hatchX + hatchW + hatchH; x += 16) {
      ctx.beginPath(); ctx.moveTo(x, hatchY + hatchH); ctx.lineTo(x + hatchH, hatchY); ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = colorWithAlpha(c.muted, mode === "anchor" ? 0.2 : 0.72);
    ctx.lineWidth = 2;
    roundedRect(ctx, hatchX, hatchY, hatchW, hatchH, 16);
    ctx.stroke();

    const egoX = width * 0.5;
    const egoY = height * 0.56;
    ctx.fillStyle = c.accent;
    roundedRect(ctx, egoX - 12, egoY - 22, 24, 44, 7);
    ctx.fill();

    ctx.fillStyle = colorWithAlpha(c.accent, 0.12);
    ctx.beginPath();
    ctx.moveTo(egoX, egoY - 22);
    ctx.lineTo(egoX - width * 0.17, egoY - height * 0.36);
    ctx.lineTo(egoX + width * 0.17, egoY - height * 0.36);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = c.text;
    ctx.font = "600 13px ui-sans-serif";
    ctx.fillText("Historical observations", width * 0.08, height * 0.15);
    ctx.fillText("Unseen region", width * 0.66, height * 0.15);

    if (mode === "unified") {
      ctx.strokeStyle = c.accent;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(width * 0.2, height * 0.84);
      ctx.quadraticCurveTo(width * 0.5, height * 0.95, width * 0.8, height * 0.84);
      ctx.stroke();
      ctx.fillStyle = c.accent;
      ctx.fillText("region-aware unified inference", width * 0.35, height * 0.94);
    }
  }

  function colorWithAlpha(hex, alpha) {
    const value = hex.replace("#", "");
    const r = parseInt(value.slice(0, 2), 16);
    const g = parseInt(value.slice(2, 4), 16);
    const b = parseInt(value.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  const savedTheme = localStorage.getItem("interview-theme");
  const initialTheme = savedTheme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  root.dataset.theme = initialTheme;
  themeButton.textContent = initialTheme === "dark" ? "浅色" : "深色";
  distillStage.dataset.mode = "anchor";
  updateTimer();
  setSlide(0);

  let resizeHandle = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeHandle);
    resizeHandle = setTimeout(() => setSlide(currentSlide), 120);
  });

  document.querySelectorAll("video").forEach((video) => {
    video.addEventListener("error", () => {
      video.closest("figure, .method-visual")?.classList.add("media-error");
    });
  });
})();
