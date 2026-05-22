window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Icon effects: float + hover flutter + glow
function setupIconEffects() {
    const icon = document.querySelector('.publication-icon');
    if (!icon) return;
    icon.addEventListener('mouseenter', () => {
        icon.classList.add('spinning');
    });
    icon.addEventListener('mouseleave', () => {
        icon.classList.remove('spinning');
    });
}

// 3D tilt on stat cards
function setupTiltEffect() {
    document.querySelectorAll('.stat-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transition = 'transform 0.1s ease, box-shadow 0.1s ease';
        });
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width  - 0.5;
            const y = (e.clientY - rect.top)  / rect.height - 0.5;
            card.style.transform = `perspective(600px) rotateY(${x * 16}deg) rotateX(${-y * 16}deg) translateY(-6px)`;
            card.style.boxShadow = `${-x * 12}px ${y * 12}px 24px rgba(37,99,235,0.15)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transition = 'transform 0.4s ease, box-shadow 0.4s ease';
            card.style.transform = '';
            card.style.boxShadow = '';
        });
    });
}

// Stat card count-up animation
function setupCountUp() {
    const statNumbers = document.querySelectorAll('.stat-number[data-target]');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseInt(el.dataset.target);
            const prefix = el.dataset.prefix || '';
            const suffix = el.dataset.suffix || '';
            const duration = 1200;
            const startTime = performance.now();

            function update(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
                const current = Math.round(eased * target);
                el.textContent = prefix + current + suffix;
                if (progress < 1) requestAnimationFrame(update);
            }
            requestAnimationFrame(update);
            observer.unobserve(el);
        });
    }, { threshold: 0.5 });

    statNumbers.forEach(el => observer.observe(el));
}

// Scroll reveal animation
function setupScrollReveal() {
    const elements = document.querySelectorAll('.hero:not(:first-of-type), .section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.03 });

    elements.forEach(el => {
        el.classList.add('scroll-reveal');
        observer.observe(el);
    });
}

// Video autoplay when each video block scrolls into view.
function setupVideoAutoplay() {
    const videoBlocks = document.querySelectorAll('.publication-hero-bg, .video-carousel, .video-single, .video-stack');
    const gridVideos = Array.from(document.querySelectorAll('.video-grid video')).filter(video => !video.closest('.video-carousel'));
    if (videoBlocks.length === 0 && gridVideos.length === 0) return;

    function getActiveCarouselVideos(carousel) {
        const slides = Array.from(carousel.querySelectorAll('.video-carousel-slide'));
        const dots = Array.from(carousel.querySelectorAll('.video-carousel-dot'));
        let activeIndex = dots.findIndex(dot => dot.classList.contains('active'));
        if (activeIndex < 0) activeIndex = 0;
        const activeSlide = slides[activeIndex];
        return activeSlide ? Array.from(activeSlide.querySelectorAll('video')) : [];
    }

    function getPlayableVideos(block) {
        if (block.classList.contains('video-carousel')) {
            return getActiveCarouselVideos(block);
        }

        const syncMaster = block.querySelector('.video-sync-master');
        if (syncMaster) return [syncMaster];

        return Array.from(block.querySelectorAll('video'));
    }

    function getAllVideos(block) {
        return Array.from(block.querySelectorAll('video'));
    }

    function playBlock(block) {
        getPlayableVideos(block).forEach(video => {
            video.play().catch(() => {});
        });
    }

    function pauseBlock(block) {
        getAllVideos(block).forEach(video => {
            if (!video.paused) video.pause();
        });
    }

    function playVideo(video) {
        video.play().catch(() => {});
    }

    function pauseVideo(video) {
        if (!video.paused) video.pause();
    }

    const blockObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && entry.intersectionRatio >= 0.2) {
                playBlock(entry.target);
            } else {
                pauseBlock(entry.target);
            }
        });
    }, {
        rootMargin: '-8% 0px -8% 0px',
        threshold: [0, 0.2, 0.5]
    });

    const gridObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && entry.intersectionRatio >= 0.45) {
                playVideo(entry.target);
            } else {
                pauseVideo(entry.target);
            }
        });
    }, {
        rootMargin: '-10% 0px -10% 0px',
        threshold: [0, 0.25, 0.45, 0.75]
    });

    videoBlocks.forEach(block => blockObserver.observe(block));
    gridVideos.forEach(video => gridObserver.observe(video));
}

function setupCustomVideoControls() {
    var videos = Array.prototype.slice.call(document.querySelectorAll('main#main-content video'));

    videos.forEach(function (video) {
        if (video.closest('.publication-hero-bg')) return;
        if (video.dataset.customVideoControls === 'true') return;

        var parent = video.parentElement;
        if (!parent) return;

        var host = document.createElement('div');
        host.className = 'video-custom-control-host';
        parent.insertBefore(host, video);
        host.appendChild(video);

        var button = document.createElement('button');
        button.className = 'video-center-play';
        button.type = 'button';
        button.innerHTML = '<span aria-hidden="true"></span>';
        host.appendChild(button);

        video.dataset.customVideoControls = 'true';
        video.removeAttribute('controls');

        function updateButton() {
            var isPaused = video.paused || video.ended;
            host.classList.toggle('is-video-paused', isPaused);
            button.classList.toggle('is-playing', !isPaused);
            button.setAttribute('aria-label', isPaused ? 'Play video' : 'Pause video');
        }

        function toggleVideo(event) {
            if (event) event.preventDefault();
            if (video.paused || video.ended) {
                video.play().catch(function () {});
            } else {
                video.pause();
            }
            button.blur();
        }

        button.addEventListener('click', toggleVideo);
        video.addEventListener('click', toggleVideo);
        video.addEventListener('play', updateButton);
        video.addEventListener('pause', updateButton);
        video.addEventListener('ended', updateButton);
        updateButton();
    });
}

// Sync BEV video height to match paired Video height
function setupPlanningHeights() {
    const grid = document.querySelectorAll('.planning-pair');
    // Pairs are: [vid0, bev1, vid2, bev3, vid4, bev5, ...]
    // In each grid, odd-index .planning-pair cells are BEV
    const pairs = [];
    for (let i = 0; i < grid.length; i += 2) {
        pairs.push({ vid: grid[i].querySelector('video'), bev: grid[i+1] && grid[i+1].querySelector('video') });
    }

    function syncHeights() {
        pairs.forEach(({ vid, bev }) => {
            if (!vid || !bev) return;
            const h = vid.offsetHeight;
            if (h > 0) bev.style.height = h + 'px';
        });
    }

    pairs.forEach(({ vid }) => {
        if (!vid) return;
        vid.addEventListener('loadedmetadata', syncHeights);
        vid.addEventListener('resize', syncHeights);
    });
    window.addEventListener('resize', syncHeights);
    setTimeout(syncHeights, 500);
}

// Custom video carousel: left/right arrows + dots + keyboard navigation
function setupVideoCarousels() {
    document.querySelectorAll('.video-carousel:not(.video-carousel--stack-vertical)').forEach(carousel => {
        const slidesEl = carousel.querySelector('.video-carousel-slides');
        const slides = Array.from(carousel.querySelectorAll('.video-carousel-slide'));
        const prevBtn = carousel.querySelector('.video-carousel-arrow.prev');
        const nextBtn = carousel.querySelector('.video-carousel-arrow.next');
        const dotsContainer = carousel.querySelector('.video-carousel-dots');
        const counterEl = carousel.querySelector('.video-carousel-counter');
        const total = slides.length;
        let currentIndex = 0;

        if (total === 0) return;

        function setVideoLoaded(video, shouldLoad) {
            const source = video.querySelector('source');
            if (!source) return;

            if (!source.dataset.src && source.getAttribute('src')) {
                source.dataset.src = source.getAttribute('src');
            }

            const targetSrc = source.dataset.src;
            const currentSrc = source.getAttribute('src');

            if (shouldLoad) {
                if (targetSrc && currentSrc !== targetSrc) {
                    source.setAttribute('src', targetSrc);
                    video.load();
                }
                return;
            }

            if (currentSrc) {
                video.pause();
                source.removeAttribute('src');
                video.load();
            }
        }

        if (total <= 1) {
            if (prevBtn) prevBtn.style.display = 'none';
            if (nextBtn) nextBtn.style.display = 'none';
        }

        if (dotsContainer) {
            slides.forEach((_, i) => {
                const dot = document.createElement('button');
                dot.className = 'video-carousel-dot' + (i === 0 ? ' active' : '');
                dot.setAttribute('aria-label', 'Go to slide ' + (i + 1));
                dot.addEventListener('click', () => goTo(i));
                dotsContainer.appendChild(dot);
            });
        }

        function updateCounter() {
            if (counterEl) counterEl.textContent = (currentIndex + 1) + ' / ' + total;
        }

        function goTo(idx) {
            currentIndex = ((idx % total) + total) % total;
            slidesEl.style.transform = 'translateX(-' + (currentIndex * 100) + '%)';
            if (dotsContainer) {
                dotsContainer.querySelectorAll('.video-carousel-dot').forEach((d, i) => {
                    d.classList.toggle('active', i === currentIndex);
                });
            }
            slides.forEach((s, i) => {
                s.querySelectorAll('video').forEach((v) => {
                    setVideoLoaded(v, i === currentIndex);
                    if (i === currentIndex) {
                        v.play().catch(() => {});
                    } else {
                        v.pause();
                    }
                });
            });
            updateCounter();
        }

        if (prevBtn) prevBtn.addEventListener('click', () => goTo(currentIndex - 1));
        if (nextBtn) nextBtn.addEventListener('click', () => goTo(currentIndex + 1));

        carousel.tabIndex = 0;
        carousel.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft')  { e.preventDefault(); goTo(currentIndex - 1); }
            if (e.key === 'ArrowRight') { e.preventDefault(); goTo(currentIndex + 1); }
        });

        // Touch swipe support
        let touchStartX = null;
        carousel.addEventListener('touchstart', (e) => {
            touchStartX = e.touches[0].clientX;
        }, { passive: true });
        carousel.addEventListener('touchend', (e) => {
            if (touchStartX === null) return;
            const dx = e.changedTouches[0].clientX - touchStartX;
            if (Math.abs(dx) > 40) {
                goTo(currentIndex + (dx < 0 ? 1 : -1));
            }
            touchStartX = null;
        });

        goTo(0);
        updateCounter();
    });
}

/** nuScenes 并排：仅「播放」时顺带起播另一只；不绑暂停/进度，避免互相卡死 */
function setupNuscenesPairedVideos() {
    var sec = document.getElementById('sec-streaming-nusc');
    if (!sec) return;
    sec.querySelectorAll('.video-carousel-slide--pair').forEach(function (slide) {
        var videos = Array.prototype.slice.call(slide.querySelectorAll('video'));
        if (videos.length < 2) return;
        var depth = 0;
        videos.forEach(function (v) {
            v.addEventListener('play', function () {
                if (depth > 0) return;
                depth += 1;
                videos.forEach(function (other) {
                    if (other === v) return;
                    other.play().catch(function () {});
                });
                depth -= 1;
            });
        });
    });
}

function setupSyncedVideoStacks() {
    document.querySelectorAll('[data-video-sync]').forEach(function (stack) {
        var videos = Array.prototype.slice.call(stack.querySelectorAll('video'));
        var master = stack.querySelector('.video-sync-master') || videos[0];
        var followers = videos.filter(function (video) { return video !== master; });
        var syncing = false;

        if (!master || followers.length === 0) return;

        function syncTimes(force) {
            followers.forEach(function (video) {
                if (force || Math.abs(video.currentTime - master.currentTime) > 0.2) {
                    video.currentTime = master.currentTime;
                }
            });
        }

        master.addEventListener('play', function () {
            if (syncing) return;
            syncing = true;
            syncTimes(true);
            followers.forEach(function (video) {
                video.play().catch(function () {});
            });
            syncing = false;
        });

        master.addEventListener('pause', function () {
            if (syncing) return;
            followers.forEach(function (video) { video.pause(); });
        });

        master.addEventListener('seeking', function () { syncTimes(true); });
        master.addEventListener('ratechange', function () {
            followers.forEach(function (video) { video.playbackRate = master.playbackRate; });
        });
        master.addEventListener('timeupdate', function () { syncTimes(false); });
    });
}

function captureVideoFrame(src, timeSeconds) {
    return new Promise(function (resolve) {
        var video = document.createElement('video');
        var cleanup = function () {
            if (video.parentNode) video.parentNode.removeChild(video);
        };

        video.preload = 'auto';
        video.muted = true;
        video.playsInline = true;
        video.setAttribute('playsinline', '');
        video.style.position = 'absolute';
        video.style.left = '-9999px';
        video.style.top = '0';
        video.style.width = '1px';
        video.style.height = '1px';
        video.style.opacity = '0';
        video.src = src;
        document.body.appendChild(video);

        var fallback = function () {
            var canvas = document.createElement('canvas');
            canvas.width = 1280;
            canvas.height = 720;
            var ctx = canvas.getContext('2d');
            ctx.fillStyle = '#0b0f18';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#8ba3c7';
            ctx.font = 'bold 48px Inter, sans-serif';
            ctx.fillText('Frame unavailable', 44, 96);
            cleanup();
            resolve(canvas.toDataURL('image/png'));
        };

        video.addEventListener('error', fallback, { once: true });
        video.addEventListener('loadedmetadata', function () {
            var duration = Number.isFinite(video.duration) ? video.duration : timeSeconds;
            var targetTime = Math.max(0.1, Math.min(timeSeconds, Math.max(0.1, duration - 0.15)));

            var onSeeked = function () {
                try {
                    var canvas = document.createElement('canvas');
                    canvas.width = 1280;
                    canvas.height = 720;
                    var ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    cleanup();
                    resolve(canvas.toDataURL('image/jpeg', 0.9));
                } catch (err) {
                    fallback();
                }
            };

            video.addEventListener('seeked', onSeeked, { once: true });
            try {
                video.currentTime = targetTime;
            } catch (err) {
                fallback();
            }
        }, { once: true });
    });
}

function createNoiseDataUrl() {
    var canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    var ctx = canvas.getContext('2d');
    var imageData = ctx.createImageData(canvas.width, canvas.height);
    for (var i = 0; i < imageData.data.length; i += 4) {
        var value = 210 + Math.floor(Math.random() * 45);
        imageData.data[i] = value;
        imageData.data[i + 1] = value;
        imageData.data[i + 2] = value;
        imageData.data[i + 3] = Math.floor(Math.random() * 80);
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

async function setupMethodOverviewDiagram() {
    var host = document.getElementById('method-overview-diagram');
    if (!host) return;

    var videoSources = [
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/106_43da50265dfa4876bdf1efed38e7a261.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/109_1f048fa210c64e80b9d57e3dc757702c.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/116_852e22dbdbbf422e99224c488d7618bf.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/128_637fc435f95f4bc1b72f09811f2d78d2.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/142_21bb21c84c3e43908cd554db2686278f.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/19_b9db804449d34414ba98868e9034abd3.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/22_806011e8a7114eca971fd0f6b95bcd4b.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/25_8d56b71c09c04ece827107b3c103498f.mp4',
        'paper_videos_final/imported_nusc_hires_200ckpt_hud/nusc_hires_200ckpt_hud/35_c28c950afda14f7b93ff956799873dff.mp4'
    ];

    host.classList.remove('is-ready');
    host.innerHTML = '<div class="method-overview-loading">Drawing method diagram...</div>';

    try {
        var framePromises = videoSources.map(function (src, index) {
            return captureVideoFrame(src, 1.1 + index * 0.45);
        });
        var frames = await Promise.all(framePromises);
        var noiseUrl = createNoiseDataUrl();

        var tile = function (frameUrl, extraClass, focus) {
            var bgPos = focus || 'center center';
            return '<div class="method-frame ' + extraClass + '" style="background-image:url(\'' + frameUrl + '\');background-position:' + bgPos + ';"></div>';
        };

        var prepTiles = [
            tile(frames[0], '', 'center center'),
            tile(frames[1], '', 'center center'),
            tile(frames[2], '', 'center center'),
            tile(frames[3], '', 'center center'),
            tile(frames[4], '', 'center center'),
            tile(frames[5], '', 'center center'),
            tile(frames[6], '', 'center center'),
            tile(frames[7], '', 'center center'),
            tile(frames[8], '', 'center center')
        ].join('');

        var noisyTiles = [
            tile(frames[0], 'method-frame--noisy', 'center center'),
            tile(frames[1], 'method-frame--noisy', 'center center'),
            tile(frames[2], 'method-frame--noisy', 'center center'),
            tile(frames[3], 'method-frame--noisy', 'center center'),
            tile(frames[4], 'method-frame--strong-noisy', 'center center'),
            tile(frames[5], 'method-frame--noisy', 'center center'),
            tile(frames[6], 'method-frame--strong-noisy', 'center center'),
            tile(frames[7], 'method-frame--noisy', 'center center'),
            tile(frames[8], 'method-frame--strong-noisy', 'center center')
        ].join('');

        host.innerHTML = [
            '<div class="method-stage-banner method-stage-banner--stage1">Stage1: Conditional Driving WWM (Sec. 4.1)</div>',
            '<div class="method-arrow method-arrow--1" aria-hidden="true"><div class="method-arrow-box">&#9654;</div></div>',
            '<div class="method-stage-banner method-stage-banner--stage2">Stage2: Scheduled Rollout Recovery (Sec. 4.2)</div>',
            '<div class="method-arrow method-arrow--2" aria-hidden="true"><div class="method-arrow-box">&#9654;</div></div>',
            '<div class="method-stage-banner method-stage-banner--stage3">Stage3: Teacher Rollout DMD (Sec. 4.3)</div>',

            '<div class="method-region method-region--left">',
            '  <div class="method-input-stack">',
            '    <div class="method-input-card"><span class="method-input-icon"><i class="fas fa-road"></i></span><span class="method-input-label">HD Map</span></div>',
            '    <div class="method-input-card"><span class="method-input-icon"><i class="fas fa-cube"></i></span><span class="method-input-label">3D Bboxs</span></div>',
            '    <div class="method-input-card"><span class="method-input-icon"><i class="fas fa-crosshairs"></i></span><span class="method-input-label">Action</span></div>',
            '  </div>',
            '  <div class="method-model-box">',
            '    <div class="method-model-line method-model-line--top">Conditional</div>',
            '    <div class="method-model-network" aria-hidden="true">',
            '      <svg viewBox="0 0 1200 980" preserveAspectRatio="none">',
            '        <defs>',
            '          <linearGradient id="methodNetGrad" x1="0" x2="1">',
            '            <stop offset="0%" stop-color="#224fba"/>',
            '            <stop offset="100%" stop-color="#1f4aa1"/>',
            '          </linearGradient>',
            '        </defs>',
            '        <g fill="none" stroke="url(#methodNetGrad)" stroke-width="10" stroke-linecap="round" stroke-linejoin="round">',
            '          <path d="M80 140 L340 140 L600 280 L860 140 L1120 140"/>',
            '          <path d="M80 320 L340 320 L600 460 L860 320 L1120 320"/>',
            '          <path d="M80 500 L340 500 L600 640 L860 500 L1120 500"/>',
            '          <path d="M80 680 L340 680 L600 820 L860 680 L1120 680"/>',
            '          <path d="M340 140 L340 320 L340 500 L340 680"/>',
            '          <path d="M600 280 L600 460 L600 640 L600 820"/>',
            '          <path d="M860 140 L860 320 L860 500 L860 680"/>',
            '        </g>',
            '        <g fill="#1e52b6">',
            '          <circle cx="80" cy="140" r="34"/>',
            '          <circle cx="80" cy="320" r="34"/>',
            '          <circle cx="80" cy="500" r="34"/>',
            '          <circle cx="80" cy="680" r="34"/>',
            '          <circle cx="340" cy="140" r="34"/>',
            '          <circle cx="340" cy="320" r="34"/>',
            '          <circle cx="340" cy="500" r="34"/>',
            '          <circle cx="340" cy="680" r="34"/>',
            '          <circle cx="860" cy="140" r="34"/>',
            '          <circle cx="860" cy="320" r="34"/>',
            '          <circle cx="860" cy="500" r="34"/>',
            '          <circle cx="860" cy="680" r="34"/>',
            '          <circle cx="1120" cy="140" r="34"/>',
            '          <circle cx="1120" cy="320" r="34"/>',
            '          <circle cx="1120" cy="500" r="34"/>',
            '          <circle cx="1120" cy="680" r="34"/>',
            '        </g>',
          '      </svg>',
            '    </div>',
            '    <div class="method-model-line method-model-line--bottom">Driving WWM</div>',
            '  </div>',
            '</div>',

            '<div class="method-region method-region--middle">',
            '  <div class="method-region-heading">Long Horizon Data Preparation</div>',
            '  <div class="method-frame-grid method-frame-grid--clean" style="--method-noise-url:url(\'' + noiseUrl + '\');">' + prepTiles + '</div>',
            '</div>',

            '<div class="method-region method-region--right">',
            '  <div class="method-region-heading method-region-heading--green">Long Horizon Rollout</div>',
            '  <div class="method-frame-grid method-frame-grid--noisy" style="--method-noise-url:url(\'' + noiseUrl + '\');">' + noisyTiles + '</div>',
            '  <div class="method-rollout-footer">Rollout-capable Base Model</div>',
            '</div>'
        ].join('');

        host.classList.add('is-ready');
    } catch (err) {
        console.error('Method diagram render failed:', err);
        host.innerHTML = '<div class="method-overview-loading">Method diagram failed to render.</div>';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    setupIconEffects();
    setupTiltEffect();
    setupCountUp();
    setupScrollReveal();
    setupPlanningHeights();
    setupVideoCarousels();
    setupCustomVideoControls();
    setupNuscenesPairedVideos();
    setupSyncedVideoStacks();
    setupVideoAutoplay();
});
