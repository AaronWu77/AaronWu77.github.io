/* --- 摄影集“灯箱”效果逻辑 --- */

// 这是一个独立的功能块，只有当页面上存在 id="lightbox" 时才会生效
const lightbox = document.getElementById('lightbox');
if (lightbox) {
    const lightboxImg = document.getElementById('lightbox-img');
    const captionText = document.getElementById('caption');
    const closeBtn = document.querySelector('.close-btn');

    // 1. 监听所有图片的点击事件
    const galleryContainer = document.querySelector('.gallery-container');
    if (galleryContainer) {
        galleryContainer.addEventListener('click', (e) => {
            if (e.target.tagName === 'IMG') {
                const img = e.target;
                
                // 显示灯箱
                lightbox.style.display = "flex";
                setTimeout(() => lightbox.classList.add('show'), 10);
                
                // 智能判断：如果标签上有 data-full-src (高清图地址)，就加载高清图；否则还用原来的图
                const highResUrl = img.getAttribute('data-full-src');
                lightboxImg.src = highResUrl ? highResUrl : img.src; 
                
                const container = img.closest('.stream-item') || img.closest('.photo-item');
                const titleSource = container ? container.querySelector('.stream-caption, .photo-overlay') : null;
                const metaSources = container ? Array.from(container.querySelectorAll('.stream-meta')) : [];

                captionText.innerHTML = '';
                if (titleSource && titleSource.textContent.trim()) {
                    const titleLine = document.createElement('div');
                    titleLine.className = 'lightbox-title';
                    titleLine.textContent = titleSource.textContent.trim();
                    captionText.appendChild(titleLine);
                }
                metaSources.forEach((meta) => {
                    if (!meta.textContent.trim()) return;
                    const metaLine = document.createElement('div');
                    metaLine.className = 'lightbox-meta';
                    metaLine.textContent = meta.textContent.trim();
                    captionText.appendChild(metaLine);
                });
                if (!captionText.childNodes.length) {
                    captionText.textContent = '';
                }
            }
        });
    }

    // 2. 关闭灯箱的函数
    function closeLightbox() {
        lightbox.classList.remove('show');
        setTimeout(() => {
            lightbox.style.display = "none";
        }, 300);
    }

    // 点击关闭按钮
    if (closeBtn) {
        closeBtn.addEventListener('click', closeLightbox);
    }

    // 点击灯箱背景也能关闭
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });

    // 按 ESC 键关闭
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && lightbox.classList.contains('show')) {
            closeLightbox();
        }
    });
}

/* --- 图片加载优化：多尺寸 WebP 响应式加载 --- */
const PHOTO_OPTIMIZED_WIDTHS = [480, 960, 1600];
const webpAvailabilityCache = new Map();

function getOptimizedBasePath(src) {
    const url = new URL(src, window.location.href);
    const marker = '/assets/images/';
    const idx = url.pathname.indexOf(marker);
    if (idx === -1) return null;
    const relPath = url.pathname.slice(idx + marker.length);
    const dotIndex = relPath.lastIndexOf('.');
    const base = dotIndex === -1 ? relPath : relPath.slice(0, dotIndex);
    return '/assets/images/optimized/' + base;
}

function buildWebpSrcset(src) {
    const base = getOptimizedBasePath(src);
    if (!base) return null;
    return PHOTO_OPTIMIZED_WIDTHS.map((width) => {
        const path = encodeURI(base + '-' + width + 'w.webp');
        return `${path} ${width}w`;
    }).join(', ');
}

function checkWebpAvailability(src) {
    const base = getOptimizedBasePath(src);
    if (!base) return Promise.resolve(false);
    if (webpAvailabilityCache.has(base)) {
        return Promise.resolve(webpAvailabilityCache.get(base));
    }
    const probe = encodeURI(base + '-' + PHOTO_OPTIMIZED_WIDTHS[0] + 'w.webp');
    return fetch(probe, { method: 'HEAD' })
        .then((response) => {
            const ok = response.ok;
            webpAvailabilityCache.set(base, ok);
            return ok;
        })
        .catch(() => {
            webpAvailabilityCache.set(base, false);
            return false;
        });
}

function upgradeResponsiveImage(img, sizesAttr) {
    if (!img || img.dataset.optimized === 'true') return;
    const originalSrc = img.getAttribute('src');
    if (!originalSrc) return;

    checkWebpAvailability(originalSrc).then((available) => {
        if (!available || img.dataset.optimized === 'true') return;
        const webpSrcset = buildWebpSrcset(originalSrc);
        if (!webpSrcset) return;

        if (img.closest('.gallery-container') && !img.getAttribute('data-full-src')) {
            img.setAttribute('data-full-src', originalSrc);
        }

        img.decoding = 'async';
        if (!img.getAttribute('loading')) {
            img.setAttribute('loading', 'lazy');
        }
        if (sizesAttr) {
            img.setAttribute('sizes', sizesAttr);
        }

        const picture = document.createElement('picture');
        const source = document.createElement('source');
        source.type = 'image/webp';
        source.srcset = webpSrcset;
        if (sizesAttr) {
            source.setAttribute('sizes', sizesAttr);
        }
        picture.appendChild(source);

        const parent = img.parentNode;
        parent.insertBefore(picture, img);
        picture.appendChild(img);
        img.dataset.optimized = 'true';
    });
}

function applyResponsiveImages() {
    const coverSizes = '(max-width: 600px) 100vw, (max-width: 900px) 50vw, 33vw';
    const streamSizes = '(max-width: 600px) 100vw, (max-width: 1200px) 90vw, 1000px';
    const gallerySizes = '(max-width: 800px) 100vw, 400px';

    const coverImages = Array.from(document.querySelectorAll('.album-cover-img'));
    coverImages.forEach((img, index) => {
        if (index === 0) img.setAttribute('fetchpriority', 'high');
        upgradeResponsiveImage(img, coverSizes);
    });

    const streamImages = Array.from(document.querySelectorAll('.stream-img'));
    streamImages.forEach((img, index) => {
        if (index === 0) img.setAttribute('fetchpriority', 'high');
        upgradeResponsiveImage(img, streamSizes);
    });

    document.querySelectorAll('.gallery-container img').forEach((img) => {
        upgradeResponsiveImage(img, gallerySizes);
    });
}

applyResponsiveImages();

/* --- 动态激光背景特效 (Canvas) --- */
/* 霓虹线条背景：长线条、随机角度、缓慢漂浮、呼吸闪烁 */
(function() {
    const canvas = document.createElement('canvas');
    canvas.id = 'cyber-bg';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.zIndex = '-2';
    canvas.style.pointerEvents = 'none';
    canvas.style.background = 'transparent';
    document.body.prepend(canvas);

    const ctx = canvas.getContext('2d');
    let width, height;
    let lines = [];

    // --- 鼠标交互追踪 ---
    let mouse = { x: null, y: null };
    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });
    window.addEventListener('mouseleave', () => {
        mouse.x = null;
        mouse.y = null;
    });

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    class NeonLine {
        constructor() {
            this.reset(true);
        }

        reset(initial = false) {
            // 随机中心点
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            
            // 随机长度
            this.length = Math.random() * width * 0.8 + 500;
            
            // 随机角度
            this.angle = Math.random() * Math.PI * 2;
            
            // 漂浮速度 (不负责弹拨，只负责移动)
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.vr = (Math.random() - 0.5) * 0.002;

            // --- 琴弦物理属性 ---
            this.bend = 0;       // 当前弯曲度 (位移)
            this.bendVel = 0;    // 弯曲速度
            this.tension = 0.04; // 张力降低，回弹更柔和
            this.damping = 0.9;  // 阻尼增加 (值越小停得越快)，震动时间变短
            this.maxBend = 200;  // 最大弯曲限制

            const isPurple = Math.random() > 0.3;
            this.colorBase = isPurple ? '188, 19, 254' : '0, 243, 255';
            
            this.opacity = Math.random() * 0.5;
            this.opacitySpeed = (Math.random() * 0.01 + 0.005) * (Math.random() > 0.5 ? 1 : -1);
            this.lineWidth = Math.random() * 2 + 1;
        }

        update() {
            // 1. 基础漂浮
            this.x += this.vx;
            this.y += this.vy;
            this.angle += this.vr;
            
            // 呼吸效果
            this.opacity += this.opacitySpeed;
            if (this.opacity > 0.6 || this.opacity < 0.1) {
                this.opacitySpeed = -this.opacitySpeed;
            }

            // 边界检查
            const margin = 500;
            if (this.x < -margin || this.x > width + margin || 
                this.y < -margin || this.y > height + margin) {
                this.reset();
            }

            // 2. 琴弦交互逻辑 (重点)
            if (mouse.x !== null && mouse.y !== null) {
                // 计算当前线条的两个端点 (基于当前位置和角度)
                const halfLen = this.length / 2;
                const cos = Math.cos(this.angle);
                const sin = Math.sin(this.angle);
                
                // 世界坐标系下的端点
                const p1x = this.x - halfLen * cos;
                const p1y = this.y - halfLen * sin;
                const p2x = this.x + halfLen * cos;
                const p2y = this.y + halfLen * sin;

                // 计算鼠标到线段的距离 (点到线段距离公式)
                // 向量 P1->P2
                const lx = p2x - p1x;
                const ly = p2y - p1y;
                // 向量 P1->Mouse
                const mx = mouse.x - p1x;
                const my = mouse.y - p1y;
                
                // 投影比 t = (M · L) / |L|^2
                const t = Math.max(0, Math.min(1, (mx * lx + my * ly) / (lx * lx + ly * ly)));
                
                // 线段上离鼠标最近的点 P_closest
                const cx = p1x + t * lx;
                const cy = p1y + t * ly;
                
                // 鼠标到最近点的距离
                const dist = Math.sqrt((mouse.x - cx) ** 2 + (mouse.y - cy) ** 2);
                
                // 如果鼠标触碰到琴弦 (交互半径)
                const stringRadius = 60; // 敏感度：在这个距离内才触发拨动
                
                if (dist < stringRadius) {
                    // 计算鼠标相对于直线的侧向速度或位置来决定拨动方向
                    // 这里简化：鼠标在哪一侧，就往哪一侧“压”一点，或者直接给速度
                    
                    // 为了让效果明显，我们计算一个强力拨动力
                    // 力的方向 = 鼠标到线条最近点的向量 (推开)
                    // 或者更真实的：我们假设鼠标抓住了线并拉动它
                    
                    // 这里使用简单的“推开”力作为拨动冲量
                    // 叉乘判断方向：(P2-P1) x (Mouse-P1)
                    const crossProduct = lx * my - ly * mx; 
                    // crossProduct > 0 在左侧，< 0 在右侧 (相对于向量方向)
                    
                    // 我们给 bendVel 施加一个与鼠标距离成反比的力
                    // 力度要大才能看出来震动
                    // 如果鼠标在线条的某一侧，就往那一侧推 bend
                    
                    // 一个更简单的拨动判定：上一帧不在半径内，这一帧在 -> 触发一次猛烈的 VELOCITY
                    // 但为了持续交互，我们用连续力：
                    
                    const force = (stringRadius - dist) / stringRadius; // 0~1 的力
                    // 这里的 100 是最大拨动速度
                    // 使用 sign(crossProduct) 来决定往哪边弯曲
                    const direction = crossProduct > 0 ? 1 : -1;
                    
                    // 施加力到速度上 (像弹簧被拉伸)
                    // 降低拨动力度，从 2 降到 0.8
                    this.bendVel += force * direction * 0.8; 
                }
            }

            // 3. 弹簧物理模拟 (Hooke's Law)
            // 恢复力 = -k * x
            const springForce = -this.tension * this.bend;
            this.bendVel += springForce;
            this.bendVel *= this.damping; // 阻力
            this.bend += this.bendVel;
            
            // 限制最大弯曲，防止炸裂
            if (this.bend > this.maxBend) this.bend = this.maxBend;
            if (this.bend < -this.maxBend) this.bend = -this.maxBend;
        }

        draw() {
            ctx.save();
            ctx.translate(this.x, this.y);
            ctx.rotate(this.angle);
            
            // 使用 Quadratic Curve 绘制弯曲
            // 起点 (-halfLen, 0)
            // 终点 (halfLen, 0)
            // 控制点 (0, this.bend) -> 注意：贝塞尔曲线控制点的 bend 需要大概是实际弯曲高度的 2 倍才能达到视觉高度
            
            const halfLen = this.length / 2;
            
            ctx.beginPath();
            ctx.moveTo(-halfLen, 0);
            
            // 如果弯曲很小，就画直线省资源 (可选优化，但为了平滑还是画曲线)
            if (Math.abs(this.bend) < 0.1) {
                ctx.lineTo(halfLen, 0);
            } else {
                ctx.quadraticCurveTo(0, this.bend, halfLen, 0);
            }
            
            // 只有当弯曲剧烈时，增加亮度 (高亮反馈)
            const bendIntensity = Math.min(1, Math.abs(this.bend) / 50); // 0~1
            const currentOpacity = Math.min(1, this.opacity + bendIntensity * 0.8);
            
            // 光晕
            ctx.shadowBlur = 10 + bendIntensity * 20; // 越弯光晕越大
            ctx.shadowColor = `rgba(${this.colorBase}, 1)`;
            ctx.strokeStyle = `rgba(${this.colorBase}, ${currentOpacity})`;
            ctx.lineWidth = this.lineWidth + bendIntensity * 2; // 越弯线越粗
            
            ctx.stroke();
            
            // 如果震动很厉害，加一道白色高亮核心 (像能量过载)
            if (bendIntensity > 0.3) {
                ctx.strokeStyle = `rgba(255, 255, 255, ${bendIntensity})`;
                ctx.lineWidth = 1;
                ctx.shadowBlur = 0;
                ctx.stroke();
            }

            ctx.restore();
        }
    }

    // 初始化线条数量
    const lineCount = 15;
    for (let i = 0; i < lineCount; i++) {
        lines.push(new NeonLine());
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        // 叠加模式，让交错的地方更亮，更有光的感觉
        ctx.globalCompositeOperation = 'lighter';

        lines.forEach(line => {
            line.update();
            line.draw();
        });
        
        ctx.globalCompositeOperation = 'source-over';
        requestAnimationFrame(animate);
    }

    animate();
})();
