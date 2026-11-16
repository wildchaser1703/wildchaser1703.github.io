import { BACKGROUND, FRAME, createSculpture, createLineData, normalizeVariant } from './sculpture-shape.js';
import { createBloom } from './sculpture-bloom.js';

const instances = new WeakMap();
const vertexShader = `
  attribute vec3 threadColor;
  attribute float threadOpacity;
  uniform float time;
  uniform float pixelRatio;
  uniform vec3 pointer;
  varying vec3 ink;
  varying float alpha;
  void main() {
    vec3 p = position;
    // A small travelling breath, not a rotating object or a particle system.
    p.x += 0.018 * sin(time * 0.31 + position.y * 1.4) * sin(time * 0.19);
    p.y += 0.022 * sin(time * 0.23 + position.x * 1.1) * sin(time * 0.17);
    p.z += 0.025 * sin(time * 0.27 + position.x + position.y) * sin(time * 0.21);
    vec2 delta = p.xy - pointer.xy;
    float influence = exp(-dot(delta, delta) / 0.45) * pointer.z;
    p.xy += delta * influence * 0.12;
    p.y += influence * 0.065;
    p.z += influence * 0.16;
    ink = threadColor;
    // Native GL lines are one physical pixel: retain ink density on Retina.
    alpha = min(threadOpacity * pixelRatio, 1.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;
const fragmentShader = `
  varying vec3 ink;
  varying float alpha;
  void main() {
    gl_FragColor = vec4(ink, alpha);
    #include <colorspace_fragment>
  }
`;

/** Initialize one markup-contract figure. Returns an idempotent cleanup hook. */
export function initSculpture(figure) {
  if (instances.has(figure)) return instances.get(figure);
  const stage = figure.querySelector('.sculpture-stage');
  const canvas = figure.querySelector('canvas.sculpture-canvas');
  const poster = figure.querySelector('img.sculpture-poster');
  if (!stage || !canvas || !poster) return () => {};
  const toggle = figure.querySelector('[data-sculpture-toggle]');
  const next = figure.querySelector('[data-sculpture-next]');
  const status = figure.querySelector('[data-sculpture-state]');
  const counter = figure.querySelector('[data-sculpture-variant]');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const small = window.matchMedia('(max-width: 640px), (pointer: coarse)');
  const dark = window.matchMedia('(prefers-color-scheme: dark)');
  const listeners = new AbortController();
  const newSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
  let variant = normalizeVariant(newSeed());
  let posterURL;
  let quality = small.matches ? 'mobile' : 'desktop';
  let userPaused = false;
  let destroyed = false;
  let failed = false;
  let pageHidden = false;
  let generation = 0;
  let renderer, scene, camera, geometry, material, lines, three, bloom;
  let frame = 0;
  let elapsed = 0;
  let previousTime = 0;
  let width = 0;
  let height = 0;
  const target = { x: 0, y: 0, strength: 0 };
  const initialRect = stage.getBoundingClientRect();
  let inView = initialRect.bottom > 0 && initialRect.top < window.innerHeight
    && initialRect.right > 0 && initialRect.left < window.innerWidth;

  function listen(element, name, handler, options = {}) {
    element?.addEventListener(name, handler, { ...options, signal: listeners.signal });
  }

  function updateEdition() {
    figure.dataset.variant = String(variant);
    const paths = createSculpture(variant, 'preview').map(({points,color,opacity}) => {
      let d = '';
      for (let i=0;i<points.length;i+=3) d += `${i?'L':'M'}${(450+points[i]*140).toFixed(1)} ${(380-points[i+1]*140).toFixed(1)}`;
      return `<path d="${d}" stroke="rgb(${color.join(',')})" stroke-opacity="${opacity}"/>`;
    }).join('');
    if (posterURL) URL.revokeObjectURL(posterURL);
    posterURL = URL.createObjectURL(new Blob([`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 760"><g fill="none" stroke-width="1.3">${paths}</g></svg>`],{type:'image/svg+xml'}));
    poster.src = posterURL;
    if (counter) counter.textContent = `Seed ${variant.toString(16).padStart(8,'0')}`;
  }

  function showRenderer(mode) {
    figure.dataset.renderer = mode;
    const live = mode === 'webgl';
    if (!live) figure.dataset.bloom = 'off';
    canvas.hidden = !live;
    canvas.style.visibility = live ? 'visible' : 'hidden';
    canvas.setAttribute('aria-hidden', 'true');
    // Preserve the image's accessible description while WebGL is visible.
    // Visibility hides it from AT too, so transfer its description to the stage.
    poster.hidden = live;
    poster.style.visibility = live ? 'hidden' : 'visible';
    if (live) {
      stage.setAttribute('role', 'img');
      stage.setAttribute('aria-label', poster.alt || 'An asymmetric loop woven from fine gold, orange and red threads.');
    } else {
      stage.removeAttribute('role');
      stage.removeAttribute('aria-label');
    }
    if (toggle) toggle.hidden = !live;
    if (next) next.hidden = false;
    const controls = figure.querySelector('.sculpture-controls, [data-sculpture-controls]');
    if (controls) controls.hidden = false;
  }

  function canAnimate() {
    return Boolean(renderer && !destroyed && !reduced.matches && !userPaused
      && !document.hidden && !pageHidden && inView && width > 0 && height > 0);
  }

  function stop() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    previousTime = 0;
  }

  function syncMotion() {
    const running = canAnimate();
    figure.dataset.paused = String(!running);
    if (toggle) {
      toggle.textContent = userPaused ? 'Resume' : 'Pause';
      toggle.setAttribute('aria-label', userPaused ? 'Resume sculpture motion' : 'Pause sculpture motion');
      toggle.setAttribute('aria-pressed', String(userPaused));
    }
    if (!running) stop();
    else if (!frame) frame = requestAnimationFrame(tick);
  }

  function release() {
    stop();
    canvas.removeEventListener('webglcontextlost', contextLost);
    bloom?.dispose();
    bloom = undefined;
    geometry?.dispose();
    material?.dispose();
    renderer?.dispose();
    geometry = material = renderer = scene = camera = lines = undefined;
  }

  function fallback(permanent = false) {
    generation++;
    failed ||= permanent;
    release();
    showRenderer('static');
    syncMotion();
  }

  function contextLost(event) {
    event.preventDefault();
    // Keep the matching poster and variation control; do not retry a lost GPU.
    fallback(true);
  }

  function draw() {
    if (!renderer || !width || !height) return;
    try {
      if (bloom) {
        try {
          bloom.render(scene, camera);
          figure.dataset.bloom = 'on';
          return;
        } catch {
          // A post-processing failure should not take the artwork away.
          bloom.dispose();
          bloom = undefined;
          renderer.setRenderTarget(null);
        }
      }
      figure.dataset.bloom = 'off';
      renderer.render(scene, camera);
    }
    catch { fallback(true); }
  }

  function tick(timestamp) {
    frame = 0;
    if (!canAnimate()) { syncMotion(); return; }
    if (previousTime && timestamp - previousTime < 1000 / 30) {
      frame = requestAnimationFrame(tick);
      return;
    }
    const dt = previousTime ? Math.min((timestamp - previousTime) / 1000, 0.05) : 0;
    previousTime = timestamp;
    elapsed += dt;
    material.uniforms.time.value = elapsed;
    const pointer = material.uniforms.pointer.value;
    const ease = 1 - Math.exp(-dt * 6);
    pointer.x += (target.x - pointer.x) * ease;
    pointer.y += (target.y - pointer.y) * ease;
    pointer.z += (target.strength - pointer.z) * ease;
    // A small viewpoint change exposes the actual depth between woven layers.
    // Settle back to the poster composition when the pointer leaves the work.
    lines.rotation.x += (-pointer.y * pointer.z * 0.04 - lines.rotation.x) * ease;
    lines.rotation.y += (pointer.x * pointer.z * 0.045 - lines.rotation.y) * ease;
    draw();
    if (canAnimate()) frame = requestAnimationFrame(tick);
  }

  function replaceGeometry() {
    const data = createLineData(createSculpture(variant, quality));
    // Shader output uses Three's sRGB conversion, so decode vertex ink first.
    const decode = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    for (let i = 0; i < data.colors.length; i++) data.colors[i] = decode(data.colors[i]);
    const replacement = new three.BufferGeometry();
    replacement.setAttribute('position', new three.BufferAttribute(data.positions, 3));
    replacement.setAttribute('threadColor', new three.BufferAttribute(data.colors, 3));
    replacement.setAttribute('threadOpacity', new three.BufferAttribute(data.opacities, 1));
    if (lines) lines.geometry = replacement;
    geometry?.dispose();
    geometry = replacement;
    if (!lines) {
      lines = new three.LineSegments(geometry, material);
      lines.frustumCulled = false;
      scene.add(lines);
    }
  }

  function resize() {
    if (!renderer) return;
    try {
      const rect = stage.getBoundingClientRect();
      width = Math.round(rect.width);
      height = Math.round(rect.height);
      if (!width || !height) { syncMotion(); return; }
      const newQuality = small.matches ? 'mobile' : 'desktop';
      if (newQuality !== quality) { quality = newQuality; replaceGeometry(); }
      const pixelRatio = Math.min(window.devicePixelRatio || 1, quality === 'mobile' ? 1.5 : 2);
      renderer.setPixelRatio(pixelRatio);
      material.uniforms.pixelRatio.value = pixelRatio;
      renderer.setSize(width, height, false);
      bloom?.resize(width, height, pixelRatio, quality === 'mobile');
      // Contain the exact poster frame, even in a tall mobile stage.
      const aspect = width / height;
      const halfHeight = Math.max(FRAME.height / 2, FRAME.width / (2 * aspect));
      camera.left = -halfHeight * aspect;
      camera.right = halfHeight * aspect;
      camera.top = halfHeight;
      camera.bottom = -halfHeight;
      camera.updateProjectionMatrix();
      draw();
      syncMotion();
    } catch { fallback(true); }
  }

  async function start() {
    if (destroyed || failed || reduced.matches || renderer) return;
    const token = ++generation;
    try {
      // No Three request and no context creation at all for reduced motion.
      // Current Three supports WebGL2; use the real canvas, not a spare context.
      if (typeof canvas.getContext !== 'function' || !('WebGL2RenderingContext' in window)) return;
      const context = canvas.getContext('webgl2', {
        alpha: true, antialias: true, depth: false, stencil: false,
        powerPreference: 'low-power', preserveDrawingBuffer: false,
      });
      if (!context || context.isContextLost()) { fallback(true); return; }
      const module = await import('/assets/js/vendor/three.module.js');
      if (destroyed || token !== generation || reduced.matches) return;
      three = module;
      canvas.addEventListener('webglcontextlost', contextLost);
      renderer = new three.WebGLRenderer({ canvas, context, antialias: true, alpha: true });
      renderer.setClearColor(BACKGROUND, 0);
      renderer.outputColorSpace = three.SRGBColorSpace;
      // Shader compile failures may otherwise only log and leave a blank hero.
      renderer.debug.onShaderError = () => { throw new Error('Sculpture shader unavailable'); };
      scene = new three.Scene();
      camera = new three.OrthographicCamera(-3.2, 3.2, 2.7, -2.7, 0.1, 20);
      camera.position.z = 8;
      material = new three.ShaderMaterial({
        uniforms: { time: { value: elapsed }, pixelRatio: { value: 1 }, pointer: { value: new three.Vector3() } },
        vertexShader, fragmentShader, transparent: true, depthTest: false, depthWrite: false,
      });
      bloom = createBloom(three, renderer);
      bloom.setTheme(dark.matches);
      replaceGeometry();
      resize();
      if (!renderer) return;
      showRenderer('webgl');
      syncMotion();
    } catch {
      if (!destroyed && token === generation) fallback(true);
    }
  }

  showRenderer('static');
  updateEdition();
  syncMotion();
  if (toggle) toggle.type = 'button';
  if (next) next.type = 'button';
  listen(toggle, 'click', () => {
    if (!renderer) return;
    userPaused = !userPaused;
    syncMotion();
    if (status) status.textContent = userPaused ? 'Motion paused.' : 'Motion resumed.';
  });
  listen(next, 'click', () => {
    const seed = newSeed();
    variant = seed === variant ? normalizeVariant(seed + 1) : seed;
    updateEdition();
    target.strength = 0;
    if (renderer) {
      try {
        replaceGeometry();
        material.uniforms.pointer.value.set(0, 0, 0);
        draw();
      } catch { fallback(true); }
    }
    if (status) status.textContent = 'New woven field generated.';
  });
  listen(stage, 'pointermove', event => {
    // Passive mouse/hovering pen only: touch remains native page scrolling.
    if (!canAnimate() || event.pointerType === 'touch' || (event.pointerType === 'pen' && event.buttons)) return;
    const rect = stage.getBoundingClientRect();
    target.x = camera.left + (event.clientX - rect.left) / rect.width * (camera.right - camera.left);
    target.y = camera.top - (event.clientY - rect.top) / rect.height * (camera.top - camera.bottom);
    target.strength = 1;
  }, { passive: true });
  listen(stage, 'pointerleave', () => { target.strength = 0; }, { passive: true });
  listen(stage, 'pointercancel', () => { target.strength = 0; }, { passive: true });
  listen(document, 'visibilitychange', () => { target.strength = 0; syncMotion(); });
  listen(window, 'resize', resize, { passive: true });
  listen(dark, 'change', () => { bloom?.setTheme(dark.matches); draw(); });
  listen(window, 'pagehide', event => {
    pageHidden = true;
    if (event.persisted) syncMotion();
    else destroy();
  });
  listen(window, 'pageshow', () => { pageHidden = false; syncMotion(); });

  const motionChanged = () => {
    if (reduced.matches) fallback();
    else void start();
  };
  // MediaQueryList.addListener is needed for older Safari.
  if (reduced.addEventListener) reduced.addEventListener('change', motionChanged);
  else reduced.addListener(motionChanged);

  const intersection = typeof IntersectionObserver === 'function'
    ? new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      if (!inView) target.strength = 0;
      syncMotion();
    }, { threshold: 0 }) : null;
  intersection?.observe(stage);
  // A scroll fallback keeps the pause contract when IO is unavailable.
  if (!intersection) listen(window, 'scroll', () => {
    const rect = stage.getBoundingClientRect();
    inView = rect.bottom > 0 && rect.top < window.innerHeight && rect.right > 0 && rect.left < window.innerWidth;
    syncMotion();
  }, { passive: true });
  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(resize) : null;
  observer?.observe(stage);

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    generation++;
    listeners.abort();
    intersection?.disconnect();
    observer?.disconnect();
    if (reduced.removeEventListener) reduced.removeEventListener('change', motionChanged);
    else reduced.removeListener(motionChanged);
    const activeRenderer = renderer;
    release();
    activeRenderer?.forceContextLoss();
    showRenderer('static');
    if (next) next.hidden = true;
    syncMotion();
    instances.delete(figure);
    if (posterURL) URL.revokeObjectURL(posterURL);
  }

  instances.set(figure, destroy);
  // Keep the poster and page responsive while the browser paints its content.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (typeof requestIdleCallback === 'function') requestIdleCallback(() => void start(), { timeout: 1500 });
    else setTimeout(() => void start(), 0);
  }));
  return destroy;
}

export function initSculptures(root = document) {
  return [...root.querySelectorAll('figure[data-sculpture]')].map(initSculpture);
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => initSculptures(), { once: true });
  else initSculptures();
}
