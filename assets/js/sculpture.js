import { BACKGROUND, FRAME, createMeshData, createPosterSvg, normalizeVariant, nextVariation } from './sculpture-shape.js';
import { createBloom } from './sculpture-bloom.js';

const instances = new WeakMap();
const vertexShader = `
  attribute vec3 folioColor;
  uniform float time;
  uniform vec3 pointer;
  varying vec3 ink;
  varying vec3 surfaceNormal;
  varying vec3 location;
  void main() {
    vec3 p = position;
    float breath = sin(time * .42 + p.y * 1.3);
    p.x += .035 * breath * (1.0 - abs(p.y) / 2.5);
    p.z += .045 * sin(time * .35 + p.y * 1.5);
    p.z += pointer.z * .035 * sin(p.y + pointer.x);
    ink = folioColor;
    surfaceNormal = normalize(normalMatrix * normal);
    location = p;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;
const fragmentShader = `
  uniform float time;
  varying vec3 ink;
  varying vec3 surfaceNormal;
  varying vec3 location;
  void main() {
    vec3 n = normalize(surfaceNormal) * (gl_FrontFacing ? 1.0 : -1.0);
    vec3 key = normalize(vec3(2.8 + .45 * sin(time * .23), 1.6, 2.0));
    float diffuse = max(dot(n, key), 0.0);
    float rim = pow(1.0 - abs(n.z), 3.0);
    float specular = pow(max(dot(n, normalize(key + vec3(0,0,1))),0.0), 42.0);
    // Thin flowing ink traces, interrupted by a quiet travelling signal.
    float trace = pow(.5 + .5 * sin(location.y * 66.0 + location.x * 12.0), 28.0);
    float pulse = pow(.5 + .5 * sin(location.y * 2.0 - time * .65), 16.0);
    vec3 color = ink * (.24 + .95 * diffuse) + vec3(1.0,.60,.24) * (specular*.42 + rim*.22);
    color *= 1.0 - trace * .12;
    color += vec3(.3,.1,.02) * trace * pulse;
    gl_FragColor = vec4(color, 1.0);
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
  const recentVariants = [variant];
  let posterURL;
  let mobile = small.matches;
  let userPaused = false;
  let destroyed = false;
  let failed = false;
  let pageHidden = false;
  let generation = 0;
  let renderer, scene, camera, geometry, material, mesh, three, bloom;
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
    if (posterURL) URL.revokeObjectURL(posterURL);
    posterURL = URL.createObjectURL(new Blob([createPosterSvg(variant)], {type:'image/svg+xml'}));
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
      stage.setAttribute('aria-label', poster.alt || 'Five folded folios in ochre and vermilion, joined along a flowing spine.');
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
    geometry = material = renderer = scene = camera = mesh = undefined;
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
    // A small viewpoint change exposes the actual depth between the folded folios.
    // Settle back to the poster composition when the pointer leaves the work.
    mesh.rotation.x += (-pointer.y * pointer.z * 0.04 - mesh.rotation.x) * ease;
    mesh.rotation.y += (pointer.x * pointer.z * 0.045 - mesh.rotation.y) * ease;
    draw();
    if (canAnimate()) frame = requestAnimationFrame(tick);
  }

  function replaceGeometry() {
    const data = createMeshData(variant);
    // Shader output uses Three's sRGB conversion, so decode vertex ink first.
    const decode = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    for (let i = 0; i < data.colors.length; i++) data.colors[i] = decode(data.colors[i]);
    const replacement = new three.BufferGeometry();
    replacement.setAttribute('position', new three.BufferAttribute(data.positions, 3));
    replacement.setAttribute('folioColor', new three.BufferAttribute(data.colors, 3));
    replacement.setAttribute('normal', new three.BufferAttribute(data.normals, 3));
    replacement.setIndex(data.indices);
    if (mesh) mesh.geometry = replacement;
    geometry?.dispose();
    geometry = replacement;
    if (!mesh) {
      mesh = new three.Mesh(geometry, material);
      mesh.frustumCulled = false;
      scene.add(mesh);
    }
  }

  function resize() {
    if (!renderer) return;
    try {
      const rect = stage.getBoundingClientRect();
      width = Math.round(rect.width);
      height = Math.round(rect.height);
      if (!width || !height) { syncMotion(); return; }
      mobile = small.matches;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2);
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(width, height, false);
      bloom?.resize(width, height, pixelRatio, mobile);
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
        alpha: true, antialias: true, depth: true, stencil: false,
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
        uniforms: { time: { value: elapsed }, pointer: { value: new three.Vector3() } },
        vertexShader, fragmentShader, side: three.DoubleSide, depthTest: true, depthWrite: true,
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
    variant = nextVariation(variant, newSeed, recentVariants);
    recentVariants.push(variant);
    if (recentVariants.length > 32) recentVariants.shift();
    updateEdition();
    target.strength = 0;
    if (renderer) {
      try {
        replaceGeometry();
        material.uniforms.pointer.value.set(0, 0, 0);
        draw();
      } catch { fallback(true); }
    }
    if (status) status.textContent = 'New folio edition generated.';
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
