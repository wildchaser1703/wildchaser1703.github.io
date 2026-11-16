// Each visit starts with a fresh selection, then rotates while visible. Image links
// remain usable with JavaScript disabled, reduced motion, or a failed download.
const collectionData = document.querySelector('#home-collections');
if (collectionData) {
  const collections = JSON.parse(collectionData.textContent);
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  for (const section of document.querySelectorAll('[data-collection]')) {
    const kind = section.dataset.collection;
    const entries = collections[kind];
    const figures = [...section.querySelectorAll('figure')];
    const controls = section.querySelector('.collection-controls');
    const next = controls.querySelector('[data-collection-next]');
    const pause = controls.querySelector('[data-collection-pause]');
    const status = controls.querySelector('[role="status"]');
    const storageKey = 'home-selection-' + kind;
    let previousVisit = [];
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey));
      if (Array.isArray(saved)) previousVisit = saved.filter(value => typeof value === 'string').slice(0, figures.length);
    } catch { /* Storage may be disabled; fresh random selections still work. */ }
    let paused = motion.matches, busy = false, visible = false, timer;
    let bag = [];
    const source = item => kind === 'art' ? `/assets/art/${encodeURIComponent(item.file)}` : `/assets/img/food/${encodeURIComponent(item.file)}`;
    const shuffle = values => {
      for (let i = values.length-1; i>0; i--) {
        const j = Math.floor(Math.random()*(i+1));
        [values[i],values[j]] = [values[j],values[i]];
      }
      return values;
    };
    function resetTimer() {
      clearTimeout(timer);
      if (paused || motion.matches || document.hidden || !visible) return;
      timer = setTimeout(async () => {
        if (!section.querySelector('a[data-gallery]:focus-visible') && !document.querySelector('.image-viewer[open]')) await advance(false);
        resetTimer();
      }, 6000);
    }
    function syncPause() {
      pause.textContent = motion.matches ? 'Rotation off' : paused ? 'Resume rotation' : 'Pause rotation';
      pause.setAttribute('aria-pressed', String(paused || motion.matches));
      pause.disabled = motion.matches;
      pause.title = motion.matches ? 'Automatic rotation is off for reduced motion. Use More to browse.' : '';
      resetTimer();
    }
    async function advance(manual = true, initial = false) {
      if (busy || document.querySelector('.image-viewer[open]')) return;
      busy = true;
      next.disabled = true;
      const current = new Set(figures.map(figure => figure.querySelector('a').getAttribute('href')));
      if (initial) previousVisit.forEach(url => current.add(url));
      // Keep enough candidates if the collection ever becomes smaller.
      if (entries.filter(item => !current.has(source(item))).length < figures.length) {
        current.clear();
        if (initial) previousVisit.forEach(url => current.add(url));
      }
      if (entries.filter(item => !current.has(source(item))).length < figures.length) current.clear();
      const selected = [];
      while (selected.length < figures.length) {
        if (!bag.length) bag = shuffle(entries.filter(item => !current.has(source(item)) && !selected.includes(item)));
        const item = bag.pop();
        if (item && !current.has(source(item)) && !selected.includes(item)) selected.push(item);
      }
      try {
        const loaded = await Promise.all(selected.map(async item => {
          const image = new Image();
          image.src = kind === 'food' && item.preview ? `/assets/img/food/previews/${encodeURIComponent(item.preview)}` : source(item);
          await image.decode();
          return image.src;
        }));
        // An image viewer may have opened while the next selection downloaded.
        if (document.querySelector('.image-viewer[open]') || (!manual && !initial && (paused || motion.matches || document.hidden || !visible || section.querySelector('a[data-gallery]:focus-visible')))) return;
        selected.forEach((item,i) => {
          const figure = figures[i], link = figure.querySelector('a'), image = link.querySelector('img');
          link.href = source(item);
          link.setAttribute('aria-label', kind === 'art' ? `View ${item.title}` : `View full-size photo: ${item.alt}`);
          image.src = loaded[i];
          image.alt = kind === 'art' ? item.title : item.alt;
          image.width = item.width;
          image.height = item.height;
          if (kind === 'art') {
            figure.querySelector('figcaption strong').textContent = item.title;
            figure.querySelector('figcaption span').textContent = item.caption;
          } else figure.querySelector('figcaption').textContent = item.caption;
        });
        try { localStorage.setItem(storageKey, JSON.stringify(selected.map(source))); } catch {}
        if (manual) status.textContent = 'Three more ' + (kind === 'art' ? 'paintings.' : 'photographs.');
      } catch {
        if (manual) status.textContent = 'Could not load the next selection. Please try again.';
      } finally {
        busy = false;
        next.disabled = false;
        if (initial) section.dataset.collectionReady = 'true';
      }
    }
    controls.hidden = false;
    next.addEventListener('click', async () => { await advance(); resetTimer(); });
    pause.addEventListener('click', () => { paused = !paused; syncPause(); });
    motion.addEventListener('change', () => { paused = motion.matches; syncPause(); });
    document.addEventListener('visibilitychange', resetTimer);
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; resetTimer(); }, {threshold:.15}).observe(section);
    } else visible = true;
    window.addEventListener('pagehide', () => clearTimeout(timer));
    window.addEventListener('pageshow', event => {
      if (event.persisted) void advance(false, true);
      resetTimer();
    });
    syncPause();
    // Initial selection also changes for visitors who prefer reduced motion.
    void advance(false, true);
  }
}

const searchInput = document.querySelector('#search-input');
const menuButton = document.querySelector('.menu-toggle');
const mainNavigation = document.querySelector('#main-navigation');
if (menuButton && mainNavigation) {
  const smallScreen = matchMedia('(max-width: 760px)');
  function closeMenu(restoreFocus = false) {
    mainNavigation.hidden = smallScreen.matches;
    menuButton.setAttribute('aria-expanded', 'false');
    if (restoreFocus) menuButton.focus();
  }
  function syncMenu() {
    menuButton.hidden = !smallScreen.matches;
    closeMenu();
  }
  menuButton.addEventListener('click', () => {
    const opening = menuButton.getAttribute('aria-expanded') !== 'true';
    menuButton.setAttribute('aria-expanded', String(opening));
    mainNavigation.hidden = !opening;
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && smallScreen.matches && !mainNavigation.hidden) closeMenu(true);
  });
  document.addEventListener('click', event => {
    if (smallScreen.matches && !event.target.closest('.site-header')) closeMenu();
  });
  mainNavigation.addEventListener('click', event => {
    if (event.target.closest('a')) closeMenu();
  });
  smallScreen.addEventListener('change', syncMenu);
  syncMenu();
}

const directoryFilters = document.querySelector('.directory-filters');
if (directoryFilters) {
  const rows = [...document.querySelectorAll('.directory-row')];
  const preview = document.querySelector('.directory-preview');
  const previewImage = preview.querySelector('img');
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let previewAnimation;
  function showPreview(row) {
    if (!row || previewImage.getAttribute('src') === row.dataset.preview) return;
    previewAnimation?.cancel();
    previewImage.src = row.dataset.preview;
    preview.querySelector('[data-preview-title]').textContent = row.dataset.previewTitle;
    preview.querySelector('[data-preview-caption]').textContent = row.dataset.previewCaption;
    if (!reduceMotion.matches && preview.animate) {
      previewAnimation = preview.animate([{opacity:.4,transform:'translateY(4px)'},{opacity:1,transform:'translateY(0)'}],{duration:180,easing:'ease-out'});
    }
  }
  for (const row of rows) {
    row.addEventListener('pointerenter', () => showPreview(row));
    row.addEventListener('focus', () => showPreview(row));
  }
  directoryFilters.hidden = false;
  directoryFilters.addEventListener('click', event => {
    const button = event.target.closest('[data-filter]');
    if (!button) return;
    for (const item of directoryFilters.querySelectorAll('button')) item.setAttribute('aria-pressed', String(item === button));
    for (const row of rows) row.hidden = button.dataset.filter !== 'All' && row.dataset.kind !== button.dataset.filter;
    const visible = rows.filter(row => !row.hidden);
    directoryFilters.querySelector('[role="status"]').textContent = `${visible.length} ${visible.length === 1 ? 'project' : 'projects'}`;
    showPreview(visible[0]);
  });
}

if (typeof HTMLDialogElement !== 'undefined') {
  const viewer = document.createElement('dialog');
  viewer.className = 'image-viewer';
  viewer.setAttribute('aria-label', 'Image viewer');
  viewer.innerHTML = '<div class="viewer-bar"><span class="viewer-count" aria-live="polite"></span><button class="viewer-close" type="button" aria-label="Close image viewer">✕</button></div><div class="viewer-stage"><img alt=""></div><div class="viewer-bottom"><p class="viewer-caption" aria-live="polite"></p><div class="viewer-controls"><button class="viewer-prev" type="button">← Previous</button><button class="viewer-next" type="button">Next →</button></div></div>';
  document.body.append(viewer);
  const image = viewer.querySelector('img');
  const caption = viewer.querySelector('.viewer-caption');
  const count = viewer.querySelector('.viewer-count');
  const previous = viewer.querySelector('.viewer-prev');
  const next = viewer.querySelector('.viewer-next');
  let items = [], index = 0, opener, priorOverflow;
  function show() {
    const link = items[index];
    const source = link.querySelector('img');
    image.src = link.href;
    image.alt = source.alt;
    const details = link.closest('figure')?.querySelector('figcaption');
    caption.textContent = details ? [...details.children].map(part => part.textContent.trim()).filter(Boolean).join(' — ') || details.textContent.trim() : source.alt;
    count.textContent = `${index + 1} / ${items.length}`;
    previous.disabled = index === 0;
    next.disabled = index === items.length - 1;
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('a[data-gallery]');
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    opener = link;
    items = [...document.querySelectorAll(`a[data-gallery="${link.dataset.gallery}"]`)];
    index = items.indexOf(link);
    show();
    priorOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    viewer.showModal();
    viewer.querySelector('.viewer-close').focus();
  });
  previous.addEventListener('click', () => { if (index > 0) { index--; show(); } });
  next.addEventListener('click', () => { if (index < items.length - 1) { index++; show(); } });
  viewer.querySelector('.viewer-close').addEventListener('click', () => viewer.close());
  viewer.addEventListener('close', () => { image.removeAttribute('src'); document.body.style.overflow = priorOverflow; opener?.focus(); });
  viewer.addEventListener('click', event => { if (event.target === viewer) viewer.close(); });
  viewer.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft' && index > 0) { event.preventDefault(); index--; show(); }
    if (event.key === 'ArrowRight' && index < items.length - 1) { event.preventDefault(); index++; show(); }
  });
}
if (searchInput) {
  const results = document.querySelector('#search-results');
  const status = document.querySelector('#search-status');
  fetch('/search-index.json').then(response => {
    if (!response.ok) throw new Error('Search index unavailable');
    return response.json();
  }).then(posts => {
    function render() {
      const query = searchInput.value.trim().toLocaleLowerCase();
      results.replaceChildren();
      if (!query) {
        status.textContent = 'Start typing to find a note.';
        return;
      }
      const terms = query.split(/\s+/);
      const matches = posts.filter(post => terms.every(term => `${post.title} ${post.text} ${post.categories.join(' ')} ${post.tags.join(' ')}`.toLocaleLowerCase().includes(term)));
      status.textContent = `${matches.length} ${matches.length === 1 ? 'note' : 'notes'} found.`;
      for (const post of matches) {
        const link = document.createElement('a');
        const time = document.createElement('time');
        const title = document.createElement('span');
        const arrow = document.createElement('span');
        link.className = 'post-row';
        link.href = post.url;
        time.textContent = post.date;
        title.textContent = post.title;
        arrow.textContent = '↗';
        arrow.setAttribute('aria-hidden', 'true');
        link.append(time, title, arrow);
        results.append(link);
      }
    }
    searchInput.addEventListener('input', render);
    render();
  }).catch(() => { status.textContent = 'Search is unavailable. Please browse the archive instead.'; });
}

document.addEventListener('click', async event => {
  const button = event.target.closest('.code-header button');
  if (!button) return;
  const code = button.closest('.highlighter-rouge')?.querySelector('.rouge-code pre');
  if (!code) return;
  try {
    await navigator.clipboard.writeText(code.textContent);
    button.setAttribute('aria-label', 'Copied');
    button.classList.add('copied');
    setTimeout(() => { button.setAttribute('aria-label', 'copy'); button.classList.remove('copied'); }, 2000);
  } catch {
    button.setAttribute('aria-label', 'Copy unavailable; select code manually');
  }
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(registrations => Promise.all(registrations.filter(registration => new URL(registration.active?.scriptURL || registration.waiting?.scriptURL || registration.installing?.scriptURL || location.origin).pathname === '/sw.min.js').map(registration => registration.update()))).catch(() => {});
}
