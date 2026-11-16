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
