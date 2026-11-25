(() => {
  const controls = document.querySelector('[data-atlas-controls]');
  if (controls) {
    const search = document.querySelector('#atlas-search');
    const category = document.querySelector('#atlas-category');
    const region = document.querySelector('#atlas-region');
    const status = document.querySelector('[data-atlas-status]');
    const empty = document.querySelector('[data-atlas-empty]');
    const rows = [...document.querySelectorAll('.atlas-row')];
    const params = new URLSearchParams(location.search);
    search.value = params.get('q') || '';
    category.value = params.get('medium') || '';
    region.value = params.get('region') || '';
    const update = (writeUrl = true) => {
      const terms = search.value.trim().normalize('NFKC').toLocaleLowerCase().split(/\s+/).filter(Boolean);
      let count = 0;
      for (const row of rows) {
        const match = terms.every(term => row.dataset.search.includes(term)) && (!category.value || row.dataset.category === category.value) && (!region.value || row.dataset.regions.split('|').includes(region.value));
        row.hidden = !match;
        if (match) count++;
      }
      status.textContent = `${count} ${count === 1 ? 'tradition' : 'traditions'}`;
      empty.hidden = count !== 0;
      if (writeUrl) {
        const url = new URL(location.href);
        for (const [key, value] of [['q', search.value.trim()], ['medium', category.value], ['region', region.value]]) value ? url.searchParams.set(key, value) : url.searchParams.delete(key);
        history.replaceState(null, '', url);
      }
    };
    search.addEventListener('input', () => update());
    category.addEventListener('change', () => update());
    region.addEventListener('change', () => update());
    document.querySelector('#atlas-reset').addEventListener('click', () => { search.value = ''; category.value = ''; region.value = ''; update(); search.focus(); });
    controls.hidden = false;
    update(false);
  }

  const zoom = document.querySelector('#atlas-zoom');
  if (zoom && typeof zoom.showModal === 'function') {
    document.addEventListener('click', event => {
      const link = event.target.closest('[data-zoom-src]');
      if (!link) return;
      event.preventDefault();
      const image = zoom.querySelector('img');
      image.src = link.dataset.zoomSrc;
      image.alt = link.dataset.zoomAlt;
      zoom.showModal();
    });
    zoom.addEventListener('click', event => { if (event.target === zoom) zoom.close(); });
  }

  const buttons = [...document.querySelectorAll('[data-save-work]')];
  const savedPage = document.querySelector('[data-saved-page]');
  if (!buttons.length && !savedPage) return;
  const key = 'indian-art-atlas-saved-v1';
  let stored;
  try {
    const value = localStorage.getItem(key);
    const parsed = value ? JSON.parse(value) : [];
    stored = new Set(Array.isArray(parsed) ? parsed.filter(id => typeof id === 'string' && /^[a-z0-9-]+$/.test(id)).slice(0, 500) : []);
    localStorage.setItem(key, JSON.stringify([...stored]));
  } catch {
    if (savedPage) {
      const status = savedPage.querySelector('[data-saved-status]');
      savedPage.querySelector('[data-saved-controls]').hidden = false;
      status.textContent = 'Saving is unavailable in this browser. You can still browse these records.';
    }
    return;
  }
  const sharedIds = new URLSearchParams(location.search).get('works');
  const cards = savedPage ? [...savedPage.querySelectorAll('[data-saved-grid] [data-work-id]')] : [];
  const validIds = new Set(cards.map(card => card.dataset.workId));
  let shared = sharedIds !== null ? new Set(sharedIds.split(',').filter(id => validIds.has(id))) : null;
  const refresh = () => {
    for (const button of buttons) {
      const isSaved = stored.has(button.dataset.saveWork);
      button.hidden = false;
      button.setAttribute('aria-pressed', String(isSaved));
      button.textContent = isSaved ? 'Remove from my collection' : 'Save to my collection';
    }
    if (!savedPage) return;
    const selection = shared || new Set([...stored].filter(id => validIds.has(id)));
    for (const card of cards) card.hidden = !selection.has(card.dataset.workId);
    savedPage.querySelector('[data-saved-controls]').hidden = false;
    savedPage.querySelector('[data-saved-empty]').hidden = selection.size > 0;
    savedPage.querySelector('[data-saved-status]').textContent = `${selection.size} ${selection.size === 1 ? 'work' : 'works'} ${shared ? 'in this shared selection' : 'saved in this browser'}.`;
    const url = new URL(location.origin + location.pathname);
    url.searchParams.set('works', [...selection].sort().join(','));
    savedPage.querySelector('#share-link').value = url.href;
    savedPage.querySelector('[data-save-shared]').hidden = !shared || !shared.size;
  };
  for (const button of buttons) button.addEventListener('click', () => {
    const id = button.dataset.saveWork;
    if (stored.has(id)) stored.delete(id); else stored.add(id);
    try { localStorage.setItem(key, JSON.stringify([...stored])); } catch { return; }
    refresh();
  });
  if (savedPage) {
    savedPage.querySelector('[data-save-shared]').addEventListener('click', () => {
      for (const id of shared || []) stored.add(id);
      try { localStorage.setItem(key, JSON.stringify([...stored])); } catch { return; }
      shared = null;
      history.replaceState(null, '', location.pathname);
      refresh();
    });
    savedPage.querySelector('[data-copy-link]').addEventListener('click', async () => {
      const input = savedPage.querySelector('#share-link');
      const status = savedPage.querySelector('[data-share-status]');
      try { await navigator.clipboard.writeText(input.value); status.textContent = 'Link copied.'; }
      catch { input.focus(); input.select(); status.textContent = 'Select and copy the link above.'; }
    });
  }
  refresh();
})();
