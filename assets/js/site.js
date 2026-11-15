const searchInput = document.querySelector('#search-input');
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
  navigator.serviceWorker.getRegistrations().then(registrations => Promise.all(registrations.filter(registration => new URL(registration.active?.scriptURL || registration.waiting?.scriptURL || registration.installing?.scriptURL || location.origin).pathname === '/sw.min.js').map(registration => registration.unregister()))).catch(() => {});
}
