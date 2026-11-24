const controls = document.querySelector('[data-atlas-controls]');
if (controls) {
  const search = document.querySelector('#atlas-search');
  const category = document.querySelector('#atlas-category');
  const region = document.querySelector('#atlas-region');
  const status = document.querySelector('[data-atlas-status]');
  const empty = document.querySelector('[data-atlas-empty]');
  const rows = [...document.querySelectorAll('.atlas-row')];
  const update = () => {
    const query = search.value.trim().toLocaleLowerCase();
    let count = 0;
    for (const row of rows) {
      const match = (!query || row.dataset.search.includes(query)) && (!category.value || row.dataset.category === category.value) && (!region.value || row.dataset.regions.split('|').includes(region.value));
      row.hidden = !match;
      if (match) count++;
    }
    status.textContent = `${count} ${count === 1 ? 'tradition' : 'traditions'}`;
    empty.hidden = count !== 0;
  };
  search.addEventListener('input', update);
  category.addEventListener('change', update);
  region.addEventListener('change', update);
  document.querySelector('#atlas-reset').addEventListener('click', () => { search.value = ''; category.value = ''; region.value = ''; update(); search.focus(); });
  controls.hidden = false;
  update();
}
