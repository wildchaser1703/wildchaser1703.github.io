import { readdir, readFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import * as cheerio from 'cheerio';

const root = path.resolve('dist');
const legacy = JSON.parse(await readFile('data/legacy-posts.json', 'utf8'));
const taxonomy = JSON.parse(await readFile('data/legacy-taxonomy.json', 'utf8'));
const failures = [];
let accessibleAnchors = 0, focusableBlocks = 0;
async function files(directory, extension = '.html') {
  const found = [];
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const location = path.join(directory, item.name);
    if (item.isDirectory()) found.push(...await files(location, extension));
    else if (item.name.endsWith(extension)) found.push(location);
  }
  return found;
}
async function exists(file) { try { return await stat(file); } catch { return null; } }
const sources = new Map();
for (const file of await files('_posts', '.md')) {
  const slug = path.basename(file, '.md').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  if (sources.has(slug)) failures.push(`Duplicate post source: ${slug}`);
  sources.set(slug, file);
}
if (Object.keys(legacy).length !== 19) failures.push('Expected 19 published legacy post hashes');
for (const slug of Object.keys(legacy)) {
  const source = sources.get(slug);
  if (!source) failures.push(`Missing legacy source: ${slug}`);
  else if (createHash('sha256').update(await readFile(source)).digest('hex') !== legacy[slug].sourceHash) failures.push(`Changed source bytes: ${source}`);
  const location = path.join(root, 'posts', slug, 'index.html');
  const info = await exists(location);
  if (!info) { failures.push(`Missing post: ${slug}`); continue; }
  const oldPage = cheerio.load(legacy[slug].html);
  const newPage = cheerio.load(await readFile(location, 'utf8'));
  if (!newPage('article[data-preserved="true"]').length) failures.push(`Legacy article is no longer preserved: ${slug}`);
  for (const element of newPage('.article-content a.anchor').toArray()) {
    const anchor = newPage(element);
    const heading = anchor.parent().clone();
    heading.find('a.anchor').remove();
    const label = anchor.attr('aria-label')?.trim();
    if (!label || !heading.text().trim() || !label.includes(heading.text().trim())) failures.push(`Heading permalink needs an accessible name: ${slug}${anchor.attr('href')}`);
    accessibleAnchors++;
  }
  for (const element of newPage('.article-content .highlight, .article-content pre, .article-content .katex-display').toArray()) {
    const block = newPage(element);
    if (block.parents('.highlight, pre, .katex-display').length) {
      if (block.attr('tabindex') === '0') failures.push(`Nested code/math block adds a duplicate tab stop: ${slug}`);
      continue;
    }
    if (block.attr('tabindex') !== '0' || block.attr('role') !== 'group' || !block.attr('aria-label')?.trim()) failures.push(`Outer code/math block needs keyboard focus and a name: ${slug}`);
    focusableBlocks++;
  }
  for (const element of oldPage('[id]').toArray()) {
    const id = oldPage(element).attr('id');
    if (!newPage(`[id="${id.replaceAll('"', '\\"')}"]`).length) failures.push(`Missing fragment ${slug}#${id}`);
  }
  if (newPage('article[data-preserved="true"]').length) {
    const content = newPage('.article-content');
    content.find('.article-toc').remove();
    content.find('.katex-display').each((index, element) => {
      const expression = newPage(element).find('annotation[encoding="application/x-tex"]').first().text();
      newPage(element).replaceWith(`\\[${expression}\\]`);
    });
    content.find('.katex').each((index, element) => {
      const expression = newPage(element).find('annotation[encoding="application/x-tex"]').first().text();
      newPage(element).replaceWith(`\\(${expression}\\)`);
    });
    const normalize = value => value.replace(/\s+/g, ' ').trim();
    if (normalize(content.text()) !== normalize(oldPage.root().text())) failures.push(`Changed rendered content: ${slug}`);
  }
}
for (const kind of ['categories', 'tags']) for (const slug of Object.keys(taxonomy[kind])) {
  const location = path.join(root, kind, slug, 'index.html');
  if (!await exists(location)) { failures.push(`Missing ${kind}: ${slug}`); continue; }
  const $ = cheerio.load(await readFile(location, 'utf8'));
  for (const member of taxonomy[kind][slug]) {
    if (!$(`a[href="/posts/${member}/"]`).length) failures.push(`Missing ${kind} member: ${slug}/${member}`);
  }
}
const pages = await files(root);
let checked = 0;
const fragments = new Map();
async function ids(file) {
  if (!fragments.has(file)) {
    const $ = cheerio.load(await readFile(file, 'utf8'));
    fragments.set(file, new Set($('[id],a[name]').map((index, element) => $(element).attr('id') || $(element).attr('name')).get()));
  }
  return fragments.get(file);
}
for (const file of pages) {
  const $ = cheerio.load(await readFile(file, 'utf8'));
  for (const element of $('a[href],img[src],script[src],link[href]').toArray()) {
    const ref = $(element).attr('href') || $(element).attr('src');
    if (!ref || (!ref.startsWith('/') && !ref.startsWith('#')) || ref.startsWith('//')) continue;
    checked++;
    const base = `https://example.test/${path.relative(root, file).replace(/index\.html$/, '')}`;
    const url = new URL(ref, base);
    let pathname;
    try { pathname = decodeURIComponent(url.pathname); } catch { failures.push(`Bad URL ${ref} in ${file}`); continue; }
    let target = path.resolve(root, `.${pathname}`);
    if (target !== root && !target.startsWith(`${root}${path.sep}`)) { failures.push(`Unsafe path ${ref}`); continue; }
    let info = await exists(target);
    if (info?.isDirectory()) { target = path.join(target, 'index.html'); info = await exists(target); }
    if (!info) { failures.push(`Missing ${ref} in ${path.relative(root, file)}`); continue; }
    if (url.hash && target.endsWith('.html')) {
      let fragment;
      try { fragment = decodeURIComponent(url.hash.slice(1)); } catch { failures.push(`Bad fragment ${ref}`); continue; }
      if (fragment && !(await ids(target)).has(fragment)) failures.push(`Missing fragment ${ref} in ${path.relative(root, file)}`);
    }
  }
}
const feed = cheerio.load(await readFile(path.join(root, 'feed.xml'), 'utf8'), { xmlMode: true });
const feedIds = new Set(feed('entry > id').map((index, element) => feed(element).text()).get());
for (const slug of Object.keys(legacy)) if (!feedIds.has(`https://wildchaser1703.github.io/posts/${encodeURIComponent(slug)}/`)) failures.push(`Missing feed entry: ${slug}`);
const home = cheerio.load(await readFile(path.join(root, 'index.html'), 'utf8'));
const portfolio = JSON.parse(await readFile('data/portfolio.json', 'utf8'));
const engineering = JSON.parse(await readFile('data/engineering-projects.json', 'utf8'));
const notes = JSON.parse(await readFile('data/field-notes.json', 'utf8'));
const art = cheerio.load(await readFile(path.join(root, 'art/index.html'), 'utf8'));
const directory = cheerio.load(await readFile(path.join(root, 'projects/index.html'), 'utf8'));
const check = (condition, message) => { if (!condition) failures.push(message); };
check(home('.hero-lede').text().trim() === 'I build the machinery around language models.', 'Homepage needs a clear technical introduction');
check(home('.hero-detail').text().trim() === 'Where an answer comes from, what a system does, and what happens when it fails.', 'Homepage introduction needs its supporting question');
check(home('.writing-intro').text().includes('Thinking in public'), 'Homepage writing needs its narrative connection');
check(home('.personal-section h2').text().trim() === 'Other ways of paying attention.', 'Creative work should not be split off as an afterthought');
check(home('.site-footer .footer-links a').length === 4 && home('.site-footer .footer-links a').map((index, element) => home(element).attr('href')).get().every(href => /^https:\/\//.test(href)), 'Footer should contain only social links');
check(home('.home-art-grid .art-item').length === 3 && home('.home-art-grid a[data-gallery="home-art"]').length === 3, 'Homepage must show three selected artworks in its own gallery');
check(portfolio.art.length === 10 && art('.art-item').length === 10 && art('a[data-gallery="art"]').length === 10, 'Full art collection must contain ten works');
check(!portfolio.art.some(item => /krishna|statue of liberty/i.test(item.title)), 'Removed artwork remains in portfolio data');
for (const [label, $] of [['home', home], ['art', art]]) {
  check(!/krishna|statue of liberty/i.test($('main').text()), `Removed artwork remains on ${label}`);
  for (const element of $('.art-item').toArray()) {
    const title = $(element).find('img').attr('alt');
    const item = portfolio.art.find(item => item.title === title);
    check(item && $(element).find('a').attr('href') === `/assets/art/${encodeURIComponent(item.file)}`, `Incorrect art link: ${label}/${title}`);
  }
}
for (const file of ['assets/art/4c6a3e30-3653-4cbc-830d-8539896ef074.jpg', 'dist/assets/art/4c6a3e30-3653-4cbc-830d-8539896ef074.jpg']) check(!await exists(file), `Removed painting file remains: ${file}`);
const navigation = ['Projects', 'Writing', 'Art', 'Food', 'Search'];
for (const file of pages) {
  const $ = cheerio.load(await readFile(file, 'utf8'));
  const links = $('.site-header nav a');
  check(JSON.stringify(links.map((i, el) => $(el).attr('href')).get()) === JSON.stringify(navigation.map(name => `/${name.toLowerCase()}/`)), `Incorrect navigation: ${path.relative(root, file)}`);
  check(!$('.site-header nav').is('[hidden]') && $('.menu-toggle[hidden][aria-expanded="false"]').length === 1, `Navigation must work without JavaScript: ${path.relative(root, file)}`);
}
const sculpture = home('figure[data-sculpture]');
check(sculpture.length === 1 && sculpture.attr('data-renderer') === 'static' && sculpture.attr('data-paused') === 'true' && sculpture.attr('data-variant') === '0', 'Sculpture must start as a static poster');
check(sculpture.find('img.sculpture-poster:not([hidden])').length === 1 && sculpture.find('[data-sculpture-next]').length === 1 && sculpture.find('[data-sculpture-toggle]').length === 1, 'Missing sculpture poster or controls');
const selectedEngineering = engineering.filter(item => !item.archived);
check(selectedEngineering.length === 6, 'Expected six curated ML/AI projects');
check(directory('.directory-row[data-kind]').length === 6 && directory('.directory-row[hidden]').length === 0, 'Selected directory must show six projects');
const kinds = ['All', 'LLM systems', 'Machine learning'];
check(JSON.stringify(directory('.directory-filters button[data-filter]').map((i, el) => directory(el).attr('data-filter')).get()) === JSON.stringify(kinds), 'Incorrect directory filters');
for (const kind of kinds) check(directory(`button[data-filter="${kind}"]`).attr('aria-pressed') === String(kind === 'All'), `Incorrect initial filter state: ${kind}`);
const archive = cheerio.load(await readFile(path.join(root, 'projects/archive/index.html'), 'utf8'));
check(archive('.directory-row').length === 9, 'Earlier work must remain accessible in the archive');
for (const slug of portfolio.projects.map(p=>p.slug)) {
  check(archive(`.directory-row[href="/projects/${slug}/"]`).length === 1, `Missing archived project: ${slug}`);
  check(directory(`.directory-row[href="/projects/${slug}/"]`).length === 0, `Archived project promoted into selection: ${slug}`);
}
const expectedItems = selectedEngineering.map(item => ({...item,route:`/projects/${item.slug}/`}));
for (const item of expectedItems) {
  const row = directory(`.directory-row[href="${item.route}"]`);
  check(row.length === 1 && row.attr('data-kind') === item.kind, `Missing or miscategorized directory item: ${item.slug}`);
  check(row.attr('data-preview')?.startsWith(`/assets/img/project-${item.slug}.svg?v=`) && row.find('img').attr('src') === row.attr('data-preview'), `Missing desktop/mobile preview: ${item.slug}`);
}
check(directory('a[href="/projects/retail-demand-forecasting-system/"]').length === 1, 'Keep forecasting visible as a workbench');
for (const project of selectedEngineering) check(directory(`a[href="/projects/${project.slug}/"]`).text().includes(project.stage), `Missing readiness label: ${project.slug}`);
const featuredProjects = ['durable-tool-runs'];
check(home('.selected-project').length === featuredProjects.length, 'Homepage should feature the current engineering project');
for (const slug of featuredProjects) check(home(`.selected-project a[href="/projects/${slug}/"]`).length > 0, `Missing selected AI project: ${slug}`);
for (const project of engineering) {
  const $ = cheerio.load(await readFile(path.join(root, 'projects', project.slug, 'index.html'), 'utf8'));
  check($('.case-body').length === 1, `Missing AI case body: ${project.slug}`);
  // Preserve source-reviewed results and caveats rather than inventing benchmark expectations.
  for (const text of [project.result, ...project.limitations]) check($('.case-body').text().includes(text), `Missing evidence limitation in ${project.slug}: ${text}`);
  for (const evidence of project.evidence) check($(`.case-body a[href="${evidence.url}"]`).length > 0, `Missing source evidence: ${project.slug}/${evidence.label}`);
}
const foodPhotos = JSON.parse(await readFile('data/food.json', 'utf8'));
const foodPage = cheerio.load(await readFile(path.join(root, 'food/index.html'), 'utf8'));
if (foodPhotos.length !== 76 || foodPage('.food-photo').length !== 76 || foodPage('.food-photo a[data-gallery="food"]').length !== 76) failures.push('Food collection must contain 76 photos');
check(foodPage('.food-selection').length === 3 && foodPage('.food-selection a[data-gallery="food-selected"]').length === 3, 'Food opener must have three independently grouped selections');
for (const element of foodPage('.food-selection').toArray()) {
  const photo = foodPhotos.find(photo => foodPage(element).find('a').attr('href') === `/assets/img/food/${encodeURIComponent(photo.file)}`);
  check(photo && foodPage(element).find('figcaption').text() === photo.caption, 'Curated food caption differs from the collection');
}
for (const [index, photo] of foodPhotos.entries()) {
  if (typeof photo.caption !== 'string' || !/^(?:\S+\s+){1,2}\S+$/.test(photo.caption.trim())) failures.push(`Food caption must have two or three words: ${photo.file}`);
  if (foodPage('.food-photo').eq(index).find('figcaption').text() !== photo.caption) failures.push(`Food caption is missing: ${photo.file}`);
}
const search = JSON.parse(await readFile(path.join(root, 'search-index.json'), 'utf8'));
for (const slug of Object.keys(legacy)) if (!search.some(item => item.url === `/posts/${encodeURIComponent(slug)}/`)) failures.push(`Missing search entry: ${slug}`);
if (!pages.length || !Object.keys(legacy).length) failures.push('No pages or published posts were checked');
if (!accessibleAnchors || !focusableBlocks) failures.push('No accessible heading anchors or focusable code/math blocks were checked');
if (failures.length) { console.error(failures.join('\n')); process.exitCode = 1; }
else {
  console.log(`Checked ${pages.length} pages, ${checked} internal references, ${Object.keys(legacy).length} published posts and their fragments.`);
  console.log('Verified 19 source-byte hashes, redesigned navigation, 6 selected projects, 9 archived entries, 6 ML/AI case studies, 3 selected / 10 full artworks, and 3 selected / 76 full food photos.');
  console.log(`Verified ${accessibleAnchors} named heading permalinks and ${focusableBlocks} keyboard-focusable code/math blocks.`);
}
