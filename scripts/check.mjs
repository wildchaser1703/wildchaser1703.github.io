import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import * as cheerio from 'cheerio';

const root = path.resolve('dist');
const legacy = JSON.parse(await readFile('data/legacy-posts.json', 'utf8'));
const taxonomy = JSON.parse(await readFile('data/legacy-taxonomy.json', 'utf8'));
const failures = [];
async function files(directory) {
  const found = [];
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const location = path.join(directory, item.name);
    if (item.isDirectory()) found.push(...await files(location));
    else if (item.name.endsWith('.html')) found.push(location);
  }
  return found;
}
async function exists(file) { try { return await stat(file); } catch { return null; } }
for (const slug of Object.keys(legacy)) {
  const location = path.join(root, 'posts', slug, 'index.html');
  const info = await exists(location);
  if (!info) { failures.push(`Missing post: ${slug}`); continue; }
  const oldPage = cheerio.load(legacy[slug].html);
  const newPage = cheerio.load(await readFile(location, 'utf8'));
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
if (home('.art-item').length !== portfolio.art.length || portfolio.art.some(item => /krishna/i.test(item.title))) failures.push('Artwork selection is incorrect');
if (home('.site-header a[href="/about/"]').length) failures.push('Removed About section is still in the navigation');
for (const project of portfolio.projects) if (!home(`a[href="/projects/${project.slug}/"]`).length) failures.push(`Missing project: ${project.slug}`);
const search = JSON.parse(await readFile(path.join(root, 'search-index.json'), 'utf8'));
for (const slug of Object.keys(legacy)) if (!search.some(item => item.url === `/posts/${encodeURIComponent(slug)}/`)) failures.push(`Missing search entry: ${slug}`);
if (!pages.length || !Object.keys(legacy).length) failures.push('No pages or published posts were checked');
if (failures.length) { console.error(failures.join('\n')); process.exitCode = 1; }
else console.log(`Checked ${pages.length} pages, ${checked} internal references, ${Object.keys(legacy).length} published posts and their fragments.`);
