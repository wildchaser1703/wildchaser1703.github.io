import { createHash } from 'node:crypto';
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import footnote from 'markdown-it-footnote';
import taskLists from 'markdown-it-task-lists';
import hljs from 'highlight.js';
import katex from 'katex';
import * as cheerio from 'cheerio';
import sharp from 'sharp';
import { arrow, navigation, siteFooter, homePage, directoryItems, directoryPage, engineeringPage, diagramSvg, artPage, foodPage } from './design-components.mjs';
import { generatePosters } from './sculpture-poster.mjs';
import { justBecausePage, atlasPage, traditionPage, worksPage, workPage, makersPage, makerPage, regionsPage, regionPage, regionSlug, collections, collectionsPage, curatedPage, savedPage, contributePage } from './just-because.mjs';

const traditions = JSON.parse(await readFile('data/indian-art.json', 'utf8'));
const atlasWorks = JSON.parse(await readFile('data/atlas-works.json', 'utf8'));
const atlasMakers = JSON.parse(await readFile('data/atlas-makers.json', 'utf8'));
const curiosityProjects = JSON.parse(await readFile('data/curiosity-projects.json', 'utf8'));
const indiaOutline = JSON.parse(await readFile('data/india-outline.json', 'utf8'));
const atlasGeography = JSON.parse(await readFile('data/atlas-geography.json', 'utf8'));
const atlasRegions = [...new Set(traditions.flatMap(form => form.locations))].sort();
const registeredPlaces = [...atlasGeography.states, ...atlasGeography.unionTerritories, ...atlasGeography.beyondIndia];
if (atlasGeography.states.length !== 28 || atlasGeography.unionTerritories.length !== 8 || new Set(registeredPlaces.map(place => place.name)).size !== registeredPlaces.length || new Set(registeredPlaces.map(place => regionSlug(place.name))).size !== registeredPlaces.length || registeredPlaces.some(place => !atlasRegions.includes(place.name)) || atlasRegions.some(region => !registeredPlaces.some(place => place.name === region))) throw new Error('Atlas geography must cover all states, union territories and listed cross-border places.');
for (const [label, records] of [['tradition', traditions], ['work', atlasWorks], ['maker', atlasMakers]]) {
  if (new Set(records.map(record => record.slug)).size !== records.length || records.some(record => !/^[a-z0-9-]+$/.test(record.slug))) throw new Error(`Invalid or duplicate ${label} slug.`);
}
for (const form of traditions) {
  if (!form.sources?.length || !form.sources.every(source => /^https:\/\//.test(source.url))) throw new Error(`Missing source for ${form.slug}`);
  for (const index of [...(form.traditionSourceRefs || []), ...(form.processSourceRefs || [])]) if (!Number.isInteger(index) || !form.sources[index]) throw new Error(`Invalid source reference for ${form.slug}`);
}
const formIds = new Set(traditions.map(form => form.slug));
if (atlasWorks.some(work => !formIds.has(work.tradition) || !work.recordUrl || !work.title || !work.maker || (work.image && (!work.imageSource || !work.credit)))) throw new Error('Invalid artwork provenance.');
if (atlasMakers.some(maker => !maker.url || maker.traditions.some(slug => !formIds.has(slug)))) throw new Error('Invalid maker attribution.');
if (collections.some(group => group.works.some(slug => !atlasWorks.some(work => work.slug === slug)))) throw new Error('Invalid curated collection.');
const root = process.cwd();
const out = path.join(root, 'dist');
const portfolio = JSON.parse(await readFile('data/portfolio.json', 'utf8'));
const foodPhotos = JSON.parse(await readFile('data/food.json', 'utf8'));
const engineering = JSON.parse(await readFile('data/engineering-projects.json', 'utf8'));
const fieldNotes = JSON.parse(await readFile('data/field-notes.json', 'utf8'));
const legacy = JSON.parse(await readFile('data/legacy-posts.json', 'utf8'));
const legacyTaxonomy = JSON.parse(await readFile('data/legacy-taxonomy.json', 'utf8'));
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const slugify = value => String(value).toLowerCase().trim().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '');
const sha = value => createHash('sha256').update(value).digest('hex');
// A release must never reuse cached styles or modules from a different build.
const assetInputs = ['package-lock.json', 'scripts/build.mjs', 'scripts/design-components.mjs', 'scripts/just-because.mjs', 'data/indian-art.json', 'data/atlas-works.json', 'data/atlas-makers.json', 'data/curiosity-projects.json', 'data/india-outline.json', 'data/atlas-geography.json', 'assets/css/site.css', ...(await readdir('assets/js')).filter(file => file.endsWith('.js')).sort().map(file => `assets/js/${file}`)];
const assetVersion = sha(Buffer.concat(await Promise.all(assetInputs.map(file => readFile(file))))).slice(0, 16);
const versionAssets = html => html
  .replace(/((?:src|href)="\/assets\/(?:css|js)\/[^"?]+\.(?:css|js))"/g, `$1?v=${assetVersion}"`)
  .replace(/((?:src|data-preview)="\/assets\/img\/(?:project-[^"?]+\.svg|sculpture-\d+\.webp))"/g, `$1?v=${assetVersion}"`);
function renderMath(html) {
  const $ = cheerio.load(html, null, false);
  const nodes = [];
  function collect(node) {
    if (node.type === 'text') nodes.push(node);
    for (const child of node.children || []) collect(child);
  }
  collect($.root()[0]);
  for (const node of nodes) {
    if ($(node).parents('pre, code').length) continue;
    const expression = /\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g;
    if (!expression.test(node.data)) continue;
    expression.lastIndex = 0;
    let result = '', position = 0, match;
    while ((match = expression.exec(node.data))) {
      result += esc(node.data.slice(position, match.index));
      result += katex.renderToString(match[1] ?? match[2], { displayMode: match[1] !== undefined, throwOnError: false });
      position = expression.lastIndex;
    }
    result += esc(node.data.slice(position));
    $(node).replaceWith(result);
  }
  return $.html();
}
const markdown = new MarkdownIt({ html: true, linkify: true, typographer: true, highlight(source, language) {
  if (language && hljs.getLanguage(language)) return `<pre class="hljs"><code>${hljs.highlight(source, { language }).value}</code></pre>`;
  return `<pre class="hljs"><code>${esc(source)}</code></pre>`;
}}).use(footnote).use(taskLists, { enabled: false });
const originalHeading = markdown.renderer.rules.heading_open || ((tokens, index, options, env, self) => self.renderToken(tokens, index, options));
markdown.renderer.rules.heading_open = (tokens, index, options, env, self) => {
  const text = tokens[index + 1]?.content || 'section';
  const base = slugify(text) || 'section';
  const count = env.headings.get(base) || 0;
  env.headings.set(base, count + 1);
  const prior = env.legacyHeadings?.get(base)?.[count];
  tokens[index].attrSet('id', prior || (count ? `${base}-${count}` : base));
  return originalHeading(tokens, index, options, env, self);
};

async function sourceFiles(directory) {
  const found = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const location = path.join(directory, entry.name);
    if (entry.isDirectory()) found.push(...await sourceFiles(location));
    else if (/^\d{4}-\d{2}-\d{2}-.+\.md$/.test(entry.name)) found.push(location);
  }
  return found;
}

const posts = [];
for (const file of await sourceFiles('_posts')) {
  const source = await readFile(file);
  const { data, content } = matter(source);
  const slug = path.basename(file, '.md').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  const snapshot = legacy[slug];
  const original = snapshot && snapshot.sourceHash === sha(source);
  const legacyHeadings = new Map();
  if (snapshot && !original) {
    const $ = cheerio.load(snapshot.html);
    $('h1[id],h2[id],h3[id],h4[id],h5[id],h6[id]').each((index, element) => {
      const heading = $(element).clone();
      heading.find('a.anchor').remove();
      const key = slugify(heading.text());
      const ids = legacyHeadings.get(key) || [];
      ids.push($(element).attr('id'));
      legacyHeadings.set(key, ids);
    });
    console.warn(`Rendering edited legacy article ${file} from Markdown; verify its formatting and fragments.`);
  }
  const date = new Date(data.date || path.basename(file).slice(0, 10));
  const title = data.title || snapshot?.title;
  if (!title || Number.isNaN(date.getTime())) throw new Error(`Invalid title or date in ${file}`);
  posts.push({ slug, title, date, categories: data.categories || [], tags: data.tags || [], original, html: original ? renderMath(snapshot.html) : renderMath(markdown.render(content, { headings: new Map(), legacyHeadings })), source: content });
}
posts.sort((left, right) => right.date - left.date || left.title.localeCompare(right.title));
const bySlug = new Map(posts.map(post => [post.slug, post]));
if (Object.keys(legacy).some(slug => !bySlug.has(slug))) throw new Error('A published article is missing from _posts.');

const dateText = date => new Intl.DateTimeFormat('en', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' }).format(date);
const postUrl = post => `/posts/${encodeURIComponent(post.slug)}/`;
const postRow = post => `<a class="post-row" href="${postUrl(post)}"><time datetime="${post.date.toISOString()}">${dateText(post.date)}</time><span>${esc(post.title)}</span>${arrow}</a>`;
const mark = '<img class="flow-mark" src="/assets/img/flow-mark.png" width="1254" height="1254" alt="" aria-hidden="true">';
const header = navigation(mark);
const footer = siteFooter().replace('<footer class="site-footer">', '<footer class="site-footer" id="contact">');
function page(title, body, options = {}) {
  const canonical = `https://wildchaser1703.github.io${options.url || '/'}`;
  const description = options.description || 'Toshali Mohapatra — AI engineer. Projects in retrieval, agents, and machine learning; writing, art, and photography.';
  const shareImage = `https://wildchaser1703.github.io/assets/img/${(options.url || '').startsWith('/just-because/indian-art/') ? 'indian-art-share.png' : 'flow-mark.png'}`;
  const section = (options.url || '/').split('/')[1];
  const activeSection = ['posts', 'archives', 'categories', 'tags'].includes(section) ? 'writing' : ['notes', 'studies'].includes(section) ? 'projects' : section;
  const activeHeader = header.replace(`href="/${activeSection}/"`, `href="/${activeSection}/" aria-current="page"`);
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)} · Toshali Mohapatra</title><meta name="description" content="${esc(description)}"><meta property="og:type" content="website"><meta property="og:title" content="${esc(title)} · Toshali Mohapatra"><meta property="og:description" content="${esc(description)}"><meta property="og:url" content="${esc(canonical)}"><meta property="og:image" content="${esc(shareImage)}"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="${esc(title)} · Toshali Mohapatra"><meta name="twitter:description" content="${esc(description)}"><meta name="twitter:image" content="${esc(shareImage)}"><link rel="canonical" href="${esc(canonical)}"><link rel="icon" type="image/png" href="/assets/img/flow-mark.png"><link rel="alternate" type="application/atom+xml" title="Toshali Mohapatra" href="/feed.xml"><link rel="stylesheet" href="/assets/css/katex.min.css"><link rel="stylesheet" href="/assets/css/site.css"><link rel="stylesheet" href="/assets/css/licensed-fonts.css"><script defer src="/assets/js/site.js"></script>${options.script || ''}</head><body>${activeHeader}<main id="main" tabindex="-1">${body}</main>${footer}</body></html>`;
}
async function writePage(route, contents) {
  const relative = route.replace(/^\/+|\/+$/g, '');
  const target = relative ? path.join(out, relative, 'index.html') : path.join(out, 'index.html');
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, versionAssets(contents));
}
function field() {
  let lines = '';
  for (let column = 0; column < 18; column++) for (let row = 0; row < 11; row++) {
    const variation = ((column * column + row) * 17) % 19;
    const x = column * 30 + 24, y = row * 30 + variation + 25;
    lines += `<line x1="${x}" y1="${y}" x2="${x + variation - 6}" y2="${y - 13 - Math.floor(variation / 2)}"/>`;
  }
  return `<svg viewBox="0 0 570 370" class="hero-field" aria-hidden="true">${lines}</svg>`;
}
const featuredSlugs = ['context-engineering', 'model-drift-in-production', 'persistence-against-the-tide'];
const featured = featuredSlugs.map(slug => bySlug.get(slug)).filter(Boolean);
const directory = directoryItems(engineering, portfolio, fieldNotes);
const home = homePage({engineering, art:portfolio.art, food:foodPhotos, writing:featured.map(postRow).join('')}) + `<section class="curiosity-page home-curiosity"><p class="kicker">MADE FROM CURIOSITY</p><h2>Just because.</h2><p>A growing home for projects beyond the expected, beginning with the Indian Art Atlas.</p><a class="text-link" href="/just-because/">Explore Just Because ↗</a></section>`;

await rm(out, { recursive: true, force: true });
await mkdir(path.join(out, 'assets'), { recursive: true });
await generatePosters(new URL('../dist/assets/img/', import.meta.url));
for (const directory of ['art', 'img', 'downloads']) await cp(path.join('assets', directory), path.join(out, 'assets', directory), { recursive: true });
// Keep the original logo source; the interface only needs a small raster.
await sharp('assets/img/flow-mark.png').resize(128, 128).png({ compressionLevel: 9 }).toFile(path.join(out, 'assets/img/flow-mark.png'));
const socialCard = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630"><rect width="1200" height="630" fill="#f6f2e9"/><rect x="42" y="42" width="1116" height="546" fill="none" stroke="#b9aa9a"/><text x="94" y="135" fill="#79564c" font-size="24" font-family="sans-serif" letter-spacing="4">JUST BECAUSE / VISUAL ARTS &amp; CRAFTS</text><text x="90" y="290" fill="#2b2926" font-family="Georgia,serif" font-size="110">Indian</text><text x="90" y="415" fill="#9d5945" font-family="Georgia,serif" font-style="italic" font-size="110">Art Atlas.</text><text x="96" y="525" fill="#514842" font-family="sans-serif" font-size="28">Toshali Mohapatra · A living index</text><circle cx="997" cy="311" r="86" fill="none" stroke="#9d5945" stroke-width="3"/><circle cx="997" cy="311" r="60" fill="none" stroke="#9d5945" stroke-width="2"/><circle cx="997" cy="311" r="28" fill="#9d5945"/></svg>`;
await sharp(Buffer.from(socialCard)).png().toFile(path.join(out, 'assets/img/indian-art-share.png'));
await cp('data/india-outline-license.txt', path.join(out, 'assets/img/india-outline-license.txt'));
await mkdir(path.join(out, 'assets', 'fonts'), { recursive: true });
// The design uses system fallbacks until licensed Sequel Sans / PP Fraktion Mono files are supplied.
// No fonts are fetched from third-party services.
const licensedFonts = [['sequel-sans-regular.woff2', 'Sequel Sans', 400], ['sequel-sans-semibold.woff2', 'Sequel Sans', 600], ['pp-fraktion-mono-regular.woff2', 'PP Fraktion Mono', 400]];
let licensedCss = '';
for (const [file, family, weight] of licensedFonts) {
  try { await stat(path.join('assets/fonts', file)); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
  await cp(path.join('assets/fonts', file), path.join(out, 'assets/fonts', file));
  licensedCss += `@font-face{font-family:'${family}';src:url('/assets/fonts/${file}') format('woff2');font-style:normal;font-weight:${weight};font-display:swap}\n`;
}
await mkdir(path.join(out, 'assets', 'css'), { recursive: true });
await mkdir(path.join(out, 'assets', 'js'), { recursive: true });
await writeFile(path.join(out, 'assets/css/licensed-fonts.css'), licensedCss || '/* Licensed typefaces pending; system fallbacks are intentional. */\n');
await mkdir(path.join(out, 'assets/js/vendor'), {recursive:true});
await cp('node_modules/three/build/three.module.min.js',path.join(out,'assets/js/vendor/three.module.js'));
await cp('node_modules/three/build/three.core.min.js',path.join(out,'assets/js/vendor/three.core.min.js'));
await cp('node_modules/three/LICENSE',path.join(out,'assets/js/vendor/THREE-LICENSE.txt'));
for (const file of ['site.js', 'indian-art.js', 'wait-study.js', 'sculpture.js', 'sculpture-shape.js', 'sculpture-model.js', 'sculpture-bloom.js']) await cp(path.join('assets/js', file), path.join(out, 'assets/js', file));
for (const file of ['site.js', 'sculpture.js', 'sculpture-shape.js', 'vendor/three.module.js']) {
  const target = path.join(out, 'assets/js', file);
  const source = await readFile(target, 'utf8');
  await writeFile(target, source.replace(/(['"])((?:\.\/|\/assets\/js\/)[^'"?]+\.js|\/search-index\.json)\1/g, `$1$2?v=${assetVersion}$1`));
}
await cp('assets/css/site.css', path.join(out, 'assets/css/site.css'));
await cp('node_modules/katex/dist/katex.min.css', path.join(out, 'assets/css/katex.min.css'));
await cp('node_modules/katex/dist/fonts', path.join(out, 'assets/css/fonts'), { recursive: true });
await cp('public/sw.min.js', path.join(out, 'sw.min.js'));
for (const item of directory) await writeFile(path.join(out, 'assets/img', `project-${item.slug}.svg`), diagramSvg(item));
await writePage('/', page('Home', home, {description:'Systems around language models: evidence, failure, writing, art, and photography by Toshali Mohapatra.',script:'<script type="module" src="/assets/js/sculpture.js"></script>'}));
await writePage('/projects/', page('Projects', directoryPage(directory.filter(item => engineering.some(project => project.slug === item.slug && !project.archived))), {url:'/projects/'}));
await writePage('/projects/archive/', page('Project archive', directoryPage(directory.filter(item => !engineering.some(project => project.slug === item.slug && !project.archived)), true), {url:'/projects/archive/'}));
await writePage('/art/', page('Art', artPage(portfolio.art), {url:'/art/'}));
await writePage('/just-because/', page('Just Because', justBecausePage(curiosityProjects), {url:'/just-because/',description:'A collection of personal projects made from curiosity.'}));
await writePage('/just-because/indian-art/', page('Indian Art Atlas', atlasPage(traditions, atlasGeography), {url:'/just-because/indian-art/',description:'An evolving index of visual arts and crafts.',script:'<script defer src="/assets/js/indian-art.js"></script>'}));
for (const [index, form] of traditions.entries()) {
  if (!/^[a-z0-9-]+$/.test(form.slug) || !form.sources?.length) throw new Error('Invalid atlas record');
  const route = `/just-because/indian-art/${form.slug}/`;
  await writePage(route, page(`${form.title} · Indian Art Atlas`, traditionPage(form, index, atlasWorks, atlasMakers), {url:route, description:form.intro, script:'<script defer src="/assets/js/indian-art.js"></script>'}));
}
const atlasBase = '/just-because/indian-art/';
const atlasScript = '<script defer src="/assets/js/indian-art.js"></script>';
const atlasExtraPaths = ['works/', 'makers/', 'regions/', 'collections/', 'collection/', 'contribute/', ...atlasWorks.map(w => `works/${w.slug}/`), ...atlasMakers.map(m => `makers/${m.slug}/`), ...atlasRegions.map(r => `regions/${regionSlug(r)}/`), ...collections.map(c => `collections/${c.slug}/`)].map(path => atlasBase + path);
const atlasRender = async (route, title, body, description) => writePage(route, page(`${title} · Indian Art Atlas`, body, {url:route, description, script:atlasScript}));
await atlasRender(`${atlasBase}works/`, 'Works', worksPage(atlasWorks, traditions), 'Explore documented works in the Indian Art Atlas.');
for (const work of atlasWorks) await atlasRender(`${atlasBase}works/${work.slug}/`, work.title, workPage(work, traditions, atlasWorks, atlasMakers), work.description || `Explore ${work.title} in the Indian Art Atlas.`);
await atlasRender(`${atlasBase}makers/`, 'Makers', makersPage(atlasMakers, traditions), 'Meet artists and artisans in the Indian Art Atlas.');
for (const maker of atlasMakers) await atlasRender(`${atlasBase}makers/${maker.slug}/`, maker.name, makerPage(maker, traditions, atlasWorks), maker.description);
await atlasRender(`${atlasBase}regions/`, 'Places', regionsPage(traditions, indiaOutline, atlasGeography), 'Explore traditions by place in the Indian Art Atlas.');
for (const region of atlasRegions) await atlasRender(`${atlasBase}regions/${regionSlug(region)}/`, region, regionPage(region, traditions), `Explore traditions connected with ${region}.`);
await atlasRender(`${atlasBase}collections/`, 'Collections', collectionsPage(), 'Explore curated paths and create a collection of art and craft records.');
for (const collection of collections) await atlasRender(`${atlasBase}collections/${collection.slug}/`, collection.title, curatedPage(collection, atlasWorks, traditions), collection.description);
await atlasRender(`${atlasBase}collection/`, 'Saved works', savedPage(atlasWorks, traditions), 'Save and share a personal selection of works.');
await atlasRender(`${atlasBase}contribute/`, 'Contribute', contributePage(), 'Share a correction, source or first-hand perspective with the Indian Art Atlas.');
for (const project of engineering) await writePage(`/projects/${project.slug}/`, page(project.title, engineeringPage(project), {url:`/projects/${project.slug}/`,description:project.summary}));

if (!Array.isArray(foodPhotos)) throw new Error('Food photos must be an array.');
await Promise.all(foodPhotos.map(async (photo, index) => {
  if (!photo || typeof photo.file !== 'string' || !photo.file || path.basename(photo.file) !== photo.file || typeof photo.alt !== 'string' || !photo.alt.trim() || (photo.caption !== undefined && typeof photo.caption !== 'string') || (photo.preview !== undefined && (typeof photo.preview !== 'string' || !photo.preview || path.basename(photo.preview) !== photo.preview))) {
    throw new Error(`Invalid food photo entry at index ${index}.`);
  }
  const hasDimensions = photo.width !== undefined || photo.height !== undefined;
  if (hasDimensions && (!Number.isSafeInteger(photo.width) || photo.width <= 0 || !Number.isSafeInteger(photo.height) || photo.height <= 0)) {
    throw new Error(`Invalid food photo dimensions at index ${index}.`);
  }
  if (!(await stat(path.join('assets/img/food', photo.file))).isFile()) throw new Error(`Missing food photo: ${photo.file}`);
  if (photo.preview && !(await stat(path.join('assets/img/food/previews', photo.preview))).isFile()) throw new Error(`Missing food preview: ${photo.preview}`);
}));
await writePage('/food/', page('Food', foodPage(foodPhotos), {url:'/food/',description:'Meals, drinks, and sweet things. Food photographs by Toshali Mohapatra.'}));

const browseNav = `<nav class="browse-nav" aria-label="Browse writing"><a href="/writing/">Latest</a><a href="/archives/">Archive</a><a href="/categories/">Categories</a><a href="/tags/">Tags</a><a href="/search/">Search ↗</a></nav>`;
function writingPage(title, entries, intro = '') {
  return `<section class="inner-page">${browseNav}<header class="page-heading"><p class="kicker">WORDS &amp; IDEAS</p><h1>${esc(title)}</h1>${intro ? `<p>${esc(intro)}</p>` : ''}</header><div class="post-list">${entries.map(postRow).join('')}</div></section>`;
}
await writePage('/writing/', page('Writing', writingPage('Writing', posts.slice(0, 10), 'Notes on systems, research, and the things I am still working out.') + (posts.length > 10 ? '<div class="pagination"><a href="/page2/">Older writing →</a></div>' : ''), { url: '/writing/' }));
for (let current = 2; current <= Math.ceil(posts.length / 10); current++) {
  const previous = current === 2 ? '/writing/' : `/page${current - 1}/`;
  const next = current < Math.ceil(posts.length / 10) ? `<a href="/page${current + 1}/">Older writing →</a>` : '';
  await writePage(`/page${current}/`, page(`Writing · ${current}`, writingPage('More writing', posts.slice((current - 1) * 10, current * 10)) + `<nav class="pagination" aria-label="Writing pages"><a href="${previous}">← Newer writing</a>${next}</nav>`, { url: `/page${current}/` }));
}
await writePage('/archives/', page('Archive', writingPage('The archive', posts, 'All notes, in one place.'), { url: '/archives/' }));

const taxonomy = { categories: new Map(), tags: new Map() };
for (const kind of ['categories', 'tags']) {
  for (const [slug, members] of Object.entries(legacyTaxonomy[kind])) taxonomy[kind].set(slug, { label: slug, posts: members.map(member => bySlug.get(member)).filter(Boolean) });
  for (const post of posts) for (const value of post[kind]) {
    const slug = slugify(value);
    if (!taxonomy[kind].has(slug)) taxonomy[kind].set(slug, { label: value, posts: [] });
    const group = taxonomy[kind].get(slug);
    group.label = value;
    if (!group.posts.includes(post)) group.posts.push(post);
  }
  const groups = [...taxonomy[kind].entries()].sort((left, right) => left[1].label.localeCompare(right[1].label));
  const label = kind === 'tags' ? 'Tags' : 'Categories';
  const cards = groups.map(([slug, group]) => `<a class="taxonomy-item" href="/${kind}/${encodeURIComponent(slug)}/"><span>${esc(group.label)}</span><small>${group.posts.length} ${group.posts.length === 1 ? 'note' : 'notes'}</small><span aria-hidden="true">↗</span></a>`).join('');
  await writePage(`/${kind}/`, page(label, `<section class="inner-page">${browseNav}<header class="page-heading"><p class="kicker">FIND A THREAD</p><h1>${label}</h1></header><div class="taxonomy-grid">${cards}</div></section>`, { url: `/${kind}/` }));
  for (const [slug, group] of groups) {
    group.posts.sort((left, right) => right.date - left.date);
    await writePage(`/${kind}/${slug}/`, page(group.label, writingPage(group.label, group.posts), { url: `/${kind}/${slug}/` }));
  }
}

for (const post of posts) {
  const categories = post.categories.map(value => `<a href="/categories/${slugify(value)}/">${esc(value)}</a>`).join(' ');
  const tags = post.tags.map(value => `<a href="/tags/${slugify(value)}/">${esc(value)}</a>`).join(' ');
  const article = cheerio.load(post.html, null, false);
  // Add accessible controls to the rendering without changing published source or text.
  article('a.anchor').each((index, element) => {
    const heading = article(element).parent().clone();
    heading.find('a.anchor').remove();
    article(element).attr('aria-label', `Link to ${heading.text().trim()}`);
  });
  article('.highlight, pre, .katex-display').each((index, element) => {
    if (article(element).parents('.highlight, pre, .katex-display').length) return;
    article(element).attr({ tabindex: '0', role: 'group', 'aria-label': article(element).hasClass('katex-display') ? 'Equation' : 'Code example' });
  });
  const renderedArticle = article.html();
  const firstHeading = article.root().children().first();
  const normalizeTitle = text => text.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/\s+/g, ' ').trim();
  const titleInContent = firstHeading.is('h1') && normalizeTitle(firstHeading.text()) === normalizeTitle(post.title);
  const headings = article('h2[id],h3[id]').map((index, element) => {
    const heading = article(element).clone();
    heading.find('a.anchor').remove();
    return `<li class="${element.tagName === 'h3' ? 'subheading' : 'heading'}"><a href="#${encodeURIComponent(article(element).attr('id'))}">${esc(heading.text().trim())}</a></li>`;
  }).get().join('');
  const contents = headings ? `<details class="article-toc"><summary>On this page</summary><ol>${headings}</ol></details>` : '';
  const articleHtml = titleInContent ? renderedArticle.replace(/^(\s*<h1\b[^>]*>[\s\S]*?<\/h1>)/i, (match, heading) => heading + contents) : renderedArticle;
  const body = `<article class="article-page"${post.original ? ' data-preserved="true"' : ''}${titleInContent ? ' data-title-in-content="true"' : ''}><a class="back-link" href="/writing/">← All writing</a><header class="article-header"><p class="kicker">${dateText(post.date)} / WRITING</p>${titleInContent ? '' : `<h1>${esc(post.title)}</h1>`}<div class="article-topics">${categories}</div></header>${titleInContent ? '' : contents}<div class="article-content">${articleHtml}</div><footer class="article-footer"><p class="kicker">FOLLOW THE THREAD</p><div class="article-tags">${tags}</div><a class="text-link" href="/writing/">← More writing</a></footer></article>`;
  await writePage(`/posts/${post.slug}/`, page(post.title, body, { url: postUrl(post), description: post.source.replace(/[#*`\[\]]/g, '').replace(/\s+/g, ' ').trim().slice(0, 155) }));
}
const searchIndex = posts.map(post => ({ title: post.title, url: postUrl(post), date: dateText(post.date), text: post.source.replace(/\s+/g, ' ').slice(0, 10000), categories: post.categories, tags: post.tags }));
for (const note of fieldNotes) searchIndex.push({ title: note.title, url: `/notes/${note.slug}/`, date: 'Field note', text: note.paragraphs.join(' '), categories: ['Field notes'], tags: [] });
for (const project of engineering) searchIndex.push({title:project.title,url:`/projects/${project.slug}/`,date:'AI engineering',text:[project.summary,project.problem,...project.approach].join(' '),categories:['AI engineering'],tags:project.stack});
for (const project of portfolio.projects) searchIndex.push({ title: project.title, url: `/projects/${project.slug}/`, date: 'Project', text: project.description, categories: ['Projects'], tags: [] });
searchIndex.push({title:'Just Because',url:'/just-because/',date:'Personal projects',text:'Projects made from curiosity, including the Indian Art Atlas',categories:['Projects'],tags:['Art']});
searchIndex.push({title:'Indian Art Atlas',url:atlasBase,date:'Visual arts & crafts',text:'Traditions, places, makers and documented works in Indian visual arts and crafts',categories:['Indian Art Atlas'],tags:['Art','Craft']});
for (const form of traditions) searchIndex.push({title:form.title,url:`${atlasBase}${form.slug}/`,date:'Art tradition',text:[form.intro,form.region,form.materials,...form.tradition,...form.process,...form.localNames.map(name=>name.text)].join(' '),categories:['Indian Art Atlas',form.category],tags:form.locations});
for (const work of atlasWorks) searchIndex.push({title:work.title,url:`${atlasBase}works/${work.slug}/`,date:'Artwork record',text:[work.maker,work.description,work.medium].join(' '),categories:['Indian Art Atlas'],tags:[work.tradition]});
for (const maker of atlasMakers) searchIndex.push({title:maker.name,url:`${atlasBase}makers/${maker.slug}/`,date:'Artist or maker',text:maker.description,categories:['Indian Art Atlas'],tags:maker.traditions});
await writeFile(path.join(out, 'search-index.json'), JSON.stringify(searchIndex));
await writePage('/search/', page('Search', `<section class="inner-page">${browseNav}<header class="page-heading"><p class="kicker">FIND A THREAD</p><h1>Search</h1><label class="search-label" for="search-input">Search writing, projects, notes, and the art atlas</label><input id="search-input" class="search-input" type="search" autocomplete="off" placeholder="Try ‘machine learning’"><p id="search-status" aria-live="polite">Start typing to find a note.</p></header><div id="search-results" class="post-list"></div><noscript><p>Search requires JavaScript. You can browse the <a href="/archives/">archive</a> instead.</p></noscript></section>`, { url: '/search/' }));

const study = `<article class="study-page"><a class="back-link" href="/studies/">← All studies</a><p class="kicker">STUDY 01 / TIME &amp; PERCEPTION</p><h1>What makes a wait feel long?</h1><p class="study-intro">The same four words take four seconds to arrive. Does seeing them sooner change the wait?</p><div class="experiment"><div class="experiment-head"><span>AN EXPERIMENT / 4 SECONDS</span><span>PRESET TRACE</span></div><p>This is a browser simulation, not a live model benchmark. Choose how the same scheduled output becomes visible.</p><div class="controls"><button type="button" data-mode="all">Show at once</button><button type="button" data-mode="ready">Show as ready</button><button type="button" data-stop disabled>Stop</button></div><p class="status" aria-live="polite" data-status>Choose a mode to begin.</p><div class="output" role="status" aria-label="Simulation output" aria-live="polite" data-output></div><div class="trace" role="group" data-trace aria-label="Timing marks"><span>1 s</span><span>2 s</span><span>3 s</span><span>4 s</span></div></div><section class="study-copy"><p class="kicker">THE COMPARISON / NOMINAL TIMINGS</p><h2>Same work. A different view of the wait.</h2><div class="comparison"><div><h3>Show at once</h3><p>First visible: <strong>4 s</strong><br>Final output: <strong>4 s</strong></p></div><div><h3>Show as ready</h3><p>First visible: <strong>1 s</strong><br>Final output: <strong>4 s</strong></p></div></div><p>The preset work schedule is one word each second. Only visibility changes: all four appear at the end in one mode, while each appears as ready in the other. In an actual service, time to first token and total time are separate metrics.</p><h2>Which wait felt longer?</h2><p>There is no answer being collected here. Your impression is a personal observation, not a research claim.</p><h3>Method</h3><p>Four preset words are scheduled at one-second intervals. The marks above represent each visible word; this is a simulation, not an inference trace.</p></section></article>`;
await writePage('/studies/', page('Studies', `<section class="inner-page"><header class="page-heading"><p class="kicker">SMALL EXPERIMENTS</p><h1>Questions, seen<br><em>from a few angles.</em></h1><p>Small experiments for thinking through how things work.</p></header><a class="study-card" href="/studies/what-makes-a-wait-feel-long/"><div><p class="kicker">STUDY 01 / TIME</p><h2>What makes a wait feel long?</h2><p>Four words, two ways of seeing them arrive.</p><span class="text-link">Open the study ↗</span></div>${field()}</a></section>`, { url: '/studies/' }));
await writePage('/studies/what-makes-a-wait-feel-long/', page('What makes a wait feel long?', study, { url: '/studies/what-makes-a-wait-feel-long/', script: '<script type="module" src="/assets/js/wait-study.js"></script>' }));
for (const note of fieldNotes) {
  const body = `<article class="article-page"><a class="back-link" href="/projects/">← Projects &amp; notes</a><header class="article-header"><p class="kicker">FROM THE FRUGAL SCIENCE LAB NOTEBOOK</p><h1>${esc(note.title)}</h1><p>${esc(note.intro)}</p></header><div class="article-content">${note.paragraphs.map(paragraph => `<p>${esc(paragraph)}</p>`).join('')}</div></article>`;
  await writePage(`/notes/${note.slug}/`, page(note.title, body, { url: `/notes/${note.slug}/` }));
}
for (const project of portfolio.projects) {
  const body = `<article class="article-page"><a class="back-link" href="/projects/">← Projects &amp; notes</a><header class="article-header"><p class="kicker">Browser &amp; code experiments</p><h1>${esc(project.title)}</h1></header><figure class="experiment-diagram"><img src="/assets/img/project-${esc(project.slug)}.svg" width="960" height="640" alt="Interaction diagram for ${esc(project.title)}"></figure><div class="article-content"><p>${esc(project.description)}</p><p><a href="${esc(project.url)}">View source on GitHub ↗</a></p></div></article>`;
  await writePage(`/projects/${project.slug}/`, page(project.title, body, { url: `/projects/${project.slug}/` }));
}
await writePage('/about/', page('About', '<section class="inner-page"><header class="page-heading"><p class="kicker">A LITTLE ABOUT ME</p><h1>About Toshali</h1></header><p>I’m Toshali Mohapatra, an AI engineer who likes thinking about how things work and what they mean to people.</p><p>This site brings together my engineering projects and writing, alongside art, photography and projects made out of curiosity.</p><p><a class="text-link" href="/just-because/">Explore Just Because ↗</a></p><p>Contact links and social profiles live in the footer.</p></section>', {url:'/about/'}));

const feedEntries = posts.map(post => `<entry><title>${esc(post.title)}</title><id>https://wildchaser1703.github.io${postUrl(post)}</id><link href="https://wildchaser1703.github.io${postUrl(post)}"/><updated>${post.date.toISOString()}</updated><content type="html">${esc(post.html)}</content></entry>`).join('');
await writeFile(path.join(out, 'feed.xml'), `<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Toshali Mohapatra</title><id>https://wildchaser1703.github.io/</id><link href="https://wildchaser1703.github.io/feed.xml" rel="self"/><updated>${(posts[0]?.date || new Date(0)).toISOString()}</updated>${feedEntries}</feed>`);
const paths = ['/', '/projects/', '/art/', '/about/', '/just-because/', '/just-because/indian-art/', ...traditions.map(form => `/just-because/indian-art/${form.slug}/`), ...atlasExtraPaths, ...engineering.map(item=>`/projects/${item.slug}/`), '/food/', '/writing/', '/archives/', '/categories/', '/tags/', '/search/', '/studies/', '/studies/what-makes-a-wait-feel-long/', ...portfolio.projects.map(project => `/projects/${project.slug}/`), ...fieldNotes.map(note => `/notes/${note.slug}/`), ...posts.map(postUrl), ...[...taxonomy.categories.keys()].map(slug => `/categories/${slug}/`), ...[...taxonomy.tags.keys()].map(slug => `/tags/${slug}/`)];
await writeFile(path.join(out, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${paths.map(url => `<url><loc>https://wildchaser1703.github.io${esc(url)}</loc></url>`).join('')}</urlset>`);
await writeFile(path.join(out, 'robots.txt'), 'User-agent: *\nAllow: /\nSitemap: https://wildchaser1703.github.io/sitemap.xml\n');
await writeFile(path.join(out, '404.html'), page('Not found', `<section class="inner-page"><p class="kicker">404 / LOST IN THE MARGINS</p><h1>That page isn’t here.</h1><a class="button" href="/">Back to the start ↗</a></section>`, { url: '/404.html' }));
await writeFile(path.join(out, '.nojekyll'), '');
await writeFile(path.join(out, '404.html'), versionAssets(await readFile(path.join(out, '404.html'), 'utf8')));
console.log(`Built ${posts.length} posts, ${taxonomy.categories.size} categories, and ${taxonomy.tags.size} tags.`);
