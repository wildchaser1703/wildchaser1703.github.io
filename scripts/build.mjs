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

const root = process.cwd();
const out = path.join(root, 'dist');
const portfolio = JSON.parse(await readFile('data/portfolio.json', 'utf8'));
const foodPhotos = JSON.parse(await readFile('data/food.json', 'utf8'));
const fieldNotes = JSON.parse(await readFile('data/field-notes.json', 'utf8'));
const legacy = JSON.parse(await readFile('data/legacy-posts.json', 'utf8'));
const legacyTaxonomy = JSON.parse(await readFile('data/legacy-taxonomy.json', 'utf8'));
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const slugify = value => String(value).toLowerCase().trim().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '');
const sha = value => createHash('sha256').update(value).digest('hex');
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
const postRow = post => `<a class="post-row" href="${postUrl(post)}"><time datetime="${post.date.toISOString()}">${dateText(post.date)}</time><span>${esc(post.title)}</span><span aria-hidden="true">↗</span></a>`;
const mark = '<img class="flow-mark" src="/assets/img/flow-mark.png" width="1254" height="1254" alt="" aria-hidden="true">';
const header = `<header class="site-header"><a class="brand" href="/" aria-label="Toshali Mohapatra, home"><span class="brand-mark" aria-hidden="true">${mark}</span><span>Toshali<br>Mohapatra</span></a><nav aria-label="Main navigation"><a href="/writing/">Writing</a><a href="/#projects">Projects</a><a href="/#art">Art</a><a href="/food/">Food</a><a href="/studies/">Studies</a><a class="search-link" href="/search/" aria-label="Search writing">Search <span aria-hidden="true">↗</span></a></nav></header>`;
const footer = `<footer class="site-footer"><div><span class="footer-mark" aria-hidden="true">${mark}</span><p>Made of questions,<br>ideas &amp; experiments.</p></div><div class="footer-links"><a href="/writing/">Writing</a><a href="/categories/">Categories</a><a href="/tags/">Tags</a><a href="/feed.xml">RSS</a><a href="mailto:toshalimohapatra1@gmail.com">Say hello ↗</a></div><small>© ${new Date().getUTCFullYear()} Toshali Mohapatra</small></footer>`;
function page(title, body, options = {}) {
  const canonical = `https://wildchaser1703.github.io${options.url || '/'}`;
  const description = options.description || 'Notes, experiments, and creative work by Toshali Mohapatra.';
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)} · Toshali Mohapatra</title><meta name="description" content="${esc(description)}"><link rel="canonical" href="${esc(canonical)}"><link rel="icon" type="image/png" href="/assets/img/flow-mark.png"><link rel="alternate" type="application/atom+xml" title="Toshali Mohapatra" href="/feed.xml"><link rel="stylesheet" href="/assets/css/katex.min.css"><link rel="stylesheet" href="/assets/css/site.css"><script defer src="/assets/js/site.js"></script>${options.script || ''}</head><body>${header}<main id="main">${body}</main>${footer}</body></html>`;
}
async function writePage(route, contents) {
  const relative = route.replace(/^\/+|\/+$/g, '');
  const target = relative ? path.join(out, relative, 'index.html') : path.join(out, 'index.html');
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, contents);
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
const projects = portfolio.projects.map((item, index) => `<a class="project-row" href="/projects/${esc(item.slug)}/"><span class="project-number">${String(index + 1).padStart(2, '0')}</span><span>${esc(item.title)}</span><span aria-hidden="true">↗</span></a>`).join('');
const fieldNoteCards = fieldNotes.map(item => `<a class="project-row" href="/notes/${esc(item.slug)}/"><span class="project-number">NOTE</span><span>${esc(item.title)}</span><span aria-hidden="true">↗</span></a>`).join('');
const art = portfolio.art.map((item, index) => `<figure class="art-item"><div class="art-image"><img src="/assets/art/${esc(item.file)}" alt="${esc(item.title)}" ${index > 2 ? 'loading="lazy"' : ''}></div><figcaption><strong>${esc(item.title)}</strong><span>${esc(item.caption)}</span></figcaption></figure>`).join('');
const favorites = portfolio.favorites.map(item => `<div class="favorite-item"><span class="kicker">${esc(item.kind)}</span><p>${esc(item.title)}</p>${item.kind === 'Food' ? '<a class="text-link food-home-link" href="/food/">Food photos <span aria-hidden="true">↗</span></a>' : ''}</div>`).join('');
const home = `<section class="home-hero"><div class="hero-copy"><p class="kicker">An ever-evolving collection</p><h1>Curious by<br><em>nature.</em></h1><p class="hero-lede">I’m Toshali. I build AI systems, draw, and follow ideas wherever they lead.</p><div class="hero-actions"><a class="button" href="/writing/">Explore the writing <span aria-hidden="true">↗</span></a><a class="text-link" href="#projects">See what I make <span aria-hidden="true">↓</span></a></div></div><div class="hero-visual"><div class="field-meta"><span>A SMALL STUDY IN VARIATION</span><span>01 / 01</span></div>${field()}<div class="field-caption"><span>A line, repeated.<br>Never quite the same.</span><span>FIG. 01</span></div></div><span class="hero-index" aria-hidden="true">T / M</span></section><section class="section-block" id="writing"><div class="section-heading"><div><p class="kicker">01 / WORDS &amp; IDEAS</p><h2>Notes from<br><em>the field.</em></h2></div><p>Thoughts on AI, engineering, and figuring things out as I go.</p></div><div class="post-list">${featured.map(postRow).join('')}</div><a class="text-link section-link" href="/writing/">All writing <span aria-hidden="true">↗</span></a></section><section class="section-block project-section" id="projects"><div class="section-heading"><div><p class="kicker">02 / EXPERIMENTS</p><h2>Making things<br><em>to learn things.</em></h2></div><p>Little explorations in code, interaction, and design—collected from my portfolio.</p></div><div class="projects-grid"><div class="project-note"><span class="project-glyph" aria-hidden="true">${mark}</span><p>Some ideas are best understood by making them.</p><small>CODE / PLAY / REPEAT</small></div><div class="project-list">${projects}${fieldNoteCards}</div></div></section><section class="section-block art-section" id="art"><div class="section-heading"><div><p class="kicker">03 / THE BONUS SECTION</p><h2>Beyond<br><em>the screen.</em></h2></div><p>A few paintings and pieces from my collection. A different way to notice the world.</p></div><div class="art-grid">${art}</div></section><section class="section-block favorites-section"><div class="section-heading"><div><p class="kicker">04 / LITTLE JOYS</p><h2>A few <em>favorites.</em></h2></div><p>A film, a book, and a kind of food I keep coming back to.</p></div><div class="favorites-grid">${favorites}</div></section><section class="closing"><span aria-hidden="true">${mark}</span><p>Always learning.<br><em>Always making.</em></p><a class="button" href="mailto:toshalimohapatra1@gmail.com">Say hello ↗</a></section>`;

await rm(out, { recursive: true, force: true });
await mkdir(path.join(out, 'assets'), { recursive: true });
for (const directory of ['art', 'img']) await cp(path.join('assets', directory), path.join(out, 'assets', directory), { recursive: true });
await mkdir(path.join(out, 'assets', 'fonts'), { recursive: true });
for (const file of ['inter-variable.ttf', 'inter-OFL.txt']) await cp(path.join('assets/fonts', file), path.join(out, 'assets/fonts', file));
await mkdir(path.join(out, 'assets', 'css'), { recursive: true });
await mkdir(path.join(out, 'assets', 'js'), { recursive: true });
for (const file of ['site.js', 'wait-study.js']) await cp(path.join('assets/js', file), path.join(out, 'assets/js', file));
await cp('assets/css/site.css', path.join(out, 'assets/css/site.css'));
await cp('node_modules/katex/dist/katex.min.css', path.join(out, 'assets/css/katex.min.css'));
await cp('node_modules/katex/dist/fonts', path.join(out, 'assets/css/fonts'), { recursive: true });
await cp('public/sw.min.js', path.join(out, 'sw.min.js'));
await writePage('/', page('Home', home));

if (!Array.isArray(foodPhotos)) throw new Error('Food photos must be an array.');
const foodGallery = (await Promise.all(foodPhotos.map(async (photo, index) => {
  if (!photo || typeof photo.file !== 'string' || !photo.file || path.basename(photo.file) !== photo.file || typeof photo.alt !== 'string' || !photo.alt.trim() || (photo.caption !== undefined && typeof photo.caption !== 'string') || (photo.preview !== undefined && (typeof photo.preview !== 'string' || !photo.preview || path.basename(photo.preview) !== photo.preview))) {
    throw new Error(`Invalid food photo entry at index ${index}.`);
  }
  const hasDimensions = photo.width !== undefined || photo.height !== undefined;
  if (hasDimensions && (!Number.isSafeInteger(photo.width) || photo.width <= 0 || !Number.isSafeInteger(photo.height) || photo.height <= 0)) {
    throw new Error(`Invalid food photo dimensions at index ${index}.`);
  }
  if (!(await stat(path.join('assets/img/food', photo.file))).isFile()) throw new Error(`Missing food photo: ${photo.file}`);
  const src = `/assets/img/food/${encodeURIComponent(photo.file)}`;
  if (photo.preview && !(await stat(path.join('assets/img/food/previews', photo.preview))).isFile()) throw new Error(`Missing food preview: ${photo.preview}`);
  const displaySrc = photo.preview ? `/assets/img/food/previews/${encodeURIComponent(photo.preview)}` : src;
  const dimensions = hasDimensions ? ` width="${photo.width}" height="${photo.height}"` : '';
  return `<figure class="food-photo"><a href="${src}" aria-label="View full-size photo: ${esc(photo.alt)}"><img src="${displaySrc}" alt="${esc(photo.alt)}"${dimensions} decoding="async"${index > 2 ? ' loading="lazy"' : ''}></a>${photo.caption ? `<figcaption>${esc(photo.caption)}</figcaption>` : ''}</figure>`;
}))).join('');
const foodBody = `<section class="inner-page food-page"><header class="page-heading"><p class="kicker">PHOTO COLLECTION</p><h1>Food</h1><p>Meals, drinks, and sweet things.</p></header>${foodPhotos.length ? `<div class="food-gallery" role="group" aria-label="Food photographs">${foodGallery}</div>` : '<p class="food-empty">No photos to show yet.</p>'}</section>`;
await writePage('/food/', page('Food', foodBody, { url: '/food/', description: 'Food photographs from Toshali Mohapatra.' }));

const browseNav = `<nav class="browse-nav" aria-label="Browse writing"><a href="/writing/">Latest</a><a href="/archives/">Archive</a><a href="/categories/">Categories</a><a href="/tags/">Tags</a><a href="/search/">Search ↗</a></nav>`;
function writingPage(title, entries, intro = '') {
  return `<section class="inner-page">${browseNav}<header class="page-heading"><p class="kicker">WORDS &amp; IDEAS</p><h1>${esc(title)}</h1>${intro ? `<p>${esc(intro)}</p>` : ''}</header><div class="post-list">${entries.map(postRow).join('')}</div></section>`;
}
await writePage('/writing/', page('Writing', writingPage('Writing', posts.slice(0, 10), 'Notes on machines, people, and everything in between.') + (posts.length > 10 ? '<div class="pagination"><a href="/page2/">Older writing →</a></div>' : ''), { url: '/writing/' }));
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
  const firstHeading = article.root().children().first();
  const normalizeTitle = text => text.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/\s+/g, ' ').trim();
  const titleInContent = firstHeading.is('h1') && normalizeTitle(firstHeading.text()) === normalizeTitle(post.title);
  const headings = article('h2[id],h3[id]').map((index, element) => {
    const heading = article(element).clone();
    heading.find('a.anchor').remove();
    return `<li class="${element.tagName === 'h3' ? 'subheading' : 'heading'}"><a href="#${encodeURIComponent(article(element).attr('id'))}">${esc(heading.text().trim())}</a></li>`;
  }).get().join('');
  const contents = headings ? `<details class="article-toc"><summary>On this page</summary><ol>${headings}</ol></details>` : '';
  const articleHtml = titleInContent ? post.html.replace(/^(\s*<h1\b[^>]*>[\s\S]*?<\/h1>)/i, (match, heading) => heading + contents) : post.html;
  const body = `<article class="article-page"${post.original ? ' data-preserved="true"' : ''}${titleInContent ? ' data-title-in-content="true"' : ''}><a class="back-link" href="/writing/">← All writing</a><header class="article-header"><p class="kicker">${dateText(post.date)} / WRITING</p>${titleInContent ? '' : `<h1>${esc(post.title)}</h1>`}<div class="article-topics">${categories}</div></header>${titleInContent ? '' : contents}<div class="article-content">${articleHtml}</div><footer class="article-footer"><p class="kicker">FOLLOW THE THREAD</p><div class="article-tags">${tags}</div><a class="text-link" href="/writing/">← More writing</a></footer></article>`;
  await writePage(`/posts/${post.slug}/`, page(post.title, body, { url: postUrl(post), description: post.source.replace(/[#*`\[\]]/g, '').replace(/\s+/g, ' ').trim().slice(0, 155) }));
}
const searchIndex = posts.map(post => ({ title: post.title, url: postUrl(post), date: dateText(post.date), text: post.source.replace(/\s+/g, ' ').slice(0, 10000), categories: post.categories, tags: post.tags }));
for (const note of fieldNotes) searchIndex.push({ title: note.title, url: `/notes/${note.slug}/`, date: 'Field note', text: note.paragraphs.join(' '), categories: ['Field notes'], tags: [] });
for (const project of portfolio.projects) searchIndex.push({ title: project.title, url: `/projects/${project.slug}/`, date: 'Project', text: project.description, categories: ['Projects'], tags: [] });
await writeFile(path.join(out, 'search-index.json'), JSON.stringify(searchIndex));
await writePage('/search/', page('Search', `<section class="inner-page">${browseNav}<header class="page-heading"><p class="kicker">FIND A THREAD</p><h1>Search the writing.</h1><label class="search-label" for="search-input">Search by title, topic, or word</label><input id="search-input" class="search-input" type="search" autocomplete="off" placeholder="Try ‘machine learning’"><p id="search-status" aria-live="polite">Start typing to find a note.</p></header><div id="search-results" class="post-list"></div><noscript><p>Search requires JavaScript. You can browse the <a href="/archives/">archive</a> instead.</p></noscript></section>`, { url: '/search/' }));

const study = `<article class="study-page"><a class="back-link" href="/studies/">← All studies</a><p class="kicker">STUDY 01 / TIME &amp; PERCEPTION</p><h1>What makes a wait feel long?</h1><p class="study-intro">The same four words take four seconds to arrive. Does seeing them sooner change the wait?</p><div class="experiment"><div class="experiment-head"><span>AN EXPERIMENT / 4 SECONDS</span><span>PRESET TRACE</span></div><p>This is a browser simulation, not a live model benchmark. Choose how the same scheduled output becomes visible.</p><div class="controls"><button type="button" data-mode="all">Show at once</button><button type="button" data-mode="ready">Show as ready</button><button type="button" data-stop disabled>Stop</button></div><p class="status" aria-live="polite" data-status>Choose a mode to begin.</p><div class="output" aria-label="Simulation output" aria-live="polite" data-output></div><div class="trace" data-trace aria-label="Timing marks"><span>1 s</span><span>2 s</span><span>3 s</span><span>4 s</span></div></div><section class="study-copy"><p class="kicker">THE COMPARISON / NOMINAL TIMINGS</p><h2>Same work. A different view of the wait.</h2><div class="comparison"><div><h3>Show at once</h3><p>First visible: <strong>4 s</strong><br>Final output: <strong>4 s</strong></p></div><div><h3>Show as ready</h3><p>First visible: <strong>1 s</strong><br>Final output: <strong>4 s</strong></p></div></div><p>The preset work schedule is one word each second. Only visibility changes: all four appear at the end in one mode, while each appears as ready in the other. In an actual service, time to first token and total time are separate metrics.</p><h2>Which wait felt longer?</h2><p>There is no answer being collected here. Your impression is a personal observation, not a research claim.</p><h3>Method</h3><p>Four preset words are scheduled at one-second intervals. The marks above represent each visible word; this is a simulation, not an inference trace.</p></section></article>`;
await writePage('/studies/', page('Studies', `<section class="inner-page"><header class="page-heading"><p class="kicker">SMALL EXPERIMENTS</p><h1>Questions, seen<br><em>from a few angles.</em></h1><p>Small experiments for thinking through how things work.</p></header><a class="study-card" href="/studies/what-makes-a-wait-feel-long/"><div><p class="kicker">STUDY 01 / TIME</p><h2>What makes a wait feel long?</h2><p>Four words, two ways of seeing them arrive.</p><span class="text-link">Open the study ↗</span></div>${field()}</a></section>`, { url: '/studies/' }));
await writePage('/studies/what-makes-a-wait-feel-long/', page('What makes a wait feel long?', study, { url: '/studies/what-makes-a-wait-feel-long/', script: '<script type="module" src="/assets/js/wait-study.js"></script>' }));
for (const note of fieldNotes) {
  const body = `<article class="article-page"><a class="back-link" href="/#projects">← Projects &amp; notes</a><header class="article-header"><p class="kicker">FROM THE FRUGAL SCIENCE LAB NOTEBOOK</p><h1>${esc(note.title)}</h1><p>${esc(note.intro)}</p></header><div class="article-content">${note.paragraphs.map(paragraph => `<p>${esc(paragraph)}</p>`).join('')}</div></article>`;
  await writePage(`/notes/${note.slug}/`, page(note.title, body, { url: `/notes/${note.slug}/` }));
}
for (const project of portfolio.projects) {
  const body = `<article class="article-page"><a class="back-link" href="/#projects">← Projects &amp; notes</a><header class="article-header"><p class="kicker">CODE / PLAY / REPEAT</p><h1>${esc(project.title)}</h1></header><div class="article-content"><p>${esc(project.description)}</p><p><a href="${esc(project.url)}">View source on GitHub ↗</a></p></div></article>`;
  await writePage(`/projects/${project.slug}/`, page(project.title, body, { url: `/projects/${project.slug}/` }));
}
await writePage('/about/', '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/"><link rel="canonical" href="https://wildchaser1703.github.io/"><title>Moved</title></head><body><p>This page has moved to the <a href="/">homepage</a>.</p></body></html>');

const feedEntries = posts.map(post => `<entry><title>${esc(post.title)}</title><id>https://wildchaser1703.github.io${postUrl(post)}</id><link href="https://wildchaser1703.github.io${postUrl(post)}"/><updated>${post.date.toISOString()}</updated><content type="html">${esc(post.html)}</content></entry>`).join('');
await writeFile(path.join(out, 'feed.xml'), `<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Toshali Mohapatra</title><id>https://wildchaser1703.github.io/</id><link href="https://wildchaser1703.github.io/feed.xml" rel="self"/><updated>${(posts[0]?.date || new Date(0)).toISOString()}</updated>${feedEntries}</feed>`);
const paths = ['/', '/food/', '/writing/', '/archives/', '/categories/', '/tags/', '/search/', '/studies/', '/studies/what-makes-a-wait-feel-long/', ...portfolio.projects.map(project => `/projects/${project.slug}/`), ...fieldNotes.map(note => `/notes/${note.slug}/`), ...posts.map(postUrl), ...[...taxonomy.categories.keys()].map(slug => `/categories/${slug}/`), ...[...taxonomy.tags.keys()].map(slug => `/tags/${slug}/`)];
await writeFile(path.join(out, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${paths.map(url => `<url><loc>https://wildchaser1703.github.io${esc(url)}</loc></url>`).join('')}</urlset>`);
await writeFile(path.join(out, 'robots.txt'), 'User-agent: *\nAllow: /\nSitemap: https://wildchaser1703.github.io/sitemap.xml\n');
await writeFile(path.join(out, '404.html'), page('Not found', `<section class="inner-page"><p class="kicker">404 / LOST IN THE MARGINS</p><h1>That page isn’t here.</h1><a class="button" href="/">Back to the start ↗</a></section>`, { url: '/404.html' }));
await writeFile(path.join(out, '.nojekyll'), '');
console.log(`Built ${posts.length} posts, ${taxonomy.categories.size} categories, and ${taxonomy.tags.size} tags.`);
