// Shared presentation for the portfolio. Published article bodies are rendered separately.
export const escape = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const e = escape;
export const arrow = '<svg class="arrow-icon" aria-hidden="true" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 19 19 5M5 5h14v14"/></svg>';
const artUrl = item => `/assets/art/${encodeURIComponent(item.file)}`;
const previewUrl = item => `/assets/img/project-${item.slug}.svg`;
export const projectUrl = item => item.route || `/projects/${item.slug}/`;

export function navigation(mark) {
  return `<a class="skip-link" href="#main">Skip to content</a><header class="site-header"><a class="brand" href="/" aria-label="Toshali Mohapatra, home"><span class="brand-mark" aria-hidden="true">${mark}</span><span>Toshali Mohapatra</span></a><button type="button" class="menu-toggle" aria-expanded="false" aria-controls="main-navigation" hidden>Menu <span aria-hidden="true">+</span></button><nav id="main-navigation" aria-label="Main navigation">${[['Projects','/projects/'],['Writing','/writing/'],['Art','/art/'],['Food','/food/'],['About','/#about']].map(([name,url]) => `<a href="${url}">${name}</a>`).join('')}<a class="search-link" href="/search/" aria-label="Search the site">Search ${arrow}</a></nav></header>`;
}

export function siteFooter() {
  return `<footer class="site-footer"><div class="footer-contact"><p class="kicker">Get in touch</p><a href="mailto:toshalimohapatra1@gmail.com">Say hello ${arrow}</a><p>AI engineering, art, and things in between.</p></div><div class="footer-links"><a href="/projects/">Projects</a><a href="/writing/">Writing</a><a href="/art/">Art</a><a href="/food/">Food</a><a href="/#about">About</a><a href="https://github.com/wildchaser1703">GitHub ${arrow}</a><a href="https://x.com/sudoher">X ${arrow}</a><a href="https://linkedin.com/in/toshalimohapatra">LinkedIn ${arrow}</a><a href="/feed.xml">RSS</a><a href="/studies/">Studies</a></div><div class="footer-bottom"><span>Toshali Mohapatra</span><small>© ${new Date().getUTCFullYear()}</small><a href="#main">Back to top ↑</a></div></footer>`;
}

export function artFigure(item, index = 0, group = 'art') {
  return `<figure class="art-item"><a class="art-image" data-gallery="${group}" href="${artUrl(item)}" aria-label="View ${e(item.title)}"><img src="${artUrl(item)}" width="${item.width}" height="${item.height}" alt="${e(item.title)}"${index > 0 ? ' loading="lazy"' : ''} decoding="async"></a><figcaption><strong>${e(item.title)}</strong><span>${e(item.caption)}</span></figcaption></figure>`;
}

export function foodFigure(photo, { index = 0, group = 'food', className = 'food-photo' } = {}) {
  const src = `/assets/img/food/${encodeURIComponent(photo.file)}`;
  const display = photo.preview ? `/assets/img/food/previews/${encodeURIComponent(photo.preview)}` : src;
  return `<figure class="${className}" style="--ratio:${photo.width / photo.height}"><a data-gallery="${group}" href="${src}" aria-label="View full-size photo: ${e(photo.alt)}"><img src="${display}" alt="${e(photo.alt)}" width="${photo.width}" height="${photo.height}"${index > 2 ? ' loading="lazy"' : ''} decoding="async"></a><figcaption>${e(photo.caption)}</figcaption></figure>`;
}

export function sculpture() {
  return `<figure class="sculpture" data-sculpture data-renderer="static" data-variant="0" data-paused="true" aria-label="Marginalia, an interactive procedural sculpture"><div class="sculpture-stage"><img class="sculpture-poster" src="/assets/img/sculpture-0.webp" alt="Five folded folios in ochre and vermilion, joined along a flowing spine." width="900" height="760" fetchpriority="high"><canvas class="sculpture-canvas" aria-hidden="true"></canvas></div><figcaption><details class="sculpture-info"><summary>Marginalia <span class="sr-only">— about this piece</span></summary><p>Named for notes in the margins, Marginalia brings my writing, art, and love of systems into five folded pages that move together.</p><span data-sculpture-variant>Seed 00000000</span></details><div class="sculpture-controls"><button type="button" data-sculpture-toggle hidden>Pause motion</button><button type="button" data-sculpture-next hidden>New variation ↗</button></div></figcaption><span class="sr-only" data-sculpture-state role="status"></span></figure>`;
}

function projectCard(project, index) {
  return `<article class="selected-project" aria-labelledby="project-${e(project.slug)}">
    <div class="selected-project-copy"><span class="kicker">${String(index + 1).padStart(2, '0')} / ${e(project.stage || project.kind)}</span><h3 id="project-${e(project.slug)}"><a href="${projectUrl(project)}">${e(project.title)}</a></h3><p>${e(project.homeSummary ?? project.summary)}</p>
    <p class="project-decision">${e(project.decisions[0].body)}</p>
    <p class="project-limit"><strong>Current scope</strong> ${e(project.limitations[0])}</p>
    <div class="project-links"><a class="text-link" href="${projectUrl(project)}">Read project notes ${arrow}</a><a class="text-link" href="${e(project.evidence[0].url)}">${e(project.evidence[0].label)} ${arrow}</a></div></div>
    <figure class="selected-project-image"><img src="${previewUrl(project)}" alt="${e(project.diagram.caption)}" width="960" height="640" loading="lazy"><figcaption>System sketch · <a href="${e(project.source)}">View repository ${arrow}</a></figcaption></figure>
  </article>`;
}

export function homePage({ engineering, art, food, writing, favorites = [] }) {
  const selectedArt = ['Day for The Hunt','Rain','Cat'].map(title => art.find(item => item.title === title)).filter(Boolean);
  const selectedFood = [40,29,69].map(i => food[i]).filter(Boolean);
  const selectedProjects = ['agentic-insight-analyst', 'depth-estimation'].map(slug => engineering.find(project => project.slug === slug)).filter(Boolean);
  return `<section class="home-hero"><div class="hero-copy"><h1>Toshali<br>Mohapatra<span class="title-period">.</span></h1><p class="hero-lede">Most of the work lives below the sentence.</p><div class="hero-actions"><a class="button" href="#projects">Explore my systems <svg class="arrow-icon" aria-hidden="true" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 4v16M5 13l7 7 7-7"/></svg></a></div></div>${sculpture()}</section>
  <section class="section-block selected-work" id="projects"><div class="section-heading"><h2>Selected projects</h2><a class="text-link" href="/projects/">All projects ${arrow}</a></div><div class="selected-projects">${selectedProjects.map(projectCard).join('')}</div></section>
  <section class="section-block writing-section" id="writing"><div class="writing-intro"><h2>Writing</h2><p>Notes on the decisions behind AI systems, and what happens when theory meets practice.</p><a class="text-link" href="/writing/">All writing ${arrow}</a></div><div class="post-list">${writing}</div></section>
  <section class="section-block personal-section" aria-labelledby="personal-heading"><div class="section-heading"><h2 id="personal-heading">Away from the keyboard.</h2><p>I also paint, write, and take far too many photos of food.</p></div><div class="personal-collections">
    <section class="home-art" id="art" data-collection="art" aria-labelledby="art-heading"><div class="collection-heading"><h3 id="art-heading">Art</h3><a class="text-link" href="/art/">The collection ${arrow}</a></div><div class="collection-controls" hidden><button type="button" data-collection-next>More paintings ↗</button><button type="button" data-collection-pause aria-pressed="false">Pause rotation</button><span class="sr-only" role="status"></span></div><div class="home-art-grid">${selectedArt.map((item,i) => artFigure(item,i+1,'home-art')).join('')}</div></section>
    <section class="home-food" id="food" data-collection="food" aria-labelledby="food-heading"><div class="collection-heading"><h3 id="food-heading">Food</h3><a class="text-link" href="/food/">All ${food.length} photographs ${arrow}</a></div><div class="collection-controls" hidden><button type="button" data-collection-next>More photographs ↗</button><button type="button" data-collection-pause aria-pressed="false">Pause rotation</button><span class="sr-only" role="status"></span></div><div class="home-food-grid">${selectedFood.map((photo,index) => foodFigure(photo,{index:index+3,group:'home-food'})).join('')}</div></section>
  </div></section>
  <section class="section-block home-about" id="about" aria-labelledby="about-heading"><div><h2 id="about-heading">A little more about me</h2><div class="about-socials"><a href="mailto:toshalimohapatra1@gmail.com">Email</a><a href="https://github.com/wildchaser1703">GitHub</a><a href="https://x.com/sudoher">X</a><a href="https://linkedin.com/in/toshalimohapatra">LinkedIn</a></div></div><div class="about-copy"><p>This site brings together the things I build, the ideas I write about, and the paintings and photographs I make along the way.</p><p>You’ll find engineering projects alongside small browser experiments, research notes, and quite a few meals.</p><div class="favorites-grid">${favorites.map(item=>`<div class="favorite-item"><span class="kicker">${e(item.kind)}</span><p>${e(item.title)}</p></div>`).join('')}</div></div></section>
  <script type="application/json" id="home-collections">${JSON.stringify({art,food}).replaceAll('<','\\u003c')}</script>`;
}

export function artPage(art) {
  return `<section class="inner-page gallery-page"><header class="page-heading"><p class="kicker">Off the screen / ${art.length} works</p><h1>Art</h1><p>Paintings, drawings, and a different way of paying attention.</p></header><div class="art-grid">${art.map((item,i) => artFigure(item,i)).join('')}</div></section>`;
}

export function foodPage(photos) {
  // These three photographs open with warm wood, bright colour and a broad table scene.
  const selections = [40,29,69].map(i => photos[i]).filter(Boolean);
  return `<section class="inner-page food-page"><header class="page-heading"><p class="kicker">At the table / ${photos.length} photographs</p><h1>Food</h1><p>Meals, drinks, and sweet things.</p></header><div class="food-opening">${selections.map((photo,index) => foodFigure(photo,{index,group:'food-selected',className:'food-selection'})).join('')}</div><div class="collection-heading"><h2>The whole collection</h2><span class="kicker">${photos.length} photographs</span></div><div class="food-gallery" role="group" aria-label="Food photographs">${photos.map((photo,index) => foodFigure(photo,{index:index+3})).join('')}</div></section>`;
}

export function directoryItems(engineering, portfolio, notes) {
  return [...engineering, ...portfolio.projects.map(project => ({...project,kind:'Experiments',summary:project.description,diagram:{caption:'Interaction diagram',nodes: project.slug === 'type-and-speak' ? ['Text','Voice settings','Speech'] : project.slug === 'louboutin-3d-card' ? ['Pointer','Transform','Product card'] : project.slug === 'morse-code' ? ['Text','Morse code','Tones'] : project.slug === 'counter' ? ['Input','State','Display'] : project.slug === 'flip-colour' ? ['Click','Colour','Canvas'] : ['Sign in','Transition','Sign up']}})), {slug:'wait-study',title:'What makes a wait feel long?',kind:'Experiments',summary:'A four-second browser simulation of two ways to reveal the same output.',route:'/studies/what-makes-a-wait-feel-long/',diagram:{caption:'Two output schedules with the same completion time',nodes:['Start','One word / second','4 seconds']}}, ...notes.map(note => ({...note,kind:'Field notes',summary:note.intro,route:`/notes/${note.slug}/`,diagram:{caption:'From the Frugal Science Lab Notebook',nodes:note.slug.includes('paper') ? ['Paper','Fibres','Structure'] : ['Geometry','Constraint','Motion']}}))];
}

export function directoryPage(items, archive = false) {
  const first = items[0];
  return `<section class="inner-page directory-page"><header class="page-heading"><p class="kicker">${archive ? 'Earlier work / Experiments' : 'Machine learning / LLM systems'}</p><h1>${archive ? 'Project archive' : 'Projects'}</h1><p>${archive ? 'Earlier experiments, learning projects, and field notes.' : 'Selected work in model training, multimodal learning, retrieval, and agents.'}</p></header><div class="directory-filters" role="group" aria-label="Filter projects" hidden>${['All', ...new Set(items.map(item => item.kind))].map((kind,i) => `<button type="button" data-filter="${kind}" aria-pressed="${i===0}">${kind}</button>`).join('')}<span data-filter-status role="status">${items.length} projects</span></div><div class="directory-layout"><div class="directory-list">${items.map((item,i) => `<a class="directory-row" href="${projectUrl(item)}" data-kind="${item.kind}" data-preview="${previewUrl(item)}" data-preview-title="${e(item.title)}" data-preview-caption="${e(item.diagram.caption)}"><span class="directory-number">${String(i+1).padStart(2,'0')}</span><div><h2>${e(item.title)}</h2><p>${e(item.summary)}</p><span class="directory-kind">${e(item.kind)}${item.stage ? ` · ${e(item.stage)}` : ''}</span></div><span class="directory-arrow" aria-hidden="true">↗</span><img class="directory-inline-preview" src="${previewUrl(item)}" alt="${e(item.diagram.caption)}" width="960" height="640" loading="lazy"></a>`).join('')}</div><figure class="directory-preview" aria-hidden="true"><img src="${previewUrl(first)}" alt="" width="960" height="640"><figcaption><span data-preview-title>${e(first.title)}</span><span data-preview-caption>${e(first.diagram.caption)}</span></figcaption></figure></div><p class="directory-study-link"><a class="text-link" href="${archive ? '/projects/' : '/projects/archive/'}">${archive ? 'Selected ML / AI projects' : 'Earlier projects and experiments'} ${arrow}</a></p><p class="directory-study-link"><a class="text-link" href="/studies/">Browse interactive studies ${arrow}</a></p></section>`;
}

export function engineeringPage(project) {
  return `<article class="inner-page case-study"><a class="back-link" href="/projects/">← All projects</a><header class="page-heading"><p class="kicker">${e(project.kind)} / ${e(project.stage || 'Project notes')}</p><h1>${e(project.title)}</h1><p>${e(project.summary)}</p><div class="project-stack">${project.stack.map(item=>`<span>${e(item)}</span>`).join('')}</div><a class="button" href="${e(project.source)}">View source on GitHub ${arrow}</a></header><figure class="case-diagram"><img src="${previewUrl(project)}" alt="${e(project.diagram.caption)}" width="960" height="640"><figcaption>${e(project.diagram.caption)}</figcaption></figure><div class="case-body"><section><h2>The problem</h2><p>${e(project.problem)}</p></section><section><h2>How it works</h2>${project.approach.map(text=>`<p>${e(text)}</p>`).join('')}</section><section><h2>Engineering decisions</h2>${project.decisions.map(item=>`<h3>${e(item.title)}</h3><p>${e(item.body)}</p>`).join('')}</section><section><h2>What’s implemented</h2><p>${e(project.result)}</p></section><section><h2>Scope and limitations</h2><ul>${project.limitations.map(item=>`<li>${e(item)}</li>`).join('')}</ul></section><section><h2>Explore the source</h2><ul class="source-links">${project.evidence.map(item=>`<li><a href="${e(item.url)}">${e(item.label)} ${arrow}</a></li>`).join('')}</ul></section></div></article>`;
}

export function diagramSvg(item) {
  // Arrows live only in the gutters; every label fits inside a padded node.
  const nodes = item.diagram.nodes;
  const positions = nodes.length > 3 ? [[64,180],[366,180],[668,180],[668,360],[366,360]] : [[64,270],[366,270],[668,270]];
  const labelLines = text => {
    const lines = [''];
    for (const word of text.split(' ')) {
      if ((lines.at(-1) + ' ' + word).trim().length > 17) lines.push(word);
      else lines[lines.length-1] = (lines.at(-1) + ' ' + word).trim();
    }
    return lines;
  };
  const boxes = nodes.map((text,i) => {
    const [x,y] = positions[i];
    const lines = labelLines(text);
    return `<g class="node"><rect x="${x}" y="${y}" width="228" height="92"/><text class="step" x="${x+16}" y="${y+23}">${String(i+1).padStart(2,'0')}</text><text class="label" x="${x+114}" y="${y+57-(lines.length-1)*11}" text-anchor="middle">${lines.map((line,j)=>`<tspan x="${x+114}" dy="${j?22:0}">${e(line)}</tspan>`).join('')}</text></g>`;
  }).join('');
  const links = nodes.slice(1).map((_,i) => {
    const [x,y] = positions[i], [nx,ny] = positions[i+1];
    const d = y !== ny ? `M${x+114} ${y+100}V${ny-10}` : nx > x ? `M${x+236} ${y+46}H${nx-10}` : `M${x-8} ${y+46}H${nx+238}`;
    return `<path class="connector" d="${d}" marker-end="url(#arrow)"/>`;
  }).join('');
  const loop = item.slug === 'agentic-insight-analyst' ? '<path class="connector" d="M782 460V520H328V316H480V282" marker-end="url(#arrow)"/><text class="note" x="64" y="552">Observe → plan again, until the stopping rule is met.</text>' : '';
  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640" role="img" aria-labelledby="title"><title id="title">${e(item.diagram.caption)}</title><style>
    :root{color-scheme:light dark;--bg:#f0ede5;--surface:#f8f6f0;--text:#272723;--muted:#67655d;--border:#bdb7a8;--accent:#a53224}
    @media(prefers-color-scheme:dark){:root{--bg:#252522;--surface:#30302b;--text:#f4f1e8;--muted:#bdb9ad;--border:#626157;--accent:#ffad7c}}
    .background{fill:var(--bg)}.node rect{fill:var(--surface);stroke:var(--border)}text{font-family:Arial,sans-serif;fill:var(--text)}.label{font-size:18px}.step,.note{font:13px monospace;fill:var(--muted)}.connector{fill:none;stroke:var(--accent);stroke-width:2}.arrow{fill:none;stroke:var(--accent);stroke-width:1.5}
  </style><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow" d="M1 1 9 5 1 9"/></marker></defs><rect class="background" width="960" height="640"/><text class="note" x="64" y="64">${e(item.kind.toUpperCase())} / SYSTEM SKETCH</text>${links}${loop}${boxes}<text class="note" x="64" y="606">${e(item.title)}</text></svg>`;
}
