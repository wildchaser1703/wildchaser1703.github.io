// Shared presentation for the portfolio. Published article bodies are rendered separately.
export const escape = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const e = escape;
export const arrow = '<svg class="arrow-icon" aria-hidden="true" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 19 19 5M5 5h14v14"/></svg>';
const artUrl = item => `/assets/art/${encodeURIComponent(item.file)}`;
const previewUrl = item => `/assets/img/project-${item.slug}.svg`;
export const projectUrl = item => item.route || `/projects/${item.slug}/`;

export function navigation(mark) {
  return `<a class="skip-link" href="#main">Skip to content</a><header class="site-header"><a class="brand" href="/" aria-label="Toshali Mohapatra, home"><span class="brand-mark" aria-hidden="true">${mark}</span><span>Toshali Mohapatra</span></a><button type="button" class="menu-toggle" aria-expanded="false" aria-controls="main-navigation" hidden>Menu <span aria-hidden="true">+</span></button><nav id="main-navigation" aria-label="Main navigation">${[['Projects','/projects/'],['Writing','/writing/'],['Art','/art/'],['Food','/food/'],['About','/about/']].map(([name,url]) => `<a href="${url}">${name}</a>`).join('')}<a class="search-link" href="/search/" aria-label="Search the site">Search ${arrow}</a></nav></header>`;
}

export function siteFooter() {
  return `<footer class="site-footer"><div class="footer-contact"><p class="kicker">Get in touch</p><a href="mailto:toshalimohapatra1@gmail.com">Say hello ${arrow}</a><p>AI engineering, art, and things in between.</p></div><div class="footer-links"><a href="/projects/">Projects</a><a href="/writing/">Writing</a><a href="/art/">Art</a><a href="/food/">Food</a><a href="/about/">About</a><a href="https://github.com/wildchaser1703">GitHub ${arrow}</a><a href="/feed.xml">RSS</a><a href="/studies/">Studies</a></div><div class="footer-bottom"><span>Toshali Mohapatra</span><small>© ${new Date().getUTCFullYear()}</small><a href="#main">Back to top ↑</a></div></footer>`;
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
  return `<figure class="sculpture" data-sculpture data-renderer="static" data-variant="0" data-paused="true" aria-label="Woven field, an interactive procedural sculpture"><div class="sculpture-stage"><img class="sculpture-poster" src="/assets/img/sculpture-0.webp" alt="Fine vermilion, orange and gold strands weave into an open, folded sculpture." width="900" height="760" fetchpriority="high"><canvas class="sculpture-canvas" aria-hidden="true"></canvas></div><figcaption><div><span class="kicker">Woven field</span><span class="sculpture-description">A study in connection and constraint.</span></div><div class="sculpture-controls"><span data-sculpture-variant>01 / 03</span><button type="button" data-sculpture-toggle hidden>Pause motion</button><button type="button" data-sculpture-next hidden>New variation ↗</button></div></figcaption><span class="sr-only" data-sculpture-state role="status"></span></figure>`;
}

function projectCard(project, index) {
  return `<a class="selected-project" href="${projectUrl(project)}"><div class="selected-project-image"><img src="${previewUrl(project)}" alt="${e(project.diagram.caption)}" width="960" height="640" loading="lazy"></div><div class="selected-project-meta"><span class="kicker">${String(index + 1).padStart(2, '0')} / ${e(project.kind)}</span>${arrow}</div><h3>${e(project.title)}</h3><p>${e(project.summary)}</p></a>`;
}

export function homePage({ engineering, art, food, writing }) {
  const selectedArt = ['Day for The Hunt','Rain','Cat'].map(title => art.find(item => item.title === title)).filter(Boolean);
  const selectedFood = [40,29,69].map(i => food[i]).filter(Boolean);
  return `<section class="home-hero"><div class="hero-copy"><p class="kicker"><span class="status-dot" aria-hidden="true"></span> AI Engineer</p><h1>Toshali<br>Mohapatra<span class="title-period">.</span></h1><p class="hero-lede">I build AI systems.<br>I also paint, write, and take far too many photos of food.</p><div class="hero-actions"><a class="button" href="/projects/">View projects ${arrow}</a><a class="text-link" href="/writing/">Read writing ${arrow}</a></div><p class="hero-footnote">Retrieval · Agents · Machine learning</p></div>${sculpture()}</section>
  <section class="section-block selected-work" id="projects"><div class="section-heading"><div><p class="kicker">01 / Engineering</p><h2>Selected projects</h2></div><a class="text-link" href="/projects/">The full directory ${arrow}</a></div><div class="selected-projects">${engineering.map(projectCard).join('')}</div></section>
  <section class="section-block writing-section" id="writing"><div class="writing-intro"><p class="kicker">02 / Writing</p><h2>Thinking<br>out loud.</h2><p>Notes on the decisions behind AI systems, and what happens when theory meets practice.</p><a class="text-link" href="/writing/">All writing ${arrow}</a></div><div class="post-list">${writing}</div></section>
  <section class="section-block home-art" id="art"><div class="section-heading"><div><p class="kicker">03 / Art</p><h2>Paint on my hands.</h2></div><a class="text-link" href="/art/">The art collection ${arrow}</a></div><div class="home-art-grid">${selectedArt.map((item,i) => artFigure(item,i+1,'home-art')).join('')}</div></section>
  <section class="section-block home-food"><div class="section-heading"><div><p class="kicker">04 / Food</p><h2>Usually, before the first bite.</h2></div><a class="text-link" href="/food/">All ${food.length} photographs ${arrow}</a></div><div class="home-food-grid">${selectedFood.map((photo,index) => foodFigure(photo,{index:index+3,group:'home-food'})).join('')}</div></section>`;
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

export function directoryPage(items) {
  const first = items[0];
  return `<section class="inner-page directory-page"><header class="page-heading"><p class="kicker">Engineering / Experiments / Notes</p><h1>Projects</h1><p>AI systems, browser experiments, and ideas worked out by making things.</p></header><div class="directory-filters" role="group" aria-label="Filter projects" hidden>${['All','AI engineering','Experiments','Field notes'].map((kind,i) => `<button type="button" data-filter="${kind}" aria-pressed="${i===0}">${kind}</button>`).join('')}<span data-filter-status role="status">${items.length} projects</span></div><div class="directory-layout"><div class="directory-list">${items.map((item,i) => `<a class="directory-row" href="${projectUrl(item)}" data-kind="${item.kind}" data-preview="${previewUrl(item)}" data-preview-title="${e(item.title)}" data-preview-caption="${e(item.diagram.caption)}"><span class="directory-number">${String(i+1).padStart(2,'0')}</span><div><h2>${e(item.title)}</h2><p>${e(item.summary)}</p><span class="directory-kind">${e(item.kind)}</span></div><span class="directory-arrow" aria-hidden="true">↗</span><img class="directory-inline-preview" src="${previewUrl(item)}" alt="${e(item.diagram.caption)}" width="960" height="640" loading="lazy"></a>`).join('')}</div><figure class="directory-preview" aria-hidden="true"><img src="${previewUrl(first)}" alt="" width="960" height="640"><figcaption><span data-preview-title>${e(first.title)}</span><span data-preview-caption>${e(first.diagram.caption)}</span></figcaption></figure></div><p class="directory-study-link"><a class="text-link" href="/studies/">Browse interactive studies ${arrow}</a></p></section>`;
}

export function engineeringPage(project) {
  return `<article class="inner-page case-study"><a class="back-link" href="/projects/">← All projects</a><header class="page-heading"><p class="kicker">AI engineering / Project notes</p><h1>${e(project.title)}</h1><p>${e(project.summary)}</p><div class="project-stack">${project.stack.map(item=>`<span>${e(item)}</span>`).join('')}</div><a class="button" href="${e(project.source)}">View source on GitHub ${arrow}</a></header><figure class="case-diagram"><img src="${previewUrl(project)}" alt="${e(project.diagram.caption)}" width="960" height="640"><figcaption>${e(project.diagram.caption)}</figcaption></figure><div class="case-body"><section><h2>The problem</h2><p>${e(project.problem)}</p></section><section><h2>How it works</h2>${project.approach.map(text=>`<p>${e(text)}</p>`).join('')}</section><section><h2>Engineering decisions</h2>${project.decisions.map(item=>`<h3>${e(item.title)}</h3><p>${e(item.body)}</p>`).join('')}</section><section><h2>What’s implemented</h2><p>${e(project.result)}</p></section><section><h2>Scope and limitations</h2><ul>${project.limitations.map(item=>`<li>${e(item)}</li>`).join('')}</ul></section><section><h2>Explore the source</h2><ul class="source-links">${project.evidence.map(item=>`<li><a href="${e(item.url)}">${e(item.label)} ${arrow}</a></li>`).join('')}</ul></section></div></article>`;
}

export function aboutPage(portfolio) {
  return `<section class="inner-page about-page"><header class="page-heading"><p class="kicker">A little about me</p><h1>Engineer.<br>Also, a person.</h1></header><div class="about-layout"><div class="about-copy"><p class="about-lede">I’m Toshali, an AI engineer. I work on retrieval, agents, and machine learning systems.</p><p>This site brings together the things I build, the ideas I write about, and the paintings and photographs I make along the way.</p><p>You’ll find engineering projects alongside small browser experiments, research notes, and quite a few meals.</p><div class="about-links"><a class="button" href="mailto:toshalimohapatra1@gmail.com">Say hello ${arrow}</a><a class="text-link" href="https://github.com/wildchaser1703">GitHub ${arrow}</a></div><div class="favorites-grid">${portfolio.favorites.map(item=>`<div class="favorite-item"><span class="kicker">${e(item.kind)}</span><p>${e(item.title)}</p></div>`).join('')}</div></div>${artFigure(portfolio.art.find(item=>item.title==='Evening Still') || portfolio.art[0],0,'about-art')}</div></section>`;
}

export function diagramSvg(item) {
  const start = `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640"><defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#d7d1c5"/></pattern><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1 9 5 1 9" fill="none" stroke="#a53224" stroke-width="1.5"/></marker></defs><rect width="960" height="640" fill="#f0ede5"/><rect width="960" height="640" fill="url(#grid)"/><text x="40" y="46" fill="#6c665b" font-family="monospace" font-size="13" letter-spacing="1">${e(item.kind.toUpperCase())} / SYSTEM SKETCH</text>`;
  const label = (x,y,text,size=18,color='#272723') => `<text x="${x}" y="${y}" text-anchor="middle" fill="${color}" font-family="Arial,sans-serif" font-size="${size}">${e(text)}</text>`;
  const box = (x,y,w,h,text,fill='#f8f6f0') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="#bdb7a8"/>${label(x+w/2,y+h/2+6,text)}`;
  const link = d => `<path d="${d}" fill="none" stroke="#a53224" stroke-width="1.8" marker-end="url(#arrow)"/>`;
  let drawing = '';
  if (item.slug === 'production-rag-system') {
    drawing = `${label(145,154,'01 / INGEST',13,'#67655d')}${label(448,154,'02 / RETRIEVE',13,'#67655d')}${label(784,154,'03 / GENERATE',13,'#67655d')}`;
    drawing += [0,1,2].map(i=>`<g transform="translate(${74+i*20},${212+i*24})"><rect width="104" height="126" fill="#f8f6f0" stroke="#a99e89"/><path d="M18 30h68m-68 16h54m-54 16h68m-68 16h45" stroke="#b9b2a3" stroke-width="2"/></g>`).join('');
    drawing += label(145,445,'Web documents') + link('M232 310H310');
    drawing += `<g fill="#e2a22d">${Array.from({length:9},(_,i)=>`<rect x="${333+(i%3)*35}" y="${255+Math.floor(i/3)*35}" width="25" height="25" opacity="${.4+(i%3)*.2}"/>`).join('')}</g>`;
    drawing += label(376,392,'Chunks') + link('M438 310H477') + box(493,268,127,84,'Index','#f4d58f') + label(556,392,'ChromaDB');
    drawing += link('M620 310H695') + box(711,260,164,100,'Answer','#e9b3a1');
    drawing += box(493,450,127,50,'Question') + link('M556 450V356') + label(783,392,'Retrieved context',16);
    drawing += `<path d="M302 190V514M660 190V514" stroke="#d0c9bb" stroke-dasharray="3 7"/>`;
  } else if (item.slug === 'agentic-insight-analyst') {
    drawing = box(70,283,158,64,'Goal + CSV') + link('M228 315H315');
    drawing += `<circle cx="505" cy="320" r="150" fill="none" stroke="#d6c5a6" stroke-width="28"/><path d="M505 170A150 150 0 0 1 655 320M505 470A150 150 0 0 1 355 320" fill="none" stroke="#a53224" stroke-width="2" marker-end="url(#arrow)"/>`;
    drawing += box(430,145,150,54,'Plan','#f4d58f') + box(570,293,170,54,'Select tool','#e9b3a1') + box(420,443,170,54,'Evaluate','#f4d58f') + box(285,293,150,54,'Observe');
    drawing += label(505,309,'WORKING',14,'#67655d') + label(505,334,'MEMORY',14,'#67655d');
    drawing += link('M740 320H784') + box(799,288,126,64,'Report') + label(807,400,'Stop rule met',15,'#67655d');
  } else if (item.slug === 'retail-demand-forecasting-system') {
    drawing = box(45,280,160,70,'Sales records') + link('M205 315H250') + box(266,280,162,70,'Validate + split');
    drawing += link('M428 315H467V211H506') + link('M467 315V419H506');
    drawing += box(522,180,180,62,'Baselines','#f4d58f') + box(522,388,180,62,'Tree models','#e9b3a1');
    drawing += label(612,150,'Reference forecasts',15,'#67655d') + label(612,486,'Temporal features',15,'#67655d');
    drawing += link('M702 211H748V315H782') + link('M702 419H748V315');
    drawing += box(798,268,128,94,'Compare') + label(480,556,'TRAIN  /  VALIDATE  /  TEST',14,'#67655d');
  } else {
    const nodes = item.diagram.nodes;
    const spacing = 720 / Math.max(nodes.length-1,1);
    drawing = nodes.map((text,i)=> {
      const x = 120+i*spacing;
      return `${i ? link(`M${x-spacing+65} 320H${x-70}`) : ''}<circle cx="${x}" cy="320" r="55" fill="${i===nodes.length-1?'#e9b3a1':'#f4d58f'}" stroke="#bcb09a"/>${label(x,326,String(i+1).padStart(2,'0'),24)}${label(x,428,text,20)}`;
    }).join('');
  }
  return `${start}${drawing}<text x="40" y="606" fill="#6c665b" font-family="monospace" font-size="13">${e(item.title)}</text></svg>`;
}
