// A small project registry: add future projects here without changing site navigation.
import { escape as e, arrow } from './design-components.mjs';

const projects = [
  { title: 'Indian Art Atlas', label: '01 / VISUAL ARTS & CRAFTS', description: 'An evolving collection of visual traditions, the people who make them, and the objects that carry their stories.', url: '/just-because/indian-art/' }
];

const shortlist = [
  ['Pattachitra', 'Odisha', 'Painting', '/just-because/indian-art/pattachitra/'],
  ['Mithila / Madhubani', 'Mithila, including Bihar and Nepal', 'Painting', ''],
  ['Warli', 'Western India', 'Painting', ''],
  ['Gond painting', 'Central India', 'Painting', ''],
  ['Kalamkari', 'Andhra Pradesh', 'Textile', ''],
  ['Dhokra', 'Eastern and central India', 'Metal', ''],
  ['Longpi pottery', 'Manipur', 'Clay and stone', '']
];

function puriFigure() {
  return `<figure class="atlas-featured-work"><img src="https://thumb.wikimedia.org/wikipedia/commons/thumb/8/85/Jagannatha%2C_Balabhadra%2C_and_Subhadra_in_the_Jagganatha_Temple_LACMA_M.82.107.jpg/960px-Jagannatha%2C_Balabhadra%2C_and_Subhadra_in_the_Jagganatha_Temple_LACMA_M.82.107.jpg?utm_campaign=index&amp;utm_content=thumbnail&amp;utm_source=commons.wikimedia.org" alt="A colourful painting depicting the Jagannath temple in Puri and its surrounding shrines" width="2100" height="1639" loading="lazy" decoding="async"><figcaption><span>Unknown artist, <em>Jagannatha, Balabhadra, and Subhadra in the Jagannatha Temple</em>, c. 1875–1900. Opaque watercolor on cloth. Indian Art Special Purpose Fund, LACMA M.82.107.</span> <a href="https://collections.lacma.org/object/39937">Museum record ↗</a> · <a href="https://commons.wikimedia.org/wiki/File:Jagannatha,_Balabhadra,_and_Subhadra_in_the_Jagganatha_Temple_LACMA_M.82.107.jpg">Public-domain image via Wikimedia Commons ↗</a><br><span>LACMA describes this as a Puri pilgrimage painting; it is included as regional context.</span></figcaption></figure>`;
}

export function justBecausePage() {
  return `<section class="curiosity-page">
    <header class="curiosity-hero"><p class="kicker">A PERSONAL COLLECTION / ALWAYS IN PROGRESS</p><h1>Just<br><em>because.</em></h1><p>Little corners of the internet, made from curiosity. A place to collect the things I want to learn more about.</p><a class="text-link" href="#projects">Explore the collection ↓</a></header>
    <div class="curiosity-heading" id="projects"><p class="kicker">THE COLLECTION</p><h2>One thing leads<br>to another.</h2><p>Independent projects, gathered in one place.</p></div>
    <div class="curiosity-projects">${projects.map(project => `<article class="curiosity-card"><p class="kicker">${e(project.label)}</p><h3><a href="${e(project.url)}">${e(project.title)}</a></h3><p>${e(project.description)}</p><a class="text-link" href="${e(project.url)}">Explore the atlas ${arrow}</a></article>`).join('')}</div>
  </section>`;
}

export function atlasPage() {
  return `<section class="curiosity-page atlas-page"><a class="back-link" href="/just-because/">← Just Because</a>
    <header class="atlas-hero"><div><p class="kicker">AN EVOLVING INDEX / VISUAL ARTS & CRAFTS</p><h1>Indian<br><em>Art Atlas.</em></h1></div><div><p>A place to follow the materials, stories, places and people behind visual art and craft traditions across India.</p><p class="atlas-note">The first field note is here. The rest of the collection is a research shortlist, still being verified.</p></div></header>
    ${puriFigure()}
    <div class="atlas-section-head"><p class="kicker">01 / START EXPLORING</p><h2>Traditions & materials</h2><p>Many traditions cross borders and evolve through the work of individual makers. These places are starting points, not fixed boundaries.</p></div>
    <div class="atlas-list">${shortlist.map(([name,region,medium,url],i)=>`<article class="atlas-row"><span class="atlas-number">${String(i+1).padStart(2,'0')}</span><div><h3>${url ? `<a href="${url}">${e(name)}</a>` : e(name)}</h3><p>${e(region)}</p></div><span class="atlas-medium">${e(medium)}</span>${url ? `<a class="atlas-link" href="${url}" aria-label="Read about ${e(name)}">Explore ↗</a>` : '<span class="atlas-status">Research shortlist</span>'}</article>`).join('')}</div>
    <aside class="atlas-method"><p class="kicker">HOW THIS IS MADE</p><h2>People before patterns.</h2><p>This is a growing research project. Entries link back to their sources, name makers where known, and distinguish a tradition from an individual work. Dates, meanings and regional names can differ across sources.</p><p>Images are added after their reuse terms have been checked. Source links do not mean every image on those sites is available for reuse.</p></aside>
  </section>`;
}

export function pattachitraPage() {
  return `<article class="curiosity-page tradition-page"><a class="back-link" href="/just-because/indian-art/">← Indian Art Atlas</a>
    <header class="tradition-hero"><p class="kicker">FIELD NOTE 01 / ODISHA / PAINTING</p><h1>Patta<span>chitra.</span></h1><p class="tradition-lede">A painting tradition with stories in its borders, lines and colours.</p><p class="atlas-note">A starting point, not a complete account. Practices and materials vary across artists and places.</p></header>
    ${puriFigure()}
    <div class="tradition-grid"><aside><p class="kicker">THE SHORT VERSION</p><dl><dt>Region</dt><dd>Odisha, India</dd><dt>Form</dt><dd>Painting</dd><dt>Materials</dt><dd>Cloth and pigments, among other surfaces</dd></dl></aside>
    <div class="tradition-body"><section><p class="kicker">01 / THE TRADITION</p><h2>Stories on a prepared surface.</h2><p>In Odisha, Pattachitra encompasses painted works on cloth and other surfaces. The Government of Odisha describes subjects ranging from religious and mythological narratives to folk themes, including stories associated with Jagannath and Krishna.</p><p>Traditions travel and change: artists also work on objects such as wooden boxes, toys and coconut shells. An individual work’s medium and story should be read from its own record.</p></section>
    <section><p class="kicker">02 / A WAY OF MAKING</p><h2>Cloth, colour, line.</h2><p>One traditional cloth preparation described by the Government of Odisha uses chalk and glue made from tamarind seeds. The border is made first; colours and fine brush lines follow. This is one documented process, not a recipe every artist must follow.</p><p>Its account describes pigments made from vegetable, earth and mineral sources. Materials and working methods vary across makers and over time.</p></section>
    <section><p class="kicker">03 / LOOK CLOSER</p><h2>One work, one record.</h2><p>The National Gallery of Victoria records <em>Pattachitra with Kali defeating Shiva</em>, a work made in Odisha in the mid-20th century using coloured pigments on paper. The gallery credits it as a gift of Jim Masselos, 2026. The individual maker is not named in the record.</p><p><a class="text-link" href="https://www.ngv.vic.gov.au/explore/collection/work/155146/">View the gallery record ↗</a></p><p class="atlas-note">The gallery labels this work public domain but specifies credit and reproduction conditions. The image is not reproduced here.</p></section>
    <section><p class="kicker">04 / SOURCES & FURTHER READING</p><h2>Follow the thread.</h2><ol class="source-list"><li><a href="https://odisha.gov.in/en/odisha-tourism/patta-chitra">Government of Odisha: Patta Chitra</a> — process, materials and themes.</li><li><a href="https://www.ngv.vic.gov.au/explore/collection/work/155146/">National Gallery of Victoria: Pattachitra with Kali defeating Shiva</a> — object record and reproduction conditions.</li><li><a href="https://odishacraftsmuseum.odisha.gov.in/getArtifact?galleryId=2">Odisha Crafts Museum: Traditional Painting Gallery</a> — context on painting and other media.</li></ol><p class="atlas-note">This note will grow as more maker-led and community sources are reviewed.</p></section></div></div>
  </article>`;
}
