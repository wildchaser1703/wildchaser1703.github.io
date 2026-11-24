import {variationDistance} from '../assets/js/sculpture-shape.js';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, readFile } from 'node:fs/promises';
import net from 'node:net';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';

const foodPhotos = JSON.parse(await readFile('data/food.json', 'utf8'));
const portfolio = JSON.parse(await readFile('data/portfolio.json', 'utf8'));
const engineering = JSON.parse(await readFile('data/engineering-projects.json', 'utf8'));
const legacy = JSON.parse(await readFile('data/legacy-posts.json', 'utf8'));
const screenshotDir = process.env.SCREENSHOT_DIR ? path.resolve(process.env.SCREENSHOT_DIR) : null;
if (screenshotDir) await mkdir(screenshotDir, { recursive: true });
const probe = net.createServer();
await new Promise((resolve, reject) => probe.once('error', reject).listen(0, '127.0.0.1', resolve));
const port = probe.address().port;
await new Promise(resolve => probe.close(resolve));
const base = `http://127.0.0.1:${port}`;
const server = spawn(process.execPath, ['scripts/serve.mjs'], { env: { ...process.env, PORT: String(port) }, stdio: 'inherit' });
let serverError;
server.on('error', error => { serverError = error; });
let browser;
const errors = [];
const threeRequests = [];

// Retry observable state, avoiding arbitrary animation and network-idle delays.
async function eventually(check, message, timeout = 10000) {
  const deadline = Date.now() + timeout;
  let lastError;
  do {
    try { await check(); return; } catch (error) { lastError = error; }
    await delay(50);
  } while (Date.now() < deadline);
  throw new Error(message, { cause: lastError });
}
const attribute = (locator, name, value) => eventually(async () => assert.equal(await locator.getAttribute(name), value), `Expected ${name}=${value}`);
const textIs = (locator, value) => eventually(async () => assert.equal(await locator.textContent(), value), `Expected text: ${value}`);
const focused = locator => eventually(async () => assert.equal(await locator.evaluate(element => element === document.activeElement), true), 'Focus was not restored');
function observe(page, label, reduced = false) {
  page.setDefaultTimeout(10000);
  page.on('pageerror', error => errors.push(`${label}: ${error.message}`));
  if (reduced) page.on('request', request => {
    if (/\/three[^/]*\.js(?:\?|$)/i.test(request.url())) threeRequests.push(`${label}: ${request.url()}`);
  });
}
async function visit(page, route, scriptsEnabled = true) {
  const response = await page.goto(base + route, { waitUntil: 'domcontentloaded' });
  assert.equal(response.status(), 200, `${route} status`);
  if (route === '/' && scriptsEnabled) await page.locator('[data-collection="art"][data-collection-ready="true"]').waitFor();
  if (route === '/' && scriptsEnabled) await page.locator('[data-collection="food"][data-collection-ready="true"]').waitFor();
}
async function decode(image) {
  await image.evaluate(async element => {
    element.loading = 'eager';
    await element.decode();
    if (!element.complete || !element.naturalWidth) throw new Error(`Image failed: ${element.src}`);
  });
}
async function screenshot(page, name, fullPage = true) {
  if (!screenshotDir) return;
  await page.evaluate(() => document.fonts.ready);
  for (const img of await page.locator('main img').all()) {
    if (!await img.isVisible()) continue;
    const sourceIsLocal = await img.evaluate(element => new URL(element.src).origin === location.origin);
    // Atlas art is externally hosted and lazy-loaded. Scroll and decode it for
    // both desktop and mobile captures so the screenshots show real content.
    const isAtlasArt = await img.evaluate(element => Boolean(element.closest('.atlas-featured-work, .atlas-card-image')));
    if (!sourceIsLocal && !isAtlasArt) continue;
    if (!fullPage && !await img.evaluate(element => { const r = element.getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0; })) continue;
    if (fullPage) await img.scrollIntoViewIfNeeded();
    await decode(img);
  }
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  const file = path.join(screenshotDir, `${name}.png`);
  await page.screenshot({ path: file, fullPage: fullPage && height < 30000, animations: 'disabled' });
  console.log(`Screenshot: ${file}`);
}
async function checkNavigation(page, width) {
  const nav = page.getByRole('navigation', { name: 'Main navigation', includeHidden: true });
  const menu = page.locator('.menu-toggle');
  assert.deepEqual(await nav.locator('a').evaluateAll(links => links.map(link => link.getAttribute('href'))), ['/projects/', '/writing/', '/art/', '/food/', '/just-because/', '/search/']);
  if (width <= 760) {
    await menu.waitFor({ state: 'visible' });
    await nav.waitFor({ state: 'hidden' });
    await attribute(menu, 'aria-expanded', 'false');
    assert.equal(await nav.evaluate(element => element.hidden), true);
    await menu.click();
    await nav.waitFor({ state: 'visible' });
    await attribute(menu, 'aria-expanded', 'true');
    await nav.locator('a').first().focus();
    await page.keyboard.press('Escape');
    await nav.waitFor({ state: 'hidden' });
    await attribute(menu, 'aria-expanded', 'false');
    await focused(menu);
  } else {
    await nav.waitFor({ state: 'visible' });
    await menu.waitFor({ state: 'hidden' });
  }
}
async function checkGallery(page, selector, count, captions) {
  const links = page.locator(selector);
  assert.equal(await links.count(), count, `${selector} count`);
  const opener = links.first();
  await opener.click();
  const viewer = page.getByRole('dialog', { name: 'Image viewer' });
  await viewer.waitFor({ state: 'visible' });
  await textIs(viewer.locator('.viewer-count'), `1 / ${count}`);
  await textIs(viewer.locator('.viewer-caption'), captions[0]);
  await decode(viewer.locator('.viewer-stage img'));
  assert.equal(await viewer.locator('.viewer-prev').isDisabled(), true);
  await page.keyboard.press('ArrowRight');
  await textIs(viewer.locator('.viewer-count'), `2 / ${count}`);
  await textIs(viewer.locator('.viewer-caption'), captions[1]);
  await viewer.locator('.viewer-prev').click();
  await textIs(viewer.locator('.viewer-count'), `1 / ${count}`);
  await page.keyboard.press('Escape');
  await viewer.waitFor({ state: 'hidden' });
  await focused(opener);
  await links.last().click();
  await textIs(viewer.locator('.viewer-count'), `${count} / ${count}`);
  assert.equal(await viewer.locator('.viewer-next').isDisabled(), true);
  await viewer.locator('.viewer-close').click();
  await viewer.waitFor({ state: 'hidden' });
  await focused(links.last());
}
async function artCaptions(page, selector) {
  return page.locator(selector).evaluateAll(links => links.map(link => [...link.closest('figure').querySelector('figcaption').children].map(part => part.textContent.trim()).join(' — ')));
}
async function checkDirectory(page, width) {
  const rows = page.locator('.directory-row[data-kind]');
  assert.equal(await rows.count(), 7);
  const filters = page.locator('.directory-filters');
  await filters.waitFor({ state: 'visible' });
  for (const [kind, count] of [['Machine learning', 3], ['LLM systems', 4], ['All', 7]]) {
    const button = filters.getByRole('button', { name: kind, exact: true });
    await button.click();
    await attribute(button, 'aria-pressed', 'true');
    assert.equal(await filters.locator('[aria-pressed="true"]').count(), 1);
    assert.equal(await page.locator('.directory-row:not([hidden])').count(), count, `${kind} count`);
    assert.equal(await page.locator('.directory-row:visible').count(), count, `${kind} visible count`);
    for (const row of await rows.all()) assert.equal(await row.evaluate(element => element.hidden), kind !== 'All' && await row.getAttribute('data-kind') !== kind);
  }
  if (width === 1440) {
    const preview = page.locator('.directory-preview img');
    for (const [index, action] of [[1, 'hover'], [2, 'focus']]) {
      await rows.nth(index)[action]();
      await attribute(preview, 'src', await rows.nth(index).getAttribute('data-preview'));
      await preview.waitFor({ state: 'visible' });
      await decode(preview);
    }
  } else if (width <= 760) {
    for (const row of await rows.all()) {
      const image = row.locator('img');
      await image.waitFor({ state: 'visible' });
      assert.equal(await image.getAttribute('src'), await row.getAttribute('data-preview'));
      await decode(image);
    }
  }
}
async function checkStaticSculpture(page, name) {
  const figure = page.locator('figure[data-sculpture]');
  const poster = figure.locator('.sculpture-poster');
  await figure.locator('[data-sculpture-next]').waitFor({ state: 'visible' });
  await attribute(figure, 'data-renderer', 'static');
  await attribute(figure, 'data-bloom', 'off');
  await attribute(figure, 'data-paused', 'true');
  await poster.waitFor({ state: 'visible' });
  await decode(poster);
  const original = await poster.getAttribute('src');
  const variant = await figure.getAttribute('data-variant');
  await figure.locator('[data-sculpture-next]').click();
  await eventually(async () => assert.notEqual(await figure.getAttribute('data-variant'), variant), `${name}: static variation`);
  assert.notEqual(await poster.getAttribute('src'), original);
  await decode(poster);
  await attribute(figure, 'data-renderer', 'static');
  await attribute(figure, 'data-paused', 'true');
}

try {
  await eventually(async () => {
    if (serverError) throw serverError;
    assert.equal((await fetch(base, { signal: AbortSignal.timeout(2000) })).ok, true);
  }, 'Preview server did not start');
  browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'], ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) });
  assert.equal(Object.keys(legacy).length, 19);
  for (const slug of Object.keys(legacy)) {
    const response = await fetch(`${base}/posts/${encodeURIComponent(slug)}/`);
    assert.equal(response.status, 200, `Legacy HTTP route: ${slug}`);
    assert.match(response.headers.get('content-type'), /text\/html/);
    assert.match(await response.text(), /data-preserved="true"/, `Legacy body: ${slug}`);
  }
  const viewports = [{ width: 1440, height: 900, name: 'desktop' }, { width: 390, height: 844, name: 'mobile' }, { width: 320, height: 800, name: 'narrow' }, { width: 844, height: 390, name: 'landscape' }];
  const atlasSlugs = ['pattachitra', 'mithila', 'gond', 'warli', 'kalamkari', 'dhokra', 'longpi'];
  const routes = ['/', '/projects/', '/projects/archive/', '/art/', '/food/', '/just-because/', '/just-because/indian-art/', ...atlasSlugs.map(slug => `/just-because/indian-art/${slug}/`), '/about/', '/writing/', '/archives/', '/categories/', '/tags/', '/posts/context-engineering/', '/posts/llms-101/', '/notes/art-and-science-of-paper/', '/projects/counter/', '/studies/what-makes-a-wait-feel-long/', ...engineering.map(item => `/projects/${item.slug}/`)];
  const captures = new Map([['/', 'home'], ['/projects/', 'projects'], ['/art/', 'art'], ['/food/', 'food'], ['/just-because/indian-art/', 'indian-art-atlas'], ['/just-because/indian-art/pattachitra/', 'pattachitra'], ['/posts/context-engineering/', 'article']]);
  for (const { width, height, name } of viewports) {
    const page = await browser.newPage({ viewport: { width, height }, reducedMotion: 'reduce' });
    observe(page, name, true);
    for (const route of routes) {
      await visit(page, route);
      assert.equal(await page.locator('.site-header').count(), 1, `${route} header`);
      assert.equal(await page.locator('.site-footer').count(), 1, `${route} footer`);
      await decode(page.locator('.brand .flow-mark'));
      assert.equal(await page.locator('.brand .flow-mark').isVisible(), true);
      assert.equal(await page.locator('link[rel="icon"]').getAttribute('href'), '/assets/img/flow-mark.png');
      await page.evaluate(() => document.fonts.ready);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `${route} overflow at ${width}x${height}`);
      if (route === '/') {
        await checkNavigation(page, width);
        const selector = '.home-art-grid a[data-gallery="home-art"]';
        await checkGallery(page, selector, 3, await artCaptions(page, selector));
        assert.doesNotMatch(await page.locator('main').textContent(), /Krishna|Statue of Liberty/i);
        await checkStaticSculpture(page, name);
      }
      if (route === '/projects/') await checkDirectory(page, width);
      if (route === '/just-because/indian-art/') {
        const rows = page.locator('.atlas-row');
        const visibleRows = page.locator('.atlas-row:visible');
        const search = page.locator('#atlas-search');
        const category = page.locator('#atlas-category');
        const reset = page.locator('#atlas-reset');
        assert.equal(await rows.count(), 7);
        assert.equal(await visibleRows.count(), 7);
        if (width === 1440) {
          for (const image of await page.locator('.atlas-featured-work img, .atlas-card-image img').all()) {
            await image.scrollIntoViewIfNeeded();
            await decode(image);
          }
        }
        await search.fill('pattachitra');
        assert.equal(await visibleRows.count(), 1, `Atlas search at ${width}`);
        assert.match(await visibleRows.first().textContent(), /pattachitra/i);
        await search.fill('no-such-art-form-12345');
        assert.equal(await visibleRows.count(), 0, 'Atlas should display no matches');
        assert.ok((await page.locator('[data-atlas-status]').textContent()).trim());
        await reset.click();
        assert.equal(await visibleRows.count(), 7, 'Reset clears search');
        await category.selectOption({ label: 'Pottery' });
        assert.equal(await visibleRows.count(), 1, 'Pottery filter');
        assert.match(await visibleRows.first().textContent(), /longpi/i);
        await reset.click();
        assert.equal(await visibleRows.count(), 7, 'Reset clears category');
        await page.locator('#atlas-region').selectOption({ label: 'Odisha' });
        assert.equal(await visibleRows.count(), 2, 'Odisha filter includes Pattachitra and Dhokra');
        await reset.click();
        assert.equal(await visibleRows.count(), 7, 'Reset clears region');
        await search.focus();
        await page.keyboard.type('warli');
        assert.equal(await visibleRows.count(), 1, 'Keyboard search');
        assert.match(await visibleRows.first().textContent(), /warli/i);
        await reset.click();
      }
      if (route === '/art/') {
        await checkGallery(page, '.art-item a[data-gallery="art"]', 10, portfolio.art.map(item => `${item.title} — ${item.caption}`));
        assert.doesNotMatch(await page.locator('main').textContent(), /Krishna|Statue of Liberty/i);
        for (const image of await page.locator('.art-item img').all()) await decode(image);
      }
      if (route === '/food/') {
        assert.equal(await page.locator('.food-photo img').count(), 76);
        assert.deepEqual(await page.locator('.food-photo figcaption').allTextContents(), foodPhotos.map(photo => photo.caption));
        await checkGallery(page, '.food-photo a[data-gallery="food"]', 76, foodPhotos.map(photo => photo.caption));
        await checkGallery(page, '.food-selection a[data-gallery="food-selected"]', 3, await page.locator('.food-selection figcaption').allTextContents());
        if (width === 1440) {
          for (const [index, image] of (await page.locator('.food-photo img').all()).entries()) {
            await decode(image);
            const photo = foodPhotos[index];
            const size = await image.evaluate(element => ({ width: element.naturalWidth, height: element.naturalHeight }));
            // Collection dimensions describe previews; links serve larger originals.
            assert.deepEqual(size, { width: photo.width, height: photo.height }, `Food preview dimensions: ${photo.file}`);
            const link = image.locator('..');
            assert.equal(await link.getAttribute('href'), `/assets/img/food/${encodeURIComponent(photo.file)}`);
            const original = await page.evaluate(async src => {
              const img = new Image(); img.src = src;
              try { await img.decode(); } catch { throw new Error(`Cannot decode original food photo: ${src}`); }
              return { width: img.naturalWidth, height: img.naturalHeight };
            }, await link.getAttribute('href'));
            assert.ok(original.width >= size.width && original.height >= size.height, `Full-size food photo: ${photo.file}`);
            assert.ok(Math.abs(original.width / original.height - size.width / size.height) < 0.01, `Original food aspect ratio: ${photo.file}`);
          }
        }
      }
      if (route === '/posts/llms-101/') assert.equal(await page.locator('.katex-display').count(), 4);
      if (route === '/posts/context-engineering/') {
        await page.locator('.article-toc summary').click();
        await page.locator('.article-toc a').first().click();
        assert.equal(new URL(page.url()).hash, '#the-fundamental-constraint');
        await page.locator('.article-toc summary').click();
      }
      if (engineering.some(item => route === `/projects/${item.slug}/`)) assert.equal(await page.locator('.case-body').count(), 1);
      if (captures.has(route) && ['desktop', 'mobile'].includes(name)) {
        await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
        await screenshot(page, `${name}-${captures.get(route)}`, route !== '/food/');
      }
    }
    await visit(page, '/');
    if (width <= 760) await page.locator('.menu-toggle').click();
    await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Food', exact: true }).click();
    await page.waitForURL(`${base}/food/`);
    await visit(page, '/search/');
    await page.locator('#search-input').fill('context engineering');
    await page.locator('#search-results a[href="/posts/context-engineering/"]').waitFor();
    await page.locator('#search-input').fill('flexures');
    await page.locator('#search-results a[href="/notes/compliant-design-flexures/"]').waitFor();
    await page.close();
    console.log(`Browser layout, navigation, galleries, filters and search passed at ${width}x${height}.`);
  }
  assert.deepEqual(threeRequests, [], 'Reduced motion must not fetch any Three modules');

  // Control browser time so partial-output and cancellation checks cannot miss their windows.
  const study = await browser.newPage();
  observe(study, 'study');
  await study.clock.install();
  await study.clock.pauseAt(new Date(Date.now() + 1000));
  await visit(study, '/studies/what-makes-a-wait-feel-long/');
  await study.locator('[data-mode="ready"]').focus();
  await study.keyboard.press('Enter');
  await focused(study.locator('[data-mode="ready"]'));
  await study.clock.runFor(1000);
  await textIs(study.locator('[data-output]'), 'The');
  await study.clock.runFor(3000);
  await textIs(study.locator('[data-output]'), 'The wait is over.');
  await study.locator('[data-mode="all"]').click();
  await study.clock.runFor(3999);
  await textIs(study.locator('[data-output]'), '');
  await study.clock.runFor(1);
  await textIs(study.locator('[data-output]'), 'The wait is over.');
  await study.locator('[data-mode="ready"]').click();
  await study.locator('[data-stop]').click();
  await study.clock.runFor(5000);
  await textIs(study.locator('[data-output]'), '');
  assert.equal(await study.locator('[data-stop]').isDisabled(), true);
  await study.close();

  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, reducedMotion: 'no-preference' });
    observe(page, `WebGL ${width}`);
    await visit(page, '/');
    const figure = page.locator('figure[data-sculpture]');
    await figure.scrollIntoViewIfNeeded();
    await attribute(figure, 'data-renderer', 'webgl');
    await attribute(figure, 'data-bloom', 'on');
    await attribute(figure, 'data-paused', 'false');
    await figure.locator('canvas').waitFor({ state: 'visible' });
    assert.equal(await figure.locator('canvas').evaluate(canvas => canvas.width > 0 && canvas.height > 0), true);
    const toggle = figure.locator('[data-sculpture-toggle]');
    await toggle.click();
    await attribute(figure, 'data-paused', 'true');
    await attribute(toggle, 'aria-pressed', 'true');
    const variants = new Set();
    for (let index = 0; index < 7; index++) {
      const variant = await figure.getAttribute('data-variant');
      variants.add(variant);
      if (screenshotDir && index < 3) {
        const file = path.join(screenshotDir, `${width === 1440 ? 'desktop' : 'mobile'}-sculpture-${variant}.png`);
        await figure.screenshot({ path: file, animations: 'disabled' });
        console.log(`Screenshot: ${file}`);
      }
      await figure.locator('[data-sculpture-next]').click();
      await eventually(async () => assert.notEqual(await figure.getAttribute('data-variant'), variant), 'WebGL variation did not change');
      assert.ok(variationDistance(Number(await figure.getAttribute('data-variant')), Number(variant)) > .2, 'New variation must visibly change the sculpture');
      await attribute(figure, 'data-renderer', 'webgl');
      await attribute(figure, 'data-bloom', 'on');
      await attribute(figure, 'data-paused', 'true');
    }
    assert.equal(variants.size, 7);
    // Resize and a system-theme change must redraw the bloom while paused.
    await page.setViewportSize({ width: width + 20, height: 800 });
    await page.emulateMedia({ colorScheme: 'dark' });
    await attribute(figure, 'data-bloom', 'on');
    await attribute(figure, 'data-paused', 'true');
    await page.setViewportSize({ width, height: 900 });
    await toggle.click();
    await attribute(figure, 'data-paused', 'false');
    await attribute(toggle, 'aria-pressed', 'false');
    await page.locator('.site-footer').scrollIntoViewIfNeeded();
    await attribute(figure, 'data-paused', 'true');
    await figure.scrollIntoViewIfNeeded();
    await attribute(figure, 'data-paused', 'false');
    // A user pause must survive leaving and returning to the artwork.
    await toggle.click();
    await page.locator('.site-footer').scrollIntoViewIfNeeded();
    await figure.scrollIntoViewIfNeeded();
    await attribute(figure, 'data-paused', 'true');
    await attribute(toggle, 'aria-pressed', 'true');
    // Exercise actual GPU-context loss, not only a missing-context mock.
    await figure.locator('canvas').evaluate(canvas => {
      const extension = canvas.getContext('webgl2').getExtension('WEBGL_lose_context');
      if (!extension) throw new Error('Context-loss extension unavailable in test browser');
      extension.loseContext();
    });
    await checkStaticSculpture(page, 'lost WebGL context');
    await page.close();
  }
  const bloomFailure = await browser.newPage({ reducedMotion: 'no-preference' });
  observe(bloomFailure, 'bloom failure');
  await bloomFailure.route('**/sculpture-bloom.js*', route => route.fulfill({
    contentType: 'text/javascript',
    body: 'export function createBloom() { return {resize(){},setTheme(){},dispose(){},render(){throw Error("Framebuffer unavailable")}}; }',
  }));
  await visit(bloomFailure, '/');
  await attribute(bloomFailure.locator('[data-sculpture]'), 'data-renderer', 'webgl');
  await attribute(bloomFailure.locator('[data-sculpture]'), 'data-bloom', 'off');
  await bloomFailure.locator('[data-sculpture-next]').click();
  await attribute(bloomFailure.locator('[data-sculpture]'), 'data-renderer', 'webgl');
  await bloomFailure.close();

  const transitions = await browser.newPage({ viewport: {width:1440,height:900}, reducedMotion:'no-preference' });
  observe(transitions, 'navigation transitions');
  await visit(transitions, '/');
  await transitions.locator('.writing-section').evaluate(element => scrollTo({top:element.getBoundingClientRect().top + scrollY - innerHeight - 60,behavior:'instant'}));
  await eventually(async () => assert.equal(await transitions.locator('.writing-section').evaluate(el => getComputedStyle(el).opacity), '0'), 'Below-fold section blends into the background');
  const drift = await transitions.locator('.writing-section').evaluate(element => new Promise(resolve => {
    const positions = [];
    const top = element.getBoundingClientRect().top + scrollY;
    let start;
    scrollTo({top:top - 100,behavior:'smooth'});
    function sample(time) {
      start ??= time;
      positions.push(element.getBoundingClientRect().top + scrollY);
      if (time - start < 600) requestAnimationFrame(sample);
      else resolve(Math.max(...positions) - Math.min(...positions));
    }
    requestAnimationFrame(sample);
  }));
  assert.ok(drift < 1, 'A reveal must not move content independently of scrolling');
  // Start a fresh visit: the preceding scroll may already reveal this section.
  await visit(transitions, '/');
  await transitions.locator('.personal-section').evaluate(element => scrollTo({top:element.getBoundingClientRect().top + scrollY - innerHeight - 60,behavior:'instant'}));
  await eventually(async () => assert.equal(await transitions.locator('.personal-section').evaluate(el => getComputedStyle(el).opacity), '0'), 'Below-fold personal section dissolves');
  await transitions.emulateMedia({reducedMotion:'reduce'});
  await eventually(async () => assert.equal(await transitions.locator('.personal-section').evaluate(element => element.getAnimations().length), 0), 'Reduced motion should cancel the entrance');
  await eventually(async () => assert.equal(await transitions.locator('.personal-section').evaluate(element => getComputedStyle(element).opacity), '1'), 'Reduced motion restores full opacity');
  // The primary link keeps native fragment/history behaviour with motion off.
  await transitions.locator('.hero-actions a').click();
  assert.equal(new URL(transitions.url()).hash, '#projects');
  await transitions.emulateMedia({reducedMotion:'no-preference'});
  await transitions.locator('#main-navigation a[href="/projects/"]').click();
  await transitions.waitForURL(base + '/projects/');
  assert.equal(await transitions.locator('main > :first-child').evaluate(element => getComputedStyle(element).animationName), 'page-enter');
  await eventually(async () => assert.equal(await transitions.locator('main > :first-child').evaluate(element => getComputedStyle(element).opacity), '1'), 'Page entrance should finish visibly');
  await transitions.goBack();
  await transitions.locator('.hero-copy h1').waitFor();
  assert.equal(await transitions.evaluate(() => getComputedStyle(document.body).opacity), '1');
  await transitions.locator('.footer-links a[href="https://github.com/wildchaser1703"]').focus();
  await eventually(async () => assert.equal(await transitions.locator('.site-footer').evaluate(element => element.getAnimations().length), 0), 'Keyboard focus should cancel the entrance');
  await transitions.close();

  const fallback = await browser.newPage({ reducedMotion: 'no-preference' });
  observe(fallback, 'no WebGL');
  await fallback.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    window.webglAttempts = 0;
    HTMLCanvasElement.prototype.getContext = function (type, ...args) {
      if (/webgl/i.test(type)) { window.webglAttempts++; return null; }
      return original.call(this, type, ...args);
    };
  });
  await visit(fallback, '/');
  await checkStaticSculpture(fallback, 'no WebGL');
  assert.ok(await fallback.evaluate(() => window.webglAttempts > 0), 'No-WebGL test must exercise context creation');
  await fallback.close();

  for (const width of [1440, 390]) {
    const noScript = await browser.newContext({ javaScriptEnabled: false, viewport: { width, height: 900 } });
    const page = await noScript.newPage();
    observe(page, `no JS ${width}`);
    await visit(page, '/', false);
    await page.locator('.site-header nav').waitFor({ state: 'visible' });
    await page.locator('.menu-toggle').waitFor({ state: 'hidden' });
    await page.locator('.sculpture-poster').waitFor({ state: 'visible' });
    await decode(page.locator('.sculpture-poster'));
    assert.equal(await page.locator('.home-art-grid a[data-gallery="home-art"]').count(), 3);
    await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Projects', exact: true }).click();
    await page.waitForURL(`${base}/projects/`);
    assert.equal(await page.locator('.directory-row:visible').count(), 7);
    for (const link of await page.locator('.directory-row').all()) assert.equal((await page.request.get(base + await link.getAttribute('href'))).status(), 200);
    for (const [route, selector, count] of [['/art/', '.art-item a', 10], ['/food/', '.food-photo a', 76]]) {
      await visit(page, route);
      assert.equal(await page.locator(selector).count(), count);
      for (const link of await page.locator(selector).all()) {
        assert.equal(await link.isVisible(), true);
        assert.match(await link.getAttribute('href'), /^\/assets\/.*\.jpg$/);
      }
      const target = await page.locator(selector).first().getAttribute('href');
      await page.locator(selector).first().click();
      await page.waitForURL(base + target);
      assert.equal((await page.request.get(base + target)).status(), 200);
    }
    await visit(page, '/just-because/indian-art/', false);
    assert.equal(await page.locator('.atlas-row:visible').count(), 7, 'All atlas entries must work without JavaScript');
    for (const slug of atlasSlugs) {
      const route = `/just-because/indian-art/${slug}/`;
      assert.equal(await page.locator(`.atlas-row h3 a[href="${route}"]`).isVisible(), true, `Atlas link without JS: ${slug}`);
      assert.equal((await page.request.get(base + route)).status(), 200, `Atlas route without JS: ${slug}`);
    }
    await visit(page, '/food/');
    assert.equal(await page.locator('.food-selection a:visible').count(), 3);
    await visit(page, '/studies/what-makes-a-wait-feel-long/');
    assert.equal(await page.locator('.comparison').isVisible(), true);
    await noScript.close();
  }
  assert.deepEqual(errors, [], 'Uncaught browser errors');
  console.log('Browser checks passed at desktop, mobile, narrow, and landscape widths; 19 legacy HTTP routes, study timing, WebGL, reduced-motion, no-WebGL, and no-JavaScript fallbacks verified.');
} finally {
  await browser?.close();
  server.kill('SIGTERM');
}
