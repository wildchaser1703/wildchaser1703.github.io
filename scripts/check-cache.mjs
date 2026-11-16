import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

// Exercise a real service worker and HTTP cache, not request interception.
const legacyWorker = `
self.addEventListener('install', e => e.waitUntil((async () => {
  const cache = await caches.open('chirpy-regression');
  await cache.put('/', new Response('<html><head><title>Loading</title></head><body></body></html>', {headers:{'Content-Type':'text/html'}}));
  await self.skipWaiting();
})()));
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => e.respondWith(caches.match(e.request).then(r => r || fetch(e.request))));`;
const retirement = await readFile('public/sw.min.js', 'utf8');
const oldRetirement = retirement.replace(/await Promise\.all\(windows\.map\(client => client\.navigate\(client\.url\)\.catch\(\(\) => \{\}\)\)\);/, '');
let worker = legacyWorker;
let staleCSS = true;
const root = path.resolve('dist');
const mime = { '.html':'text/html', '.js':'application/javascript', '.css':'text/css', '.json':'application/json', '.svg':'image/svg+xml', '.png':'image/png', '.webp':'image/webp', '.jpg':'image/jpeg', '.woff2':'font/woff2' };
const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  if (url.pathname === '/sw.min.js') { response.writeHead(200, {'Content-Type':'application/javascript','Cache-Control':'no-store'}).end(worker); return; }
  if (url.pathname === '/seed/') { response.writeHead(200, {'Content-Type':'text/html'}).end('<!doctype html><title>Cache fixture</title><body>Cache fixture</body>'); return; }
  if (url.pathname === '/assets/css/site.css' && !url.search && staleCSS) {
    response.writeHead(200, {'Content-Type':'text/css','Cache-Control':'max-age=3600'}).end('body { opacity: 0 }'); return;
  }
  let relative = decodeURIComponent(url.pathname);
  if (relative.endsWith('/')) relative += 'index.html';
  const file = path.resolve(root, `.${relative}`);
  if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
  try { const body = await readFile(file); response.writeHead(200, {'Content-Type': mime[path.extname(file)] || 'application/octet-stream'}).end(body); }
  catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({headless:true,...(process.env.CHROME_BIN ? {executablePath:process.env.CHROME_BIN} : {})});
try {
  for (const repaired of [false, true]) {
    worker = legacyWorker;
    const context = await browser.newContext({reducedMotion:'reduce'});
    const page = await context.newPage();
    await page.goto(base + '/seed/');
    await page.evaluate(async () => {
      await navigator.serviceWorker.register('/sw.min.js');
      await navigator.serviceWorker.ready;
      await caches.open('unrelated-app-cache');
    });
    await page.waitForFunction(() => !!navigator.serviceWorker.controller);
    await page.goto(base + '/');
    assert.equal(await page.title(), 'Loading', 'legacy cache should reproduce the blank page');
    worker = repaired ? retirement : oldRetirement;
    await page.evaluate(async () => { const registration = await navigator.serviceWorker.getRegistration(); await registration.update(); });
    await page.waitForFunction(async () => (await navigator.serviceWorker.getRegistrations()).length === 0);
    if (repaired) {
      await page.locator('.hero-copy h1').waitFor();
      assert.equal(await page.title(), 'Home · Toshali Mohapatra');
      assert.equal(await page.evaluate(() => getComputedStyle(document.body).opacity), '1');
      await page.reload();
      assert.equal(await page.locator('.hero-copy h1').isVisible(), true);
    } else {
      assert.equal(await page.title(), 'Loading', 'unregistering alone leaves the blank cached document');
    }
    if (repaired) assert.deepEqual(await page.evaluate(() => caches.keys()), ['unrelated-app-cache']);
    await context.close();
  }
  const page = await browser.newPage({reducedMotion:'reduce'});
  await page.goto(base + '/seed/');
  await page.evaluate(async () => {
    const link = document.createElement('link'); link.rel = 'stylesheet'; link.href = '/assets/css/site.css';
    document.head.append(link); await new Promise(resolve => link.onload = resolve);
  });
  assert.equal(await page.evaluate(() => getComputedStyle(document.body).opacity), '0');
  staleCSS = false;
  await page.goto(base + '/');
  assert.equal(await page.evaluate(() => getComputedStyle(document.body).opacity), '1', 'new release bypasses cached old stylesheet');
  const references = await page.locator('link[rel="stylesheet"],script[src]').evaluateAll(elements => elements.map(el => el.href || el.src));
  assert.ok(references.every(url => new URL(url).searchParams.has('v')));
  console.log('Reproduced stale-worker blank page; verified automatic recovery, no reload loop, unrelated cache preservation, and cached-CSS release isolation.');
} finally { await browser.close(); server.close(); }
