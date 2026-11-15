import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import net from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';

const probe = net.createServer();
await new Promise((resolve, reject) => probe.once('error', reject).listen(0, '127.0.0.1', resolve));
const port = probe.address().port;
await new Promise(resolve => probe.close(resolve));
const base = `http://127.0.0.1:${port}`;
const server = spawn(process.execPath, ['scripts/serve.mjs'], { env: { ...process.env, PORT: String(port) }, stdio: 'inherit' });
let browser;
try {
  let ready = false;
  for (let attempt = 0; attempt < 50; attempt++) {
    try { const response = await fetch(base); if (response.ok) { ready = true; break; } } catch {}
    await delay(100);
  }
  if (!ready) throw new Error('Preview server did not start');
  browser = await chromium.launch({ headless: true, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) });
  for (const width of [1440, 375, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 800 }, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const route of ['/', '/writing/', '/archives/', '/categories/', '/tags/', '/posts/context-engineering/', '/posts/llms-101/', '/notes/art-and-science-of-paper/', '/projects/counter/', '/studies/what-makes-a-wait-feel-long/']) {
      const response = await page.goto(base + route, { waitUntil: 'networkidle' });
      assert.equal(response.status(), 200, `${route} status`);
      assert.equal(await page.locator('.site-header').count(), 1, `${route} header`);
      assert.equal(await page.locator('.site-footer').count(), 1, `${route} footer`);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `${route} overflow at ${width}px`);
      if (route === '/' && width === 1440) {
        assert.equal(await page.locator('.art-image img').count(), 11);
        assert.equal(await page.getByText('Krishna', { exact: false }).count(), 0);
        for (const image of await page.locator('.art-image img').all()) {
          await image.scrollIntoViewIfNeeded();
          await image.evaluate(element => element.decode());
        }
      }
      if (route === '/posts/llms-101/') assert.equal(await page.locator('.katex-display').count(), 4);
      if (route === '/posts/context-engineering/') {
        await page.locator('.article-toc summary').click();
        await page.locator('.article-toc a').first().click();
        assert.equal(new URL(page.url()).hash, '#the-fundamental-constraint');
      }
    }
    await page.goto(`${base}/search/`);
    await page.locator('#search-input').fill('context engineering');
    await page.locator('#search-results a[href="/posts/context-engineering/"]').waitFor();
    await page.locator('#search-input').fill('flexures');
    await page.locator('#search-results a[href="/notes/compliant-design-flexures/"]').waitFor();
    assert.deepEqual(errors, [], `browser errors at ${width}px`);
    await page.close();
  }
  const page = await browser.newPage();
  await page.goto(`${base}/studies/what-makes-a-wait-feel-long/`);
  await page.locator('[data-mode="ready"]').focus();
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('[data-mode="ready"]').evaluate(element => element === document.activeElement), true);
  await delay(1200);
  assert.equal(await page.locator('[data-output]').textContent(), 'The');
  await delay(3100);
  assert.equal(await page.locator('[data-output]').textContent(), 'The wait is over.');
  await page.locator('[data-mode="all"]').click();
  await delay(1200);
  assert.equal(await page.locator('[data-output]').textContent(), '');
  await delay(3100);
  assert.equal(await page.locator('[data-output]').textContent(), 'The wait is over.');
  await page.locator('[data-mode="ready"]').click();
  await page.locator('[data-stop]').click();
  await delay(1100);
  assert.equal(await page.locator('[data-output]').textContent(), '');
  await page.close();
  const noScript = await browser.newContext({ javaScriptEnabled: false });
  const staticPage = await noScript.newPage();
  await staticPage.goto(`${base}/studies/what-makes-a-wait-feel-long/`);
  assert.equal(await staticPage.locator('.comparison').isVisible(), true);
  await noScript.close();
  console.log('Browser checks passed at desktop, mobile, and narrow widths.');
} finally {
  await browser?.close();
  server.kill('SIGTERM');
}
