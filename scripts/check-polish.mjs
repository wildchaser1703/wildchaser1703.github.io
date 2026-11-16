import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import net from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';
import { createSculpture, FRAME } from '../assets/js/sculpture-shape.js';

const engineering = JSON.parse(await readFile('data/engineering-projects.json','utf8'));
const data = {art:JSON.parse(await readFile('data/portfolio.json','utf8')).art,food:JSON.parse(await readFile('data/food.json','utf8'))};
const probe = net.createServer();
await new Promise(resolve => probe.listen(0,'127.0.0.1',resolve));
const port = probe.address().port;
await new Promise(resolve => probe.close(resolve));
const base = 'http://127.0.0.1:' + port;
const server = spawn(process.execPath,['scripts/serve.mjs'],{env:{...process.env,PORT:String(port)},stdio:'inherit'});
const directory = process.env.SCREENSHOT_DIR;
if(directory) await mkdir(directory,{recursive:true});
let browser;
const errors = [], reports = [];
const poll = async (fn,message) => {
  for(let i=0;i<100;i++) {
    if(await fn()) return;
    await delay(50);
  }
  throw Error(message);
};
try {
  await poll(async()=>{try{return (await fetch(base)).ok;}catch{return false;}},'Server unavailable');
  browser = await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader','--use-angle=swiftshader'],...(process.env.CHROME_BIN ? {executablePath:process.env.CHROME_BIN}:{})});
  // Novel seeds must stay finite, within the frame, and produce different work.
  const fingerprints = new Set();
  for(let seed=1000;seed<1040;seed++) {
    const strands=createSculpture(seed,'preview');
    fingerprints.add(strands[0].points.slice(0,6).join(','));
    for(const {points} of strands) for(let i=0;i<points.length;i+=3) {
      assert.ok(Number.isFinite(points[i+2]));
      assert.ok(Math.abs(points[i]) < FRAME.width/2 && Math.abs(points[i+1]) < FRAME.height/2,'Generated weave escaped its frame');
    }
  }
  assert.equal(fingerprints.size,40);
  for(const colorScheme of ['light','dark']) for(const width of [1440,390]) {
    const page=await browser.newPage({viewport:{width,height:900},colorScheme,reducedMotion:'reduce'});
    page.on('pageerror',error=>errors.push(error.message));
    for(const route of ['/','/projects/','/art/','/food/','/posts/context-engineering/','/posts/llms-101/','/studies/what-makes-a-wait-feel-long/']) {
      assert.equal((await page.goto(base+route)).status(),200);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).colorScheme),colorScheme);
      if(route==='/') {
        await page.locator('[data-collection="art"][data-collection-ready="true"]').waitFor();
        await page.locator('[data-collection="food"][data-collection-ready="true"]').waitFor();
        for(const url of ['https://x.com/sudoher','https://linkedin.com/in/toshalimohapatra','mailto:toshalimohapatra1@gmail.com']) assert.equal(await page.locator('.about-socials a[href="'+url+'"]').count(),1);
        await page.locator('.hero-actions a').click();
        assert.equal(new URL(page.url()).hash, '#projects');
        assert.ok(await page.locator('#projects').evaluate(el => Math.abs(el.getBoundingClientRect().top) < 150));
        await page.locator('.sculpture-info summary').focus();
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('.sculpture-info').getAttribute('open'), '');
        assert.equal(await page.locator('[data-sculpture-variant]').isVisible(), true);
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('.sculpture-info').getAttribute('open'), null);
        assert.equal(await page.locator('.favorite-item').count(),3);
        for(const kind of ['art','food']) {
          const section=page.locator('[data-collection="'+kind+'"]');
          const links=section.locator('a[data-gallery]');
          const before=await links.evaluateAll(items=>items.map(x=>x.getAttribute('href')));
          const heights=await section.locator('figure a').evaluateAll(items=>items.map(x=>x.getBoundingClientRect().height));
          await section.locator('[data-collection-next]').click();
          await poll(async()=>(await links.first().getAttribute('href'))!==before[0],'Manual gallery rotation did not change');
          const after=await links.evaluateAll(items=>items.map(x=>x.getAttribute('href')));
          assert.equal(new Set(after).size,3);
          assert.ok(after.every(x=>!before.includes(x)));
          assert.deepEqual(await section.locator('figure a').evaluateAll(items=>items.map(x=>x.getBoundingClientRect().height)),heights,'Rotation shifts image frames');
          for(const figure of await section.locator('figure').all()) {
            const link=figure.locator('a'), img=link.locator('img');
            const file=decodeURIComponent((await link.getAttribute('href')).split('/').at(-1));
            const item=data[kind].find(x=>x.file===file);
            assert.ok(item);
            assert.equal(await img.getAttribute('alt'),kind==='art'?item.title:item.alt);
            assert.equal((await figure.locator('figcaption').textContent()).includes(item.caption),true);
            assert.ok(await link.evaluate(x=>parseFloat(getComputedStyle(x).paddingLeft)>=10));
          }
          await links.first().click();
          await page.locator('.image-viewer[open]').waitFor();
          assert.equal(await page.locator('.viewer-stage img').getAttribute('src'),new URL(after[0],base).href);
          await page.keyboard.press('Escape');
        }
      }
      if(process.env.AXE_PATH) {
        await page.addScriptTag({path:process.env.AXE_PATH});
        const violations=await page.evaluate(async()=>(await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}})).violations.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})));
        reports.push({route,width,colorScheme,violations});
        assert.deepEqual(violations,[],JSON.stringify({route,width,colorScheme,violations}));
      }
      if(directory && ['/', '/projects/','/art/','/food/'].includes(route)) {
        await page.evaluate(async()=>{
          for(const img of document.querySelectorAll('main img')) { img.loading='eager'; try {await img.decode();}catch{} }
          scrollTo({top:0,behavior:'instant'});
        });
        const name=route==='/'?'home':route.replaceAll('/','');
        await page.screenshot({path:directory+'/'+colorScheme+'-'+width+'-'+name+'.png',fullPage:route!=='/food/'});
      }
    }
    await page.goto(base+'/about/');
    await page.waitForURL(base+'/#about');
    assert.equal(await page.locator('.home-hero').count(),1);
    assert.equal(await page.locator('#about h2').textContent(), 'A little more about me');
    assert.ok(await page.locator('#about').evaluate(el => Math.abs(el.getBoundingClientRect().top) < 150));
    // SVG label bounds must fit inside their boxes in both themes.
    for(const project of engineering) {
      await page.goto(base+'/assets/img/project-'+project.slug+'.svg');
      const bad=await page.locator('.node').evaluateAll(nodes=>nodes.flatMap(node=>{
        const rect=node.querySelector('rect').getBBox();
        return [...node.querySelectorAll('text')].filter(text=>{
          const b=text.getBBox();
          return b.x<rect.x+12 || b.y<rect.y+8 || b.x+b.width>rect.x+rect.width-12 || b.y+b.height>rect.y+rect.height-8;
        }).map(text=>text.textContent);
      }));
      assert.deepEqual(bad,[],project.slug+' text outside box');
      const crossings=await page.locator('.connector').evaluateAll(paths=>{
        const boxes=[...document.querySelectorAll('.node rect')].map(x=>x.getBBox());
        return paths.filter(path=>{
          for(let length=0;length<=path.getTotalLength();length+=2) {
            const p=path.getPointAtLength(length);
            if(boxes.some(b=>p.x>b.x && p.x<b.x+b.width && p.y>b.y && p.y<b.y+b.height)) return true;
          }
          return false;
        }).length;
      });
      assert.equal(crossings,0,project.slug+' arrow crosses a node');
      if(directory && width===1440) await page.screenshot({path:directory+'/'+colorScheme+'-'+project.slug+'.png'});
    }
    await page.close();
    console.log(colorScheme+' theme, galleries, About, social links and diagrams passed at '+width+'px.');
  }
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  page.on('pageerror',error=>errors.push(error.message));
  await page.clock.install();
  await page.goto(base);
  await page.locator('[data-collection="food"][data-collection-ready="true"]').waitFor();
  await page.locator('[data-collection="art"][data-collection-ready="true"]').waitFor();
  // A pointer resting over a gallery must not prevent the requested rotation.
  for (const kind of ['art','food']) {
    const gallery = page.locator('[data-collection="'+kind+'"]');
    await gallery.locator('a[data-gallery]').first().hover();
    await delay(100);
    const previous = await gallery.locator('a[data-gallery]').evaluateAll(items=>items.map(x=>x.getAttribute('href')));
    await page.clock.fastForward(6100);
    await poll(async()=>(await gallery.locator('a[data-gallery]').first().getAttribute('href'))!==previous[0],kind+' did not rotate while hovered');
  }
  const section=page.locator('[data-collection="food"]');
  await section.scrollIntoViewIfNeeded();
  await page.mouse.move(0,0);
  await delay(100);
  const first=section.locator('a[data-gallery]').first();
  const before=await first.getAttribute('href');
  await page.clock.fastForward(6100);
  await poll(async()=>await first.getAttribute('href')!==before,'Automatic rotation did not run');
  await section.locator('[data-collection-pause]').click();
  const paused=await first.getAttribute('href');
  await page.mouse.move(0,0);
  await page.locator('body').click({position:{x:1,y:1}});
  await page.clock.fastForward(30000);
  assert.equal(await first.getAttribute('href'),paused,'Paused selection changed');
  await section.locator('[data-collection-pause]').click();
  await first.click();
  await page.clock.fastForward(30000);
  assert.equal(await first.getAttribute('href'),paused,'Selection changed underneath viewer');
  await page.keyboard.press('Escape');
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.clock.fastForward(30000);
  assert.equal(await first.getAttribute('href'),paused,'Reduced-motion selection changed');
  for(const colorScheme of ['dark','light']) {
    await page.emulateMedia({colorScheme});
    assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).colorScheme),colorScheme,'Theme did not follow a live system change');
  }
  await page.close();
  // Reloads must start with three different images, even with reduced motion.
  for(const storageAvailable of [true,false]) {
    const context=await browser.newContext({reducedMotion:'reduce'});
    if(!storageAvailable) await context.addInitScript(()=>{
      Storage.prototype.getItem=Storage.prototype.setItem=()=>{throw new DOMException('Storage disabled','SecurityError');};
    });
    const returning=await context.newPage();
    returning.on('pageerror',error=>errors.push(error.message));
    const previous={art:[],food:[]};
    for(let visit=0;visit<4;visit++) {
      await returning.goto(base);
      for(const kind of ['art','food']) {
        const gallery=returning.locator('[data-collection="'+kind+'"][data-collection-ready="true"]');
        await gallery.waitFor();
        const current=await gallery.locator('a[data-gallery]').evaluateAll(items=>items.map(x=>x.getAttribute('href')));
        assert.equal(new Set(current).size,3);
        if(storageAvailable) assert.ok(current.every(url=>!previous[kind].includes(url)),kind+' repeats images from the previous visit');
        previous[kind]=current;
      }
    }
    await returning.evaluate(()=>dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true})));
    for(const kind of ['art','food']) {
      const gallery=returning.locator('[data-collection="'+kind+'"]');
      await poll(async()=>{
        const current=await gallery.locator('a[data-gallery]').evaluateAll(items=>items.map(x=>x.getAttribute('href')));
        return current.every(url=>!previous[kind].includes(url));
      },kind+' did not refresh on history restoration');
    }
    await context.close();
  }
  assert.deepEqual(errors,[]);
  if(directory) await writeFile(directory+'/accessibility-results.json',JSON.stringify(reports,null,2));
  console.log('40 generated seeds; fresh selections across reloads, disabled-storage fallback, rotation while hovered, pause, viewer stability and reduced motion passed.');
} finally {
  await browser?.close();
  server.kill();
}
