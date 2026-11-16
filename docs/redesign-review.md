# Portfolio redesign review

This release implements the agreed direction: an AI engineer's portfolio with original procedural art, source-backed engineering projects, readable writing, and separate painting and food collections. It uses the repository's existing build, test, and GitHub Pages workflow; repository selection, credentials, and network policy remain unchanged.

## Stage 1 — Structure and engineering identity

- Homepage leads with Toshali's name, AI engineering role, and links to projects and writing.
- Three featured projects: Documentation RAG Pipeline, Agentic Insight Analyst, and Retail Demand Forecasting.
- Each has a dedicated case study with problem, approach, engineering decisions, implementation scope, limitations, and commit-pinned source links. Public source was read; the remote applications and their model calls were not executed. No benchmark or production claims are added.
- Project directory includes these projects, the existing six code experiments, the waiting-time study, and two field notes. Filters enhance a complete static list. Desktop hover and keyboard focus update the preview; mobile keeps previews inline.
- Main navigation: Projects, Writing, Art, Food, About, Search. Existing study, post, category, tag, and archive routes remain available.
- `/about/` is now a real page. Existing favourites appear there.
- Source references: [project-source-evidence.md](project-source-evidence.md).

Review routes: `/`, `/projects/`, `/projects/production-rag-system/`, `/about/`.

## Stage 2 — Procedural artwork and visual system

- Original Three.js artwork, **Woven field**, uses 960 desktop or 600 mobile strands over an asymmetric folded annulus. Warp and weft share seeded geometry with three curated compositions.
- Local pointer displacement, subtle ambient motion, pause/resume and variation controls. Touch scrolling remains native.
- Rendering pauses offscreen and when the tab is hidden. User pause persists on returning to the artwork. Reduced motion avoids downloading Three or creating a WebGL context. WebGL failures/context loss fall back to the matching poster.
- All runtime modules are served locally. No model calls, external fonts, analytics, or CDN dependencies.
- Posters are generated from the same geometry and encoded to WebP at build time. Only the current variant's poster is loaded.
- Warm off-white and charcoal interface; dark red controls; yellow/orange/red in the artwork. SVG arrows avoid font-dependent missing glyphs.
- **Typography limitation:** Sequel Sans and PP Fraktion Mono are commercial fonts. User authorized temporary fallbacks because licences are not yet available. CSS declares the intended families and uses Arial/Helvetica and Courier New/system monospace until licensed webfonts are supplied. The reference site's font files have not been copied. Final typography is pending, not represented as verified.

Review routes: `/` with all three variations, reduced motion, and WebGL disabled.

## Stage 3 — Art, food, reading, and interaction

- Statue of Liberty was removed from the portfolio and the source asset removed from the site. Ten paintings remain, with existing titles and captions.
- Homepage selects three paintings; `/art/` preserves natural proportions in an asymmetric desktop arrangement and a large single-column mobile gallery.
- All 76 food photographs remain in the complete `/food/` collection. Coffee, ice cream, and a banana-leaf meal introduce it as a small curated sequence. The opener does not replace or reorder the complete collection.
- Desktop food rows accommodate image proportions; mobile uses two columns with occasional full-width photos. Existing food captions and descriptive alt text from the prior draft remain unchanged.
- Shared native-dialog viewer supports previous/next, keyboard arrows, Escape, focus restoration, and uncropped originals. Image links remain usable without JavaScript.
- Article layout has restrained typography, contents links, equations and horizontally scrollable code. Published Markdown and preserved HTML snapshots are untouched.
- Small-screen navigation supports keyboard operation, Escape, and breakpoint changes. The full navigation remains available without JavaScript.

Review routes: `/art/`, `/food/`, `/writing/`, `/posts/context-engineering/`, `/posts/llms-101/`.

## Stage 4 — Verification and release boundary

Build and static checks:

```sh
npm ci
npm run build
npm test
```

Browser checks (uses Playwright's installed Chromium by default):

```sh
npx playwright install chromium
npm run test:browser
```

In this cloud workspace:

```sh
npm ci --cache /tmp/toshali-npm-cache
CHROME_BIN=/usr/bin/chromium npm run test:browser
npm run dev
```

Preview serves `dist` at `http://127.0.0.1:4173`. `PORT` changes the preview port. All dependencies, including Three.js and the poster encoder, are locked in `package-lock.json`. Existing CI invokes build/static/browser checks and deploys the generated `dist` artifact after a successful push to `main`; its workflow configuration is unchanged.

A task-local review bundle under `/workspace/redesign-review` records before/after post hashes, existing-URL checks, screenshots, diff, and validation output. The permanent tests verify legacy post source hashes and fragments as well as new interactions. The bundle contains a separate pre-existing-draft patch so the earlier gallery/caption changes remain distinguishable.

Physical-device GPU performance and the final licensed fonts need a separate review before claiming those are verified.
