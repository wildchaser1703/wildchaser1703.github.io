# Maintaining the Indian Art Atlas

The atlas is part of the existing static website. The build makes HTML pages in advance; search, filtering, image enlargement and saved collections run in the visitor's browser. No database or account is required. Saved selections stay in that browser; the share link contains the selected work identifiers.

## Where the information lives

- `data/indian-art.json`: a tradition's name, places, materials, background, process and sources. `traditionSourceRefs` and `processSourceRefs` are zero-based positions in that entry's `sources` list. Each entry needs a unique lowercase, hyphenated `slug`.
- `data/atlas-works.json`: individual works or explicitly labelled groups, their makers, dates, media and collection records. A work's `tradition` must match a tradition's slug. If the maker or date is unknown, say so instead of guessing.
- `data/atlas-makers.json`: short, sourced maker profiles. `traditions` contains matching tradition slugs. Optional `aliases` connect differently spelled names to the right records; verify that they refer to the same person first.
- `data/atlas-geography.json`: the directory of states, Union Territories and cross-border places. Coordinates are approximate markers, not the full extent or a precise location of a tradition. The build verifies coverage of 28 states and eight Union Territories.
- `data/curiosity-projects.json`: the projects shown under Just Because. Each project also needs its own page built in `scripts/build.mjs`.
- `data/india-outline.json` and `data/india-outline-license.txt`: a simplified, illustrative map outline and its licence.

## Research and image rights

Link claims to reliable sources and keep the source's uncertainty. Practices cross state and national borders, and different artists make different choices. A photograph's licence does not by itself establish rights in the artwork it depicts.

Only add an image after checking both the work and the image's reuse terms. For a documented work, `image` is the larger image, `preview` can be a smaller image for the gallery, `imageSource` links to the image's rights information, `credit` identifies the creator or institution and any changes, and `licenseUrl` links to the licence where applicable. Include the object record separately in `recordUrl`. If permission is unclear, leave out the image and link to the source record. Context photographs belong in a tradition's `illustration` field, with the same image and credit fields; do not present a general photograph as a specific work.

## Build and verify

With Node.js installed, run:

```sh
npm ci
npm run build
npm test
npm run test:browser
```

The checks cover existing post content and URLs, internal links, geography, source references, search, image loading, saved and shared collections, keyboard and mobile behaviour, and browsing without JavaScript. Browser checks need Playwright's Chromium; install it with `npx playwright install --with-deps chromium` when needed. Check the resulting pages and images visually as well as relying on automated tests.
