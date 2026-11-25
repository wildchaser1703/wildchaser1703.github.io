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

## Publish an artwork or exhibition

Keep an existing work's slug and object source stable. Add a row to `data/atlas-works.json` for a distinct work, with its title, maker, date, medium, context and object URL. Set `places` only when the work or its named practice has a documented regional connection. A related tradition is optional for an independently catalogued regional work. The place directory lists all 28 states and eight Union Territories even when an artwork image is pending.

Create a matching entry in `data/atlas-image-audit.json`. Record the object and image sources separately, the photographer or provider, licence and policy URLs, exact-match evidence, date checked, and any outstanding permissions. Keep unresolved records discoverable from the collection’s research section. See the [image audit](indian-art-image-audit.md) for the current place gaps and draft permission requests.

For a verified reproduction, use Sharp to auto-orient and encode full-frame WebP files under `assets/img/museum/`. Produce a thumbnail, preview and larger view, fitting within 480×672, 960×1344 and 2000×2800 without enlargement. Record the output widths, large height, matching local file paths, descriptive alt text and visible credit in the work record. Keep the original source URL in the audit. The site’s generated pages and assets currently fit within the existing Pages limit.

Add exhibitions in `data/atlas-exhibitions.json` with a unique slug, a curatorial question, an introduction and an ordered list of work slugs and original labels. The build requires verified images for every stop. Exhibition pages link to object records and appear in site search and the sitemap. Existing curated collections and saved selections use their original URLs.

Pull requests run the existing Build and Deploy workflow. After checks pass and the PR is merged, the workflow on `main` deploys GitHub Pages. Verify the museum entrance, image loading, a regional page, site search, and save and share interactions on the public domain.
