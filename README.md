# Toshali Mohapatra’s site

This site uses a Node.js static build. It combines writing, small projects, studies, and artwork in one design.

## Develop

Use Node.js 22 or newer:

```sh
npm ci
npm run dev
```

The preview serves the generated `dist` directory on port 4173. Set `PORT` to use a different port. Run `npm run build && npm test` to build and check the site without starting a server.

For the browser interaction checks, run `npx playwright install chromium` once and then `npm run test:browser`. On Linux, Playwright may also require its documented system dependencies. Set `CHROME_BIN` to use an existing compatible Chromium executable.

## Content

Published Markdown remains in `_posts`. `data/legacy-posts.json` contains preserved renderings and hashes for the existing articles so their formatting and heading fragments remain stable. When an existing article is edited, the build renders its Markdown and reuses known heading IDs where possible; the checks report missing legacy fragments. New Markdown posts use a dated filename, front matter with a title and date, and optional `categories` and `tags` arrays.

Portfolio projects and artwork are listed in `data/portfolio.json`; selected notebook excerpts are in `data/field-notes.json`. Artwork is stored locally under `assets/art`.

## Deployment

Pull requests run the Node build and checks. A successful push to `main` builds the site and deploys `dist` to GitHub Pages. The small service-worker migration at `/sw.min.js` retires the previous site's cached assets for returning visitors.
