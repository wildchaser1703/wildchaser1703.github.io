# Portfolio typefaces

The intended design uses **Sequel Sans** for display/body text and **PP Fraktion Mono** for short interface labels. These commercial families require appropriate webfont licences. The user has authorized temporary system fallbacks until licensed files are available.

To activate licensed webfonts, supply these authorized WOFF2 files here:

- `sequel-sans-regular.woff2` — regular, weight 400
- `sequel-sans-semibold.woff2` — semibold, weight 600
- `pp-fraktion-mono-regular.woff2` — regular, weight 400

The build detects and copies only files that exist and generates `dist/assets/css/licensed-fonts.css`. There are no requests for missing files. The final fonts require desktop/mobile layout and readability review after installation. Do not download commercial font binaries from reference sites.

Older open-source font files remain in the repository for history but are not shipped by the current build. Their licence files remain with them.
