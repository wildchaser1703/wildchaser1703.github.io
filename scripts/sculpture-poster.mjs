// Run from any working directory: node scripts/sculpture-poster.mjs
// The geometry module is deliberately independent of Three and the browser.
import { mkdir } from 'node:fs/promises';
import sharp from 'sharp';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { FRAME, VARIANT_COUNT, createSculpture } from '../assets/js/sculpture-shape.js';

export async function generatePosters(destination = new URL('../assets/img/', import.meta.url)) {
  await mkdir(destination, { recursive: true });
  const width = 900;
  const height = 760;
  const scale = Math.min(width / FRAME.width, height / FRAME.height);
  for (let variant = 0; variant < VARIANT_COUNT; variant++) {
    const strands = createSculpture(variant);
    const paths = strands.map(({ points, color, opacity }) => {
      let d = '';
      for (let i = 0; i < points.length; i += 3) {
        const x = (width / 2 + points[i] * scale).toFixed(1);
        const y = (height / 2 - points[i + 1] * scale).toFixed(1);
        d += `${i ? 'L' : 'M'}${x} ${y}`;
      }
      return `<path d="${d}" stroke="rgb(${color.join(',')})" stroke-opacity="${opacity}"/>`;
    });
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
<title id="title">Woven field — study ${String(variant + 1).padStart(2, '0')}</title>
<desc id="desc">Fine saffron, amber and vermilion threads cross an asymmetric folded loop, leaving an open aperture at its centre.</desc>
<g fill="none" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
${paths.join('\n')}
</g>
</svg>
`;
    const target = new URL(`sculpture-${variant}.webp`, destination);
    await sharp(Buffer.from(svg)).webp({quality:88}).toFile(fileURLToPath(target));
    console.log(`${fileURLToPath(target)} — ${strands.length} strands, optimized WebP`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await generatePosters();
