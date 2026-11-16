import { mkdir } from 'node:fs/promises';
import sharp from 'sharp';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { VARIANT_COUNT, createPosterSvg } from '../assets/js/sculpture-shape.js';
export async function generatePosters(destination = new URL('../assets/img/', import.meta.url)) {
  await mkdir(destination,{recursive:true});
  for(let variant=0;variant<VARIANT_COUNT;variant++) {
    const target=new URL(`sculpture-${variant}.webp`,destination);
    await sharp(Buffer.from(createPosterSvg(variant))).webp({quality:88}).toFile(fileURLToPath(target));
    console.log(`${fileURLToPath(target)} — Blender folio, optimized WebP`);
  }
}
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) await generatePosters();
