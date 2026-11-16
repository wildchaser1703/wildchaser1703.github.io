/**
 * Woven field — an original, seeded woven ribbon. No DOM or Three dependency.
 * Coordinates are already oriented for an orthographic camera looking down -Z.
 * Both the live sculpture and its SVG editions consume these exact strands.
 */
export const BACKGROUND = '#f8f6f0';
export const VARIANT_COUNT = 3;
export const FRAME = Object.freeze({ width: 6.4, height: 5.4 });
export const QUALITY = Object.freeze({
  preview: { warp: 100, weft: 180, around: 80, across: 20 },
  desktop: { warp: 300, weft: 660, around: 240, across: 40 },
  mobile: { warp: 180, weft: 420, around: 160, across: 28 },
});

const TAU = Math.PI * 2;
const editions = [
  { seed: 1703, phase: 0.2, oval: 1.08, lift: 0.64, twist: 0.85, rx: -0.46, ry: -0.28, rz: -0.48 },
  { seed: 4817, phase: 1.25, oval: 0.92, lift: 0.78, twist: 1.04, rx: -0.58, ry: 0.32, rz: 0.57 },
  { seed: 9029, phase: -0.75, oval: 1.17, lift: 0.53, twist: 0.72, rx: -0.38, ry: -0.42, rz: -0.85 },
];

function randomGenerator(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let n = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    n = (n + Math.imul(n ^ (n >>> 7), 61 | n)) ^ n;
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
}

export function normalizeVariant(variant = 0) {
  const value = Number(variant);
  return Number.isFinite(value) ? Math.trunc(value) >>> 0 : 0;
}

const editionCache = new Map();
function edition(seed) {
  seed = normalizeVariant(seed);
  if (seed < editions.length) return editions[seed];
  if (!editionCache.has(seed)) {
    const random = randomGenerator(seed);
    if (editionCache.size > 16) editionCache.clear();
    editionCache.set(seed, {seed,phase:random()*TAU,oval:.85+random()*.35,lift:.35+random()*.55,twist:.5+random()*.7,rx:-.7+random()*.5,ry:-.5+random(),rz:random()*TAU});
  }
  return editionCache.get(seed);
}

/** Point on a folded annulus; v runs from inner to outer selvedge. */
export function surfacePoint(u, v, variant = 0, weave = 0) {
  const e = edition(variant);
  const q = u + e.phase;
  const radius = 1.34 + 0.15 * Math.cos(3 * q + 0.6) + 0.1 * Math.sin(2 * q);
  const width = 0.57 + 0.16 * Math.sin(q - 0.8) + 0.075 * Math.cos(2 * q + 1);
  const twist = 0.34 + e.twist * Math.sin(q - 0.5) + 0.3 * Math.sin(2 * q + 0.7);
  const radial = radius + width * v * Math.cos(twist) + 0.075 * v * v * Math.sin(3 * q);
  let x = e.oval * radial * Math.cos(u) + 0.12 * Math.sin(2 * q);
  let y = radial * Math.sin(u) + 0.08 * Math.cos(3 * q);
  let z = e.lift * Math.sin(2 * q - 0.45) + 0.21 * Math.cos(q + 0.4)
    + width * v * Math.sin(twist) + 0.19 * v * v * Math.cos(q - 0.6) + weave;
  // Fixed camera orientation, shared with the poster projection.
  [y, z] = [y * Math.cos(e.rx) - z * Math.sin(e.rx), y * Math.sin(e.rx) + z * Math.cos(e.rx)];
  [x, z] = [x * Math.cos(e.ry) + z * Math.sin(e.ry), -x * Math.sin(e.ry) + z * Math.cos(e.ry)];
  [x, y] = [x * Math.cos(e.rz) - y * Math.sin(e.rz), x * Math.sin(e.rz) + y * Math.cos(e.rz)];
  return [x, y, z];
}

const palette = [[238, 183, 32], [231, 151, 20], [224, 105, 27], [209, 64, 38], [169, 45, 38]];
function threadColor(heat) {
  const t = Math.max(0, Math.min(0.9999, heat)) * (palette.length - 1);
  const index = Math.floor(t);
  return palette[index].map((channel, i) => Math.round(channel + (palette[index + 1][i] - channel) * (t - index)));
}

/**
 * 960 real continuous strands on desktop, 600 on small screens.
 * Warp follows the loop; weft crosses it on a slight bias. Minute alternating
 * normal offsets suggest over/under crossings without tubes or a solid mesh.
 */
export function createSculpture(variant = 0, quality = 'desktop') {
  variant = normalizeVariant(variant);
  const { warp, weft, around, across } = QUALITY[quality] || QUALITY.desktop;
  const shape = edition(variant);
  const random = randomGenerator(shape.seed);
  const strands = [];
  for (let family = 0; family < 2; family++) {
    const count = family === 0 ? warp : weft;
    const segments = family === 0 ? around : across;
    for (let index = 0; index < count; index++) {
      const fraction = (index + 0.5 + (random() - 0.5) * 0.28) / count;
      const phase = random() * TAU;
      const points = new Float32Array((segments + 1) * 3);
      let depth = 0;
      for (let step = 0; step <= segments; step++) {
        const t = step / segments;
        const u = family === 0 ? t * TAU : fraction * TAU + 0.12 * Math.sin(t * Math.PI - 0.5);
        const v = family === 0 ? fraction * 2 - 1 + 0.002 * Math.sin(5 * u + phase) : t * 2 - 1;
        const weave = 0.003 * Math.sin(family === 0 ? u * weft / 2 : v * warp * Math.PI / 2) * (family === 0 ? 1 : -1);
        const point = surfacePoint(u, v, variant, weave);
        points.set(point, step * 3);
        depth += point[2];
      }
      const heat = family === 0
        ? 0.1 + 0.72 * fraction + (random() - 0.5) * 0.15
        : 0.37 + 0.3 * Math.sin(fraction * TAU + shape.phase) + random() * 0.16;
      strands.push({ points, color: threadColor(heat), opacity: family === 0 ? 0.52 : 0.64, family, depth: depth / (segments + 1) });
    }
  }
  // Back-to-front order is shared by SVG and the transparent GPU line batch.
  strands.sort((a, b) => a.depth - b.depth);
  return strands;
}

/** Expand polylines to one GPU LineSegments batch; RGB stays in sRGB here. */
export function createLineData(strands) {
  const vertices = strands.reduce((sum, strand) => sum + (strand.points.length / 3 - 1) * 2, 0);
  const positions = new Float32Array(vertices * 3);
  const colors = new Float32Array(vertices * 3);
  const opacities = new Float32Array(vertices);
  let offset = 0;
  for (const strand of strands) {
    for (let point = 3; point < strand.points.length; point += 3) {
      for (const source of [point - 3, point]) {
        positions.set(strand.points.subarray(source, source + 3), offset * 3);
        colors.set(strand.color.map(channel => channel / 255), offset * 3);
        opacities[offset++] = strand.opacity;
      }
    }
  }
  return { positions, colors, opacities };
}
