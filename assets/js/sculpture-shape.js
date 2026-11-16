/** Marginalia: the evaluated Blender folios, shared by WebGL and static editions. */
import model from './sculpture-model.js';
export const BACKGROUND = '#f8f6f0';
export const FRAME = Object.freeze({width:6.4,height:5.4});
export const VARIANT_COUNT = 3;
export const normalizeVariant = value => Number.isFinite(Number(value)) ? Math.trunc(Number(value)) >>> 0 : 0;
// Independent continuous parameters, not a catalogue of preset poses.
function edition(seed) {
  let n=normalizeVariant(seed);
  const random=()=>{
    n=(n+0x6d2b79f5)>>>0;
    let t=Math.imul(n^(n>>>15),1|n);
    t^=t+Math.imul(t^(t>>>7),61|t);
    return ((t^(t>>>14))>>>0)/4294967296;
  };
  const range=(min,max)=>min+(max-min)*random();
  return {
    width:range(.78,1.35),height:range(.84,1.12),fan:range(.04,.56),
    depth:range(1.05,2.25),twist:range(-1.05,1.05),bow:range(-.5,.5),
    yaw:range(-1,1),tilt:range(-.5,.5),curl:range(-.35,.35),splay:range(-.24,.24),
  };
}
function transform(points, layers, e) {
  const result=new Float32Array(points.length);
  for(let i=0;i<points.length;i+=3){
    const leaf=(typeof layers==='number'?layers:layers[i/3])-2;
    const t=Math.max(-1,Math.min(1,points[i+1]/2));
    let x=points[i]*e.width+leaf*e.fan*.42*(.3+Math.cos(t*1.2));
    let y=points[i+1]*e.height+leaf*e.splay*Math.sin(t*Math.PI)*.3;
    let z=points[i+2]*e.depth+e.bow*Math.sin(t*Math.PI)+leaf*.08*Math.sin(t*2)+e.curl*Math.sin(t*3+leaf*.35)*(1-t*t);
    const angle=e.twist*t+e.yaw, c=Math.cos(angle),s=Math.sin(angle);
    [x,z]=[x*c+z*s,-x*s+z*c];
    result[i]=x*Math.cos(e.tilt)-y*Math.sin(e.tilt);
    result[i+1]=x*Math.sin(e.tilt)+y*Math.cos(e.tilt);
    result[i+2]=z;
  }
  return result;
}
const palette=[[176,48,24],[206,71,26],[227,104,35],[239,150,62],[248,196,112]];
const cache=new Map();
function geometry(seed) {
  seed=normalizeVariant(seed);
  if(cache.has(seed)) return cache.get(seed);
  const pose=edition(seed), positions=transform(model.positions,model.layers,pose);
  // Leave breathing/parallax room inside the shared poster and camera frame.
  let extentX=0,extentY=0;
  for(let i=0;i<positions.length;i+=3){extentX=Math.max(extentX,Math.abs(positions[i]));extentY=Math.max(extentY,Math.abs(positions[i+1]));}
  const scale=Math.min(1,2.7/extentX,2.25/extentY);
  for(let i=0;i<positions.length;i++) positions[i]*=scale;
  // Recompute smooth normals after the nonlinear fold deformation; rotating the
  // original normals would flatten or incorrectly light the new creases.
  const normals=new Float32Array(positions.length);
  for(let i=0;i<model.indices.length;i+=3){
    const a=model.indices[i]*3,b=model.indices[i+1]*3,c=model.indices[i+2]*3;
    const ux=positions[b]-positions[a],uy=positions[b+1]-positions[a+1],uz=positions[b+2]-positions[a+2];
    const vx=positions[c]-positions[a],vy=positions[c+1]-positions[a+1],vz=positions[c+2]-positions[a+2];
    const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
    for(const index of [a,b,c]){normals[index]+=nx;normals[index+1]+=ny;normals[index+2]+=nz;}
  }
  for(let i=0;i<normals.length;i+=3){const length=Math.hypot(normals[i],normals[i+1],normals[i+2])||1;normals[i]/=length;normals[i+1]/=length;normals[i+2]/=length;}
  const result={positions,normals,pose,scale};
  if(cache.size>=4) cache.delete(cache.keys().next().value);
  cache.set(seed,result);return result;
}
/** Screen-space displacement stops "new" editions from looking almost identical. */
export function variationDistance(a,b) {
  const p=geometry(a).positions,q=geometry(b).positions;
  let sum=0,count=0;
  for(let i=0;i<p.length;i+=39){sum+=(p[i]-q[i])**2+(p[i+1]-q[i+1])**2;count++;}
  return Math.sqrt(sum/count);
}
export function nextVariation(current,randomSeed,recent=[]) {
  let best=current,bestDistance=-1;
  // Bounded search, also works if a random source repeats the same seed.
  for(let i=0;i<8;i++) {
    const candidate=normalizeVariant(randomSeed()+Math.imul(current+1,0x85ebca6b)+Math.imul(i,0x9e3779b9));
    if(candidate===current || recent.includes(candidate)) continue;
    const distance=variationDistance(current,candidate);
    if(distance>bestDistance){best=candidate;bestDistance=distance;}
    if(distance>=.32) return candidate;
  }
  if(best===current) {
    // Even a pathological source cannot trap the control on the current seed.
    best=normalizeVariant(current+1);
    while(recent.includes(best) || best===current) best=normalizeVariant(best+1);
  }
  return best;
}
export function createSculpture(seed=0) {
  const {pose,scale}=geometry(seed);
  return model.guides.map(guide=>({points:transform(guide.points,guide.layer,pose).map(v=>v*scale),color:palette[guide.layer],opacity:.7}));
}
export function createMeshData(seed=0) {
  const {positions,normals}=geometry(seed);
  return {positions,normals,indices:model.indices,
    colors:new Float32Array(model.layers.flatMap(layer=>palette[layer].map(c=>c/255)))};
}
/** Orthographic painter's edition: no GPU or external asset needed for fallback. */
export function createPosterSvg(seed=0) {
  const {positions:p,normals:n,indices}=createMeshData(seed),faces=[];
  const project=i=>`${(450+p[i]*140).toFixed(1)},${(380-p[i+1]*140).toFixed(1)}`;
  for(let i=0;i<indices.length;i+=3){
    const a=indices[i]*3,b=indices[i+1]*3,c=indices[i+2]*3;
    const light=.48+.52*Math.abs((n[a]+n[b]+n[c])*.15+(n[a+1]+n[b+1]+n[c+1])*.2+(n[a+2]+n[b+2]+n[c+2])*.2);
    const color=palette[model.layers[indices[i]]].map(v=>Math.min(255,Math.round(v*light)));
    faces.push({z:p[a+2]+p[b+2]+p[c+2],svg:`<path d="M${project(a)}L${project(b)}L${project(c)}Z" fill="rgb(${color})" stroke="rgb(${color})" stroke-width=".35"/>`});
  }
  faces.sort((a,b)=>a.z-b.z);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="900" height="760" viewBox="0 0 900 760"><title>Marginalia — folded folios in ochre and vermilion</title>${faces.map(f=>f.svg).join('')}</svg>`;
}
