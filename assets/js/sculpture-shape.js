/** Marginalia: the evaluated Blender folios, shared by WebGL and static editions. */
import model from './sculpture-model.js';
export const BACKGROUND = '#f8f6f0';
export const FRAME = Object.freeze({width:6.4,height:5.4});
export const VARIANT_COUNT = 3;
export const normalizeVariant = value => Number.isFinite(Number(value)) ? Math.trunc(Number(value)) >>> 0 : 0;
// Four distinct poses retain the authored folios while changing their volume.
const poses = [
  {width:1.15, height:.94, fan:.34, depth:1.65, twist:.38, bow:.12, yaw:-.55, tilt:.12},
  {width:.92, height:1.08, fan:.10, depth:1.95, twist:-.60, bow:.35, yaw:.62, tilt:-.24},
  {width:1.22, height:.88, fan:.46, depth:1.35, twist:.72, bow:-.26, yaw:.28, tilt:.34},
  {width:.95, height:1.02, fan:.25, depth:2.1, twist:-.28, bow:-.38, yaw:-.78, tilt:-.18},
];
function edition(seed) {
  let n=normalizeVariant(seed);
  const random=()=>{n=(Math.imul(n,1664525)+1013904223)>>>0;return n/4294967296;};
  const pose=poses[n % poses.length];
  return {...pose, yaw:pose.yaw+(random()-.5)*.28, twist:pose.twist+(random()-.5)*.18};
}
function transform(points, layers, e) {
  const result=new Float32Array(points.length);
  for(let i=0;i<points.length;i+=3){
    const leaf=(typeof layers==='number'?layers:layers[i/3])-2;
    const t=Math.max(-1,Math.min(1,points[i+1]/2));
    let x=points[i]*e.width+leaf*e.fan*.42*(.3+Math.cos(t*1.2));
    let y=points[i+1]*e.height;
    let z=points[i+2]*e.depth+e.bow*Math.sin(t*Math.PI)+leaf*.08*Math.sin(t*2);
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
