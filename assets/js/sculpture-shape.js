/** Marginalia: the evaluated Blender folios, shared by WebGL and static editions. */
import model from './sculpture-model.js';
export const BACKGROUND = '#f8f6f0';
export const FRAME = Object.freeze({width:6.4,height:5.4});
export const VARIANT_COUNT = 3;
export const normalizeVariant = value => Number.isFinite(Number(value)) ? Math.trunc(Number(value)) >>> 0 : 0;
function edition(seed) {
  let n=normalizeVariant(seed);
  const random=()=>{n=(Math.imul(n,1664525)+1013904223)>>>0;return n/4294967296;};
  return {angle:(random()-.5)*.32, width:.92+random()*.16};
}
function transform(points,seed,normal=false) {
  const e=edition(seed),c=Math.cos(e.angle),s=Math.sin(e.angle),result=new Float32Array(points.length);
  for(let i=0;i<points.length;i+=3){
    const x=points[i]*(normal?1/e.width:e.width), y=points[i+1],z=points[i+2];
    result[i]=x*c-y*s;result[i+1]=x*s+y*c;result[i+2]=z;
  }
  return result;
}
const palette=[[176,48,24],[206,71,26],[227,104,35],[239,150,62],[248,196,112]];
export function createSculpture(seed=0) {
  return model.guides.map(guide=>({points:transform(guide.points,seed),color:palette[guide.layer],opacity:.7}));
}
export function createMeshData(seed=0) {
  return {positions:transform(model.positions,seed),normals:transform(model.normals,seed,true),indices:model.indices,
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
