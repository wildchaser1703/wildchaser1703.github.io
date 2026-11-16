import assert from 'node:assert/strict';
import {createMeshData, FRAME, nextVariation, variationDistance} from '../assets/js/sculpture-shape.js';
const fingerprints=new Set();
for(let index=0;index<512;index++) {
  const seed=index<256?index:(Math.imul(index,2654435761)>>>0);
  const {positions,normals}=createMeshData(seed);
  fingerprints.add(Array.from(positions.filter((_,i)=>i%123===0)).join(','));
  for(let i=0;i<positions.length;i+=3) {
    assert.ok(Number.isFinite(positions[i+2]));
    assert.ok(Math.abs(positions[i])<FRAME.width/2-.2);
    assert.ok(Math.abs(positions[i+1])<FRAME.height/2-.2);
    assert.ok(Math.abs(Math.hypot(normals[i],normals[i+1],normals[i+2])-1)<.001);
  }
}
assert.equal(fingerprints.size,512,'Distinct seeds must produce distinct sampled geometry');
assert.deepEqual(createMeshData(123456).positions,createMeshData(123456).positions);
let current=0;const recent=[current];
for(let i=0;i<64;i++) {
  let draws=0;
  const next=nextVariation(current,()=>{draws++;return 0;},recent);
  assert.ok(draws<=8);assert.ok(!recent.includes(next));
  assert.ok(variationDistance(current,next)>.2);
  recent.push(next);if(recent.length>32)recent.shift();current=next;
}
console.log('512 distinct procedural editions: finite geometry, safe framing and unit normals; bounded selection and recent-repeat avoidance passed.');
