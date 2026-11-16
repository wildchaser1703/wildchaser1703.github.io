import assert from 'node:assert/strict';
import {createMeshData, FRAME} from '../assets/js/sculpture-shape.js';
// Exercise every fold family and varied seeds: deformation must keep the mesh
// in frame, normals usable for lighting, and editions structurally different.
for(let seed=0;seed<160;seed++) {
  const {positions,normals}=createMeshData(seed*7919);
  for(let i=0;i<positions.length;i+=3) {
    assert.ok(Number.isFinite(positions[i+2]));
    assert.ok(Math.abs(positions[i])<FRAME.width/2-.2);
    assert.ok(Math.abs(positions[i+1])<FRAME.height/2-.2);
    assert.ok(Math.abs(Math.hypot(normals[i],normals[i+1],normals[i+2])-1)<.001);
  }
}
for(let a=0;a<4;a++) for(let b=a+1;b<4;b++) {
  const p=createMeshData(a).positions,q=createMeshData(b).positions;
  const rms=Math.sqrt(p.reduce((sum,v,i)=>sum+(v-q[i])**2,0)/(p.length/3));
  assert.ok(rms>.35,`Fold families ${a} and ${b} are too similar: ${rms}`);
}
console.log('160 sculptural editions stay in frame with unit normals; all four fold families are distinct.');
