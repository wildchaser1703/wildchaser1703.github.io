"""Blender source for Marginalia. Rebuild: blender -b --python art-source/marginalia.py
Five open folios follow an S-shaped spine. Export evaluated Blender geometry;
the website needs neither Blender nor a model-loader dependency at build time.
"""
import bpy, math, json
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
positions=[]; normals=[]; indices=[]; layers=[]; guides=[]
N=72; M=10
for layer in range(5):
    verts=[]; faces=[]
    for i in range(N+1):
        t=i/N
        # Each leaf has its own open end, but shares the same flowing spine.
        y=(t-.5)*3.65
        spine=.83*math.sin((t-.5)*math.tau)
        twist=1.05*math.sin(t*math.pi*2-.6)+.32
        width=.16+.62*math.sin(math.pi*t)**.65
        fan=(layer-2)*.19
        for j in range(M+1):
            v=j/M*2-1
            x=spine+v*width*math.cos(twist)+fan*(.4+math.sin(math.pi*t))
            yy=y+v*.23*math.sin(twist)+(layer-2)*.055*math.cos(math.pi*t)
            z=.44*math.cos(t*math.tau)+v*width*math.sin(twist)+fan*.78
            # Fine ribbing catches the light without a texture download.
            z+=.013*math.cos(v*math.pi*5)*math.sin(math.pi*t)
            verts.append((x,yy,z))
    for i in range(N):
        for j in range(M):
            a=i*(M+1)+j; faces.append((a,a+1,a+M+2,a+M+1))
    mesh=bpy.data.meshes.new('Folio mesh');mesh.from_pydata(verts,[],faces);mesh.update()
    obj=bpy.data.objects.new('Folio %02d'%(layer+1),mesh);bpy.context.collection.objects.link(obj)
    mat=bpy.data.materials.new('Ochre paper %02d'%layer);mat.diffuse_color=(.95,.24+layer*.09,.055,1);mat.metallic=.35;mat.roughness=.32;obj.data.materials.append(mat)
    for poly in mesh.polygons: poly.use_smooth=True
    modifier=obj.modifiers.new('Paper edge','SOLIDIFY');modifier.thickness=.018;modifier.offset=0
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());surface=evaluated.to_mesh();surface.calc_loop_triangles()
    offset=len(positions)//3
    for vertex in surface.vertices:
        positions.extend(round(v,5) for v in vertex.co);normals.extend(round(v,5) for v in vertex.normal);layers.append(layer)
    for tri in surface.loop_triangles:indices.extend(offset+i for i in tri.vertices)
    evaluated.to_mesh_clear()
    # Sparse fine contours for the static ink edition and the live signal paths.
    for j in range(0,M+1,2):guides.append({'layer':layer,'points':[round(c,5) for i in range(N+1) for c in verts[i*(M+1)+j]]})
# Source studio, for editing/inspection in Blender.
bpy.ops.object.camera_add(location=(0,0,9));camera=bpy.context.object;camera.name='Front orthographic';camera.rotation_euler=(0,0,0);camera.data.type='ORTHO';camera.data.ortho_scale=5.4;bpy.context.scene.camera=camera
for name,location,power,color in [('Key',(3,4,5),450,(1,.72,.4)),('Rim',(-3,1,3),350,(1,.22,.06))]:
    bpy.ops.object.light_add(type='AREA',location=location);light=bpy.context.object;light.name=name;light.data.energy=power;light.data.color=color;light.data.shape='DISK';light.data.size=4;light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler()
bpy.context.scene.render.film_transparent=True
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'art-source/marginalia.blend'))
data={'positions':positions,'normals':normals,'indices':indices,'layers':layers,'guides':guides}
(ROOT/'assets/js/sculpture-model.js').write_text('// Generated from art-source/marginalia.blend by marginalia.py; do not hand-edit.\nexport default '+json.dumps(data,separators=(',',':'))+';\n')
print('Exported',len(positions)//3,'vertices and',len(indices)//3,'triangles')
