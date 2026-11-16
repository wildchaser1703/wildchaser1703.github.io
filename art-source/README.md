# Marginalia

An original sculpture for Toshali's homepage: five folded folios along an open S-shaped spine. The open leaves reference writing and the spaces between them suggest independent paths held together by a system. The warm palette belongs to the existing site.

`marginalia.blend` is the editable Blender 4.3 source: five named mesh objects with live Solidify modifiers, materials, a camera and studio lights. `marginalia.py` recreates it and exports the evaluated Blender meshes into `assets/js/sculpture-model.js`.

From the repository root:

```sh
blender -b --python art-source/marginalia.py
npm run build
npm test
npm run dev
```

Blender is needed only to edit/regenerate the artwork, not for normal site builds or CI. The exported model is 8,030 vertices / 16,040 triangles, rendered as one indexed mesh. No remote model host, texture downloads or additional runtime dependencies. The model module is about 694 KiB uncompressed / 229 KiB gzip. Blender sources are not copied to the deployed site.

Three.js supplies small, continuous deformations, eased pointer parallax, moving shader lighting, fine ink traces and a restrained bloom. New variations sample ten continuous deformation parameters from a 32-bit seed (4,294,967,296 possible seeds), rather than choosing a preset family. A bounded search prefers visibly different projected geometry, and a bounded recent-seed list avoids repeats from the last 32 editions. Surface normals are recomputed after deformation so the lighting follows the folds. Touch scroll is native. Motion pauses when offscreen, hidden or explicitly paused; reduced motion and GPU failure use a static edition derived from the same geometry. Print and no-JavaScript use a build-time WebP poster.

The shader intentionally uses a stylized material rather than attempting to reproduce Blender's studio render exactly. Geometry, normals and fold thickness originate in Blender; lighting and deformation happen in the browser.
