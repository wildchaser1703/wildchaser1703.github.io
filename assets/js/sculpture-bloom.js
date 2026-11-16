// Small, transparent-background bloom: render the sculpture once, blur its brighter
// highlights at reduced resolution, then composite beneath the crisp original.
// Uses RGBA8 targets, so it needs no floating-point framebuffer extensions.
export function createBloom(three, renderer) {
  const options = { depthBuffer: false, stencilBuffer: false };
  const source = new three.WebGLRenderTarget(1, 1, { ...options, depthBuffer: true, samples: 2 });
  const horizontal = new three.WebGLRenderTarget(1, 1, options);
  const vertical = new three.WebGLRenderTarget(1, 1, options);
  const vertexShader = `
    varying vec2 uvBloom;
    void main() {
      uvBloom = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `;
  const blur = new three.ShaderMaterial({
    uniforms: {
      image: { value: source.texture },
      direction: { value: new three.Vector2() },
      extract: { value: true },
    },
    vertexShader,
    fragmentShader: `
      uniform sampler2D image;
      uniform vec2 direction;
      uniform bool extract;
      varying vec2 uvBloom;
      vec4 inkAt(vec2 point) {
        vec4 ink = texture2D(image, point);
        if (extract) {
          float brightness = max(ink.r, max(ink.g, ink.b)) / max(ink.a, 0.0001);
          ink *= smoothstep(0.45, 0.85, brightness);
        }
        return ink;
      }
      void main() {
        vec4 glow = inkAt(uvBloom) * 0.227027;
        glow += inkAt(uvBloom + direction * 1.384615) * 0.316216;
        glow += inkAt(uvBloom - direction * 1.384615) * 0.316216;
        glow += inkAt(uvBloom + direction * 3.230769) * 0.070270;
        glow += inkAt(uvBloom - direction * 3.230769) * 0.070270;
        gl_FragColor = glow;
      }
    `,
    depthTest: false, depthWrite: false, blending: three.NoBlending,
  });
  const composite = new three.ShaderMaterial({
    uniforms: {
      original: { value: source.texture },
      blurred: { value: vertical.texture },
      strength: { value: 0.65 },
    },
    vertexShader,
    fragmentShader: `
      uniform sampler2D original;
      uniform sampler2D blurred;
      uniform float strength;
      varying vec2 uvBloom;
      void main() {
        vec4 ink = texture2D(original, uvBloom);
        vec4 glow = texture2D(blurred, uvBloom) * strength;
        float alpha = ink.a + glow.a * (1.0 - ink.a);
        vec3 colour = ink.rgb + glow.rgb * (1.0 - ink.a * 0.8);
        // Targets contain premultiplied linear ink. Convert straight colour to
        // sRGB first, then premultiply for the transparent browser canvas.
        gl_FragColor = vec4(colour / max(alpha, 0.0001), alpha);
        #include <colorspace_fragment>
        gl_FragColor.rgb *= gl_FragColor.a;
      }
    `,
    depthTest: false, depthWrite: false, blending: three.NoBlending,
  });
  const geometry = new three.PlaneGeometry(2, 2);
  const quad = new three.Mesh(geometry, blur);
  quad.frustumCulled = false;
  const screen = new three.Scene();
  screen.add(quad);
  const camera = new three.Camera();
  let blurWidth = 1, blurHeight = 1;

  return {
    resize(width, height, pixelRatio, mobile) {
      source.setSize(Math.round(width * pixelRatio), Math.round(height * pixelRatio));
      const scale = Math.min(0.5, (mobile ? 256 : 512) / Math.max(width, height));
      blurWidth = Math.max(1, Math.round(width * scale));
      blurHeight = Math.max(1, Math.round(height * scale));
      horizontal.setSize(blurWidth, blurHeight);
      vertical.setSize(blurWidth, blurHeight);
    },
    setTheme(dark) { composite.uniforms.strength.value = dark ? 0.28 : 0.12; },
    render(scene, sceneCamera) {
      renderer.setRenderTarget(source);
      renderer.render(scene, sceneCamera);
      quad.material = blur;
      blur.uniforms.image.value = source.texture;
      blur.uniforms.extract.value = true;
      blur.uniforms.direction.value.set(2.5 / blurWidth, 0);
      renderer.setRenderTarget(horizontal);
      renderer.render(screen, camera);
      blur.uniforms.image.value = horizontal.texture;
      blur.uniforms.extract.value = false;
      blur.uniforms.direction.value.set(0, 2.5 / blurHeight);
      renderer.setRenderTarget(vertical);
      renderer.render(screen, camera);
      quad.material = composite;
      renderer.setRenderTarget(null);
      renderer.render(screen, camera);
    },
    dispose() {
      source.dispose(); horizontal.dispose(); vertical.dispose();
      blur.dispose(); composite.dispose(); geometry.dispose();
    },
  };
}
