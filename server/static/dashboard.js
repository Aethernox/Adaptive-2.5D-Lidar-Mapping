/* 
 * High-Performance GPU WebGL 3D Perception Workstation.
 * Features:
 *   - Solid 3D Polyhedral Ego Vehicle with headlights, glass canopy, and chassis glow.
 *   - 3D Planned Collision-Avoidance Trajectory Ribbon on the roadway.
 *   - Soft Alpha-Splatting Shader merging high-density points into solid surfaces.
 *   - Multi-Street Network Navigation, Obstacle Avoidance HUD, and 3D Bounding Boxes.
 */
(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const $$ = s => [...document.querySelectorAll(s)];

  const canvas = $('lidarCanvas');
  const overlay = $('sceneOverlay');
  const mapImg = $('mapImg');
  const mapOverlay = $('mapOverlay');
  const mapWrap = $('mapWrap');
  const mapHoverHud = $('mapHoverHud');

  const gl = canvas.getContext('webgl', {
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: false
  });

  const state = {
    frame: null,
    history: [],
    cache: [],
    cacheIndex: -1,
    playing: true,
    busy: false,
    mode: 'prediction',
    pointSize: 5,
    temporal: 1,
    showGrid: true,
    trails: true,
    scanBeam: true,
    scanAngle: 0,
    selectedId: null,
    lockedCamera: false,
    chart: 'fps',
    camera: 'ego',
    yaw: -0.85,
    pitch: 0.48,
    distance: 54,
    target: [10, 0, 0],
    pan: [0, 0, 0],
    drag: null,
    lastMvp: null,
    classVisible: new Set([0, 1, 2, 3, 4, 5]),
    staticVisible: true,
    dynamicVisible: true,
    ranges: [[0, 10], [10, 25], [25, 50], [50, 100]],
    density: 1,
    speed: 1.0,
    graph: { fps: [], latency: [], points: [], objects: [] }
  };

  // Vivid Semantic Class Palette (RGB in [0, 1])
  const semantic = [
    [0.18, 0.74, 0.38], // 0: Drivable terrain (Emerald Green)
    [0.76, 0.66, 0.28], // 1: Non-drivable / curb / sidewalk (Gold/Amber)
    [0.88, 0.22, 0.28], // 2: Static Structure / Building (Ruby Crimson)
    [1.00, 0.60, 0.10], // 3: Static Pole / Cone (Bright Orange)
    [1.00, 0.16, 0.68], // 4: Pedestrian / Cyclist (Neon Magenta)
    [0.00, 0.88, 1.00], // 5: Vehicle / Truck (Electric Cyan)
  ];

  const dynamicIds = new Set([4, 5]);
  let program, cloudBuffer, lineBuffer, solidCarBuffer, ribbonBuffer, scanBuffer;
  let pointCount = 0, lineCount = 0, solidCarVertexCount = 0, ribbonVertexCount = 0;

  // Shader sources
  const shaders = {
    vert: `
      attribute vec3 p;
      attribute vec3 c;
      attribute float q;
      attribute float a;
      attribute float intensity;
      
      uniform mat4 mvp;
      uniform float pointSize;
      
      varying vec4 vColor;
      varying float vAlpha;
      
      void main() {
        gl_Position = mvp * vec4(p, 1.0);
        float dist = max(8.0, gl_Position.w);
        gl_PointSize = (2.5 + q * pointSize) * (195.0 / dist);
        gl_PointSize = clamp(gl_PointSize, 2.0, 36.0);
        vColor = vec4(c, 1.0);
        vAlpha = a;
      }
    `,
    frag: `
      precision mediump float;
      varying vec4 vColor;
      varying float vAlpha;
      
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float distSq = dot(coord, coord);
        if (distSq > 0.25) discard;
        
        // Soft Gaussian alpha splatting for seamless solid surfaces
        float splat = 1.0 - smoothstep(0.04, 0.25, distSq);
        float core = 1.0 - smoothstep(0.01, 0.16, distSq);
        vec3 col = vColor.rgb * (0.85 + 0.35 * core);
        gl_FragColor = vec4(col, vAlpha * (0.5 * splat + 0.5 * core));
      }
    `,
    lineVert: `
      attribute vec3 p;
      attribute vec3 c;
      uniform mat4 mvp;
      varying vec3 vColor;
      void main() {
        gl_Position = mvp * vec4(p, 1.0);
        vColor = c;
      }
    `,
    lineFrag: `
      precision mediump float;
      varying vec3 vColor;
      void main() {
        gl_FragColor = vec4(vColor, 0.75);
      }
    `,
    meshVert: `
      attribute vec3 p;
      attribute vec3 c;
      attribute float a;
      uniform mat4 mvp;
      varying vec4 vColor;
      void main() {
        gl_Position = mvp * vec4(p, 1.0);
        vColor = vec4(c, a);
      }
    `,
    meshFrag: `
      precision mediump float;
      varying vec4 vColor;
      void main() {
        gl_FragColor = vColor;
      }
    `
  };

  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw Error(gl.getShaderInfoLog(s));
    }
    return s;
  };

  const makeProgram = (v, f) => {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, v));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, f));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw Error(gl.getProgramInfoLog(p));
    }
    return p;
  };

  function initGL() {
    if (!gl) {
      if ($('statusText')) $('statusText').textContent = 'WEBGL UNAVAILABLE';
      return;
    }
    program = makeProgram(shaders.vert, shaders.frag);
    program.lines = makeProgram(shaders.lineVert, shaders.lineFrag);
    program.mesh = makeProgram(shaders.meshVert, shaders.meshFrag);

    cloudBuffer = gl.createBuffer();
    lineBuffer = gl.createBuffer();
    solidCarBuffer = gl.createBuffer();
    ribbonBuffer = gl.createBuffer();
    scanBuffer = gl.createBuffer();

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    buildSceneLines();
    buildSolidEgoCar();
    requestAnimationFrame(render);
  }

  function buildSolidEgoCar() {
    // Construct a Solid 3D Polyhedral Autonomous Sports Sedan
    const v = [];
    const addTri = (p1, p2, p3, col, alpha = 0.96) => {
      v.push(...p1, ...col, alpha);
      v.push(...p2, ...col, alpha);
      v.push(...p3, ...col, alpha);
    };
    const addQuad = (p1, p2, p3, p4, col, alpha = 0.96) => {
      addTri(p1, p2, p3, col, alpha);
      addTri(p1, p3, p4, col, alpha);
    };

    // Palette for solid car
    const cBody = [0.0, 0.85, 1.0];        // Cyber Electric Cyan
    const cDark = [0.05, 0.12, 0.20];       // Dark Obsidian Aerokit
    const cGlass = [0.08, 0.25, 0.42];      // Panoramic Tinted Cockpit Glass
    const cHeadlight = [1.0, 0.95, 0.65];   // Projector Headlights
    const cTaillight = [1.0, 0.12, 0.25];   // Neon Tail Brake Bar
    const cWheel = [0.04, 0.08, 0.12];      // Solid Wheels
    const cRim = [0.0, 0.95, 1.0];          // Rim Accents

    // Key body vertices in ego coordinates (X fwd, Y left, Z up)
    const hoodFwd = [2.3, 0.0, 0.55];
    const hoodLeft = [2.2, 0.95, 0.62];
    const hoodRight = [2.2, -0.95, 0.62];
    const noseLowL = [2.4, 0.9, 0.15];
    const noseLowR = [2.4, -0.9, 0.15];
    const noseCenter = [2.45, 0.0, 0.2];

    const cowlL = [0.85, 0.9, 0.85];
    const cowlR = [0.85, -0.9, 0.85];
    const cowlCenter = [0.9, 0.0, 0.88];

    const roofFrontL = [0.25, 0.72, 1.48];
    const roofFrontR = [0.25, -0.72, 1.48];
    const roofRearL = [-1.35, 0.72, 1.46];
    const roofRearR = [-1.35, -0.72, 1.46];

    const trunkL = [-2.2, 0.9, 0.82];
    const trunkR = [-2.2, -0.9, 0.82];
    const trunkEnd = [-2.35, 0.0, 0.78];
    const rearLowL = [-2.3, 0.85, 0.18];
    const rearLowR = [-2.3, -0.85, 0.18];

    const sideLowL = [0.0, 0.95, 0.18];
    const sideLowR = [0.0, -0.95, 0.18];

    // 1. Front Bumper & Hood
    addTri(hoodFwd, hoodLeft, cowlL, cBody);
    addTri(hoodFwd, cowlL, cowlCenter, cBody);
    addTri(hoodFwd, cowlCenter, cowlR, cBody);
    addTri(hoodFwd, cowlR, hoodRight, cBody);

    // Front Grill & Bumper
    addQuad(noseLowL, hoodLeft, hoodFwd, noseCenter, cDark);
    addQuad(noseCenter, hoodFwd, hoodRight, noseLowR, cDark);

    // Projector Headlights
    addQuad([2.35, 0.65, 0.42], [2.25, 0.88, 0.55], [2.15, 0.85, 0.6], [2.25, 0.62, 0.48], cHeadlight, 1.0);
    addQuad([2.35, -0.65, 0.42], [2.25, -0.62, 0.48], [2.15, -0.85, 0.6], [2.25, -0.88, 0.55], cHeadlight, 1.0);

    // 2. Tinted Windshield Glass
    addQuad(cowlL, cowlR, roofFrontR, roofFrontL, cGlass, 0.85);

    // 3. Panoramic Roof
    addQuad(roofFrontL, roofFrontR, roofRearR, roofRearL, cDark, 0.98);

    // 4. Rear Window Glass
    addQuad(roofRearL, roofRearR, trunkR, trunkL, cGlass, 0.85);

    // 5. Rear Trunk Deck & Spoiler
    addTri(trunkEnd, trunkL, roofRearL, cBody);
    addTri(trunkEnd, roofRearL, roofRearR, cBody);
    addTri(trunkEnd, roofRearR, trunkR, cBody);

    // Rear Bumper & Neon Tail Light Bar
    addQuad(rearLowL, trunkL, trunkR, rearLowR, cDark);
    addQuad([-2.32, 0.85, 0.72], [-2.32, -0.85, 0.72], [-2.36, -0.85, 0.8], [-2.36, 0.85, 0.8], cTaillight, 1.0);

    // 6. Solid Side Body Panels & Doors
    addQuad(sideLowL, noseLowL, hoodLeft, cowlL, cBody);
    addQuad(sideLowL, cowlL, trunkL, rearLowL, cBody);
    addQuad(sideLowR, cowlR, hoodRight, noseLowR, cBody);
    addQuad(sideLowR, rearLowR, trunkR, cowlR, cBody);

    // Side Windows
    addTri(cowlL, roofFrontL, roofRearL, cGlass, 0.82);
    addTri(cowlL, roofRearL, trunkL, cGlass, 0.82);
    addTri(cowlR, roofRearR, roofFrontR, cGlass, 0.82);
    addTri(cowlR, trunkR, roofRearR, cGlass, 0.82);

    // 7. Solid 3D Wheels & Glowing Rims
    const wheelPositions = [
      [1.45, 0.96, 0.32],
      [1.45, -0.96, 0.32],
      [-1.45, 0.96, 0.32],
      [-1.45, -0.96, 0.32]
    ];
    const rW = 0.34, dW = 0.16;
    for (let [wx, wy, wz] of wheelPositions) {
      for (let i = 0; i < 12; i++) {
        let a1 = (i / 12) * Math.PI * 2, a2 = ((i + 1) / 12) * Math.PI * 2;
        let p1 = [wx + rW * Math.cos(a1), wy, wz + rW * Math.sin(a1)];
        let p2 = [wx + rW * Math.cos(a2), wy, wz + rW * Math.sin(a2)];
        let p1_in = [wx + rW * Math.cos(a1), wy - Math.sign(wy) * dW, wz + rW * Math.sin(a1)];
        let p2_in = [wx + rW * Math.cos(a2), wy - Math.sign(wy) * dW, wz + rW * Math.sin(a2)];

        addTri([wx, wy, wz], p1, p2, cRim, 1.0);
        addQuad(p1, p2, p2_in, p1_in, cWheel, 1.0);
      }
    }

    // 8. Roof Autonomous LiDAR Sensor Dome
    const lDome = [-0.5, 0.0, 1.58];
    const rD = 0.22, hD = 0.22;
    for (let i = 0; i < 12; i++) {
      let a1 = (i / 12) * Math.PI * 2, a2 = ((i + 1) / 12) * Math.PI * 2;
      let p1 = [lDome[0] + rD * Math.cos(a1), lDome[1] + rD * Math.sin(a1), lDome[2]];
      let p2 = [lDome[0] + rD * Math.cos(a2), lDome[1] + rD * Math.sin(a2), lDome[2]];
      let p1_top = [lDome[0] + rD * Math.cos(a1), lDome[1] + rD * Math.sin(a1), lDome[2] + hD];
      let p2_top = [lDome[0] + rD * Math.cos(a2), lDome[1] + rD * Math.sin(a2), lDome[2] + hD];

      addQuad(p1, p2, p2_top, p1_top, [0.0, 1.0, 0.85], 1.0);
      addTri([lDome[0], lDome[1], lDome[2] + hD], p1_top, p2_top, [0.0, 0.8, 1.0], 1.0);
    }

    // 9. Projected Headlight Cones on Road Ahead
    addTri([2.3, 0.7, 0.45], [14.0, 4.5, -0.02], [14.0, -1.5, -0.02], [1.0, 0.95, 0.6], 0.18);
    addTri([2.3, -0.7, 0.45], [14.0, 1.5, -0.02], [14.0, -4.5, -0.02], [1.0, 0.95, 0.6], 0.18);

    gl.bindBuffer(gl.ARRAY_BUFFER, solidCarBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
    solidCarVertexCount = v.length / 7;
  }

  function buildSceneLines() {
    const d = [];
    const add = (a, b, c = [0.12, 0.28, 0.38]) => d.push(...a, ...c, ...b, ...c);

    // 100m x 100m Ground grid
    for (let i = -100; i <= 100; i += 10) {
      const col = (i === 0) ? [0.25, 0.45, 0.6] : [0.08, 0.18, 0.26];
      add([i, -100, -0.06], [i, 100, -0.06], col);
      add([-100, i, -0.06], [100, i, -0.06], col);
    }

    // Polar range rings (10m, 25m, 50m, 100m)
    for (let r of [10, 25, 50, 100]) {
      const ringCol = r === 10 ? [0.0, 0.8, 1.0] : (r === 25 ? [0.0, 0.6, 0.85] : [0.08, 0.32, 0.42]);
      for (let i = 0; i < 90; i++) {
        let a = (i / 90) * Math.PI * 2, b = ((i + 1) / 90) * Math.PI * 2;
        add([r * Math.cos(a), r * Math.sin(a), -0.04], [r * Math.cos(b), r * Math.sin(b), -0.04], ringCol);
      }
    }

    // Coordinate Frame tripod at origin
    add([0, 0, 0], [6, 0, 0], [1.0, 0.2, 0.35]); // +X Forward
    add([0, 0, 0], [0, 6, 0], [0.2, 0.95, 0.55]); // +Y Left
    add([0, 0, 0], [0, 0, 5], [0.2, 0.6, 1.0]);  // +Z Up

    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(d), gl.STATIC_DRAW);
    lineCount = d.length / 6;
  }

  function updatePlannedTrajectoryRibbon(waypoints) {
    if (!waypoints || waypoints.length < 2) {
      ribbonVertexCount = 0;
      return;
    }

    const d = [];
    const ribbonHalfWidth = 0.95; // 1.9m wide corridor
    const cRibbon = [0.0, 0.95, 0.75];

    for (let i = 0; i < waypoints.length - 1; i++) {
      let p1 = waypoints[i];
      let p2 = waypoints[i + 1];
      let dx = p2[0] - p1[0], dy = p2[1] - p1[1];
      let l = Math.hypot(dx, dy) || 1;
      let nx = -dy / l * ribbonHalfWidth, ny = dx / l * ribbonHalfWidth;

      let alpha1 = Math.max(0.08, 0.45 * (1.0 - i / waypoints.length));
      let alpha2 = Math.max(0.08, 0.45 * (1.0 - (i + 1) / waypoints.length));

      // Left & right vertices for quad
      d.push(p1[0] + nx, p1[1] + ny, p1[2], ...cRibbon, alpha1);
      d.push(p1[0] - nx, p1[1] - ny, p1[2], ...cRibbon, alpha1);
      d.push(p2[0] + nx, p2[1] + ny, p2[2], ...cRibbon, alpha2);

      d.push(p1[0] - nx, p1[1] - ny, p1[2], ...cRibbon, alpha1);
      d.push(p2[0] - nx, p2[1] - ny, p2[2], ...cRibbon, alpha2);
      d.push(p2[0] + nx, p2[1] + ny, p2[2], ...cRibbon, alpha2);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, ribbonBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(d), gl.DYNAMIC_DRAW);
    ribbonVertexCount = d.length / 7;
  }

  // Linear algebra helpers
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const matPerspective = (f, a, n, z) => {
    let q = 1 / Math.tan(f / 2), nf = 1 / (n - z);
    return [q / a, 0, 0, 0, 0, q, 0, 0, 0, 0, (z + n) * nf, -1, 0, 0, 2 * z * n * nf, 0];
  };
  const norm = v => {
    let l = Math.hypot(...v) || 1;
    return v.map(x => x / l);
  };
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
  function lookAt(e, t) {
    let z = norm([e[0] - t[0], e[1] - t[1], e[2] - t[2]]);
    let x = norm(cross([0, 0, 1], z));
    if (Math.hypot(...x) < 0.1) x = [1, 0, 0];
    let y = cross(z, x);
    return [
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -x[0]*e[0] - x[1]*e[1] - x[2]*e[2],
      -y[0]*e[0] - y[1]*e[1] - y[2]*e[2],
      -z[0]*e[0] - z[1]*e[1] - z[2]*e[2], 1
    ];
  }
  function mul(a, b) {
    let o = Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
    }
    return o;
  }

  function camera() {
    let t = state.target.map((v, i) => v + state.pan[i]);
    let e;

    // Follow locked object if active
    if (state.lockedCamera && state.selectedId !== null) {
      const obj = (state.frame?.objects || []).find(o => o.id === state.selectedId);
      if (obj) {
        t = [obj.x, obj.y, (obj.z_max || 1.5) / 2];
      }
    }

    if (state.camera === 'top') {
      e = [t[0], t[1], 105];
    } else if (state.camera === 'front') {
      e = [-12, 0, 4.5];
      t = [65, 0, 1.2];
    } else if (state.camera === 'rear') {
      e = [25, 0, 5.5];
      t = [-55, 0, 1.2];
    } else {
      e = [
        t[0] + state.distance * Math.cos(state.pitch) * Math.cos(state.yaw),
        t[1] + state.distance * Math.cos(state.pitch) * Math.sin(state.yaw),
        t[2] + state.distance * Math.sin(state.pitch)
      ];
    }
    return { e, t };
  }

  function resize(c) {
    let r = Math.min(window.devicePixelRatio || 1, 2);
    let w = Math.max(1, Math.floor(c.clientWidth * r));
    let h = Math.max(1, Math.floor(c.clientHeight * r));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    return [w, h, r];
  }

  function render() {
    if (gl && program) {
      let [w, h] = resize(canvas);
      let cam = camera();
      let mvp = mul(matPerspective(1.02, w / h, 0.1, 400), lookAt(cam.e, cam.t));
      state.lastMvp = mvp;

      gl.viewport(0, 0, w, h);
      gl.clearColor(0.012, 0.028, 0.048, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // 1. Scene Lines (Grid, Coordinate tripod)
      gl.useProgram(program.lines);
      gl.uniformMatrix4fv(gl.getUniformLocation(program.lines, 'mvp'), false, new Float32Array(mvp));
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
      let lp = gl.getAttribLocation(program.lines, 'p');
      let lc = gl.getAttribLocation(program.lines, 'c');
      gl.enableVertexAttribArray(lp);
      gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(lc);
      gl.vertexAttribPointer(lc, 3, gl.FLOAT, false, 24, 12);
      gl.drawArrays(gl.LINES, 0, lineCount);

      // 2. Solid 3D Ego Vehicle Model
      gl.useProgram(program.mesh);
      gl.uniformMatrix4fv(gl.getUniformLocation(program.mesh, 'mvp'), false, new Float32Array(mvp));
      gl.bindBuffer(gl.ARRAY_BUFFER, solidCarBuffer);
      let mp = gl.getAttribLocation(program.mesh, 'p');
      let mc = gl.getAttribLocation(program.mesh, 'c');
      let ma = gl.getAttribLocation(program.mesh, 'a');
      gl.enableVertexAttribArray(mp);
      gl.vertexAttribPointer(mp, 3, gl.FLOAT, false, 28, 0);
      gl.enableVertexAttribArray(mc);
      gl.vertexAttribPointer(mc, 3, gl.FLOAT, false, 28, 12);
      gl.enableVertexAttribArray(ma);
      gl.vertexAttribPointer(ma, 1, gl.FLOAT, false, 28, 24);
      gl.drawArrays(gl.TRIANGLES, 0, solidCarVertexCount);

      // 3. 3D Planned Collision-Avoidance Trajectory Ribbon
      if (ribbonVertexCount > 0) {
        gl.bindBuffer(gl.ARRAY_BUFFER, ribbonBuffer);
        gl.vertexAttribPointer(mp, 3, gl.FLOAT, false, 28, 0);
        gl.vertexAttribPointer(mc, 3, gl.FLOAT, false, 28, 12);
        gl.vertexAttribPointer(ma, 1, gl.FLOAT, false, 28, 24);
        gl.drawArrays(gl.TRIANGLES, 0, ribbonVertexCount);
      }

      // 4. Animated 360° LiDAR Scan Beam Fan
      if (state.scanBeam) {
        state.scanAngle = (state.scanAngle + 0.045) % (Math.PI * 2);
        drawScanFan(mvp);
      }

      // 5. Point Cloud Points with Soft Alpha Splatting
      gl.useProgram(program);
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'mvp'), false, new Float32Array(mvp));
      gl.uniform1f(gl.getUniformLocation(program, 'pointSize'), state.pointSize);
      gl.bindBuffer(gl.ARRAY_BUFFER, cloudBuffer);

      let p = gl.getAttribLocation(program, 'p');
      let c = gl.getAttribLocation(program, 'c');
      let q = gl.getAttribLocation(program, 'q');
      let a = gl.getAttribLocation(program, 'a');
      let intensity = gl.getAttribLocation(program, 'intensity');

      gl.enableVertexAttribArray(p);
      gl.vertexAttribPointer(p, 3, gl.FLOAT, false, 36, 0);
      gl.enableVertexAttribArray(c);
      gl.vertexAttribPointer(c, 3, gl.FLOAT, false, 36, 12);
      gl.enableVertexAttribArray(q);
      gl.vertexAttribPointer(q, 1, gl.FLOAT, false, 36, 24);
      gl.enableVertexAttribArray(a);
      gl.vertexAttribPointer(a, 1, gl.FLOAT, false, 36, 28);
      gl.enableVertexAttribArray(intensity);
      gl.vertexAttribPointer(intensity, 1, gl.FLOAT, false, 36, 32);

      gl.drawArrays(gl.POINTS, 0, pointCount);

      // 6. Overlays & Canvas HUD
      drawOverlay(mvp, w, h);
      drawMap();
      drawChart();
    }
    requestAnimationFrame(render);
  }

  function drawScanFan(mvp) {
    const d = [];
    const nSteps = 24;
    const fanAngle = 0.55;
    const r = 85.0;
    const origin = [-0.5, 0.0, 1.72];

    for (let i = 0; i <= nSteps; i++) {
      let ang = state.scanAngle - fanAngle * (1.0 - i / nSteps);
      let x = r * Math.cos(ang), y = r * Math.sin(ang);
      let alpha = 0.35 * (i / nSteps);
      d.push(...origin, 0.0, 0.95, 1.0, alpha);
      d.push(x, y, -0.05, 0.0, 0.8, 1.0, 0.0);
    }

    gl.useProgram(program.mesh);
    gl.uniformMatrix4fv(gl.getUniformLocation(program.mesh, 'mvp'), false, new Float32Array(mvp));
    gl.bindBuffer(gl.ARRAY_BUFFER, scanBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(d), gl.DYNAMIC_DRAW);

    let p = gl.getAttribLocation(program.mesh, 'p');
    let c = gl.getAttribLocation(program.mesh, 'c');
    let a = gl.getAttribLocation(program.mesh, 'a');
    gl.enableVertexAttribArray(p);
    gl.vertexAttribPointer(p, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(c);
    gl.vertexAttribPointer(c, 3, gl.FLOAT, false, 28, 12);
    gl.enableVertexAttribArray(a);
    gl.vertexAttribPointer(a, 1, gl.FLOAT, false, 28, 24);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, d.length / 7);
  }

  function colorFor(v) {
    let x = v[0], y = v[1], z = v[2], cls = v[3], conf = v[4], intensity = v[5] ?? conf;
    let r = Math.hypot(x, y);

    if (state.mode === 'ground_truth') {
      return (v.length > 6 && v[6] >= 0) ? (semantic[v[6]] || [0.55, 0.55, 0.55]) : [0.22, 0.25, 0.28];
    }
    if (state.mode === 'error') {
      if (v.length > 6 && v[6] >= 0) {
        return cls === v[6] ? [0.15, 0.75, 0.45] : [1.0, 0.15, 0.25];
      }
      let c = clamp(conf, 0, 1);
      return [lerp(1.0, 0.1, c), lerp(0.2, 0.9, c), 0.3];
    }
    if (state.mode === 'intensity') {
      let t = clamp(intensity, 0, 1);
      if (t < 0.5) {
        return [lerp(0.05, 0.2, t * 2), lerp(0.3, 0.8, t * 2), lerp(0.9, 0.95, t * 2)];
      } else {
        return [lerp(0.2, 1.0, (t - 0.5) * 2), lerp(0.8, 0.95, (t - 0.5) * 2), lerp(0.95, 0.4, (t - 0.5) * 2)];
      }
    }
    if (state.mode === 'height') {
      let t = clamp((z + 1.2) / 5.5, 0, 1);
      return [
        clamp(1.5 - Math.abs(t * 4 - 3), 0, 1),
        clamp(1.5 - Math.abs(t * 4 - 2), 0, 1),
        clamp(1.5 - Math.abs(t * 4 - 1), 0, 1)
      ];
    }
    if (state.mode === 'distance') {
      let t = clamp(r / 100.0, 0, 1);
      return [lerp(0.0, 1.0, t), lerp(0.9, 0.2, t), lerp(1.0, 0.1, t)];
    }
    if (state.mode === 'dynamic') {
      if (dynamicIds.has(cls)) {
        return cls === 5 ? [0.0, 0.95, 1.0] : [1.0, 0.2, 0.8];
      }
      return [0.08, 0.14, 0.18];
    }
    if (state.mode === 'hazard') {
      if (r < 12.0) return [1.0, 0.15, 0.3];
      if (r < 25.0) return [1.0, 0.7, 0.0];
      return [0.15, 0.85, 0.4];
    }
    return semantic[cls] || [0.55, 0.55, 0.55];
  }

  function setCloud() {
    if (!gl || !state.frame) return;
    let d = [];
    let frames = state.history.slice(-state.temporal);
    let step = state.density;

    frames.forEach((f, fi) => {
      let age = (fi + 1) / frames.length;
      let pts = f.point_cloud || [];
      for (let i = 0; i < pts.length; i += step) {
        let v = pts[i];
        let cls = v[3];
        let r = Math.hypot(v[0], v[1]);

        if (!state.classVisible.has(cls)) continue;
        if (dynamicIds.has(cls) ? !state.dynamicVisible : !state.staticVisible) continue;
        if (!state.ranges.some(z => r >= z[0] && r < z[1])) continue;

        let col = colorFor(v);
        let alpha = (fi === frames.length - 1 ? 1.0 : (0.18 + 0.38 * age)) * (0.55 + 0.45 * (v[4] || 0.8));
        d.push(
          v[0], v[1], Math.max(-1.5, v[2]),
          col[0], col[1], col[2],
          0.85 + (v[4] || 0.5),
          alpha,
          v[5] ?? 0.5
        );
      }
    });

    pointCount = d.length / 9;
    gl.bindBuffer(gl.ARRAY_BUFFER, cloudBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(d), gl.DYNAMIC_DRAW);
    if ($('cloudMeta')) $('cloudMeta').textContent = `${pointCount.toLocaleString()} GPU POINTS · HIGH-DENSITY SCAN`;
  }

  function project(v, m, w, h) {
    let x = v[0], y = v[1], z = v[2];
    let qx = m[0]*x + m[4]*y + m[8]*z + m[12];
    let qy = m[1]*x + m[5]*y + m[9]*z + m[13];
    let qw = m[3]*x + m[7]*y + m[11]*z + m[15];
    if (qw <= 0.01) return null;
    return [w * (qx / qw * 0.5 + 0.5), h * (0.5 - qy / qw * 0.5), qw];
  }

  function drawOverlay(mvp, w, h) {
    let [ow, oh] = resize(overlay);
    let ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, ow, oh);
    ctx.scale(ow / w, oh / h);

    let objs = state.frame?.objects || [];

    // 1. Motion Trails
    if (state.trails) {
      for (let obj of objs) {
        if (!obj.is_dynamic) continue;
        let hist = state.history.map(f => (f.objects || []).find(o => o.id === obj.id)).filter(Boolean);
        if (hist.length > 1) {
          ctx.beginPath();
          hist.forEach((o, i) => {
            let p = project([o.x, o.y, (o.z_min ?? 0) + 0.3], mvp, w, h);
            if (!p) return;
            i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
          });
          ctx.strokeStyle = obj.kind === 'vehicle' ? 'rgba(0, 240, 255, 0.55)' : 'rgba(255, 42, 141, 0.55)';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    }

    // 2. 3D Bounding Boxes & HUD Labels
    objs.forEach(obj => {
      let z0 = obj.z_min ?? -0.2, z1 = obj.z_max ?? 1.6;
      let hw = Math.max(obj.w || 1.8, 0.45) / 2;
      let hd = Math.max(obj.d || 4.2, 0.45) / 2;

      let corners = [];
      for (let z of [z0, z1]) {
        for (let x of [-hw, hw]) {
          for (let y of [-hd, hd]) {
            corners.push(project([obj.x + x, obj.y + y, z], mvp, w, h));
          }
        }
      }

      if (corners.some(p => !p)) return;

      let edges = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
      ];

      let active = obj.id === state.selectedId;
      let strokeCol = active ? '#fff480' : (
        obj.kind === 'vehicle' ? '#00f0ff' : (
          obj.kind === 'pedestrian' ? '#ff2a8d' : (
            obj.kind === 'cyclist' ? '#ffb700' : (
              obj.kind === 'pole' ? '#ff9d00' : '#ff4d6d'
            )
          )
        )
      );

      ctx.fillStyle = active ? 'rgba(255, 244, 128, 0.16)' : (
        obj.kind === 'vehicle' ? 'rgba(0, 240, 255, 0.08)' : 'rgba(255, 42, 141, 0.08)'
      );
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      ctx.lineTo(corners[1][0], corners[1][1]);
      ctx.lineTo(corners[3][0], corners[3][1]);
      ctx.lineTo(corners[2][0], corners[2][1]);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = strokeCol;
      ctx.lineWidth = active ? 2.5 : 1.5;
      ctx.beginPath();
      edges.forEach(([a, b]) => {
        ctx.moveTo(corners[a][0], corners[a][1]);
        ctx.lineTo(corners[b][0], corners[b][1]);
      });
      ctx.stroke();

      let dist = obj.distance ?? Math.hypot(obj.x, obj.y);
      let vx = obj.vx || 0, vy = obj.vy || 0;
      let sp = Math.hypot(vx, vy);

      if (sp > 0.2) {
        let pStart = project([obj.x, obj.y, z0 + 0.4], mvp, w, h);
        let pEnd = project([obj.x + vx * 1.4, obj.y + vy * 1.4, z0 + 0.4], mvp, w, h);
        if (pStart && pEnd) {
          ctx.beginPath();
          ctx.moveTo(pStart[0], pStart[1]);
          ctx.lineTo(pEnd[0], pEnd[1]);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      let top = project([obj.x, obj.y, z1 + 0.35], mvp, w, h);
      if (top && top[2] > 0) {
        let tagText = `${obj.kind.toUpperCase()} #${String(obj.id).padStart(2, '0')}`;
        if (obj.is_dynamic && sp > 0.1) {
          tagText += ` · ${sp.toFixed(1)}m/s`;
        }
        tagText += ` · ${dist.toFixed(1)}m`;

        ctx.font = '10px "JetBrains Mono", Consolas, monospace';
        let textWidth = ctx.measureText(tagText).width;

        ctx.fillStyle = active ? 'rgba(255, 244, 128, 0.95)' : 'rgba(6, 16, 26, 0.88)';
        ctx.fillRect(top[0] - textWidth / 2 - 5, top[1] - 14, textWidth + 10, 16);
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = 1;
        ctx.strokeRect(top[0] - textWidth / 2 - 5, top[1] - 14, textWidth + 10, 16);

        ctx.fillStyle = active ? '#041018' : '#e6f4f8';
        ctx.fillText(tagText, top[0] - textWidth / 2, top[1] - 3);
      }
    });

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  function drawMap() {
    let f = state.frame;
    if (!f) return;
    let [w, h] = resize(mapOverlay);
    let ctx = mapOverlay.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    let scale = w / 200.0;
    const cv = (x, y) => [w / 2 + x * scale, h / 2 - y * scale];

    if (state.showGrid) {
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.28)';
      ctx.lineWidth = 1;
      for (let t of f.adaptive_grid?.tiers || []) {
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, t.r_max * scale, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    for (let cell of f.adaptive_grid?.cells || []) {
      let [x, y, , cls, conf, dr] = cell;
      let [px, py] = cv(x, y);
      let r = Math.max(1.2, dr * scale * 0.7);
      let col = semantic[cls] || [0.5, 0.5, 0.5];
      ctx.fillStyle = `rgba(${Math.round(col[0] * 255)}, ${Math.round(col[1] * 255)}, ${Math.round(col[2] * 255)}, ${Math.min(0.55, conf * 0.4)})`;
      ctx.fillRect(px - r, py - r, r * 2, r * 2);
    }

    // Planned trajectory line on map
    if (f.ego?.planned_path && f.ego.planned_path.length > 1) {
      ctx.beginPath();
      f.ego.planned_path.forEach((p, i) => {
        let [px, py] = cv(p[0], p[1]);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.strokeStyle = '#00ff9d';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    for (let o of f.objects || []) {
      let [x, y] = cv(o.x, o.y);
      let active = o.id === state.selectedId;
      ctx.strokeStyle = active ? '#fff7a0' : (
        o.kind === 'vehicle' ? '#00f0ff' : (
          o.kind === 'pedestrian' ? '#ff2a8d' : '#ffb700'
        )
      );
      ctx.lineWidth = active ? 2 : 1;
      let bw = Math.max(4, (o.w || 1.8) * scale);
      let bd = Math.max(4, (o.d || 4.2) * scale);
      ctx.strokeRect(x - bw / 2, y - bd / 2, bw, bd);

      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(`#${o.id}`, x + bw / 2 + 2, y + 3);
    }
  }

  function drawChart() {
    let cv = $('telemetryChart');
    if (!cv) return;
    let [w, h] = resize(cv);
    let ctx = cv.getContext('2d');
    let key = state.chart;
    let data = state.graph[key] || [];

    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) return;

    let max = Math.max(...data, 1);
    let min = Math.min(...data, 0);
    let span = Math.max(max - min, max * 0.1, 1);

    ctx.strokeStyle = 'rgba(25, 55, 75, 0.35)';
    ctx.beginPath();
    for (let i = 0; i < 3; i++) {
      let y = 8 + i * (h - 16) / 2;
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    ctx.beginPath();
    data.forEach((v, i) => {
      let x = (i / (data.length - 1)) * w;
      let y = h - 7 - ((v - min) / span) * (h - 16);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.8;
    ctx.stroke();

    ctx.fillStyle = '#94b8c2';
    ctx.font = '9px "JetBrains Mono", monospace';
    let latest = data[data.length - 1];
    ctx.fillText(`${key.toUpperCase()}: ${latest.toFixed(key === 'points' || key === 'objects' ? 0 : 1)}`, 6, 12);
  }

  function fmtMs(v) {
    return Number.isFinite(v) ? `${v.toFixed(1)} ms` : 'N/A';
  }

  const pct = v => v == null ? 'N/A' : `${(v * 100).toFixed(1)}%`;

  function updateInspector() {
    let o = (state.frame?.objects || []).find(x => x.id === state.selectedId);
    let p = $('inspectorContent');
    if (!p) return;

    if (!o) {
      p.innerHTML = `
        <div class="empty-inspector">
          <span class="empty-icon">🎯</span>
          SELECT ANY TRACKED OBJECT OR OBSTACLE
          <small>CLICK A 3D BOUNDING BOX OR 2.5D MAP TARGET</small>
        </div>
      `;
      return;
    }

    let sp = o.speed ?? Math.hypot(o.vx || 0, o.vy || 0);
    let dist = o.distance ?? Math.hypot(o.x, o.y);
    let dir = o.heading ?? (Math.atan2(o.vy || 0, o.vx || 0) * 180 / Math.PI);
    let risk = o.risk_level ?? (dist < 12 ? 'CRITICAL' : (dist < 25 ? 'WARNING' : 'SAFE'));
    let badgeCls = o.kind === 'vehicle' ? 'badge-veh' : (o.kind === 'pedestrian' ? 'badge-ped' : 'badge-stat');
    let riskCls = risk === 'CRITICAL' ? 'risk-critical' : (risk === 'WARNING' ? 'risk-warning' : 'risk-safe');

    p.innerHTML = `
      <div class="object-card">
        <div class="object-header">
          <span class="object-badge ${badgeCls}">${o.kind.toUpperCase()} #${String(o.id).padStart(2, '0')}</span>
          <span class="risk-badge ${riskCls}">${risk}</span>
        </div>
        <div class="object-grid">
          <div><small>3D POSITION (X, Y)</small><b>${o.x.toFixed(1)}, ${o.y.toFixed(1)} M</b></div>
          <div><small>RADIAL DISTANCE</small><b>${dist.toFixed(1)} M</b></div>
          <div><small>SPEED</small><b>${sp.toFixed(1)} M/S (${(sp * 3.6).toFixed(0)} KM/H)</b></div>
          <div><small>HEADING / BEARING</small><b>${Number.isFinite(dir) ? dir.toFixed(0) : '0'}°</b></div>
          <div><small>BOUNDS (W × D × H)</small><b>${(o.w || 1.8).toFixed(1)} × ${(o.d || 4.2).toFixed(1)} × ${(o.h || 1.5).toFixed(1)} M</b></div>
          <div><small>LiDAR RETURNS</small><b>${o.n_points || '--'} PTS</b></div>
          <div><small>TIME TO COLLISION</small><b>${o.ttc != null ? `${o.ttc.toFixed(1)} S` : 'CLEAR'}</b></div>
          <div><small>TRACK PERSISTENCE</small><b>${o.age > 1 ? `${o.age} FRAMES` : 'INITIALIZING'}</b></div>
        </div>
        <button class="tool-button lock-cam-btn" id="lockCamButton">${state.lockedCamera ? '🔓 UNLOCK CAMERA' : '🔒 LOCK CAM ON TARGET'}</button>
      </div>
    `;

    const lockBtn = $('lockCamButton');
    if (lockBtn) {
      lockBtn.onclick = () => {
        state.lockedCamera = !state.lockedCamera;
        updateInspector();
      };
    }
  }

  function updateUI(d) {
    if ($('mapImg')) $('mapImg').src = 'data:image/png;base64,' + d.image_b64;
    if ($('topFrame')) $('topFrame').textContent = String(d.frame_idx).padStart(4, '0');
    if ($('frameIndex')) $('frameIndex').textContent = String(d.frame_idx).padStart(4, '0');
    if ($('jumpFrame')) $('jumpFrame').value = d.frame_idx;
    if ($('frameScrubber')) $('frameScrubber').value = d.frame_idx;

    if ($('datasetValue')) $('datasetValue').textContent = d.dataset_type === 'kitti' ? 'KITTI / SEMANTICKITTI' : 'CITY ROAD NETWORK';
    if ($('streetNameValue')) $('streetNameValue').textContent = (d.ego?.street_name || 'GRAND AVENUE').toUpperCase();
    if ($('sensorValue')) $('sensorValue').textContent = d.sensor;
    if ($('topObstacleCount')) $('topObstacleCount').textContent = `${(d.objects || []).length} OBJECTS`;

    // Autonomy HUD Updates
    let ego = d.ego || {};
    let decision = ego.decision || 'AUTONOMOUS_CRUISE_CLEAR';
    if ($('decisionText')) $('decisionText').textContent = decision.replace(/_/g, ' ');
    if ($('hudStreet')) $('hudStreet').textContent = (ego.street_name || 'GRAND AVE').toUpperCase();
    if ($('steerValue')) $('steerValue').textContent = `${(ego.steer_angle || 0).toFixed(1)}°`;
    if ($('speedValue')) $('speedValue').textContent = `${(ego.speed_kmh || (ego.speed || 4.2) * 3.6).toFixed(1)} KM/H`;

    if ($('statusText')) $('statusText').textContent = 'AUTONOMOUS NAV · 60 FPS';
    if ($('fpsValue')) $('fpsValue').innerHTML = `${d.fps.toFixed(1)}<em>FPS</em>`;
    if ($('totalMs')) $('totalMs').textContent = fmtMs(d.total_ms);
    if ($('nPoints')) $('nPoints').textContent = d.n_points.toLocaleString();
    if ($('inferMs')) $('inferMs').textContent = fmtMs(d.latency_ms?.infer_ms ?? d.latency_ms?.inference_ms);
    if ($('fuseMs')) $('fuseMs').textContent = fmtMs(d.latency_ms?.fuse_ms ?? d.latency_ms?.grid_fusion_ms);
    if ($('trackMs')) $('trackMs').textContent = fmtMs(d.latency_ms?.track_ms ?? d.latency_ms?.tracking_ms);
    if ($('nObjects')) $('nObjects').textContent = (d.objects || []).length;
    if ($('senseMs')) $('senseMs').textContent = fmtMs(d.latency_ms?.sense_ms ?? d.latency_ms?.data_load_ms);
    if ($('renderMs')) $('renderMs').textContent = fmtMs(d.latency_ms?.render_ms);

    if ($('adaptCells')) $('adaptCells').textContent = d.memory.adaptive_cells.toLocaleString();
    if ($('adaptMb')) $('adaptMb').textContent = `${(d.memory.adaptive_bytes / 1e6).toFixed(2)} MB`;
    if ($('reduction')) $('reduction').textContent = `${d.memory.reduction_factor.toFixed(1)}× REDUCTION`;
    if ($('mapCells')) $('mapCells').textContent = `${d.memory.adaptive_cells.toLocaleString()} CELLS`;

    if ($('groundTruthMode')) $('groundTruthMode').disabled = !d.ground_truth_available;
    if ($('errorMode')) $('errorMode').disabled = !d.ground_truth_available;
    if (!d.ground_truth_available && ['ground_truth', 'error'].includes(state.mode)) {
      state.mode = 'prediction';
      activate('#renderModes button', '[data-mode="prediction"]');
      setCloud();
    }

    let body = $('accBody');
    if (body) {
      body.innerHTML = '';
      (d.accuracy_by_range || []).forEach(r => {
        body.insertAdjacentHTML('beforeend', `<tr><td>${r.range}</td><td>${pct(r.accuracy)}</td><td>${pct(r.miou)}</td></tr>`);
      });
    }

    state.graph.fps.push(d.fps);
    state.graph.latency.push(d.total_ms);
    state.graph.points.push(d.n_points);
    state.graph.objects.push((d.objects || []).length);
    Object.values(state.graph).forEach(a => { if (a.length > 80) a.shift(); });

    // Update 3D Trajectory Ribbon
    updatePlannedTrajectoryRibbon(ego.planned_path);

    updateInspector();
  }

  function receive(d, cache = true) {
    state.frame = d;
    state.history.push(d);
    if (state.history.length > 50) state.history.shift();
    if (cache) {
      state.cache.push(d);
      if (state.cache.length > 100) state.cache.shift();
      state.cacheIndex = state.cache.length - 1;
    }
    setCloud();
    updateUI(d);

    let load = $('loadingScreen');
    if (load) {
      if ($('loadingStep')) $('loadingStep').textContent = 'GPU ACCELERATION ACTIVE · PERCEPTION ONLINE';
      setTimeout(() => {
        load.style.opacity = '0';
        setTimeout(() => load.remove(), 500);
      }, 250);
    }
  }

  async function next() {
    if (state.busy || state.frame?.at_end) return;
    state.busy = true;
    try {
      let r = await fetch('/api/frame');
      if (!r.ok) throw Error();
      let d = await r.json();
      receive(d);
      if (d.at_end) {
        state.playing = false;
        if ($('playPause')) $('playPause').textContent = '▶';
      }
    } catch (e) {
      if ($('statusText')) $('statusText').textContent = 'DATA STREAM PAUSED';
    } finally {
      state.busy = false;
    }
  }

  async function seek(n) {
    if (state.busy) return;
    state.busy = true;
    if ($('statusText')) $('statusText').textContent = 'REBUILDING REPLAY STATE';
    try {
      let r = await fetch('/api/seek', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: n })
      });
      if (!r.ok) throw Error();
      state.history = [];
      state.cache = [];
      receive(await r.json());
    } catch (e) {
      if ($('statusText')) $('statusText').textContent = 'SEEK UNAVAILABLE';
    } finally {
      state.busy = false;
    }
  }

  function resetView() {
    state.camera = 'ego';
    state.yaw = -0.85;
    state.pitch = 0.48;
    state.distance = 54;
    state.target = [10, 0, 0];
    state.pan = [0, 0, 0];
    state.lockedCamera = false;
    activate('#cameraModes button', '[data-camera="ego"]');
  }

  function activate(group, chosen) {
    $$(group).forEach(b => b.classList.toggle('active', b.matches(chosen)));
  }

  function chooseObject(clientX, clientY) {
    if (!state.lastMvp) return;
    let r = canvas.getBoundingClientRect();
    let best = null;
    let bestD = 48;

    for (let o of state.frame?.objects || []) {
      let p = project([o.x, o.y, (o.z_max ?? 1.5) / 2], state.lastMvp, canvas.width, canvas.height);
      if (!p) continue;
      let px = (p[0] / canvas.width) * r.width;
      let py = (p[1] / canvas.height) * r.height;
      let d = Math.hypot(clientX - r.left - px, clientY - r.top - py);
      if (d < bestD) {
        bestD = d;
        best = o;
      }
    }

    if (best) {
      state.selectedId = best.id;
      updateInspector();
    }
  }

  // 3D Canvas Pointer Events
  canvas.addEventListener('pointerdown', e => {
    state.drag = { x: e.clientX, y: e.clientY, shift: e.shiftKey };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => state.drag = null);
  canvas.addEventListener('pointermove', e => {
    let d = state.drag;
    if (!d) return;
    let dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (d.shift) {
      state.pan[0] -= dx * 0.08;
      state.pan[1] += dy * 0.08;
    } else {
      state.yaw -= dx * 0.007;
      state.pitch = clamp(state.pitch - dy * 0.007, 0.06, 1.52);
      state.camera = 'free';
      activate('#cameraModes button', '[data-camera="free"]');
    }
    d.x = e.clientX;
    d.y = e.clientY;
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    state.distance = clamp(state.distance + e.deltaY * 0.05, 8, 180);
    state.camera = 'free';
    activate('#cameraModes button', '[data-camera="free"]');
  }, { passive: false });
  canvas.addEventListener('click', e => chooseObject(e.clientX, e.clientY));

  // 2.5D Map Hover HUD
  mapWrap.addEventListener('mousemove', e => {
    let r = mapWrap.getBoundingClientRect();
    let px = (e.clientX - r.left) / r.width;
    let py = (e.clientY - r.top) / r.height;
    let wx = (px - 0.5) * 200.0;
    let wy = (0.5 - py) * 200.0;
    let dist = Math.hypot(wx, wy);

    if (mapHoverHud) mapHoverHud.textContent = `X: ${wx.toFixed(1)}m | Y: ${wy.toFixed(1)}m | R: ${dist.toFixed(1)}m`;
  });
  mapWrap.addEventListener('mouseleave', () => {
    if (mapHoverHud) mapHoverHud.textContent = 'HOVER GRID CELL';
  });
  mapWrap.addEventListener('click', e => {
    let r = mapWrap.getBoundingClientRect();
    let px = (e.clientX - r.left) / r.width;
    let py = (e.clientY - r.top) / r.height;
    let wx = (px - 0.5) * 200.0;
    let wy = (0.5 - py) * 200.0;

    let best = null, bestD = 8.0;
    for (let o of state.frame?.objects || []) {
      let d = Math.hypot(o.x - wx, o.y - wy);
      if (d < bestD) {
        bestD = d;
        best = o;
      }
    }
    if (best) {
      state.selectedId = best.id;
      updateInspector();
    }
  });

  // UI Event Bindings
  if ($('resetCamera')) $('resetCamera').onclick = resetView;
  if ($('gridToggle')) {
    $('gridToggle').onclick = () => {
      state.showGrid = !state.showGrid;
      $('gridToggle').classList.toggle('active', state.showGrid);
      $('gridToggle').textContent = state.showGrid ? 'POLAR RINGS' : 'RINGS OFF';
    };
  }
  if ($('scanBeamToggle')) {
    $('scanBeamToggle').onclick = () => {
      state.scanBeam = !state.scanBeam;
      $('scanBeamToggle').classList.toggle('active', state.scanBeam);
      $('scanBeamToggle').textContent = state.scanBeam ? 'SCANNER ON' : 'SCANNER OFF';
    };
  }

  $$('#renderModes button').forEach(b => b.onclick = () => {
    state.mode = b.dataset.mode;
    activate('#renderModes button', `[data-mode="${state.mode}"]`);
    setCloud();
  });

  $$('#cameraModes button').forEach(b => b.onclick = () => {
    state.camera = b.dataset.camera;
    activate('#cameraModes button', `[data-camera="${state.camera}"]`);
    if (state.camera === 'orbit') state.target = [10, 0, 0];
  });

  $$('#semanticLegend input').forEach(i => i.onchange = () => {
    let c = +i.dataset.class;
    i.checked ? state.classVisible.add(c) : state.classVisible.delete(c);
    setCloud();
  });

  if ($('staticToggle')) {
    $('staticToggle').onchange = e => {
      state.staticVisible = e.target.checked;
      setCloud();
    };
  }
  if ($('dynamicToggle')) {
    $('dynamicToggle').onchange = e => {
      state.dynamicVisible = e.target.checked;
      setCloud();
    };
  }

  $$('[data-range]').forEach(i => i.onchange = () => {
    state.ranges = $$('[data-range]:checked').map(x => x.dataset.range.split(',').map(Number));
    setCloud();
  });

  if ($('densitySelect')) {
    $('densitySelect').onchange = e => {
      state.density = +e.target.value;
      setCloud();
    };
  }
  if ($('pointSize')) {
    $('pointSize').oninput = e => {
      state.pointSize = +e.target.value;
    };
  }
  if ($('trailsToggle')) {
    $('trailsToggle').onchange = e => {
      state.trails = e.target.checked;
    };
  }

  $$('#temporalWindow button').forEach(b => b.onclick = () => {
    state.temporal = +b.dataset.window;
    activate('#temporalWindow button', `[data-window="${state.temporal}"]`);
    setCloud();
  });

  $$('#telemetryPanel [data-chart]').forEach(b => b.onclick = () => {
    state.chart = b.dataset.chart;
    activate('#telemetryPanel [data-chart]', `[data-chart="${state.chart}"]`);
  });

  $$('.collapse').forEach(b => b.onclick = () => {
    let p = $(b.dataset.collapse);
    if (p) {
      p.classList.toggle('collapsed');
      b.textContent = p.classList.contains('collapsed') ? '＋' : '−';
    }
  });

  if ($('speedSelect')) {
    $('speedSelect').onchange = e => {
      state.speed = +e.target.value;
    };
  }

  if ($('playPause')) {
    $('playPause').onclick = () => {
      state.playing = !state.playing;
      $('playPause').textContent = state.playing ? '❚❚' : '▶';
    };
  }
  if ($('nextFrame')) $('nextFrame').onclick = next;
  if ($('prevFrame')) $('prevFrame').onclick = () => seek(Math.max(+$('frameScrubber').min, (state.frame?.frame_idx ?? +$('frameScrubber').min) - 1));
  if ($('resetBtn')) $('resetBtn').onclick = () => seek(+$('frameScrubber').min);
  if ($('jumpButton')) $('jumpButton').onclick = () => seek(+$('jumpFrame').value);
  if ($('frameScrubber')) $('frameScrubber').onchange = e => seek(+e.target.value);

  async function loadConfig() {
    try {
      let c = await (await fetch('/api/config')).json();
      if ($('dataMode')) $('dataMode').value = c.dataset_type;
      
      const seqGroup = $('seqGroup');
      const seqSelect = $('sequenceSelect');
      if (seqGroup && seqSelect) {
        if (c.dataset_type === 'kitti') {
          seqGroup.style.display = 'flex';
          seqSelect.innerHTML = (c.available_sequences || ['00']).map(s => `<option value="${s}">${s}</option>`).join('');
          seqSelect.value = c.current_sequence;
        } else {
          seqGroup.style.display = 'none';
        }
      }

      if ($('frameScrubber')) {
        $('frameScrubber').min = c.frame_min;
        $('frameScrubber').max = c.frame_max;
      }
      if ($('jumpFrame')) {
        $('jumpFrame').min = c.frame_min;
        $('jumpFrame').max = c.frame_max;
      }
    } catch (e) {
      if ($('statusText')) $('statusText').textContent = 'CONFIG RETRIEVAL FAILED';
    }
  }

  async function selectDataset() {
    if (state.busy) return;
    state.busy = true;
    if ($('statusText')) $('statusText').textContent = 'SWITCHING DATASET';
    try {
      const mode = $('dataMode')?.value || 'synthetic';
      const seq = $('sequenceSelect')?.value || '00';
      let r = await fetch('/api/dataset/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_type: mode,
          sequence: seq
        })
      });
      let d = await r.json();
      if (!r.ok) throw Error(d.error || 'Dataset switch failed');
      state.frame = null;
      state.history = [];
      state.cache = [];
      await loadConfig();
      state.busy = false;
      await next();
    } catch (e) {
      if ($('statusText')) $('statusText').textContent = `SOURCE ERROR: ${e.message}`;
    } finally {
      state.busy = false;
    }
  }

  if ($('dataMode')) $('dataMode').onchange = selectDataset;
  if ($('sequenceSelect')) $('sequenceSelect').onchange = selectDataset;

  if ($('presentationButton')) $('presentationButton').onclick = () => document.body.classList.toggle('presentation');
  document.addEventListener('keydown', e => {
    if (e.key.toLowerCase() === 'p' && !['INPUT', 'SELECT'].includes(document.activeElement.tagName)) {
      document.body.classList.toggle('presentation');
    }
  });

  async function loop() {
    if (state.playing) {
      await next();
    }
    setTimeout(loop, Math.max(35, 200 / state.speed));
  }

  try {
    initGL();
    loadConfig();
    loop();
  } catch (e) {
    console.error(e);
    if ($('statusText')) $('statusText').textContent = 'INITIALIZATION ERROR';
  }
})();
