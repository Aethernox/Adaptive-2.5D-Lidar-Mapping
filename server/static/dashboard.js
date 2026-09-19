/* Live perception workstation.  Points stay in typed GPU buffers; DOM is
   reserved for controls and diagnostics, never used as a point-cloud scene. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id), $$ = s => [...document.querySelectorAll(s)];
  const canvas = $('lidarCanvas'), overlay = $('sceneOverlay'), map = $('mapOverlay');
  const gl = canvas.getContext('webgl', { antialias: false, alpha: false, preserveDrawingBuffer: false });
  const state = {
    frame: null, history: [], cache: [], cacheIndex: -1, playing: true, busy: false,
    mode: 'semantic', pointSize: 3, temporal: 1, showGrid: true, trails: true,
    selectedId: null, chart: 'fps', camera: 'ego', yaw: -.85, pitch: .47, distance: 62,
    target: [15, 0, 0], pan: [0, 0, 0], drag: null, lastMvp: null,
    classVisible: new Set([0,1,2,3,4,5]), staticVisible: true, dynamicVisible: true,
    ranges: [[0,10],[10,25],[25,50],[50,100]], density: 1, speed: 1, graph: {fps:[],latency:[],points:[]}
  };
  const semantic = [[.22,.65,.31],[.74,.67,.25],[.87,.27,.24],[.96,.62,.18],[1,.24,.72],[.12,.58,1]];
  const dynamicIds = new Set([4,5]);
  let program, cloudBuffer, lineBuffer, pointCount = 0, lineCount = 0;

  const shaders = {
    vert: `attribute vec3 p; attribute vec3 c; attribute float q; attribute float a;
      uniform mat4 mvp; uniform float pointSize; varying vec4 v;
      void main(){ gl_Position=mvp*vec4(p,1.0); gl_PointSize=(1.5+q*pointSize)*(180.0/max(20.0,gl_Position.w)); v=vec4(c,a); }`,
    frag: `precision mediump float; varying vec4 v; void main(){ vec2 d=gl_PointCoord-vec2(.5); float r=dot(d,d); if(r>.25) discard; float glow=1.0-smoothstep(.08,.25,r); gl_FragColor=vec4(v.rgb*(.72+.45*glow),v.a*glow); }`,
    lineVert: `attribute vec3 p; attribute vec3 c; uniform mat4 mvp; varying vec3 v; void main(){gl_Position=mvp*vec4(p,1.0);v=c;}`,
    lineFrag: `precision mediump float; varying vec3 v; void main(){gl_FragColor=vec4(v,.45);}`
  };
  const compile = (type, src) => { const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s); if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s)); return s; };
  const makeProgram = (v,f) => { const p=gl.createProgram(); gl.attachShader(p,compile(gl.VERTEX_SHADER,v)); gl.attachShader(p,compile(gl.FRAGMENT_SHADER,f)); gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p)); return p; };
  function initGL(){
    if(!gl) { $('statusText').textContent='WEBGL UNAVAILABLE'; return; }
    program=makeProgram(shaders.vert,shaders.frag); program.lines=makeProgram(shaders.lineVert,shaders.lineFrag);
    cloudBuffer=gl.createBuffer(); lineBuffer=gl.createBuffer(); gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    buildSceneLines(); requestAnimationFrame(render);
  }
  function buildSceneLines(){
    const d=[], add=(a,b,c=[.13,.31,.38])=>d.push(...a,...c,...b,...c);
    // Ground grid plus ego vehicle coordinate frame and low-poly vehicle footprint.
    for(let i=-100;i<=100;i+=10){add([i,-100,-.06],[i,100,-.06]);add([-100,i,-.06],[100,i,-.06]);}
    for(let r of [10,25,50,100]) for(let i=0;i<80;i++){let a=i/80*Math.PI*2,b=(i+1)/80*Math.PI*2;add([r*Math.cos(a),r*Math.sin(a),-.04],[r*Math.cos(b),r*Math.sin(b),-.04],[.10,.27,.33]);}
    add([0,0,0],[5,0,0],[.95,.25,.32]); add([0,0,0],[0,5,0],[.25,.9,.54]); add([0,0,0],[0,0,4],[.25,.55,1]);
    const car=[[-2,-1,0],[3,-1,0],[3,1,0],[-2,1,0]], roof=[[-.7,-.8,.85],[1.7,-.8,.85],[1.7,.8,.85],[-.7,.8,.85]];
    for(let i=0;i<4;i++){add(car[i],car[(i+1)%4],[.30,.83,.9]);add(roof[i],roof[(i+1)%4],[.30,.83,.9]);add(car[i],roof[i],[.18,.5,.58]);}
    gl.bindBuffer(gl.ARRAY_BUFFER,lineBuffer); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(d),gl.STATIC_DRAW); lineCount=d.length/6;
  }
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)), lerp=(a,b,t)=>a+(b-a)*t;
  const matPerspective=(f,a,n,z)=>{let q=1/Math.tan(f/2),nf=1/(n-z);return[q/a,0,0,0,0,q,0,0,0,0,(z+n)*nf,-1,0,0,2*z*n*nf,0]};
  const norm=v=>{let l=Math.hypot(...v)||1;return v.map(x=>x/l)}, cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  function lookAt(e,t){let z=norm([e[0]-t[0],e[1]-t[1],e[2]-t[2]]),x=norm(cross([0,0,1],z)); if(Math.hypot(...x)<.1)x=[1,0,0];let y=cross(z,x);return[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-x[0]*e[0]-x[1]*e[1]-x[2]*e[2],-y[0]*e[0]-y[1]*e[1]-y[2]*e[2],-z[0]*e[0]-z[1]*e[1]-z[2]*e[2],1]}
  function mul(a,b){let o=Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];return o}
  function camera(){
    let e,t=state.target.map((v,i)=>v+state.pan[i]);
    if(state.camera==='top') e=[t[0],t[1],105];
    else if(state.camera==='front') e=[-9,0,4],t=[65,0,1];
    else if(state.camera==='rear') e=[22,0,5],t=[-55,0,1];
    else { e=[t[0]+state.distance*Math.cos(state.pitch)*Math.cos(state.yaw),t[1]+state.distance*Math.cos(state.pitch)*Math.sin(state.yaw),t[2]+state.distance*Math.sin(state.pitch)]; }
    return {e,t};
  }
  function resize(c){let r=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.floor(c.clientWidth*r)),h=Math.max(1,Math.floor(c.clientHeight*r));if(c.width!==w||c.height!==h){c.width=w;c.height=h;}return[w,h,r]}
  function render(){
    if(gl&&program){let [w,h]=resize(canvas),cam=camera(),mvp=mul(matPerspective(1.02,w/h,.1,350),lookAt(cam.e,cam.t));state.lastMvp=mvp;
      gl.viewport(0,0,w,h);gl.clearColor(.012,.028,.045,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program.lines);gl.uniformMatrix4fv(gl.getUniformLocation(program.lines,'mvp'),false,new Float32Array(mvp));gl.bindBuffer(gl.ARRAY_BUFFER,lineBuffer);let lp=gl.getAttribLocation(program.lines,'p'),lc=gl.getAttribLocation(program.lines,'c');gl.enableVertexAttribArray(lp);gl.vertexAttribPointer(lp,3,gl.FLOAT,false,24,0);gl.enableVertexAttribArray(lc);gl.vertexAttribPointer(lc,3,gl.FLOAT,false,24,12);gl.drawArrays(gl.LINES,0,lineCount);
      gl.useProgram(program);gl.uniformMatrix4fv(gl.getUniformLocation(program,'mvp'),false,new Float32Array(mvp));gl.uniform1f(gl.getUniformLocation(program,'pointSize'),state.pointSize);gl.bindBuffer(gl.ARRAY_BUFFER,cloudBuffer);let p=gl.getAttribLocation(program,'p'),c=gl.getAttribLocation(program,'c'),q=gl.getAttribLocation(program,'q'),a=gl.getAttribLocation(program,'a');gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,3,gl.FLOAT,false,32,0);gl.enableVertexAttribArray(c);gl.vertexAttribPointer(c,3,gl.FLOAT,false,32,12);gl.enableVertexAttribArray(q);gl.vertexAttribPointer(q,1,gl.FLOAT,false,32,24);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,1,gl.FLOAT,false,32,28);gl.drawArrays(gl.POINTS,0,pointCount);
      drawOverlay(mvp,w,h); drawMap(); drawChart(); }
    requestAnimationFrame(render);
  }
  function colorFor(v){let x=v[0],y=v[1],z=v[2],cls=v[3],conf=v[4],intensity=v[5]??conf,r=Math.hypot(x,y);
    if(state.mode==='intensity') return [intensity*.3+.08,intensity*.75+.12,intensity*.9+.1];
    if(state.mode==='height'){let t=clamp((z+1)/4,0,1);return [lerp(.15,.96,t),lerp(.25,.35,t),lerp(.85,.22,t)];}
    if(state.mode==='distance'){let t=clamp(r/100,0,1);return [lerp(.12,.98,t),lerp(.85,.25,t),lerp(.95,.12,t)];}
    if(state.mode==='dynamic') return dynamicIds.has(cls)?(cls===5?[.2,.8,1]:[1,.3,.7]):[.12,.18,.22];
    return semantic[cls]||[.55,.55,.55];
  }
  function setCloud(){
    if(!gl||!state.frame)return;let d=[],frames=state.history.slice(-state.temporal),step=state.density;
    frames.forEach((f,fi)=>{let age=(fi+1)/frames.length, pts=f.point_cloud||[];for(let i=0;i<pts.length;i+=step){let v=pts[i],cls=v[3],r=Math.hypot(v[0],v[1]);if(!state.classVisible.has(cls)||(dynamicIds.has(cls)?!state.dynamicVisible:!state.staticVisible)||!state.ranges.some(z=>r>=z[0]&&r<z[1]))continue;let col=colorFor(v),alpha=(fi===frames.length-1?1:.13+.36*age)*(.45+.55*v[4]);d.push(v[0],v[1],Math.max(-1,v[2]),col[0],col[1],col[2],.65+v[4],alpha);}});
    pointCount=d.length/8;gl.bindBuffer(gl.ARRAY_BUFFER,cloudBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(d),gl.DYNAMIC_DRAW);
    $('cloudMeta').textContent=`${pointCount.toLocaleString()} GPU POINTS · ${state.temporal>1?`MEMORY ${state.temporal} FRAMES`:'CURRENT FRAME'}`;
  }
  function project(v,m,w,h){let x=v[0],y=v[1],z=v[2],qx=m[0]*x+m[4]*y+m[8]*z+m[12],qy=m[1]*x+m[5]*y+m[9]*z+m[13],qw=m[3]*x+m[7]*y+m[11]*z+m[15];if(qw<=.01)return null;return [w*(qx/qw*.5+.5),h*(.5-qy/qw*.5),qw]}
  function drawOverlay(mvp,w,h){
    let [ow,oh]=resize(overlay),ctx=overlay.getContext('2d');ctx.clearRect(0,0,ow,oh);ctx.scale(ow/w,oh/h);let objs=state.frame?.objects||[];
    if(state.trails){for(let obj of objs){let hist=state.history.map(f=>(f.objects||[]).find(o=>o.id===obj.id)).filter(Boolean);ctx.beginPath();hist.forEach((o,i)=>{let p=project([o.x,o.y,(o.z_min??0)+.4],mvp,w,h);if(!p)return;i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);});ctx.strokeStyle=obj.kind==='vehicle'?'rgba(74,208,255,.48)':'rgba(255,89,201,.48)';ctx.lineWidth=1;ctx.stroke();}}
    objs.forEach(obj=>{let z0=obj.z_min??0,z1=obj.z_max??1.6,hw=Math.max(obj.w||1,.45)/2,hd=Math.max(obj.d||1,.45)/2,corners=[];for(let z of [z0,z1])for(let x of [-hw,hw])for(let y of [-hd,hd])corners.push(project([obj.x+x,obj.y+y,z],mvp,w,h));if(corners.some(p=>!p))return;let edges=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]],active=obj.id===state.selectedId;ctx.strokeStyle=active?'#fff3a8':obj.kind==='vehicle'?'#54d9f0':'#ff5dcc';ctx.lineWidth=active?2:1;ctx.beginPath();edges.forEach(([a,b])=>{ctx.moveTo(corners[a][0],corners[a][1]);ctx.lineTo(corners[b][0],corners[b][1]);});ctx.stroke();let top=project([obj.x,obj.y,z1+.3],mvp,w,h),speed=Math.hypot(obj.vx||0,obj.vy||0);if(top&&top[2]>0){ctx.fillStyle=active?'#fff':'#bed6dc';ctx.font='10px ui-monospace,Consolas';ctx.fillText(`${obj.kind.toUpperCase()} #${String(obj.id).padStart(2,'0')}  ${speed.toFixed(1)}m/s`,top[0]+6,top[1]-5);let end=project([obj.x+(obj.vx||0)*1.2,obj.y+(obj.vy||0)*1.2,z0+.5],mvp,w,h);let start=project([obj.x,obj.y,z0+.5],mvp,w,h);if(end&&start){ctx.beginPath();ctx.moveTo(start[0],start[1]);ctx.lineTo(end[0],end[1]);ctx.stroke();}}});ctx.setTransform(1,0,0,1,0,0);
  }
  function drawMap(){
    let f=state.frame;if(!f)return;let [w,h]=resize(map),ctx=map.getContext('2d');ctx.clearRect(0,0,w,h);let scale=w/200;
    const cv=(x,y)=>[w/2+x*scale,h/2-y*scale];if(state.showGrid){ctx.strokeStyle='rgba(99,214,227,.22)';ctx.lineWidth=1;for(let t of f.adaptive_grid?.tiers||[]){ctx.beginPath();ctx.arc(w/2,h/2,t.r_max*scale,0,Math.PI*2);ctx.stroke();}}
    for(let cell of f.adaptive_grid?.cells||[]){let [x,y,,cls,conf,dr,dt]=cell,[px,py]=cv(x,y),r=Math.max(1,dr*scale*.6);ctx.fillStyle=`rgba(${Math.round((semantic[cls]||[.5,.5,.5])[0]*255)},${Math.round((semantic[cls]||[.5,.5,.5])[1]*255)},${Math.round((semantic[cls]||[.5,.5,.5])[2]*255)},${Math.min(.42,conf*.3)})`;ctx.fillRect(px-r,py-r,r*2,r*2);}
    for(let o of f.objects||[]){let [x,y]=cv(o.x,o.y);ctx.strokeStyle=o.id===state.selectedId?'#fff7a0':o.kind==='vehicle'?'#55d9ea':'#ff66c4';ctx.strokeRect(x-Math.max(3,o.w*scale/2),y-Math.max(3,o.d*scale/2),Math.max(6,o.w*scale),Math.max(6,o.d*scale));ctx.fillStyle=ctx.strokeStyle;ctx.fillText(`#${o.id}`,x+4,y-4);}
  }
  function drawChart(){let cv=$('telemetryChart'),[w,h]=resize(cv),ctx=cv.getContext('2d'),key=state.chart,data=state.graph[key];ctx.clearRect(0,0,w,h);if(data.length<2)return;let max=Math.max(...data,1),min=Math.min(...data,0),span=Math.max(max-min,max*.08,1);ctx.strokeStyle='rgba(64,126,142,.28)';ctx.beginPath();for(let i=0;i<3;i++){let y=8+i*(h-16)/2;ctx.moveTo(0,y);ctx.lineTo(w,y)}ctx.stroke();ctx.beginPath();data.forEach((v,i)=>{let x=i/(data.length-1)*w,y=h-7-(v-min)/span*(h-16);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle='#54d9ea';ctx.lineWidth=1.5;ctx.stroke();ctx.fillStyle='#83a5ad';ctx.font='9px ui-monospace';ctx.fillText(`${key.toUpperCase()}  ${data[data.length-1].toFixed(key==='points'?0:1)}`,5,11)}
  function fmtMs(v){return Number.isFinite(v)?`${v.toFixed(1)} ms`:'N/A'} const pct=v=>v==null?'N/A':`${(v*100).toFixed(1)}%`;
  function updateInspector(){let o=(state.frame?.objects||[]).find(x=>x.id===state.selectedId),p=$('inspectorContent');if(!o){p.innerHTML='<div class="empty-inspector">SELECT A TRACKED OBJECT<br><span>CLICK A BOX IN THE 3D VIEW</span></div>';return}let sp=Math.hypot(o.vx||0,o.vy||0),dist=Math.hypot(o.x,o.y),dir=Math.atan2(o.vy||0,o.vx||0)*180/Math.PI;p.innerHTML=`<div class="object-data"><div class="accent"><small>OBJECT ID</small><b>${o.kind.toUpperCase()} #${String(o.id).padStart(2,'0')}</b></div><div><small>TRACK STATE</small><b>${o.age>1?'CONFIRMED':'INITIALIZING'}</b></div><div><small>POSITION</small><b>${o.x.toFixed(1)}, ${o.y.toFixed(1)} M</b></div><div><small>DISTANCE</small><b>${dist.toFixed(1)} M</b></div><div><small>VELOCITY</small><b>${sp.toFixed(1)} M/S</b></div><div><small>DIRECTION</small><b>${Number.isFinite(dir)?dir.toFixed(0):'N/A'}°</b></div><div><small>EXTENTS</small><b>${o.w.toFixed(1)} × ${o.d.toFixed(1)} M</b></div><div><small>CONFIDENCE</small><b>N/A</b></div></div>`}
  function updateUI(d){
    $('mapImg').src='data:image/png;base64,'+d.image_b64;$('topFrame').textContent=String(d.frame_idx).padStart(4,'0');$('frameIndex').textContent=String(d.frame_idx).padStart(4,'0');$('jumpFrame').value=d.frame_idx;$('frameScrubber').value=d.frame_idx;$('statusText').textContent='LIVE · PIPELINE ACTIVE';$('fpsValue').innerHTML=`${d.fps.toFixed(1)}<em>FPS</em>`;$('totalMs').textContent=fmtMs(d.total_ms);$('nPoints').textContent=d.n_points.toLocaleString();$('inferMs').textContent=fmtMs(d.latency_ms.infer_ms);$('fuseMs').textContent=fmtMs(d.latency_ms.fuse_ms);$('trackMs').textContent=fmtMs(d.latency_ms.track_ms);$('nObjects').textContent=d.objects.length;$('senseMs').textContent=fmtMs(d.latency_ms.sense_ms);$('renderMs').textContent=fmtMs(d.latency_ms.render_ms);$('adaptCells').textContent=d.memory.adaptive_cells.toLocaleString();$('adaptMb').textContent=`${(d.memory.adaptive_bytes/1e6).toFixed(2)} MB`;$('reduction').textContent=`${d.memory.reduction_factor.toFixed(0)}×`;$('mapCells').textContent=`${d.memory.adaptive_cells.toLocaleString()} CELLS`;
    let body=$('accBody');body.innerHTML='';d.accuracy_by_range.forEach(r=>body.insertAdjacentHTML('beforeend',`<tr><td>${r.range}</td><td>${pct(r.accuracy)}</td><td>${pct(r.miou)}</td></tr>`));
    state.graph.fps.push(d.fps);state.graph.latency.push(d.total_ms);state.graph.points.push(d.n_points);Object.values(state.graph).forEach(a=>{if(a.length>70)a.shift()});updateInspector();
  }
  function receive(d,cache=true){state.frame=d;state.history.push(d);if(state.history.length>50)state.history.shift();if(cache){state.cache.push(d);if(state.cache.length>100)state.cache.shift();state.cacheIndex=state.cache.length-1;}setCloud();updateUI(d);let load=$('loadingScreen');if(load){$('loadingStep').textContent='GPU RENDERER READY · LIVE DATA LINKED';setTimeout(()=>{load.style.opacity='0';setTimeout(()=>load.remove(),500)},250)}}
  async function next(){if(state.busy)return;state.busy=true;try{let r=await fetch('/api/frame');if(!r.ok)throw Error();receive(await r.json());}catch(e){$('statusText').textContent='DATA STREAM UNAVAILABLE';}finally{state.busy=false}}
  async function seek(n){if(state.busy)return;state.busy=true;$('statusText').textContent='REBUILDING PIPELINE STATE';try{let r=await fetch('/api/seek',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame:n})});if(!r.ok)throw Error();state.history=[];state.cache=[];receive(await r.json());}catch(e){$('statusText').textContent='SEEK UNAVAILABLE';}finally{state.busy=false}}
  function resetView(){state.camera='ego';state.yaw=-.85;state.pitch=.47;state.distance=62;state.target=[15,0,0];state.pan=[0,0,0];activate('#cameraModes button','[data-camera="ego"]')}
  function activate(group,chosen){$$(group).forEach(b=>b.classList.toggle('active',b.matches(chosen)))}
  function chooseObject(x,y){if(!state.lastMvp)return;let r=canvas.getBoundingClientRect(),best=null,bestD=42;for(let o of state.frame?.objects||[]){let p=project([o.x,o.y,(o.z_max??1.5)/2],state.lastMvp,canvas.width,canvas.height);if(!p)continue;let px=(p[0]/canvas.width)*r.width,py=(p[1]/canvas.height)*r.height,d=Math.hypot(x-r.left-px,y-r.top-py);if(d<bestD){bestD=d;best=o}}if(best){state.selectedId=best.id;updateInspector()}}
  canvas.addEventListener('pointerdown',e=>{state.drag={x:e.clientX,y:e.clientY,shift:e.shiftKey};canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointerup',()=>state.drag=null);canvas.addEventListener('pointermove',e=>{let d=state.drag;if(!d)return;let dx=e.clientX-d.x,dy=e.clientY-d.y;if(d.shift){state.pan[0]-=dx*.09;state.pan[1]+=dy*.09}else{state.yaw-=dx*.008;state.pitch=clamp(state.pitch-dy*.008,.08,1.47);state.camera='free';activate('#cameraModes button','[data-camera="free"]')}d.x=e.clientX;d.y=e.clientY});canvas.addEventListener('wheel',e=>{e.preventDefault();state.distance=clamp(state.distance+e.deltaY*.05,15,150);state.camera='free';activate('#cameraModes button','[data-camera="free"]')},{passive:false});canvas.addEventListener('dblclick',e=>chooseObject(e.clientX,e.clientY));
  $('resetCamera').onclick=resetView;$('gridToggle').onclick=()=>{state.showGrid=!state.showGrid;$('gridToggle').classList.toggle('active',state.showGrid);$('gridToggle').textContent=state.showGrid?'GRID ON':'GRID OFF'};
  $$('#renderModes button').forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;activate('#renderModes button',`[data-mode="${state.mode}"]`);setCloud()});
  $$('#cameraModes button').forEach(b=>b.onclick=()=>{state.camera=b.dataset.camera;activate('#cameraModes button',`[data-camera="${state.camera}"]`);if(state.camera==='orbit')state.target=[10,0,0]});
  $$('#semanticLegend input').forEach(i=>i.onchange=()=>{let c=+i.dataset.class;i.checked?state.classVisible.add(c):state.classVisible.delete(c);setCloud()});
  $('staticToggle').onchange=e=>{state.staticVisible=e.target.checked;setCloud()};$('dynamicToggle').onchange=e=>{state.dynamicVisible=e.target.checked;setCloud()};$$('[data-range]').forEach(i=>i.onchange=()=>{state.ranges=$$('[data-range]:checked').map(x=>x.dataset.range.split(',').map(Number));setCloud()});$('densitySelect').onchange=e=>{state.density=+e.target.value;setCloud()};$('pointSize').oninput=e=>state.pointSize=+e.target.value;$('trailsToggle').onchange=e=>state.trails=e.target.checked;
  $$('#temporalWindow button').forEach(b=>b.onclick=()=>{state.temporal=+b.dataset.window;activate('#temporalWindow button',`[data-window="${state.temporal}"]`);setCloud()});$$('#telemetryPanel [data-chart]').forEach(b=>b.onclick=()=>{state.chart=b.dataset.chart;activate('#telemetryPanel [data-chart]',`[data-chart="${state.chart}"]`)});
  $$('.collapse').forEach(b=>b.onclick=()=>{let p=$(b.dataset.collapse);p.classList.toggle('collapsed');b.textContent=p.classList.contains('collapsed')?'＋':'−'});$('speedSelect').onchange=e=>state.speed=+e.target.value;
  $('playPause').onclick=()=>{state.playing=!state.playing;$('playPause').textContent=state.playing?'❚❚':'▶'};$('nextFrame').onclick=next;$('prevFrame').onclick=()=>{if(state.cacheIndex>0){state.cacheIndex--;receive(state.cache[state.cacheIndex],false)}};$('resetBtn').onclick=()=>seek(1);$('jumpButton').onclick=()=>seek(+$('jumpFrame').value);$('frameScrubber').onchange=e=>seek(+e.target.value);
  $('presentationButton').onclick=()=>document.body.classList.toggle('presentation');document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='p'&&!['INPUT','SELECT'].includes(document.activeElement.tagName))document.body.classList.toggle('presentation')});
  async function loop(){if(state.playing)await next();setTimeout(loop,Math.max(50,240/state.speed))}
  try{initGL();loop()}catch(e){console.error(e);$('statusText').textContent='RENDERER INITIALIZATION FAILED'}
})();
