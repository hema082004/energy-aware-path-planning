from flask import Flask, request, jsonify, Response
import numpy as np
import heapq
import math

app = Flask(__name__)

HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Energy-Aware Path Planning</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:#0f172a;color:#e5e7eb}
header{padding:20px 28px;background:#111827;border-bottom:1px solid #334155}
h1{margin:0;font-size:28px}.sub{color:#94a3b8;margin-top:6px}
nav{display:flex;gap:8px;padding:12px 20px;background:#111827;position:sticky;top:0;z-index:5}
nav button{background:#1e293b;color:#e5e7eb;border:1px solid #475569;padding:10px 16px;border-radius:8px;cursor:pointer}
nav button.active{background:#2563eb}
.page{display:none;padding:20px}.page.active{display:block}
.layout{display:grid;grid-template-columns:300px 1fr;gap:18px}.panel,.card{background:#111827;border:1px solid #334155;border-radius:12px;padding:18px}
label{display:block;margin:12px 0 5px;color:#cbd5e1}input{width:100%;padding:9px;border-radius:7px;border:1px solid #475569;background:#0f172a;color:white}
button.action{width:100%;margin-top:10px;padding:11px;border:0;border-radius:8px;background:#2563eb;color:white;cursor:pointer}
button.action:hover{background:#1d4ed8}.status{margin-top:12px;padding:10px;border-radius:8px;background:#172554;color:#bfdbfe}
#plot2d{height:650px}#plot3d,#plotTop{height:620px}
.row{display:flex;gap:12px;flex-wrap:wrap}.row .card{flex:1;min-width:180px}
.small{color:#94a3b8;font-size:13px}.metric{font-size:24px;font-weight:bold;margin-top:5px}
@media(max-width:850px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>Energy-Aware Path Planning in 2D and 3D Environment</h1><div class="sub">Vercel web version — A* path planning, optimized paths, candidates and 3D terrain</div></header>
<nav>
<button class="tab active" data-page="home">Home</button>
<button class="tab" data-page="candidates">Candidate Paths</button>
<button class="tab" data-page="comparison">Comparison</button>
</nav>

<section id="home" class="page active">
<div class="layout">
<div class="panel">
<h2>Controls</h2>
<label>Map Size: <span id="sizeVal">20</span></label>
<input id="size" type="range" min="10" max="50" value="20">
<label>Obstacle Density: <span id="densityVal">0.25</span></label>
<input id="density" type="range" min="0" max="0.5" step="0.01" value="0.25">
<label>Start X</label><input id="sx" type="number" min="0" value="2">
<label>Start Y</label><input id="sy" type="number" min="0" value="2">
<label>Goal X</label><input id="gx" type="number" min="0" value="18">
<label>Goal Y</label><input id="gy" type="number" min="0" value="14">
<label><input id="dynamic" type="checkbox" style="width:auto"> Enable Dynamic Obstacles</label>
<button class="action" onclick="generateMap()">Generate Map</button>
<button class="action" onclick="runAStar()">Run A* Algorithm</button>
<button class="action" onclick="optimizePath()">Optimized Path</button>
<button class="action" onclick="show3D()">Show 3D Terrain</button>
<button class="action" onclick="startRobot()">Start Robot Simulation</button>
<div id="status" class="status">Generate a map first.</div>
</div>
<div class="card"><h2>2D Visualization</h2><div id="plot2d"></div></div>
</div>
<div class="card" style="margin-top:18px"><h2>3D Terrain</h2><div id="plot3d"></div><div id="plotTop"></div></div>
</section>

<section id="candidates" class="page">
<div class="card"><h2>Candidate Path Generation</h2>
<button class="action" onclick="generateCandidates()">Generate Candidate Paths (2D + 3D)</button>
<div id="candidateStatus" class="status">Run A* on Home first.</div>
<div id="candidate2d" style="height:650px"></div>
<div id="candidate3d" style="height:650px"></div>
</div>
</section>

<section id="comparison" class="page">
<div class="card"><h2>Performance Comparison</h2>
<button class="action" onclick="showComparison()">Show Comparison Graphs</button>
<div id="comparisonPlot" style="height:500px"></div>
<div id="comparisonText" class="status"></div>
</div>
</section>

<script>
let state={grid:null,path:null,opt:null,start:[2,2],goal:[18,14],Z:null,candidates:[]};

document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
 b.classList.add('active');
 document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
 document.getElementById(b.dataset.page).classList.add('active');
});
size.oninput=()=>{sizeVal.textContent=size.value; const n=+size.value; gx.max=gy.max=n-1;};
density.oninput=()=>densityVal.textContent=density.value;

function setStatus(t){status.textContent=t}
function neighbors(x,y,n){return [[x+1,y],[x-1,y],[x,y+1],[x,y-1]].filter(([a,b])=>a>=0&&b>=0&&a<n&&b<n)}
function heuristic(a,b){return Math.hypot(a[0]-b[0],a[1]-b[1])}
function astar(grid,start,goal){
 const n=grid.length, key=p=>p[0]+','+p[1], open=[[heuristic(start,goal),0,start]], came=new Map(), g=new Map([[key(start),0]]), closed=new Set();
 while(open.length){
  open.sort((a,b)=>a[0]-b[0]); const [,cost,cur]=open.shift(), ck=key(cur);
  if(closed.has(ck)) continue; closed.add(ck);
  if(cur[0]===goal[0]&&cur[1]===goal[1]){
   let p=[cur]; let k=ck;
   while(came.has(k)){let q=came.get(k);p.push(q);k=key(q)}
   return p.reverse();
  }
  for(const nb of neighbors(cur[0],cur[1],n)){
   if(grid[nb[1]][nb[0]]===1) continue;
   const nk=key(nb), ng=cost+1;
   if(!g.has(nk)||ng<g.get(nk)){g.set(nk,ng);came.set(nk,cur);open.push([ng+heuristic(nb,goal),ng,nb])}
  }
 }
 return null;
}
function makeGrid(n,d){
 let g=Array.from({length:n},()=>Array.from({length:n},()=>Math.random()<d?1:0));
 const s=[+sx.value,+sy.value], t=[+gx.value,+gy.value]; g[s[1]][s[0]]=0;g[t[1]][t[0]]=0;
 return g;
}
function terrain(n){let z=[];for(let y=0;y<n;y++){let r=[];for(let x=0;x<n;x++){let X=x*50/(n-1),Y=y*50/(n-1);r.push(5+.6*Math.sin(X/6)+.6*Math.cos(Y/7)+.4*Math.sin((X+Y)/10))}z.push(r)}return z}
function draw2d(robot=null){
 const traces=[{z:state.grid,type:'heatmap',colorscale:[[0,'#f8fafc'],[1,'#111827']],showscale:false}];
 if(state.path) traces.push({x:state.path.map(p=>p[0]),y:state.path.map(p=>p[1]),mode:'lines',line:{color:'blue',width:4},name:'A* Path'});
 if(state.opt) traces.push({x:state.opt.map(p=>p[0]),y:state.opt.map(p=>p[1]),mode:'lines',line:{color:'lime',width:4,dash:'dash'},name:'Optimized Path'});
 traces.push({x:[state.start[0]],y:[state.start[1]],mode:'markers',marker:{size:13,color:'green'},name:'Start'});
 traces.push({x:[state.goal[0]],y:[state.goal[1]],mode:'markers',marker:{size:13,color:'red'},name:'Goal'});
 if(robot) traces.push({x:[robot[0]],y:[robot[1]],mode:'markers',marker:{size:18,color:'orange'},name:'Robot'});
 Plotly.newPlot('plot2d',traces,{paper_bgcolor:'#111827',plot_bgcolor:'#111827',font:{color:'#e5e7eb'},xaxis:{title:'X',dtick:1},yaxis:{title:'Y',autorange:'reversed',dtick:1},legend:{orientation:'h'}},{responsive:true});
}
function generateMap(){
 const n=+size.value; state.start=[+sx.value,+sy.value];state.goal=[+gx.value,+gy.value];
 if(state.start[0]>=n||state.start[1]>=n||state.goal[0]>=n||state.goal[1]>=n){setStatus('Start and goal must be inside the map.');return}
 state.grid=makeGrid(n,+density.value);state.path=null;state.opt=null;state.Z=terrain(n);
 draw2d();setStatus('Map Generated');
}
function runAStar(){
 if(!state.grid){setStatus('Generate map first!');return}
 state.path=astar(state.grid,state.start,state.goal);
 if(state.path){draw2d();setStatus('A* Path Generated — '+state.path.length+' points.')}
 else setStatus('No Path Found. Try lower obstacle density.');
}
function optimizePath(){
 if(!state.path){setStatus('Run A* first!');return}
 let s=state.path.map(p=>[...p]);
 for(let i=1;i<s.length-1;i++){let ax=Math.round((s[i-1][0]+s[i+1][0])/2),ay=Math.round((s[i-1][1]+s[i+1][1])/2);if(ax>=0&&ay>=0&&ax<state.grid.length&&ay<state.grid.length&&state.grid[ay][ax]===0)s[i]=[ax,ay]}
 state.opt=s;draw2d();setStatus('Optimized Path Generated — '+s.length+' points.');
}
function show3D(){
 if(!state.path){setStatus('Run A* first!');return}
 const n=state.grid.length,z=state.Z,X=Array.from({length:n},(_,x)=>x*50/(n-1)),Y=Array.from({length:n},(_,y)=>y*50/(n-1));
 let traces=[{x:X,y:Y,z:z,type:'surface',colorscale:'Viridis',opacity:.85,name:'Terrain'}];
 function line3(p,c,name){return {x:p.map(q=>X[q[0]]),y:p.map(q=>Y[q[1]]),z:p.map(q=>z[q[1]][q[0]]),type:'scatter3d',mode:'lines',line:{color:c,width:7},name:name}}
 traces.push(line3(state.path,'red','A* Path')); if(state.opt)traces.push(line3(state.opt,'lime','Optimized Path'));
 traces.push({x:[X[state.start[0]]],y:[Y[state.start[1]]],z:[z[state.start[1]][state.start[0]]],type:'scatter3d',mode:'markers',marker:{size:8,color:'green'},name:'Start'});
 traces.push({x:[X[state.goal[0]]],y:[Y[state.goal[1]]],z:[z[state.goal[1]][state.goal[0]]],type:'scatter3d',mode:'markers',marker:{size:8,color:'blue'},name:'Goal'});
 let ox=[],oy=[],oz=[];for(let y=0;y<n;y++)for(let x=0;x<n;x++)if(state.grid[y][x]){ox.push(X[x]);oy.push(Y[y]);oz.push(z[y][x])}
 traces.push({x:ox,y:oy,z:oz,type:'scatter3d',mode:'markers',marker:{size:4,color:'white'},name:'Obstacles'});
 Plotly.newPlot('plot3d',traces,{paper_bgcolor:'#111827',font:{color:'#fff'},scene:{xaxis_title:'X',yaxis_title:'Y',zaxis_title:'Height'}},{responsive:true});
 const top=[{z:z,type:'heatmap',colorscale:'Viridis',name:'Height'},{x:state.path.map(p=>p[0]),y:state.path.map(p=>p[1]),mode:'lines',line:{color:'red',width:4},name:'A* Path'}];
 if(state.opt)top.push({x:state.opt.map(p=>p[0]),y:state.opt.map(p=>p[1]),mode:'lines',line:{color:'lime',width:4},name:'Optimized Path'});
 Plotly.newPlot('plotTop',top,{paper_bgcolor:'#111827',font:{color:'#fff'},yaxis:{autorange:'reversed'},title:'Top View of 3D Environment'},{responsive:true});
 setStatus('3D Terrain displayed.');
}
function startRobot(){
 if(!state.opt){setStatus('Generate Optimized Path first!');return}
 let i=0; const timer=setInterval(()=>{if(i>=state.opt.length){clearInterval(timer);setStatus('Robot Reached Goal!');return} draw2d(state.opt[i]);setStatus('Robot simulation: step '+(i+1)+' / '+state.opt.length);i++},150);
}
function generateCandidates(){
 if(!state.path){candidateStatus.textContent='Run A* Algorithm on Home first!';return}
 const paths=[state.path];
 for(let k=0;k<10;k++){let p=state.path.map(q=>[...q]);for(let i=1;i<p.length-1;i++){let dx=p[i+1][0]-p[i-1][0],dy=p[i+1][1]-p[i-1][1],norm=Math.hypot(dx,dy);if(norm){let px=-dy/norm,py=dx/norm,nx=Math.round(p[i][0]+px*(Math.random()*4-2)),ny=Math.round(p[i][1]+py*(Math.random()*4-2));if(nx>=0&&ny>=0&&nx<state.grid.length&&ny<state.grid.length&&state.grid[ny][nx]===0)p[i]=[nx,ny]}}paths.push(p)}
 state.candidates=paths;
 let tr=paths.slice(1).map((p,i)=>({x:p.map(q=>q[0]),y:p.map(q=>q[1]),mode:'lines',line:{dash:'dot',width:1},name:'Candidate '+(i+1)}));
 tr.push({x:state.path.map(p=>p[0]),y:state.path.map(p=>p[1]),mode:'lines',line:{color:'blue',width:4},name:'Main A* Path'});
 Plotly.newPlot('candidate2d',tr,{paper_bgcolor:'#111827',font:{color:'#fff'},yaxis:{autorange:'reversed'},title:'Candidate Paths (2D)'},{responsive:true});
 const n=state.grid.length,z=state.Z,X=Array.from({length:n},(_,x)=>x*50/(n-1)),Y=Array.from({length:n},(_,y)=>y*50/(n-1));
 let t=[{x:X,y:Y,z:z,type:'surface',opacity:.7,colorscale:'Viridis',name:'Terrain'}];
 paths.forEach((p,i)=>t.push({x:p.map(q=>X[q[0]]),y:p.map(q=>Y[q[1]]),z:p.map(q=>z[q[1]][q[0]]),type:'scatter3d',mode:'lines',line:{width:i===0?7:3},name:i===0?'Main A* Path':'Candidate '+i}));
 Plotly.newPlot('candidate3d',t,{paper_bgcolor:'#111827',font:{color:'#fff'},scene:{xaxis_title:'X',yaxis_title:'Y',zaxis_title:'Height'},title:'Candidate Paths (3D)'},{responsive:true});
 candidateStatus.textContent='Generated 10 candidate paths plus the main A* path.';
}
function showComparison(){
 if(!state.path){comparisonText.textContent='Run A* first.';return}
 const a=state.path.length,o=state.opt?state.opt.length:null;
 const labels=['A*'];const vals=[a];if(o){labels.push('Optimized');vals.push(o)}
 Plotly.newPlot('comparisonPlot',[{x:labels,y:vals,type:'bar'}],{paper_bgcolor:'#111827',plot_bgcolor:'#111827',font:{color:'#fff'},yaxis:{title:'Path Length'},title:'Path Length Comparison'},{responsive:true});
 comparisonText.innerHTML='<b>A* path length:</b> '+a+(o?'<br><b>Optimized path length:</b> '+o:'')+'<br><b>Start:</b> ('+state.start.join(', ')+') &nbsp; <b>Goal:</b> ('+state.goal.join(', ')+')';
}
generateMap();
</script>
</body></html>"""

@app.route("/", methods=["GET", "POST"])
@app.route("/api", methods=["GET", "POST"])
@app.route("/api/index.py", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return jsonify({"status": "ok", "message": "Vercel API is running."})
    return Response(HTML, mimetype="text/html")

if __name__ == "__main__":
    app.run()
