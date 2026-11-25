"use strict";
const el = (id) => document.getElementById(id);
const text = (id, value) => { el(id).textContent = value; };
const show = (id, visible) => el(id).classList.toggle("hidden", !visible);
const pretty = (value) => JSON.stringify(value, null, 2);
const names = {
  assassin:"标飞", dragon_knight:"龙骑士", cleric:"牧师", archer:"弓箭手",
  fire_poison:"火毒（合成）", ice_lightning:"冰雷（合成）", bandit:"刀飞", blood_knight:"血骑士（合成）", builtin:"内置两图", pixel:"像素基线",
  "three-map-fixture":"外部三图", "lifecycle-model":"已验收的混合模型",
  "branch-grove":"林地岔路", "grove-shortcut":"直达遗迹", standard:"逐图战斗、拾取、对话", "talk-then-clear":"先对话再清场", "three-map-visits":"三图访问",
  "dead-forest-drill":"死亡之林双图演练（合成）", "dead-forest-hunt":"两图各击败一只怪物",
  cms_dead_forest:"死亡之林", dead_forest_1:"死亡之林Ⅰ", dead_forest_2:"死亡之林Ⅱ",
  cms_east_hill:"射手村东部", branch_fixture:"分支路线样例", henesys:"射手村", east_hill:"射手村东部小山",
  east_grass:"射手村东部草丛", east_forest:"射手村东部树林", mushroom_forest:"青蘑菇树林",
  queued:"排队中", running:"运行中", finished:"已执行", cancelled:"已取消",
  interrupted:"已中断", error:"出错", completed:"目标完成", stopped:"已停止",
  failed:"目标失败", verified:"校验完成", run:"视觉闭环", replay:"动作重放",
  reinfer:"感知重推理", export_bundle:"导出分层包", verify_bundle:"验证分层包"
};
const label = (key) => names[key] || key;
let catalog = null, jobs = [], selected = "", detail = null, evidence = "summary";
let traceOffset = 0, activeView = "runs", sim = null, simEvidence = "observation";
let playTimer = null, selectionRevision = 0, refreshing = false, simBusy = false;
let recordingSessions = [], recordingCoverages = [], selectedRecording = "", recordingDetail = null;
let recordingPlayTimer = null, recordingRevision = 0, recordingsLoaded = false;

async function api(path, body) {
  const options = body === undefined ? {} : {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
  };
  const response = await fetch("/api/" + path, options);
  if (!response.ok) {
    let msg = response.statusText;
    try { const data = await response.json(); msg = typeof data.detail === "string" ? data.detail : pretty(data.detail); }
    catch { /* A transport rejection can have an empty body. */ }
    throw new Error(msg || String(response.status));
  }
  return response.json();
}
function fail(error) { text("global-error", error.message || String(error)); show("global-error", true); }
function bind(id, type, handler) {
  el(id).addEventListener(type, (event) => {
    event.preventDefault();
    show("global-error", false);
    Promise.resolve().then(() => handler(event)).catch(fail);
  });
}
function options(id, items) {
  const select = el(id), previous = select.value;
  select.replaceChildren(...items.map(([key, name]) => new Option(name, key)));
  if (items.some(([key]) => key === previous)) select.value = previous;
}
function displayContent() {
  const groups = [
    ["职业", Object.entries(catalog.classes).map(([key, value]) =>
      [key, [value.version, ...value.skills].join(" · ")])],
    ["地图", Object.entries(catalog.maps).map(([key, value]) => [key, value.join(" → ")])],
    ["感知方式", catalog.models.map(key => [key, key])]
  ];
  const cards = el("content-cards"); cards.replaceChildren();
  for (const [title, entries] of groups) {
    const card = document.createElement("section"); card.className = "panel content-group";
    const header = document.createElement("h2"); header.textContent = title; card.append(header);
    for (const [key, info] of entries) {
      const item = document.createElement("div"); item.className = "content-item";
      const strong = document.createElement("strong"); strong.textContent = label(key);
      const small = document.createElement("small"); small.textContent = info;
      item.append(strong, small); card.append(item);
    }
    cards.append(card);
  }
  for (const id of ["class-select", "sim-class"]) options(id, Object.keys(catalog.classes).map(k=>[k,label(k)]));
  for (const id of ["map-select", "sim-map"]) options(id, Object.keys(catalog.maps).map(k=>[k,label(k)]));
  options("model-select", catalog.models.map(k=>[k,label(k)]));
  displayMissions();
  const graphs = Object.keys(catalog.world_graphs);
  show("routes-empty", !graphs.length);
  show("routes-workspace", graphs.length > 0);
  options("route-graph", graphs.map(k => [k, label(k)]));
  if (graphs.length) displayRouteNodes();
  text("bootstrap-json", pretty(catalog));
}
function displayMissions() {
  const ids = catalog.mission_options[el("map-select").value];
  options("mission-select", ids.map(id => [id, label(id)]));
}
bind("map-select", "change", displayMissions);
let routeRevision = 0;
function clearRoute() {
  routeRevision++;
  text("route-status", "选择起终点后查询路线。");
  el("route-steps").replaceChildren();
}
function displayRouteNodes() {
  const graphId = el("route-graph").value;
  const graph = catalog.world_graphs[graphId];
  const nodes = graph.nodes.map(k => [k, label(k)]);
  options("route-source", nodes);
  options("route-target", nodes);
  el("route-source").value = graph.nodes.includes("henesys") ? "henesys" : graph.nodes[0];
  el("route-target").value = graph.nodes.includes("mushroom_forest") ? "mushroom_forest" : graph.nodes.at(-1);
  text("route-scope", graph.scope === "game_reference" ? "按游戏资料登记的连接规划；实录通行情况待验证。" : "按独立模拟样例的有向连接规划。");
  clearRoute();
}
async function queryRoute() {
  clearRoute();
  const revision = routeRevision;
  const graph = el("route-graph").value;
  const query = new URLSearchParams({source:el("route-source").value, target:el("route-target").value});
  text("route-status", "正在查询…");
  let response;
  try { response = await api("world-graphs/" + encodeURIComponent(graph) + "/route?" + query); }
  catch (error) { if (revision === routeRevision) text("route-status", error.message); return; }
  if (revision !== routeRevision) return;
  const route = response.route;
  if (!route) { text("route-status", "已登记的出口无法到达目的地。"); return; }
  if (!route.portals.length) { text("route-status", "起点和终点相同，无需换图。"); return; }
  text("route-status", route.portals.length + " 次换图 · 总代价 " + route.total_cost);
  for (const edge of route.portals) {
    const step = document.createElement("li");
    step.textContent = label(edge.source) + " → " + label(edge.target) + "（出口 ID " + edge.portal_id + "）";
    el("route-steps").append(step);
  }
}
bind("route-graph", "change", displayRouteNodes);
bind("route-source", "change", clearRoute);
bind("route-target", "change", clearRoute);
bind("route-form", "submit", queryRoute);
function chooseView(view) {
  if (!["runs","recordings","simulator","routes","keys","content"].includes(view)) view = "runs";
  activeView = view; stopPlay(); stopRecordingPlay();
  document.querySelectorAll(".view").forEach(e=>e.classList.toggle("hidden", e.id !== "view-"+view));
  document.querySelectorAll("[data-view]").forEach(e=>e.classList.toggle("active", e.dataset.view===view));
  text("view-label", {runs:"任务与回放", recordings:"真实录像复核", simulator:"单步模拟器", routes:"地图路线", keys:"角色键位", content:"内容与模型"}[view]);
  if (view === "recordings" && !recordingsLoaded) loadRecordings().catch(fail);
}
function setupForm() {
  const kind = el("run-kind").value, run = kind === "run";
  show("run-config-fields",run); show("run-source-fields",!run);
  const inputKind = kind === "verify_bundle" ? "export_bundle" : "run";
  const available = jobs.filter(j=>j.state==="finished" && j.kind===inputKind);
  options("source-select", available.map(j=>[j.id,label(j.kind)+" · "+j.id.slice(0,8)]));
  el("submit-run").disabled = !catalog || (!run && !available.length);
}
function drawJobs() {
  text("run-count",jobs.length); show("runs-empty",!jobs.length); show("run-table",!!jobs.length);
  const list=el("run-list"); list.replaceChildren();
  for (const job of jobs) {
    const row=document.createElement("tr");
    for (const value of [label(job.kind), label(job.outcome||job.state),
                         new Date(job.created).toLocaleString("zh-CN")]) {
      const cell=document.createElement("td"); cell.textContent=value; row.append(cell);
    }
    const cell=document.createElement("td"), button=document.createElement("button");
    button.className="button subtle"; button.textContent="查看 "+job.id.slice(0,6);
    button.addEventListener("click",()=>chooseJob(job.id).catch(fail));
    cell.append(button); row.append(cell); list.append(row);
  }
  setupForm();
}
async function refresh() {
  if (refreshing) return;
  refreshing=true;
  try {
    if (!catalog) { catalog=await api("content"); displayContent(); }
    jobs=await api("jobs"); drawJobs();
    text("connection-label","本机已连接"); text("server-version","Maple Vision · 离线工作台");
    el("connection-dot").className="status-dot";
    if (selected && (!detail || ["queued","running"].includes(detail.state))) await loadJob();
  } catch(error) {
    text("connection-label","连接失败"); el("connection-dot").className="status-dot bad"; throw error;
  } finally { refreshing=false; }
}
async function chooseJob(id) {
  selected=id; detail=null; selectionRevision++; traceOffset=0; evidence="summary"; stopPlay();
  el("frame-range").value=0; await loadJob();
}
async function loadJob() {
  const id=selected, version=selectionRevision;
  const data=await api("jobs/"+id);
  if (id!==selected || version!==selectionRevision) return;
  detail=data; show("run-detail",true); text("detail-status",label(data.outcome || data.state));
  text("detail-heading",label(data.kind)+" · "+id.slice(0,8)); show("cancel-job",data.state==="queued");
  const result=data.result || {}, box=el("result-metrics"); box.replaceChildren();
  for (const [name,val] of [["结果",label(data.outcome||data.state)],["动作数",result.steps??"—"],["可看帧数",data.frames.length]]) {
    const item=document.createElement("div"); item.className="metric";
    const dt=document.createElement("dt"),dd=document.createElement("dd"); dt.textContent=name; dd.textContent=val;
    item.append(dt,dd); box.append(item);
  }
  el("frame-range").max=Math.max(0,data.frames.length-1);
  drawFrame(); await drawEvidence();
}
function stopPlay(){ clearInterval(playTimer); playTimer=null; text("frame-play","▷"); }
function drawFrame() {
  const range=el("frame-range"), frame=detail?.frames[Number(range.value)];
  show("replay-frame",!!frame); show("frame-empty",!frame);
  for (const id of ["frame-prev","frame-play","frame-range","frame-next"]) el(id).disabled=!frame;
  text("frame-caption",frame==="final"?"终帧":(frame?"第 "+Number(frame)+" 帧":""));
  if (frame) el("replay-frame").src="/api/jobs/"+selected+"/frames/"+frame;
}
function moveFrame(step) {
  const range=el("frame-range"); range.value=Math.max(0,Math.min(+range.max,+range.value+step)); drawFrame();
}
async function drawEvidence() {
  const id=selected, tab=evidence, offset=traceOffset;
  document.querySelectorAll("[data-evidence]").forEach(b=>{
    const active=b.dataset.evidence===tab; b.classList.toggle("active",active); b.setAttribute("aria-selected",active);
  });
  show("trace-controls",tab==="trace");
  if (!detail) return;
  if (tab==="trace") {
    const rows=await api("jobs/"+id+"/trace?offset="+offset);
    if (id!==selected || evidence!==tab || traceOffset!==offset) return;
    text("evidence-json",pretty(rows)); text("trace-page",rows.length?(offset+1)+"–"+(offset+rows.length):"没有更多记录");
    el("trace-prev").disabled=!offset; el("trace-next").disabled=rows.length<4;
  } else {
    text("evidence-json",pretty(tab==="config"?detail.config:(detail.result || {state:detail.state,error:detail.error})));
  }
}

function stopRecordingPlay() {
  clearTimeout(recordingPlayTimer); recordingPlayTimer=null; text("recording-play","▷");
}
function recordingLabel(session) {
  const start=session.source_start_frame || 0;
  return "单窗口 · " + session.source_name + " · 源帧 " + start + "–" + session.source_end_frame;
}
function coverageLabel(coverage) {
  return "连续覆盖 · " + coverage.source_name + " · 源帧 " + coverage.decoded_source_range.join("–");
}
const recordingEventNames={
  rope_contact_visible:"接绳可见",
  layer_change_visible:"层位变化可见",
  level_change_visible:"层位变化可见",
  attack_motion_visible:"攻击动作可见",
  damage_feedback_visible:"伤害反馈可见"
};
function recordingEvents() {
  if (!recordingDetail) return [];
  if (recordingDetail.viewer_kind==="coverage") {
    return recordingDetail.reviewed_visible_events.map(event=>({
      ...event,
      start_index:recordingDetail.frames.findIndex(frame=>frame.source_sequence===event.start_source_sequence),
      end_index:recordingDetail.frames.findIndex(frame=>frame.source_sequence===event.end_source_sequence),
      display_range:"源帧 "+event.start_source_sequence+"–"+event.end_source_sequence
    }));
  }
  const events=[];
  for (const item of recordingDetail.reviewed_event_reports) for (const event of item.report.events) {
    events.push({
      ...event,
      reviewer:item.report.reviewer,
      verified:item.verification.verified,
      start_index:event.start.sequence,
      end_index:event.end.sequence,
      display_range:"窗口帧 "+event.start.sequence+"–"+event.end.sequence
    });
  }
  return events.sort((a,b)=>a.start_index-b.start_index || a.end_index-b.end_index || a.kind.localeCompare(b.kind));
}
function updateRecordingEventNavigation(current) {
  const events=recordingEvents().filter(event=>event.start_index>=0 && event.end_index>=event.start_index);
  const active=events.filter(event=>event.start_index<=current && current<=event.end_index);
  const previous=[...events].reverse().find(event=>event.start_index<current);
  const next=events.find(event=>event.start_index>current);
  el("recording-event-prev").disabled=!previous;
  el("recording-event-next").disabled=!next;
  if (active.length) {
    text("recording-event-status","当前："+active.map(event=>(recordingEventNames[event.kind] || event.kind)+"（"+event.display_range+"）").join("；"));
  } else {
    text("recording-event-status","当前帧无已复核事件 · 共 "+events.length+" 条");
  }
}
async function loadRecordings() {
  const version=++recordingRevision;
  [recordingSessions,recordingCoverages]=await Promise.all([api("recordings"),api("recording-coverages")]);
  if (version!==recordingRevision) return;
  recordingsLoaded=true;
  const choices=[
    ...recordingCoverages.map(coverage=>["coverage:"+coverage.coverage_sha256,coverageLabel(coverage)]),
    ...recordingSessions.map(session=>["session:"+session.session_id,recordingLabel(session)])
  ];
  text("recording-count",choices.length);
  show("recordings-empty",!choices.length);
  show("recordings-workspace",!!choices.length);
  if (!choices.length) { selectedRecording=""; recordingDetail=null; return; }
  options("recording-select",choices);
  if (!choices.some(([identity])=>identity===selectedRecording)) {
    selectedRecording=choices[0][0];
    el("recording-select").value=selectedRecording;
  }
  await loadRecording();
}
async function loadRecording() {
  const identity=selectedRecording, version=++recordingRevision;
  stopRecordingPlay(); recordingDetail=null;
  const coverage=identity.startsWith("coverage:"), key=identity.slice(identity.indexOf(":")+1);
  const data=await api((coverage?"recording-coverages/":"recordings/")+key);
  if (identity!==selectedRecording || version!==recordingRevision) return;
  recordingDetail={...data,viewer_kind:coverage?"coverage":"session",viewer_key:key};
  const start=coverage?data.decoded_source_range[0]:(data.source_start_frame || 0);
  const end=coverage?data.decoded_source_range[1]:data.source_end_frame;
  text("recording-source",data.source_name+" · SHA-256 "+data.source_sha256.slice(0,16)+"… · "+data.permission+" / "+data.license);
  const events=coverage?data.reviewed_events:data.reviewed_event_reports.reduce((total,item)=>total+item.report.events.length,0);
  const metrics=el("recording-metrics"); metrics.replaceChildren();
  for (const [name,value] of [["帧数",coverage?data.decoded_frame_count:data.frame_count],["可见事件",events],["源帧范围",start+"–"+end]]) {
    const item=document.createElement("div"); item.className="metric";
    const dt=document.createElement("dt"), dd=document.createElement("dd"); dt.textContent=name; dd.textContent=value;
    item.append(dt,dd); metrics.append(item);
  }
  const range=el("recording-range"); range.max=Math.max(0,data.frames.length-1); range.value=0;
  drawRecordingFrame();
}
function drawRecordingFrame() {
  const range=el("recording-range"), frame=recordingDetail?.frames[Number(range.value)];
  show("recording-frame",!!frame); show("recording-frame-empty",!frame);
  for (const id of ["recording-prev","recording-play","recording-range","recording-next"]) el(id).disabled=!frame;
  if (!frame) {
    el("recording-event-prev").disabled=true; el("recording-event-next").disabled=true;
    text("recording-frame-caption",""); text("recording-event-status","当前帧无已复核事件。"); text("recording-evidence",""); return;
  }
  updateRecordingEventNavigation(Number(range.value));
  const coverage=recordingDetail.viewer_kind==="coverage";
  const source=coverage?frame.source_sequence:(recordingDetail.source_start_frame || 0)+frame.sequence;
  text("recording-frame-caption","源帧 "+source+" · 窗口帧 "+frame.sequence+" · "+frame.timestamp_ms+" ms");
  el("recording-frame").src=coverage
    ? "/api/recording-coverages/"+recordingDetail.viewer_key+"/frames/"+source
    : "/api/recordings/"+recordingDetail.viewer_key+"/frames/"+frame.sequence;
  if (coverage) {
    const visibleNow=recordingDetail.reviewed_visible_events
      .filter(event=>event.start_source_sequence<=source && source<=event.end_source_sequence)
      .map(event=>event.kind);
    text("recording-evidence",pretty({
      coverage:{coverage_sha256:recordingDetail.coverage_sha256,verified:recordingDetail.verified,contiguous:recordingDetail.contiguous,decoder_stop_reached:recordingDetail.decoder_stop_reached,windows:recordingDetail.windows},
      current_frame:{...frame,visible_events:visibleNow},
      reviewed_visible_events:recordingDetail.reviewed_visible_events,
      limitations:recordingDetail.limitations
    }));
  } else {
    const reports=recordingDetail.reviewed_event_reports.map(item=>({
      verified:item.verification.verified,
      scope:item.verification.scope,
      reviewer:item.report.reviewer,
      visible_events:item.report.events,
      limitations:item.report.limitations
    }));
    const visibleNow=[];
    for (const report of reports) for (const event of report.visible_events) {
      if (event.start.sequence<=frame.sequence && frame.sequence<=event.end.sequence) visibleNow.push(event.kind);
    }
    text("recording-evidence",pretty({
      session:{session_id:recordingDetail.session_id,validated:recordingDetail.validated,timing_accuracy:recordingDetail.timing_accuracy,timebase:recordingDetail.timebase,truncated:recordingDetail.truncated,completion_reason:recordingDetail.completion_reason},
      current_frame:{...frame,source_sequence:source,visible_events:visibleNow},
      reviewed_event_reports:reports,
      limitations:recordingDetail.limitations
    }));
  }
}
function moveRecordingFrame(step) {
  const range=el("recording-range"); range.value=Math.max(0,Math.min(+range.max,+range.value+step)); drawRecordingFrame();
}
function moveRecordingEvent(direction) {
  const range=el("recording-range"), current=Number(range.value);
  const events=recordingEvents().filter(event=>event.start_index>=0 && event.end_index>=event.start_index);
  const target=direction<0
    ? [...events].reverse().find(event=>event.start_index<current)
    : events.find(event=>event.start_index>current);
  if (!target) return;
  stopRecordingPlay(); range.value=target.start_index; drawRecordingFrame();
}
function playRecordingFrame() {
  const range=el("recording-range");
  if (+range.value>=+range.max) return stopRecordingPlay();
  const current=recordingDetail.frames[+range.value], next=recordingDetail.frames[+range.value+1];
  moveRecordingFrame(1);
  recordingPlayTimer=setTimeout(playRecordingFrame,Math.max(20,Math.min(500,next.timestamp_ms-current.timestamp_ms)));
}
function drawSim(data) {
  show("sim-key-log", false);
  sim=data; sessionStorage.setItem("maple-session",data.id);
  show("sim-controls",true); show("sim-frame",true); show("sim-empty",false);
  el("sim-frame").src="data:image/png;base64,"+data.image;
  text("sim-clock",data.timestamp_ms+" ms · 第 "+data.sequence+" 帧");
  text("sim-json",pretty(data[simEvidence]));
  show("sim-outcome",!!data.receipt);
  if (data.receipt) text("sim-outcome",(data.receipt.accepted?"已接受":"未接受")+" · "+data.receipt.reason);
}
async function simOperation(operation) {
  if (simBusy || !sim) return;
  simBusy=true;
  const controls=["sim-step","sim-save","sim-restore","sim-reset","sim-create"].map(el);
  controls.forEach(button=>{button.disabled=true;});
  try {
    const request={operation,revision:sim.revision};
    if(operation==="step") {
      const kind=el("sim-action").value;
      request.action={kind,
        direction:["move","climb","jump","jump_attack","attack","cast"].includes(kind)?+el("sim-direction").value:0,
        skill:kind==="cast" ? el("sim-skill").value : "basic",
        target:["interact","portal"].includes(kind)?el("sim-target").value:""};
    }
    drawSim(await api("sessions/"+sim.id,request));
    if (operation !== "step") {
      text("sim-outcome", {save:"已保存当前状态",restore:"已恢复存档",reset:"已重置到初始状态"}[operation]);
      show("sim-outcome",true);
    }
  } catch(error) { drawSim(await api("sessions/"+sim.id)); throw error; }
  finally { simBusy=false; controls.forEach(button=>{button.disabled=false;}); }
}
document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>{location.hash=b.dataset.view;});
window.addEventListener("hashchange",()=>chooseView(location.hash.slice(1)));
document.querySelectorAll("[data-evidence]").forEach(b=>b.onclick=()=>{evidence=b.dataset.evidence;drawEvidence().catch(fail);});
document.querySelectorAll("[data-sim-evidence]").forEach(b=>b.onclick=()=>{
  simEvidence=b.dataset.simEvidence;
  document.querySelectorAll("[data-sim-evidence]").forEach(node=>{
    const active=node===b; node.classList.toggle("active",active); node.setAttribute("aria-selected",active);
  });
  if (sim) text("sim-json",pretty(sim[simEvidence]));
});
bind("refresh","click",refresh); bind("run-kind","change",setupForm);
bind("run-form","submit",async()=>{
  const kind=el("run-kind").value, body={kind};
  if(kind==="run") {
    Object.assign(body,{job:el("class-select").value,map_id:el("map-select").value,
      model:el("model-select").value,mission:el("mission-select").value,seed:+el("seed").value,max_steps:+el("max-steps").value,fault:el("fault-select").value});
    const profile = el("run-profile").value.trim();
    if (profile) body.input_profile = storedKeyProfile(profile);
  }
  else body.source=el("source-select").value;
  el("submit-run").disabled=true;
  try { const created=await api("jobs",body); await chooseJob(created.id); await refresh(); }
  finally { setupForm(); }
});
bind("cancel-job","click",async()=>{ await api("jobs/"+selected+"/cancel",{}); await loadJob(); await refresh(); });
bind("frame-range","input",drawFrame); bind("frame-prev","click",()=>moveFrame(-1)); bind("frame-next","click",()=>moveFrame(1));
bind("frame-play","click",()=>{
  if(playTimer) return stopPlay();
  const range=el("frame-range"); if (+range.value===+range.max) range.value=0;
  drawFrame(); text("frame-play","Ⅱ");
  playTimer=setInterval(()=>{ if(+range.value>=+range.max) stopPlay(); else moveFrame(1); },200);
});
bind("recording-select","change",async()=>{selectedRecording=el("recording-select").value; await loadRecording();});
bind("recording-range","input",drawRecordingFrame);
bind("recording-prev","click",()=>moveRecordingFrame(-1));
bind("recording-next","click",()=>moveRecordingFrame(1));
bind("recording-event-prev","click",()=>moveRecordingEvent(-1));
bind("recording-event-next","click",()=>moveRecordingEvent(1));
bind("recording-play","click",()=>{
  if(recordingPlayTimer) return stopRecordingPlay();
  const range=el("recording-range"); if(+range.value===+range.max) range.value=0;
  drawRecordingFrame(); text("recording-play","Ⅱ"); playRecordingFrame();
});
bind("trace-next","click",async()=>{traceOffset+=4; await drawEvidence();});
bind("trace-prev","click",async()=>{traceOffset=Math.max(0,traceOffset-4); await drawEvidence();});
bind("sim-form","submit",async()=>drawSim(await api("sessions",{job:el("sim-class").value,map_id:el("sim-map").value,seed:+el("sim-seed").value})));
for(const op of ["step","save","reset","restore"]) bind("sim-"+op,"click",()=>simOperation(op));
chooseView(location.hash.slice(1)); refresh().catch(fail);
const saved=sessionStorage.getItem("maple-session");
if(saved) api("sessions/"+saved).then(drawSim).catch(()=>sessionStorage.removeItem("maple-session"));
setInterval(()=>{ if (activeView==="runs" && !document.hidden) refresh().catch(fail); },2000);

function keyProfileId() {
  return "maple.keybindings.v1." + el("keys-character").value.trim();
}
function storedKeyProfile(profile) {
  const saved = localStorage.getItem("maple.keybindings.v1." + profile);
  if (saved === null) throw new Error("请先在角色键位页确认该角色。");
  const bindings = [];
  for (const line of saved.split("\n").filter(Boolean)) {
    const split = line.indexOf("=");
    if (split < 1) throw new Error("保存的角色键位格式无效，请重新确认。");
    bindings.push({skill:line.slice(0,split).trim(),key:line.slice(split+1).trim()});
  }
  if (!bindings.length) throw new Error("该角色没有已确认的键位。");
  return {schema_version:1,profile_id:profile,bindings};
}
bind("keys-export", "click", async () => {
  const profile = storedKeyProfile(el("keys-character").value.trim());
  const validated = await api("input-profile", profile);
  const url = URL.createObjectURL(new Blob([pretty(validated) + "\n"], {type:"application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "input-bindings.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
function clearKeyImage() {
  const image = el("keys-image");
  if (image.hasAttribute("src")) URL.revokeObjectURL(image.src);
  image.removeAttribute("src");
  el("keys-image-file").value = "";
  show("keys-image", false);
}
bind("keys-character", "input", () => {
  clearKeyImage();
  const saved = localStorage.getItem(keyProfileId());
  el("keys-list").value = saved || "";
  text("keys-status", saved ? "已载入此角色上次确认的键位，请与游戏设置核对。" : "");
});
bind("keys-list", "input", () => text("keys-status", "修改尚未保存。"));
bind("keys-image-file", "change", () => {
  const file = el("keys-image-file").files[0];
  clearKeyImage();
  if (!file) return;
  if (!["image/png","image/jpeg","image/webp"].includes(file.type)) {
    text("keys-status", "请选择 PNG、JPEG 或 WebP 图片。"); return;
  }
  el("keys-image").src = URL.createObjectURL(file);
  show("keys-image", true);
});
let keyEditRevision = 0;
bind("keys-import", "change", async () => {
  const input = el("keys-import");
  const file = input.files[0];
  input.value = "";
  if (!file) return;
  const revision = ++keyEditRevision;
  try {
    if (file.size > 65536) throw new Error("请选择 64 KiB 以内的键位配置。");
    const source = new TextDecoder("utf-8", {fatal:true}).decode(await file.arrayBuffer());
    const profile = await api("input-profile", JSON.parse(source));
    if (revision !== keyEditRevision) return;
    const editor = el("keys-list"), character = el("keys-character");
    const representable = value => value === value.trim() && !/[\r\n]/.test(value);
    if (!representable(profile.profile_id) || profile.profile_id.length > character.maxLength ||
        profile.bindings.some(({skill, key}) => !representable(skill) || skill.includes("=") || !representable(key))) {
      throw new Error("此配置含当前编辑器无法完整保留的名称或键位。");
    }
    const draft = profile.bindings.map(({skill, key}) => skill + " = " + key).join("\n");
    if (draft.length > editor.maxLength) throw new Error("键位表超出当前编辑器的长度限制。");
    if (character.value.trim() !== profile.profile_id) clearKeyImage();
    character.value = profile.profile_id;
    editor.value = draft;
    text("keys-status", "已载入 " + profile.bindings.length + " 项草稿，核对后点击确认并保存。");
  } catch (error) {
    if (revision === keyEditRevision) text("keys-status", "导入失败：" + error.message);
  }
});
for (const id of ["keys-list", "keys-character"]) {
  el(id).addEventListener("input", () => { keyEditRevision++; });
}
bind("keys-form", "submit", async () => {
  const revision = ++keyEditRevision;
  const profile = el("keys-character").value.trim();
  const lines = el("keys-list").value.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  if (!el("keys-character").value.trim()) {
    text("keys-status", "请填写区服和角色。"); return;
  }
  const bindings = [];
  const result = [];
  for (const line of lines) {
    const split = line.indexOf("=");
    const skill = line.slice(0, split).trim(), key = line.slice(split + 1).trim();
    if (split < 1 || !skill || !key) {
      text("keys-status", "请按「技能名 = 键位」逐行填写。"); return;
    }
    bindings.push({key, skill});
    result.push(skill + " = " + key);
  }
  if (bindings.length) {
    try {
      await api("input-profile", {schema_version:1, profile_id:profile, bindings});
    } catch (error) {
      if (revision !== keyEditRevision) return;
      const messages = {
        "input profile contains duplicate normalized keys":"同一个键位出现了两次，请核对。",
        "input profile contains duplicate normalized skills":"同一个技能出现了两次，请核对。"
      };
      text("keys-status", messages[error.message] || "键位未保存：" + error.message);
      return;
    }
  }
  if (revision !== keyEditRevision) return;
  localStorage.setItem(keyProfileId(), result.join("\n"));
  text("keys-status", "已确认并保存 " + result.length + " 项绑定。");
});

function keyResult(receipt) {
  if (!receipt.events.includes("attack")) return receipt.reason;
  const hits = receipt.events.filter(event => event === "attack_hit").length;
  return "本步直接命中 " + hits + " 段";
}
el("sim-key").addEventListener("keydown", async event => {
  if (event.key === "Tab" || event.isComposing) return;
  event.preventDefault();
  if (event.repeat || simBusy || !sim) return;
  if ((event.ctrlKey && event.key !== "Control") || (event.altKey && event.key !== "Alt") ||
      (event.metaKey && event.key !== "Meta") || (event.shiftKey && event.key !== "Shift")) {
    text("sim-key-status", "当前试放支持单键。"); return;
  }
  const profile = el("sim-profile").value.trim();
  const saved = localStorage.getItem("maple.keybindings.v1." + profile);
  if (!profile || saved === null) { text("sim-key-status","请先在角色键位页确认该角色。"); return; }
  const bindings = Object.create(null);
  for (const line of saved.split("\n").filter(Boolean)) {
    const split = line.indexOf("=");
    bindings[line.slice(split + 1).trim()] = line.slice(0, split).trim();
  }
  const key = event.key === " " ? "Space" : event.key;
  simBusy = true;
  try {
    const data = await api("sessions/" + sim.id + "/key-step", {revision:sim.revision, key, bindings, profile_id:profile});
    drawSim(data);
    text("sim-key-status", key + " · " + (data.receipt.accepted ? "已执行" : "未执行") + " · " + keyResult(data.receipt));
  } catch(error) { fail(error); }
  finally { simBusy = false; }
});
bind("sim-key-history", "click", async () => {
  const id = sim.id;
  const rows = await api("sessions/" + id + "/key-log");
  if (sim.id !== id) return;
  text("sim-key-log", rows.map(r => (r.profile_id ? r.profile_id + " · " : "") + "第 " + r.sequence + " 帧 · " + r.key_name + " → " + (r.skill || "未绑定") + " · " + keyResult(r) + " " + r.events.join("、")).join("\n") || "暂无按键记录");
  show("sim-key-log", true);
});
