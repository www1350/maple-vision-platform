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
let trialEvidenceLoaded = false;

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
  if (!["runs","recordings","trial","simulator","routes","keys","content"].includes(view)) view = "runs";
  activeView = view; stopPlay(); stopRecordingPlay();
  document.querySelectorAll(".view").forEach(e=>e.classList.toggle("hidden", e.id !== "view-"+view));
  document.querySelectorAll("[data-view]").forEach(e=>e.classList.toggle("active", e.dataset.view===view));
  text("view-label", {runs:"任务与回放", recordings:"真实录像复核", trial:"试用验收账本", simulator:"单步模拟器", routes:"地图路线", keys:"角色键位", content:"内容与模型"}[view]);
  if (view === "recordings" && !recordingsLoaded) loadRecordings().catch(fail);
  if (view === "trial" && !trialEvidenceLoaded) loadTrialEvidence().catch(fail);
}

async function loadTrialEvidence() {
  const data = await api("trial-evidence");
  trialEvidenceLoaded = true;
  show("trial-empty", !data.configured);
  show("trial-workspace", data.configured);
  if (!data.configured) return;
  const laneNames = {real_recording:"真实录像", synthetic_mechanics:"合成机制", role_binding:"角色键位", ammunition:"耗镖"};
  const cards = el("trial-lanes"); cards.replaceChildren();
  for (const [lane, facts] of Object.entries(data.evidence_lanes)) {
    const card = document.createElement("section"); card.className = "panel content-group";
    const heading = document.createElement("h2"); heading.textContent = laneNames[lane] || lane; card.append(heading);
    for (const [key, value] of Object.entries(facts)) {
      const item = document.createElement("div"); item.className = "content-item";
      const strong = document.createElement("strong"); strong.textContent = key;
      const small = document.createElement("small"); small.textContent = typeof value === "object" ? pretty(value) : String(value);
      item.append(strong, small); card.append(item);
    }
    cards.append(card);
  }
  text("trial-status", data.verified ? "四栏已复验" : "未通过");
  text("trial-json", pretty(data));
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
  const seek=el("recording-source-frame");
  seek.min=start; seek.max=end; seek.value=start;
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
    if (profile) {
      body.input_profile = storedKeyProfile(profile);
      const receipt = localStorage.getItem(keyConfirmationId(profile));
      if (receipt === null) throw new Error("该角色缺少键位确认凭证，请在角色键位页重新核对并保存。");
      try { body.input_confirmation = JSON.parse(receipt); }
      catch { throw new Error("保存的键位确认凭证无效，请在角色键位页重新核对并保存。"); }
    }
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
bind("recording-seek","submit",()=>{
  if (!recordingDetail) return;
  const source=el("recording-source-frame").valueAsNumber;
  const start=recordingDetail.viewer_kind==="coverage"
    ? recordingDetail.decoded_source_range[0]
    : (recordingDetail.source_start_frame || 0);
  const index=source-start;
  if (!Number.isSafeInteger(index) || !recordingDetail.frames[index]) return;
  stopRecordingPlay(); el("recording-range").value=index; drawRecordingFrame();
});
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
function keyConfirmationId(profile) {
  return "maple.keyconfirmation.v1." + profile;
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
function downloadJson(value, name) {
  const url = URL.createObjectURL(new Blob([pretty(value) + "\n"], {type:"application/json"}));
  const link = document.createElement("a");
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
bind("keys-export", "click", async () => {
  const profile = storedKeyProfile(el("keys-character").value.trim());
  const validated = await api("input-profile", profile);
  downloadJson(validated, "input-bindings.json");
});
bind("keys-export-confirmation", "click", () => {
  const profile = el("keys-character").value.trim();
  const source = localStorage.getItem(keyConfirmationId(profile));
  if (source === null) { text("keys-status", "请先核对并保存此角色的键位确认凭证。"); return; }
  try { downloadJson(JSON.parse(source), "input-binding-confirmation.json"); }
  catch { text("keys-status", "保存的确认凭证无效，请重新核对并保存。"); }
});
let keyEditRevision = 0, keyImageRevision = 0, keyImageEvidence = null, keyImageBusy = false;
let rowAnchors = null, rowReviewDraft = null, keyGridReview = null;
bind("keys-row-start", "click", () => {
  rowAnchors = []; rowReviewDraft = null; keyGridReview = null;
  el("keys-row-crops").replaceChildren();
  show("keys-row-confirm", false);
  text("keys-row-status", "依次点击截图中的 A、L 键中心。");
});
function orderedRowKeys(keys) {
  return [..."ASDFGHJKL"].filter(key => keys.has(key));
}
function updateRowDraftStatus() {
  const keys = orderedRowKeys(rowReviewDraft.confirmed);
  text("keys-row-status", "当前选择 " + (keys.join("、") || "0 项") +
    "。点九格可纠正，再确认复核结果。");
}
bind("keys-row-confirm", "click", () => {
  if (!rowReviewDraft) return;
  keyGridReview = {
    schema_version:1,
    row:"ASDFGHJKL",
    anchors:rowReviewDraft.anchors,
    algorithm_candidate_keys:rowReviewDraft.algorithm,
    confirmed_content_keys:orderedRowKeys(rowReviewDraft.confirmed),
    user_reviewed:true
  };
  show("keys-row-confirm", false);
  text("keys-row-status", "九格复核已确认，将写入本次键位确认凭证。");
});
el("keys-image").addEventListener("click", event => {
  if (!rowAnchors || !keyImageEvidence) return;
  const image = event.currentTarget, rect = image.getBoundingClientRect();
  rowAnchors.push({
    x:(event.clientX - rect.left) * image.naturalWidth / rect.width,
    y:(event.clientY - rect.top) * image.naturalHeight / rect.height
  });
  if (rowAnchors.length === 1) { text("keys-row-status", "现在点击 L 键中心。"); return; }
  const [a, l] = rowAnchors;
  rowAnchors = null;
  const pitch = (l.x - a.x) / 8, size = Math.round(pitch * 0.9);
  if (pitch < 8 || pitch > 160 || Math.abs(l.y - a.y) > pitch / 3 ||
      a.x - size / 2 < 0 || l.x + size / 2 > image.naturalWidth ||
      Math.min(a.y,l.y) - size / 2 < 0 || Math.max(a.y,l.y) + size / 2 > image.naturalHeight) {
    text("keys-row-status", "定位超出范围，请重新选择同一行的 A、L 中心。"); return;
  }
  const keys = "ASDFGHJKL", marked = [];
  for (let i = 0; i < keys.length; i++) {
    const tile = document.createElement("canvas");
    tile.width = tile.height = size;
    const ctx = tile.getContext("2d");
    ctx.drawImage(image, Math.round(a.x + pitch*i - size/2),
      Math.round(a.y + (l.y-a.y)*i/8 - size/2), size, size, 0, 0, size, size);
    // Sample below the printed key name, away from the keycap border.
    const pixels = ctx.getImageData(Math.floor(size*.19), Math.floor(size*.52),
      Math.ceil(size*.62), Math.ceil(size*.26)).data;
    let dark = 0;
    for (let p = 0; p < pixels.length; p += 4) {
      if (pixels[p+3] === 255 && Math.max(pixels[p],pixels[p+1],pixels[p+2]) < 195) dark++;
    }
    const candidate = dark * 4 / pixels.length >= .35;
    if (candidate) marked.push(keys[i]);
    const label = document.createElement("button");
    label.type = "button";
    label.className = "key-review-cell";
    tile.setAttribute("aria-hidden", "true");
    label.append(tile, keys[i] + (candidate ? " · 待核" : ""));
    el("keys-row-crops").append(label);
  }
  rowReviewDraft = {
    anchors:[
      {key:"A",x:a.x,y:a.y},
      {key:"L",x:l.x,y:l.y}
    ],
    algorithm:marked,
    confirmed:new Set(marked)
  };
  for (const cell of el("keys-row-crops").children) {
    const key = cell.textContent.trim()[0];
    cell.classList.toggle("selected", rowReviewDraft.confirmed.has(key));
    cell.addEventListener("click", () => {
      keyGridReview = null;
      if (rowReviewDraft.confirmed.has(key)) rowReviewDraft.confirmed.delete(key);
      else rowReviewDraft.confirmed.add(key);
      cell.classList.toggle("selected", rowReviewDraft.confirmed.has(key));
      show("keys-row-confirm", true);
      updateRowDraftStatus();
    });
  }
  show("keys-row-confirm", true);
  text("keys-row-status", "深色内容候选 " + (marked.join("、") || "0 项") +
    "。点九格可纠正，再确认复核结果。");
});
el("keys-detail").addEventListener("change", () => {
  el("keys-image").style.maxWidth = el("keys-detail").checked ? "none" : "100%";
});
function clearKeyImage() {
  keyImageRevision++; keyImageEvidence = null; keyImageBusy = false;
  const image = el("keys-image");
  if (image.hasAttribute("src")) URL.revokeObjectURL(image.src);
  image.removeAttribute("src");
  image.style.maxWidth = "100%";
  el("keys-detail").checked = false;
  rowAnchors = null; rowReviewDraft = null; keyGridReview = null;
  el("keys-row-crops").replaceChildren();
  text("keys-row-status", "");
  show("keys-row-confirm", false);
  show("keys-row-start", false);
  show("keys-detail-toggle", false);
  el("keys-image-file").value = "";
  show("keys-image", false);
}
bind("keys-character", "input", () => {
  clearKeyImage();
  const saved = localStorage.getItem(keyProfileId());
  el("keys-list").value = saved || "";
  const receipt = localStorage.getItem(keyConfirmationId(el("keys-character").value.trim()));
  text("keys-status", saved ? "已载入此角色上次确认的键位" + (receipt ? "和确认凭证。" : "，请与游戏设置核对。") : "");
});
bind("keys-list", "input", () => text("keys-status", "修改尚未保存。"));
bind("keys-image-file", "change", async () => {
  const file = el("keys-image-file").files[0];
  clearKeyImage();
  if (!file) return;
  if (!["image/png","image/jpeg","image/webp"].includes(file.type)) {
    text("keys-status", "请选择 PNG、JPEG 或 WebP 图片。"); return;
  }
  if (file.size < 1 || file.size > 16777216) {
    text("keys-status", "键位截图必须在 16 MiB 以内。"); return;
  }
  const revision = keyImageRevision;
  keyImageBusy = true;
  const image = el("keys-image");
  image.src = URL.createObjectURL(file); show("keys-image", true);
  try {
    await image.decode();
    if (revision !== keyImageRevision) return;
    if (!image.naturalWidth || !image.naturalHeight || image.naturalWidth > 8192 ||
        image.naturalHeight > 8192 || image.naturalWidth * image.naturalHeight > 33554432) {
      throw new Error("键位截图尺寸超过 32 百万像素限制。");
    }
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    if (revision !== keyImageRevision) return;
    keyImageEvidence = {
      sha256:[...new Uint8Array(digest)].map(value => value.toString(16).padStart(2,"0")).join(""),
      bytes:file.size, media_type:file.type, width:image.naturalWidth, height:image.naturalHeight
    };
    show("keys-detail-toggle", true);
    show("keys-row-start", true);
    text("keys-status", "截图已在浏览器内计算哈希；请人工核对技能和键位后保存。未自动识别技能。");
  } catch (error) {
    if (revision === keyImageRevision) { clearKeyImage(); text("keys-status", "截图读取失败：" + error.message); }
  } finally { if (revision === keyImageRevision) keyImageBusy = false; }
});
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
  if (keyImageBusy) { text("keys-status", "截图仍在计算哈希，请稍后再保存。"); return; }
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
  if (!bindings.length) {
    localStorage.setItem(keyProfileId(), "");
    localStorage.removeItem(keyConfirmationId(profile));
    text("keys-status", "已解除此角色的全部绑定；空表不生成确认凭证。");
    return;
  }
  const imageRevision = keyImageRevision;
  let reviewedGrid = false;
  try {
    const inputProfile = {schema_version:1, profile_id:profile, bindings};
    const receiptRequest = {
      schema_version:keyGridReview ? 2 : 1,
      profile:inputProfile,
      comparison_source:keyImageEvidence ? "browser_selected_image_digest" : "live_window_manual_comparison",
      source_image:keyImageEvidence,
      user_confirmed:true,
      automatic_skill_recognition:false
    };
    if (keyGridReview) receiptRequest.key_grid_review = keyGridReview;
    const receipt = await api("input-profile/confirmation", receiptRequest);
    if (revision !== keyEditRevision || imageRevision !== keyImageRevision) return;
    reviewedGrid = Boolean(receipt.key_grid_review);
    localStorage.setItem(keyProfileId(), result.join("\n"));
    localStorage.setItem(keyConfirmationId(profile), JSON.stringify(receipt));
  } catch (error) {
      if (revision !== keyEditRevision || imageRevision !== keyImageRevision) return;
      const messages = {
        "input profile contains duplicate normalized keys":"同一个键位出现了两次，请核对。",
        "input profile contains duplicate normalized skills":"同一个技能出现了两次，请核对。"
      };
      text("keys-status", messages[error.message] || "键位未保存：" + error.message);
      return;
  }
  text("keys-status", "已确认并保存 " + result.length + " 项绑定和独立确认凭证" +
    (reviewedGrid ? "，九格人工复核已写入" : "") + "；未自动识别技能。");
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
