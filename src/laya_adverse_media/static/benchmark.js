const $ = (s) => document.querySelector(s);
const start = $("#start-button"), pause = $("#pause-button"), errorBox = $("#error-message");
const saveRun = $("#save-run-button"), loadRun = $("#load-run-button"), savedRun = $("#saved-run");
let timer, lastId;
let probabilitySeries = [];
let probabilityTotal = 0;
let benchmarkStatus = "idle";
const pct = (v) => `${((v || 0) * 100).toFixed(1)}%`;
const clock = (v) => `${Math.floor((v || 0) / 60)}:${Math.floor((v || 0) % 60).toString().padStart(2, "0")}`;
function fail(message = "") { errorBox.textContent = message; errorBox.hidden = !message; }
async function request(path, body) { const response = await fetch(path, body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); const data = await response.json(); if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Benchmark request failed"); return data; }
async function refreshRuns(selected = savedRun.value) {
  const data = await request("/v1/benchmark/runs");
  savedRun.replaceChildren();
  if (!data.runs.length) savedRun.add(new Option("No saved runs", ""));
  data.runs.forEach(run => savedRun.add(new Option(`${run.name} · ${run.completed.toLocaleString()} samples · ${new Date(run.created_at).toLocaleString()}`, run.run_id)));
  if ([...savedRun.options].some(option => option.value === selected)) savedRun.value = selected;
  loadRun.disabled = !savedRun.value || benchmarkStatus === "running";
}
function bar(name, value) { $(`#${name}-value`).textContent = pct(value); $(`#${name}-bar`).style.width = pct(value); }
function audit(data = {}) {
  const sources = Object.entries(data.annotators || {}).map(([key, value]) => `${key}: ${value.toLocaleString()}`).join(" / ");
  $("#provenance").innerHTML = `<b>${(data.blind_articles || 0).toLocaleString()} articles / ${(data.human_gold_rows || 0).toLocaleString()} source dispositions</b><span>${(data.usable_gold_rows || 0).toLocaleString()} Terra annotations${sources ? ` / ${sources}` : ""}</span>`;
  if (data.error) fail(data.error);
}
function current(item) {
  if (!item) return;
  if (item.article_id !== lastId) {
    lastId = item.article_id;
    $("#entity-name").textContent = item.entity_name; $("#article-id").textContent = item.article_id; $("#article-text").textContent = item.article;
  }
  const prediction = item.prediction, gold = item.gold;
  $("#sample-time").textContent = `${item.elapsed_seconds.toFixed(2)}s`; $("#sample-state").textContent = "Complete"; $("#individual-progress").className = "done";
  $("#correctness").textContent = gold ? (item.correct ? "Match" : "Mismatch") : "No truth"; $("#correctness").className = `badge ${gold ? (item.correct ? "match" : "mismatch") : ""}`;
  $("#gold-verdict").textContent = gold?.label_name || "Unavailable"; $("#gold-annotator").textContent = gold ? `${gold.annotator || "Unknown annotator"} (${gold.source})` : "No truth for this sample"; $("#laya-verdict").textContent = prediction.label_name;
  $("#laya-confidence").textContent = `${pct(Math.max(prediction.probabilities.negative, prediction.probabilities.positive))} winning probability`; $("#gold-rationale").textContent = gold?.rationale || "No rationale supplied.";
  bar("negative", prediction.probabilities.negative); bar("positive", prediction.probabilities.positive); $("#negative-target").classList.toggle("truth", gold?.label === 2); $("#positive-target").classList.toggle("truth", gold?.label === 1);
}
function matrix(source, report) {
  const data = report.confusion_matrices[source], values = data.values;
  [["pp",0,0],["pn",0,1],["np",1,0],["nn",1,1]].forEach(([id,row,column]) => $(`#${source}-${id}`).textContent = values[row][column].toLocaleString());
  $(`#${source}-compared`).textContent = `${data.compared.toLocaleString()} compared`;
}
function drawConfidence() {
  const canvas = $("#confidence-chart"), ratio = window.devicePixelRatio || 1, width = canvas.clientWidth, height = 250;
  canvas.width = Math.max(1, Math.floor(width * ratio)); canvas.height = height * ratio;
  const context = canvas.getContext("2d"); context.scale(ratio, ratio); context.clearRect(0, 0, width, height);
  const left = 46, right = 12, top = 12, bottom = 28, plotWidth = width - left - right, plotHeight = height - top - bottom;
  context.font = "10px DM Sans"; context.fillStyle = "#68716d"; context.strokeStyle = "#d8ded8"; context.lineWidth = 1;
  [100, 0].forEach(value => { const y = top + ((100 - value) / 100) * plotHeight; context.beginPath(); context.moveTo(left, y); context.lineTo(width - right, y); context.stroke(); context.fillText(`${value}%`, 4, y + 3); });
  const midpoint = top + plotHeight / 2;
  const drawThreshold = () => {
    context.save(); context.strokeStyle = "#4f5753"; context.setLineDash([5, 4]); context.beginPath(); context.moveTo(left, midpoint); context.lineTo(width - right, midpoint); context.stroke(); context.restore();
    context.fillStyle = "#4f5753"; context.fillText("50%", 4, midpoint + 3);
  };
  if (!probabilitySeries.length) { drawThreshold(); context.fillText("Probability history appears as samples complete", left + 12, top + plotHeight / 2 - 9); return; }
  const total = Math.max(1, probabilityTotal, probabilitySeries[probabilitySeries.length - 1].index);
  const bins = new Map();
  probabilitySeries.forEach(point => {
    const column = Math.min(Math.floor(plotWidth) - 1, Math.floor((point.index - 1) / total * plotWidth));
    const bin = bins.get(column) || {negative: 0, positive: 0, count: 0};
    bin.negative += Math.abs(point.negative); bin.positive += point.positive; bin.count += 1;
    bins.set(column, bin);
  });
  bins.forEach((bin, column) => {
    const negativeHeight = bin.negative / bin.count / 100 * plotHeight;
    const positiveHeight = bin.positive / bin.count / 100 * plotHeight;
    context.fillStyle = "#be4037"; context.fillRect(left + column, top, 1, negativeHeight);
    context.fillStyle = "#167454"; context.fillRect(left + column, top + negativeHeight, 1, positiveHeight);
  });
  const latest = probabilitySeries[probabilitySeries.length - 1];
  const cursorX = left + Math.min(plotWidth, latest.index / total * plotWidth);
  context.save(); context.strokeStyle = "#202622"; context.lineWidth = 1; context.beginPath(); context.moveTo(cursorX, top); context.lineTo(cursorX, top + plotHeight); context.stroke(); context.restore();
  drawThreshold();
  const latestText = `Sample ${latest.index.toLocaleString()}  N ${Math.abs(latest.negative).toFixed(1)}%  P ${latest.positive.toFixed(1)}%`;
  const latestWidth = context.measureText(latestText).width + 12;
  context.fillStyle = "rgba(255, 255, 255, 0.92)"; context.fillRect(width - right - latestWidth, top + 6, latestWidth, 20);
  context.fillStyle = "#202622"; context.textAlign = "right"; context.fillText(latestText, width - right - 6, top + 20); context.textAlign = "left";
  context.fillStyle = "#68716d"; context.fillText("Sample 1", left, height - 7); context.textAlign = "right"; context.fillText(`Sample ${total.toLocaleString()}`, width - right, height - 7); context.textAlign = "left";
}
function render(report) {
  audit(report.audit); const running = report.status === "running", paused = report.status === "paused";
  const startingRun = running && benchmarkStatus !== "running" && benchmarkStatus !== "paused";
  if (startingRun || !probabilityTotal) probabilityTotal = report.total || report.audit.blind_articles || 0;
  benchmarkStatus = report.status;
  start.textContent = paused ? "Resume run" : running ? "Running" : report.status === "complete" ? "Run again" : "Start run"; start.disabled = running; pause.disabled = !running;
  saveRun.disabled = report.completed === 0; loadRun.disabled = running || !savedRun.value;
  $("#routing-mode").disabled = $("#sample-limit").disabled = running || paused;
  $("#progress-count").textContent = `${report.completed.toLocaleString()} / ${report.total.toLocaleString()}`; $("#progress-percent").textContent = pct(report.progress); $("#overall-progress").style.width = pct(report.progress);
  const primary = report.metrics.gpt.compared ? report.metrics.gpt : report.metrics.human;
  $("#accuracy").textContent = pct(primary.accuracy); $("#precision").textContent = pct(primary.precision_negative); $("#recall").textContent = pct(primary.recall_negative); $("#f1").textContent = pct(primary.f1_negative); $("#average-time").textContent = `${report.timing.average_seconds.toFixed(2)}s`;
  $("#elapsed-time").textContent = clock(report.timing.elapsed_seconds); $("#inference-time").textContent = clock(report.timing.inference_seconds); $("#throughput").textContent = `${report.timing.items_per_second.toFixed(2)} / sec`; $("#completed-count").textContent = report.completed.toLocaleString();
  if (report.completed < probabilitySeries.length) probabilitySeries = [];
  report.probability_series.forEach(point => { if (point.index > probabilitySeries.length) probabilitySeries.push(point); });
  matrix("human", report); matrix("gpt", report); drawConfidence(); current(report.current);
  if (report.pending) { $("#sample-state").textContent = `Classifying next: ${report.pending.entity_name}`; $("#individual-progress").className = "scanning"; }
  if (report.error) fail(report.error); clearTimeout(timer); timer = setTimeout(status, running ? 200 : 1000);
}
async function status() { try { render(await (await fetch(`/v1/benchmark?series_after=${probabilitySeries.length}`)).json()); } catch (error) { fail(error.message); timer = setTimeout(status, 1500); } }
async function command(path, body = {}) { fail(); render(await request(path, body)); }
start.onclick = async () => { try { const value = $("#sample-limit").value; await command("/v1/benchmark/start", {routing_mode:$("#routing-mode").value, limit:value ? Number(value) : null}); } catch (error) { fail(error.message); } };
pause.onclick = async () => { try { await command("/v1/benchmark/pause"); } catch (error) { fail(error.message); } };
$("#reset-button").onclick = async () => { try { lastId = null; probabilitySeries = []; probabilityTotal = 0; await command("/v1/benchmark/reset"); } catch (error) { fail(error.message); } };
saveRun.onclick = async () => { try { const name = $("#run-name").value.trim(); if (!name) { $("#run-name").focus(); throw new Error("Enter a name for this run"); } const data = await request("/v1/benchmark/runs/save", {name}); $("#run-name").value = ""; await refreshRuns(data.saved.run_id); } catch (error) { fail(error.message); } };
loadRun.onclick = async () => { try { if (!savedRun.value) return; lastId = null; probabilitySeries = []; probabilityTotal = 0; await command("/v1/benchmark/runs/load", {run_id:savedRun.value}); } catch (error) { fail(error.message); } };
savedRun.onchange = () => { loadRun.disabled = !savedRun.value || benchmarkStatus === "running"; };
async function health() { try { const data = await (await fetch("/health")).json(), ready = data.status === "ok"; $("#service-state").className = `service-state ${ready ? "ready" : data.status === "warming" ? "" : "degraded"}`; $("#status-label").textContent = ready ? "Model ready" : data.status === "warming" ? "Warming model" : "Model unavailable"; if (data.status === "warming") setTimeout(health, 1500); if (data.error) fail(data.error); } catch (error) { fail(error.message); } }
window.addEventListener("resize", drawConfidence); health(); refreshRuns().catch(error => fail(error.message)); status();
