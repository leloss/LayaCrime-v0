const $ = (s) => document.querySelector(s);
const start = $("#start-button"), pause = $("#pause-button"), errorBox = $("#error-message");
const loadRun = $("#load-run-button"), savedRun = $("#saved-run");
const labelSet = $("#label-set"), labelFile = $("#label-file"), uploadLabels = $("#upload-label-set");
const resultsDialog = $("#results-dialog"), resultsBody = $("#results-body");
const promptDialog = $("#prompt-dialog");
let timer, lastId;
let probabilitySeries = [];
let probabilityTotal = 0;
let benchmarkStatus = "idle";
let modelLoading = false;
let modelReady = false;
let availableModels = [];
let deletableLabelSets = new Set();
let savedRunIds = new Set();
let activeModelId = "auto";
let lastSavedRunId = null;
let installPollTimer = null;
let installLogIndex = 0;
let installationAwaitingActivation = false;
let activeLabelName = "External labels";
let resultFilter = null;
const resultPageSize = 100;
const customModelAction = "__add-custom-model";
const pct = (v) => `${((v || 0) * 100).toFixed(1)}%`;
const clock = (v) => `${Math.floor((v || 0) / 60)}:${Math.floor((v || 0) % 60).toString().padStart(2, "0")}`;
function fail(message = "") { errorBox.textContent = message; errorBox.hidden = !message; }
function applySelectedModelPrompt() { $("#prompt-summary").textContent = PromptViewer.summary(selectedModel()); }
function openPrompt() {
  const model = selectedModel();
  $("#prompt-title").textContent = model ? `${model.label} prompt` : "Benchmark prompt";
  PromptViewer.render($("#prompt-view"), model);
  promptDialog.showModal();
}
async function request(path, body) { const response = await fetch(path, body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); const data = await response.json(); if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Benchmark request failed"); return data; }
function selectedModel() { return availableModels.find(item => item.id === $("#routing-mode").value); }
function selectedModelLabel() {
  return selectedModel()?.label || availableModels.find(item => item.id === activeModelId)?.label || "the selected model";
}
function updateIdleArticleCopy() {
  if (benchmarkStatus !== "idle" || lastId) return;
  $("#article-text").textContent = `Start the run to classify the first labeled article. Each sample is sent to ${selectedModelLabel()} without its label.`;
}
function updateModelInstallAction() {
  const model = selectedModel();
  if (!model) { $("#install-model").hidden = true; return; }
  $("#install-model").hidden = !model.installable || (model.installed && model.runtime_available);
  $("#install-model").textContent = model.installation_state === "missing" ? "Install model" : "Repair installation";
}
function requestCredentials(model) {
  const dialog = $("#credentials-dialog"), form = $("#credentials-form");
  $("#credentials-title").textContent = model.action_only ? "Add endpoint" : `Connect ${model.label}`;
  $("#azure-deployment").value = model.action_only ? "" : (model.azure_deployment || model.label.toLowerCase());
  $("#azure-api-key").value = "";
  dialog.showModal();
  $("#azure-endpoint").focus();
  return new Promise(resolve => {
    const finish = value => { form.removeEventListener("submit", submit); $("#credentials-cancel").removeEventListener("click", cancel); $("#credentials-close").removeEventListener("click", cancel); dialog.removeEventListener("cancel", cancelEvent); $("#azure-api-key").value = ""; dialog.close(); resolve(value); };
    const submit = event => { event.preventDefault(); if (!form.reportValidity()) return; finish({endpoint:$("#azure-endpoint").value.trim(), deployment:$("#azure-deployment").value.trim(), api_key:$("#azure-api-key").value}); };
    const cancel = () => finish(null);
    const cancelEvent = event => { event.preventDefault(); finish(null); };
    form.addEventListener("submit", submit); $("#credentials-cancel").addEventListener("click", cancel); $("#credentials-close").addEventListener("click", cancel); dialog.addEventListener("cancel", cancelEvent);
  });
}
function requestInstallation(model, custom = false) {
  const dialog = $("#install-dialog"), form = $("#install-form");
  clearTimeout(installPollTimer); installLogIndex = 0;
  $("#install-title").textContent = custom ? "Add custom model" : `Install ${model.label}`;
  $("#install-description").textContent = "The source is saved with the model so it can be installed, activated, and reused without changing the application catalog.";
  $("#install-name-field").hidden = !custom;
  $("#install-name").required = custom;
  $("#install-name").value = custom ? "" : model.label;
  $("#install-repo-id").value = custom ? "" : (model.repo_id || "");
  $("#install-repo-id").readOnly = !custom && model.provider !== "llama.cpp";
  $("#install-path-field").hidden = !custom && model.provider !== "llama.cpp";
  $("#install-hf-path").required = custom || model.provider === "llama.cpp";
  $("#install-hf-path").value = custom ? "" : (model.huggingface_path || model.filename || "");
  $("#install-token").value = "";
  $("#install-close").onclick = null;
  $("#install-config").hidden = false; $("#install-progress").hidden = true; $("#install-submit").hidden = false; $("#install-cancel").hidden = false; $("#install-stop").hidden = true; $("#install-done").hidden = true; $("#install-close").disabled = false; $("#install-error").hidden = true; $("#install-log").textContent = "Waiting for installation output...";
  dialog.showModal();
  return new Promise(resolve => {
    const finish = value => { form.removeEventListener("submit", submit); $("#install-cancel").removeEventListener("click", cancel); $("#install-close").removeEventListener("click", cancel); dialog.removeEventListener("cancel", cancelEvent); $("#install-token").value = ""; if (!value) dialog.close(); resolve(value); };
    const submit = event => { event.preventDefault(); if (!form.reportValidity()) return; finish({name:$("#install-name").value.trim(), repo_id:$("#install-repo-id").value.trim(), huggingface_path:$("#install-hf-path").value.trim(), huggingface_token:$("#install-token").value || null}); };
    const cancel = () => finish(null);
    const cancelEvent = event => { event.preventDefault(); finish(null); };
    form.addEventListener("submit", submit); $("#install-cancel").addEventListener("click", cancel); $("#install-close").addEventListener("click", cancel); dialog.addEventListener("cancel", cancelEvent);
  });
}
const formatBytes = value => { if (!value) return "0 B"; const units = ["B", "KB", "MB", "GB"]; const unit = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024))); return `${(value / 1024 ** unit).toFixed(unit ? 1 : 0)} ${units[unit]}`; };
function showInstallationProgress() {
  $("#install-config").hidden = true; $("#install-progress").hidden = false; $("#install-submit").hidden = true; $("#install-cancel").hidden = true; $("#install-stop").hidden = false; $("#install-done").hidden = true; $("#install-close").disabled = true; $("#install-error").hidden = true;
}
function renderInstallation(data) {
  const phase = (data.phase || data.status || "preparing").replaceAll("_", " ");
  $("#install-phase").textContent = phase;
  const progress = Number.isFinite(data.progress) ? data.progress : null;
  $("#install-percent").textContent = progress === null ? "Working..." : `${(progress * 100).toFixed(1)}%`;
  $("#install-progress-bar").className = progress === null && data.status === "running" ? "indeterminate" : "";
  $("#install-progress-bar").style.width = progress === null ? "" : `${progress * 100}%`;
  $("#install-bytes").textContent = data.total_bytes ? `${formatBytes(data.downloaded_bytes)} of ${formatBytes(data.total_bytes)}` : data.phase === "building runtime" ? "Compiling llama.cpp runtime" : "Preparing installation";
  if (data.logs?.length) { if ($("#install-log").textContent.startsWith("Waiting for")) $("#install-log").textContent = ""; data.logs.forEach(row => { $("#install-log").textContent += `${row.message}\n`; installLogIndex = Math.max(installLogIndex, row.index); }); $("#install-log").scrollTop = $("#install-log").scrollHeight; }
  if (data.status === "failed") { $("#install-error").textContent = data.error || "Installation failed"; $("#install-error").hidden = false; }
}
async function waitForInstallation() {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const data = await request(`/v1/benchmark/models/install?logs_after=${installLogIndex}`);
        renderInstallation(data);
        if (data.status === "complete") return resolve(data);
        if (["failed", "stopped"].includes(data.status)) return reject(new Error(data.error || `Installation ${data.status}`));
        installPollTimer = setTimeout(poll, 500);
      } catch (error) { reject(error); }
    };
    poll();
  });
}
function finishInstallationVerification(error = null) {
  if (!installationAwaitingActivation) return;
  installationAwaitingActivation = false;
  $("#install-percent").textContent = error ? "Failed" : "Ready";
  $("#install-progress-bar").className = "";
  $("#install-progress-bar").style.width = error ? "0%" : "100%";
  $("#install-bytes").textContent = error ? "Model could not be activated" : "Model files and runtime are ready";
  $("#install-stop").hidden = true;
  if (!error) {
    if ($("#install-dialog").open) $("#install-dialog").close();
    return;
  }
  $("#install-done").hidden = false;
  $("#install-close").disabled = false;
  $("#install-close").onclick = () => $("#install-dialog").close();
  if (error) { $("#install-error").textContent = error.message; $("#install-error").hidden = false; }
}
async function runInstallation(model, settings) {
  showInstallationProgress(); $("#model-status-label").textContent = "Installing requirements";
  try {
    const started = await request("/v1/benchmark/models/install", {model_id:model.id, ...settings}); renderInstallation(started); await waitForInstallation(); await loadModels(model.id); $("#install-percent").textContent = "Verifying..."; $("#install-progress-bar").className = "indeterminate"; $("#install-progress-bar").style.width = ""; $("#install-bytes").textContent = "Files installed; loading model and confirming GPU offload"; $("#install-stop").hidden = true; installationAwaitingActivation = true; return true;
  } catch (error) { $("#install-stop").hidden = true; $("#install-done").hidden = false; $("#install-close").disabled = false; $("#install-close").onclick = () => $("#install-dialog").close(); $("#install-error").textContent = error.message; $("#install-error").hidden = false; throw error; }
}
async function installSelectedModel() {
  const model = selectedModel();
  if (!model || !model.installable) return false;
  const settings = await requestInstallation(model);
  return settings ? runInstallation(model, settings) : false;
}
async function addCustomModel() {
  const settings = await requestInstallation({label:"Custom model"}, true);
  if (!settings) return false;
  const created = await request("/v1/benchmark/models/custom", {name:settings.name, repo_id:settings.repo_id, huggingface_path:settings.huggingface_path});
  const model = created.model;
  await loadModels(model.id);
  return runInstallation(model, settings);
}
async function refreshRuns(selected = savedRun.value) {
  const data = await request("/v1/benchmark/runs");
  savedRunIds = new Set(data.runs.map(run => run.run_id));
  savedRun.replaceChildren();
  if (!data.runs.length) savedRun.add(new Option("No saved runs", ""));
  data.runs.forEach(run => savedRun.add(new Option(`${run.name} · ${run.completed.toLocaleString()} samples · ${new Date(run.created_at).toLocaleString()}`, run.run_id)));
  if ([...savedRun.options].some(option => option.value === selected)) savedRun.value = selected;
  loadRun.disabled = !savedRun.value || benchmarkStatus === "running";
}
function bar(name, value) { $(`#${name}-value`).textContent = pct(value); $(`#${name}-bar`).style.width = pct(value); }
function audit(data = {}) {
  activeLabelName = data.label_set?.label || activeLabelName;
  $("#label-set-matrix-name").textContent = activeLabelName;
  $("#provenance").innerHTML = `<b>${(data.blind_articles || 0).toLocaleString()} blind articles</b><span>${(data.usable_gold_rows || 0).toLocaleString()} usable labels · ${activeLabelName}</span>`;
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
  $("#gold-verdict").textContent = gold?.label_name || "Unavailable"; $("#gold-source").textContent = gold ? activeLabelName : "No truth for this sample"; $("#laya-verdict").textContent = prediction.label_name;
  const isLanguageModel = ["azure", "llama.cpp"].includes(prediction.routing?.provider);
  $("#laya-confidence").textContent = isLanguageModel ? "Discrete 0% / 100% decision" : `${pct(Math.max(prediction.probabilities.negative, prediction.probabilities.positive))} class probability · ${pct(prediction.confidence)} certainty`; $("#gold-rationale").textContent = gold?.rationale || "No rationale supplied.";
  bar("negative", prediction.probabilities.negative); bar("positive", prediction.probabilities.positive); $("#negative-target").classList.toggle("truth", gold?.label === 2); $("#positive-target").classList.toggle("truth", gold?.label === 1);
}
function matrix(source, report) {
  const data = report.confusion_matrices[source], values = data.values;
  [["pp",0,0],["pn",0,1],["np",1,0],["nn",1,1]].forEach(([id,row,column]) => $(`#${source}-${id}`).textContent = values[row][column].toLocaleString());
  $(`#${source}-compared`).textContent = `${data.compared.toLocaleString()} compared`;
}
function resultLabel(value) { return value === 2 ? "negative" : "positive"; }
function renderResultDetail(item) {
  $("#result-entity").textContent = item.entity_name;
  $("#result-article-id").textContent = item.article_id;
  $("#result-elapsed").textContent = `${item.prediction.elapsed_seconds.toFixed(2)}s`;
  $("#result-truth").textContent = resultLabel(item.truth.label);
  $("#result-prediction").textContent = item.prediction.label_name;
  $("#result-article").textContent = item.article;
  $("#result-rationale").textContent = item.truth.rationale || "No rationale supplied.";
  $("#result-correctness").textContent = item.correct ? "Match" : "Mismatch";
  $("#result-correctness").className = `badge ${item.correct ? "match" : "mismatch"}`;
  $("#result-negative-value").textContent = pct(item.prediction.probabilities.negative);
  $("#result-positive-value").textContent = pct(item.prediction.probabilities.positive);
  $("#result-negative-bar").style.width = pct(item.prediction.probabilities.negative);
  $("#result-positive-bar").style.width = pct(item.prediction.probabilities.positive);
}
function clearResultDetail() {
  $("#result-entity").textContent = "No matching results";
  $("#result-article-id").textContent = "-";
  $("#result-elapsed").textContent = "-";
  $("#result-truth").textContent = "-";
  $("#result-prediction").textContent = "-";
  $("#result-article").textContent = "This confusion-matrix cell has no completed results.";
  $("#result-rationale").textContent = "No rationale supplied.";
  $("#result-correctness").textContent = "No result";
  $("#result-correctness").className = "badge";
  $("#result-negative-value").textContent = "0.0%";
  $("#result-positive-value").textContent = "0.0%";
  $("#result-negative-bar").style.width = "0%";
  $("#result-positive-bar").style.width = "0%";
}
function selectResult(row, item) {
  resultsBody.querySelectorAll("tr").forEach(candidate => candidate.classList.remove("selected"));
  row.classList.add("selected");
  renderResultDetail(item);
}
function resultRow(item) {
  const row = document.createElement("tr");
  row.tabIndex = 0;
  [item.entity_name, resultLabel(item.truth.label), item.prediction.label_name, pct(item.prediction.confidence)].forEach(value => {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  });
  row.addEventListener("click", () => selectResult(row, item));
  row.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectResult(row, item);
    }
  });
  return row;
}
async function loadMatrixResults(offset = 0) {
  const {truth, prediction} = resultFilter;
  const data = await request(`/v1/benchmark/results?truth_label=${truth}&prediction_label=${prediction}&offset=${offset}&limit=${resultPageSize}`);
  resultFilter.offset = offset;
  resultsBody.replaceChildren(...data.items.map(resultRow));
  $("#results-summary").textContent = `${data.total.toLocaleString()} results · ${activeLabelName}`;
  const first = data.total ? offset + 1 : 0;
  const last = Math.min(offset + data.items.length, data.total);
  $("#results-page").textContent = `${first.toLocaleString()}–${last.toLocaleString()} of ${data.total.toLocaleString()}`;
  $("#results-previous").disabled = offset === 0;
  $("#results-next").disabled = offset + data.items.length >= data.total;
  if (data.items.length) selectResult(resultsBody.firstElementChild, data.items[0]);
  else clearResultDetail();
}
async function openMatrixResults(truth, prediction) {
  resultFilter = {truth, prediction, offset: 0};
  $("#results-title").textContent = `Truth ${resultLabel(truth)} · Laya ${resultLabel(prediction)}`;
  resultsDialog.showModal();
  try { await loadMatrixResults(); } catch (error) { resultsDialog.close(); fail(error.message); }
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
  audit(report.audit); const running = report.status === "running";
  const interrupted = ["paused", "error"].includes(report.status) && report.completed < report.total;
  const startingRun = running && benchmarkStatus !== "running" && benchmarkStatus !== "paused";
  if (startingRun || !probabilityTotal) probabilityTotal = report.total || report.audit.blind_articles || 0;
  benchmarkStatus = report.status;
  const interruptedModelNeedsActivation = interrupted && report.model_id && report.model_id !== activeModelId;
  start.textContent = interrupted ? "Resume run" : running ? "Running" : report.status === "complete" ? "Run again" : "Start run"; start.disabled = running || modelLoading || !modelReady || interruptedModelNeedsActivation; pause.disabled = !running;
  loadRun.disabled = running || !savedRun.value;
  $("#routing-mode").disabled = running || (interrupted && !interruptedModelNeedsActivation) || modelLoading; $("#sample-limit").disabled = running || interrupted;
  labelSet.disabled = running || interrupted; labelFile.disabled = running || interrupted; uploadLabels.disabled = running || interrupted || !labelFile.files.length;
  $("#progress-count").textContent = `${report.completed.toLocaleString()} / ${report.total.toLocaleString()}`; $("#progress-percent").textContent = pct(report.progress); $("#overall-progress").style.width = pct(report.progress);
  const primary = report.metrics.gold;
  $("#accuracy").textContent = pct(primary.accuracy); $("#precision").textContent = pct(primary.precision_negative); $("#recall").textContent = pct(primary.recall_negative); $("#f1").textContent = pct(primary.f1_negative); $("#average-time").textContent = `${report.timing.average_seconds.toFixed(2)}s`;
  $("#elapsed-time").textContent = clock(report.timing.elapsed_seconds); $("#inference-time").textContent = clock(report.timing.inference_seconds); $("#throughput").textContent = `${report.timing.items_per_second.toFixed(2)} / sec`; $("#completed-count").textContent = report.completed.toLocaleString();
  const usage = report.usage || {}; $("#token-count").textContent = ((usage.input_tokens || 0) + (usage.output_tokens || 0)).toLocaleString(); $("#metered-cost").textContent = `$${(usage.cost_usd || 0).toFixed(4)}`; $("#truncated-count").textContent = (usage.truncated_examples || 0).toLocaleString();
  if (report.saved_run) { $("#auto-save-status").textContent = `Saved as ${report.saved_run.name}`; if (report.saved_run.run_id !== lastSavedRunId) { lastSavedRunId = report.saved_run.run_id; refreshRuns(report.saved_run.run_id).catch(error => fail(error.message)); } } else if (running) $("#auto-save-status").textContent = "Run in progress · results will save automatically"; else $("#auto-save-status").textContent = "Results save automatically when the run finishes";
  if (report.completed < probabilitySeries.length) probabilitySeries = [];
  report.probability_series.forEach(point => { if (point.index > probabilitySeries.length) probabilitySeries.push(point); });
  matrix("gold", report); drawConfidence(); current(report.current);
  if (report.pending) {
    $("#entity-name").textContent = report.pending.entity_name;
    $("#article-id").textContent = report.pending.article_id;
    if (typeof report.pending.article === "string") {
      $("#article-text").textContent = report.pending.article;
    } else if (!report.current) {
      $("#article-text").textContent = `${selectedModelLabel()} is classifying this article without access to its label.`;
    }
    $("#sample-state").textContent = "Classifying";
    $("#individual-progress").className = "scanning";
  } else if (running && !report.current) {
    $("#entity-name").textContent = "Preparing first sample";
    $("#article-id").textContent = "Selecting article";
    $("#article-text").textContent = `${selectedModelLabel()} is preparing to classify the first article without access to its label.`;
    $("#sample-state").textContent = "Starting";
    $("#individual-progress").className = "scanning";
  } else if (!report.current && report.status === "idle") {
    updateIdleArticleCopy();
  }
  if (report.error) fail(report.error); clearTimeout(timer); if (running) timer = setTimeout(status, 200);
}
async function status() { try { render(await (await fetch(`/v1/benchmark?series_after=${probabilitySeries.length}`)).json()); } catch (error) { fail(error.message); timer = setTimeout(status, 1500); } }
async function command(path, body = {}) { fail(); render(await request(path, body)); }
const groundTruthValue = (datasetId, labelSetId) => `${datasetId || ""}\u001f${labelSetId}`;
async function loadLabelSets(selected) { const data = await request("/v1/benchmark/label-sets"); deletableLabelSets = new Set(data.label_sets.filter(item => item.deletable).map(item => groundTruthValue(item.dataset_id, item.id))); labelSet.replaceChildren(...data.label_sets.map(item => new Option(`${item.dataset_label} · ${item.label} · ${item.rows.toLocaleString()} labels`, groundTruthValue(item.dataset_id, item.id)))); labelSet.value = selected || groundTruthValue(data.dataset?.id, data.active_label_set); const option = labelSet.selectedOptions[0]; if (option) activeLabelName = option.textContent.replace(/^.* · ([^·]+) · [\d,]+ labels$/, "$1").trim(); }
async function activateSelectedLabels() { try { const [dataset_id, label_set_id] = labelSet.value.split("\u001f"); const data = await request("/v1/benchmark/label-sets/activate", {dataset_id:dataset_id || null, label_set_id}); lastId = null; probabilitySeries = []; probabilityTotal = 0; activeLabelName = data.audit.label_set.label; await refreshRuns(); audit(data.audit); await status(); } catch (error) { fail(error.message); await loadLabelSets(); } }
async function uploadLabelFile() { try { const file = labelFile.files[0]; if (!file) return; uploadLabels.disabled = true; const data = await request("/v1/benchmark/label-sets/upload", {name:$("#label-set-name").value.trim() || file.name.replace(/\.jsonl$/i, ""), content:await file.text()}); await loadLabelSets(data.active_label_set); activeLabelName = data.audit.label_set.label; audit(data.audit); labelFile.value = ""; $("#label-set-name").value = ""; } catch (error) { fail(error.message); } finally { uploadLabels.disabled = !labelFile.files.length; } }
start.onclick = async () => { try { const value = $("#sample-limit").value; await command("/v1/benchmark/start", {routing_mode:$("#routing-mode").value, limit:value ? Number(value) : null}); } catch (error) { fail(error.message); } };
pause.onclick = async () => { try { await command("/v1/benchmark/pause"); } catch (error) { fail(error.message); } };
$("#reset-button").onclick = async () => { try { lastId = null; probabilitySeries = []; probabilityTotal = 0; await command("/v1/benchmark/reset"); } catch (error) { fail(error.message); } };
loadRun.onclick = async () => { try { if (!savedRun.value) return; lastId = null; probabilitySeries = []; probabilityTotal = 0; await command("/v1/benchmark/runs/load", {run_id:savedRun.value}); await loadLabelSets(); } catch (error) { fail(error.message); } };
savedRun.onchange = () => { loadRun.disabled = !savedRun.value || benchmarkStatus === "running"; };
labelSet.onchange = activateSelectedLabels;
labelFile.onchange = () => { uploadLabels.disabled = !labelFile.files.length || benchmarkStatus === "running" || benchmarkStatus === "paused"; };
uploadLabels.onclick = uploadLabelFile;
document.querySelectorAll(".matrix-cell").forEach(cell => cell.addEventListener("click", () => openMatrixResults(Number(cell.dataset.truth), Number(cell.dataset.prediction))));
$("#results-close").onclick = () => resultsDialog.close();
$("#results-previous").onclick = () => loadMatrixResults(Math.max(0, resultFilter.offset - resultPageSize)).catch(error => fail(error.message));
$("#results-next").onclick = () => loadMatrixResults(resultFilter.offset + resultPageSize).catch(error => fail(error.message));
$("#see-prompt").onclick = openPrompt;
$("#prompt-close").onclick = () => promptDialog.close();
$("#prompt-done").onclick = () => promptDialog.close();
async function health() { try { const data = await (await fetch("/health")).json(), ready = data.status === "ok"; $("#model-health").className = `model-health ${ready ? "ready" : data.status === "warming" ? "" : "degraded"}`; $("#model-status-label").textContent = ready ? "Model ready" : data.status === "warming" ? "Loading catalog" : "Laya unavailable"; await loadModels(); if (data.status === "warming") setTimeout(health, 1500); if (data.error && !availableModels.length) fail(data.error); } catch (error) { fail(error.message); } }
async function loadModels(selectedId = null) {
  const response = await fetch("/v1/benchmark/models"); if (!response.ok) return;
  const data = await response.json();
  const select = $("#routing-mode"), selected = selectedId || data.active_model || select.value;
  availableModels = data.models; activeModelId = data.active_model || "auto";
  const endpointAction = data.models.find(model => model.provider === "azure" && model.action_only);
  const groupSpecs = [
    ["Decision Models", model => model.category === "decision", null],
    ["Cloud LLMs", model => model.provider === "azure" && !model.action_only, endpointAction ? new Option(endpointAction.action_label, endpointAction.id) : null],
    ["Self-hosted LLMs", model => model.provider === "llama.cpp", new Option("Add custom model…", customModelAction)],
  ];
  const groups = groupSpecs.map(([label, includes, action]) => {
    const group = document.createElement("optgroup"); group.label = label;
    const models = data.models.filter(includes);
    models.forEach(model => group.append(new Option(`${model.label}${model.installable && !model.installed ? model.installation_state === "missing" ? " · install required" : " · repair required" : !model.installable && model.runtime_available === false ? " · setup required" : ""}`, model.id)));
    if (!models.length && !action) { const unavailable = new Option(`${label} unavailable`, `__${label.toLowerCase().replaceAll(" ", "-")}-unavailable`); unavailable.disabled = true; group.append(unavailable); }
    if (action) group.append(action);
    return group;
  });
  const values = data.models.map(model => model.id); const content = [...groups];
  if (!values.includes(selected)) { const placeholder = new Option("Select a model", "", true, true); placeholder.disabled = true; content.unshift(placeholder); }
  select.replaceChildren(...content); if ([...select.options].some(option => option.value === selected)) select.value = selected;
  applySelectedModelPrompt(); updateModelInstallAction(); modelReady = availableModels.some(model => model.id === activeModelId); start.disabled = !modelReady;
}
async function activateSelectedModel() { if ($("#routing-mode").value === customModelAction) { try { if (await addCustomModel()) return activateSelectedModel(); await loadModels(activeModelId); } catch (error) { await loadModels(activeModelId); fail(error.message); } return; } applySelectedModelPrompt(); updateModelInstallAction(); const model = selectedModel(); if (!model) return; modelLoading = true; modelReady = false; $("#routing-mode").disabled = true; start.disabled = true; $("#model-health").className = "model-health"; $("#model-status-label").textContent = "Loading model"; fail(); try { if (model.installable && (!model.installed || !model.runtime_available) && !(await installSelectedModel())) { $("#routing-mode").value = activeModelId; updateModelInstallAction(); modelReady = true; return; } const credentials = model.credential_required ? await requestCredentials(model) : {}; if (model.credential_required && !credentials) { $("#routing-mode").value = activeModelId; updateModelInstallAction(); modelReady = true; return; } await request("/v1/benchmark/models/activate", {model_id:model.id, ...credentials}); activeModelId = model.id; modelReady = true; finishInstallationVerification(); $("#model-health").className = "model-health ready"; $("#model-status-label").textContent = "Model ready"; } catch (error) { finishInstallationVerification(error); $("#routing-mode").value = activeModelId; updateModelInstallAction(); $("#model-health").className = "model-health degraded"; $("#model-status-label").textContent = "Model unavailable"; fail(error.message); } finally { modelLoading = false; $("#routing-mode").disabled = benchmarkStatus === "running" || benchmarkStatus === "paused"; start.disabled = benchmarkStatus === "running" || !modelReady; } }
async function syncActiveModel() { if (modelLoading) return; try { const response = await fetch("/v1/benchmark/models"); if (!response.ok) return; const data = await response.json(); availableModels = data.models; activeModelId = data.active_model || "auto"; const select = $("#routing-mode"); if ([...select.options].some(option => option.value === activeModelId)) select.value = activeModelId; applySelectedModelPrompt(); updateModelInstallAction(); modelReady = true; select.disabled = benchmarkStatus === "running" || benchmarkStatus === "paused"; start.disabled = benchmarkStatus === "running"; $("#model-health").className = "model-health ready"; $("#model-status-label").textContent = "Model ready"; } catch { $("#model-health").className = "model-health degraded"; $("#model-status-label").textContent = "Model unavailable"; } }
$("#routing-mode").onchange = () => { updateIdleArticleCopy(); activateSelectedModel(); };
new ResourcePicker($("#routing-mode"), {
  isDeletable: id => availableModels.some(model => model.id === id && model.deletable),
  onDelete: async id => {
    if (activeModelId === id) {
      $("#routing-mode").value = "auto";
      await activateSelectedModel();
      if (activeModelId === id) throw new Error("Select another active model before deleting this one.");
    }
    await request("/v1/benchmark/models/delete", {model_id:id});
    await loadModels();
  },
  onError: error => fail(error.message),
});
new ResourcePicker(labelSet, {
  isDeletable: id => deletableLabelSets.has(id),
  onDelete: async id => {
    const [dataset_id, label_set_id] = id.split("\u001f");
    await request("/v1/benchmark/label-sets/delete", {dataset_id:dataset_id || null, label_set_id});
    await loadLabelSets();
    await status();
  },
  onError: error => fail(error.message),
});
new ResourcePicker(savedRun, {
  isDeletable: id => savedRunIds.has(id),
  onDelete: async id => {
    await request("/v1/benchmark/runs/delete", {run_id:id});
    await refreshRuns();
  },
  onError: error => fail(error.message),
});
$("#install-model").onclick = () => installSelectedModel().then(installed => { if (installed) return activateSelectedModel(); updateModelInstallAction(); }).catch(error => fail(error.message));
$("#install-stop").onclick = () => request("/v1/benchmark/models/install/stop", {}).then(renderInstallation).catch(error => fail(error.message));
$("#install-done").onclick = () => $("#install-dialog").close();
window.addEventListener("resize", drawConfidence); health(); window.setInterval(() => syncActiveModel().then(updateIdleArticleCopy), 1500); loadLabelSets().catch(error => fail(error.message)); refreshRuns().catch(error => fail(error.message)); status();
