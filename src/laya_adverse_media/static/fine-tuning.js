const $ = (selector) => document.querySelector(selector);
const prepareButton = $("#prepare-button");
const trainButton = $("#train-button");
const stopButton = $("#stop-button");
const loadTrainingButton = $("#load-training");
const liveTrainingButton = $("#live-training");
const reportSelect = $("#training-report");
const logView = $("#process-log");
const errorBox = $("#error-message");
const hubDialog = $("#hub-dialog");
const hubForm = $("#hub-form");
let timer;
let inferenceModelTimer;
let trainingOwnsGpu = false;
let defaultsApplied = false;
let latestLogIndex = 0;
let currentJobStartedAt = null;
let datasetPresets = [];
let modelPresets = [];
let reportIds = new Set();
let trainingStrategies = [];
let authorTrainingPreset = null;
let configurationDisabled = false;
let currentLossHistory = [];
let actionError = "";
let lastStatus = "idle";
let promptQuestion = "";
let promptCriteria = [];
const promptDialog = $("#prompt-dialog");
const hasNumber = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
const percent = value => hasNumber(value) ? `${(Number(value) * 100).toFixed(1)}%` : "-";
const decimal = value => hasNumber(value) ? Number(value).toFixed(4) : "-";

function fail(message = "") {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

async function refreshInferenceModel() {
  clearTimeout(inferenceModelTimer);
  try {
    if (trainingOwnsGpu) {
      $("#model-health").className = "model-health";
      $("#model-status-label").textContent = "Inference released · training owns GPU";
      $("#model-health").title = "Inference models were unloaded before training started";
      return;
    }
    const healthResponse = await fetch("/health");
    const health = await healthResponse.json();
    if (health.status !== "ok") {
      const warming = health.status === "warming";
      $("#model-health").className = `model-health ${warming ? "" : "degraded"}`;
      $("#model-status-label").textContent = warming ? "Inference model loading" : "Inference model unavailable";
      $("#model-health").title = health.error || "";
      return;
    }
    const response = await fetch("/v1/models");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail?.message || "Inference model unavailable");
    if (data.active_model === "loading") {
      $("#model-health").className = "model-health";
      $("#model-status-label").textContent = "Inference model loading";
    } else if (data.active_model === "unavailable") {
      $("#model-health").className = "model-health degraded";
      $("#model-status-label").textContent = "Inference model unavailable";
    } else {
      const active = data.active_model || "auto";
      const model = data.models.find(item => item.id === active);
      const label = model?.label || active;
      $("#model-health").className = "model-health ready";
      $("#model-status-label").textContent = `Inference: ${label}`;
      $("#model-health").title = `Active inference model: ${label}`;
    }
  } catch (error) {
    $("#model-health").className = "model-health degraded";
    $("#model-status-label").textContent = "Inference unavailable";
    $("#model-health").title = error.message;
  } finally {
    inferenceModelTimer = setTimeout(refreshInferenceModel, 1500);
  }
}

async function request(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    const validation = Array.isArray(detail)
      ? detail.map(item => `${item.loc?.slice(1).join(".") || "request"}: ${item.msg}`).join("; ")
      : null;
    throw new Error(typeof detail === "string" ? detail : validation || detail?.message || "Fine-tuning request failed");
  }
  return data;
}

function value(id) { return $(`#${id}`).value.trim(); }
function number(id) { return Number($(`#${id}`).value); }

const partitionDescriptions = {
  holdout_80_20: "Default deterministic 80% training / 20% calibration split. Connected entities and shared articles stay together.",
  holdout_70_30: "Deterministic 70% training / 30% calibration split with connected groups kept intact.",
  holdout_60_40: "Deterministic 60% training / 40% calibration split with connected groups kept intact.",
  holdout_50_50: "Deterministic 50% training / 50% calibration split with connected groups kept intact.",
  group_k_fold: "The selected grouped fold is calibration; every other fold is training. Rotate the index for grouped cross-validation.",
  leave_one_group_out: "The selected connected entity/article group is calibration; every other group is training.",
};

function updatePartitionControls() {
  const strategy = value("partition-strategy");
  const isKFold = strategy === "group_k_fold";
  const hasFold = isKFold || strategy === "leave_one_group_out";
  $("#fold-count-field").hidden = !isKFold;
  $("#fold-index-field").hidden = !hasFold;
  $("#fold-count").disabled = configurationDisabled || !isKFold;
  $("#fold-index").disabled = configurationDisabled || !hasFold;
  const preset = datasetPresets.find(item => item.id === value("dataset-preset"));
  const prepared = preset?.prepared_partition?.strategy;
  const preparedLabel = prepared
    ? [...$("#partition-strategy").options].find(option => option.value === prepared)?.textContent || prepared
    : null;
  const mismatch = Boolean(prepared && prepared !== strategy);
  const preparedStatus = mismatch
    ? ` Prepared tensors use ${preparedLabel}; run Prepare dataset again before training.`
    : prepared ? ` Prepared tensors use ${preparedLabel}.` : " Run Prepare dataset to create tensors with this partition.";
  $("#partition-description").textContent = `${partitionDescriptions[strategy]}${preparedStatus}`;
  trainButton.disabled = configurationDisabled || mismatch;
}

function syncPreparedPartitions(defaults) {
  if (!defaults?.dataset_presets?.length || !datasetPresets.length) return;
  const partitions = new Map(
    defaults.dataset_presets.map(preset => [preset.id, preset.prepared_partition || null])
  );
  datasetPresets.forEach(preset => {
    if (partitions.has(preset.id)) preset.prepared_partition = partitions.get(preset.id);
  });
  updatePartitionControls();
}

function updatePromptSummary() {
  const summary = promptQuestion
    ? `${promptQuestion.replace("{entity_name}", "Entity")} · ${promptCriteria.length} criteria`
    : "Custom files use the default training prompt";
  $("#prompt-summary").textContent = summary;
}

function applyDatasetPrompt(preset) {
  promptQuestion = preset?.prompt?.question || "";
  promptCriteria = structuredClone(preset?.prompt?.criteria || []);
  updatePromptSummary();
  $("#edit-prompt").disabled = configurationDisabled || !preset?.dataset_id;
  $("#edit-prompt").title = preset?.dataset_id ? "Edit this dataset's training prompt" : "Prompt editing requires a dataset preset";
}

function criterionRow(criterion = {decision: "negative", text: ""}) {
  const row = document.createElement("div"); row.className = "criterion-row";
  const decision = document.createElement("select"); decision.append(new Option("Negative", "negative"), new Option("Positive", "positive")); decision.value = criterion.decision; decision.setAttribute("aria-label", "Criterion decision");
  const text = document.createElement("textarea"); text.value = criterion.text; text.maxLength = 4000; text.required = true; text.setAttribute("aria-label", "Criterion text");
  const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "×"; remove.title = "Remove criterion"; remove.setAttribute("aria-label", "Remove criterion"); remove.onclick = () => row.remove();
  row.append(decision, text, remove); return row;
}

function openPromptEditor() {
  $("#prompt-question").value = promptQuestion;
  $("#criteria-list").replaceChildren(...promptCriteria.map(criterionRow));
  promptDialog.showModal();
}

async function savePrompt() {
  const preset = datasetPresets.find(item => item.id === value("dataset-preset"));
  if (!preset?.dataset_id) return;
  const question = $("#prompt-question").value.trim();
  const criteria = [...$("#criteria-list").children].map(row => ({decision: row.querySelector("select").value, text: row.querySelector("textarea").value.trim()}));
  if (!question) return $("#prompt-question").reportValidity();
  if (criteria.length < 2) return fail("Prompt criteria must contain at least two entries.");
  const blank = [...$("#criteria-list textarea")].find(input => !input.value.trim());
  if (blank) return blank.reportValidity();
  if (!["negative", "positive"].every(decision => criteria.some(criterion => criterion.decision === decision))) return fail("Prompt criteria must include negative and positive decisions.");
  try {
    const data = await request("/v1/fine-tuning/prompt", {dataset_id: preset.dataset_id, question, criteria});
    datasetPresets.filter(item => item.dataset_id === preset.dataset_id).forEach(item => { item.prompt = structuredClone(data.prompt); });
    applyDatasetPrompt(preset);
    fail();
    promptDialog.close();
  } catch (error) { fail(error.message); }
}

function preparePayload() {
  return {
    label_source: "external",
    corpus: value("corpus"),
    labels: value("labels"),
    model_dir: value("prepare-model-dir"),
    output_dir: value("prepared-output"),
    seed: number("prepare-seed"),
    partition_strategy: value("partition-strategy"),
    fold_count: number("fold-count"),
    fold_index: number("fold-index"),
    dataset_id: datasetPresets.find(item => item.id === value("dataset-preset"))?.dataset_id || null,
  };
}

function trainPayload() {
  return {
    strategy: value("training-strategy"),
    partition_strategy: value("partition-strategy"),
    prepared_dir: value("prepared-dir"),
    model_dir: value("train-model-dir"),
    output_dir: value("model-output"),
    gpu_count: number("gpu-count"),
    epochs: number("epochs"),
    batch_size: number("batch-size"),
    eval_batch_size: number("eval-batch-size"),
    gradient_accumulation: number("gradient-accumulation"),
    group_size: number("group-size"),
    rl_weight: number("rl-weight"),
    encoder_lr: number("encoder-lr"),
    head_lr: number("head-lr"),
    weight_decay: number("weight-decay"),
    sigma_start: number("sigma-start"),
    sigma_end: number("sigma-end"),
    encoder_warmup_epochs: number("encoder-warmup-epochs"),
    lr_warmup_ratio: number("lr-warmup-ratio"),
    label_smoothing: number("label-smoothing"),
    seed: number("train-seed"),
    log_every: number("log-every"),
  };
}

function requestHubSettings() {
  const slug = value => value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const model = slug(value("model-preset") || "laya");
  const dataset = slug(value("dataset-preset") || "dataset");
  const strategy = slug(value("training-strategy") || "custom");
  const repository = `layacrime-${model}-${dataset}-${strategy}`
    .slice(0, 96)
    .replace(/[._-]+$/, "");
  $("#hub-repo-id").value = `username/${repository}`;
  $("#hub-private").checked = false;
  hubDialog.showModal();
  $("#hub-repo-id").focus();
  return new Promise(resolve => {
    const finish = settings => {
      hubForm.removeEventListener("submit", submit);
      $("#hub-skip").removeEventListener("click", skip);
      $("#hub-close").removeEventListener("click", skip);
      hubDialog.removeEventListener("cancel", cancel);
      $("#hub-token").value = "";
      hubDialog.close();
      resolve(settings);
    };
    const submit = event => {
      event.preventDefault();
      if (!hubForm.reportValidity()) return;
      finish({
        huggingface_repo_id: value("hub-repo-id"),
        huggingface_token: value("hub-token"),
        huggingface_private: $("#hub-private").checked,
      });
    };
    const skip = () => finish(null);
    const cancel = event => {
      event.preventDefault();
      finish(null);
    };
    hubForm.addEventListener("submit", submit);
    $("#hub-skip").addEventListener("click", skip);
    $("#hub-close").addEventListener("click", skip);
    hubDialog.addEventListener("cancel", cancel);
  });
}

function applyDatasetPreset() {
  const preset = datasetPresets.find(item => item.id === value("dataset-preset"));
  if (!preset) return;
  $("#corpus").value = preset.corpus;
  $("#labels").value = preset.labels;
  $("#prepared-output").value = preset.prepared_dir;
  $("#prepared-dir").value = preset.prepared_dir;
  $("#model-output").value = preset.output_dir;
  if (
    preset.prepared_partition?.strategy
    && [...$("#partition-strategy").options].some(option => option.value === preset.prepared_partition.strategy)
  ) {
    $("#partition-strategy").value = preset.prepared_partition.strategy;
    updatePartitionControls();
  }
  applyDatasetPrompt(preset);
  if (preset.recommended_strategy && trainingStrategies.some(strategy => strategy.id === preset.recommended_strategy)) {
    $("#training-strategy").value = preset.recommended_strategy;
    applyTrainingStrategy(true, true);
  } else {
    applyTrainingStrategy(false, true);
  }
}

function applyModelPreset() {
  const preset = modelPresets.find(item => item.id === value("model-preset"));
  if (!preset) return;
  $("#prepare-model-dir").value = preset.path;
  $("#train-model-dir").value = preset.path;
}

function applyAuthorTrainingPreset() {
  if (!authorTrainingPreset) return;
  $("#training-strategy").value = "balanced";
  const fields = {
    "gpu-count": "gpu_count",
    "epochs": "epochs",
    "batch-size": "batch_size",
    "eval-batch-size": "eval_batch_size",
    "gradient-accumulation": "gradient_accumulation",
    "group-size": "group_size",
    "rl-weight": "rl_weight",
    "encoder-lr": "encoder_lr",
    "head-lr": "head_lr",
    "weight-decay": "weight_decay",
    "sigma-start": "sigma_start",
    "sigma-end": "sigma_end",
    "encoder-warmup-epochs": "encoder_warmup_epochs",
    "lr-warmup-ratio": "lr_warmup_ratio",
    "label-smoothing": "label_smoothing",
    "train-seed": "seed",
    "log-every": "log_every",
  };
  Object.entries(fields).forEach(([id, key]) => { $(`#${id}`).value = authorTrainingPreset[key]; });
  applyTrainingStrategy();
  fail();
}

const strategyParameterFields = {
  "epochs": "epochs",
  "group-size": "group_size",
  "encoder-lr": "encoder_lr",
  "head-lr": "head_lr",
  "rl-weight": "rl_weight",
  "weight-decay": "weight_decay",
  "sigma-start": "sigma_start",
  "sigma-end": "sigma_end",
  "encoder-warmup-epochs": "encoder_warmup_epochs",
  "lr-warmup-ratio": "lr_warmup_ratio",
  "label-smoothing": "label_smoothing",
};

function applyTrainingStrategy(populateParameters = true, updateOutput = true) {
  const strategy = trainingStrategies.find(item => item.id === value("training-strategy"));
  if (strategy && populateParameters) {
    Object.entries(strategyParameterFields).forEach(([id, key]) => {
      $(`#${id}`).value = strategy.parameters[key];
    });
  }
  if (strategy && updateOutput) {
    const dataset = datasetPresets.find(item => item.id === value("dataset-preset"));
    if (dataset) $("#model-output").value = `${dataset.output_dir}-${strategy.id}`;
  }
  $("#strategy-description").textContent = strategy?.description || "Use the optimization parameters below without automatic overrides.";
  Object.keys(strategyParameterFields).forEach(id => {
    $(`#${id}`).disabled = configurationDisabled;
  });
}

function applyDefaults(defaults) {
  if (defaultsApplied || !defaults) return;
  $("#corpus").value = defaults.corpus;
  $("#labels").value = defaults.labels;
  $("#prepared-output").value = defaults.prepared_dir;
  $("#prepared-dir").value = defaults.prepared_dir;
  $("#model-output").value = defaults.output_dir;
  datasetPresets = defaults.dataset_presets || [];
  modelPresets = defaults.model_presets || [];
  trainingStrategies = defaults.training_strategies || [];
  authorTrainingPreset = defaults.author_training_preset || null;
  $("#reset-author-parameters").disabled = !authorTrainingPreset;
  const modelSelect = $("#model-preset");
  modelSelect.replaceChildren(...modelPresets.map(preset => new Option(preset.label, preset.id)));
  modelSelect.disabled = !modelPresets.length;
  if (modelPresets.length) {
    modelSelect.value = modelPresets.some(preset => preset.id === "auto") ? "auto" : modelPresets[0].id;
    applyModelPreset();
  }
  const presetSelect = $("#dataset-preset");
  presetSelect.replaceChildren(...datasetPresets.map(preset => new Option(preset.label, preset.id)));
  presetSelect.add(new Option("Custom label file", "custom"));
  if (datasetPresets.length) {
    presetSelect.value = datasetPresets[0].id;
  }
  const strategySelect = $("#training-strategy");
  strategySelect.replaceChildren(...trainingStrategies.map(strategy => new Option(strategy.label, strategy.id)));
  strategySelect.add(new Option("Custom parameters", "custom"));
  strategySelect.value = defaults.default_training_strategy || "balanced";
  applyTrainingStrategy();
  if (datasetPresets.length) applyDatasetPreset();
  defaultsApplied = true;
}

function setInputsDisabled(disabled) {
  configurationDisabled = disabled;
  document.querySelectorAll(".configuration input, .configuration select").forEach(control => {
    control.disabled = disabled;
  });
  prepareButton.disabled = disabled;
  trainButton.disabled = disabled;
  stopButton.disabled = !disabled;
  $("#reset-author-parameters").disabled = disabled || !authorTrainingPreset;
  $("#model-preset").disabled = disabled || !modelPresets.length;
  updatePartitionControls();
  $("#edit-prompt").disabled = disabled || !datasetPresets.find(item => item.id === value("dataset-preset"))?.dataset_id;
  applyTrainingStrategy(false, false);
}

async function loadReports() {
  const data = await request("/v1/fine-tuning/reports");
  const selected = reportSelect.value;
  reportIds = new Set(data.reports.map(report => report.run_id));
  reportSelect.replaceChildren();
  if (!data.reports.length) reportSelect.add(new Option("No completed trainings found", ""));
  data.reports.forEach(report => {
    const date = new Date(report.updated_at).toLocaleString();
    reportSelect.add(new Option(`${report.label} · ${report.epochs} epochs · ${date}`, report.run_id));
  });
  if ([...reportSelect.options].some(option => option.value === selected)) reportSelect.value = selected;
  loadTrainingButton.disabled = !reportSelect.value;
}

function appendLogs(logs) {
  if (!logs.length) return;
  if (logView.textContent.startsWith("Waiting for")) logView.textContent = "";
  const pinned = logView.scrollHeight - logView.scrollTop - logView.clientHeight < 35;
  logs.forEach(row => {
    logView.textContent += `${row.message}\n`;
    latestLogIndex = Math.max(latestLogIndex, row.index);
  });
  if (pinned) logView.scrollTop = logView.scrollHeight;
}

function drawLossChart(history = [], epochs = 0) {
  const canvas = $("#loss-chart");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = 220;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const left = 42, right = 14, top = 14, bottom = 28;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const losses = history.flatMap(row => [row.train_loss, row.validation_loss]).filter(Number.isFinite);
  const maxLoss = Math.max(1, ...losses) * 1.08;
  const totalEpochs = Math.max(epochs, history.at(-1)?.epoch || 1, 2);
  const x = epoch => left + (epoch - 1) / (totalEpochs - 1) * plotWidth;
  const y = loss => top + (1 - loss / maxLoss) * plotHeight;
  context.font = "10px DM Sans";
  context.fillStyle = "#68716d";
  context.strokeStyle = "#d8ded8";
  context.lineWidth = 1;
  [0, .5, 1].forEach(portion => {
    const value = maxLoss * portion;
    const lineY = y(value);
    context.beginPath(); context.moveTo(left, lineY); context.lineTo(width - right, lineY); context.stroke();
    context.fillText(value.toFixed(2), 4, lineY + 3);
  });
  if (!history.length) {
    context.fillText("Curves appear after the first epoch", left + 10, top + plotHeight / 2);
  } else {
    const plot = (key, color) => {
      context.beginPath(); context.strokeStyle = color; context.lineWidth = 2;
      history.forEach((row, index) => { const pointX = x(row.epoch), pointY = y(row[key]); if (index) context.lineTo(pointX, pointY); else context.moveTo(pointX, pointY); });
      context.stroke(); context.fillStyle = color;
      history.forEach(row => { context.beginPath(); context.arc(x(row.epoch), y(row[key]), 3, 0, Math.PI * 2); context.fill(); });
    };
    plot("train_loss", "#167454");
    plot("validation_loss", "#be4037");
  }
  context.fillStyle = "#68716d"; context.fillText("Epoch 1", left, height - 7);
  context.textAlign = "right"; context.fillText(`Epoch ${totalEpochs}`, width - right, height - 7); context.textAlign = "left";
  const latest = history.at(-1);
  const best = history
    .filter(row => Number.isFinite(row.validation_loss))
    .reduce((selected, row) => !selected || row.validation_loss < selected.validation_loss ? row : selected, null);
  $("#loss-epoch-latest").textContent = latest?.epoch ?? "-";
  $("#train-loss-latest").textContent = latest ? latest.train_loss.toFixed(4) : "-";
  $("#validation-loss-latest").textContent = latest ? latest.validation_loss.toFixed(4) : "-";
  $("#loss-best-epoch").textContent = best?.epoch ?? "-";
  $("#train-loss-best").textContent = decimal(best?.train_loss);
  $("#validation-loss-best").textContent = decimal(best?.validation_loss);
}

function drawAccuracyChart(history = [], epochs = 0) {
  const canvas = $("#accuracy-chart");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = 220;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const left = 42, right = 14, top = 14, bottom = 28;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const totalEpochs = Math.max(epochs, history.at(-1)?.epoch || 1, 2);
  const x = epoch => left + (epoch - 1) / (totalEpochs - 1) * plotWidth;
  const y = accuracy => top + (1 - accuracy) * plotHeight;
  context.font = "10px DM Sans";
  context.fillStyle = "#68716d";
  context.strokeStyle = "#d8ded8";
  context.lineWidth = 1;
  [0, .5, 1].forEach(value => {
    const lineY = y(value);
    context.beginPath(); context.moveTo(left, lineY); context.lineTo(width - right, lineY); context.stroke();
    context.fillText(`${Math.round(value * 100)}%`, 4, lineY + 3);
  });
  const accuracyHistory = history.filter(row => Number.isFinite(row.validation_accuracy));
  if (!accuracyHistory.length) {
    context.fillText("Curve appears after the first epoch", left + 10, top + plotHeight / 2);
  } else {
    const plot = (rows, key, color) => {
      if (!rows.length) return;
      context.beginPath(); context.strokeStyle = color; context.lineWidth = 2;
      rows.forEach((row, index) => {
        const pointX = x(row.epoch), pointY = y(row[key]);
        if (index) context.lineTo(pointX, pointY); else context.moveTo(pointX, pointY);
      });
      context.stroke(); context.fillStyle = color;
      rows.forEach(row => { context.beginPath(); context.arc(x(row.epoch), y(row[key]), 3, 0, Math.PI * 2); context.fill(); });
    };
    plot(accuracyHistory, "validation_accuracy", "#be4037");
  }
  context.fillStyle = "#68716d"; context.fillText("Epoch 1", left, height - 7);
  context.textAlign = "right"; context.fillText(`Epoch ${totalEpochs}`, width - right, height - 7); context.textAlign = "left";
  const latest = history.at(-1);
  const best = accuracyHistory.reduce((selected, row) => !selected || row.validation_accuracy > selected.validation_accuracy ? row : selected, null);
  $("#accuracy-epoch-latest").textContent = latest?.epoch ?? "-";
  $("#train-accuracy-latest").textContent = percent(latest?.train_accuracy);
  $("#validation-accuracy-latest").textContent = percent(latest?.validation_accuracy);
  $("#accuracy-best-epoch").textContent = best?.epoch ?? "-";
  $("#train-accuracy-best").textContent = percent(best?.train_accuracy);
  $("#validation-accuracy-best").textContent = percent(best?.validation_accuracy);
}

function renderReport(data) {
  const report = data.report;
  $("#training-results").hidden = !report;
  if (!report) return;
  const history = data.loss_history || [];
  const selected = history.find(row => row.epoch === report.best_epoch) || history.at(-1);
  const partition = data.report?.partition;
  $("#report-name").textContent = data.run?.label || "Completed training";
  $("#partition-result").textContent = partition?.label || partition?.strategy || "Legacy split";
  $("#report-best-epoch").textContent = report.best_epoch ?? "-";
  $("#report-validation-accuracy").textContent = percent(selected?.validation_accuracy);
  $("#report-validation-loss").textContent = decimal(report.best_validation_loss);
  $("#report-temperature").textContent = decimal(report.temperature_choice);
}

function render(data, scheduleRefresh = true) {
  applyDefaults(data.defaults);
  syncPreparedPartitions(data.defaults);
  if (data.started_at !== currentJobStartedAt) {
    currentJobStartedAt = data.started_at;
    latestLogIndex = 0;
    logView.textContent = data.started_at ? "" : "Waiting for a preparation or training job.";
  }
  appendLogs(data.logs || []);
  const active = data.status === "running" || data.status === "stopping";
  const nextTrainingOwnsGpu = active && data.phase === "training";
  if (nextTrainingOwnsGpu !== trainingOwnsGpu) {
    trainingOwnsGpu = nextTrainingOwnsGpu;
    refreshInferenceModel();
  }
  const enabled = data.defaults?.enabled !== false;
  setInputsDisabled(active || !enabled);
  reportSelect.disabled = active;
  loadTrainingButton.disabled = active || !reportSelect.value;
  if (!enabled) fail("Fine-tuning is disabled on this server.");
  $("#job-state").className = `job-state ${data.status}`;
  $("#job-status-label").textContent = data.historical ? "Loaded report" : data.status.charAt(0).toUpperCase() + data.status.slice(1);
  $("#phase-badge").className = `status-badge ${data.status}`;
  $("#phase-badge").textContent = data.historical ? "Archived" : data.status;
  $("#phase").textContent = data.phase ? data.phase.charAt(0).toUpperCase() + data.phase.slice(1) : "Not started";
  $("#pid").textContent = data.pid || "-";
  $("#epoch-progress").textContent = data.progress?.epoch ? `${data.progress.epoch} / ${data.progress.epochs}` : "-";
  $("#batch-progress").textContent = data.progress?.batch ?? "-";
  $("#loss-progress").textContent = data.progress?.loss?.toFixed(4) ?? "-";
  $("#sigma-progress").textContent = data.progress?.sigma?.toFixed(3) ?? "-";
  $("#started-at").textContent = data.started_at ? new Date(data.started_at).toLocaleString() : "-";
  $("#exit-code").textContent = data.exit_code ?? "-";
  $("#time-label").textContent = data.historical ? "Saved" : "Started";
  $("#command").textContent = data.historical ? `Loaded report: ${data.run.label}` : data.command?.length ? data.command.map(part => part.includes(" ") ? `"${part}"` : part).join(" ") : "No command launched.";
  currentLossHistory = data.loss_history || [];
  drawLossChart(currentLossHistory, data.progress?.epochs || number("epochs"));
  drawAccuracyChart(currentLossHistory, data.progress?.epochs || number("epochs"));
  renderReport(data);
  fail(actionError || data.error || "");
  clearTimeout(timer);
  if (data.status === "complete" && lastStatus !== "complete" && !data.historical) loadReports().catch(error => fail(error.message));
  lastStatus = data.status;
  if (scheduleRefresh) timer = setTimeout(refresh, active ? 750 : 2500);
}

async function refresh() {
  try {
    const data = await request(`/v1/fine-tuning?logs_after=${latestLogIndex}`);
    if (data.started_at !== currentJobStartedAt) {
      latestLogIndex = 0;
      const full = await request("/v1/fine-tuning?logs_after=0");
      render(full);
    } else {
      render(data);
    }
  } catch (error) {
    fail(error.message);
    clearTimeout(timer);
    timer = setTimeout(refresh, 2000);
  }
}

prepareButton.onclick = async () => {
  try {
    actionError = "";
    liveTrainingButton.hidden = true;
    fail();
    render(await request("/v1/fine-tuning/prepare", preparePayload()));
  } catch (error) { actionError = error.message; fail(actionError); }
};
trainButton.onclick = async () => {
  try {
    actionError = "";
    liveTrainingButton.hidden = true;
    fail();
    const hubSettings = await requestHubSettings();
    render(await request("/v1/fine-tuning/train", {
      ...trainPayload(),
      ...(hubSettings || {}),
    }));
  } catch (error) { actionError = error.message; fail(actionError); }
};
stopButton.onclick = async () => {
  try {
    actionError = "";
    fail();
    render(await request("/v1/fine-tuning/stop", {}));
  } catch (error) { actionError = error.message; fail(actionError); }
};
reportSelect.onchange = () => { loadTrainingButton.disabled = !reportSelect.value; };
$("#model-preset").onchange = applyModelPreset;
$("#training-strategy").onchange = () => applyTrainingStrategy();
$("#reset-author-parameters").onclick = applyAuthorTrainingPreset;
loadTrainingButton.onclick = async () => {
  try {
    actionError = "";
    clearTimeout(timer);
    const loaded = await request("/v1/fine-tuning/reports/load", {run_id: reportSelect.value});
    liveTrainingButton.hidden = false;
    render(loaded, false);
  } catch (error) { actionError = error.message; fail(actionError); }
};
liveTrainingButton.onclick = () => {
  liveTrainingButton.hidden = true;
  currentJobStartedAt = "__refresh_live__";
  refresh();
};
$("#copy-log").onclick = async event => {
  const button = event.currentTarget;
  let copyArea;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(logView.textContent);
    } else {
      copyArea = document.createElement("textarea");
      copyArea.value = logView.textContent;
      copyArea.readOnly = true;
      copyArea.style.position = "fixed";
      copyArea.style.opacity = "0";
      document.body.append(copyArea);
      copyArea.select();
      if (!document.execCommand("copy")) throw new Error("Copy command was rejected");
    }
    button.textContent = "Copied";
    window.setTimeout(() => { button.textContent = "Copy log"; }, 1400);
  } catch {
    fail("Could not copy the process output. Select the log text and copy it manually.");
  } finally {
    copyArea?.remove();
  }
};
$("#clear-log").onclick = () => { logView.textContent = ""; };
$("#dataset-preset").onchange = applyDatasetPreset;
new ResourcePicker($("#model-preset"), {
  isDeletable: id => modelPresets.some(preset => preset.id === id && preset.deletable),
  onDelete: async id => {
    await request("/v1/benchmark/models/delete", {model_id:id});
    modelPresets = modelPresets.filter(preset => preset.id !== id);
    const select = $("#model-preset");
    select.replaceChildren(...modelPresets.map(preset => new Option(preset.label, preset.id)));
    select.value = modelPresets.some(preset => preset.id === "auto") ? "auto" : modelPresets[0]?.id || "";
    applyModelPreset();
  },
  onError: error => fail(error.message),
});
new ResourcePicker(reportSelect, {
  isDeletable: id => reportIds.has(id),
  onDelete: async id => {
    const data = await request("/v1/fine-tuning/reports/delete", {run_id:id});
    modelPresets = data.model_presets;
    const modelSelect = $("#model-preset");
    const selectedModel = modelSelect.value;
    modelSelect.replaceChildren(...modelPresets.map(preset => new Option(preset.label, preset.id)));
    modelSelect.value = modelPresets.some(preset => preset.id === selectedModel) ? selectedModel : "auto";
    applyModelPreset();
    await loadReports();
  },
  onError: error => fail(error.message),
});
new ResourcePicker($("#dataset-preset"), {
  isDeletable: id => datasetPresets.some(preset => preset.id === id && preset.deletable),
  onDelete: async id => {
    const removed = datasetPresets.find(preset => preset.id === id);
    if (!removed) return;
    await request("/v1/benchmark/label-sets/delete", {
      dataset_id:removed.dataset_id || null,
      label_set_id:removed.label_set_id,
    });
    datasetPresets = datasetPresets.filter(preset => !(
      preset.label_set_id === removed.label_set_id
      && preset.dataset_id === removed.dataset_id
    ));
    const select = $("#dataset-preset");
    select.replaceChildren(...datasetPresets.map(preset => new Option(preset.label, preset.id)));
    select.add(new Option("Custom label file", "custom"));
    select.value = datasetPresets[0]?.id || "custom";
    applyDatasetPreset();
  },
  onError: error => fail(error.message),
});
$("#partition-strategy").onchange = updatePartitionControls;
$("#edit-prompt").onclick = openPromptEditor;
$("#add-criterion").onclick = () => $("#criteria-list").append(criterionRow());
$("#prompt-close").onclick = () => promptDialog.close();
$("#cancel-prompt").onclick = () => promptDialog.close();
$("#save-prompt").onclick = savePrompt;
window.addEventListener("resize", () => {
  drawLossChart(currentLossHistory, number("epochs"));
  drawAccuracyChart(currentLossHistory, number("epochs"));
});
loadReports().catch(error => fail(error.message));
refreshInferenceModel();
refresh();
