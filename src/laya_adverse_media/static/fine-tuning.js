const $ = (selector) => document.querySelector(selector);
const prepareButton = $("#prepare-button");
const trainButton = $("#train-button");
const stopButton = $("#stop-button");
const logView = $("#process-log");
const errorBox = $("#error-message");
let timer;
let defaultsApplied = false;
let latestLogIndex = 0;
let currentJobStartedAt = null;

function fail(message = "") {
  errorBox.textContent = message;
  errorBox.hidden = !message;
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
    throw new Error(typeof detail === "string" ? detail : detail?.message || "Fine-tuning request failed");
  }
  return data;
}

function value(id) { return $(`#${id}`).value.trim(); }
function number(id) { return Number($(`#${id}`).value); }

function preparePayload() {
  return {
    label_source: value("label-source"),
    data_dir: value("data-dir"),
    corpus: value("corpus"),
    annotations: value("annotations"),
    model_dir: value("prepare-model-dir"),
    output_dir: value("prepared-output"),
    seed: number("prepare-seed"),
  };
}

function trainPayload() {
  return {
    prepared_dir: value("prepared-dir"),
    model_dir: value("train-model-dir"),
    output_dir: value("model-output"),
    resume_from: value("resume-from") || null,
    gpu_count: number("gpu-count"),
    epochs: number("epochs"),
    batch_size: number("batch-size"),
    eval_batch_size: number("eval-batch-size"),
    gradient_accumulation: number("gradient-accumulation"),
    group_size: number("group-size"),
    encoder_lr: number("encoder-lr"),
    head_lr: number("head-lr"),
    weight_decay: number("weight-decay"),
    sigma_start: number("sigma-start"),
    sigma_end: number("sigma-end"),
    seed: number("train-seed"),
    log_every: number("log-every"),
  };
}

function applyDefaults(defaults) {
  if (defaultsApplied || !defaults) return;
  $("#corpus").value = defaults.corpus;
  $("#annotations").value = defaults.annotations;
  $("#data-dir").value = defaults.data_dir;
  $("#prepare-model-dir").value = defaults.model_dir;
  $("#train-model-dir").value = defaults.model_dir;
  $("#prepared-output").value = defaults.prepared_dir;
  $("#prepared-dir").value = defaults.prepared_dir;
  $("#model-output").value = defaults.output_dir;
  defaultsApplied = true;
}

function setInputsDisabled(disabled) {
  document.querySelectorAll(".configuration input, .configuration select").forEach(control => {
    control.disabled = disabled;
  });
  prepareButton.disabled = disabled;
  trainButton.disabled = disabled;
  stopButton.disabled = !disabled;
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

function render(data) {
  applyDefaults(data.defaults);
  if (data.started_at !== currentJobStartedAt) {
    currentJobStartedAt = data.started_at;
    latestLogIndex = 0;
    logView.textContent = data.started_at ? "" : "Waiting for a preparation or training job.";
  }
  appendLogs(data.logs || []);
  const active = data.status === "running" || data.status === "stopping";
  setInputsDisabled(active);
  $("#job-state").className = `job-state ${data.status}`;
  $("#status-label").textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
  $("#phase-badge").className = `status-badge ${data.status}`;
  $("#phase-badge").textContent = data.status;
  $("#phase").textContent = data.phase ? data.phase.charAt(0).toUpperCase() + data.phase.slice(1) : "Not started";
  $("#pid").textContent = data.pid || "-";
  $("#epoch-progress").textContent = data.progress?.epoch ? `${data.progress.epoch} / ${data.progress.epochs}` : "-";
  $("#batch-progress").textContent = data.progress?.batch ?? "-";
  $("#loss-progress").textContent = data.progress?.loss?.toFixed(4) ?? "-";
  $("#sigma-progress").textContent = data.progress?.sigma?.toFixed(3) ?? "-";
  $("#started-at").textContent = data.started_at ? new Date(data.started_at).toLocaleString() : "-";
  $("#exit-code").textContent = data.exit_code ?? "-";
  $("#command").textContent = data.command?.length ? data.command.map(part => part.includes(" ") ? `"${part}"` : part).join(" ") : "No command launched.";
  fail(data.error || "");
  clearTimeout(timer);
  timer = setTimeout(refresh, active ? 750 : 2500);
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

function toggleLabelSource() {
  const azure = value("label-source") === "azure";
  document.querySelectorAll(".azure-field").forEach(field => field.hidden = !azure);
  document.querySelectorAll(".human-field").forEach(field => field.hidden = azure);
}

prepareButton.onclick = async () => {
  try {
    fail();
    render(await request("/v1/fine-tuning/prepare", preparePayload()));
  } catch (error) { fail(error.message); }
};
trainButton.onclick = async () => {
  try {
    fail();
    render(await request("/v1/fine-tuning/train", trainPayload()));
  } catch (error) { fail(error.message); }
};
stopButton.onclick = async () => {
  try {
    fail();
    render(await request("/v1/fine-tuning/stop", {}));
  } catch (error) { fail(error.message); }
};
$("#clear-log").onclick = () => { logView.textContent = ""; };
$("#label-source").onchange = toggleLabelSource;
toggleLabelSource();
refresh();
