const $ = selector => document.querySelector(selector);
const form = $("#test-form");
const modelSelect = $("#model-id");
const runButton = $("#run-button");
const errorBox = $("#error-message");
const articleInput = $("#article-text");
const promptDialog = $("#prompt-dialog");
const customModelAction = "__add-custom-model";
let modelReady = false;
let modelChanging = false;
let availableModels = [];
let activeModelId = "auto";
let installPollTimer = null;
let installLogIndex = 0;
const pct = value => `${((value || 0) * 100).toFixed(1)}%`;

function fail(message = "") {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || "Request failed");
  }
  return data;
}

function setService(status) {
  const ready = status === "ok";
  $("#model-health").className = `model-health ${ready ? "ready" : status === "warming" ? "" : "degraded"}`;
  $("#model-status-label").textContent = ready ? "Model ready" : status === "warming" ? "Loading model" : status === "installing" ? "Installing requirements" : "Model unavailable";
}

function selectedModel() {
  return availableModels.find(item => item.id === modelSelect.value);
}

function setActiveModel() {
  $("#active-model").textContent = selectedModel()?.label || "No model available";
}

function applySelectedModelPrompt() {
  $("#prompt-summary").textContent = PromptViewer.summary(selectedModel());
}

function openPrompt() {
  const model = selectedModel();
  $("#prompt-title").textContent = model ? `${model.label} prompt` : "Inference prompt";
  PromptViewer.render($("#prompt-view"), model);
  promptDialog.showModal();
}

function updateModelInstallAction() {
  const model = selectedModel();
  $("#install-model").hidden = !model?.installable || (model.installed && model.runtime_available);
  if (model) $("#install-model").textContent = model.installation_state === "missing" ? "Install model" : "Repair installation";
}

function optionLabel(model) {
  if (model.installable && !model.installed) {
    return `${model.label} · ${model.installation_state === "missing" ? "install required" : "repair required"}`;
  }
  if (!model.installable && model.runtime_available === false) return `${model.label} · setup required`;
  return model.label;
}

function renderModelOptions(data, selected) {
  const endpointAction = data.models.find(model => model.provider === "azure" && model.action_only);
  const groupSpecs = [
    ["Decision Models", model => model.category === "decision", null],
    ["Cloud LLMs", model => model.provider === "azure" && !model.action_only, endpointAction ? new Option(endpointAction.action_label, endpointAction.id) : null],
    ["Self-hosted LLMs", model => model.provider === "llama.cpp", new Option("Add custom model…", customModelAction)],
  ];
  const groups = groupSpecs.map(([label, includes, action]) => {
    const group = document.createElement("optgroup");
    group.label = label;
    const models = data.models.filter(includes);
    models.forEach(model => group.append(new Option(optionLabel(model), model.id)));
    if (!models.length && !action) {
      const unavailable = new Option(`${label} unavailable`, `__${label.toLowerCase().replaceAll(" ", "-")}-unavailable`);
      unavailable.disabled = true;
      group.append(unavailable);
    }
    if (action) group.append(action);
    return group;
  });
  const content = [...groups];
  if (!data.models.some(model => model.id === selected)) {
    const placeholder = new Option("Select a model", "", true, true);
    placeholder.disabled = true;
    content.unshift(placeholder);
  }
  modelSelect.replaceChildren(...content);
  if ([...modelSelect.options].some(option => option.value === selected)) modelSelect.value = selected;
}

async function loadModels(selectedId = null) {
  const data = await request("/v1/benchmark/models");
  availableModels = data.models;
  activeModelId = data.active_model || "auto";
  renderModelOptions(data, selectedId || activeModelId);
  applySelectedModelPrompt();
  updateModelInstallAction();
  setActiveModel();
  modelReady = availableModels.some(model => model.id === activeModelId && model.runtime_available !== false);
  modelSelect.disabled = !availableModels.length;
  runButton.disabled = !modelReady;
  setService(modelReady ? "ok" : "unavailable");
  if (data.laya_error && !modelReady) fail(data.laya_error);
}

function requestCredentials(model) {
  const dialog = $("#credentials-dialog"), credentialsForm = $("#credentials-form");
  $("#credentials-title").textContent = model.action_only ? "Add endpoint" : `Connect ${model.label}`;
  $("#azure-deployment").value = model.action_only ? "" : (model.azure_deployment || model.label.toLowerCase());
  $("#azure-api-key").value = "";
  dialog.showModal();
  $("#azure-endpoint").focus();
  return new Promise(resolve => {
    const finish = value => {
      credentialsForm.removeEventListener("submit", submit);
      $("#credentials-cancel").removeEventListener("click", cancel);
      $("#credentials-close").removeEventListener("click", cancel);
      dialog.removeEventListener("cancel", cancelEvent);
      $("#azure-api-key").value = "";
      dialog.close();
      resolve(value);
    };
    const submit = event => {
      event.preventDefault();
      if (!credentialsForm.reportValidity()) return;
      finish({endpoint: $("#azure-endpoint").value.trim(), deployment: $("#azure-deployment").value.trim(), api_key: $("#azure-api-key").value});
    };
    const cancel = () => finish(null);
    const cancelEvent = event => { event.preventDefault(); finish(null); };
    credentialsForm.addEventListener("submit", submit);
    $("#credentials-cancel").addEventListener("click", cancel);
    $("#credentials-close").addEventListener("click", cancel);
    dialog.addEventListener("cancel", cancelEvent);
  });
}

function requestInstallation(model, custom = false) {
  const dialog = $("#install-dialog"), installForm = $("#install-form");
  clearTimeout(installPollTimer);
  installLogIndex = 0;
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
  $("#install-config").hidden = false;
  $("#install-progress").hidden = true;
  $("#install-submit").hidden = false;
  $("#install-cancel").hidden = false;
  $("#install-stop").hidden = true;
  $("#install-done").hidden = true;
  $("#install-close").disabled = false;
  $("#install-error").hidden = true;
  $("#install-log").textContent = "Waiting for installation output...";
  dialog.showModal();
  return new Promise(resolve => {
    const finish = value => {
      installForm.removeEventListener("submit", submit);
      $("#install-cancel").removeEventListener("click", cancel);
      $("#install-close").removeEventListener("click", cancel);
      dialog.removeEventListener("cancel", cancelEvent);
      $("#install-token").value = "";
      if (!value) dialog.close();
      resolve(value);
    };
    const submit = event => {
      event.preventDefault();
      if (!installForm.reportValidity()) return;
      finish({
        name: $("#install-name").value.trim(),
        repo_id: $("#install-repo-id").value.trim(),
        huggingface_path: $("#install-hf-path").value.trim(),
        huggingface_token: $("#install-token").value || null,
      });
    };
    const cancel = () => finish(null);
    const cancelEvent = event => { event.preventDefault(); finish(null); };
    installForm.addEventListener("submit", submit);
    $("#install-cancel").addEventListener("click", cancel);
    $("#install-close").addEventListener("click", cancel);
    dialog.addEventListener("cancel", cancelEvent);
  });
}

const formatBytes = value => {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const unit = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** unit).toFixed(unit ? 1 : 0)} ${units[unit]}`;
};

function renderInstallation(data) {
  $("#install-phase").textContent = (data.phase || data.status || "preparing").replaceAll("_", " ");
  const progress = Number.isFinite(data.progress) ? data.progress : null;
  $("#install-percent").textContent = progress === null ? "Working..." : `${(progress * 100).toFixed(1)}%`;
  $("#install-progress-bar").className = progress === null && data.status === "running" ? "indeterminate" : "";
  $("#install-progress-bar").style.width = progress === null ? "" : `${progress * 100}%`;
  $("#install-bytes").textContent = data.total_bytes
    ? `${formatBytes(data.downloaded_bytes)} of ${formatBytes(data.total_bytes)}`
    : data.phase === "building runtime" ? "Compiling llama.cpp runtime" : "Preparing installation";
  if (data.logs?.length) {
    if ($("#install-log").textContent.startsWith("Waiting for")) $("#install-log").textContent = "";
    data.logs.forEach(row => {
      $("#install-log").textContent += `${row.message}\n`;
      installLogIndex = Math.max(installLogIndex, row.index);
    });
    $("#install-log").scrollTop = $("#install-log").scrollHeight;
  }
  if (data.status === "failed") {
    $("#install-error").textContent = data.error || "Installation failed";
    $("#install-error").hidden = false;
  }
}

function waitForInstallation() {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const data = await request(`/v1/benchmark/models/install?logs_after=${installLogIndex}`);
        renderInstallation(data);
        if (data.status === "complete") return resolve(data);
        if (["failed", "stopped"].includes(data.status)) return reject(new Error(data.error || `Installation ${data.status}`));
        installPollTimer = setTimeout(poll, 500);
      } catch (error) {
        reject(error);
      }
    };
    poll();
  });
}

function showInstallationFailure(error) {
  $("#install-stop").hidden = true;
  $("#install-done").hidden = false;
  $("#install-close").disabled = false;
  $("#install-error").textContent = error.message;
  $("#install-error").hidden = false;
}

async function runInstallation(model, settings) {
  $("#install-config").hidden = true;
  $("#install-progress").hidden = false;
  $("#install-submit").hidden = true;
  $("#install-cancel").hidden = true;
  $("#install-stop").hidden = false;
  $("#install-close").disabled = true;
  setService("installing");
  try {
    renderInstallation(await request("/v1/benchmark/models/install", {model_id: model.id, ...settings}));
    await waitForInstallation();
    await loadModels(model.id);
    $("#install-dialog").close();
    return true;
  } catch (error) {
    showInstallationFailure(error);
    throw error;
  }
}

async function installSelectedModel() {
  const model = selectedModel();
  if (!model?.installable) return false;
  const settings = await requestInstallation(model);
  return settings ? runInstallation(model, settings) : false;
}

async function addCustomModel() {
  const settings = await requestInstallation({label: "Custom model"}, true);
  if (!settings) return false;
  const created = await request("/v1/benchmark/models/custom", {
    name: settings.name,
    repo_id: settings.repo_id,
    huggingface_path: settings.huggingface_path,
  });
  await loadModels(created.model.id);
  return runInstallation(created.model, settings);
}

function restoreActiveSelection() {
  if ([...modelSelect.options].some(option => option.value === activeModelId)) modelSelect.value = activeModelId;
  applySelectedModelPrompt();
  updateModelInstallAction();
  setActiveModel();
}

async function activateSelectedModel() {
  fail();
  if (modelSelect.value === customModelAction) {
    try {
      if (await addCustomModel()) return activateSelectedModel();
      await loadModels(activeModelId);
    } catch (error) {
      await loadModels(activeModelId).catch(() => {});
      fail(error.message);
    }
    return;
  }
  const model = selectedModel();
  if (!model) return;
  setActiveModel();
  applySelectedModelPrompt();
  updateModelInstallAction();
  modelChanging = true;
  modelReady = false;
  modelSelect.disabled = true;
  runButton.disabled = true;
  setService("warming");
  try {
    if (model.installable && (!model.installed || !model.runtime_available) && !(await installSelectedModel())) {
      restoreActiveSelection();
      modelReady = true;
      return;
    }
    const credentials = model.credential_required ? await requestCredentials(model) : {};
    if (model.credential_required && !credentials) {
      restoreActiveSelection();
      modelReady = true;
      return;
    }
    await request("/v1/benchmark/models/activate", {model_id: model.id, ...credentials});
    activeModelId = model.id;
    modelReady = true;
  } catch (error) {
    restoreActiveSelection();
    modelReady = false;
    fail(error.message);
  } finally {
    modelChanging = false;
    modelSelect.disabled = false;
    runButton.disabled = !modelReady;
    setService(modelReady ? "ok" : "unavailable");
  }
}

async function syncActiveModel() {
  if (modelChanging || document.querySelector("dialog[open]")) return;
  try {
    const data = await request("/v1/benchmark/models");
    availableModels = data.models;
    if ((data.active_model || "auto") === activeModelId) return;
    activeModelId = data.active_model || "auto";
    restoreActiveSelection();
    modelReady = availableModels.some(model => model.id === activeModelId);
    runButton.disabled = !modelReady;
    setService(modelReady ? "ok" : "unavailable");
  } catch {
    setService("unavailable");
  }
}

async function start() {
  const health = await request("/health").catch(() => ({status: "degraded"}));
  if (health.status === "warming") {
    setService("warming");
    window.setTimeout(() => start().catch(error => fail(error.message)), 1200);
    return;
  }
  await loadModels();
}

function render(result, elapsed) {
  const negative = result.probabilities.negative;
  const positive = result.probabilities.positive;
  const languageModel = ["azure", "llama.cpp"].includes(result.routing?.provider);
  $("#decision-title").textContent = result.entity_name;
  $("#decision").textContent = result.decision;
  $("#confidence").textContent = languageModel
    ? "Discrete 0% / 100% decision"
    : `${pct(Math.max(negative, positive))} class probability · ${pct(result.confidence)} certainty`;
  $("#negative-value").textContent = pct(negative);
  $("#positive-value").textContent = pct(positive);
  $("#negative-bar").style.width = pct(negative);
  $("#positive-bar").style.width = pct(positive);
  $("#result-model").textContent = selectedModel()?.label || modelSelect.selectedOptions[0].textContent;
  $("#review-detail").textContent = languageModel
    ? "Language models do not report certainty"
    : result.needs_review ? "Manual review recommended" : "Confidence threshold met";
  $("#result-time").textContent = `${elapsed.toFixed(2)}s`;
  $("#review-state").textContent = languageModel ? "LLM decision" : result.needs_review ? "Review" : "Confident";
  $("#review-state").className = `badge ${languageModel ? "" : result.needs_review ? "review" : "clear"}`;
  $("#elapsed-time").textContent = `${elapsed.toFixed(2)}s`;
}

articleInput.addEventListener("input", () => {
  $("#character-count").textContent = `${articleInput.value.length.toLocaleString()} characters`;
});
modelSelect.addEventListener("change", activateSelectedModel);
form.addEventListener("submit", async event => {
  event.preventDefault();
  fail();
  runButton.disabled = true;
  runButton.textContent = "Running...";
  const started = performance.now();
  try {
    const data = await request("/v1/adverse-media", {
      entity_name: $("#entity-name").value.trim(),
      article: articleInput.value.trim(),
      model_id: modelSelect.value === "auto" ? null : modelSelect.value,
    });
    render(data, (performance.now() - started) / 1000);
  } catch (error) {
    fail(error.message);
  } finally {
    runButton.disabled = modelSelect.disabled || !modelReady;
    runButton.textContent = "Run test";
  }
});

$("#see-prompt").onclick = openPrompt;
$("#prompt-close").onclick = () => promptDialog.close();
$("#prompt-done").onclick = () => promptDialog.close();
$("#install-model").onclick = () => installSelectedModel()
  .then(installed => { if (installed) return activateSelectedModel(); updateModelInstallAction(); })
  .catch(error => fail(error.message));
$("#install-stop").onclick = () => request("/v1/benchmark/models/install/stop", {}).then(renderInstallation).catch(error => fail(error.message));
$("#install-done").onclick = () => $("#install-dialog").close();

start().catch(error => { setService("degraded"); fail(error.message); });
window.setInterval(syncActiveModel, 1500);
