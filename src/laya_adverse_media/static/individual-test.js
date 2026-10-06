const $ = selector => document.querySelector(selector);
const form = $("#test-form");
const modelSelect = $("#model-id");
const runButton = $("#run-button");
const errorBox = $("#error-message");
const articleInput = $("#article-text");
const promptDialog = $("#prompt-dialog");
let modelReady = false;
let modelChanging = false;
let availableModels = [];
let questionTemplate = "How does this article portray {entity_name} regarding criminal behavior or intent? Judge only the named entity, not other people or organizations.";
let promptCriteria = [
  {decision: "negative", text: "negative: the article credibly associates the entity with alleged, investigated, charged, convicted, sanctioned, or admitted criminal behavior or intent"},
  {decision: "positive", text: "positive: the article does not associate the entity with criminal behavior or intent, or identifies the entity only as a victim, witness, investigator, or unrelated party"},
];
const pct = value => `${((value || 0) * 100).toFixed(1)}%`;

function fail(message = "") {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function setService(status) {
  const ready = status === "ok";
  $("#model-health").className = `model-health ${ready ? "ready" : status === "warming" ? "" : "degraded"}`;
  $("#model-status-label").textContent = ready ? "Model ready" : status === "warming" ? "Warming model" : "Model unavailable";
}

function setActiveModel() {
  $("#active-model").textContent = modelSelect.selectedOptions[0]?.textContent || "No model available";
}

function applySelectedModelPrompt() {
  const model = availableModels.find(item => item.id === modelSelect.value);
  if (!model?.prompt) return;
  questionTemplate = model.prompt.question;
  promptCriteria = structuredClone(model.prompt.criteria);
  $("#prompt-summary").textContent = `${questionTemplate.replace("{entity_name}", "Entity")} · ${promptCriteria.length} criteria`;
}

function criterionRow(criterion = {decision: "negative", text: ""}) {
  const row = document.createElement("div");
  row.className = "criterion-row";
  const decision = document.createElement("select");
  decision.append(new Option("Negative", "negative"), new Option("Positive", "positive"));
  decision.value = criterion.decision;
  decision.setAttribute("aria-label", "Criterion decision");
  const text = document.createElement("textarea");
  text.value = criterion.text;
  text.maxLength = 4000;
  text.required = true;
  text.setAttribute("aria-label", "Criterion text");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "×";
  remove.title = "Remove criterion";
  remove.setAttribute("aria-label", "Remove criterion");
  remove.onclick = () => row.remove();
  row.append(decision, text, remove);
  return row;
}

function openPromptEditor() {
  $("#prompt-question").value = questionTemplate;
  $("#criteria-list").replaceChildren(...promptCriteria.map(criterionRow));
  promptDialog.showModal();
}

function savePrompt() {
  const question = $("#prompt-question").value.trim();
  const criteria = [...$("#criteria-list").children].map(row => ({
    decision: row.querySelector("select").value,
    text: row.querySelector("textarea").value.trim(),
  }));
  if (!question) return $("#prompt-question").reportValidity();
  if (criteria.some(criterion => !criterion.text)) return [...$("#criteria-list textarea")].find(input => !input.value.trim()).reportValidity();
  if (!["negative", "positive"].every(decision => criteria.some(criterion => criterion.decision === decision))) {
    fail("Prompt criteria must include negative and positive decisions");
    return;
  }
  questionTemplate = question;
  promptCriteria = criteria;
  $("#prompt-summary").textContent = `${questionTemplate.replace("{entity_name}", "Entity")} · ${criteria.length} criteria`;
  fail();
  promptDialog.close();
}

async function activateSelectedModel() {
  setActiveModel();
  applySelectedModelPrompt();
  modelChanging = true;
  modelReady = false;
  modelSelect.disabled = true;
  runButton.disabled = true;
  setService("warming");
  fail();
  try {
    const response = await fetch("/v1/models/activate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model_id: modelSelect.value}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not load model");
    modelReady = true;
    setService("ok");
  } catch (error) {
    setService("unavailable");
    fail(error.message);
  } finally {
    modelChanging = false;
    modelSelect.disabled = false;
    runButton.disabled = !modelReady;
  }
}

async function syncActiveModel() {
  if (modelChanging) return;
  try {
    const response = await fetch("/v1/models");
    if (!response.ok) return;
    const data = await response.json();
    if (data.active_model === "loading") {
      modelReady = false;
      modelSelect.disabled = true;
      runButton.disabled = true;
      setService("warming");
      return;
    }
    if (data.active_model === "unavailable") {
      modelReady = false;
      runButton.disabled = true;
      setService("unavailable");
      return;
    }
    const active = data.active_model || "auto";
    const option = [...modelSelect.options].find(item => item.value === active);
    if (!option) return;
    modelSelect.value = active;
    availableModels = data.models;
    applySelectedModelPrompt();
    modelReady = true;
    modelSelect.disabled = false;
    runButton.disabled = false;
    setActiveModel();
    setService("ok");
  } catch {
    setService("unavailable");
  }
}

async function loadModels() {
  const healthResponse = await fetch("/health");
  const health = await healthResponse.json();
  setService(health.status);
  if (health.error) throw new Error(health.error);
  if (health.status !== "ok") {
    window.setTimeout(() => loadModels().catch(error => fail(error.message)), 1200);
    return;
  }
  const response = await fetch("/v1/models");
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not load models");
  availableModels = data.models;
  modelSelect.replaceChildren(...data.models.map(model => new Option(model.label, model.id)));
  if ([...modelSelect.options].some(option => option.value === data.active_model)) modelSelect.value = data.active_model;
  applySelectedModelPrompt();
  modelReady = Boolean(data.models.length);
  modelSelect.disabled = !data.models.length;
  runButton.disabled = !data.models.length;
  setActiveModel();
}

function render(result, elapsed) {
  const negative = result.probabilities.negative;
  const positive = result.probabilities.positive;
  $("#decision-title").textContent = result.entity_name;
  $("#decision").textContent = result.decision;
  $("#confidence").textContent = `${pct(Math.max(negative, positive))} class probability · ${pct(result.confidence)} certainty`;
  $("#negative-value").textContent = pct(negative);
  $("#positive-value").textContent = pct(positive);
  $("#negative-bar").style.width = pct(negative);
  $("#positive-bar").style.width = pct(positive);
  $("#result-model").textContent = modelSelect.selectedOptions[0].textContent;
  $("#review-detail").textContent = result.needs_review ? "Manual review recommended" : "Confidence threshold met";
  $("#result-time").textContent = `${elapsed.toFixed(2)}s`;
  $("#review-state").textContent = result.needs_review ? "Review" : "Confident";
  $("#review-state").className = `badge ${result.needs_review ? "review" : "clear"}`;
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
    const response = await fetch("/v1/adverse-media", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        entity_name: $("#entity-name").value.trim(),
        article: articleInput.value.trim(),
        question: questionTemplate,
        criteria: promptCriteria,
        model_id: modelSelect.value === "auto" ? null : modelSelect.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      throw new Error(typeof detail === "string" ? detail : detail?.message || "Inference failed");
    }
    render(data, (performance.now() - started) / 1000);
  } catch (error) {
    fail(error.message);
  } finally {
    runButton.disabled = modelSelect.disabled || !modelReady;
    runButton.textContent = "Run test";
  }
});

$("#edit-prompt").onclick = openPromptEditor;
$("#add-criterion").onclick = () => $("#criteria-list").append(criterionRow());
$("#prompt-close").onclick = () => promptDialog.close();
$("#cancel-prompt").onclick = () => promptDialog.close();
$("#save-prompt").onclick = savePrompt;

loadModels().catch(error => { setService("degraded"); fail(error.message); });
window.setInterval(syncActiveModel, 1500);
