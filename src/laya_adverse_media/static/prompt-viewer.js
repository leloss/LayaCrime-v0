// Read-only view of what a model is asked. Every model decides only between
// negative and positive, so prompts are shown for transparency, not edited.
const PromptViewer = (() => {
  const DECISION_TITLES = {negative: "Negative", positive: "Positive"};

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function section(title, ...children) {
    const block = element("section", "prompt-view-section");
    block.append(element("h3", "", title), ...children);
    return block;
  }

  function summary(model) {
    if (model?.prompt) {
      return model.prompt.question.replace("{entity_name}", "Entity");
    }
    if (model?.instruction) return "Fixed language-model instruction";
    if (model?.method) return "Fixed method · no prompt";
    return "Select a model to see its prompt";
  }

  function render(container, model) {
    const blocks = [];
    if (model?.prompt) {
      const question = element("p", "prompt-view-text", model.prompt.question);
      const note = element("small", "", "{entity_name} is replaced with the entity being classified.");
      blocks.push(section("Question", question, note));
      const list = element("dl", "prompt-view-criteria");
      model.prompt.criteria.forEach(criterion => {
        list.append(
          element("dt", criterion.decision, DECISION_TITLES[criterion.decision] || criterion.decision),
          element("dd", "", criterion.text),
        );
      });
      blocks.push(section("Decisions", list));
    } else if (model?.instruction) {
      blocks.push(section("Instruction", element("pre", "prompt-view-text", model.instruction)));
      blocks.push(section("Input", element("p", "prompt-view-text", "The entity name and the article text are sent after this instruction. The model returns label 1 (positive) or label 2 (negative).")));
    } else if (model?.method) {
      blocks.push(section("Method", element("p", "prompt-view-text", model.method)));
      blocks.push(section("Prompt", element("p", "prompt-view-text", "This baseline does not use a prompt.")));
    } else {
      blocks.push(element("p", "prompt-view-text", "No model is selected."));
    }
    blocks.push(element("p", "prompt-view-note", "Prompts are fixed: every model decides only between negative and positive."));
    container.replaceChildren(...blocks);
  }

  return {summary, render};
})();
