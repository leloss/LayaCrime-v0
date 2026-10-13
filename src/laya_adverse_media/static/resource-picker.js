class ResourcePicker {
  constructor(select, {isDeletable = () => false, onDelete, onError = () => {}} = {}) {
    this.select = select;
    this.isDeletable = isDeletable;
    this.onDelete = onDelete;
    this.onError = onError;
    this.busy = false;

    this.root = document.createElement("div");
    this.root.className = "resource-picker";
    this.trigger = document.createElement("button");
    this.trigger.type = "button";
    this.trigger.className = "resource-picker-trigger";
    this.trigger.setAttribute("aria-haspopup", "listbox");
    this.trigger.setAttribute("aria-expanded", "false");
    this.menu = document.createElement("div");
    this.menu.className = "resource-picker-menu";
    this.menu.hidden = true;
    this.menu.setAttribute("role", "listbox");
    this.root.append(this.trigger, this.menu);
    select.classList.add("resource-picker-native");
    select.setAttribute("aria-hidden", "true");
    select.tabIndex = -1;
    select.insertAdjacentElement("afterend", this.root);

    this.trigger.addEventListener("click", () => this.toggle());
    this.trigger.addEventListener("keydown", event => {
      if (event.key === "ArrowDown" && this.menu.hidden) {
        event.preventDefault();
        this.open();
      }
    });
    this.menu.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        event.preventDefault();
        this.close();
        this.trigger.focus();
      }
    });
    select.addEventListener("change", () => this.sync());
    document.addEventListener("click", event => {
      if (!this.root.contains(event.target)) this.close();
    });
    new MutationObserver(() => this.refresh()).observe(select, {
      attributes: true,
      attributeFilter: ["disabled"],
      childList: true,
      subtree: true,
    });
    this.refresh();
  }

  refresh() {
    const content = [...this.select.children].map(child => {
      if (child.tagName !== "OPTGROUP") return this.row(child);
      const group = document.createElement("div");
      group.className = "resource-picker-group";
      group.setAttribute("role", "group");
      group.setAttribute("aria-label", child.label);
      const heading = document.createElement("div");
      heading.className = "resource-picker-group-heading";
      heading.textContent = child.label;
      group.append(heading, ...[...child.children].map(option => this.row(option)));
      return group;
    });
    this.menu.replaceChildren(...content);
    this.sync();
  }

  row(option) {
    const row = document.createElement("div");
    row.className = "resource-picker-row";
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(option.selected));

    const choice = document.createElement("button");
    choice.type = "button";
    choice.className = "resource-picker-choice";
    choice.textContent = option.textContent;
    choice.disabled = option.disabled;
    choice.addEventListener("click", () => {
      this.select.value = option.value;
      this.select.dispatchEvent(new Event("change", {bubbles: true}));
      this.close();
    });
    row.append(choice);

    if (option.value && this.isDeletable(option.value)) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "resource-picker-delete";
      remove.innerHTML = "&#128465;";
      remove.title = `Delete ${option.textContent}`;
      remove.setAttribute("aria-label", `Delete ${option.textContent}`);
      remove.addEventListener("click", async event => {
        event.stopPropagation();
        if (this.busy || !confirm(`Delete "${option.textContent}"? This cannot be undone.`)) return;
        this.busy = true;
        this.sync();
        try {
          await this.onDelete(option.value, option.textContent);
          this.close();
        } catch (error) {
          this.onError(error);
        } finally {
          this.busy = false;
          this.refresh();
        }
      });
      row.append(remove);
    }
    return row;
  }

  sync() {
    const selected = this.select.selectedOptions[0];
    this.trigger.textContent = selected?.textContent || "Select an item";
    this.trigger.disabled = this.select.disabled || this.busy;
    this.menu.querySelectorAll(".resource-picker-row").forEach((row, index) => {
      row.setAttribute("aria-selected", String(this.select.options[index]?.selected || false));
    });
  }

  toggle() {
    if (this.menu.hidden) this.open(); else this.close();
  }

  open() {
    if (this.trigger.disabled) return;
    this.menu.hidden = false;
    this.trigger.setAttribute("aria-expanded", "true");
    const selected = this.menu.querySelector('[aria-selected="true"] .resource-picker-choice');
    (selected || this.menu.querySelector(".resource-picker-choice:not(:disabled)"))?.focus();
  }

  close() {
    this.menu.hidden = true;
    this.trigger.setAttribute("aria-expanded", "false");
  }
}

window.ResourcePicker = ResourcePicker;
