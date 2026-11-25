"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const input = $("leaf-file");
  const form = $("predict-form");
  const drop = $("drop-zone");
  const empty = $("drop-empty");
  const preview = $("drop-preview");
  const previewImage = $("preview-image");
  const selectedName = $("selected-name");
  const submit = $("analyze-button");
  const buttonLabel = $("button-label");
  const buttonIcon = $("button-icon");
  const feedback = $("form-feedback");
  const resultEmpty = $("result-empty");
  const resultContent = $("result-content");
  const results = $("confidence-list");
  const status = $("model-status");
  const statusText = $("model-status-text");
  const labels = { angular_leaf_spot: "Angular leaf spot", bean_rust: "Bean rust", healthy: "Healthy leaf" };
  const details = {
    angular_leaf_spot: { title: "Signs of angular leaf spot", emoji: "🍂", text: "The model spotted patterns associated with angular leaf spot. Different leaf diseases can look similar, so treat this as a starting point." },
    bean_rust: { title: "Signs of bean rust", emoji: "🍁", text: "The model spotted patterns associated with bean rust. Check the leaf in person and consult a reliable plant-care resource before taking action." },
    healthy: { title: "Looking leafy & lovely!", emoji: "🌱", text: "Your photo looks most like the healthy leaf examples in the training dataset. Keep observing your plant as it grows." },
  };
  const MAX_SIZE = 5 * 1024 * 1024;
  const TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  let chosen = null;
  let previewUrl = null;
  let pending = false;

  function showError(message) { feedback.textContent = message; }
  function clearResult() { resultContent.classList.add("hidden"); resultEmpty.classList.remove("hidden"); results.replaceChildren(); }
  function selectFile(file) {
    if (pending) return;
    if (!file) return;
    if (!TYPES.has(file.type)) { showError("Please choose a JPG, PNG, or WebP image."); return; }
    if (file.size === 0) { showError("That image is empty. Please try another."); return; }
    if (file.size > MAX_SIZE) { showError("That file is too large. Please choose an image under 5 MB."); return; }
    showError("");
    chosen = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    selectedName.textContent = file.name;
    empty.classList.add("hidden");
    preview.classList.remove("hidden");
    submit.disabled = false;
    clearResult();
  }

  drop.addEventListener("click", () => { if (!pending) input.click(); });
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (!pending) input.click(); }
  });
  input.addEventListener("change", () => selectFile(input.files?.[0]));
  for (const name of ["dragenter", "dragover"]) {
    drop.addEventListener(name, (event) => { event.preventDefault(); if (!pending) drop.classList.add("drag-over"); });
  }
  for (const name of ["dragleave", "drop"]) drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("drag-over"); });
  drop.addEventListener("drop", (event) => { if (!pending) selectFile(event.dataTransfer?.files?.[0]); });

  const percent = (value) => `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
  function renderResult(data) {
    if (!Object.prototype.hasOwnProperty.call(details, data.label) || !data.scores) throw new Error("The server returned an unexpected prediction.");
    const match = details[data.label];
    $("result-title").textContent = match.title;
    $("result-detail").textContent = match.text;
    $("result-illustration").textContent = match.emoji;
    $("result-chip").textContent = "RESULT READY";
    results.replaceChildren();
    for (const [key, display] of Object.entries(labels)) {
      const numeric = Number(data.scores[key]);
      if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) throw new Error("The server returned an unexpected model score.");
      const row = document.createElement("div");
      row.className = "score-row";
      const top = document.createElement("div");
      top.className = "score-row-top";
      const name = document.createElement("span");
      name.textContent = display;
      const number = document.createElement("span");
      number.className = "score-percent";
      number.textContent = percent(numeric);
      top.append(name, number);
      const track = document.createElement("div");
      track.className = "score-track";
      const fill = document.createElement("div");
      fill.className = `score-fill${data.label === key ? " is-top" : ""}`;
      fill.style.width = percent(numeric);
      track.appendChild(fill);
      row.append(top, track);
      results.appendChild(row);
    }
    resultEmpty.classList.add("hidden");
    resultContent.classList.remove("hidden");
    if (window.innerWidth < 790) $("result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function setBusy(active) {
    pending = active;
    submit.disabled = active || !chosen;
    submit.classList.toggle("is-loading", active);
    buttonLabel.textContent = active ? "Listening to your leaf…" : "Analyze my leaf";
    buttonIcon.textContent = active ? "◌" : "↗";
    drop.setAttribute("aria-busy", active ? "true" : "false");
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!chosen || pending) return;
    showError("");
    setBusy(true);
    const body = new FormData();
    body.append("file", chosen, chosen.name);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const response = await fetch("/predict", { method: "POST", body, signal: controller.signal });
      if (!response.ok) {
        let reason = "Please try again in a moment.";
        try { const json = await response.json(); if (typeof json.detail === "string") reason = json.detail; } catch { /* keep default */ }
        throw new Error(reason);
      }
      renderResult(await response.json());
    } catch (error) {
      showError(error?.name === "AbortError" ? "The request took too long. The free cloud service may be waking up; please try again." : (error?.message || "We couldn't analyze that image. Please try again."));
    } finally { clearTimeout(timeout); setBusy(false); }
  });
  $("try-again").addEventListener("click", () => {
    chosen = null; input.value = "";
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    previewImage.removeAttribute("src");
    preview.classList.add("hidden");
    empty.classList.remove("hidden");
    clearResult();
    showError("");
    submit.disabled = true;
    drop.focus();
    $("try-it").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  async function checkModel() {
    try {
      const res = await fetch("/ready", { cache: "no-store" });
      if (!res.ok) throw new Error("not ready");
      statusText.textContent = "Model is ready";
      status.classList.remove("is-offline");
    } catch {
      statusText.textContent = "Model is waking up";
      status.classList.add("is-offline");
    }
  }
  checkModel();
})();
