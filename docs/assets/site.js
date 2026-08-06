const state = {
  data: null,
  view: "full",
  category: "all",
  search: "",
  sortKey: "accuracy",
  sortDirection: "descending",
};

const stageKeys = ["CFA Level I", "CFA Level II", "CFA Level III", "FRM Part I", "FRM Part II"];

function valueFor(row, key) {
  return stageKeys.includes(key) ? row.stages[key] : row[key];
}

function sortRows(rows) {
  const direction = state.sortDirection === "ascending" ? 1 : -1;
  return rows.slice().sort((left, right) => {
    const a = valueFor(left, state.sortKey);
    const b = valueFor(right, state.sortKey);
    if (typeof a === "string") return direction * a.localeCompare(b);
    return direction * ((a ?? Number.NEGATIVE_INFINITY) - (b ?? Number.NEGATIVE_INFINITY));
  });
}

function filteredRows(view) {
  return view.rows.filter((row) => {
    const categoryMatch = state.category === "all" || row.category === state.category;
    const modelMatch = row.model.toLowerCase().includes(state.search);
    return categoryMatch && modelMatch;
  });
}

function cell(text, className = "") {
  const element = document.createElement("td");
  element.textContent = text;
  element.className = className;
  return element;
}

function score(value) {
  return value == null ? "—" : Number(value).toFixed(2);
}

function renderRows(view) {
  const body = document.querySelector("#leaderboard-body");
  const rows = sortRows(filteredRows(view));
  body.replaceChildren();
  if (rows.length === 0) {
    const row = document.createElement("tr");
    const message = cell("No models match these filters.", "empty-state");
    message.colSpan = 10;
    row.append(message);
    body.append(row);
    return 0;
  }
  rows.forEach((model, index) => {
    const row = document.createElement("tr");
    row.append(
      cell(String(index + 1), "rank"),
      cell(model.model, "model"),
      cell(model.category, "category"),
      ...stageKeys.map((key) => cell(score(model.stages[key]), "numeric")),
      cell(score(model.accuracy), "numeric primary-score"),
      cell(score(model.adjusted_accuracy), "numeric"),
    );
    body.append(row);
  });
  return rows.length;
}

function renderSummary(view) {
  document.querySelector("#view-title").textContent = view.title;
  document.querySelector("#view-description").textContent = view.description;
  document.querySelector("#view-items").textContent = view.items.toLocaleString();
  document.querySelector("#view-chance").textContent = view.chance_accuracy.toFixed(2) + "%";
  document.querySelector("#view-below").textContent = view.below_chance_systems + " / 17";
}

function render() {
  const view = state.data.views[state.view];
  renderSummary(view);
  const count = renderRows(view);
  document.querySelector("#results-status").textContent = count + " models shown";
}

function updateSort(button) {
  const key = button.dataset.sort;
  if (state.sortKey === key) {
    state.sortDirection = state.sortDirection === "descending" ? "ascending" : "descending";
  } else {
    state.sortKey = key;
    state.sortDirection = typeof valueFor(state.data.views[state.view].rows[0], key) === "string"
      ? "ascending" : "descending";
  }
  document.querySelectorAll("th[aria-sort]").forEach((heading) => {
    heading.setAttribute("aria-sort", "none");
  });
  button.closest("th").setAttribute("aria-sort", state.sortDirection);
  render();
}

function bindControls() {
  document.querySelector("#view-filter").addEventListener("change", (event) => {
    state.view = event.target.value;
    render();
  });
  document.querySelector("#category-filter").addEventListener("change", (event) => {
    state.category = event.target.value;
    render();
  });
  document.querySelector("#model-search").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    render();
  });
  document.querySelectorAll("[data-sort]").forEach((button) => {
    button.addEventListener("click", () => updateSort(button));
  });
}

async function loadLeaderboard() {
  const status = document.querySelector("#results-status");
  try {
    const response = await fetch("data/leaderboard.json");
    if (!response.ok) throw new Error("HTTP " + response.status);
    state.data = await response.json();
    const category = document.querySelector("#category-filter");
    state.data.categories.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      category.append(option);
    });
    bindControls();
    render();
    document.documentElement.dataset.ready = "true";
  } catch (error) {
    status.textContent = "Leaderboard data could not be loaded.";
    document.querySelector("#leaderboard-body").replaceChildren();
    document.documentElement.dataset.ready = "error";
    console.error("Leaderboard load failed", error);
  }
}

window.FinExamLeaderboard = {sortRows, filteredRows};
loadLeaderboard();
