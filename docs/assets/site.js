const state = {
  data: null,
  category: "all",
  search: "",
  sortKey: "accuracy",
  sortDirection: "descending",
};

const stageKeys = ["CFA Level I", "CFA Level II", "CFA Level III", "FRM Part I", "FRM Part II"];

const categoryLabels = {
  "Proprietary API served": "Proprietary",
  "Open weight reasoning": "Open-weight reasoning",
  "Finance specialized": "Finance-specialized",
};

function valueFor(row, key) {
  return stageKeys.includes(key) ? row.stages[key] : row[key];
}

function sortRows(rows) {
  const direction = state.sortDirection === "ascending" ? 1 : -1;
  return rows.slice().sort((left, right) => {
    const a = valueFor(left, state.sortKey);
    const b = valueFor(right, state.sortKey);
    const comparison = typeof a === "string"
      ? a.localeCompare(b)
      : (a ?? Number.NEGATIVE_INFINITY) - (b ?? Number.NEGATIVE_INFINITY);
    return comparison === 0 ? left.model.localeCompare(right.model) : direction * comparison;
  });
}

function filteredRows(rows) {
  return rows.filter((row) => {
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

function rankCell(rank) {
  const element = cell("", "rank-cell");
  const marker = document.createElement("span");
  marker.className = rank <= 3 ? "rank-mark top-rank" : "rank-mark";
  marker.textContent = String(rank);
  element.append(marker);
  return element;
}

function renderRows(rows) {
  const body = document.querySelector("#leaderboard-body");
  body.replaceChildren();
  if (rows.length === 0) {
    const row = document.createElement("tr");
    const message = cell("No models match these filters.", "empty-state");
    message.colSpan = 9;
    row.append(message);
    body.append(row);
    return;
  }

  rows.forEach((model, index) => {
    const row = document.createElement("tr");
    row.append(
      rankCell(index + 1),
      cell(model.model, "model-name"),
      cell(categoryLabels[model.category], "access-type"),
      cell(score(model.accuracy), "overall-score"),
      ...stageKeys.map((key) => cell(score(model.stages[key]))),
    );
    body.append(row);
  });
}

function updateSortState() {
  document.querySelectorAll("th[aria-sort]").forEach((heading) => {
    heading.setAttribute("aria-sort", "none");
  });
  const active = document.querySelector(`[data-sort="${state.sortKey}"]`);
  active.closest("th").setAttribute("aria-sort", state.sortDirection);
}

function render() {
  const sourceRows = state.data.views.full.rows;
  const rows = sortRows(filteredRows(sourceRows));
  renderRows(rows);
  updateSortState();
  document.querySelector("#results-status").textContent = `${rows.length} of ${sourceRows.length} models`;
}

function updateSort(key) {
  if (state.sortKey === key) {
    state.sortDirection = state.sortDirection === "descending" ? "ascending" : "descending";
  } else {
    state.sortKey = key;
    state.sortDirection = key === "model" ? "ascending" : "descending";
  }
  render();
}

function bindControls() {
  document.querySelectorAll("[data-category]").forEach((button) => {
    button.addEventListener("click", () => {
      state.category = button.dataset.category;
      document.querySelectorAll("[data-category]").forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      render();
    });
  });

  document.querySelector("#model-search").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    render();
  });

  document.querySelectorAll("[data-sort]").forEach((button) => {
    button.addEventListener("click", () => updateSort(button.dataset.sort));
  });
}

async function loadLeaderboard() {
  const status = document.querySelector("#results-status");
  try {
    const response = await fetch("data/leaderboard.json");
    if (!response.ok) throw new Error("HTTP " + response.status);
    state.data = await response.json();
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

window.FinExamLeaderboard = { state, sortRows, filteredRows, updateSort };
loadLeaderboard();
