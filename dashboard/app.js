const state = {
  summary: [],
  recommendations: [],
  runIds: [],
  comparisonRows: [],
};

const protocolDetails = {
  NL: {
    label: "Natural Language",
    text: "Plain prose between agents with explicit instructions to avoid structured formatting.",
    best: "Low overhead baseline",
    risk: "Intermediate state can be underspecified.",
  },
  MARKDOWN: {
    label: "Markdown",
    text: "Headings, bullet points, and numbered steps for inspectable intermediate reasoning.",
    best: "Strong math trace readability",
    risk: "Completion tokens grow quickly.",
  },
  JSON: {
    label: "JSON",
    text: "Validated structured objects using response_format and one parse retry.",
    best: "Compact, machine-readable math runs",
    risk: "Can compress away nuance.",
  },
  SHARED_MEMORY: {
    label: "Shared Memory",
    text: "A full blackboard snapshot is injected into downstream agents.",
    best: "Best reading/news quality",
    risk: "Highest prompt-token overhead.",
  },
};

const protocolOrder = ["NL", "MARKDOWN", "JSON", "SHARED_MEMORY"];

const examples = {
  math: "Janet has 24 apples. She gives 6 to her friend, buys 12 more, and then sells half of the total for $3 each. How much money does she make?",
  reading: "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was named after engineer Gustave Eiffel, whose company designed and built the tower. Question: Who was the Eiffel Tower named after?",
  news: "A city council approved a $42 million transit plan on Monday, adding 18 electric buses and three new rapid routes by 2027. Summarize the key facts.",
};

function $(selector) {
  return document.querySelector(selector);
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function htmlEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function setupNav() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      button.classList.add("active");
      $(`#${button.dataset.section}`).classList.add("active");
    });
  });
}

async function loadHealth() {
  const health = await api("/api/health");
  $("#keyDot").classList.toggle("ok", health.has_openai_key);
  $("#keyStatus").textContent = health.has_openai_key ? "Local key loaded" : "No local key";
}

async function loadSummary() {
  const data = await api("/api/summary");
  state.summary = data.summary;
  state.recommendations = data.recommendations;

  $("#topMetrics").innerHTML = [
    ["Pipeline runs", data.runCount || 360, "Saved experiment executions"],
    ["Message logs", data.messageCount || 1080, "Planner, Executor, Integrator traces"],
    ["Protocols", data.protocols.length, "NL, Markdown, JSON, Shared Memory"],
    ["Domains", data.domains.length, "Math, Reading, News"],
  ].map(metricCard).join("");

  renderProtocolCards();
  renderProtocolTable();
  renderWinnerCards();
  renderSummaryTable();
  renderRecommendations();
  drawCharts();
}

function metricCard([label, value, note]) {
  return `<article class="metric"><span>${label}</span><strong>${formatNumber(value)}</strong><p>${note}</p></article>`;
}

function renderProtocolCards() {
  $("#protocolCards").innerHTML = Object.entries(protocolDetails).map(([key, item]) => `
    <article class="panel">
      <p class="eyebrow">${key}</p>
      <h4>${item.label}</h4>
      <p>${item.text}</p>
      <span class="pill">${item.best}</span>
      <span class="pill">${item.risk}</span>
    </article>
  `).join("");
}

function tableFromRows(rows, columns) {
  return `
    <table>
      <thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `<tr>${columns.map((c) => `<td>${c.format ? c.format(row[c.key], row) : htmlEscape(row[c.key])}</td>`).join("")}</tr>`).join("")}
      </tbody>
    </table>
  `;
}

function renderProtocolTable() {
  const domain = $("#protocolDomain").value;
  const rows = state.summary.filter((row) => row.Domain === domain)
    .sort((a, b) => b["Completion Rate"] - a["Completion Rate"]);
  $("#protocolTable").innerHTML = tableFromRows(rows, [
    { key: "Protocol", label: "Protocol" },
    { key: "Completion Rate", label: "Completion", format: (v) => formatNumber(v, 3) },
    { key: "Mean Tokens", label: "Mean tokens", format: (v) => formatNumber(v) },
    { key: "Mean Prompt Tok", label: "Prompt", format: (v) => formatNumber(v) },
    { key: "Mean Compl Tok", label: "Completion tok", format: (v) => formatNumber(v) },
    { key: "Mean Latency (ms)", label: "Latency", format: (v) => `${formatNumber(v)} ms` },
  ]);
}

function renderWinnerCards() {
  $("#winnerCards").innerHTML = state.recommendations.map((r) => `
    <article class="metric">
      <span>${r.domain}</span>
      <strong>${r.qualityProtocol}</strong>
      <p>Quality winner: score ${formatNumber(r.qualityScore, 3)}, ${formatNumber(r.qualityTokens)} tokens</p>
      <p>Cost winner: <b>${r.costProtocol}</b>, ${formatNumber(r.costTokens)} tokens</p>
    </article>
  `).join("");
}

function renderSummaryTable() {
  $("#summaryTable").innerHTML = tableFromRows(state.summary, [
    { key: "Protocol", label: "Protocol" },
    { key: "Domain", label: "Domain" },
    { key: "Mean Tokens", label: "Tokens", format: (v) => formatNumber(v) },
    { key: "Mean Prompt Tok", label: "Prompt", format: (v) => formatNumber(v) },
    { key: "Mean Compl Tok", label: "Completion", format: (v) => formatNumber(v) },
    { key: "Mean Latency (ms)", label: "Latency", format: (v) => `${formatNumber(v)} ms` },
    { key: "Completion Rate", label: "Score", format: (v) => formatNumber(v, 3) },
  ]);
}

function renderRecommendations() {
  const rows = state.recommendations.map((r) => ({
    domain: r.domain,
    quality: r.qualityProtocol,
    cost: r.costProtocol,
    note: r.domain === "MATH"
      ? "Markdown wins completion; JSON is the budget option."
      : "Shared Memory wins quality; NL is the budget option.",
  }));
  $("#recommendTable").innerHTML = tableFromRows(rows, [
    { key: "domain", label: "Domain" },
    { key: "quality", label: "Quality first" },
    { key: "cost", label: "Cost first" },
    { key: "note", label: "Reason" },
  ]);
}

function drawBarChart(canvasId, field, maxValue = null) {
  const canvas = $(`#${canvasId}`);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const rows = state.summary;
  const colors = { MATH: "#2f6f55", READING: "#315f83", NEWS: "#a05d28" };
  const padding = { left: 44, right: 10, top: 18, bottom: 54 };
  const width = rect.width - padding.left - padding.right;
  const height = rect.height - padding.top - padding.bottom;
  const max = maxValue || Math.max(...rows.map((r) => Number(r[field])));
  const barW = width / rows.length * 0.72;

  ctx.strokeStyle = "#ddd8cc";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + height);
  ctx.lineTo(padding.left + width, padding.top + height);
  ctx.stroke();

  rows.forEach((row, i) => {
    const value = Number(row[field]);
    const x = padding.left + (width / rows.length) * i + 4;
    const h = (value / max) * height;
    const y = padding.top + height - h;
    ctx.fillStyle = colors[row.Domain] || "#2f6f55";
    ctx.fillRect(x, y, barW, h);
    ctx.save();
    ctx.translate(x + barW / 2, padding.top + height + 8);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = "#67706f";
    ctx.font = "11px system-ui";
    ctx.fillText(`${row.Protocol}/${row.Domain[0]}`, 0, 0);
    ctx.restore();
  });
}

function drawCharts() {
  requestAnimationFrame(() => {
    drawBarChart("tokenChart", "Mean Tokens");
    drawBarChart("completionChart", "Completion Rate", 0.9);
  });
}

async function loadConfig() {
  const config = await api("/api/config");
  $("#configBlock").textContent = JSON.stringify(config, null, 2);
  $("#configBlock").classList.toggle("hidden");
}

async function loadLogs() {
  const protocol = $("#logProtocol").value;
  const domain = $("#logDomain").value;
  const sender = $("#logSender").value;
  const runId = $("#logRun").value || "";
  const data = await api(`/api/messages?protocol=${protocol}&domain=${domain}&sender=${sender}&run_id=${runId}`);
  state.runIds = data.runIds || [];
  const runSelect = $("#logRun");
  const current = runSelect.value;
  runSelect.innerHTML = `<option value="">First matching messages</option>` + state.runIds.map((id) => `<option ${id === current ? "selected" : ""}>${id}</option>`).join("");
  $("#messageList").innerHTML = data.messages.map((m) => `
    <article class="message-card">
      <div class="message-meta">
        <span>${m.run_id}</span><span>${m.protocol}</span><span>${m.task_domain}</span>
        <span>${m.sender} -> ${m.receiver}</span><span>${m.total_tokens} tokens</span><span>${formatNumber(m.latency_ms)} ms</span>
      </div>
      <pre>${htmlEscape(m.content)}</pre>
    </article>
  `).join("") || `<p class="muted">No messages match these filters.</p>`;
}

function setupDemo() {
  $("#exampleSelect").addEventListener("change", (event) => {
    $("#taskInput").value = examples[event.target.value] || "";
    renderProtocolPreview();
  });
  $("#taskInput").addEventListener("input", renderProtocolPreview);
  document.querySelectorAll(".protocolChoice").forEach((input) => {
    input.addEventListener("change", renderProtocolPreview);
  });
  $("#runBtn").addEventListener("click", runProtocolComparison);
  renderProtocolPreview();
}

function selectedProtocols() {
  return [...document.querySelectorAll(".protocolChoice:checked")].map((input) => input.value);
}

function protocolPreviewText(protocol, task) {
  const clipped = task ? task.slice(0, 180) + (task.length > 180 ? "..." : "") : "[your task]";
  if (protocol === "NL") {
    return `Plain English message carrying the task intent:\n${clipped}`;
  }
  if (protocol === "MARKDOWN") {
    return `## Task\n- Input: ${clipped}\n- Expected handoff: structured steps and concise result`;
  }
  if (protocol === "JSON") {
    return JSON.stringify({ task: clipped, expected_output: "structured intermediate result", protocol: "JSON" }, null, 2);
  }
  return JSON.stringify({
    blackboard_state: {
      task_prompt: clipped,
      planner_notes: "written after Planner runs",
      executor_result: "written after Executor runs",
    },
  }, null, 2);
}

function renderProtocolPreview() {
  const task = $("#taskInput").value.trim();
  const protocols = selectedProtocols();
  $("#protocolPreview").innerHTML = protocols.map((protocol) => `
    <article class="preview-card">
      <div class="message-meta"><span class="pill">${protocol}</span><span>${protocolDetails[protocol]?.label || protocol}</span></div>
      <pre>${htmlEscape(protocolPreviewText(protocol, task))}</pre>
    </article>
  `).join("") || `<p class="muted">Select at least one protocol to preview.</p>`;
}

function protocolAxisLabel(protocol) {
  if (protocol === "SHARED_MEMORY") return ["Shared", "Memory"];
  if (protocol === "MARKDOWN") return ["Markdown"];
  return [protocol];
}

function sortByProtocolOrder(rows) {
  return [...rows].sort((a, b) => protocolOrder.indexOf(a.protocol) - protocolOrder.indexOf(b.protocol));
}

function drawDemoMetricChart(canvasId, rows, field, color, yAxisLabel) {
  const canvas = $(`#${canvasId}`);
  if (!canvas || !rows.length) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const padding = { left: 72, right: 18, top: 22, bottom: 68 };
  const width = rect.width - padding.left - padding.right;
  const height = rect.height - padding.top - padding.bottom;
  const max = Math.max(...rows.map((r) => Number(r[field]) || 0), 1);
  const yMax = Math.ceil(max * 1.12);
  const slotW = width / rows.length;
  const barW = Math.min(62, slotW * 0.46);

  ctx.textBaseline = "middle";
  ctx.font = "12px system-ui";
  ctx.fillStyle = "#67706f";
  ctx.save();
  ctx.translate(18, padding.top + height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(yAxisLabel, 0, 0);
  ctx.restore();

  ctx.strokeStyle = "#e4dfd4";
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = yMax * tick / 4;
    const y = padding.top + height - (value / yMax) * height;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + width, y);
    ctx.stroke();
    ctx.fillStyle = "#67706f";
    ctx.fillText(formatNumber(value), padding.left - 10, y);
  }

  ctx.strokeStyle = "#cfc8ba";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + height);
  ctx.lineTo(padding.left + width, padding.top + height);
  ctx.stroke();

  rows.forEach((row, i) => {
    const value = Number(row[field]) || 0;
    const centerX = padding.left + slotW * i + slotW / 2;
    const x = centerX - barW / 2;
    const h = (value / yMax) * height;
    const y = padding.top + height - h;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, barW, Math.max(h, 1));
    ctx.fillStyle = "#191923";
    ctx.font = "700 12px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(value), centerX, y - 6);
    ctx.fillStyle = "#67706f";
    ctx.font = "12px system-ui";
    ctx.textBaseline = "top";
    protocolAxisLabel(row.protocol).forEach((label, line) => {
      ctx.fillText(label, centerX, padding.top + height + 14 + line * 15);
    });
  });
}

function renderComparisonCharts(rows) {
  state.comparisonRows = rows;
  requestAnimationFrame(() => {
    drawDemoMetricChart("demoTokenChart", rows, "total_tokens", "#2f6f55", "Tokens");
    drawDemoMetricChart("demoLatencyChart", rows, "latency_ms", "#315f83", "Latency (ms)");
  });
}

async function runProtocolComparison() {
  const task = $("#taskInput").value.trim();
  if (!task) return;
  const protocols = selectedProtocols();
  if (!protocols.length) {
    $("#runStatus").textContent = "Select at least one protocol.";
    return;
  }
  $("#runBtn").disabled = true;
  $("#runStatus").textContent = `Running ${protocols.length} protocol${protocols.length > 1 ? "s" : ""} for comparison...`;
  $("#runOutput").classList.remove("empty");
  $("#runOutput").innerHTML = `<p class="muted">Working. This runs the same task once per selected protocol.</p>`;
  try {
    const result = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify({
        task,
        model: $("#modelSelect").value,
        protocols,
      }),
    });
    const rows = [...result.rows].sort((a, b) => a.total_tokens - b.total_tokens);
    $("#runStatus").textContent = `Done: ${result.domain} (${formatNumber(result.confidence, 2)}). Fewest tokens: ${result.winners.fewest_tokens}; fastest: ${result.winners.fastest}.`;
    $("#runOutput").innerHTML = `
      <article class="recommendation-card">
        <p class="eyebrow">Recommendation for this input</p>
        <h4>${result.recommendation.recommended_protocol || "No recommendation"}</h4>
        <p>${htmlEscape(result.recommendation.reason)}</p>
        <div class="message-meta">
          ${result.recommendation.alternatives.map((alt) => `
            <span class="pill">${alt.label}: ${alt.protocol} (${formatNumber(alt.tokens)} tok, ${formatNumber(alt.latency_ms)} ms)</span>
          `).join("")}
        </div>
      </article>
      <div class="message-meta">
        <span class="pill">${result.domain}</span>
        <span class="pill">fewest tokens: ${result.winners.fewest_tokens}</span>
        <span class="pill">cheapest: ${result.winners.cheapest}</span>
        <span class="pill">fastest: ${result.winners.fastest}</span>
      </div>
      ${tableFromRows(rows, [
        { key: "protocol", label: "Protocol" },
        { key: "total_tokens", label: "Total tokens", format: (v) => formatNumber(v) },
        { key: "prompt_tokens", label: "Prompt", format: (v) => formatNumber(v) },
        { key: "completion_tokens", label: "Completion", format: (v) => formatNumber(v) },
        { key: "estimated_cost_usd", label: "Est. cost", format: (v) => `$${formatNumber(v, 5)}` },
        { key: "latency_ms", label: "Latency", format: (v) => `${formatNumber(v)} ms` },
        { key: "tokens_per_second", label: "Tokens/sec", format: (v) => formatNumber(v, 2) },
      ])}
    `;
    renderComparisonCharts(sortByProtocolOrder(result.rows));
  } catch (error) {
    $("#runStatus").textContent = "Run failed.";
    $("#runOutput").innerHTML = `<p class="muted">${htmlEscape(error.message)}</p>`;
  } finally {
    $("#runBtn").disabled = false;
  }
}

function setupEvents() {
  $("#protocolDomain").addEventListener("change", renderProtocolTable);
  $("#configBtn").addEventListener("click", loadConfig);
  ["#logProtocol", "#logDomain", "#logSender", "#logRun"].forEach((selector) => {
    $(selector).addEventListener("change", loadLogs);
  });
  $("#loadLogs").addEventListener("click", loadLogs);
  window.addEventListener("resize", () => {
    drawCharts();
    if (state.comparisonRows.length) renderComparisonCharts(state.comparisonRows);
  });
}

async function init() {
  setupNav();
  setupEvents();
  setupDemo();
  await Promise.all([loadHealth(), loadSummary()]);
  await loadLogs();
}

init().catch((error) => {
  document.body.insertAdjacentHTML("beforeend", `<div style="position:fixed;right:20px;bottom:20px;background:#9a3f3f;color:white;padding:16px;border-radius:12px;">${htmlEscape(error.message)}</div>`);
});
