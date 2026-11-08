// Live session view: snapshot from /api/sessions/{id}/state, then incremental SSE events.
(() => {
  const SID = window.SID;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const PALETTE = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"];
  const PHASES = ["understand", "plan", "wave1", "wave1-seg2", "wave2", "verify", "report"];
  const S = { runs: {}, charts: {}, cfgColor: {}, decisions: [], created: null, status: "", sandboxed: false };

  const color = (cfg) => {
    const base = cfg.split("~")[0];
    if (!(base in S.cfgColor)) S.cfgColor[base] = PALETTE[Object.keys(S.cfgColor).length % PALETTE.length];
    return S.cfgColor[base];
  };
  const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 }));
  const chartKey = (r) => `w${r.wave}|${r.env}`;
  const chartTitle = (r) => (r.wave === 1 ? `Wave 1 · ${r.env} · screening (1 seed)` : r.wave === 2 ? `Wave 2 · ${r.env} · confirmation` : `Red-team · ${r.env}`);

  function ensureChart(r) {
    const key = chartKey(r);
    if (S.charts[key]) return S.charts[key];
    const box = document.createElement("div");
    box.className = "chart card";
    box.innerHTML = `<h4>${esc(chartTitle(r))}</h4><div class="canvas"><canvas></canvas></div>`;
    const order = (k) => (k.startsWith("w1") ? 0 : k.startsWith("w2") ? 1 : 2);
    const kids = [...$("charts").children];
    const after = kids.find((c) => order(c.dataset.key) > order(key));
    box.dataset.key = key;
    $("charts").insertBefore(box, after || null);
    const css = getComputedStyle(document.documentElement);
    const grid = css.getPropertyValue("--grid").trim(), ink = css.getPropertyValue("--muted").trim();
    const chart = new Chart(box.querySelector("canvas"), {
      type: "line",
      data: { datasets: [] },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "linear", title: { display: true, text: "env steps", color: ink }, ticks: { color: ink, callback: (v) => (v >= 1000 ? v / 1000 + "k" : v) }, grid: { color: grid } },
          y: { title: { display: true, text: "return", color: ink }, ticks: { color: ink }, grid: { color: grid } },
        },
        plugins: { legend: { position: "bottom", labels: { color: ink, boxWidth: 10, font: { size: 11 }, filter: (it, d) => d.datasets[it.datasetIndex].showInLegend } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y)} @ ${fmt(c.parsed.x)}` } } },
      },
    });
    S.charts[key] = { chart, ds: {}, legend: new Set() };
    return S.charts[key];
  }

  function addPoints(r, pts) {
    if (r.kind === "verify") return;
    const c = ensureChart(r);
    let ds = c.ds[r.id];
    if (!ds) {
      const base = r.config.split("~")[0];
      const first = !c.legend.has(base); c.legend.add(base);
      ds = { label: r.is_fork ? `${r.config} (fork)` : base, data: [], borderColor: color(r.config), backgroundColor: color(r.config),
        borderWidth: r.is_fork ? 2 : 1.5, borderDash: r.is_fork ? [5, 4] : [], pointRadius: 0, tension: 0.2, showInLegend: first || r.is_fork };
      c.ds[r.id] = ds; c.chart.data.datasets.push(ds);
    }
    for (const [x, y] of pts) if (!ds.data.length || x > ds.data[ds.data.length - 1].x) ds.data.push({ x, y });
    if (["STOPPED", "FAILED", "CONTAINED"].includes(r.status)) ds.borderColor = ds.borderColor.slice(0, 7) + "77";
    scheduleChart(c);
  }
  const pending = new Set();
  function scheduleChart(c) {
    pending.add(c);
    if (pending.size === 1) requestAnimationFrame(() => { pending.forEach((x) => x.chart.update("none")); pending.clear(); });
  }

  // ---------------- runs table
  function renderRuns() {
    const rows = Object.values(S.runs).filter((r) => r.kind !== "verify")
      .sort((a, b) => a.wave - b.wave || a.env.localeCompare(b.env) || a.config.localeCompare(b.config) || a.seed - b.seed || a.segment - b.segment);
    $("runs").innerHTML = rows.map((r) => {
      const span = Math.max(1, r.total - r.start_at), done = Math.max(0, Math.min(1, (r.step - r.start_at) / span));
      const pct = r.status === "DONE" ? 100 : Math.round(done * 100);
      const wave = r.wave === 0 ? "red-team" : `${r.wave}${r.wave === 1 ? "." + r.segment : ""}`;
      const note = r.note ? ` title="${esc(r.note)}"` : "";
      return `<tr class="${r.is_fork ? "fork" : ""}"><td><span class="dot" style="background:${color(r.config)}"></span>${esc(r.config)}</td>
        <td>${esc(r.env)}</td><td>${r.seed}</td><td>${wave}</td>
        <td><span class="status s-${r.status}"${note}>${r.status}</span></td>
        <td><div class="bar"><span style="width:${pct}%"></span></div><span class="mono small">${fmt(r.step)}/${fmt(r.total)}</span></td>
        <td class="mono">${fmt(r.ret_recent)}</td><td class="mono">${fmt(r.eval_mean)}</td>
        <td>${(r.flags || []).map((f) => `<span class="flag">${esc(f)}</span>`).join(" ")}</td></tr>`;
    }).join("");
  }

  // ---------------- cards
  function renderBrief(b) {
    if (!b) return;
    $("brief").classList.remove("muted");
    $("brief").innerHTML = `<dl class="kv">
      <dt>Problem</dt><dd>${esc(b.problem_type)}${b.feasible ? "" : " · <b>not supported</b>"}</dd>
      <dt>Data</dt><dd>${esc(b.data_source.kind)}: ${esc(b.data_source.ref)}${b.also_evaluate_on?.length ? " → " + b.also_evaluate_on.map(esc).join(", ") : ""}</dd>
      <dt>Metric</dt><dd>${esc(b.metric)}</dd><dt>Budget</dt><dd>${b.budget_minutes ? b.budget_minutes + " min" : "default"}</dd></dl>
      <ul class="qs">${b.questions.map((q) => `<li>${esc(q)}</li>`).join("")}</ul>${b.scope_note ? `<p class="warn-text">${esc(b.scope_note)}</p>` : ""}`;
  }
  function renderPlan(p, budget) {
    if (!p) return;
    $("plan").classList.remove("muted");
    const bud = budget ? `<p class="small muted">Wave 1: ${fmt(budget.b1)} + ${fmt(budget.b1)} steps · wave 2: ${fmt(budget.b2)} steps × ${budget.seeds} seeds · est. ${budget.est_minutes} min at ${fmt(budget.sps)} sps × ${budget.slots} slots</p>` : "";
    $("plan").innerHTML = `<p>${esc(p.task_summary)}</p>${bud}<ul class="cfgs">${p.configs.map((c) => `<li><span class="dot" style="background:${color(c.name)}"></span><b>${esc(c.name)}</b>${c.name === p.hypothesis_config ? ' <span class="tag">hypothesis</span>' : ""} <span class="mono small">${esc(Object.entries(c.params).map(([k, v]) => `${k}=${v}`).join(", ") || "defaults")}</span><div class="small muted">${esc(c.rationale)}</div></li>`).join("")}</ul>`;
    renderPreds(p.predictions, S.outcomes);
  }
  function renderPreds(preds, outcomes) {
    if (!preds) return;
    const out = Object.fromEntries((outcomes || []).map((o) => [o.id, o.outcome]));
    $("preds").classList.remove("muted");
    $("preds").innerHTML = `<ul class="preds">${preds.map((p) => `<li><span class="pid">${esc(p.id)}</span> ${esc(p.statement)} <span class="p">p=${p.p}</span>${out[p.id] ? ` <span class="outcome o-${out[p.id]}">${out[p.id]}</span>` : ""}</li>`).join("")}</ul>`;
  }

  // ---------------- feeds
  const ACTION_LABEL = { stop: "STOP", fork: "FORK", rescue: "RESCUE", cut: "CUT", rung: "RUNG", select: "SELECT", plan: "PLAN", brief: "BRIEF", note: "NOTE", trim: "TRIM" };
  function addDecision(d, prepend = true) {
    const li = document.createElement("li");
    const act = d.action.startsWith("reject") ? "REJECTED " + d.action.slice(7).toUpperCase() : (ACTION_LABEL[d.action] || d.action.toUpperCase());
    li.className = `dec a-${d.action.split("-")[0]}`;
    li.innerHTML = `<span class="time mono">${new Date(d.ts * 1000).toLocaleTimeString()}</span><span class="actor ac-${esc(d.actor)}">${esc(d.actor)}</span><b>${esc(act)}</b>${d.run_id ? ` <span class="mono small">${esc(d.run_id.split("-").slice(1).join("-"))}</span>` : ""}<div class="reason">${esc(d.reason)}</div>`;
    prepend ? $("decisions").prepend(li) : $("decisions").append(li);
  }
  function addEvent(ev) {
    if (ev.kind === "phase") return setPhase(ev.payload.phase, ev.payload.detail);
    if (ev.kind === "flag") {
      const li = document.createElement("li");
      li.innerHTML = `<span class="time mono">${new Date(ev.ts * 1000).toLocaleTimeString()}</span><span class="flag">${esc(ev.payload.flag)}</span> ${esc(ev.payload.config)} <span class="mono small muted">${esc(ev.payload.run_id.split("-").slice(1).join("-"))}</span>`;
      $("flags").prepend(li);
    }
    if (ev.kind === "containment" || ev.kind === "redteam") {
      const li = document.createElement("li");
      li.className = ev.kind === "containment" ? "contained" : "";
      li.innerHTML = ev.kind === "containment"
        ? `<span class="time mono">${new Date(ev.ts * 1000).toLocaleTimeString()}</span><b>CONTAINED</b> ${esc(ev.payload.config)}<div class="reason mono">${esc(ev.payload.reason)}</div>`
        : `<span class="time mono">${new Date(ev.ts * 1000).toLocaleTimeString()}</span>red-team <b>${esc(ev.payload.preset)}</b> launched<div class="reason small">${esc(ev.payload.note)}</div>`;
      $("contain").prepend(li);
    }
    if (ev.kind === "error") setPhase("failed", ev.payload.error);
  }

  function setPhase(phase, detail) {
    S.status = phase;
    const idx = PHASES.indexOf(phase);
    document.querySelectorAll("#stepper li").forEach((li, i) => {
      li.className = phase === "done" || (idx >= 0 && i < idx) ? "done" : i === idx ? "active" : "";
    });
    $("status").textContent = phase;
    $("status").className = `status s-${phase}`;
    if (detail) $("phase-detail").textContent = detail;
    const finished = ["done", "failed", "unsupported", "interrupted", "cancelled"].includes(phase);
    $("cancel").classList.toggle("hidden", finished);
    if (phase === "done") $("report-link").classList.remove("hidden");
  }

  function renderRedteam(presets, sandboxed) {
    $("contain-note").textContent = sandboxed
      ? "Send a malicious reward function with the AST check turned off: the gVisor sandbox has to hold on its own."
      : "Red-team presets are disabled: this node runs the local dev backend (no sandbox). Point the control plane at a gVisor worker to use them.";
    $("redteam").innerHTML = presets.map((p) => `<button class="chip danger" data-p="${p}" ${sandboxed ? "" : "disabled"}>${p}</button>`).join("");
    $("redteam").querySelectorAll("button").forEach((b) => (b.onclick = async () => {
      const r = await fetch(`/api/sessions/${SID}/redteam/${b.dataset.p}`, { method: "POST" });
      if (!r.ok) alert((await r.json()).detail);
    }));
  }

  // ---------------- boot
  async function boot() {
    const st = await (await fetch(`/api/sessions/${SID}/state`)).json();
    S.created = st.created_at;
    S.outcomes = st.predictions;
    renderBrief(st.brief);
    renderPlan(st.plan, st.budget);
    st.runs.forEach((r) => (S.runs[r.id] = r));
    st.runs.forEach((r) => addPoints(r, st.series[r.id] || []));
    renderRuns();
    st.decisions.forEach((d) => addDecision(d));
    st.events.forEach(addEvent);
    setPhase(st.phase, null);
    if (st.report_ready) $("report-link").classList.remove("hidden");
    renderRedteam(st.redteam_presets, st.system.sandboxed);

    const es = new EventSource(`/sessions/${SID}/stream`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "points") { S.runs[ev.run_id] = ev.run; addPoints(ev.run, ev.points); renderRunsSoon(); }
      else if (ev.type === "run") { S.runs[ev.run.id] = ev.run; if (S.charts[chartKey(ev.run)]) addPoints(ev.run, []); renderRunsSoon(); }
      else if (ev.type === "features") { ev.features.forEach((f) => { if (S.runs[f.run_id]) S.runs[f.run_id].flags = f.flags; }); renderRunsSoon(); }
      else if (ev.type === "decision") addDecision(ev);
      else if (ev.type === "event") { addEvent(ev); if (ev.kind === "phase" && ev.payload.phase === "brief") renderBrief(ev.payload); }
      else if (ev.type === "plan") renderPlan(ev.plan, ev.budget);
      else if (ev.type === "report") { $("report-link").classList.remove("hidden"); refreshOutcomes(); }
    };
    $("cancel").onclick = async () => { if (confirm("Cancel this session and kill its runs?")) await fetch(`/api/sessions/${SID}/cancel`, { method: "POST" }); };
    setInterval(() => {
      if (!S.created) return;
      const s = Math.round(Date.now() / 1000 - S.created);
      $("elapsed").textContent = `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
    }, 1000);
    // brief arrives as a decision; refresh cards when it does
    setInterval(async () => {
      if (["done", "failed", "unsupported", "interrupted"].includes(S.status) && S.cardsFinal) return;
      if (!$("brief").classList.contains("muted") && !$("plan").classList.contains("muted")) { S.cardsFinal = true; return; }
      const st2 = await (await fetch(`/api/sessions/${SID}/state`)).json();
      renderBrief(st2.brief); renderPlan(st2.plan, st2.budget);
    }, 3000);
  }
  let rt = null;
  function renderRunsSoon() { if (!rt) rt = setTimeout(() => { rt = null; renderRuns(); }, 400); }
  async function refreshOutcomes() {
    const st = await (await fetch(`/api/sessions/${SID}/state`)).json();
    S.outcomes = st.predictions; renderPlan(st.plan, st.budget);
  }
  boot();
})();
