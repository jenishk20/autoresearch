// Study view: one card per agent with live checkpoint videos. Snapshot + SSE.
(() => {
  const SID = window.SID;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const ROBOT = { "Hopper-v4": "Hopper", "Walker2d-v4": "Walker2d", "HalfCheetah-v4": "HalfCheetah", "InvertedDoublePendulum-v4": "Pendulum",
                  "PandaReachDense-v3": "Panda · reach", "PandaPushDense-v3": "Panda · push" };
  const NOW = { understand: "Reading the task", plan: "Planning", provision: "Preparing sandbox", wave1: "Screening",
    "wave1-seg2": "Screening", wave2: "Training", verify: "Testing on new situations", report: "Writing results",
    done: "Done", failed: "Failed", unsupported: "Not supported", interrupted: "Stopped", cancelled: "Stopped" };
  const PROG = { understand: 4, plan: 8, provision: 10, wave1: 20, "wave1-seg2": 35, wave2: 45, verify: 90, report: 96, done: 100 };
  const ATTACKS = { egress: ["🌐", "Reach the internet"], escape_write: ["📝", "Write outside the sandbox"],
                    membomb: ["💣", "Grab 8 GB of memory"], forkbomb: ["🍴", "Spawn 10,000 processes"], gaming: ["🎯", "Cheat the reward"] };
  const LETTERS = "ABCDEFGHIJ";
  const S = { runs: {}, plan: null, phase: "", created: null, cards: {}, letter: {}, selected: {} };

  const base = (c) => String(c).split("~")[0];
  const isGoal = (env) => String(env).startsWith("Panda");
  const fmt = (n) => (n == null ? "—" : Math.round(n).toLocaleString());
  const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
  const params = (p) => Object.entries(p || {}).map(([k, v]) => ({ learning_rate: `lr ${v}`, hidden: `${v} net`, ent_coef: `explore ${v}`, norm_obs: v ? "" : "raw inputs" }[k] ?? `${k} ${v}`)).filter(Boolean).join(" · ") || "defaults";
  const agentName = (cfg) => { const b = base(cfg); if (!(b in S.letter)) S.letter[b] = LETTERS[Object.keys(S.letter).length % LETTERS.length]; return `Agent ${S.letter[b]}`; };
  const score = (r) => isGoal(r.env) ? pct(r.eval_success ?? r.success) : fmt(r.eval_mean ?? r.ret_recent);
  const scoreLabel = (r) => (isGoal(r.env) ? "success" : "reward") + (r.eval_mean != null ? " · new situations" : "");

  // ---------------- agent cards
  const agentRuns = () => {
    const rs = Object.values(S.runs).filter((r) => r.kind !== "verify" && r.wave >= 1);
    if (!rs.length) return [];
    const top = Math.max(...rs.map((r) => r.wave));
    const letter = (r) => agentName(r.config);
    return rs.filter((r) => r.wave === top).sort((a, b) => a.env.localeCompare(b.env) || letter(a).localeCompare(letter(b)) || a.seed - b.seed || (a.is_fork - b.is_fork));
  };
  function ensureCard(r) {
    if (S.cards[r.id]) return S.cards[r.id];
    const el = document.createElement("div");
    el.className = "card agent";
    el.innerHTML = `<div class="stage"><div class="empty">First video at ${fmt(window.VIDEO_EVERY || 25000)} steps</div><span class="tag hidden"></span></div>
      <div class="body"><div class="row"><span class="name"></span><span class="metric"></span></div>
      <div class="row sub"><span class="cfg"></span><span class="mlabel"></span></div>
      <div class="spark"><canvas></canvas></div><div class="strip"></div></div>`;
    $("agents").appendChild(el);
    const css = getComputedStyle(document.documentElement);
    const chart = new Chart(el.querySelector("canvas"), {
      type: "line", data: { datasets: [{ data: [], borderColor: css.getPropertyValue("--accent").trim(), borderWidth: 1.6, pointRadius: 0, tension: 0.3, fill: false }] },
      options: { animation: false, responsive: true, maintainAspectRatio: false, parsing: false,
        scales: { x: { type: "linear", display: false }, y: { display: false } }, plugins: { legend: { display: false }, tooltip: { enabled: false } } },
    });
    S.cards[r.id] = { el, chart, shown: null };
    return S.cards[r.id];
  }
  function showVideo(r, v) {
    const c = ensureCard(r), stage = c.el.querySelector(".stage");
    if (c.shown === v.url) return;
    c.shown = v.url;
    stage.querySelector(".empty")?.remove();
    let vid = stage.querySelector("video");
    if (!vid) { vid = document.createElement("video"); vid.muted = true; vid.loop = true; vid.autoplay = true; vid.playsInline = true; stage.prepend(vid); }
    vid.src = v.url; vid.play().catch(() => {});
    const tag = stage.querySelector(".tag");
    tag.classList.remove("hidden");
    tag.textContent = `${v.final ? "final · " : ""}${fmt(v.step / 1000)}k steps${v.success != null ? " · " + (v.success ? "success" : "missed") : ""}`;
    c.el.querySelectorAll(".strip button").forEach((b) => b.classList.toggle("on", b.dataset.url === v.url));
  }
  function renderCard(r) {
    const c = ensureCard(r);
    c.el.querySelector(".name").textContent = `${agentName(r.config)}${r.is_fork ? " · branch" : ""}`;
    const planned = S.plan?.configs.find((x) => x.name === base(r.config));
    c.el.querySelector(".cfg").textContent = `${ROBOT[r.env] || r.env} · ${params(planned ? planned.params : {})}${r.seed > 1 ? " · seed " + r.seed : ""}`;
    c.el.querySelector(".metric").textContent = score(r);
    const trained = Math.round(100 * Math.max(0, Math.min(1, (r.step - r.start_at) / Math.max(1, r.total - r.start_at))));
    c.el.querySelector(".mlabel").textContent = r.status === "RUNNING" || r.status === "PENDING" ? `${scoreLabel(r)} · ${trained}% trained` : scoreLabel(r);
    const vids = [...(r.videos || [])];
    if (r.video_url) vids.push({ url: r.video_url, step: r.total, final: true, success: null });
    const strip = c.el.querySelector(".strip");
    if (strip.children.length !== vids.length) {
      strip.innerHTML = vids.map((v) => `<button data-url="${v.url}" title="${v.final ? "Final model on a new situation" : fmt(v.step) + " steps"}"><video src="${v.url}" muted preload="metadata"></video><span>${v.final ? "final" : fmt(v.step / 1000) + "k"}</span></button>`).join("");
      strip.querySelectorAll("button").forEach((b, i) => (b.onclick = () => { S.selected[r.id] = vids[i].url; showVideo(r, vids[i]); }));
    }
    const pick = vids.find((v) => v.url === S.selected[r.id]) || vids[vids.length - 1];
    if (pick) showVideo(r, pick);
  }
  function renderAgents() {
    const rs = agentRuns();
    const keep = new Set(rs.map((r) => r.id));
    for (const id of Object.keys(S.cards)) if (!keep.has(id)) { S.cards[id].el.remove(); delete S.cards[id]; }
    rs.forEach(renderCard);
    // one heading per robot when the study covers several; keep DOM order = sorted order
    const envs = [...new Set(rs.map((r) => r.env))];
    $("agents").querySelectorAll(".robot-h").forEach((h) => h.remove());
    rs.forEach((r, i) => {
      if (envs.length > 1 && (i === 0 || rs[i - 1].env !== r.env)) {
        const h = document.createElement("div"); h.className = "robot-h label"; h.textContent = ROBOT[r.env] || r.env;
        $("agents").appendChild(h);
      }
      $("agents").appendChild(S.cards[r.id].el);
    });
    renderRuns();
  }
  function addPoints(r, pts) {
    if (r.kind === "verify" || r.wave < 1) return;
    const c = S.cards[r.id] || (agentRuns().some((x) => x.id === r.id) ? ensureCard(r) : null);
    if (!c) return;
    const d = c.chart.data.datasets[0].data;
    for (const [x, y] of pts) if (!d.length || x > d[d.length - 1].x) d.push({ x, y });
    c.chart.update("none");
  }

  function renderRuns() {
    const rows = Object.values(S.runs).filter((r) => r.kind !== "verify");
    $("nruns").textContent = rows.length;
    $("runs").innerHTML = rows.map((r) => `<tr><td>${r.wave === 0 ? "attack" : agentName(r.config)}</td><td>${ROBOT[r.env] || r.env}</td><td>${r.seed}</td>
      <td>${r.status.toLowerCase()}</td><td>${Math.round(100 * Math.max(0, Math.min(1, (r.step - r.start_at) / Math.max(1, r.total - r.start_at))))}%</td><td>${score(r)}</td></tr>`).join("");
  }

  // ---------------- winner
  async function renderWinner() {
    try {
      const rep = await (await fetch(`/api/sessions/${SID}/report.json`)).json();
      const [env, t] = Object.entries(rep.per_env || {})[0] || [];
      const w = t && t.table && t.table[0];
      if (!w) return;
      $("winner").classList.remove("hidden");
      $("winner").innerHTML = `${w.video ? `<video src="${w.video}" autoplay muted loop playsinline></video>` : "<div></div>"}
        <div class="body"><span class="label">Winner · ${esc(ROBOT[env] || env)}</span>
        <span style="font-family:var(--serif);font-size:24px">${agentName(w.config)}</span>
        <span class="big">${w.success != null ? pct(w.success) : fmt(w.iqm)}</span>
        <span class="muted small">${w.success != null ? "success on new situations" : "reward on new situations"}</span>
        <a class="btn sm" style="align-self:flex-start;margin-top:10px" href="/sessions/${SID}/report">Full results</a></div>`;
    } catch (e) {}
  }

  // ---------------- log, phase, sandbox
  const LOG = { brief: "Read the task", plan: "Planned the agents", rung: "Kept the best half", cut: "Cut an agent", rescue: "Rescued an agent",
                select: "Started the agents", fork: "Branched an agent", stop: "Stopped an agent", trim: "Trimmed the plan" };
  function addDecision(d) {
    const li = document.createElement("li");
    const run = d.run_id && S.runs[d.run_id];
    const text = d.action.startsWith("reject") ? "Guardrail blocked an action" : (LOG[d.action] || "Reviewed progress") + (run ? ` · ${agentName(run.config)}` : "");
    li.title = d.reason || "";
    li.innerHTML = `<span class="tm">${new Date(d.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span><span>${esc(text)}</span>`;
    $("feed").prepend(li);
  }
  function addEvent(ev) {
    if (ev.kind === "phase") return setPhase(ev.payload.phase);
    if (ev.kind === "containment") {
      const name = (ev.payload.config || "").replace("redteam:", "");
      $("contain").querySelector(`[data-a="${name}"]`)?.remove();
      const li = document.createElement("li"); li.title = ev.payload.reason;
      li.innerHTML = `<b>Blocked</b> — ${esc((ATTACKS[name] || ["", name])[1])}`;
      $("contain").prepend(li);
    }
    if (ev.kind === "redteam") {
      const li = document.createElement("li"); li.dataset.a = ev.payload.preset;
      li.innerHTML = `<span class="muted">Trying — ${esc((ATTACKS[ev.payload.preset] || ["", ev.payload.preset])[1])}</span>`;
      $("contain").prepend(li);
    }
    if (ev.kind === "error") setPhase("failed");
  }
  function setPhase(p) {
    S.phase = p;
    $("now").textContent = NOW[p] || p;
    if (p in PROG) $("progress").style.width = PROG[p] + "%";
    const done = ["done", "failed", "unsupported", "interrupted", "cancelled"].includes(p);
    $("live").className = "dot " + (done ? (p === "done" ? "ok" : "warn") : "live");
    $("cancel").classList.toggle("hidden", done);
    if (p === "done") { $("report-link").classList.remove("hidden"); renderWinner(); }
  }
  function trainingProgress() {
    if (S.phase !== "wave2" && S.phase !== "wave1" && S.phase !== "wave1-seg2") return;
    const rs = agentRuns(); if (!rs.length) return;
    const f = rs.reduce((a, r) => a + Math.max(0, Math.min(1, (r.step - r.start_at) / Math.max(1, r.total - r.start_at))), 0) / rs.length;
    $("progress").style.width = (S.phase === "wave2" ? 45 + 45 * f : PROG[S.phase]) + "%";
  }
  function renderSandbox(on, presets) {
    $("sandbox").innerHTML = on ? `<span class="dot ok"></span> Each agent trains in its own sandbox · no internet` : `<span class="dot warn"></span> Dev mode · not sandboxed`;
    $("redteam").innerHTML = (presets || []).map((p) => `<button class="icon-btn" data-p="${p}" title="Try to break out: ${(ATTACKS[p] || ["", p])[1]}" ${on ? "" : "disabled"}>${(ATTACKS[p] || ["?"])[0]}</button>`).join("");
    $("redteam").querySelectorAll("button").forEach((b) => (b.onclick = async () => {
      const r = await fetch(`/api/sessions/${SID}/redteam/${b.dataset.p}`, { method: "POST" });
      if (!r.ok) alert((await r.json()).detail);
    }));
  }
  function renderPlan(p) {
    if (!p) return;
    S.plan = p;
    p.configs.forEach((c) => agentName(c.name));
    $("plan").innerHTML = p.configs.map((c) => `<div><b>${agentName(c.name)}</b> · ${esc(params(c.params))} — ${esc(c.rationale)}</div>`).join("");
  }

  let t = null;
  const soon = () => { if (!t) t = setTimeout(() => { t = null; renderAgents(); trainingProgress(); }, 400); };

  async function boot() {
    const st = await (await fetch(`/api/sessions/${SID}/state`)).json();
    S.created = st.created_at;
    renderPlan(st.plan);
    st.runs.forEach((r) => (S.runs[r.id] = r));
    renderAgents();
    st.runs.forEach((r) => addPoints(r, st.series[r.id] || []));
    st.events.forEach(addEvent);
    st.decisions.forEach(addDecision);
    setPhase(st.phase);
    trainingProgress();
    if (st.report_ready) { $("report-link").classList.remove("hidden"); renderWinner(); }
    renderSandbox(st.system.sandboxed, st.redteam_presets);
    const es = new EventSource(`/sessions/${SID}/stream`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "points") { S.runs[ev.run_id] = ev.run; addPoints(ev.run, ev.points); soon(); }
      else if (ev.type === "run" || ev.type === "video") { const r = ev.run; S.runs[r.id] = r; soon(); }
      else if (ev.type === "decision") addDecision(ev);
      else if (ev.type === "event") addEvent(ev);
      else if (ev.type === "plan") renderPlan(ev.plan);
      else if (ev.type === "report") { $("report-link").classList.remove("hidden"); renderWinner(); }
    };
    $("cancel").onclick = async () => { if (confirm("Stop this study?")) await fetch(`/api/sessions/${SID}/cancel`, { method: "POST" }); };
    setInterval(() => {
      if (!S.created || ["done", "failed", "unsupported", "interrupted", "cancelled"].includes(S.phase)) return;
      const s = Math.round(Date.now() / 1000 - S.created);
      $("elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 1000);
    const poll = setInterval(async () => {
      if (S.plan) return clearInterval(poll);
      renderPlan((await (await fetch(`/api/sessions/${SID}/state`)).json()).plan);
    }, 3000);
  }
  boot();
})();
