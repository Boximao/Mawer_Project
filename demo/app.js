const STATES = [
  "RECEIVED", "RETRIEVING", "RERANKING", "DRAFTING",
  "VALIDATING", "AWAITING_HUMAN", "RELEASED", "ABSTAINED", "FAILED"
];

const DOCS = [
  { id: "AAPL_10-Q_2025-12-27", label: "Apple 10-Q · quarter ended 27 Dec 2025", pages: 28 },
  { id: "AAPL_10-Q_2026-03-28", label: "Apple 10-Q · quarter ended 28 Mar 2026", pages: 32 },
  { id: "AAPL_10-Q_2026-06-27", label: "Apple 10-Q · quarter ended 27 Jun 2026", pages: 32 }
];

const SCRIPTS = {
  eps: {
    question: "What was Apple diluted earnings per share for the quarter ended December 27, 2025?",
    path: "answer"
  },
  tesla: {
    question: "What was Tesla FY2025 total revenue?",
    path: "abstain"
  }
};

const HITL_KEY = "workshop-demo-key-not-the-prod-secret";

let run = null;
let live = false;

function $(id) { return document.getElementById(id); }

function sha256Hex(message) {
  return crypto.subtle.digest("SHA-256", new TextEncoder().encode(message)).then((buf) =>
    [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("")
  );
}

async function hmacHex(key, msg) {
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(key),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", cryptoKey, new TextEncoder().encode(msg));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function renderDocs() {
  $("docs").innerHTML = DOCS.map(
    (d) => `<li><code>${d.id}</code> · ${d.label} · ${d.pages} pp</li>`
  ).join("");
}

function renderStates(current) {
  $("states").innerHTML = STATES.map((s) => {
    let cls = "";
    if (s === current) cls = current === "AWAITING_HUMAN" ? "on" : s === "RELEASED" ? "done" : s === "ABSTAINED" ? "block" : "on";
    else if (run && run.visited && run.visited.includes(s)) cls = "done";
    return `<li class="${cls}">${s}</li>`;
  }).join("");
}

function envelopeFrom(env) {
  return {
    run_id: env.run_id,
    state: env.state,
    human_review: env.human_review,
    answer: env.answer,
    draft_answer: env.draft_answer,
    claims: env.claims,
    citations: env.citations,
    reason_codes: env.reason_codes,
    stop_reason: env.stop_reason,
    usage: env.usage
  };
}

function envelope() {
  if (!run) return;
  $("envelope").textContent = JSON.stringify(envelopeFrom(run), null, 2);
  $("runId").textContent = run.run_id || run.id || "";
}

async function updateCanon() {
  if (!run || !["AWAITING_HUMAN", "FAILED"].includes(run.state)) {
    $("canon").textContent = "";
    $("sig").textContent = "";
    return;
  }
  const payload = JSON.stringify({ draft_answer: run.draft_answer, claims: run.claims, citations: run.citations });
  const payloadHash = await sha256Hex(payload);
  const toState = run.state === "FAILED" || (run.reason_codes || []).includes("OUT_OF_CORPUS") || run.path === "abstain" ? "ABSTAINED" : "RELEASED";
  run.canon = `v1|${run.run_id || run.id}|${run.state}|${toState}|${run.step_id || "s0"}|${payloadHash}|presenter|${run.ts || Math.floor(Date.now() / 1000)}`;
  $("canon").textContent = run.canon;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function setStage(html) {
  $("stage").classList.remove("empty");
  $("stage").innerHTML = html;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
}

async function goto(state, html) {
  run.state = state;
  if (!run.visited.includes(state)) run.visited.push(state);
  renderStates(state);
  if (html) setStage(html);
  envelope();
  await sleep(650);
}

function applyLiveEnvelope(env) {
  run = {
    ...env,
    id: env.run_id,
    visited: env.visited || [env.state],
    path: (env.reason_codes || []).includes("OUT_OF_CORPUS") ? "abstain" : "answer",
    human_review: env.human_review,
    ts: Math.floor(Date.now() / 1000)
  };
  renderStates(env.state);
  envelope();
  const kept = (env.trace && env.trace.kept) || [];
  const rows = kept.map((h) => `<div>${escapeHtml(h.chunk_id)}</div><div>p.${h.page} · ${escapeHtml(h.section)} · fusion ${h.fusion}${h.rerank != null ? " · rerank " + h.rerank : ""}</div>`).join("");
  setStage(`
    <div class="kv">
      <div>Mode</div><div>${escapeHtml((env.trace && env.trace.retrieve_mode) || "live")}</div>
      <div>Stop</div><div>${escapeHtml(env.stop_reason)}</div>
      <div>Reasons</div><div>${escapeHtml((env.reason_codes || []).join(", ") || "—")}</div>
    </div>
    <div class="kv">${rows || "<div>kept</div><div>none</div>"}</div>
    <p>Draft (not user-visible until HITL): ${escapeHtml(env.draft_answer || "null")}</p>
    <p>Envelope <code>answer</code> is ${env.answer == null ? "null" : "set"}.</p>`);
  $("approve").disabled = !(env.state === "AWAITING_HUMAN" && env.draft_answer);
  $("ack").disabled = !(
    (env.state === "AWAITING_HUMAN" && !env.draft_answer) || env.state === "FAILED"
  );
}

async function runLive(question) {
  setStage("<p>Running bounded loop (max 4 steps / 45s)…</p>");
  const r = await fetch("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  if (!r.ok) throw new Error("POST /runs " + r.status);
  const env = await r.json();
  applyLiveEnvelope(env);
  await updateCanon();
}

async function runAgent() {
  const q = $("q").value.trim();
  if (!q) return;
  $("approve").disabled = true;
  $("ack").disabled = true;
  $("sig").textContent = "";
  if (live) {
    try {
      await runLive(q);
      return;
    } catch (e) {
      setStage(`<p>Live API failed (${escapeHtml(e.message)}). Falling back to fixture.</p>`);
      live = false;
      $("modePill").textContent = "Fixture mode · no live API";
    }
  }
  await runFixture(q);
}

async function runFixture(q) {
  const path = q.toLowerCase().includes("tesla") ? "abstain" : "answer";
  run = {
    id: crypto.randomUUID(),
    run_id: null,
    path,
    state: "RECEIVED",
    visited: ["RECEIVED"],
    step_id: "s4",
    ts: Math.floor(Date.now() / 1000),
    draft_answer: null,
    claims: [],
    citations: [],
    reason_codes: [],
    stop_reason: "awaiting_human",
    usage: { steps: 0, tokens: 0, latency_ms: 0 },
    human_review: true,
    answer: null
  };
  run.run_id = run.id;
  renderStates("RECEIVED");
  envelope();

  await goto("RETRIEVING", `
    <div class="kv">
      <div>Query</div><div>${escapeHtml(q)}</div>
      <div>Vector top_k</div><div>20 · hybrid cosine · allowlisted Apple docs only</div>
      <div>Keyword top_k</div><div>20 · tsvector / lexical</div>
      <div>Fusion</div><div>RRF k=60 · discard fusion &lt; 0.40</div>
    </div>`);

  if (path === "abstain") {
    await goto("RERANKING", `<p>RRF candidates: <strong>0</strong> (Tesla is not on the allowlist). Cloud rerank not invoked.</p>`);
    await goto("DRAFTING", `<p>Generator skipped. Pretraining recall is forbidden when retrieve is empty.</p>`);
    run.reason_codes = ["OUT_OF_CORPUS", "INSUFFICIENT_EVIDENCE"];
    run.usage = { steps: 2, tokens: 0, latency_ms: 1800 };
    await goto("VALIDATING", `<p>Claim gate: no claims to score. Fail closed.</p>`);
    run.stop_reason = "empty_retrieve";
    await goto("AWAITING_HUMAN", `<p>State is <strong>AWAITING_HUMAN</strong>. <code>answer</code> stays null until a human acks the abstain.</p>`);
    $("ack").disabled = false;
    await updateCanon();
    return;
  }

  await goto("RERANKING", `
    <p>Cloud rerank or <strong>RRF-only</strong> (no local 278M reranker) → keep 6.</p>
    <div class="kv">
      <div>chunk 0.91</div><div>10-Q 27 Dec 2025 · p.4 · Statements of Operations · diluted EPS 2.84</div>
      <div>chunk 0.74</div><div>same filing · p.4 · basic EPS 2.85 (not used as the answer)</div>
      <div>dropped</div><div>fusion 0.31 footer noise</div>
    </div>`);

  run.draft_answer = "Diluted earnings per share were $2.84 for the three months ended December 27, 2025.";
  run.claims = [{ text: "Diluted EPS $2.84", entailment: 0.97, numeric_verbatim: true }];
  run.citations = [{
    document_id: "AAPL_10-Q_2025-12-27",
    filing_period: "2025-12-27",
    page: 4,
    section: "Condensed Consolidated Statements of Operations",
    quote: "Diluted $ 2.84 $ 2.40"
  }];
  run.usage = { steps: 3, tokens: 4120, latency_ms: 2400 };

  await goto("DRAFTING", `<p>Draft (not user-visible yet): ${run.draft_answer}</p><div class="cite">p.4 · ${run.citations[0].quote}</div>`);
  await goto("VALIDATING", `
    <div class="kv">
      <div>Entailment</div><div>0.97 ≥ 0.90 (quote substring + numeric verbatim)</div>
      <div>Numeric</div><div>2.84 appears verbatim in cited span</div>
      <div>Temperature</div><div>0 · top_p 1.0</div>
    </div>
    <p>Validation passed, but release still requires HITL. Envelope <code>answer</code> is null.</p>`);
  await goto("AWAITING_HUMAN", `<p>PIN 2026 then <strong>Sign &amp; release</strong>. A model emitting <code>approved: true</code> is ignored.</p>`);
  $("approve").disabled = false;
  await updateCanon();
}

async function approve(kind) {
  if (!run || !["AWAITING_HUMAN", "FAILED"].includes(run.state)) return;
  const pin = $("pin").value;
  if (live && run.run_id) {
    const toState = kind === "release" ? "RELEASED" : "ABSTAINED";
    try {
      const r = await fetch("/hitl/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: run.run_id, pin, to_state: toState, approver_id: "presenter" })
      });
      const env = await r.json();
      if (!r.ok) {
        $("sig").textContent = env.detail || "PIN rejected";
        return;
      }
      $("sig").textContent = "HMAC-SHA256 " + (env.signature || "");
      $("canon").textContent = env.canon || "";
      applyLiveEnvelope(env);
      if (env.state === "RELEASED") {
        setStage(`<p><strong>Released to presenter.</strong> ${escapeHtml(env.answer)}</p>`);
      } else {
        setStage("<p><strong>Abstain acknowledged.</strong> No figure was released.</p>");
      }
      $("approve").disabled = true;
      $("ack").disabled = true;
      return;
    } catch (e) {
      $("sig").textContent = "HITL failed: " + e.message;
      return;
    }
  }
  if (pin !== "2026") {
    $("sig").textContent = "PIN rejected";
    return;
  }
  const sig = await hmacHex(HITL_KEY, run.canon);
  $("sig").textContent = "HMAC-SHA256 " + sig;
  if (kind === "release") {
    run.state = "RELEASED";
    run.human_review = false;
    run.answer = run.draft_answer;
    run.visited.push("RELEASED");
    setStage(`<p><strong>Released to presenter.</strong> ${run.draft_answer}</p><div class="cite">${run.citations[0].document_id} · p.${run.citations[0].page}<br>${run.citations[0].quote}</div>`);
  } else {
    run.state = "ABSTAINED";
    run.visited.push("ABSTAINED");
    run.draft_answer = null;
    run.answer = null;
    setStage("<p><strong>Abstain acknowledged.</strong> Tesla is outside the Apple allowlist. No figure was generated.</p>");
  }
  renderStates(run.state);
  envelope();
  $("approve").disabled = true;
  $("ack").disabled = true;
}

async function detectLive() {
  try {
    const r = await fetch("/health", { signal: AbortSignal.timeout(800) });
    if (!r.ok) return;
    const j = await r.json();
    if (j && j.ok) {
      live = true;
      $("modePill").textContent = "Live · FastAPI " + (j.store || "");
      const c = await fetch("/corpus");
      if (c.ok) {
        const body = await c.json();
        if (body.documents && body.documents.length) {
          $("docs").innerHTML = body.documents.map(
            (d) => `<li><code>${d.document_id}</code> · ${d.form_type} · ${d.filing_period} · ${d.page_count || "?"} pp</li>`
          ).join("");
        }
      }
    }
  } catch (_) {
    live = false;
  }
}

renderDocs();
renderStates(null);
detectLive();
document.querySelectorAll("[data-script]").forEach((b) => {
  b.addEventListener("click", () => {
    $("q").value = SCRIPTS[b.dataset.script].question;
  });
});
$("run").addEventListener("click", runAgent);
$("approve").addEventListener("click", () => approve("release"));
$("ack").addEventListener("click", () => approve("ack"));
