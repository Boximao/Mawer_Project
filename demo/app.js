const STATES = [
  "RECEIVED", "RETRIEVING", "RERANKING", "DRAFTING",
  "VALIDATING", "AWAITING_HUMAN", "RELEASED", "ABSTAINED", "FAILED"
];

const DOCS = [
  { id: "aapl-10q-2025-12-27", label: "Apple 10-Q · quarter ended 27 Dec 2025", pages: 28 },
  { id: "aapl-10q-2026-03-28", label: "Apple 10-Q · quarter ended 28 Mar 2026", pages: 32 },
  { id: "aapl-10q-2026-06-27", label: "Apple 10-Q · quarter ended 27 Jun 2026", pages: 32 }
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
    else if (run && run.visited.includes(s)) cls = "done";
    return `<li class="${cls}">${s}</li>`;
  }).join("");
}

function envelope() {
  if (!run) return;
  const body = {
    run_id: run.id,
    state: run.state,
    human_review: run.state !== "RELEASED",
    answer: run.state === "RELEASED" ? run.draft_answer : null,
    draft_answer: run.draft_answer,
    claims: run.claims,
    citations: run.citations,
    reason_codes: run.reason_codes,
    stop_reason: run.stop_reason,
    usage: run.usage
  };
  $("envelope").textContent = JSON.stringify(body, null, 2);
  $("runId").textContent = run.id;
}

async function updateCanon() {
  if (!run || run.state !== "AWAITING_HUMAN") {
    $("canon").textContent = "";
    $("sig").textContent = "";
    return;
  }
  const payload = JSON.stringify({ draft_answer: run.draft_answer, claims: run.claims, citations: run.citations });
  const payloadHash = await sha256Hex(payload);
  const toState = run.path === "abstain" ? "ABSTAINED" : "RELEASED";
  run.canon = `v1|${run.id}|AWAITING_HUMAN|${toState}|${run.step_id}|${payloadHash}|presenter|${run.ts}`;
  $("canon").textContent = run.canon;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function setStage(html) {
  $("stage").classList.remove("empty");
  $("stage").innerHTML = html;
}

async function goto(state, html) {
  run.state = state;
  if (!run.visited.includes(state)) run.visited.push(state);
  renderStates(state);
  if (html) setStage(html);
  envelope();
  await sleep(650);
}

async function runAgent() {
  const q = $("q").value.trim();
  if (!q) return;
  const path = q.toLowerCase().includes("tesla") ? "abstain" : "answer";
  run = {
    id: crypto.randomUUID(),
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
    usage: { steps: 0, tokens: 0, latency_ms: 0 }
  };
  $("approve").disabled = true;
  $("ack").disabled = true;
  renderStates("RECEIVED");
  envelope();

  await goto("RETRIEVING", `
    <div class="kv">
      <div>Query</div><div>${q}</div>
      <div>Vector top_k</div><div>20 · text-embedding-3-small · pgvector cosine</div>
      <div>Keyword top_k</div><div>20 · tsvector</div>
      <div>Fusion</div><div>RRF k=60 · discard fusion &lt; 0.40</div>
    </div>`);

  if (path === "abstain") {
    await goto("RERANKING", `<p>RRF candidates: <strong>0</strong> (Tesla is not on the allowlist). BGE not invoked.</p>`);
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
    <p>BGE <code>bge-reranker-base</code> on 8 fused chunks → keep 6.</p>
    <div class="kv">
      <div>chunk 0.91</div><div>10-Q 27 Dec 2025 · p.4 · Statements of Operations · diluted EPS 2.84</div>
      <div>chunk 0.74</div><div>same filing · p.4 · basic EPS 2.85 (not used as the answer)</div>
      <div>dropped</div><div>fusion 0.31 footer noise</div>
    </div>`);

  run.draft_answer = "Diluted earnings per share were $2.84 for the three months ended December 27, 2025.";
  run.claims = [{ text: "Diluted EPS $2.84", entailment: 0.97, numeric_verbatim: true }];
  run.citations = [{
    document_id: "aapl-10q-2025-12-27",
    filing_period: "2025-12-27",
    page: 4,
    section: "Condensed Consolidated Statements of Operations",
    quote: "Diluted $ 2.84 $ 2.40"
  }];
  run.usage = { steps: 3, tokens: 4120, latency_ms: 2400 };

  await goto("DRAFTING", `<p>Draft (not user-visible yet): ${run.draft_answer}</p><div class="cite">p.4 · ${run.citations[0].quote}</div>`);
  await goto("VALIDATING", `
    <div class="kv">
      <div>Entailment</div><div>0.97 ≥ 0.90</div>
      <div>Numeric</div><div>2.84 appears verbatim in cited span</div>
      <div>Temperature</div><div>0 · top_p 1.0</div>
    </div>
    <p>Validation passed, but release still requires HITL. Envelope <code>answer</code> is null.</p>`);
  await goto("AWAITING_HUMAN", `<p>PIN 2026 then <strong>Sign &amp; release</strong>. A model emitting <code>approved: true</code> is ignored.</p>`);
  $("approve").disabled = false;
  await updateCanon();
}

async function approve(kind) {
  if (!run || run.state !== "AWAITING_HUMAN") return;
  if ($("pin").value !== "2026") {
    $("sig").textContent = "PIN rejected";
    return;
  }
  const sig = await hmacHex(HITL_KEY, run.canon);
  $("sig").textContent = "HMAC-SHA256 " + sig;
  if (kind === "release") {
    run.state = "RELEASED";
    run.visited.push("RELEASED");
    setStage(`<p><strong>Released to presenter.</strong> ${run.draft_answer}</p><div class="cite">${run.citations[0].document_id} · p.${run.citations[0].page}<br>${run.citations[0].quote}</div>`);
  } else {
    run.state = "ABSTAINED";
    run.visited.push("ABSTAINED");
    run.draft_answer = null;
    setStage("<p><strong>Abstain acknowledged.</strong> Tesla is outside the Apple allowlist. No figure was generated.</p>");
  }
  renderStates(run.state);
  envelope();
  $("approve").disabled = true;
  $("ack").disabled = true;
}

renderDocs();
renderStates(null);
document.querySelectorAll("[data-script]").forEach((b) => {
  b.addEventListener("click", () => {
    $("q").value = SCRIPTS[b.dataset.script].question;
  });
});
$("run").addEventListener("click", runAgent);
$("approve").addEventListener("click", () => approve("release"));
$("ack").addEventListener("click", () => approve("ack"));
