/* Drives the console's goal pipeline panel in jsdom.
 *
 * The panel is a viewer over a server-owned queue, so what can go wrong is on this
 * side: rendering the wrong state, sending the wrong request, or rebuilding a button
 * under the pointer while tokens stream. Each check below is one of those.
 *
 * Usage:  node tests/js/goals_panel.js [path-to-console.html]
 * Prints FAIL lines and exits non-zero on any problem.
 */
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const file = process.argv[2] ||
  path.join(__dirname, "..", "..", "ui", "console.html");
const html = fs.readFileSync(file, "utf8");

const goal = (id, text, status, extra = {}) => ({
  id, text, status, done_when: "", agents: [], route: [], answer: "", error: "",
  added: "", started: "", finished: "", ...extra });

let SNAP = {
  type: "goals", running: true, pausing: false, current: "c1", stage: "dev_agent",
  awaiting: null, note: "", settings: { plan: true, fail: "stop", model: "" },
  goals: [
    goal("d1", "finished one", "done", { route: ["dev_agent"], answer: "## Result\nall good",
                                         finished: "2026-10-30T10:00:00" }),
    goal("f1", "broke", "failed", { error: "RuntimeError: boom", finished: "2026-10-30T10:01:00" }),
    goal("c1", "running now", "running", { route: ["research_agent", "dev_agent", "tester_agent"],
                                           done_when: "tests pass" }),
    goal("p1", "<img src=x onerror=alert(1)>", "pending"),
    goal("p2", "second waiting", "pending", { agents: ["dev_agent", "tester_agent"] }),
  ],
};

const calls = [];          // [method, url, body]
let ES = null;             // the page's EventSource

const dom = new JSDOM(`<!doctype html><html><head></head><body>${html}</body></html>`, {
  runScripts: "dangerously", pretendToBeVisual: true, url: "http://127.0.0.1:8765/",
  beforeParse(win) {
    win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
    win.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
      get: () => () => ({ addColorStop() {} }), set: () => true,
    });
    win.requestAnimationFrame = () => 0;
    win.TextDecoder = TextDecoder;
    // Only the goal stream; the page also opens /api/models/watch.
    win.EventSource = class { constructor(u) { this.url = u; if (String(u).includes("/api/goals")) ES = this; } close() {} };
    win.fetch = (u, opts = {}) => {
      const url = String(u), method = opts.method || "GET";
      let body = { personas: [], tools: [], rag_tools: [], agents: [],
                   ollama: { up: false, models: [] }, running: null };
      if (url.includes("/api/goals")) {
        calls.push([method, url.replace("http://127.0.0.1:8765", ""),
                    opts.body ? JSON.parse(opts.body) : null]);
        body = SNAP;
      }
      return Promise.resolve({ ok: true, status: 200, body: null,
                               json: () => Promise.resolve(body) });
    };
  },
});

const problems = [];
const check = (ok, msg) => { if (!ok) problems.push(msg); };
const tick = (ms = 50) => new Promise(r => setTimeout(r, ms));
const doc = dom.window.document;
const $ = s => doc.querySelector(s);
const $$ = s => [...doc.querySelectorAll(s)];
const push = ev => ES.onmessage({ data: JSON.stringify(ev) });
const last = () => calls[calls.length - 1] || [];

(async () => {
  await tick(2500);        // boot animation, then loadState + gConnect
  check(ES && ES.url.endsWith("/api/goals/events"), "never subscribed to the goal stream");
  check(calls.some(c => c[0] === "GET" && c[1] === "/api/goals"), "never fetched the queue");

  // ── what it shows ─────────────────────────────────────────────────────────
  // A class rule like .g-pick{display:flex} beats the hidden attribute, so the
  // picker could never close. The page needs a global [hidden] rule with
  // !important. jsdom ignores !important in its cascade, so check the rule is there.
  const css = $$("style").map(s => s.textContent).join("\n").replace(/\s+/g, "");
  check(css.includes("[hidden]{display:none!important;}"),
        "no global [hidden]{display:none !important} rule; class display rules beat hidden");

  check($("#g-sum").textContent === "3 left · 1 done · running",
        `summary reads ${JSON.stringify($("#g-sum").textContent)}`);
  const now = $("#g-now");
  check(now.textContent.includes("running now"), "now-working card missing the running goal");
  check(now.textContent.includes("tests pass"), "now-working card missing done-when");
  const st = $$("#g-now .g-stg").map(s => [s.textContent, s.dataset.st]);
  check(JSON.stringify(st) === JSON.stringify([
    ["orchestrator · route", "done"], ["research_agent", "done"],
    ["dev_agent", "now"], ["tester_agent", "next"]]),
    `stage chips wrong: ${JSON.stringify(st)}`);

  const q = $$("#g-queue li");
  check(q.length === 2, `expected 2 queued, got ${q.length}`);
  check(q[0].querySelector('[data-s="next"]'), "head of the queue not marked next");
  check(q[0].querySelector('[data-gact="up"]').disabled, "first goal can move up");
  check(q[1].querySelector('[data-gact="down"]').disabled, "last goal can move down");
  check(!$("#g-queue img"), "goal text was rendered as markup");
  check(q[0].textContent.includes("<img"), "escaped goal text is missing");
  check(q[1].textContent.includes("dev_agent → tester_agent"), "picked agents not shown");

  const fin = $$("#g-fin li");
  check(fin.length === 2, `expected 2 finished, got ${fin.length}`);
  check(fin[0].textContent.includes("broke"), "finished list not newest first");
  check(fin[0].querySelector('[data-gact="retry"]'), "failed goal has no retry");
  check(!fin[1].querySelector('[data-gact="retry"]'), "done goal offers retry");

  // ── live work streams into the chat pane, never into the card ─────────────
  const chatText = () => $("#chat").textContent;
  check(!$("#g-now .g-live"), "the card still carries a live line");
  check([...doc.querySelectorAll("#chat .msg-user")].some(m => m.textContent.includes("running now")),
        "running goal has no header in the chat pane");
  check(chatText().includes("done when · tests pass"), "goal header missing done-when");
  const skipBtn = $('#g-now [data-gact="skip"]');
  push({ type: "stage", name: "dev_agent", model: "qwen3:8b", loaded: false, goal: "c1" });
  push({ type: "token", text: "writing the flag parser", goal: "c1" });
  push({ type: "token", text: "ignored", goal: "someone-else" });
  check(chatText().includes("writing the flag parser"), "goal tokens not streamed into chat");
  check([...doc.querySelectorAll("#chat .who")].some(w => w.textContent === "lag · dev_agent"),
        "streamed answer not labelled with its stage");
  check(!chatText().includes("ignored"), "another goal's tokens shown");
  check(skipBtn.isConnected, "a token rebuilt the card's buttons");
  check($("#g-now").textContent.includes("running now") && !$("#g-now").textContent.includes("flag parser"),
        "live tokens written into the card");

  // ── what it sends ────────────────────────────────────────────────────────
  skipBtn.click(); await tick();
  check(last()[1] === "/api/goals/skip", "skip did not post /api/goals/skip");

  $('#g-now [data-gact="remove"]').click(); await tick();
  check(last()[0] === "DELETE" && last()[1] === "/api/goals/c1", "stop & remove sent wrong request");

  $$("#g-queue li")[1].querySelector('[data-gact="up"]').click(); await tick();
  check(last()[1] === "/api/goals/p2/move?delta=-1", `move up sent ${last()[1]}`);

  $("#g-run").click(); await tick();
  check(last()[1] === "/api/goals/pause", "run button did not pause a running queue");

  $('[data-gfail="ask"]').click(); await tick();
  check(last()[1] === "/api/goals/settings" && last()[2].fail === "ask", "on-fail not sent");

  // add with two picked agents, in click order
  $("#g-agents").click();
  check(!$("#g-pick").hidden, "agent picker did not open");
  $('#g-pick [data-gpick="tester_agent"]').click();
  $('#g-pick [data-gpick="dev_agent"]').click();
  check($("#g-agents").textContent.includes("2 picked"), "picker count not shown");
  $("#g-text").value = "add a dry-run flag";
  $("#g-done").value = "a test covers it";
  $("#g-add").dispatchEvent(new dom.window.Event("submit", { cancelable: true }));
  await tick();
  const add = calls.find(c => c[0] === "POST" && c[1] === "/api/goals");
  check(add && add[2].text === "add a dry-run flag" && add[2].done_when === "a test covers it",
        "add sent the wrong text");
  check(add && JSON.stringify(add[2].agents) === '["tester_agent","dev_agent"]',
        `picked agents sent as ${add && JSON.stringify(add[2].agents)}`);
  check($("#g-text").value === "" && $("#g-pick").hidden, "form not reset after add");

  // view puts the answer in the chat, rendered
  $('#g-fin [data-gact="view"]').click(); await tick();
  check([...doc.querySelectorAll("#chat .md-h")].some(h => h.textContent === "Result"),
        "view did not render the answer into chat");

  // ── a failed goal waiting on you ─────────────────────────────────────────
  push({ ...SNAP, running: false, current: null, stage: "", awaiting: "f1",
         note: "a goal failed — retry it, or skip to carry on" });
  check(chatText().includes("goal stopped"), "chat pane did not close out the goal that left");
  check(!$("#chat .cursor"), "streaming cursor left behind after the goal ended");
  check($("#g-now").classList.contains("failed"), "awaiting goal not shown as failed");
  check($("#g-now").textContent.includes("RuntimeError: boom"), "failure reason missing");
  check($("#g-run").textContent === "run", "paused queue does not offer run");
  check(!$("#g-note").hidden && $("#g-note").textContent.includes("goal failed"), "note hidden");

  // ── idle ─────────────────────────────────────────────────────────────────
  push({ ...SNAP, running: true, current: null, stage: "", awaiting: null, note: "",
         goals: [] });
  check($("#g-now").innerHTML === "", "empty queue still shows a card");
  check($("#g-note").textContent.includes("idle"), "idle queue does not say so");
  check($("#g-queue").textContent.includes("nothing queued"), "empty queue has no hint");

  // fold persists the choice
  $("#g-fold").click();
  check($("#g-body").hidden, "fold did not hide the list");

  if (problems.length) {
    problems.forEach(p => console.log("FAIL " + p));
    process.exit(1);
  }
  console.log("goal panel ok · " + calls.length + " goal requests checked");
  process.exit(0);
})().catch(e => { console.log("FAIL harness error: " + (e.stack || e)); process.exit(1); });

dom.window.addEventListener("error", e => {
  console.log("FAIL page error: " + (e.error && e.error.stack || e.message));
  process.exit(1);
});
