/* Drives the console's agent form in jsdom and checks what it sends on save.
 *
 * The form is the only way most people edit a saved agent, and its bugs are silent:
 * a 0 in "max cycles" became null and the server refused the save, which the page
 * reported as "saved as undefined"; and with no network field, the form could not
 * say what the agent's network setting was, so re-saving did not preserve it.
 *
 * Usage:  node tests/js/agent_form.js [path-to-console.html]
 * Prints FAIL lines and exits non-zero on any problem.
 */
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const file = process.argv[2] ||
  path.join(__dirname, "..", "..", "ui", "console.html");
const html = fs.readFileSync(file, "utf8");

const posts = [];          // bodies POSTed to /api/agents
let saveReply = { status: 200, body: { saved: "x", unknown_tools: [] } };
let starts = 0;            // POSTs to /api/agent/start
let RUNNING = null;        // what /api/agent/running reports (another tab, the CLI)
let startReply = "stream"; // "stream": an open run that ends on endRun(); or "busy"
let endRun = () => {};

const dom = new JSDOM(`<!doctype html><html><head></head><body>${html}</body></html>`, {
  runScripts: "dangerously", pretendToBeVisual: true, url: "http://127.0.0.1:8765/",
  beforeParse(win) {
    win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
    win.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
      get: () => () => ({ addColorStop() {} }), set: () => true,
    });
    win.requestAnimationFrame = () => 0;
    win.TextDecoder = TextDecoder;
    win.fetch = (u, opts = {}) => {
      if (String(u).includes("/api/agent/running"))
        return Promise.resolve({ ok: true, status: 200,
                                 json: () => Promise.resolve({ running: RUNNING }) });
      if (String(u).includes("/api/agent/start")) {
        starts++;
        if (startReply === "busy")
          return Promise.resolve({ ok: false, status: 409, body: null, json: () =>
            Promise.resolve({ error: "agent 'nightly' is already running — one agent "
                                     + "at a time; stop it first", running: "nightly" }) });
        const done = new Promise(r => { endRun = () => r({ done: true }); });
        return Promise.resolve({ ok: true, status: 200,
          body: { getReader: () => ({ read: () => done, releaseLock() {} }) } });
      }
      if (String(u).endsWith("/api/agents") && opts.method === "POST") {
        posts.push(JSON.parse(opts.body));
        const { status, body } = saveReply;
        return Promise.resolve({ ok: status < 400, status,
                                 json: () => Promise.resolve(body) });
      }
      const body = String(u).includes("/api/agents") ? { agents: SAVED }
        : { personas: [{ name: "dev_agent", role: "r", model: "m", tools: [] }],
            tools: [], rag_tools: [], agents: [], ollama: { up: false, models: [] } };
      return Promise.resolve({ ok: true, status: 200, body: null,
                               json: () => Promise.resolve(body) });
    };
  },
});

const problems = [];
const check = (ok, msg) => { if (!ok) problems.push(msg); };
const tick = (ms = 50) => new Promise(r => setTimeout(r, ms));

// The page script is one IIFE, so nothing is reachable from window — drive it the
// way a person does: open the wizard, press "edit" on a saved agent, press "save".
const SAVED = [{ name: "sealed", goal: "g", description: "d", enabled: true, tools: [],
                 max_cycles: 5, network: false, running: false, unknown_tools: [],
                 state: null }];

(async () => {
  await tick(1500);                         // let the page boot
  const w = dom.window, $ = s => w.document.querySelector(s);
  const save = async () => { $("#aw-save").click(); await tick(); };

  $("#agent-open").click();
  await tick(200);                          // refreshSaved fills the list

  // 1. The input's own default (0 = until done) must reach the server as 0, not null.
  $("#aw-name").value = "a"; $("#aw-goal").value = "g"; $("#aw-cycles").value = "0";
  await save();
  check(posts.at(-1) && posts.at(-1).max_cycles === 0,
        `max_cycles 0 was sent as ${JSON.stringify(posts.at(-1) && posts.at(-1).max_cycles)}`);

  // 2. A blank field means "not set": send nothing, so the server keeps or defaults it.
  $("#aw-cycles").value = "";
  await save();
  check(posts.at(-1) && (!("max_cycles" in posts.at(-1)) ||
                         Number.isInteger(posts.at(-1).max_cycles)),
        `blank max_cycles was sent as ${JSON.stringify(posts.at(-1) && posts.at(-1).max_cycles)}`);

  // 3. Editing a sealed agent must show it sealed and save it sealed.
  const edit = [...w.document.querySelectorAll("#aw-saved button")]
    .find(b => b.textContent === "edit");
  check(edit, "the saved agent has no edit button");
  if (edit) {
    edit.click();
    check($("#aw-network") && $("#aw-network").checked === false,
          "the form has no network checkbox, or did not load network: false into it");
    await save();
    check(posts.at(-1) && posts.at(-1).network === false,
          `a sealed agent was re-saved with network ${JSON.stringify(posts.at(-1) && posts.at(-1).network)}`);
  }

  // 4. A refused save must say why, never "saved as undefined".
  saveReply = { status: 422, body: { detail: [{ loc: ["body", "max_cycles"],
                                               msg: "Input should be a valid integer" }] } };
  await save();
  const hint = $("#aw-hint").textContent;
  check(!/undefined/.test(hint) && /max_cycles/.test(hint),
        `a refused save showed: ${JSON.stringify(hint)}`);

  // 5. Cmd+Enter saves; it must not ALSO start an agent (both handlers matched it).
  saveReply = { status: 200, body: { saved: "x", unknown_tools: [] } };
  const key = (target, init) => target.dispatchEvent(
    new w.KeyboardEvent("keydown", { key: "Enter", bubbles: true, ...init }));
  const savesBefore = posts.length;
  key($("#aw-goal"), { metaKey: true });
  await tick();
  check(posts.length === savesBefore + 1, "Cmd+Enter did not save");
  check(starts === 0, `Cmd+Enter also started an agent (${starts} start request(s))`);

  // 6. Enter on a focused button presses THAT button; it must not start an agent too.
  key($("#aw-save"));
  await tick();
  check(starts === 0, `Enter on the save button started an agent (${starts})`);

  // 7. Two fast clicks on "start agent" start ONE agent, and every start control greys out.
  $("#aw-name").value = ""; $("#aw-goal").value = "g";
  $("#aw-start").click(); $("#aw-start").click();
  await tick();
  check(starts === 1, `two clicks sent ${starts} start requests`);
  check($("#agent-open").disabled, "agent mode stayed clickable while an agent ran");
  check($("#aw-start").disabled, "start agent stayed enabled while an agent ran");
  const runs = () => [...w.document.querySelectorAll("#aw-saved button")]
    .filter(b => b.textContent === "run");
  check(runs().length && runs().every(b => b.disabled),
        "a saved agent's run button stayed enabled while an agent ran");

  // 8. When the run ends, everything comes back.
  endRun();
  await tick(150);
  check(!$("#agent-open").disabled && !$("#aw-start").disabled,
        "start controls stayed grey after the run ended");

  // 9. A run started elsewhere (another tab, the CLI) greys this page out too, and
  //    the page notices when it ends.
  RUNNING = "nightly";
  $("#agent-open").click();
  await tick(150);
  check($("#agent-open").disabled && /nightly/.test($("#agent-open").textContent),
        `an agent running elsewhere was not shown: ${JSON.stringify($("#agent-open").textContent)}`);
  check(!$("#agentwiz").classList.contains("on"), "the wizard opened while another agent ran");
  RUNNING = null;
  await tick(2600);                         // the page polls for the other run's end
  check(!$("#agent-open").disabled, "the page never noticed the other run ended");

  // 10. If the server refuses anyway (a CLI start won the race), say why.
  startReply = "busy";
  $("#agent-open").click(); await tick(150);
  $("#aw-goal").value = "g"; $("#aw-start").click();
  await tick(150);
  check(/already running/.test(w.document.querySelector("#chat").textContent),
        "a 409 from the server was not shown");
  check($("#agent-open").disabled, "after a 409 the page did not grey out for the other run");
  RUNNING = null;

  finish();
})().catch(e => { problems.push("threw: " + (e.stack || e)); finish(); });

function finish() {
  if (problems.length) {
    console.error(problems.map(p => "FAIL " + p).join("\n"));
    process.exit(1);
  }
  console.log("agent form OK");
  process.exit(0);
}
