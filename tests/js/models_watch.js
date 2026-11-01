/* Drives the console's live model list in jsdom.
 *
 * The server pushes the `ollama` block of /api/state whenever the installed or
 * resident models change. Checked here: a pushed list reaches both the picker and
 * the models panel without a reload, the user's picked override survives a push,
 * and a hidden tab closes the stream (which is what stops the server-side checks).
 *
 * Usage:  node tests/js/models_watch.js [path-to-console.html]
 */
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const file = process.argv[2] || path.join(__dirname, "..", "..", "ui", "console.html");
const html = fs.readFileSync(file, "utf8");

const m = (name, extra = {}) => ({ name, tools: true, thinking: false, embedding: false,
                                   capabilities: [], ...extra });
const OLLAMA = { up: true, available: ["qwen3:8b"], loaded: [], models: [m("qwen3:8b")] };
const STATE = { chat_model: "qwen3:8b", num_ctx: 8192, ollama: OLLAMA, personas: [],
                tools: [], rag_tools: [], agents: [], chroma: { available: false, chunks: 0 } };

const sources = [];        // every EventSource the page opened
const errors = [];
let hidden = false;

const dom = new JSDOM(`<!doctype html><html><head></head><body>${html}</body></html>`, {
  runScripts: "dangerously", pretendToBeVisual: true, url: "http://127.0.0.1:8765/",
  beforeParse(win) {
    win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
    win.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
      get: () => () => ({ addColorStop() {} }), set: () => true,
    });
    win.requestAnimationFrame = () => 0;
    Object.defineProperty(win.document, "hidden", { get: () => hidden });
    win.EventSource = class {
      constructor(u) { this.url = u; this.closed = false; sources.push(this); }
      close() { this.closed = true; }
    };
    win.fetch = (u) => Promise.resolve({
      ok: true, status: 200, body: null,
      json: () => Promise.resolve(String(u).includes("/api/goals")
        ? { type: "goals", goals: [], settings: {}, running: false } : STATE),
    });
    win.onerror = (msg, s, l, c, err) => errors.push(String(err && err.stack || msg));
  },
});

const watches = () => sources.filter(s => s.url.includes("/api/models/watch"));
const push = (o) => { const w = watches().find(s => !s.closed);
                      w.onmessage({ data: JSON.stringify(o) }); };

setTimeout(() => {
  const doc = dom.window.document, win = dom.window, problems = [...errors];
  const pick = doc.querySelector("#m-pick");
  const panelText = () => doc.querySelector("#models").textContent;
  const options = () => [...pick.querySelectorAll("option")].map(o => o.value);

  if (watches().length !== 1) problems.push(`expected one model watch, got ${watches().length}`);
  else {
    // Pick a model, then a push arrives with a newly pulled one.
    pick.value = "qwen3:8b"; pick.dispatchEvent(new win.Event("change"));
    push({ ...OLLAMA, available: ["qwen3:8b", "fresh:14b"],
           loaded: [{ name: "fresh:14b", size_gb: 9.1 }],
           models: [m("qwen3:8b"), m("fresh:14b", { thinking: true })] });
    if (!options().includes("fresh:14b")) problems.push("a pushed model never reached the picker");
    if (!panelText().includes("fresh:14b")) problems.push("a pushed model never reached the models panel");
    if (!panelText().includes("9.10 GB")) problems.push("the resident model was not shown as loaded");
    if (pick.value !== "qwen3:8b") problems.push(`the picked override was lost on a push (now ${JSON.stringify(pick.value)})`);

    // An rm: the model disappears again.
    push(OLLAMA);
    if (options().includes("fresh:14b")) problems.push("a removed model stayed in the picker");

    // Hidden tab closes the stream; visible again opens a fresh one.
    hidden = true; doc.dispatchEvent(new win.Event("visibilitychange"));
    if (!watches()[0].closed) problems.push("a hidden tab kept the model watch open");
    hidden = false; doc.dispatchEvent(new win.Event("visibilitychange"));
    if (watches().filter(s => !s.closed).length !== 1)
      problems.push("coming back to the tab did not reopen the model watch");
  }
  problems.push(...errors.filter(e => !problems.includes(e)));
  if (problems.length) { console.error("MODEL WATCH FAILED:\n" + problems.join("\n")); process.exit(1); }
  console.log("model watch ok"); process.exit(0);
}, 3000);
