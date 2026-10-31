/* Boots ui/console.html in jsdom and reports anything that throws.
 *
 * Every UI change in this project has been shipped without a browser, and the failure
 * mode is always the same: a reference survives the thing it referenced, throws at
 * script level, and stops the whole page before it initialises. `node --check` cannot
 * see that — it is valid syntax. This actually executes it.
 *
 * Usage:  node tests/js/console_boot.js [path-to-console.html]
 * Exits non-zero, and prints the stack, if the page fails to boot.
 */
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const file = process.argv[2] ||
  path.join(__dirname, "..", "..", "ui", "console.html");
const html = fs.readFileSync(file, "utf8");

const errors = [], fetched = [];

// A synchronous throw inside an event listener (e.g. the send handler) surfaces as an
// uncaught exception at the node level rather than through window.onerror. Report it in
// our own format and fail, instead of letting node dump a raw stack and exit.
process.on("uncaughtException", e => {
  console.error("CONSOLE FAILED TO BOOT:\nuncaught: " + (e && e.stack || e));
  process.exit(1);
});

// Enough of /api/state to exercise every render path the boot touches.
const STATE = {
  chat_model: "ornith:latest", num_ctx: 32768,
  ollama: {
    up: true, available: ["ornith:latest", "qwen3:8b"],
    loaded: [{ name: "qwen3:8b", size_gb: 6.2 }],
    models: [
      { name: "ornith:latest", tools: true, thinking: true, embedding: false, capabilities: [] },
      { name: "qwen3:8b", tools: true, thinking: true, embedding: false, capabilities: [] },
      { name: "chatty:7b", tools: false, thinking: false, embedding: false, capabilities: [] },
      { name: "nomic-embed-text:latest", tools: false, thinking: false, embedding: true, capabilities: [] },
    ],
  },
  personas: [{ name: "dev_agent", role: "writes code", model: "qwen3:8b", tools: ["read_file"] }],
  tools: ["read_file", "write_file"], rag_tools: ["search_knowledge_base"],
  chroma: { available: true, chunks: 11 },
  agents: [{ name: "nightly", goal: "g", description: "d", enabled: true, tools: [],
             running: false, unknown_tools: [], state: null }],
};

const dom = new JSDOM(`<!doctype html><html><head></head><body>${html}</body></html>`, {
  runScripts: "dangerously", pretendToBeVisual: true, url: "http://127.0.0.1:8765/",
  beforeParse(win) {
    // jsdom gaps that a real browser does not have — shimmed so they are not
    // mistaken for page bugs.
    win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
    win.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
      get: () => () => ({ addColorStop() {} }), set: () => true,
    });
    win.requestAnimationFrame = () => 0;
    win.TextDecoder = TextDecoder;   // page-context decode of the fake stream below

    // A fake SSE stream for /api/stream, so a send is driven end to end: start →
    // stage → token → done. That is the path that calls Tps.start/bump/stop and
    // loadModel — all of which a boot-only check never reaches, and all of which
    // have gone missing in a cleanup before. A unique marker in the token lets the
    // assertion prove the answer actually rendered (the reader loop swallows a
    // throw from onEvent per-frame, so a missing helper fails silently otherwise).
    const sseBody = (frames) => {
      const enc = new TextEncoder();
      const chunks = frames.map(f => enc.encode("data: " + JSON.stringify(f) + "\n\n"));
      let i = 0;
      return { getReader: () => ({
        read: () => Promise.resolve(i < chunks.length
          ? { done: false, value: chunks[i++] } : { done: true, value: undefined }),
        releaseLock() {},
      }) };
    };
    win.fetch = (u) => {
      fetched.push(String(u));
      if (String(u).includes("/api/stream"))
        return Promise.resolve({ ok: true, status: 200, body: sseBody([
          { type: "start", stages: [{ name: "lag" }], loads: 1 },
          { type: "stage", name: "lag", model: "ornith:latest", loaded: true },
          { type: "token", text: "STREAMOK_" },
          { type: "token", text: "42" },
          { type: "done" },
        ]) });
      return Promise.resolve({ ok: true, status: 200, body: null,
                               json: () => Promise.resolve(STATE) });
    };
    win.onerror = (m, s, l, c, err) => errors.push(String(err && err.stack || m));
    win.addEventListener("unhandledrejection",
      e => errors.push("unhandled rejection: " + (e.reason && e.reason.stack || e.reason)));
  },
});

setTimeout(() => {
  const doc = dom.window.document;
  const problems = [...errors];

  // The page is only "booted" once the overlay is gone and state has been fetched.
  if (doc.querySelector("#boot")) problems.push("boot overlay never cleared");
  if (!fetched.some(u => u.includes("/api/state")))
    problems.push("/api/state was never requested — boot stopped early");

  // Booting clean is not enough: the bugs that shipped lived in the SEND path, where a
  // reference (clearEmpty, say, thinkingLine) outlived its definition and threw inside
  // the click handler — invisible to a boot-only check, because nothing throws until
  // someone sends. So drive an actual turn and prove the request leaves the page.
  const input = doc.querySelector("#input"), exec = doc.querySelector("#exec");
  if (!input || !exec) {
    problems.push("the chat input or send button is missing");
  } else {
    const before = errors.length;
    input.value = "boot probe — does send reach the server?";
    try { exec.click(); } catch (e) { errors.push("send threw: " + (e.stack || e)); }
    // The stub returns no stream body, so realTurn's own try/catch renders a fault and
    // does NOT rethrow — a clean page therefore adds no uncaught error here. What it
    // MUST do is fire the request; the broken page threw before reaching it.
    setTimeout(() => {
      if (!fetched.some(u => u.includes("/api/stream")))
        problems.push("send did nothing — /api/stream was never requested");
      // The stream carried a token marked STREAMOK_42. If it rendered, the whole
      // turn path ran — Tps.start/bump/stop and loadModel included. If it did not,
      // a helper on that path threw (swallowed per-frame) and the answer is empty.
      const chatText = (doc.querySelector("#chat") || {}).textContent || "";
      if (!chatText.includes("STREAMOK_42"))
        problems.push("the streamed answer never rendered — a turn-path helper "
          + "(Tps/loadModel/say) threw; chat was: " + JSON.stringify(chatText.slice(0, 200)));
      for (const e of errors.slice(before)) problems.push("during send: " + e);
      finish(problems);
    }, 600);
    return;
  }
  finish(problems);
}, 3000);

function finish(problems) {
  if (problems.length) {
    console.error("CONSOLE FAILED TO BOOT:\n" + problems.join("\n---\n"));
    process.exit(1);
  }
  console.log("console booted and sent cleanly · fetched: " + fetched.join(", "));
  process.exit(0);
}
