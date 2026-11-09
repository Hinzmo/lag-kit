/* Feeds the console the strings an attacker would, and checks none becomes markup.
 *
 * A model's output is not trusted: a page fetched with http_fetch can carry a prompt
 * injection, and the model may repeat what it says. The console runs on the same
 * origin as /api/stream, which reaches run_shell, so a script that gets onto this page
 * can run commands on the machine. Two ways in were found and closed:
 *   - a markdown link whose address carried a quote closed href="…" and added onmouseover
 *   - a model-invented tool name went into the activity drawer as raw HTML
 *
 * Usage:  node tests/js/render_safety.js [path-to-console.html]
 * Prints FAIL lines and exits non-zero on any problem.
 */
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const file = process.argv[2] ||
  path.join(__dirname, "..", "..", "ui", "console.html");
const html = fs.readFileSync(file, "utf8");

const EVIL = `<img src=x onerror="window.PWNED=1">`;
const LINKS = [
  `[docs](https://x.io"onmouseover=location='javascript:alert%281%29'//)`,
  `[docs](https://x.io'onmouseover='alert%281%29)`,
  `[a](https://x.io"style="position:fixed)`,
  `[fine](https://example.com/x?a=1&b=2) and **bold**`,
];

// One chat turn, streamed the way /api/stream sends it, with attacker-chosen names
// and a model answer full of hostile links.
function sseBody(frames) {
  const enc = new TextEncoder();
  const chunks = frames.map(f => enc.encode("data: " + JSON.stringify(f) + "\n\n"));
  let i = 0;
  return { getReader: () => ({
    read: () => Promise.resolve(i < chunks.length
      ? { done: false, value: chunks[i++] } : { done: true, value: undefined }),
    releaseLock() {},
  }) };
}
const FRAMES = [
  { type: "start", stages: [{ name: "lag" }], loads: 1 },
  { type: "stage", name: EVIL, model: EVIL, loaded: true },
  { type: "tool", name: EVIL, args: {}, pretty: EVIL },
  { type: "tool_result", name: EVIL, content: "result" },
  { type: "routed", lead: EVIL, agents: [{ name: EVIL, model: EVIL }] },
  { type: "token", text: LINKS.join("\n\n") },
  { type: "done" },
];
const STATE = { chat_model: "m:1", num_ctx: 8192, personas: [], tools: [], rag_tools: [],
                agents: [], chroma: { available: false, chunks: 0 },
                ollama: { up: true, available: [], loaded: [], models: [] } };

const dom = new JSDOM(`<!doctype html><html><head></head><body>${html}</body></html>`, {
  runScripts: "dangerously", pretendToBeVisual: true, url: "http://127.0.0.1:8765/",
  beforeParse(win) {
    win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
    win.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
      get: () => () => ({ addColorStop() {} }), set: () => true,
    });
    win.requestAnimationFrame = () => 0;
    win.TextDecoder = TextDecoder;
    win.EventSource = undefined;
    win.fetch = (u) => {
      if (String(u).includes("/api/stream"))
        return Promise.resolve({ ok: true, status: 200, body: sseBody(FRAMES) });
      return Promise.resolve({ ok: true, status: 200, body: null,
                               json: () => Promise.resolve(STATE) });
    };
  },
});

const problems = [];
const win = dom.window, doc = win.document;

// Anything carrying an on* attribute, or an element model text should never produce.
function injected(root) {
  const bad = [];
  for (const n of root.querySelectorAll("*")) {
    for (const a of n.attributes) if (/^on/i.test(a.name)) bad.push(`${n.tagName}[${a.name}]`);
    if (/^(IMG|SCRIPT|IFRAME|OBJECT|EMBED)$/i.test(n.tagName)) bad.push(n.tagName);
  }
  return bad;
}
let ownScripts = 0;   // the page's own <script>; anything beyond it was injected

setTimeout(() => {
  ownScripts = doc.querySelectorAll("script").length;
  const input = doc.querySelector("#input"), exec = doc.querySelector("#exec");
  input.value = "render safety probe";
  exec.click();
  setTimeout(() => {
    const bad = injected(doc.body).filter(x => x !== "SCRIPT");
    if (doc.querySelectorAll("script").length > ownScripts) bad.push("SCRIPT");
    if (bad.length) problems.push(`model text became markup: ${bad.join(", ")}`);
    const links = [...doc.querySelectorAll("#chat a")];
    for (const a of links) {
      const names = [...a.attributes].map(x => x.name).sort().join(",");
      if (names !== "href,rel,target") problems.push(`a link gained attributes: ${names}`);
    }
    // Ordinary markdown must still render, or the fix broke the feature.
    if (!links.some(a => a.getAttribute("href") === "https://example.com/x?a=1&b=2"))
      problems.push("an ordinary link stopped rendering");
    if (!doc.querySelector("#chat strong")) problems.push("bold stopped rendering");
    // And the names are still shown, as text.
    if (!doc.body.textContent.includes('<img src=x onerror="window.PWNED=1">'))
      problems.push("hostile names were dropped instead of shown as text");
    if (win.PWNED) problems.push("an injected handler ran");

    if (problems.length) {
      console.error("FAIL " + problems.join("\nFAIL "));
      process.exit(1);
    }
    console.log("render safety: model text stayed text");
    process.exit(0);
  }, 800);
}, 2500);
