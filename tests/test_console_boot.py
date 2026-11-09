"""Does the console actually boot?

Every UI change in this project has been made without a browser, and the failure has
twice been the same shape: a reference outlives the thing it referenced, throws at
script level, and the page stops before it initialises. `node --check` cannot see it —
the syntax is valid. Only executing the page can.

The last occurrence deleted a hardcoded MODELS array and left one `MODELS.forEach`
behind. The server kept returning 200 with the full page, so it looked like a backend
or Docker problem; in fact the HTML arrived fine and the script died on line 963.

Skipped unless node and jsdom are present (`cd tests/js && npm install`). A skipped
test is honest about coverage; a missing one is not.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

JS_DIR = Path(__file__).parent / "js"
HARNESS = JS_DIR / "console_boot.js"
FORM_HARNESS = JS_DIR / "agent_form.js"
GOALS_HARNESS = JS_DIR / "goals_panel.js"
CONSOLE = Path(__file__).parent.parent / "ui" / "console.html"


def _node() -> str | None:
    return shutil.which("node")


def _jsdom_available() -> bool:
    node = _node()
    if not node or not HARNESS.exists():
        return False
    env = dict(os.environ)
    extra = os.environ.get("LAG_NODE_PATH") or str(JS_DIR / "node_modules")
    env["NODE_PATH"] = extra + os.pathsep + env.get("NODE_PATH", "")
    r = subprocess.run([node, "-e", "require.resolve('jsdom')"],
                       capture_output=True, env=env, cwd=str(JS_DIR))
    return r.returncode == 0


def _run(page: Path, harness: Path = HARNESS) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    extra = os.environ.get("LAG_NODE_PATH") or str(JS_DIR / "node_modules")
    env["NODE_PATH"] = extra + os.pathsep + env.get("NODE_PATH", "")
    return subprocess.run([_node(), str(harness), str(page)],
                          capture_output=True, text=True, timeout=120, env=env)


needs_jsdom = pytest.mark.skipif(
    not _jsdom_available(),
    reason="needs node + jsdom — run `cd tests/js && npm install`")


@needs_jsdom
def test_the_console_boots_without_throwing():
    r = _run(CONSOLE)
    assert r.returncode == 0, (
        "ui/console.html failed to boot:\n" + (r.stderr or r.stdout))


@needs_jsdom
def test_the_harness_catches_a_dead_reference(tmp_path):
    """The check is only worth having if it fails on the bug that shipped."""
    broken = tmp_path / "console.html"
    broken.write_text(CONSOLE.read_text().replace(
        'const stageflow = $("#stageflow");',
        'const stageflow = $("#stageflow");\nDELETED_ARRAY.forEach(x => x);'))
    r = _run(broken)
    assert r.returncode != 0, "a dead reference must fail the boot check"
    assert "DELETED_ARRAY" in (r.stderr + r.stdout)


@needs_jsdom
def test_the_harness_catches_a_broken_send_path(tmp_path):
    """Booting clean is not enough — the bug that shipped was in the send path. Deleting
    clearEmpty's definition (exactly what happened) must now fail the check, because
    execute() throws on it before /api/stream is ever requested."""
    broken = tmp_path / "console.html"
    broken.write_text(CONSOLE.read_text().replace(
        'function clearEmpty(){ const e = chat.querySelector(".empty"); if(e) e.remove(); }',
        '/* clearEmpty deleted — the regression that broke chat */'))
    r = _run(broken)
    assert r.returncode != 0, "a broken send path must fail the boot check"
    assert "/api/stream" in (r.stderr + r.stdout) or "clearEmpty" in (r.stderr + r.stdout)


@needs_jsdom
def test_the_harness_catches_a_page_that_never_finishes_booting():
    """Throwing is not the only way to fail — a boot that stalls looks identical."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        broken = Path(d) / "console.html"
        broken.write_text(CONSOLE.read_text().replace("  loadState().then(watchModels);", "  return;"))
        r = _run(broken)
        assert r.returncode != 0
        assert "never requested" in (r.stderr + r.stdout) or "never cleared" in (r.stderr + r.stdout)


@needs_jsdom
def test_the_agent_form_saves_what_it_shows():
    """0 cycles stays 0, a sealed agent stays sealed, a refused save says why."""
    r = _run(CONSOLE, FORM_HARNESS)
    assert r.returncode == 0, "agent form:\n" + (r.stderr or r.stdout)


@needs_jsdom
def test_the_form_harness_catches_the_null_cap(tmp_path):
    """Put the shipped bug back (`|| null`) and the harness must fail on it."""
    broken = tmp_path / "console.html"
    src = CONSOLE.read_text()
    fixed = 'max_cycles: cycles === "" ? undefined : Math.max(0, parseInt(cycles, 10) || 0),'
    assert fixed in src
    broken.write_text(src.replace(fixed, 'max_cycles: parseInt(cycles || "0", 10) || null,'))
    r = _run(broken, FORM_HARNESS)
    assert r.returncode != 0 and "sent as null" in (r.stderr + r.stdout)


@needs_jsdom
def test_the_goal_panel_renders_and_sends_the_right_requests():
    r = _run(CONSOLE, GOALS_HARNESS)
    assert r.returncode == 0, "goal panel:\n" + (r.stderr or r.stdout)


@needs_jsdom
def test_the_goal_harness_catches_a_token_rebuilding_the_card(tmp_path):
    """Live events stream into the chat pane and must never re-render the card.
    Re-rendering it per token rebuilds its buttons, so a click that lands mid-stream
    hits a detached node."""
    broken = tmp_path / "console.html"
    src = CONSOLE.read_text()
    fixed = "  GS.s.event(e);\n}"
    assert fixed in src
    broken.write_text(src.replace(fixed, "  GS.s.event(e);\n  gRender();\n}"))
    r = _run(broken, GOALS_HARNESS)
    assert r.returncode != 0 and "rebuilt the card" in (r.stderr + r.stdout)


@needs_jsdom
def test_the_goal_harness_catches_live_work_left_out_of_chat(tmp_path):
    """A running goal's work belongs in the chat pane, where a chat turn's does."""
    broken = tmp_path / "console.html"
    src = CONSOLE.read_text()
    fixed = "  GS.s.event(e);\n}"
    broken.write_text(src.replace(fixed, "}"))
    r = _run(broken, GOALS_HARNESS)
    assert r.returncode != 0 and "not streamed into chat" in (r.stderr + r.stdout)


MODELS_HARNESS = JS_DIR / "models_watch.js"


@needs_jsdom
def test_the_model_list_stays_live_without_a_reload():
    r = _run(CONSOLE, MODELS_HARNESS)
    assert r.returncode == 0, "model watch:\n" + (r.stderr or r.stdout)


@needs_jsdom
def test_the_model_harness_catches_a_watch_left_open_in_a_hidden_tab(tmp_path):
    """A hidden tab must drop the stream, or the server keeps checking for nobody."""
    broken = tmp_path / "console.html"
    src = CONSOLE.read_text()
    fixed = "document.hidden ? unwatchModels() : watchModels()"
    assert fixed in src
    broken.write_text(src.replace(fixed, "watchModels()"))
    r = _run(broken, MODELS_HARNESS)
    assert r.returncode != 0 and "hidden tab kept" in (r.stderr + r.stdout)


SAFETY_HARNESS = JS_DIR / "render_safety.js"


@needs_jsdom
def test_model_text_never_becomes_markup():
    """Hostile links, tool names and stage names stream through a chat turn; none may
    reach the page as markup. The console shares an origin with /api/stream."""
    r = _run(CONSOLE, SAFETY_HARNESS)
    assert r.returncode == 0, "render safety:\n" + (r.stderr or r.stdout)


@needs_jsdom
def test_the_safety_harness_catches_the_href_quote(tmp_path):
    """Put the shipped bug back (quotes not escaped in mdRender) and it must fail."""
    broken = tmp_path / "console.html"
    src = CONSOLE.read_text()
    fixed = """  const esc0 = s => s.replace(/[&<>"']/g,"""
    assert fixed in src
    broken.write_text(src.replace(fixed, """  const esc0 = s => s.replace(/[&<>]/g,"""))
    r = _run(broken, SAFETY_HARNESS)
    assert r.returncode != 0 and "a link gained attributes" in (r.stderr + r.stdout)


@needs_jsdom
def test_the_safety_harness_catches_a_raw_tool_name(tmp_path):
    """Put the shipped bug back (the drawer's tag unescaped) and it must fail."""
    broken = tmp_path / "console.html"
    src = CONSOLE.read_text()
    fixed = '<span class="vb-tag">${esc(tag)}</span>'
    assert fixed in src
    broken.write_text(src.replace(fixed, '<span class="vb-tag">${tag}</span>'))
    r = _run(broken, SAFETY_HARNESS)
    assert r.returncode != 0 and "IMG" in (r.stderr + r.stdout)
