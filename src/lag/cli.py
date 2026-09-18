"""`lag` command-line interface."""
from __future__ import annotations
import typer

from .config import load_config
from .ollama_client import OllamaClient
from .tools import LocalToolProvider

app = typer.Typer(help="local-agent-kit \u2014 offline multi-agent harness over Ollama.")


@app.command()
def run(
    prompt: str,
    model: str = typer.Option(None, help="Override the configured model."),
    mcp: bool = typer.Option(False, "--mcp", help="Route tools through the MCP server instead of direct dispatch."),
):
    """Run a single agent on a prompt."""
    cfg = load_config()
    model = model or cfg["models"]["worker"]

    if mcp:
        from .mcp_client import MCPToolProvider
        provider = MCPToolProvider.from_config(cfg)
    else:
        provider = LocalToolProvider()

    client = OllamaClient.from_config(cfg, model)
    print(client.run(prompt, tool_provider=provider))


@app.command()
def pipeline(
    task: str,
    agents: str = typer.Option(None, help="Comma-separated persona names, in order (the first plans unless --no-plan)."),
    plan: bool = typer.Option(True, help="Let the first stage write the downstream instructions before any work starts."),
    route: bool = typer.Option(True, help="Let the orchestrator pick which agents the task needs, instead of running --agents in order."),
    mcp: bool = typer.Option(False, "--mcp", help="Give every stage the MCP toolset (fs, shell, web, memory, and search_knowledge_base) instead of only in-process tools."),
):
    """Run a project pipeline: staged agents, each building on the context above it.

    Stages run in the order given — order is meaning, and it is also the cost model,
    since consecutive stages sharing a model tag reuse the resident weights while a
    change of tag costs a full cold reload (~5s on this box).
    """
    from .pipeline import Pipeline
    cfg = load_config()

    # With --mcp, all stages share ONE provider backed by both MCP servers, so the lead
    # gains search_knowledge_base and can pull back facts it stored with remember — the
    # memory-offload loop that makes a small num_ctx workable. Without it, from_personas
    # falls back to in-process tools (no semantic KB search).
    provider = None
    if mcp:
        from .mcp_client import MCPToolProvider
        provider = MCPToolProvider.from_config(cfg)

    if route and not agents:
        typer.echo("[pipeline] routing — the orchestrator is choosing the team …")
        pipe = Pipeline.routed(task, cfg, provider)
        typer.echo(f"[pipeline] chose: {', '.join(a.name for a in pipe.stages) or '(none)'}")
    else:
        agents = agents or cfg.get("pipeline", {}).get(
            "default_agents", "architect_agent,dev_agent,senior_dev_agent"
        )
        pipe = Pipeline.from_personas([a.strip() for a in agents.split(",")], cfg, provider)
    typer.echo(f"[pipeline] {len(pipe.stages)} stage(s) · {pipe.loads()} model load(s)")
    print(pipe.run(task, plan=plan))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8765, help="Port."),
):
    """Launch the local web console (the phosphor UI) over the harness.

    Serves ui/console.html plus a small JSON API that wraps the same library the
    CLI uses. Everything binds to localhost — no cloud, no keys. Run from the repo
    root so the RAG server finds ./.chroma (the same caveat as `--mcp`).
    """
    import uvicorn

    from .web import app as web_app

    typer.echo(f"lag console → http://{host}:{port}   (Ctrl-C to stop)")
    uvicorn.run(web_app, host=host, port=port, log_level="info")


agent_app = typer.Typer(help="Agent mode: one persona pursuing a goal on its own, 24/7 if you like.")
app.add_typer(agent_app, name="agent")


def _ask(label: str, default: str = "") -> str:
    """One wizard field. Enter keeps the default; the goal is the only required one."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise typer.Abort() from None
    return val or default


@agent_app.command("start")
def agent_start(
    saved: str = typer.Argument(None, help="Name of a saved agent to run. Omit to define one."),
    goal: str = typer.Option(None, help="What the agent should achieve. Omit for the wizard."),
    success: str = typer.Option("", help="What done looks like."),
    constraints: str = typer.Option("", help="What it must not do."),
    persona: str = typer.Option("dev_agent", help="Which persona drives the loop."),
    interval: int = typer.Option(0, help="Seconds to pause between cycles (0 = flat out)."),
    max_cycles: int = typer.Option(0, help="Stop after this many cycles (0 = until done)."),
    name: str = typer.Option("default", help="Run name; also the state and stop file name."),
    max_failures: int = typer.Option(5, help="Stop after this many CONSECUTIVE failed cycles (0 = never)."),
    max_idle: int = typer.Option(5, help="Stop after this many consecutive cycles that call no tool and note no progress (0 = never)."),
    mcp: bool = typer.Option(False, "--mcp", help="Use the MCP toolset instead of in-process tools."),
):
    """Set a goal and let the agent work it until done, capped, or stopped.

    With no --goal this walks the wizard. Agent mode runs with FULL autonomy — the
    same unsandboxed shell and filesystem the rest of the CLI has — so read the goal
    back before confirming, and remember `lag agent stop` exists.
    """
    from .agent import AgentLoop, GoalSpec, stop_path

    cfg = load_config()

    # A saved agent is run as-is: its definition already holds every answer the wizard
    # would ask for, and re-asking would quietly let the saved copy drift.
    if saved:
        try:
            spec = GoalSpec.load(saved)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"  {e}")
            raise typer.Exit(1) from None
        if not spec.enabled:
            typer.echo(f"  agent {saved!r} is disabled — enable it with: "
                       f"lag agent enable {saved}")
            raise typer.Exit(1)
        return _run_spec(spec, cfg, mcp=spec.mcp)

    if not goal:
        typer.echo("\n  agent mode — set the goal\n")
        goal = _ask("goal (required)")
        if not goal:
            typer.echo("  no goal given; nothing to do.")
            raise typer.Exit(1)
        success = _ask("done when", success)
        constraints = _ask("must not", constraints)
        persona = _ask("persona", persona)
        interval = int(_ask("seconds between cycles", str(interval)) or 0)
        max_cycles = int(_ask("max cycles (0 = until done)", str(max_cycles)) or 0)
        name = _ask("run name", name)

    spec = GoalSpec(goal=goal, success=success, constraints=constraints,
                    persona=persona, interval=interval,
                    max_cycles=max_cycles or None, name=name,
                    max_failures=max_failures, max_idle=max_idle, mcp=mcp)
    return _run_spec(spec, cfg, mcp=mcp)


def _run_spec(spec, cfg, mcp: bool):
    """Print the brief, then drive the loop. Shared by ad-hoc and saved runs."""
    from .agent import AgentLoop, stop_path

    typer.echo(f"\n  goal      : {spec.goal}")
    typer.echo(f"  done when : {spec.success or '(agent decides)'}")
    typer.echo(f"  must not  : {spec.constraints or '(none)'}")
    typer.echo(f"  persona   : {spec.persona}")
    typer.echo(f"  cadence   : {'flat out' if not spec.interval else str(spec.interval) + 's between cycles'}")
    typer.echo(f"  cap       : {spec.max_cycles or 'until done or stopped'}")
    typer.echo(f"  breakers  : {spec.max_failures or 'off'} consecutive failures · "
               f"{spec.max_idle or 'off'} idle cycles")
    typer.echo(f"  tools     : {', '.join(spec.tools) if spec.tools else 'all (unrestricted)'}")
    typer.echo(f"  autonomy  : FULL — unsandboxed shell and filesystem")
    typer.echo(f"  stop with : lag agent stop {spec.name}   (or touch {stop_path(spec.name)})\n")

    unknown = spec.unknown_tools()
    if unknown:
        # A typo here silently removes a tool, which looks like a dumb model later.
        typer.echo(f"  ! unknown tools in this agent: {', '.join(unknown)}")

    provider = None
    if mcp:
        from .mcp_client import MCPToolProvider
        provider = MCPToolProvider.from_config(cfg)

    def show(ev):
        t = ev.get("type")
        if t == "cycle":
            typer.echo(f"\n[agent] cycle {ev['n']} · {ev['model']}")
        elif t == "tool":
            typer.echo(f"  ↳ {ev['name']}({str(ev.get('args'))[:80]})")
        elif t == "cycle_error":
            typer.echo(f"  ! cycle {ev['n']} failed: {ev['message']}")
        elif t == "idle":
            typer.echo(f"[agent] idle {ev['seconds']}s")
        elif t == "agent_done":
            typer.echo(f"\n[agent] DONE after {ev['cycle']} cycle(s): {ev['summary']}")
        elif t == "agent_stop":
            typer.echo(f"\n[agent] stopped ({ev['reason']}) at cycle {ev['cycle']}"
                       + (f" — {ev['detail']}" if ev.get("detail") else ""))

    state = AgentLoop(spec, cfg, provider).run(on_event=show)
    typer.echo(f"\n[agent] {state.cycle} cycle(s) · {len(state.notes)} note(s) · "
               f"{state.errors} error(s) · state: {state.path}")


@agent_app.command("new")
def agent_new(
    name: str = typer.Argument(..., help="Name for the saved agent."),
    tools: str = typer.Option("", help="Comma-separated tool allow-list. Empty = every tool."),
):
    """Define a saved agent you can run, disable, and edit later."""
    from .agent import GoalSpec

    typer.echo(f"\n  defining agent {name!r}\n")
    goal = _ask("goal (required)")
    if not goal:
        typer.echo("  no goal given; nothing saved.")
        raise typer.Exit(1)
    spec = GoalSpec(
        name=name,
        goal=goal,
        description=_ask("description"),
        success=_ask("done when"),
        constraints=_ask("must not"),
        persona=_ask("persona", "dev_agent"),
        model=_ask("model override (blank = the persona's)", ""),
        interval=int(_ask("seconds between cycles", "0") or 0),
        max_cycles=int(_ask("max cycles (0 = until done)", "0") or 0) or None,
        max_failures=int(_ask("stop after N consecutive failures", "5") or 5),
        max_idle=int(_ask("stop after N idle cycles", "5") or 5),
        tools=[t.strip() for t in tools.split(",") if t.strip()],
    )
    unknown = spec.unknown_tools()
    if unknown:
        typer.echo(f"  ! unknown tools: {', '.join(unknown)} — saved anyway, fix the file")
    path = spec.save()
    typer.echo(f"\n  saved to {path}")
    typer.echo(f"  run it with: lag agent start {name}")


@agent_app.command("list")
def agent_list():
    """List saved agents and whether each is on."""
    from .agent import AgentState, GoalSpec

    specs = GoalSpec.list_all()
    if not specs:
        typer.echo("no saved agents — define one with: lag agent new <name>")
        return
    for sp in specs:
        st = AgentState.load(sp.name)
        where = ""
        if st:
            where = f" · cycle {st.cycle}" + (" · DONE" if st.done else "")
            if st.stopped_reason:
                where += f" · stopped: {st.stopped_reason}"
        flag = "on " if sp.enabled else "off"
        typer.echo(f"  [{flag}] {sp.name:18} {sp.persona:18}{where}")
        typer.echo(f"         {sp.description or sp.goal[:80]}")


@agent_app.command("show")
def agent_show(name: str = typer.Argument(..., help="Saved agent name.")):
    """Print a saved agent's full definition."""
    from .agent import GoalSpec

    try:
        sp = GoalSpec.load(name)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"  {e}")
        raise typer.Exit(1) from None
    for k, v in vars(sp).items():
        typer.echo(f"  {k:14}: {v if v not in ([], '') else '(unset)'}")
    typer.echo(f"  file          : {GoalSpec.path_for(name)}")


def _set_enabled(name: str, on: bool):
    from .agent import GoalSpec

    try:
        sp = GoalSpec.load(name)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"  {e}")
        raise typer.Exit(1) from None
    sp.enabled = on
    sp.save()
    typer.echo(f"  {name} is now {'enabled' if on else 'disabled'}")


@agent_app.command("enable")
def agent_enable(name: str = typer.Argument(..., help="Saved agent name.")):
    """Turn a saved agent on."""
    _set_enabled(name, True)


@agent_app.command("disable")
def agent_disable(name: str = typer.Argument(..., help="Saved agent name.")):
    """Turn a saved agent off without deleting it."""
    _set_enabled(name, False)


@agent_app.command("delete")
def agent_delete(
    name: str = typer.Argument(..., help="Saved agent name."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation."),
):
    """Delete a saved agent, its run state, and its stop flag."""
    from .agent import GoalSpec

    try:
        sp = GoalSpec.load(name)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"  {e}")
        raise typer.Exit(1) from None
    if not yes:
        typer.echo(f"  about to delete {name!r} and its run history:")
        typer.echo(f"    {sp.goal[:100]}")
        if _ask("type the name to confirm") != name:
            typer.echo("  cancelled.")
            raise typer.Exit(1)
    sp.delete()
    typer.echo(f"  deleted {name}")


@agent_app.command("status")
def agent_status(name: str = typer.Argument("default", help="Run name.")):
    """Show what an agent is working on and how far it has got."""
    from .agent import AgentState

    st = AgentState.load(name)
    if not st:
        typer.echo(f"no agent state for {name!r} — start one with: lag agent start")
        raise typer.Exit(1)
    typer.echo(f"  goal    : {st.spec.get('goal','')}")
    typer.echo(f"  status  : {'DONE' if st.done else 'in progress'}")
    typer.echo(f"  cycles  : {st.cycle}   errors: {st.errors}")
    typer.echo(f"  started : {st.started}    updated: {st.updated}")
    if st.summary:
        typer.echo(f"  summary : {st.summary}")
    if st.stopped_reason:
        typer.echo(f"  stopped : {st.stopped_reason}")
    if st.notes:
        typer.echo("  progress:")
        for n in st.notes[-15:]:
            typer.echo(f"    - {n}")


@agent_app.command("stop")
def agent_stop(name: str = typer.Argument("default", help="Run name.")):
    """Ask a running agent to stop after its current cycle."""
    from .agent import stop_path

    p = stop_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("stop")
    typer.echo(f"stop requested for {name!r} — it will exit within a second of its "
               f"current cycle finishing ({p})")


study_app = typer.Typer(help="Course brain: study your own material, then get drilled on it.")
app.add_typer(study_app, name="study")


def _course(name: str):
    from .study import Course
    return Course(name, load_config())


@study_app.command("add")
def study_add(course: str, path: str):
    """Ingest notes, PDFs, notebooks, or code into a course."""
    c = _course(course)
    print(f"reading {path} ...")
    docs, chunks = c.add(path)
    if not docs:
        print("no readable material found (.md .txt .pdf .ipynb and common code files)")
        raise typer.Exit(1)
    print(f"✓ {docs} document(s) → {chunks} chunks · {c.count()} total in '{course}'")


@study_app.command("ask")
def study_ask(course: str, question: str, k: int = typer.Option(5, help="Chunks to retrieve.")):
    """Ask a question answered only from this course's material."""
    c = _course(course)
    if not c.count():
        print(f"'{course}' is empty — run: lag study add {course} <path>")
        raise typer.Exit(1)
    answer, hits = c.ask(question, k=k)
    print(f"\n{answer}\n")
    if hits:
        print("sources:")
        for h in hits:
            print(f"  · {h['source']}  (similarity {1 - h['distance']:.2f})")


@study_app.command("quiz")
def study_quiz(course: str, n: int = 5, topic: str = typer.Option(None, help="Focus one topic."),
               cards: bool = typer.Option(True, help="Seed flashcards from what you miss.")):
    """Get examined on the material, with your answers graded against the source."""
    from .study import Deck, Progress
    c = _course(course)
    if not c.count():
        print(f"'{course}' is empty — run: lag study add {course} <path>")
        raise typer.Exit(1)

    print(f"generating {n} question(s) from '{course}' ...")
    questions = c.generate(n, topic=topic)
    if not questions:
        print("could not generate questions — try adding more material")
        raise typer.Exit(1)

    progress, scored = Progress.load(course), []
    for i, q in enumerate(questions, 1):
        print(f"\n─── {i}/{len(questions)} · {q['topic']} · {q['kind']} " + "─" * 20)
        print(f"\n{q['question']}\n")
        try:
            answer = input("your answer > ")
        except (EOFError, KeyboardInterrupt):
            print("\nstopped.")
            break
        g = c.grade(q["question"], q["answer"], answer)
        mark = {"correct": "✓", "partial": "~"}.get(g["verdict"], "✗")
        print(f"\n{mark} {g['score']}/100 · {g['verdict']}")
        if g["missing"]:
            print(f"  {g['missing']}")
        if q["answer"]:
            print(f"\n  reference: {q['answer']}")
        progress.record(q["topic"], q["question"], g["score"])
        scored.append(g["score"])

    if scored:
        progress.save()
        print(f"\n{'═' * 52}\nsession: {sum(scored) // len(scored)}/100 mean over {len(scored)}")
        weak = progress.weak_topics()
        if weak:
            print(f"weak topics: {', '.join(weak)}")
        if cards and weak:
            print("\nbuilding flashcards for what you missed ...")
            deck = Deck.load(course)
            added = deck.add(c.cards_for_weaknesses(progress))
            deck.save()
            print(f"✓ {added} new card(s) · review with: lag study review {course}")


@study_app.command("cards")
def study_cards(course: str, n: int = 10, topic: str = typer.Option(None)):
    """Generate flashcards from the material."""
    from .study import Deck
    c = _course(course)
    if not c.count():
        print(f"'{course}' is empty — run: lag study add {course} <path>")
        raise typer.Exit(1)
    print(f"writing {n} card(s) ...")
    deck = Deck.load(course)
    added = deck.add(c.make_cards(n, topic=topic))
    deck.save()
    s = deck.stats()
    print(f"✓ {added} new · {s['total']} total · {s['due']} due now")


@study_app.command("review")
def study_review(course: str, limit: int = typer.Option(20, help="Max cards this session.")):
    """Review the cards that are due."""
    from .study import Deck
    deck = Deck.load(course)
    due = deck.due()[:limit]
    if not due:
        s = deck.stats()
        print(f"nothing due · {s['total']} card(s) in the deck"
              if s["total"] else f"no cards yet — run: lag study cards {course}")
        return

    print(f"{len(due)} card(s) due\n")
    done = 0
    for i, card in enumerate(due, 1):
        print(f"─── {i}/{len(due)} · {card.topic} " + "─" * 24)
        print(f"\n{card.front}\n")
        try:
            input("[enter to reveal] ")
            print(f"→ {card.back}\n")
            r = input("again / hard / good / easy  [a/h/g/e] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nstopped.")
            break
        card.review({"a": 0, "h": 3, "g": 4, "e": 5}.get(r[:1] if r else "g", 4))
        done += 1
        print()

    deck.save()
    s = deck.stats()
    print(f"{'═' * 52}\nreviewed {done} · {s['due']} still due · {s['mature']} mature")


@study_app.command("status")
def study_status(course: str = typer.Argument(None, help="Omit to list all courses.")):
    """Show what's in a course and how you're doing."""
    from . import rag
    from .study import Deck, Progress, collection_for

    if not course:
        names = [c.removeprefix("course_") for c in rag.list_collections()
                 if c.startswith("course_")]
        print("courses: " + (", ".join(names) if names else "(none yet)"))
        return

    c = _course(course)
    print(f"'{course}' · {c.count()} chunks")
    src = c.sources()
    if src:
        print("  material: " + ", ".join(src[:12]) + (" …" if len(src) > 12 else ""))

    s = Deck.load(course).stats()
    print(f"  cards: {s['total']} total · {s['due']} due · {s['new']} new · "
          f"{s['mature']} mature · {s['lapses']} lapses")

    rows = Progress.load(course).by_topic()
    if rows:
        print("  quiz history (weakest first):")
        for t, r in list(rows.items())[:10]:
            bar = "█" * int(r["mean"] / 10) + "░" * (10 - int(r["mean"] / 10))
            print(f"    {bar} {r['mean']:5.1f}  {t}  ({r['n']}×)")


if __name__ == "__main__":
    app()
