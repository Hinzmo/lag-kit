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
def swarm(
    task: str,
    mode: str = typer.Option(None, help="pipeline | parallel | crew"),
    agents: str = typer.Option("architect_agent,dev_agent,senior_dev_agent", help="Comma-separated persona names (first is coordinator in crew mode)."),
    mcp: bool = typer.Option(False, "--mcp", help="Give every agent the MCP toolset (fs, shell, web, memory, and search_knowledge_base) instead of only in-process tools."),
):
    """Run a multi-agent swarm on a task."""
    from .swarm import Swarm
    cfg = load_config()
    mode = mode or cfg["swarm"]["default_mode"]
    names = [a.strip() for a in agents.split(",")]

    # With --mcp, all agents share ONE provider backed by both MCP servers, so the
    # orchestrator gains search_knowledge_base and can pull back facts it stored with
    # remember — the memory-offload loop that makes a small num_ctx workable. Without
    # it, from_personas falls back to in-process tools (no semantic KB search).
    provider = None
    if mcp:
        from .mcp_client import MCPToolProvider
        provider = MCPToolProvider.from_config(cfg)

    print(Swarm.from_personas(names, cfg, provider).run(task, mode=mode))


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
