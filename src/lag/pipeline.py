"""Sequential project pipeline: one model resident at a time, context passed down the line.

Replaces the old three-mode `Swarm`, whose name was a lie on this box. `parallel` fired
every agent at once through a ThreadPoolExecutor, but Ollama here holds exactly ONE chat
model resident — a hard concurrency cap, measured, not a memory ceiling (probing at
num_ctx 4096/6144/8192 the second model evicted the first every time while the total sat
at ~5.9 GB of 16 GB). Concurrent agents on different tags therefore evict each other in a
loop. Sequential isn't a compromise here; it's the only shape that runs at full speed.

`crew` went for the same reason. Its coordinator bookended the run — plan, delegate,
then come *back* to synthesize — and that return trip forced a second cold reload at
~5s. Here the lead plans ONCE at the top, its per-stage instructions ride the
accumulated context down the line, and the last stage writes the answer on whatever
model is already resident. Two loads instead of three.

Stage order is meaning (design, build, review) and simultaneously the cost model:
consecutive stages sharing a tag reuse the resident weights for free, and every change
of tag costs a full reload. `run` announces each stage as LOAD or resident so that cost
is never invisible.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import load_persona
from .ollama_client import OllamaClient
from .tools import LocalToolProvider


class Agent:
    def __init__(self, persona: dict, cfg: dict, tool_provider=None,
                 model_override: str | None = None):
        self.name = persona["name"]
        self.persona = persona
        # Resolved tag, kept on the agent so the pipeline can reason about swap cost
        # without reaching back through the persona. An override replaces it outright:
        # picking one model for a run is a deliberate way to spend zero reloads, since
        # every stage then shares the resident weights.
        self.model = model_override or persona["model"]
        # from_config so every agent inherits num_ctx / keep_alive. Pipeline stages need
        # the window most: each one inherits the whole accumulated context above it.
        self.client = OllamaClient.from_config(
            cfg, self.model, system=persona.get("system_prompt"),
            think=persona.get("thinking_mode"),
        )
        self.tool_provider = tool_provider

    def run(self, task: str, on_event=None) -> str:
        return self.client.run(task, tool_provider=self.tool_provider, on_event=on_event)


class Pipeline:
    def __init__(self, stages: list[Agent], host: str):
        self.stages = stages
        self.host = host

    @classmethod
    def from_personas(cls, names: list[str], cfg: dict, tool_provider=None,
                      model_override: str | None = None):
        host = cfg["ollama_base_url"]
        provider = tool_provider or LocalToolProvider()
        stages = [Agent(load_persona(n, cfg), cfg, provider, model_override)
                  for n in names]
        return cls(stages, host)

    def loads(self) -> int:
        """How many cold model loads this ordering costs — one per change of tag."""
        tags = [s.model for s in self.stages]
        if not tags:
            return 0
        return 1 + sum(1 for a, b in zip(tags, tags[1:]) if a != b)

    @classmethod
    def routed(cls, task: str, cfg: dict, tool_provider=None,
               lead_name: str | None = None, on_event=None,
               model_override: str | None = None):
        """Let the lead CHOOSE the stages for this task instead of taking a fixed list.

        A fixed roster runs architect → dev → senior_dev whether the task is "design a
        parser" or "research three vector stores", which wastes stages and puts the
        wrong specialist on the work. Here the lead reads the task against the roster's
        role lines and returns the ordered subset that should handle it.

        Selection and briefing are ONE call, deliberately. The lead is the orchestrator
        model and would already be paying a load to write the plan, so choosing the team
        in the same pass costs nothing extra — a separate routing call would double the
        orchestrator's turn for no new information.

        Falls back to the configured default roster if the model returns something
        unusable: a bad route should degrade to the old behaviour, never to a crash.
        """
        provider = tool_provider or LocalToolProvider()
        host = cfg["ollama_base_url"]
        lead_name = lead_name or cls._default_names(cfg)[0]
        roster = cls._roster(cfg, exclude=lead_name)

        lead = Agent(load_persona(lead_name, cfg), cfg, provider, model_override)
        if on_event:
            on_event({"type": "routing", "lead": lead.name, "model": lead.model,
                      "candidates": [n for n, _ in roster]})

        listing = "\n".join(f"- {n}: {r}" for n, r in roster)
        raw = lead.run(
            "You are routing a task to a team. Choose ONLY the specialists this task "
            "actually needs, in the order they should work, and write one concrete "
            "instruction for each. Choosing fewer is better than choosing all — every "
            "extra stage costs a model load.\n\n"
            f"Available specialists:\n{listing}\n\n"
            f"Task: {task}\n\n"
            'Reply with ONLY a JSON object, no prose and no code fence:\n'
            '{"agents":[{"name":"<exact name from the list>","instruction":"<what they do>"}]}',
            on_event=on_event,
        )

        chosen = cls._parse_route(raw, {n for n, _ in roster})
        if not chosen:
            chosen = [(n, "") for n in cls._default_names(cfg) if n != lead_name]
            if on_event:
                on_event({"type": "route_fallback",
                          "agents": [n for n, _ in chosen]})

        stages = [Agent(load_persona(n, cfg), cfg, provider, model_override)
                  for n, _ in chosen]
        pipe = cls(stages, host)
        pipe.lead = lead
        pipe.brief = "\n".join(
            f"- {n}: {i}" for n, i in chosen if i
        ) or raw.strip()
        if on_event:
            on_event({"type": "routed", "lead": lead.name,
                      "agents": [{"name": a.name, "model": a.model} for a in stages],
                      "loads": pipe.loads() + 1})
        return pipe

    @staticmethod
    def _default_names(cfg: dict) -> list[str]:
        raw = cfg.get("pipeline", {}).get(
            "default_agents", "architect_agent,dev_agent,senior_dev_agent")
        return [a.strip() for a in raw.split(",") if a.strip()]

    @staticmethod
    def _roster(cfg: dict, exclude: str | None = None) -> list[tuple[str, str]]:
        """(name, role) for every persona on disk — what the lead routes against."""
        import yaml
        base = Path(cfg.get("paths", {}).get("personas", "./personas"))
        out = []
        for f in sorted(base.glob("*.yaml")):
            if f.stem == exclude:
                continue
            try:
                d = yaml.safe_load(f.read_text()) or {}
            except Exception:  # noqa: BLE001 - a broken persona must not kill routing
                continue
            out.append((d.get("name", f.stem), d.get("role", "")))
        return out

    @staticmethod
    def _parse_route(raw: str, valid: set[str]) -> list[tuple[str, str]]:
        """Pull the agent list out of the lead's reply, tolerantly.

        Small local models fence their JSON, prefix it with prose, or emit the object
        after a thinking preamble, so this scans for the first balanced object rather
        than trusting the whole string to parse. Unknown or duplicated names are
        dropped — a hallucinated persona would otherwise crash load_persona.
        """
        if not raw:
            return []
        blob = None
        for m in re.finditer(r"\{", raw):
            depth = 0
            for i in range(m.start(), len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            blob = json.loads(raw[m.start():i + 1])
                        except ValueError:
                            blob = None
                        break
            if isinstance(blob, dict) and "agents" in blob:
                break
        if not isinstance(blob, dict):
            return []

        chosen, seen = [], set()
        for item in blob.get("agents") or []:
            name = (item.get("name") if isinstance(item, dict) else item) or ""
            name = str(name).strip()
            if name in valid and name not in seen:
                seen.add(name)
                instr = item.get("instruction", "") if isinstance(item, dict) else ""
                chosen.append((name, str(instr).strip()))
        return chosen

    def _brief(self, lead: Agent, task: str, rest: list[Agent], on_event=None) -> str:
        """Ask the lead for the whole downstream plan in one pass.

        Deliberately capped. The lead usually holds a far larger window than the stages
        that read it (here 131072 vs 8192), and an over-long brief is not an error —
        the receiving stage silently drops the top of it. Bounding the brief keeps the
        plan inside the smallest window in the pipeline.
        """
        roster = ", ".join(s.name for s in rest)
        return lead.run(
            "You are opening a project pipeline. Write one concrete instruction for each "
            f"stage that follows — {roster} — in that order, so each can work from your "
            "text alone. Finish by stating what the final stage must deliver. You will "
            "not be called again, so put everything downstream needs here.\n\n"
            "Be terse: a short paragraph per stage, no preamble, no restating the task. "
            "Later stages read this on a much smaller context window than yours.\n\n"
            f"Task: {task}",
            on_event=on_event,
        )

    # Roughly 4 characters per token, and only part of the window is ours: the system
    # prompt, nine tool schemas, retrieved chunks and the model's own reply all have to
    # fit alongside. 55% is deliberately conservative — overflowing is a hard failure,
    # under-filling only costs a little context.
    _CHARS_PER_TOKEN = 4
    _USABLE = 0.55

    def _fit(self, context: str, stage: Agent, on_event=None) -> str:
        """Trim the accumulated context to what this stage's window can actually hold.

        The pipeline's whole idea is that each stage sees everything above it, but that
        grows without bound: by the third stage the task, the plan and two full outputs
        are all in play, and stages do not share a window — qwen3:8b holds 8192 where
        the lead had 32768. Overflowing is not a soft failure; the request errors.

        So keep the head (the task and the plan, which every stage needs to stay on
        brief) and the most recent output (what it is actually building on), and drop
        the middle with a visible marker rather than letting the model silently lose
        the top of its own prompt.
        """
        window = getattr(getattr(stage, "client", None), "num_ctx", None)
        if not window:
            return context
        budget = int(window * self._CHARS_PER_TOKEN * self._USABLE)
        if len(context) <= budget:
            return context

        head = budget // 3
        tail = budget - head
        marker = "\n\n[… earlier stage output trimmed to fit this stage's window …]\n\n"
        trimmed = context[:head] + marker + context[-tail:]
        if on_event:
            on_event({"type": "trimmed", "stage": stage.name, "model": stage.model,
                      "window": window, "dropped_chars": len(context) - len(trimmed)})
        print(f"[pipeline] {stage.name} · context trimmed to fit num_ctx={window} "
              f"({len(context) - len(trimmed)} chars dropped)")
        return trimmed

    def run(self, task: str, plan: bool = True, on_event=None) -> str:
        """Run every stage in order, each seeing the full context built above it.

        With `plan`, the first stage writes the downstream instructions instead of
        doing a share of the work — the lead is usually the orchestrator model, and
        this is the one place its extra reasoning is worth a load.

        on_event, when given, receives the same token/tool events the client emits plus
        a `stage` event per transition, so a UI can show which stage is talking and
        whether it cost a reload. Purely a side channel — the return value is unchanged.
        """
        if not self.stages:
            raise ValueError("pipeline needs at least one stage")

        resident: str | None = None

        def announce(stage: Agent, role: str) -> None:
            nonlocal resident
            loaded = stage.model != resident
            print(f"[pipeline] {stage.name} · {stage.model} "
                  f"({'LOAD' if loaded else 'resident'})")
            if on_event:
                on_event({"type": "stage", "name": stage.name, "model": stage.model,
                          "role": role, "loaded": loaded})
            resident = stage.model

        stages, context = self.stages, f"Task: {task}"

        brief = getattr(self, "brief", None)
        if brief:
            # routed(): the lead already chose the team and wrote their instructions in
            # the same turn, so it is NOT called again here — that would be the crew
            # coordinator's wasted second load all over again.
            context += f"\n\nPlan from {getattr(self, 'lead', None) and self.lead.name}:\n{brief}"
        elif plan and len(stages) > 1:
            lead, stages = stages[0], stages[1:]
            announce(lead, "lead")
            context += (f"\n\nPlan from {lead.name}:\n"
                        f"{self._brief(lead, task, stages, on_event=on_event)}")

        out = ""
        for stage in stages:
            announce(stage, "stage")
            out = stage.run(self._fit(context, stage, on_event), on_event=on_event)
            context += f"\n\n### {stage.name}\n{out}"
        if on_event:
            on_event({"type": "done", "answer": out})
        return out
