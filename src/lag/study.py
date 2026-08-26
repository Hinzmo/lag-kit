"""Course brain: study your own material, then get drilled on it.

Two surfaces over one corpus:

  ask   — questions answered *from your notes*, with citations. The citation is
          the point: an answer you can't trace back to a lecture is an answer you
          can't trust the night before an exam.
  quiz  — questions generated from your material, your free-text answer graded
          against the source, and a record of what you keep missing.

The quiz half is the half that works. Re-reading feels productive and barely
moves retention; being asked to produce an answer is what builds it. So the
grader is deliberately strict and always shows the source afterwards.

Each course is its own Chroma collection, so a question about transformers can't
pull in last term's material. Everything runs on the local embedder and a local
model — no account, no per-question cost, works on lecture-hall wifi.
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import rag
from .ollama_client import OllamaClient

STUDY_DIR = Path("study")
TEXT_EXT = {".md", ".txt", ".rst", ".org"}
CODE_EXT = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".sql", ".sh"}
NOTEBOOK_EXT = {".ipynb"}
PDF_EXT = {".pdf"}


def slug(name: str) -> str:
    """Course name -> filesystem- and Chroma-safe identifier."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "course"


def collection_for(course: str) -> str:
    return f"course_{slug(course)}"


# ── loading material ────────────────────────────────────────────────────────

def read_pdf(path: Path) -> str:
    """Extract text from a PDF, or explain why it couldn't."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF support needs pypdf — pip install -e '.[study]'") from None
    pages = []
    for page in PdfReader(str(path)).pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def read_notebook(path: Path) -> str:
    """Flatten a Jupyter notebook to markdown + code.

    Course material for an ML class is often notebooks, and the code cells carry
    as much of the lesson as the prose does.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        out.append(src if cell.get("cell_type") == "markdown" else f"```\n{src}\n```")
    return "\n\n".join(out)


def read_any(path: Path) -> str | None:
    """Read one file, or None if it isn't material we understand."""
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return read_pdf(path)
    if ext in NOTEBOOK_EXT:
        return read_notebook(path)
    if ext in TEXT_EXT or ext in CODE_EXT:
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def collect(target: str | Path) -> list[dict]:
    """Gather every readable file under a path (or the single file given)."""
    p = Path(target)
    paths = [p] if p.is_file() else sorted(
        q for q in p.rglob("*") if q.is_file() and not any(
            part.startswith(".") for part in q.parts))
    docs = []
    for q in paths:
        try:
            content = read_any(q)
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop an import
            print(f"  ! skipped {q.name}: {e}")
            continue
        if content and content.strip():
            docs.append({"path": str(q), "name": q.stem, "content": content})
    return docs


# ── progress ────────────────────────────────────────────────────────────────

@dataclass
class Progress:
    """Per-course record of what you've been asked and how it went."""
    course: str
    attempts: list[dict] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return STUDY_DIR / slug(self.course) / "progress.json"

    @classmethod
    def load(cls, course: str) -> Progress:
        p = STUDY_DIR / slug(course) / "progress.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return cls(course=course, attempts=d.get("attempts", []))
        return cls(course=course)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"course": self.course, "attempts": self.attempts}, indent=2), encoding="utf-8")

    def record(self, topic: str, question: str, score: int) -> None:
        self.attempts.append({
            "topic": topic, "question": question, "score": score,
            "at": time.strftime("%Y-%m-%d %H:%M"),
        })

    def by_topic(self) -> dict[str, dict]:
        """Mean score and attempt count per topic, weakest first."""
        agg: dict[str, list[int]] = {}
        for a in self.attempts:
            agg.setdefault(a["topic"], []).append(a["score"])
        rows = {t: {"n": len(v), "mean": sum(v) / len(v)} for t, v in agg.items()}
        return dict(sorted(rows.items(), key=lambda kv: kv[1]["mean"]))

    def weak_topics(self, threshold: int = 70) -> list[str]:
        return [t for t, r in self.by_topic().items() if r["mean"] < threshold]


# ── flashcards, scheduled ───────────────────────────────────────────────────
# SM-2, trimmed. The deck is not a fixed set you build once: after every quiz the
# topics you missed seed new cards, so it grows toward whatever you keep getting
# wrong instead of drilling what you already know.

DAY = 86400.0


@dataclass
class Card:
    front: str
    back: str
    topic: str
    ease: float = 2.5        # SM-2 easiness factor
    interval: float = 0.0    # days
    reps: int = 0
    lapses: int = 0
    due: float = 0.0         # epoch seconds; 0 = due now
    source: str = ""

    @property
    def id(self) -> str:
        import hashlib
        return hashlib.sha1(self.front.encode("utf-8")).hexdigest()[:16]

    def is_due(self, now: float | None = None) -> bool:
        return self.due <= (now if now is not None else time.time())

    def review(self, quality: int, now: float | None = None) -> None:
        """quality: 0 again · 3 hard · 4 good · 5 easy (SM-2 scale)."""
        now = now if now is not None else time.time()
        if quality < 3:
            # Lapse: back to the front of the queue, ease takes a small hit.
            self.reps = 0
            self.lapses += 1
            self.interval = 0
            self.ease = max(1.3, self.ease - 0.20)
            self.due = now + 600  # ten minutes, same session
            return
        self.ease = max(1.3, self.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        if self.reps == 0:
            self.interval = 1
        elif self.reps == 1:
            self.interval = 6
        else:
            self.interval = round(self.interval * self.ease, 2)
        self.reps += 1
        self.due = now + self.interval * DAY


@dataclass
class Deck:
    course: str
    cards: list[Card] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return STUDY_DIR / slug(self.course) / "deck.json"

    @classmethod
    def load(cls, course: str) -> Deck:
        p = STUDY_DIR / slug(course) / "deck.json"
        if not p.exists():
            return cls(course=course)
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(course=course, cards=[Card(**c) for c in d.get("cards", [])])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"course": self.course, "cards": [vars(c) for c in self.cards]},
            indent=2), encoding="utf-8")

    def add(self, cards: list[Card]) -> int:
        """Add cards, skipping ones already in the deck."""
        have = {c.id for c in self.cards}
        fresh = [c for c in cards if c.id not in have]
        self.cards.extend(fresh)
        return len(fresh)

    def due(self, now: float | None = None) -> list[Card]:
        now = now if now is not None else time.time()
        return sorted((c for c in self.cards if c.is_due(now)), key=lambda c: c.due)

    def stats(self) -> dict:
        now = time.time()
        return {
            "total": len(self.cards),
            "due": len(self.due(now)),
            "new": sum(1 for c in self.cards if c.reps == 0),
            "learning": sum(1 for c in self.cards if 0 < c.reps < 3),
            "mature": sum(1 for c in self.cards if c.interval >= 21),
            "lapses": sum(c.lapses for c in self.cards),
        }


# ── the course ──────────────────────────────────────────────────────────────

class Course:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.collection = collection_for(name)
        o = cfg.get("ollama", {}) or {}
        self.db = (cfg.get("paths", {}) or {}).get("chroma", rag.DB_PATH)
        self.embed = (cfg.get("models", {}) or {}).get("embed", rag.EMBED_MODEL)
        self.host = cfg.get("ollama_base_url", rag.OLLAMA_HOST)
        self._engine = None
        self._ollama_opts = o

    # -- ingest --------------------------------------------------------------
    def add(self, target: str | Path) -> tuple[int, int]:
        """Ingest a file or folder. Returns (documents, chunks)."""
        docs = collect(target)
        if not docs:
            return 0, 0
        texts, metas = [], []
        for d in docs:
            for c in rag.chunk_text(d["content"]):
                texts.append(c)
                metas.append({
                    "source": d["name"], "path": d["path"],
                    "kind": "course", "course": self.name,
                })
        n = rag.add_texts(texts, metas, prefix=f"c-{slug(self.name)}",
                          db_path=self.db, collection=self.collection,
                          model=self.embed, host=self.host)
        return len(docs), n

    @property
    def engine(self) -> rag.RAGEngine:
        if self._engine is None:
            self._engine = rag.RAGEngine(db_path=self.db, model=self.embed,
                                         host=self.host, collection=self.collection)
        return self._engine

    def count(self) -> int:
        return self.engine.collection.count()

    def sources(self) -> list[str]:
        got = self.engine.collection.get(include=["metadatas"])
        return sorted({(m or {}).get("source", "?") for m in got.get("metadatas", [])})

    # -- model ---------------------------------------------------------------
    def _client(self, role: str = "orchestrator", system: str | None = None) -> OllamaClient:
        models = self.cfg.get("models", {}) or {}
        return OllamaClient.from_config(
            self.cfg, models.get(role) or models.get("worker"), system=system)

    # -- ask -----------------------------------------------------------------
    ASK_SYSTEM = (
        "You are a study assistant. Answer ONLY from the provided course material. "
        "If the material does not contain the answer, say so plainly — do not fill the "
        "gap from general knowledge, because the student is being examined on this "
        "specific material. Cite sources inline as [source]. Be direct and concrete."
    )

    def ask(self, question: str, k: int = 5) -> tuple[str, list[dict]]:
        hits = self.engine.search(question, k=k)
        if not hits:
            return "(nothing in this course matches — add material first)", []
        context = "\n\n".join(
            f"[{h['source']}]\n{h['text']}" for h in hits)
        prompt = (f"Course material:\n\n{context}\n\n"
                  f"Question: {question}\n\nAnswer using only the material above.")
        answer = self._client("orchestrator", self.ASK_SYSTEM).run(
            prompt, tool_provider=None, verbose=False)
        return answer, hits

    # -- quiz ----------------------------------------------------------------
    GEN_SYSTEM = (
        "You write exam questions from course material. Output STRICT JSON only — "
        "no prose, no markdown fence. A list of objects with keys: "
        '"topic" (2-4 words), "question", "answer" (the correct answer, drawn only '
        'from the material), "kind" (one of: concept, implementation, tradeoff). '
        "Favour questions that require producing an explanation or writing something, "
        "not yes/no recall. Never ask about anything absent from the material."
    )

    def generate(self, n: int = 5, topic: str | None = None) -> list[dict]:
        seed = topic or "the core ideas of this course"
        hits = self.engine.search(seed, k=min(8, max(self.count(), 1)))
        if not hits:
            return []
        material = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        raw = self._client("orchestrator", self.GEN_SYSTEM).run(
            f"Course material:\n\n{material}\n\nWrite {n} exam questions as JSON.",
            tool_provider=None, verbose=False)
        return _parse_questions(raw, n)

    CARD_SYSTEM = (
        "You write flashcards from course material. Output STRICT JSON only — a list "
        'of objects with keys: "topic" (2-4 words), "front" (the prompt), "back" (the '
        "answer). One idea per card. The front must be answerable without seeing the "
        "material; the back must be short enough to check at a glance. Prefer 'why' and "
        "'what breaks if' over pure definitions. Never invent material."
    )

    def make_cards(self, n: int = 10, topic: str | None = None) -> list[Card]:
        """Generate cards from the material, optionally aimed at one topic."""
        seed = topic or "the core ideas of this course"
        hits = self.engine.search(seed, k=min(8, max(self.count(), 1)))
        if not hits:
            return []
        material = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        raw = self._client("orchestrator", self.CARD_SYSTEM).run(
            f"Course material:\n\n{material}\n\nWrite {n} flashcards as JSON.",
            tool_provider=None, verbose=False)
        src = hits[0]["source"] if hits else ""
        out = []
        for d in _parse_cards(raw, n):
            out.append(Card(front=d["front"], back=d["back"],
                            topic=d["topic"] or (topic or "general"), source=src))
        return out

    def cards_for_weaknesses(self, progress: Progress, per_topic: int = 4) -> list[Card]:
        """Seed new cards from whatever the quiz says you keep missing.

        This is the 'as it goes along' part: the deck is not static, it chases
        the topics your scores say you haven't got yet.
        """
        cards: list[Card] = []
        for t in progress.weak_topics():
            cards.extend(self.make_cards(per_topic, topic=t))
        return cards

    GRADE_SYSTEM = (
        "You grade a student's free-text answer against a reference answer. "
        "Output STRICT JSON only: {\"score\": 0-100, \"verdict\": \"correct|partial|incorrect\", "
        "\"missing\": \"what they left out or got wrong, one or two sentences\"}. "
        "Reward demonstrated understanding over exact wording. Do not be generous "
        "with a confident answer that is wrong — say so."
    )

    def grade(self, question: str, reference: str, answer: str) -> dict:
        if not answer.strip():
            return {"score": 0, "verdict": "incorrect", "missing": "No answer given."}
        raw = self._client("orchestrator", self.GRADE_SYSTEM).run(
            f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Student answer: {answer}\n\nGrade it.",
            tool_provider=None, verbose=False)
        return _parse_grade(raw)


# ── tolerant parsing ────────────────────────────────────────────────────────
# Small local models wrap JSON in prose or fences no matter how firmly the system
# prompt objects. Recover what we can rather than failing the whole quiz.

def _json_slice(raw: str, opener: str, closer: str):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    i, j = raw.find(opener), raw.rfind(closer)
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(raw[i:j + 1])
    except json.JSONDecodeError:
        return None


def _parse_questions(raw: str, n: int) -> list[dict]:
    data = _json_slice(raw, "[", "]")
    if not isinstance(data, list):
        one = _json_slice(raw, "{", "}")
        data = [one] if isinstance(one, dict) else []
    out = []
    for d in data:
        if not isinstance(d, dict) or not d.get("question"):
            continue
        out.append({
            "topic": str(d.get("topic") or "general").strip(),
            "question": str(d["question"]).strip(),
            "answer": str(d.get("answer") or "").strip(),
            "kind": str(d.get("kind") or "concept").strip(),
        })
    return out[:n]


def _parse_cards(raw: str, n: int) -> list[dict]:
    data = _json_slice(raw, "[", "]")
    if not isinstance(data, list):
        one = _json_slice(raw, "{", "}")
        data = [one] if isinstance(one, dict) else []
    out = []
    for d in data:
        if not isinstance(d, dict) or not d.get("front") or not d.get("back"):
            continue
        out.append({
            "topic": str(d.get("topic") or "general").strip(),
            "front": str(d["front"]).strip(),
            "back": str(d["back"]).strip(),
        })
    return out[:n]


def _parse_grade(raw: str) -> dict:
    d = _json_slice(raw, "{", "}")
    if not isinstance(d, dict):
        # Last resort: find a bare number so a malformed grade still counts.
        m = re.search(r"\b(\d{1,3})\b", raw)
        score = min(int(m.group(1)), 100) if m else 0
        return {"score": score, "verdict": "partial", "missing": raw.strip()[:200]}
    try:
        score = max(0, min(int(float(d.get("score", 0))), 100))
    except (TypeError, ValueError):
        score = 0
    return {
        "score": score,
        "verdict": str(d.get("verdict") or "partial"),
        "missing": str(d.get("missing") or "").strip(),
    }
