"""Study-tool tests. Nothing here touches Ollama or the network.

The two things worth pinning hard are the scheduler (pure arithmetic — if it
drifts you silently stop seeing cards) and the parsers (small local models wrap
JSON in prose no matter what the system prompt says, and a quiz that dies on a
stray ``` is a quiz you stop using).
"""
from __future__ import annotations

import pytest

pytest.importorskip("chromadb", reason="install with .[rag]")
from lag import study  # noqa: E402
from lag.study import Card, Deck, Progress  # noqa: E402

DAY = study.DAY


class TestSlug:
    @pytest.mark.parametrize("raw,want", [
        ("AI Engineering", "ai-engineering"),
        ("CS 5010: Intro!", "cs-5010-intro"),
        ("  spaced  out  ", "spaced-out"),
        ("!!!", "course"),          # never empty — it becomes a path and a collection name
    ])
    def test_slugs(self, raw, want):
        assert study.slug(raw) == want

    def test_collection_is_namespaced(self):
        assert study.collection_for("AI Eng") == "course_ai-eng"


class TestScheduler:
    """SM-2, trimmed. Quality: 0 again · 3 hard · 4 good · 5 easy."""

    def test_first_correct_review_is_one_day(self):
        c = Card(front="f", back="b", topic="t")
        c.review(4, now=0)
        assert c.reps == 1 and c.interval == 1 and c.due == DAY

    def test_second_correct_review_is_six_days(self):
        c = Card(front="f", back="b", topic="t")
        c.review(4, now=0)
        c.review(4, now=0)
        assert c.reps == 2 and c.interval == 6

    def test_intervals_grow_by_ease_after_that(self):
        c = Card(front="f", back="b", topic="t")
        for _ in range(3):
            c.review(4, now=0)
        assert c.interval > 6          # 6 * ease
        assert c.reps == 3

    def test_lapse_resets_and_reschedules_within_session(self):
        c = Card(front="f", back="b", topic="t")
        c.review(4, now=0)
        c.review(4, now=0)
        c.review(0, now=1000)          # forgot it
        assert c.reps == 0
        assert c.lapses == 1
        assert c.interval == 0
        assert c.due == 1000 + 600     # ten minutes, same sitting

    def test_ease_falls_on_lapse_but_never_below_floor(self):
        c = Card(front="f", back="b", topic="t")
        for i in range(40):
            c.review(0, now=i)
        assert c.ease == pytest.approx(1.3)   # SM-2 floor; cards never become impossible

    def test_easy_answers_raise_ease_above_default(self):
        c = Card(front="f", back="b", topic="t")
        c.review(5, now=0)
        assert c.ease > 2.5

    def test_hard_answers_lower_ease_but_still_advance(self):
        c = Card(front="f", back="b", topic="t")
        c.review(3, now=0)
        assert c.ease < 2.5
        assert c.reps == 1             # 3 is a pass, not a lapse

    def test_due_before_first_review(self):
        assert Card(front="f", back="b", topic="t").is_due(now=0)

    def test_not_due_until_the_interval_elapses(self):
        c = Card(front="f", back="b", topic="t")
        c.review(4, now=0)
        assert not c.is_due(now=DAY - 1)
        assert c.is_due(now=DAY)


class TestDeck:
    def _deck(self, tmp_path, monkeypatch, course="c"):
        monkeypatch.setattr(study, "STUDY_DIR", tmp_path)
        return Deck(course=course)

    def test_add_skips_duplicate_fronts(self, tmp_path, monkeypatch):
        d = self._deck(tmp_path, monkeypatch)
        assert d.add([Card("same", "a", "t"), Card("other", "b", "t")]) == 2
        assert d.add([Card("same", "different back", "t")]) == 0
        assert len(d.cards) == 2

    def test_due_is_ordered_and_excludes_scheduled(self, tmp_path, monkeypatch):
        d = self._deck(tmp_path, monkeypatch)
        soon, later, gone = Card("1", "a", "t"), Card("2", "b", "t"), Card("3", "c", "t")
        soon.due, later.due, gone.due = 10, 20, 10 ** 12
        d.add([later, soon, gone])
        assert [c.front for c in d.due(now=100)] == ["1", "2"]

    def test_roundtrip_preserves_schedule(self, tmp_path, monkeypatch):
        d = self._deck(tmp_path, monkeypatch)
        c = Card("f", "b", "t")
        c.review(4, now=0)
        d.add([c])
        d.save()

        loaded = Deck.load("c")
        assert len(loaded.cards) == 1
        assert loaded.cards[0].interval == 1
        assert loaded.cards[0].due == DAY

    def test_stats_counts_buckets(self, tmp_path, monkeypatch):
        d = self._deck(tmp_path, monkeypatch)
        fresh = Card("new", "b", "t")
        mature = Card("mature", "b", "t")
        mature.interval, mature.reps, mature.due = 30, 5, 10 ** 12
        d.add([fresh, mature])
        s = d.stats()
        assert s["total"] == 2 and s["new"] == 1 and s["mature"] == 1 and s["due"] == 1

    def test_missing_deck_loads_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(study, "STUDY_DIR", tmp_path)
        assert Deck.load("never-seen").cards == []


class TestProgress:
    def test_weak_topics_are_below_threshold_and_sorted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(study, "STUDY_DIR", tmp_path)
        p = Progress(course="c")
        p.record("attention", "q", 30)
        p.record("attention", "q", 50)   # mean 40
        p.record("tokenizing", "q", 95)
        p.record("embeddings", "q", 60)

        assert p.weak_topics() == ["attention", "embeddings"]
        assert list(p.by_topic()) == ["attention", "embeddings", "tokenizing"]
        assert p.by_topic()["attention"]["n"] == 2

    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(study, "STUDY_DIR", tmp_path)
        p = Progress(course="c")
        p.record("t", "q", 80)
        p.save()
        assert len(Progress.load("c").attempts) == 1


class TestTolerantParsing:
    """Local models fence, prefix, and chatter. Recover what's recoverable."""

    def test_plain_json_list(self):
        out = study._parse_questions('[{"topic":"t","question":"q","answer":"a"}]', 5)
        assert out[0]["question"] == "q"

    def test_fenced_json(self):
        raw = '```json\n[{"topic":"t","question":"q","answer":"a"}]\n```'
        assert len(study._parse_questions(raw, 5)) == 1

    def test_json_buried_in_prose(self):
        raw = 'Sure! Here are your questions:\n[{"topic":"t","question":"q","answer":"a"}]\nHope that helps.'
        assert len(study._parse_questions(raw, 5)) == 1

    def test_single_object_instead_of_list(self):
        assert len(study._parse_questions('{"topic":"t","question":"q","answer":"a"}', 5)) == 1

    def test_entries_missing_a_question_are_dropped(self):
        raw = '[{"topic":"t","answer":"a"},{"topic":"t","question":"real","answer":"a"}]'
        out = study._parse_questions(raw, 5)
        assert len(out) == 1 and out[0]["question"] == "real"

    def test_unparseable_yields_nothing_rather_than_raising(self):
        assert study._parse_questions("I cannot help with that.", 5) == []

    def test_respects_the_requested_count(self):
        raw = "[" + ",".join('{"topic":"t","question":"q%d","answer":"a"}' % i
                             for i in range(10)) + "]"
        assert len(study._parse_questions(raw, 3)) == 3

    def test_cards_need_both_sides(self):
        raw = '[{"front":"f"},{"front":"f2","back":"b2","topic":"t"}]'
        out = study._parse_cards(raw, 5)
        assert len(out) == 1 and out[0]["back"] == "b2"

    def test_grade_clamps_out_of_range_scores(self):
        assert study._parse_grade('{"score": 250, "verdict":"correct"}')["score"] == 100
        assert study._parse_grade('{"score": -10, "verdict":"incorrect"}')["score"] == 0

    def test_grade_falls_back_to_a_bare_number(self):
        g = study._parse_grade("I'd give that about 70 out of 100.")
        assert g["score"] == 70

    def test_grade_survives_total_garbage(self):
        assert study._parse_grade("no.")["score"] == 0


class TestReaders:
    def test_notebook_flattens_markdown_and_code(self, tmp_path):
        nb = tmp_path / "lesson.ipynb"
        nb.write_text('{"cells":['
                      '{"cell_type":"markdown","source":["# Attention"]},'
                      '{"cell_type":"code","source":["import torch\\n"]},'
                      '{"cell_type":"code","source":[]}]}', encoding="utf-8")
        out = study.read_notebook(nb)
        assert "# Attention" in out
        assert "```" in out and "import torch" in out

    def test_collect_reads_text_and_code_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "notes.md").write_text("lecture one", encoding="utf-8")
        (tmp_path / "sub" / "train.py").write_text("import torch", encoding="utf-8")
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")
        names = {d["name"] for d in study.collect(tmp_path)}
        assert names == {"notes", "train"}

    def test_collect_skips_dotfiles(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "x.md").write_text("secret", encoding="utf-8")
        assert study.collect(tmp_path) == []

    def test_collect_accepts_a_single_file(self, tmp_path):
        f = tmp_path / "one.md"
        f.write_text("just this", encoding="utf-8")
        assert len(study.collect(f)) == 1
