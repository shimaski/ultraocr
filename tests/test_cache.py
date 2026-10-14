from __future__ import annotations

import json
import tempfile
import unittest
import sqlite3
from pathlib import Path

from manga_rt_translator.cache import TranslationMemory, normalize_text


class TranslationMemoryTest(unittest.TestCase):
    def test_translation_memory_json_round_trip_keeps_translation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = TranslationMemory(Path(tmp) / "source.sqlite3")
            source.save(
                "I will never forgive you", "Nunca vou te perdoar", "en->pt-BR",
                "en", "pt-BR", context="Sakura speaking to Naruto",
            )
            source.save(
                "I will never forgive you", "Nunca vou te perdoar", "en->pt-BR",
                "en", "pt-BR", context="Sakura speaking to Naruto",
            )
            source.record_history(
                "I will never forgive you", "Nunca vou te perdoar", "en->pt-BR", "google"
            )
            source.save_visual(
                "page-region", "I will never forgive you", "Nunca vou te perdoar",
                "en->pt-BR", (10, 20, 80, 30),
            )
            export_path = Path(tmp) / "memory.json"
            assert source.export_json(export_path) == 1

            imported = TranslationMemory(Path(tmp) / "imported.sqlite3", fuzzy_threshold=80)
            result = imported.import_json(export_path)

            row = imported.conn.execute(
                "select frequency_count, context, source_lang, target_lang from translations"
            ).fetchone()
            hit = imported.lookup("I will never forgive ya", "en->pt-BR")
            history = imported.history()
            visual_hit = imported.get_visual("page-region", "en->pt-BR")
            source.close()
            imported.close()
            self.assertEqual(result.inserted, 1)
            self.assertEqual(result.updated, 0)
            self.assertEqual(result.kept_existing, 0)
            self.assertIsNotNone(hit)
            self.assertEqual(hit.translated_text, "Nunca vou te perdoar")
            self.assertEqual(row["frequency_count"], 2)
            self.assertEqual(row["context"], "Sakura speaking to Naruto")
            self.assertEqual((row["source_lang"], row["target_lang"]), ("en", "pt-BR"))
            self.assertEqual(history, [])
            self.assertIsNone(visual_hit)

    def test_translation_memory_import_conflict_policies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_memory = TranslationMemory(Path(tmp) / "export.sqlite3")
            export_memory.save("Good morning", "Bom dia novo", "en->pt-BR")
            export_path = Path(tmp) / "memory.json"
            export_memory.export_json(export_path)

            target = TranslationMemory(Path(tmp) / "target.sqlite3")
            target.save("Good morning", "Bom dia anterior", "en->pt-BR")

            kept = target.import_json(export_path, conflict="keep")
            kept_text = target.lookup("Good morning", "en->pt-BR").translated_text
            replaced = target.import_json(export_path)
            replaced_text = target.lookup("Good morning", "en->pt-BR").translated_text
            export_memory.close()
            target.close()
            self.assertEqual(kept.kept_existing, 1)
            self.assertEqual(kept_text, "Bom dia anterior")
            self.assertEqual(replaced.updated, 1)
            self.assertEqual(replaced_text, "Bom dia novo")

    def test_translation_memory_import_validates_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "memory.sqlite3")
            path = Path(tmp) / "invalid.json"
            valid_entry = {
                "original_text": "This must not be partly imported",
                "translated_text": "Não deve entrar parcialmente",
                "source_lang": "en",
                "target_lang": "pt-BR",
                "language_pair": "en->pt-BR",
                "frequency_count": 1,
                "context": None,
                "created_at": 1,
                "updated_at": 1,
            }
            path.write_text(
                json.dumps({
                    "format": TranslationMemory.EXPORT_FORMAT,
                    "version": TranslationMemory.EXPORT_VERSION,
                    "entries": [valid_entry, {**valid_entry, "frequency_count": True}],
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "frequency_count"):
                memory.import_json(path)
            count = memory.conn.execute("select count(*) from translations").fetchone()[0]
            memory.close()
            self.assertEqual(count, 0)

    def test_translation_memory_import_rolls_back_sql_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = TranslationMemory(Path(tmp) / "source.sqlite3")
            source.save("A new entry", "Uma entrada nova", "en->pt-BR")
            source.save("Existing entry", "Uma entrada atualizada", "en->pt-BR")
            export_path = Path(tmp) / "memory.json"
            source.export_json(export_path)

            target = TranslationMemory(Path(tmp) / "target.sqlite3")
            target.save("Existing entry", "Tradução antiga", "en->pt-BR")
            target.conn.execute(
                """
                create trigger fail_memory_import before update on translations
                when new.original_text = 'Existing entry'
                begin select raise(abort, 'simulated import failure'); end
                """
            )
            failed = False
            try:
                target.import_json(export_path)
            except sqlite3.IntegrityError:
                failed = True
            rows = target.conn.execute(
                "select original_text, translated_text from translations order by original_text"
            ).fetchall()
            source.close()
            target.close()
            self.assertTrue(failed)
            self.assertEqual(
                [(row["original_text"], row["translated_text"]) for row in rows],
                [("Existing entry", "Tradução antiga")],
            )

    def test_exact_lookup_and_frequency_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "tm.sqlite3")
            memory.save("Damn it!", "Droga!", "en->pt-BR", "en", "pt-BR")

            hit = memory.lookup("damn it!", "en->pt-BR")

            self.assertIsNotNone(hit)
            self.assertEqual(hit.translated_text, "Droga!")
            self.assertEqual(hit.source, "exact")
            memory.close()

    def test_fuzzy_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "tm.sqlite3", fuzzy_threshold=85)
            memory.save("I will never forgive you", "Eu nunca vou te perdoar", "en->pt-BR")

            hit = memory.lookup("I will never forgive ya", "en->pt-BR")

            self.assertIsNotNone(hit)
            self.assertEqual(hit.translated_text, "Eu nunca vou te perdoar")
            self.assertEqual(hit.source, "fuzzy")
            memory.close()

    def test_normalization(self) -> None:
        self.assertEqual(normalize_text("  HELLO   WORLD  "), "hello world")

    def test_translation_history_search_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "history.sqlite3")
            memory.record_history("Hello, hero", "Olá, herói", "en->pt-BR", "google")
            memory.record_history("Goodbye", "Até logo", "en->pt-BR", "deepl")

            recent = memory.history()
            assert [entry.original_text for entry in recent] == ["Goodbye", "Hello, hero"]
            assert memory.history("herói")[0].original_text == "Hello, hero"
            assert memory.history("%") == []

            memory.close()

    def test_translation_history_ignores_echo_and_blank_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "history.sqlite3")
            memory.record_history("Hello", "Hello", "en->pt-BR", "echo")
            memory.record_history(" ", "Olá", "en->pt-BR", "google")
            assert memory.history() == []
            memory.close()

    def test_manual_history_edit_updates_text_and_visual_memory_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "edited-history.sqlite3")
            memory.save("You are amazing", "Você é incrível", "en->pt-BR", "en", "pt-BR")
            memory.save("You are amazing", "Você é ótimo", "en->es", "en", "es")
            memory.save_visual(
                "manga-bubble-en", "You are amazing", "Você é incrível",
                "en->pt-BR", (10, 20, 90, 40),
            )
            memory.save_visual(
                "manga-bubble-es", "You are amazing", "Você é ótimo",
                "en->es", (10, 20, 90, 40),
            )
            memory.record_history(
                "You are amazing", "Você é incrível", "en->pt-BR", "google"
            )
            entry = memory.history()[0]

            assert memory.edit_history_translation(entry.id, "Você é sensacional!")
            assert memory.lookup("You are amazing", "en->pt-BR").translated_text == "Você é sensacional!"
            assert memory.get_visual("manga-bubble-en", "en->pt-BR").translated_text == "Você é sensacional!"
            assert memory.lookup("You are amazing", "en->es").translated_text == "Você é ótimo"
            edited = memory.history()[0]
            assert edited.translated_text == "Você é sensacional!"
            assert edited.provider == "manual"
            assert not memory.edit_history_translation(-1, "não existe")
            assert not memory.edit_history_translation(entry.id, "  ")
            memory.close()

    def test_manual_history_edit_rolls_back_cache_when_visual_update_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "edit-rollback.sqlite3")
            memory.save("Stop right there", "Pare agora", "en->pt-BR")
            memory.save_visual(
                "stop-bubble", "Stop right there", "Pare agora",
                "en->pt-BR", (1, 2, 80, 30),
            )
            memory.record_history(
                "Stop right there", "Pare agora", "en->pt-BR", "google"
            )
            entry = memory.history()[0]
            memory.conn.execute(
                """
                create trigger fail_visual_edit before update on visual_regions
                begin select raise(abort, 'simulated failure'); end
                """
            )
            with self.assertRaises(sqlite3.IntegrityError):
                memory.edit_history_translation(entry.id, "Pare já!")
            assert memory.lookup("Stop right there", "en->pt-BR").translated_text == "Pare agora"
            assert memory.get_visual("stop-bubble", "en->pt-BR").translated_text == "Pare agora"
            assert memory.history()[0].translated_text == "Pare agora"
            memory.close()

    def test_glossary_change_invalidates_only_matching_cache_and_keeps_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "glossary.sqlite3")
            memory.save("Sakura is here", "Sakura está aqui", "en->pt-BR")
            memory.save("The sky is blue", "O céu é azul", "en->pt-BR")
            memory.save_visual(
                "sakura-hash", "Sakura is here", "Sakura está aqui",
                "en->pt-BR", (1, 2, 30, 20),
            )
            memory.save_visual(
                "sky-hash", "The sky is blue", "O céu é azul",
                "en->pt-BR", (40, 2, 30, 20),
            )
            memory.record_history("Sakura is here", "Sakura está aqui", "en->pt-BR", "google")

            invalidated = memory.invalidate_glossary_cache(["Sakura"])

            assert invalidated == 1
            assert memory.lookup("Sakura is here", "en->pt-BR") is None
            assert memory.get_visual("sakura-hash", "en->pt-BR") is None
            assert memory.lookup("The sky is blue", "en->pt-BR").translated_text == "O céu é azul"
            assert memory.get_visual("sky-hash", "en->pt-BR") is not None
            assert len(memory.history()) == 1
            memory.close()

    def test_general_glossary_rule_change_clears_reusable_cache_not_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "glossary-instructions.sqlite3")
            memory.save("Good morning", "Bom dia", "en->pt-BR")
            memory.save_visual(
                "morning-hash", "Good morning", "Bom dia", "en->pt-BR", (1, 2, 30, 20)
            )
            memory.record_history("Good morning", "Bom dia", "en->pt-BR", "google")

            invalidated = memory.invalidate_glossary_cache([], clear_all=True)

            assert invalidated == 1
            assert memory.lookup("Good morning", "en->pt-BR") is None
            assert memory.get_visual("morning-hash", "en->pt-BR") is None
            assert len(memory.history()) == 1
            memory.close()

    def test_scoped_glossary_invalidation_does_not_touch_other_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = TranslationMemory(Path(tmp) / "scoped-glossary.sqlite3")
            pair = "work:abc|en->pt-BR"
            memory.save("guild hall", "salão da associação", pair)
            memory.save("guild hall", "salão da guilda", "en->pt-BR")
            memory.save_visual("work-bubble", "guild hall", "salão da associação", pair, (1, 2, 30, 20))
            memory.save_visual("global-bubble", "guild hall", "salão da guilda", "en->pt-BR", (1, 2, 30, 20))

            invalidated = memory.invalidate_glossary_cache(
                ["guild"], language_pair_prefix="work:abc|"
            )

            assert invalidated == 1
            assert memory.lookup("guild hall", pair) is None
            assert memory.get_visual("work-bubble", pair) is None
            assert memory.lookup("guild hall", "en->pt-BR").translated_text == "salão da guilda"
            assert memory.get_visual("global-bubble", "en->pt-BR") is not None
            memory.close()


if __name__ == "__main__":
    unittest.main()
