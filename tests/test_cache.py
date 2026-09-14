from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manga_rt_translator.cache import TranslationMemory, normalize_text


class TranslationMemoryTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
