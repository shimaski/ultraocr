from __future__ import annotations

import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranslationHit:
    translated_text: str
    original_text: str
    normalized_text: str
    score: float
    source: str


class TranslationMemory:
    def __init__(self, path: str | Path, fuzzy_threshold: int = 92) -> None:
        self.path = Path(path)
        self.fuzzy_threshold = fuzzy_threshold
        # The async translation thread writes while the capture thread reads;
        # a single connection guarded by a lock keeps both safe.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def lookup(self, text: str, language_pair: str) -> TranslationHit | None:
        with self._lock:
            normalized = normalize_text(text)
            exact = self.conn.execute(
                """
                select id, original_text, normalized_text, translated_text
                from translations
                where language_pair = ? and normalized_text = ?
                limit 1
                """,
                (language_pair, normalized),
            ).fetchone()
            if exact:
                self._increment_frequency(int(exact["id"]))
                hit = TranslationHit(
                    translated_text=str(exact["translated_text"]),
                    original_text=str(exact["original_text"]),
                    normalized_text=str(exact["normalized_text"]),
                    score=100.0,
                    source="exact",
                )
                if _looks_untranslated_for_ptbr(hit, language_pair):
                    return None
                return hit

            return self.lookup_fuzzy(normalized, language_pair)

    def lookup_fuzzy(self, normalized: str, language_pair: str) -> TranslationHit | None:
        with self._lock:
            candidates = self._fts_candidates(normalized, language_pair)
            if not candidates:
                candidates = self.conn.execute(
                    """
                    select id, original_text, normalized_text, translated_text
                    from translations
                    where language_pair = ?
                    order by updated_at desc
                    limit 80
                    """,
                    (language_pair,),
                ).fetchall()

            best = None
            best_score = 0.0
            for row in candidates:
                score = similarity(normalized, str(row["normalized_text"]))
                if score > best_score:
                    best_score = score
                    best = row

            if best is None or best_score < self.fuzzy_threshold:
                return None

            self._increment_frequency(int(best["id"]))
            hit = TranslationHit(
                translated_text=str(best["translated_text"]),
                original_text=str(best["original_text"]),
                normalized_text=str(best["normalized_text"]),
                score=best_score,
                source="fuzzy",
            )
            if _looks_untranslated_for_ptbr(hit, language_pair):
                return None
            return hit

    def save(
        self,
        original_text: str,
        translated_text: str,
        language_pair: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        context: str | None = None,
    ) -> None:
        with self._lock:
            normalized = normalize_text(original_text)
            now = int(time.time())
            self.conn.execute(
                """
                insert into translations(
                  original_text, normalized_text, translated_text, source_lang,
                  target_lang, language_pair, frequency_count, context,
                  created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                on conflict(language_pair, normalized_text)
                do update set
                  translated_text = excluded.translated_text,
                  frequency_count = translations.frequency_count + 1,
                  context = coalesce(excluded.context, translations.context),
                  updated_at = excluded.updated_at
                """,
                (
                    original_text,
                    normalized,
                    translated_text,
                    source_lang,
                    target_lang,
                    language_pair,
                    context,
                    now,
                    now,
                ),
            )
            row = self.conn.execute(
                """
                select id from translations
                where language_pair = ? and normalized_text = ?
                """,
                (language_pair, normalized),
            ).fetchone()
            if row:
                self.conn.execute(
                    "delete from translations_fts where rowid = ?",
                    (int(row["id"]),),
                )
                self.conn.execute(
                    """
                    insert into translations_fts(rowid, normalized_text, translated_text, language_pair)
                    values (?, ?, ?, ?)
                    """,
                    (int(row["id"]), normalized, translated_text, language_pair),
                )
            self.conn.commit()

    def get_visual(self, visual_hash: str, language_pair: str) -> TranslationHit | None:
        if not visual_hash:
            return None
        with self._lock:
            row = self.conn.execute(
                """
                select original_text, normalized_text, translated_text
                from visual_regions
                where visual_hash = ? and language_pair = ?
                limit 1
                """,
                (visual_hash, language_pair),
            ).fetchone()
            if not row:
                return None
            hit = TranslationHit(
                translated_text=str(row["translated_text"]),
                original_text=str(row["original_text"]),
                normalized_text=str(row["normalized_text"]),
                score=100.0,
                source="visual",
            )
            if _looks_untranslated_for_ptbr(hit, language_pair):
                return None
            return hit

    def save_visual(
        self,
        visual_hash: str,
        original_text: str,
        translated_text: str,
        language_pair: str,
        bbox: tuple[int, int, int, int],
    ) -> None:
        if not visual_hash:
            return
        with self._lock:
            now = int(time.time())
            self.conn.execute(
                """
                insert into visual_regions(
                  visual_hash, original_text, normalized_text, translated_text,
                  language_pair, bbox_x, bbox_y, bbox_width, bbox_height,
                  created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(visual_hash, language_pair)
                do update set
                  original_text = excluded.original_text,
                  normalized_text = excluded.normalized_text,
                  translated_text = excluded.translated_text,
                  bbox_x = excluded.bbox_x,
                  bbox_y = excluded.bbox_y,
                  bbox_width = excluded.bbox_width,
                  bbox_height = excluded.bbox_height,
                  updated_at = excluded.updated_at
                """,
                (
                    visual_hash,
                    original_text,
                    normalize_text(original_text),
                    translated_text,
                    language_pair,
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    now,
                    now,
                ),
            )
            self.conn.commit()

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                pragma journal_mode = wal;

                create table if not exists translations (
                  id integer primary key,
                  original_text text not null,
                  normalized_text text not null,
                  translated_text text not null,
                  source_lang text,
                  target_lang text,
                  language_pair text not null,
                  frequency_count integer not null default 1,
                  context text,
                  created_at integer not null,
                  updated_at integer not null,
                  unique(language_pair, normalized_text)
                );

                create virtual table if not exists translations_fts
                using fts5(
                  normalized_text,
                  translated_text,
                  language_pair,
                  tokenize='trigram'
                );

                create table if not exists visual_regions (
                  id integer primary key,
                  visual_hash text not null,
                  original_text text not null,
                  normalized_text text not null,
                  translated_text text not null,
                  language_pair text not null,
                  bbox_x integer not null,
                  bbox_y integer not null,
                  bbox_width integer not null,
                  bbox_height integer not null,
                  created_at integer not null,
                  updated_at integer not null,
                  unique(visual_hash, language_pair)
                );
                """
            )
            self.conn.commit()

    def _increment_frequency(self, translation_id: int) -> None:
        self.conn.execute(
            """
            update translations
            set frequency_count = frequency_count + 1, updated_at = ?
            where id = ?
            """,
            (int(time.time()), translation_id),
        )
        self.conn.commit()

    def _fts_candidates(self, normalized: str, language_pair: str) -> list[sqlite3.Row]:
        query = _build_fts_query(normalized)
        if not query:
            return []
        try:
            return self.conn.execute(
                """
                select t.id, t.original_text, t.normalized_text, t.translated_text
                from translations_fts
                join translations t on t.id = translations_fts.rowid
                where translations_fts match ?
                  and t.language_pair = ?
                limit 40
                """,
                (query, language_pair),
            ).fetchall()
        except sqlite3.OperationalError:
            return []


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    return text


def similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio

        return float(ratio(left, right))
    except ImportError:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio() * 100.0


def _build_fts_query(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff\u4e00-\u9fff]+", " ", text)
    tokens = [token for token in cleaned.split() if len(token) >= 3]
    return " OR ".join(tokens[:8])


def _looks_untranslated_for_ptbr(hit: TranslationHit, language_pair: str) -> bool:
    if not language_pair.endswith("->pt-BR"):
        return False
    original = (hit.original_text or "").strip()
    translated = (hit.translated_text or "").strip()
    if not original or not translated:
        return False
    if normalize_text(original) != normalize_text(translated):
        return False
    if len(original) <= 4:
        return False
    if " " not in original and original.isalpha():
        return False
    ascii_letters = sum(1 for char in original if "a" <= char.lower() <= "z")
    return ascii_letters >= max(4, len(original) // 3)
