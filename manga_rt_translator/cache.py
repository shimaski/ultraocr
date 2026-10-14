from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
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


@dataclass(frozen=True)
class TranslationHistoryEntry:
    id: int
    created_at: int
    original_text: str
    translated_text: str
    language_pair: str
    provider: str


@dataclass(frozen=True)
class TranslationMemoryImportResult:
    inserted: int
    updated: int
    kept_existing: int

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.kept_existing


class TranslationMemory:
    EXPORT_FORMAT = "manga-rt-translator.translation-memory"
    EXPORT_VERSION = 1
    MAX_IMPORT_BYTES = 50 * 1024 * 1024
    MAX_IMPORT_ENTRIES = 100_000
    MAX_ENTRY_TEXT_LENGTH = 100_000

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

    def export_json(self, destination: str | Path) -> int:
        """Write the textual translation memory as an atomic, versioned JSON file."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = self.conn.execute(
                """
                select original_text, translated_text, source_lang, target_lang,
                       language_pair, frequency_count, context, created_at, updated_at
                from translations
                order by language_pair, normalized_text
                """
            ).fetchall()
            payload = {
                "format": self.EXPORT_FORMAT,
                "version": self.EXPORT_VERSION,
                "exported_at": int(time.time()),
                "entries": [dict(row) for row in rows],
            }

        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                json.dump(payload, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, target)
        except OSError:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
            raise
        return len(rows)

    def import_json(
        self,
        source: str | Path,
        *,
        conflict: str = "replace",
    ) -> TranslationMemoryImportResult:
        """Merge a validated export; existing entries are replaced or retained."""
        if conflict not in {"replace", "keep"}:
            raise ValueError("conflict must be 'replace' or 'keep'")

        source_path = Path(source)
        if source_path.stat().st_size > self.MAX_IMPORT_BYTES:
            raise ValueError("Translation memory export is too large")
        with source_path.open("r", encoding="utf-8-sig") as stream:
            payload = json.load(stream)
        entries = self._validate_export(payload)

        inserted = updated = kept_existing = 0
        now = int(time.time())
        with self._lock:
            self.conn.execute("begin immediate")
            try:
                for entry in entries:
                    existing = self.conn.execute(
                        """
                        select id, frequency_count, created_at
                        from translations
                        where language_pair = ? and normalized_text = ?
                        """,
                        (entry["language_pair"], entry["normalized_text"]),
                    ).fetchone()
                    if existing is not None and conflict == "keep":
                        kept_existing += 1
                        continue

                    if existing is None:
                        cursor = self.conn.execute(
                            """
                            insert into translations(
                              original_text, normalized_text, translated_text,
                              source_lang, target_lang, language_pair,
                              frequency_count, context, created_at, updated_at
                            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                entry["original_text"], entry["normalized_text"],
                                entry["translated_text"], entry["source_lang"],
                                entry["target_lang"], entry["language_pair"],
                                entry["frequency_count"], entry["context"],
                                entry["created_at"], entry["updated_at"],
                            ),
                        )
                        translation_id = int(cursor.lastrowid)
                        inserted += 1
                    else:
                        translation_id = int(existing["id"])
                        created_at = min(
                            int(existing["created_at"]), entry["created_at"]
                        )
                        frequency_count = min(
                            9_223_372_036_854_775_807,
                            int(existing["frequency_count"]) + entry["frequency_count"],
                        )
                        self.conn.execute(
                            """
                            update translations set
                              original_text = ?, translated_text = ?, source_lang = ?,
                              target_lang = ?, frequency_count = ?, context = ?,
                              created_at = ?, updated_at = ?
                            where id = ?
                            """,
                            (
                                entry["original_text"], entry["translated_text"],
                                entry["source_lang"], entry["target_lang"],
                                frequency_count, entry["context"], created_at,
                                now, translation_id,
                            ),
                        )
                        updated += 1

                    self.conn.execute(
                        "delete from translations_fts where rowid = ?",
                        (translation_id,),
                    )
                    self.conn.execute(
                        """
                        insert into translations_fts(
                          rowid, normalized_text, translated_text, language_pair
                        ) values (?, ?, ?, ?)
                        """,
                        (
                            translation_id, entry["normalized_text"],
                            entry["translated_text"], entry["language_pair"],
                        ),
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return TranslationMemoryImportResult(inserted, updated, kept_existing)

    @classmethod
    def _validate_export(cls, payload: object) -> list[dict[str, object]]:
        if not isinstance(payload, dict):
            raise ValueError("Invalid translation memory export")
        if payload.get("format") != cls.EXPORT_FORMAT:
            raise ValueError("Unsupported translation memory format")
        if type(payload.get("version")) is not int or payload["version"] != cls.EXPORT_VERSION:
            raise ValueError("Unsupported translation memory version")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list) or len(raw_entries) > cls.MAX_IMPORT_ENTRIES:
            raise ValueError("Invalid translation memory entries")

        entries: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        max_integer = 9_223_372_036_854_775_807
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise ValueError("Each translation memory entry must be an object")

            original = raw.get("original_text")
            translated = raw.get("translated_text")
            language_pair = raw.get("language_pair")
            source_lang = raw.get("source_lang")
            target_lang = raw.get("target_lang")
            context = raw.get("context")
            frequency = raw.get("frequency_count")
            created_at = raw.get("created_at")
            updated_at = raw.get("updated_at")

            for field, value in (("source_lang", source_lang), ("target_lang", target_lang), ("context", context)):
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{field} must be a string or null")
                if isinstance(value, str) and len(value) > cls.MAX_ENTRY_TEXT_LENGTH:
                    raise ValueError(f"{field} exceeds the maximum length")
            if not isinstance(original, str) or not original.strip():
                raise ValueError("original_text must not be empty")
            if not isinstance(translated, str) or not translated.strip():
                raise ValueError("translated_text must not be empty")
            if len(original) > cls.MAX_ENTRY_TEXT_LENGTH or len(translated) > cls.MAX_ENTRY_TEXT_LENGTH:
                raise ValueError("Translation memory text exceeds the maximum length")
            if not isinstance(language_pair, str) or not language_pair.strip() or len(language_pair) > 64:
                raise ValueError("language_pair must be a non-empty string of at most 64 characters")
            if type(frequency) is not int or not 1 <= frequency <= 1_000_000_000:
                raise ValueError("frequency_count must be a positive integer")
            for field, value in (("created_at", created_at), ("updated_at", updated_at)):
                if type(value) is not int or not 0 <= value <= max_integer:
                    raise ValueError(f"{field} must be a non-negative integer")

            normalized = normalize_text(original)
            key = (language_pair, normalized)
            if not normalized or key in seen:
                raise ValueError("Translation memory export contains an empty or duplicate key")
            seen.add(key)
            entries.append(
                {
                    "original_text": original,
                    "normalized_text": normalized,
                    "translated_text": translated,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "language_pair": language_pair,
                    "frequency_count": frequency,
                    "context": context,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        return entries

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

    def record_history(
        self,
        original_text: str,
        translated_text: str,
        language_pair: str,
        provider: str,
    ) -> None:
        self.record_history_batch(
            [(original_text, translated_text, language_pair, provider)]
        )

    def record_history_batch(
        self, entries: list[tuple[str, str, str, str]]
    ) -> None:
        valid_entries = [
            (int(time.time()), original, translated, pair, provider)
            for original, translated, pair, provider in entries
            if original.strip() and translated.strip() and provider != "echo"
        ]
        if not valid_entries:
            return
        with self._lock:
            self.conn.executemany(
                """
                insert into translation_history(
                  created_at, original_text, translated_text, language_pair, provider
                ) values (?, ?, ?, ?, ?)
                """,
                valid_entries,
            )
            self.conn.commit()

    def history(
        self, search: str = "", limit: int = 200
    ) -> list[TranslationHistoryEntry]:
        limit = max(1, min(int(limit), 1000))
        query = search.strip()
        with self._lock:
            if query:
                escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                rows = self.conn.execute(
                    """
                    select id, created_at, original_text, translated_text, language_pair, provider
                    from translation_history
                    where original_text like ? escape '\\'
                       or translated_text like ? escape '\\'
                    order by id desc limit ?
                    """,
                    (f"%{escaped}%", f"%{escaped}%", limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    select id, created_at, original_text, translated_text, language_pair, provider
                    from translation_history order by id desc limit ?
                    """,
                    (limit,),
                ).fetchall()
        return [
            TranslationHistoryEntry(
                id=int(row["id"]),
                created_at=int(row["created_at"]),
                original_text=str(row["original_text"]),
                translated_text=str(row["translated_text"]),
                language_pair=str(row["language_pair"]),
                provider=str(row["provider"]),
            )
            for row in rows
        ]

    def edit_history_translation(self, history_id: int, translated_text: str) -> bool:
        translated_text = translated_text.strip()
        if not translated_text:
            return False
        with self._lock:
            row = self.conn.execute(
                """
                select original_text, language_pair
                from translation_history where id = ?
                """,
                (int(history_id),),
            ).fetchone()
            if row is None:
                return False
            original = str(row["original_text"])
            pair = str(row["language_pair"])
            normalized = normalize_text(original)
            parts = pair.split("->", 1)
            source_lang = parts[0] if len(parts) == 2 else None
            target_lang = parts[1] if len(parts) == 2 else None
            now = int(time.time())
            try:
                self.conn.execute("begin immediate")
                self.conn.execute(
                    """
                    insert into translations(
                      original_text, normalized_text, translated_text, source_lang,
                      target_lang, language_pair, frequency_count, context,
                      created_at, updated_at
                    ) values (?, ?, ?, ?, ?, ?, 1, null, ?, ?)
                    on conflict(language_pair, normalized_text)
                    do update set
                      translated_text = excluded.translated_text,
                      updated_at = excluded.updated_at
                    """,
                    (original, normalized, translated_text, source_lang,
                     target_lang, pair, now, now),
                )
                cache_row = self.conn.execute(
                    """
                    select id from translations
                    where language_pair = ? and normalized_text = ?
                    """,
                    (pair, normalized),
                ).fetchone()
                if cache_row is not None:
                    cache_id = int(cache_row["id"])
                    self.conn.execute(
                        "delete from translations_fts where rowid = ?", (cache_id,)
                    )
                    self.conn.execute(
                        """
                        insert into translations_fts(
                          rowid, normalized_text, translated_text, language_pair
                        ) values (?, ?, ?, ?)
                        """,
                        (cache_id, normalized, translated_text, pair),
                    )
                self.conn.execute(
                    """
                    update visual_regions set translated_text = ?, updated_at = ?
                    where language_pair = ? and normalized_text = ?
                    """,
                    (translated_text, now, pair, normalized),
                )
                self.conn.execute(
                    """
                    update translation_history
                    set translated_text = ?, provider = 'manual'
                    where id = ?
                    """,
                    (translated_text, int(history_id)),
                )
                self.conn.commit()
            except sqlite3.Error:
                self.conn.rollback()
                raise
            return True

    def invalidate_glossary_cache(
        self,
        changed_sources: list[str],
        *,
        clear_all: bool = False,
        language_pair_prefix: str | None = None,
    ) -> int:
        terms = list(dict.fromkeys(
            normalize_text(term) for term in changed_sources if normalize_text(term)
        ))
        if not clear_all and not terms:
            return 0

        def scoped(sql: str) -> tuple[str, tuple[object, ...]]:
            if language_pair_prefix is None:
                return sql, ()
            return (
                sql + " and substr(language_pair, 1, ?) = ?",
                (len(language_pair_prefix), language_pair_prefix),
            )

        with self._lock:
            if clear_all:
                query, params = scoped("select id from translations where 1 = 1")
                row_ids = [row[0] for row in self.conn.execute(query, params)]
                invalidated = len(row_ids)
                if language_pair_prefix is None:
                    self.conn.execute("delete from translations_fts")
                    self.conn.execute("delete from translations")
                    self.conn.execute("delete from visual_regions")
                else:
                    self.conn.executemany(
                        "delete from translations_fts where rowid = ?",
                        ((row_id,) for row_id in row_ids),
                    )
                    self.conn.executemany(
                        "delete from translations where id = ?",
                        ((row_id,) for row_id in row_ids),
                    )
                    query, params = scoped("delete from visual_regions where 1 = 1")
                    self.conn.execute(query, params)
            else:
                row_ids: list[int] = []
                for term in terms:
                    query, scope_params = scoped(
                        "select id from translations where instr(normalized_text, ?) > 0"
                    )
                    row_ids.extend(row[0] for row in self.conn.execute(
                        query, (term, *scope_params)
                    ))
                    query, scope_params = scoped(
                        "delete from visual_regions where instr(normalized_text, ?) > 0"
                    )
                    self.conn.execute(query, (term, *scope_params))
                row_ids = list(dict.fromkeys(row_ids))
                invalidated = len(row_ids)
                self.conn.executemany(
                    "delete from translations_fts where rowid = ?",
                    ((row_id,) for row_id in row_ids),
                )
                self.conn.executemany(
                    "delete from translations where id = ?",
                    ((row_id,) for row_id in row_ids),
                )
            self.conn.commit()
            return invalidated

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

                create table if not exists translation_history (
                  id integer primary key,
                  created_at integer not null,
                  original_text text not null,
                  translated_text text not null,
                  language_pair text not null,
                  provider text not null
                );

                create index if not exists translation_history_created_idx
                on translation_history(created_at desc, id desc);
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
