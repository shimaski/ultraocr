from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path


INVALID_WHERE = """
language_pair like '%->pt-BR'
and lower(trim(original_text)) = lower(trim(translated_text))
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove source-language fallbacks saved as PT-BR translations."
    )
    parser.add_argument("database", nargs="?", default="manga_translations.sqlite3")
    parser.add_argument("--apply", action="store_true", help="Back up and repair the DB")
    args = parser.parse_args()

    database = Path(args.database).resolve()
    if not database.exists():
        parser.error(f"database not found: {database}")

    with sqlite3.connect(database) as connection:
        translation_count = connection.execute(
            f"select count(*) from translations where {INVALID_WHERE}"
        ).fetchone()[0]
        visual_count = connection.execute(
            f"select count(*) from visual_regions where {INVALID_WHERE}"
        ).fetchone()[0]

        print(f"invalid translations: {translation_count}")
        print(f"invalid visual regions: {visual_count}")
        if not args.apply:
            print("dry run only; pass --apply to repair")
            return 0

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = database.with_name(
            f"{database.stem}.backup-{stamp}{database.suffix}"
        )
        with sqlite3.connect(backup_path) as backup:
            connection.backup(backup)

        connection.execute(
            f"""
            delete from translations_fts
            where rowid in (
                select id from translations where {INVALID_WHERE}
            )
            """
        )
        connection.execute(f"delete from translations where {INVALID_WHERE}")
        connection.execute(f"delete from visual_regions where {INVALID_WHERE}")
        connection.commit()

    print(f"backup: {backup_path}")
    print("repair complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
