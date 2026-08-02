"""
database/run_migrations.py — database/migrations/*.sql için basit takip aracı
(Fable 5 mimari denetimi madde 5, 2 Ağu 2026: 34 dosya, hiçbir "uygulandı"
takip tablosu yoktu).

Alembic değil — mevcut dosyalara dokunmadan, hangisinin uygulandığını
schema_migrations tablosunda kaydeder. asyncpg tek execute() çağrısında
çoklu SQL ifadesini desteklemediği için (çoğu migration dosyası çok-ifadeli)
psycopg2 kullanılır — run_migration_performance.py ile aynı bağlantı deseni.

İki mod:
  --baseline : Mevcut TÜM .sql dosyalarını ÇALIŞTIRMADAN "uygulanmış" olarak
               işaretler. Bu dosyalar zaten canlıda elle uygulanmıştı — bazıları
               (ör. rename_deviso_to_devisso.sql) RENAME/DROP içerdiği için
               IF NOT EXISTS gibi korumaları yok, tekrar çalıştırmak canlıda
               hataya yol açabilir. TEK SEFERLİK bootstrap adımı.
  (argümansız): schema_migrations'da kaydı OLMAYAN dosyaları gerçekten
               çalıştırıp kaydeder — bundan sonra eklenecek YENİ migration'lar
               için normal akış budur.

Kullanım:
    python -m database.run_migrations --baseline   # bir kez, şimdi
    python -m database.run_migrations              # yeni migration eklendikçe
"""

import sys
from pathlib import Path

import psycopg2

from config import Config

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _connect():
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )


def _ensure_tracking_table(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )


def _applied_filenames(cursor) -> set:
    cursor.execute("SELECT filename FROM schema_migrations")
    return {row[0] for row in cursor.fetchall()}


def _sql_files() -> list:
    return sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))


def baseline() -> list:
    """Mevcut tüm .sql dosyalarını ÇALIŞTIRMADAN uygulanmış işaretler."""
    conn = _connect()
    conn.autocommit = False
    try:
        cursor = conn.cursor()
        _ensure_tracking_table(cursor)
        conn.commit()

        applied = _applied_filenames(cursor)
        to_stamp = [f for f in _sql_files() if f not in applied]
        for filename in to_stamp:
            cursor.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (filename,))
        conn.commit()
        print(f"{len(to_stamp)} dosya baseline olarak işaretlendi (çalıştırılmadı): {to_stamp}")
        return to_stamp
    finally:
        conn.close()


def run_migrations() -> list:
    """schema_migrations'da kaydı olmayan dosyaları GERÇEKTEN çalıştırıp kaydeder."""
    conn = _connect()
    conn.autocommit = False
    newly_applied: list = []
    try:
        cursor = conn.cursor()
        _ensure_tracking_table(cursor)
        conn.commit()

        applied = _applied_filenames(cursor)
        for filename in _sql_files():
            if filename in applied:
                continue
            sql = (MIGRATIONS_DIR / filename).read_text(encoding="utf-8")
            print(f"Uygulanıyor: {filename}")
            try:
                cursor.execute(sql)
                cursor.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (filename,))
                conn.commit()
                newly_applied.append(filename)
            except Exception as exc:
                conn.rollback()
                print(f"BAŞARISIZ: {filename} — {exc}", file=sys.stderr)
                raise

        if newly_applied:
            print(f"{len(newly_applied)} migration uygulandı: {newly_applied}")
        else:
            print("Uygulanacak yeni migration yok — hepsi güncel.")
        return newly_applied
    finally:
        conn.close()


if __name__ == "__main__":
    if "--baseline" in sys.argv:
        baseline()
    else:
        run_migrations()
