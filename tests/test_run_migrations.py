"""
database/run_migrations.py — saf dosya-tarama mantığı için birim testi.

baseline()/run_migrations() gerçek DB'ye karşı elle doğrulandı (2 Ağu 2026):
34 dosya --baseline ile işaretlendi, tekrar çalıştırınca "yeni migration yok"
doğru raporlandı. Burada sadece DB'ye dokunmayan _sql_files() test edilir.
"""

from database.run_migrations import MIGRATIONS_DIR, _sql_files


def test_sql_files_returns_sorted_filenames_only():
    files = _sql_files()
    assert files == sorted(files)
    assert all(f.endswith(".sql") for f in files)


def test_sql_files_matches_real_migrations_dir():
    on_disk = {p.name for p in MIGRATIONS_DIR.glob("*.sql")}
    assert set(_sql_files()) == on_disk
