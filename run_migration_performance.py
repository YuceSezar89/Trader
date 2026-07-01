#!/usr/bin/env python3
"""
Signal Performance Migration Runner
====================================
signal_performance tablosunu oluşturur.

Kullanım:
    python run_migration_performance.py
"""

import sys
import psycopg2
from pathlib import Path

# Config'i import et
from config import Config


def run_migration():
    """Migration'ı çalıştır."""
    migration_file = Path(__file__).parent / "database" / "migrations" / "add_signal_performance.sql"
    
    if not migration_file.exists():
        print(f"❌ Migration dosyası bulunamadı: {migration_file}")
        sys.exit(1)
    
    print(f"📄 Migration dosyası: {migration_file}")
    print(f"🔗 Database: {Config.DB_NAME}")
    print()
    
    # SQL'i oku
    with open(migration_file, "r", encoding="utf-8") as f:
        sql = f.read()
    
    # Database'e bağlan
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("🔄 Migration çalıştırılıyor...")
        print()
        
        # SQL'i çalıştır
        cursor.execute(sql)
        
        print()
        print("=" * 60)
        print("✅ Migration başarıyla tamamlandı!")
        print("=" * 60)
        print()
        
        # Tablo bilgilerini göster
        cursor.execute("""
            SELECT 
                column_name, 
                data_type, 
                is_nullable
            FROM 
                information_schema.columns
            WHERE 
                table_name = 'signal_performance'
            ORDER BY 
                ordinal_position;
        """)
        
        columns = cursor.fetchall()
        print("📊 signal_performance Kolonları:")
        print("-" * 60)
        for col_name, col_type, nullable in columns:
            null_str = "NULL" if nullable == "YES" else "NOT NULL"
            print(f"  • {col_name:25} {col_type:20} {null_str}")
        
        print()
        
        # Index bilgilerini göster
        cursor.execute("""
            SELECT 
                indexname, 
                indexdef
            FROM 
                pg_indexes
            WHERE 
                tablename = 'signal_performance';
        """)
        
        indexes = cursor.fetchall()
        print("🔍 İndexler:")
        print("-" * 60)
        for idx_name, idx_def in indexes:
            print(f"  • {idx_name}")
        
        print()
        print("🎯 Sonraki Adım:")
        print("  python run_performance_update.py --test")
        print()
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Database hatası: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  SIGNAL PERFORMANCE MIGRATION")
    print("=" * 60)
    print()
    
    run_migration()
