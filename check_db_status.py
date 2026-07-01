#!/usr/bin/env python3
"""
Database Status Checker
========================
Mevcut database durumunu kontrol eder.
"""

import psycopg2
from config import Config


def check_database():
    """Database bağlantısını ve tabloları kontrol et."""
    print("=" * 60)
    print("  DATABASE STATUS CHECK")
    print("=" * 60)
    print()
    
    print(f"🔗 Bağlantı Bilgileri:")
    print(f"  Host: {Config.DB_HOST}")
    print(f"  Port: {Config.DB_PORT}")
    print(f"  Database: {Config.DB_NAME}")
    print(f"  User: {Config.DB_USER}")
    print()
    
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        print("✅ Database bağlantısı başarılı!")
        print()
        
        # Tabloları listele
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        print(f"📊 Mevcut Tablolar ({len(tables)}):")
        print("-" * 60)
        for table in tables:
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table[0]};
            """)
            count = cursor.fetchone()[0]
            print(f"  • {table[0]:30} {count:>10} rows")
        print()
        
        # signal_performance tablosu var mı?
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'signal_performance'
            );
        """)
        perf_exists = cursor.fetchone()[0]
        
        if perf_exists:
            print("✅ signal_performance tablosu mevcut")
            
            # Kolon sayısı
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.columns
                WHERE table_name = 'signal_performance';
            """)
            col_count = cursor.fetchone()[0]
            print(f"   Kolon sayısı: {col_count}")
            
            # Kayıt sayısı
            cursor.execute("SELECT COUNT(*) FROM signal_performance;")
            row_count = cursor.fetchone()[0]
            print(f"   Kayıt sayısı: {row_count}")
        else:
            print("⚠️  signal_performance tablosu YOK")
            print("   Migration çalıştırılmalı: python run_migration_performance.py")
        
        print()
        
        # View kontrolü
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.views 
                WHERE table_name = 'signal_quality_summary'
            );
        """)
        view_exists = cursor.fetchone()[0]
        
        if view_exists:
            print("✅ signal_quality_summary view mevcut")
        else:
            print("⚠️  signal_quality_summary view YOK")
        
        cursor.close()
        conn.close()
        
        print()
        print("=" * 60)
        
    except psycopg2.OperationalError as e:
        print(f"❌ Bağlantı hatası: {e}")
        print()
        print("Kontrol listesi:")
        print("  1. PostgreSQL çalışıyor mu? (pg_isready)")
        print("  2. Database oluşturuldu mu? (CREATE DATABASE trader_panel)")
        print("  3. Kullanıcı yetkisi var mı?")
        print("  4. .env dosyası doğru mu?")
    except Exception as e:
        print(f"❌ Hata: {e}")


if __name__ == "__main__":
    check_database()
