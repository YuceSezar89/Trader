#!/usr/bin/env python3
"""Veritabanı bağlantısını ve durumunu kontrol eder."""
import asyncio
from database.engine import get_session
from database.models import PriceData, Signal
from sqlalchemy import select, func, text

async def check_database():
    """Veritabanı durumunu kontrol eder."""
    print("=" * 60)
    print("TRADER_PANEL VERİTABANI DURUM KONTROLÜ")
    print("=" * 60)
    
    try:
        async with get_session() as session:
            # Bağlantı testi
            result = await session.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"\n✅ PostgreSQL Bağlantısı: BAŞARILI")
            print(f"   Versiyon: {version[:50]}...")
            
            # Veritabanı adı
            result = await session.execute(text("SELECT current_database()"))
            db_name = result.scalar()
            print(f"\n📊 Veritabanı: {db_name}")
            
            # Tablolar
            result = await session.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """))
            tables = result.scalars().all()
            print(f"\n📋 Tablolar ({len(tables)} adet):")
            for table in tables:
                print(f"   - {table}")
            
            # PriceData kayıt sayısı
            result = await session.execute(select(func.count()).select_from(PriceData))
            price_count = result.scalar()
            print(f"\n💰 price_data tablosu: {price_count:,} kayıt")
            
            # PriceData - sembol bazında
            result = await session.execute(
                select(PriceData.symbol, func.count(PriceData.symbol).label('count'))
                .group_by(PriceData.symbol)
                .order_by(func.count(PriceData.symbol).desc())
                .limit(5)
            )
            top_symbols = result.all()
            if top_symbols:
                print("   En çok veri olan 5 sembol:")
                for symbol, count in top_symbols:
                    print(f"   - {symbol}: {count:,} kayıt")
            
            # Signal kayıt sayısı
            result = await session.execute(select(func.count()).select_from(Signal))
            signal_count = result.scalar()
            print(f"\n🎯 signals tablosu: {signal_count:,} kayıt")
            
            # Signal - aktif/pasif durumu
            result = await session.execute(
                select(Signal.status, func.count(Signal.status).label('count'))
                .group_by(Signal.status)
            )
            status_counts = result.all()
            if status_counts:
                print("   Durum dağılımı:")
                for status, count in status_counts:
                    print(f"   - {status or 'NULL'}: {count:,} kayıt")
            
            # Signal - sinyal türü dağılımı
            result = await session.execute(
                select(Signal.signal_type, func.count(Signal.signal_type).label('count'))
                .where(Signal.status == 'active')
                .group_by(Signal.signal_type)
            )
            type_counts = result.all()
            if type_counts:
                print("\n   Aktif sinyal türleri:")
                for sig_type, count in type_counts:
                    print(f"   - {sig_type}: {count:,} kayıt")
            
            # Son 5 sinyal
            result = await session.execute(
                select(Signal.symbol, Signal.signal_type, Signal.timestamp, Signal.vpms_score)
                .where(Signal.status == 'active')
                .order_by(Signal.timestamp.desc())
                .limit(5)
            )
            recent_signals = result.all()
            if recent_signals:
                print("\n   Son 5 aktif sinyal:")
                for symbol, sig_type, ts, score in recent_signals:
                    score_str = f"{score:.2f}" if score is not None else "N/A"
                    print(f"   - {symbol} {sig_type} | {ts} | VPM: {score_str}")
            
            # TimescaleDB hypertable kontrolü
            result = await session.execute(text("""
                SELECT tablename 
                FROM timescaledb_information.hypertables
                WHERE schemaname = 'public'
            """))
            hypertables = result.scalars().all()
            if hypertables:
                print(f"\n⏰ TimescaleDB Hypertables ({len(hypertables)} adet):")
                for ht in hypertables:
                    print(f"   - {ht}")
            else:
                print("\n⚠️  TimescaleDB hypertables bulunamadı (normal PostgreSQL tabloları kullanılıyor)")
            
            print("\n" + "=" * 60)
            print("✅ VERİTABANI KONTROLÜ TAMAMLANDI")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_database())
