#!/bin/bash

# TRader Panel Sistem Kontrol ve Başlatma Scripti
# Bu script sistem yeniden başlatıldığında çalıştırılmalıdır

echo "🚀 TRader Panel Sistem Kontrolü Başlatılıyor..."
echo "=================================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Hata sayacı
ERROR_COUNT=0

# 1. PostgreSQL Servis Kontrolü
echo -e "${BLUE}📊 PostgreSQL Servis Kontrolü...${NC}"
if brew services list | grep -q "postgresql@17.*started"; then
    echo -e "${GREEN}✅ PostgreSQL@17 çalışıyor${NC}"
else
    echo -e "${YELLOW}⚠️  PostgreSQL@17 çalışmıyor, başlatılıyor...${NC}"
    brew services start postgresql@17
    sleep 3
    if brew services list | grep -q "postgresql@17.*started"; then
        echo -e "${GREEN}✅ PostgreSQL@17 başarıyla başlatıldı${NC}"
    else
        echo -e "${RED}❌ PostgreSQL@17 başlatılamadı${NC}"
        ((ERROR_COUNT++))
    fi
fi

# 2. Port 5432 Kontrolü
echo -e "${BLUE}🔌 Port 5432 Kontrolü...${NC}"
if lsof -i :5432 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Port 5432 aktif${NC}"
else
    echo -e "${RED}❌ Port 5432 kapalı${NC}"
    ((ERROR_COUNT++))
fi

# 3. Database Bağlantı Testi
echo -e "${BLUE}🗄️  Database Bağlantı Testi...${NC}"
cd /Users/yusuf/Documents/TRader/CascadeProjects/TRader-Panel-ASYNC

# Virtual environment kontrolü
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment bulunamadı${NC}"
    ((ERROR_COUNT++))
else
    source .venv/bin/activate
    
    # Database bağlantı testi
    DB_TEST=$(python -c "
import asyncio
from database.engine import async_engine
from sqlalchemy import text
async def test_db():
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text('SELECT 1;'))
        return True
    except Exception as e:
        print(f'DB_ERROR: {e}')
        return False
print('SUCCESS' if asyncio.run(test_db()) else 'FAILED')
" 2>&1)

    if echo "$DB_TEST" | grep -q "SUCCESS"; then
        echo -e "${GREEN}✅ Database bağlantısı başarılı${NC}"
    else
        echo -e "${RED}❌ Database bağlantı hatası: $DB_TEST${NC}"
        ((ERROR_COUNT++))
    fi
fi

# 4. Environment Variables Kontrolü
echo -e "${BLUE}🔧 Environment Variables Kontrolü...${NC}"
if [ -n "$DATABASE_URL" ]; then
    echo -e "${YELLOW}⚠️  DATABASE_URL environment variable tanımlı: $DATABASE_URL${NC}"
    echo -e "${YELLOW}   Bu kod içindeki ayarları override edebilir${NC}"
else
    echo -e "${GREEN}✅ DATABASE_URL environment variable temiz${NC}"
fi

# 5. Sistem Durumu Özeti
echo ""
echo "=================================================="
echo -e "${BLUE}📋 SİSTEM DURUMU ÖZETİ${NC}"
echo "=================================================="

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}🎉 TÜM KONTROLLER BAŞARILI!${NC}"
    echo -e "${GREEN}✅ Sistem çalışmaya hazır${NC}"
    echo ""
    echo -e "${BLUE}🚀 Sistemi başlatmak için:${NC}"
    echo "   python run_services.py"
    echo ""
    echo -e "${BLUE}📊 Panel'i açmak için:${NC}"
    echo "   streamlit run streamlit_app.py --server.port 8501"
    exit 0
else
    echo -e "${RED}❌ $ERROR_COUNT HATA TESPİT EDİLDİ${NC}"
    echo -e "${RED}⚠️  Sistem çalışmaya hazır değil${NC}"
    echo ""
    echo -e "${YELLOW}🔧 Önerilen Çözümler:${NC}"
    echo "1. PostgreSQL'i manuel başlat: brew services start postgresql@17"
    echo "2. Port çakışması varsa: lsof -i :5432"
    echo "3. Database bağlantısını kontrol et"
    exit 1
fi
