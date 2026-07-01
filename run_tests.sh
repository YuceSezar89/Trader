#!/bin/bash
# =============================================================================
# Test Runner Script
# =============================================================================
# Tüm testleri çalıştırır ve sonuçları raporlar.
#
# Kullanım:
#   ./run_tests.sh                    # Tüm testler
#   ./run_tests.sh unit               # Sadece unit testler
#   ./run_tests.sh integration        # Sadece integration testler
#   ./run_tests.sh performance        # Sadece performance testler
#   ./run_tests.sh database           # Sadece database testler
#   ./run_tests.sh signal_performance # Sadece signal performance testler
# =============================================================================

set -e  # Exit on error

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo ""
echo "=========================================="
echo "  TRader Panel Test Suite"
echo "=========================================="
echo ""

# Test tipi seç
TEST_MARKER="${1:-all}"

case $TEST_MARKER in
    all)
        echo -e "${BLUE}🧪 Tüm testler çalıştırılıyor...${NC}"
        pytest tests/ -v --tb=short
        ;;
    unit)
        echo -e "${BLUE}🧪 Unit testler çalıştırılıyor...${NC}"
        pytest tests/ -v --tb=short -m "unit"
        ;;
    integration)
        echo -e "${BLUE}🧪 Integration testler çalıştırılıyor...${NC}"
        pytest tests/ -v --tb=short -m "integration"
        ;;
    performance)
        echo -e "${BLUE}🧪 Performance testler çalıştırılıyor...${NC}"
        pytest tests/ -v --tb=short -m "performance"
        ;;
    database)
        echo -e "${BLUE}🧪 Database testler çalıştırılıyor...${NC}"
        pytest tests/ -v --tb=short -m "database"
        ;;
    signal_performance)
        echo -e "${BLUE}🧪 Signal Performance testler çalıştırılıyor...${NC}"
        pytest tests/test_signal_performance.py -v --tb=short
        ;;
    coverage)
        echo -e "${BLUE}🧪 Coverage raporu oluşturuluyor...${NC}"
        pytest tests/ -v --tb=short --cov=utils --cov=signals --cov=database --cov-report=html --cov-report=term-missing
        echo ""
        echo -e "${GREEN}✅ Coverage raporu: htmlcov/index.html${NC}"
        ;;
    quick)
        echo -e "${BLUE}🧪 Hızlı testler çalıştırılıyor (slow hariç)...${NC}"
        pytest tests/ -v --tb=short -m "not slow"
        ;;
    *)
        echo -e "${RED}❌ Geçersiz test tipi: $TEST_MARKER${NC}"
        echo ""
        echo "Kullanılabilir seçenekler:"
        echo "  all                - Tüm testler"
        echo "  unit               - Unit testler"
        echo "  integration        - Integration testler"
        echo "  performance        - Performance testler"
        echo "  database           - Database testler"
        echo "  signal_performance - Signal performance testler"
        echo "  coverage           - Coverage raporu"
        echo "  quick              - Hızlı testler (slow hariç)"
        exit 1
        ;;
esac

# Test sonucu
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo -e "  ✅ Testler Başarılı!"
    echo -e "==========================================${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}=========================================="
    echo -e "  ❌ Testler Başarısız!"
    echo -e "==========================================${NC}"
    echo ""
    exit 1
fi
