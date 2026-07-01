#!/bin/bash
# =============================================================================
# DAILY SIGNAL PERFORMANCE UPDATE
# =============================================================================
# Her gece çalışarak signal_performance verilerini günceller
# Kullanım: ./scripts/daily_performance_update.sh

set -e  # Hata olursa dur

# Renkli output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script dizini
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Log dosyası
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_performance_update_$(date +%Y%m%d).log"

# Log fonksiyonu
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

# =============================================================================
# MAIN
# =============================================================================

log "=========================================="
log "Daily Signal Performance Update Started"
log "=========================================="

# Virtual environment'ı aktif et
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    log "Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
else
    log_error "Virtual environment not found at $PROJECT_DIR/.venv"
    exit 1
fi

# Python script'i çalıştır
log "Running batch update..."
cd "$PROJECT_DIR"

python -m scripts.update_signal_performance \
    --hours-back 720 \
    --max-signals 1000 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    log_success "Batch update completed successfully"
else
    log_error "Batch update failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Eski logları temizle (30 günden eski)
log "Cleaning old logs..."
find "$LOG_DIR" -name "daily_performance_update_*.log" -mtime +30 -delete
log_success "Old logs cleaned"

log "=========================================="
log "Daily Signal Performance Update Finished"
log "=========================================="

exit 0
