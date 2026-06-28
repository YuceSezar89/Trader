#!/bin/bash
cd /Users/yusuf/Documents/TRader/CascadeProjects/TRader-Panel-ASYNC

# Mevcut tüm servis süreci öldür
for pat in "run_services.py" "live_data_manager.py"; do
    pids=$(pgrep -f "$pat" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Durduruluyor: $pat (PID: $pids)"
        kill -TERM $pids 2>/dev/null
    fi
done

# Kapanmaları için 3 saniye bekle, sonra zorla öldür
sleep 3
for pat in "run_services.py" "live_data_manager.py"; do
    pids=$(pgrep -f "$pat" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Zorla durduruluyor: $pat (PID: $pids)"
        kill -9 $pids 2>/dev/null
    fi
done

# Stale PID dosyasını temizle
rm -f run_services.pid

sleep 1
echo "Servisler başlatılıyor..."
exec .venv/bin/python run_services.py
