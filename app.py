# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
import streamlit as st

st.set_page_config(page_title="TRader Panel", layout="wide")
st.title("TRader Panel - Pes Eden Maldır !!!")

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import json
from datetime import datetime, timedelta

# =============================================================================
# ASYNC HELPER FOR STREAMLIT
# =============================================================================
# Async helper fonksiyonları kaldırıldı - run_async_safely kullanılıyor

# Async helper kaldırıldı - artık sync fonksiyonlar kullanılıyor

# =============================================================================
# THIRD PARTY IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz
import redis
import psycopg2
import psycopg2.extras
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
# Config
from config import Config

# Binance client
from binance_client import BinanceClientManager
from utils.redis_client import RedisClient
from utils.data_provider import fetch_ohlcv

# Database'den veri çekme fonksiyonu (öncelikli)
@st.cache_data(ttl=30)  # 30 saniye cache
def get_price_data_from_db(symbol: str, interval: str, limit: int):
    """Database'den fiyat verisi çeker (hızlı ve real-time)."""
    try:
        import psycopg2
        import psycopg2.extras
        import pandas as pd
        
        # Direct PostgreSQL connection
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="trader_panel",
            user="yusuf",
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        
        cursor = conn.cursor()
        query = """
        SELECT timestamp as open_time, open, high, low, close, volume
        FROM price_data 
        WHERE symbol = %s AND interval = %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        cursor.execute(query, (symbol, interval, limit))
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if not rows:
            print(f"⚠️ Database'de {symbol} {interval} verisi bulunamadı")
            return pd.DataFrame()
        
        # DataFrame'e dönüştür
        df = pd.DataFrame([dict(row) for row in rows])
        
        # Veri tiplerini düzelt
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Timestamp'leri datetime'a çevir (zaten datetime olabilir)
        if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
            df['open_time'] = pd.to_datetime(df['open_time'])
        
        # Sıralamayı düzelt (eski -> yeni)
        df = df.sort_values('open_time').reset_index(drop=True)
        
        # Date sütunu ekle (grafik için gerekli)
        df['date'] = df['open_time']
        
        print(f"✅ Database'den {symbol} verisi çekildi: {len(df)} mum (HIZLI)")
        return df
        
    except Exception as e:
        print(f"Database hatası: {e}")
        return pd.DataFrame()

# Binance API fallback fonksiyonu
def get_live_data_from_api(symbol: str, interval: str, limit: int):
    """Binance REST API'den direkt veri çeker (yavaş fallback)."""
    try:
        import requests
        import pandas as pd
        from datetime import datetime
        
        # Binance REST API endpoint
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': str(limit)
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 400:
            print(f"⚠️ Geçersiz sembol: {symbol} - Binance'de bulunamadı")
            return pd.DataFrame()
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # DataFrame'e dönüştür
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Veri tiplerini düzelt
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Timestamp'leri datetime'a çevir
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Date sütunu ekle (grafik için gerekli)
        df['date'] = df['open_time']
        
        print(f"⚠️ API'den {symbol} verisi çekildi: {len(df)} mum (YAVAŞ)")
        return df
        
    except Exception as e:
        print(f"Binance API hatası: {e}")
        return pd.DataFrame()

# Ana veri çekme fonksiyonu (akıllı seçim)
def get_chart_data(symbol: str, interval: str, limit: int):
    """Akıllı veri çekme: Önce database, sonra API fallback."""
    # 1. Önce database'den dene
    df = get_price_data_from_db(symbol, interval, limit)
    
    if not df.empty:
        # Database'den veri geldi
        data_source = "database"
        return df, data_source
    
    # 2. Database'de veri yoksa API'den çek
    print(f"⚠️ Database'de {symbol} {interval} verisi yok, API'ye geçiliyor...")
    df = get_live_data_from_api(symbol, interval, limit)
    data_source = "api"
    
    return df, data_source

# Database (Async)
from database.engine import init_db
from database import crud as db_crud

# Signals


# Indicators
from indicators.core import (
    find_support_resistance,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_adx,
    calculate_mfi,
    calculate_obv,
    calculate_cci,
    calculate_roc,
    calculate_stochastic,
    calculate_williams_r,
    calculate_rsi_sma,
    calculate_awesome_oscillator,
    calculate_stoch_rsi,
    calculate_ema,
    calculate_sma,
    calculate_donchian_channel,
    calculate_keltner_channel,
    calculate_vwap,
    calculate_parabolic_sar,
)

# =============================================================================
# INITIALIZE DATABASE
# =============================================================================


# =============================================================================
# CACHED FUNCTIONS
# =============================================================================


def get_symbols_cache_first():
    """Redis cache'den sembolleri yükle, yoksa veritabanından mevcut coinleri al"""
    # 1. Redis'ten cache'lenmiş sembolleri al
    try:
        r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
        cached_symbols = r.get("symbols_cache")
        r.close()
        
        if cached_symbols:
            symbols = json.loads(cached_symbols)
            print(f"✅ Sembol cache HIT: {len(symbols)} sembol")
            return symbols
    except Exception as e:
        print(f"Redis sembol cache hatası: {e}")
    
    # 2. Veritabanından mevcut coinleri çek (price_data tablosundan)
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,  # Direct PostgreSQL
            database="trader_panel",
            user="yusuf",
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol 
            FROM price_data 
            ORDER BY symbol
        """)
        
        rows = cursor.fetchall()
        symbols = [row['symbol'] for row in rows]
        
        cursor.close()
        conn.close()
        
        print(f"✅ Price_data'dan {len(symbols)} coin yüklendi")
        print(f"İlk 10 coin: {symbols[:10]}")
        
        # Redis'e cache'le
        try:
            r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
            r.set("symbols_cache", json.dumps(symbols), ex=600)  # 10 dakika
            r.close()
            print(f"✅ {len(symbols)} sembol cache'lendi")
        except Exception as e:
            print(f"Redis cache hatası: {e}")
        
        return symbols
        
    except Exception as e:
        print(f"Veritabanından sembol yükleme hatası: {e}")
    
    # 3. Son fallback: Sabit sembol listesi (popüler coinler)
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
        "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT",
        "XRPUSDT", "TRXUSDT", "ETCUSDT", "AVAXUSDT", "MATICUSDT",
        "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT", "MKRUSDT"
    ]
    
    print(f"⚠️ Fallback: {len(symbols)} sabit sembol kullanılıyor")
    return symbols

coin_list = get_symbols_cache_first()


# === Otomatik Sinyal Tablosu (AgGrid) ===

# --- Sinyal Sütunlarını Dinamik Olarak Tespit Eden Fonksiyon ---
def _rename_signal_columns(df, only_confirmed=True):
    """Kolon isimlerini Türkçe'ye çevir ve sırala."""
    import pandas as pd
    from utils.bar_counter import calculate_bars_since_signal
    from datetime import datetime
    
    if df.empty:
        return df
    
    print(f"DEBUG: Gelen DataFrame kolonları: {list(df.columns)}")
    print(f"DEBUG: DataFrame shape: {df.shape}")
    
    column_mapping = {
        'symbol': 'Coin',
        'signal_type': 'Sinyal Türü',
        'indicators': 'Sinyal Mantığı',
        'strength': 'Sinyal Gücü',
        'vpms_score': 'VPM Skoru',
        'timestamp': 'Tarih/Saat',
        'price': 'Fiyat ($)',
        'rsi': 'RSI Değeri',
        'momentum': 'Momentum',
        'status': 'status',
        'interval': 'Zaman Dilimi'
    }
    
    # Kolon adlarını değiştir
    df = df.rename(columns=column_mapping)
    
    # Zaman sütununu datetime'a çevir
    if 'Tarih/Saat' in df.columns:
        df["Tarih/Saat"] = pd.to_datetime(df["Tarih/Saat"], format='mixed', errors='coerce')

    # Sinyal tiplerini AL/SAT formatına çevir
    if 'Sinyal Türü' in df.columns:
        df["Sinyal Türü"] = df["Sinyal Türü"].replace({"Long": "AL", "Short": "SAT"})

        # Sadece AL ve SAT sinyallerini tut
        df = df[df["Sinyal Türü"].isin(["AL", "SAT"])]

    # "Sadece onaylı" filtresini uygula
    if only_confirmed and 'Onaylı' in df.columns:
        df = df[df['Onaylı'] == True]
    
    return df


def load_signals(hours=24):
    """Cache-first stratejisi: Önce Redis, sonra veritabanı."""
    try:
        # Sync version kullan - daha güvenli
        signals = _load_signals_cache_first_sync(hours=hours)
        if not signals:
            return pd.DataFrame()
        
        df = pd.DataFrame(signals)
        return _rename_signal_columns(df, only_confirmed=True)
    except Exception as e:
        print(f"Sinyal yükleme hatası: {e}")
        return pd.DataFrame()


def _load_signals_cache_first_sync(hours=24):
    """Cache-first stratejisi: Redis -> Database fallback (Sync version)."""
    import redis
    import json
    import psycopg2
    import psycopg2.extras
    from config import Config
    
    # 1. Redis cache'den sinyalleri dene
    try:
        r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
        cache_key = f"signals_cache:{hours}h"
        cached_data = r.get(cache_key)
        r.close()
        
        if cached_data:
            cached_signals = json.loads(cached_data)
            print(f"✅ Redis cache HIT: {len(cached_signals)} sinyal")
            # Cache'den gelen veriyi _rename_signal_columns ile işle
            df_cached = pd.DataFrame(cached_signals)
            df_cached = _rename_signal_columns(df_cached, only_confirmed=True)
            return df_cached.to_dict('records')
        else:
            print("⚠️ Redis cache MISS - veritabanına gidiliyor")
    except Exception as e:
        print(f"Redis cache hatası: {e}")
    
    # 2. Cache miss - doğrudan SQL ile çek (psycopg2)
    try:
        # Direct PostgreSQL connection
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="trader_panel",
            user="yusuf",
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        
        cursor = conn.cursor()
        query = """
        SELECT 
            symbol, signal_type, indicators, strength, vpms_score, 
            timestamp, price, rsi, momentum, status, interval
        FROM signals 
        WHERE status = 'active'
        AND timestamp >= NOW() - INTERVAL '%s hours'
        ORDER BY timestamp DESC
        LIMIT 1000
        """ % hours
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Dict listesine çevir
        db_signals = [dict(row) for row in rows]
        
        cursor.close()
        conn.close()
        
        # 3. Veriyi işle ve cache'le
        if db_signals:
            # DataFrame'e çevir ve kolon adlarını düzenle
            df_processed = pd.DataFrame(db_signals)
            df_processed = _rename_signal_columns(df_processed, only_confirmed=True)
            processed_signals = df_processed.to_dict('records')
            
            # Redis'e cache'le (5 dakika)
            try:
                r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
                cache_key = f"signals_cache:{hours}h"
                r.set(cache_key, json.dumps(processed_signals, default=str), ex=300)
                r.close()
                print(f"✅ {len(processed_signals)} sinyal yüklendi ve işlendi (sync SQL)")
            except Exception:
                pass  # Cache yazma hatası önemli değil
            
            return processed_signals
        
        return []
        
    except Exception as e:
        print(f"Veritabanı hatası: {e}")
        return []

# Async sinyal yükleme fonksiyonu kaldırıldı - sync versiyonu kullanılıyor


    # Oran (ratio) sütunlarını gizle: Sadece Alpha ve Beta kalsın
    ratio_cols_tr = [
        "Sharpe Oranı",
        "Sortino Oranı",
        "Calmar Oranı",
        "Omega Oranı",
        "Treynor Oranı",
        "Bilgi Oranı",
    ]
    existing_ratio_cols = [c for c in ratio_cols_tr if c in df.columns]
    if existing_ratio_cols:
        df = df.drop(columns=existing_ratio_cols)

    # Zaman sütununu datetime'a çevir
    if 'Tarih/Saat' in df.columns:
        df["Tarih/Saat"] = pd.to_datetime(df["Tarih/Saat"], format='mixed', errors='coerce')

    # Sinyal tiplerini AL/SAT formatına çevir
    if 'Sinyal Türü' in df.columns:
        df["Sinyal Türü"] = df["Sinyal Türü"].replace({"Long": "AL", "Short": "SAT"})

        # Sadece AL ve SAT sinyallerini tut
        df = df[df["Sinyal Türü"].isin(["AL", "SAT"])]

    # "Sadece onaylı" filtresini uygula
    if only_confirmed and 'Onaylı' in df.columns:
        df = df[df['Onaylı'] == True]

    # Varsayılan sıralama: Birleşik Skor, VPM Skoru, Sinyal Gücü, Tarih/Saat (azalan)
    sort_priority = ["Birleşik Skor", "VPM Skoru", "Sinyal Gücü", "Tarih/Saat"]
    existing_sort_cols = [c for c in sort_priority if c in df.columns]
    if existing_sort_cols:
        df = df.sort_values(by=existing_sort_cols, ascending=[False] * len(existing_sort_cols))

    return df

async def load_signals_from_redis():
    """Redis'ten canlı sinyal verilerini yükler."""
    try:
        # Tüm sembollerin canlı verilerini Redis'ten çek
        symbols = await get_active_symbols_from_redis()
        if not symbols:
            return pd.DataFrame()
        
        signals_data = []
        for symbol in symbols:
            try:
                df = await RedisClient.get_df(f"live_kline_data:{symbol}")
                if df is not None and not df.empty and len(df) >= 2:
                    # Son iki satırdan sinyal analizi yap
                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2]
                    
                    # RSI sinyali kontrol et (rsi_9 kullan)
                    if 'rsi_9' in df.columns:
                        rsi_current = last_row.get('rsi_9')
                        rsi_prev = prev_row.get('rsi_9')
                        
                        if pd.notna(rsi_current) and pd.notna(rsi_prev):
                            rsi_change = rsi_current - rsi_prev
                            
                            # RSI sinyal koşulları
                            if rsi_change > 10:  # RSI 10 puan arttı
                                signals_data.append({
                                    'Sembol': symbol,
                                    'Zaman': last_row.get('open_time', pd.Timestamp.now()),
                                    'Sinyal': 'RSI_AL',
                                    'Fiyat': last_row.get('close'),
                                    'RSI': rsi_current,
                                    'RSI_Değişim': rsi_change,
                                    'MA200': last_row.get('ma200'),
                                    'MACD': last_row.get('macd')
                                })
                            elif rsi_change < -10:  # RSI 10 puan düştü
                                signals_data.append({
                                    'Sembol': symbol,
                                    'Zaman': last_row.get('open_time', pd.Timestamp.now()),
                                    'Sinyal': 'RSI_SAT',
                                    'Fiyat': last_row.get('close'),
                                    'RSI': rsi_current,
                                    'RSI_Değişim': rsi_change,
                                    'MA200': last_row.get('ma200'),
                                    'MACD': last_row.get('macd')
                                })
                    
                    # MA200 crossover sinyali
                    if 'ma200' in df.columns:
                        price_current = last_row.get('close')
                        price_prev = prev_row.get('close')
                        ma200_current = last_row.get('ma200')
                        ma200_prev = prev_row.get('ma200')
                        
                        if all(pd.notna(x) for x in [price_current, price_prev, ma200_current, ma200_prev]):
                            # Fiyat MA200'ü yukarı kesti
                            if price_prev <= ma200_prev and price_current > ma200_current:
                                signals_data.append({
                                    'Sembol': symbol,
                                    'Zaman': last_row.get('open_time', pd.Timestamp.now()),
                                    'Sinyal': 'MA200_AL',
                                    'Fiyat': price_current,
                                    'RSI': last_row.get('rsi_9'),
                                    'MA200': ma200_current,
                                    'MACD': last_row.get('macd')
                                })
                            # Fiyat MA200'ü aşağı kesti
                            elif price_prev >= ma200_prev and price_current < ma200_current:
                                signals_data.append({
                                    'Sembol': symbol,
                                    'Zaman': last_row.get('open_time', pd.Timestamp.now()),
                                    'Sinyal': 'MA200_SAT',
                                    'Fiyat': price_current,
                                    'RSI': last_row.get('rsi_9'),
                                    'MA200': ma200_current,
                                    'MACD': last_row.get('macd')
                                })
            except Exception as e:
                continue  # Bu sembol için hata varsa atla
        
        if signals_data:
            df = pd.DataFrame(signals_data)
            # Zamana göre sırala (en yeni önce)
            if 'Zaman' in df.columns:
                df = df.sort_values('Zaman', ascending=False)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Redis'ten sinyal yükleme hatası: {str(e)}")
        return pd.DataFrame()

# Async indikatör yükleme fonksiyonu kaldırıldı - sync versiyonu kullanılıyor

def load_stats():
    """Sync database'den sinyal istatistiklerini yükle"""
    try:
        # Basit istatistik döndür
        return {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'top_symbols': []
        }
    except Exception as e:
        st.error(f"İstatistik yükleme hatası: {str(e)}")
        return {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'top_symbols': []
        }



# --- Database İstatistikleri Dashboard ---


st.subheader("🎯 Aktif Sinyal Tablosu")

# Aktif sinyal filtreleri
st.sidebar.markdown("### 🎯 Aktif Sinyal Filtreleri")
only_confirmed = st.sidebar.checkbox("Sadece onaylı", value=True)
time_filter = st.sidebar.selectbox("Zaman Aralığı", ["Son 1 saat", "Son 6 saat", "Son 12 saat", "Son 24 saat", "Son 7 gün", "Tüm Aktif Sinyaller"], index=5)

# Sinyal mantığı filtresi
signal_strategy_filter = st.sidebar.selectbox(
    "Sinyal Mantığı", 
    ["Tümü", "C20MX", "RSI Cross", "MA200 Cross"], 
    index=0,
    help="Hangi sinyal üretme algoritmasını görmek istiyorsunuz?"
)

# Veritabanı hazır - PgBouncer üzerinden bağlantı
st.toast("Veritabanı bağlantısı hazır (PgBouncer)", icon="🗄️")

# Zaman filtresini hours'a çevir
hours_map = {"Son 1 saat": 1, "Son 6 saat": 6, "Son 12 saat": 12, "Son 24 saat": 24, "Son 7 gün": 168, "Tüm Aktif Sinyaller": 8760}
selected_hours = hours_map[time_filter]

df_signals = load_signals(hours=selected_hours)
print(f"**DEBUG: load_signals sonucu - tip: {type(df_signals)}, uzunluk: {len(df_signals) if df_signals is not None else 'None'}")
print(f"**DEBUG: df_signals içeriği: {df_signals[:3] if df_signals is not None else 'Boş'}")

if df_signals is None or len(df_signals) == 0:
    st.info(f"🔍 {time_filter} içinde aktif sinyal bulunamadı.")
else:
    # Bar sayısını manuel olarak ekle
    from utils.bar_counter import calculate_bars_since_signal
    from datetime import datetime
    
    # DataFrame'i pandas DataFrame'e çevir
    df_signals = pd.DataFrame(df_signals)
    
    print(f"**DEBUG: DataFrame kolonları (bar sayısı öncesi):** {list(df_signals.columns)}")
    
    # Bar sayısı hesaplama - güncellenmiş kolon adlarını kullan
    if 'Tarih/Saat' in df_signals.columns and 'Zaman Dilimi' in df_signals.columns:
        current_time = datetime.now()
        try:
            df_signals['Bar Sayısı'] = df_signals.apply(
                lambda row: calculate_bars_since_signal(row['Tarih/Saat'], row['Zaman Dilimi'], current_time), 
                axis=1
            )
            print(f"**DEBUG: Bar Sayısı kolonu eklendi. Yeni kolonlar:** {list(df_signals.columns)}")
        except Exception as e:
            print(f"**DEBUG: Bar sayısı hesaplama hatası:** {e}")
            df_signals['Bar Sayısı'] = 0  # Fallback değer
    else:
        print(f"**DEBUG: Tarih/Saat veya Zaman Dilimi kolonu bulunamadı. Mevcut kolonlar:** {list(df_signals.columns)}")
        df_signals['Bar Sayısı'] = 0  # Fallback değer
    
    # "Sadece onaylı" filtresini uygula (varsa)
    if only_confirmed and 'Onaylı' in df_signals.columns:
        df_signals = df_signals[df_signals['Onaylı'] == True]
    
    # Sinyal mantığı filtresini uygula
    original_count = len(df_signals)
    if signal_strategy_filter != "Tümü" and 'Sinyal Mantığı' in df_signals.columns:
        if signal_strategy_filter == "C20MX":
            df_signals = df_signals[df_signals['Sinyal Mantığı'].str.contains('C20MX', na=False)]
        elif signal_strategy_filter == "RSI Cross":
            df_signals = df_signals[df_signals['Sinyal Mantığı'].str.contains('RSI_Cross', na=False)]
        elif signal_strategy_filter == "MA200 Cross":
            df_signals = df_signals[df_signals['Sinyal Mantığı'].str.contains('MA200_Cross', na=False)]
        
        filtered_count = len(df_signals)
        print(f"**DEBUG: {signal_strategy_filter} filtresi uygulandı. {original_count} → {filtered_count} sinyal")
        
        # Sidebar'da filtreleme sonucunu göster
        st.sidebar.info(f"📊 Filtreleme Sonucu:\n{original_count} → {filtered_count} sinyal")

    # Sinyal sütununda ikon göster ve Long/Short'u AL/SAT'a çevir
    def signal_icon(signal_type):
        if "Long" in str(signal_type):
            return "🟢 AL"
        elif "Short" in str(signal_type):
            return "🔴 SAT"
        else:
            return str(signal_type)

    print(f"**DEBUG: Kolon adlandırma sonrası:** {list(df_signals.columns)}")
    
    # Debug: Sinyal türü değerlerini kontrol et
    if 'Sinyal Türü' in df_signals.columns:
        print(f"**DEBUG: Sinyal Türü değerleri:** {df_signals['Sinyal Türü'].unique()}")
        print(f"**DEBUG: Sinyal Türü sayıları:** {df_signals['Sinyal Türü'].value_counts()}")
        
        # Long ve Short sinyallerini AL ve SAT olarak ayır
        df_al = df_signals[df_signals["Sinyal Türü"] == "AL"].reset_index(drop=True)
        df_sat = df_signals[df_signals["Sinyal Türü"] == "SAT"].reset_index(drop=True)
        
        print(f"**DEBUG: AL sinyalleri sayısı:** {len(df_al)}")
        print(f"**DEBUG: SAT sinyalleri sayısı:** {len(df_sat)}")
    else:
        print(f"**DEBUG: 'Sinyal Türü' kolonu bulunamadı!")
        # Fallback: Tüm sinyalleri AL olarak göster
        df_signals["Sinyal Türü"] = "AL"
        df_al = df_signals.copy()
        df_sat = pd.DataFrame()

    # AL sinyalleri tablosu
    st.markdown("### 🟢 Aktif AL Sinyalleri")
    st.write("**DEBUG AL Tablosu Kolonları:**", list(df_al.columns) if not df_al.empty else "Boş DataFrame")
    if df_al.empty:
        st.info("🔍 Aktif AL sinyali bulunamadı.")
    else:
        gb_al = GridOptionsBuilder.from_dataframe(df_al)
        gb_al.configure_pagination(paginationAutoPageSize=True)
        gb_al.configure_default_column(editable=False, groupable=True)
        
        # Özel kolon formatlaması
        if 'Bar Sayısı' in df_al.columns:
            gb_al.configure_column('Bar Sayısı', header_name='Bar Sayısı', width=100)
        if 'Sinyal Mantığı' in df_al.columns:
            gb_al.configure_column('Sinyal Mantığı', header_name='Sinyal Mantığı', width=200)
        
        for col in df_al.columns:
            gb_al.configure_column(col, header_name=col)
        grid_options_al = gb_al.build()
        AgGrid(
            df_al,
            gridOptions=grid_options_al,
            theme="streamlit",
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
        )

    # SAT sinyalleri tablosu
    st.markdown("### 🔴 Aktif SAT Sinyalleri")
    st.write("**DEBUG SAT Tablosu Kolonları:**", list(df_sat.columns) if not df_sat.empty else "Boş DataFrame")
    if df_sat.empty:
        st.info("🔍 Aktif SAT sinyali bulunamadı.")
    else:
        gb_sat = GridOptionsBuilder.from_dataframe(df_sat)
        gb_sat.configure_pagination(paginationAutoPageSize=True)
        gb_sat.configure_default_column(editable=False, groupable=True)
        
        # Özel kolon formatlaması
        if 'Bar Sayısı' in df_sat.columns:
            gb_sat.configure_column('Bar Sayısı', header_name='Bar Sayısı', width=100)
        if 'Sinyal Mantığı' in df_sat.columns:
            gb_sat.configure_column('Sinyal Mantığı', header_name='Sinyal Mantığı', width=200)
        
        for col in df_sat.columns:
            gb_sat.configure_column(col, header_name=col)
        grid_options_sat = gb_sat.build()
        AgGrid(
            df_sat,
            gridOptions=grid_options_sat,
            theme="streamlit",
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
        )




# Grafiklere crosshair eklemek için yardımcı fonksiyon
def add_crosshair_to_fig(fig):
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikethickness=1,
        spikecolor="#aaa",
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikethickness=1,
        spikecolor="#aaa",
    )
    fig.update_layout(
        hovermode="x",
        spikedistance=-1,
    )
    return fig


# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

symbols = get_symbols_cache_first()
intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]

st.sidebar.title("Kontrol Paneli")
# Güvenli varsayılan sembol seçimi
default_symbol = "ETHUSDT" if "ETHUSDT" in symbols else (symbols[0] if symbols else "BTCUSDT")
default_index = symbols.index(default_symbol) if default_symbol in symbols else 0
symbol = st.sidebar.selectbox("Sembol Seçiniz", symbols, index=default_index)
interval = st.sidebar.selectbox("Zaman Dilimi", intervals, index=intervals.index("15m"))
limit = st.sidebar.slider(
    "Mum Sayısı", min_value=50, max_value=1000, value=200, step=10
)

 


# interval değişkeni hem tekli hem çoklu sembol için ortak kullanılacak



# st_autorefresh(interval=15000, key="main_autorefresh") # Redis'e geçildiği için artık gerekli değil, manuel yenileme veya daha akıllı bir mekanizma kullanılabilir.

st.markdown("---")
# st.subheader("Fiyat Mum Grafiği (TradingView Tarzı)")



st.markdown("---")

# Layout: Sol sütunda indikatör seçenekleri, sağda grafik


# === VERİYİ HER ZAMAN EN BAŞTA ÇEK (AKILLI SEÇIM) ===
with st.spinner(f"{symbol} verisi çekiliyor..."):
    limit = 250
    df, data_source = get_chart_data(symbol, interval, limit=limit)
    
    if df is None or df.empty:
        st.error("Veri alınamadı.")
        st.stop()
    
    # Veri kaynağını göster
    if data_source == "database":
        st.success(f"✅ Database'den {len(df)} mum yüklendi (Hızlı & Real-time)")
    else:
        st.warning(f"⚠️ API'den {len(df)} mum yüklendi (Yavaş - Database'de veri yok)")
    if "date" not in df.columns:
        if "open_time" in df.columns:
            # open_time zaten datetime olabilir
            if pd.api.types.is_datetime64_any_dtype(df["open_time"]):
                df["date"] = pd.to_datetime(df["open_time"], errors="coerce")
            else:
                df["date"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        else:
            st.error(
                "DataFrame'de 'date', 'open_time' veya 'timestamp' sütunu bulunamadı."
            )
            st.stop()
    else:
        df["date"] = (
            pd.to_datetime(df["date"], unit="s", errors="coerce")
            if np.issubdtype(df["date"].dtype, np.number)  # type: ignore
            else pd.to_datetime(df["date"], errors="coerce")
        )
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        if getattr(df["date"].dt, "tz", None) is None:
            df["date"] = (
                df["date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Istanbul")
            )
        else:
            df["date"] = df["date"].dt.tz_convert("Europe/Istanbul")



# === Alt Panel İndikatörleri (Sekmeli Görünüm) ===
st.markdown("### Alt Panel İndikatörleri")
subchart_options = [
    "RSI",
    "MACD",
    "ADX",
    "MFI",
    "OBV",
    "Triple RSI",
    "Triple RSI (SMA)",
    "ATR",
    "CCI",
    "ROC",
    "Stochastic",
    "Williams %R",
    "AO",
    "StochRSI",
]
tabs = st.tabs(subchart_options)

# Her sekme için ilgili grafiği oluştur
with tabs[0]:  # RSI
    st.subheader("RSI (Relative Strength Index)")
    rsi = calculate_rsi(df, period=Config.RSI_PERIOD_DEFAULT)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(
        go.Scatter(
            x=df["date"],
            y=rsi,
            mode="lines",
            name="RSI",
            line=dict(color="#ff9800", width=2),
        )
    )
    fig_rsi.update_layout(
        yaxis=dict(title="RSI", range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig_rsi = add_crosshair_to_fig(fig_rsi)
    st.plotly_chart(fig_rsi, use_container_width=True)

with tabs[1]:  # MACD
    st.subheader("MACD (Moving Average Convergence Divergence)")
    macd, signal, hist = calculate_macd(df, fast=12, slow=26, signal=9)
    fig_macd = go.Figure()
    fig_macd.add_trace(
        go.Bar(
            x=df["date"],
            y=hist,
            name="Histogram",
            marker_color=["#43a047" if v >= 0 else "#e53935" for v in hist],
        )
    )
    fig_macd.add_trace(
        go.Scatter(
            x=df["date"],
            y=macd,
            mode="lines",
            name="MACD",
            line=dict(color="#1976d2", width=2),
        )
    )
    fig_macd.add_trace(
        go.Scatter(
            x=df["date"],
            y=signal,
            mode="lines",
            name="Signal",
            line=dict(color="#d32f2f", width=1, dash="dash"),
        )
    )
    fig_macd.update_layout(
        yaxis=dict(title="MACD"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_macd = add_crosshair_to_fig(fig_macd)
    st.plotly_chart(fig_macd, use_container_width=True)

with tabs[2]:  # ADX
    st.subheader("ADX (Average Directional Index)")
    adx, _, _ = calculate_adx(df, adxlen=14, dilen=14)
    fig_adx = go.Figure()
    fig_adx.add_trace(
        go.Scatter(
            x=df["date"],
            y=adx,
            mode="lines",
            name="ADX",
            line=dict(color="#00bcd4", width=2),
        )
    )
    fig_adx.update_layout(
        yaxis=dict(title="ADX"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_adx = add_crosshair_to_fig(fig_adx)
    st.plotly_chart(fig_adx, use_container_width=True)

with tabs[3]:  # MFI
    st.subheader("MFI (Money Flow Index)")
    mfi = calculate_mfi(df, period=Config.MFI_PERIOD)
    fig_mfi = go.Figure()
    fig_mfi.add_trace(
        go.Scatter(
            x=df["date"],
            y=mfi,
            mode="lines",
            name="MFI",
            line=dict(color="#8bc34a", width=2),
        )
    )
    fig_mfi.update_layout(
        yaxis=dict(title="MFI", range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig_mfi = add_crosshair_to_fig(fig_mfi)
    st.plotly_chart(fig_mfi, use_container_width=True)

with tabs[4]:  # OBV
    st.subheader("OBV (On-Balance Volume)")
    obv = calculate_obv(df)
    fig_obv = go.Figure()
    fig_obv.add_trace(
        go.Scatter(
            x=df["date"],
            y=obv,
            mode="lines",
            name="OBV",
            line=dict(color="#009688", width=2),
        )
    )
    fig_obv.update_layout(
        yaxis=dict(title="OBV"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_obv = add_crosshair_to_fig(fig_obv)
    st.plotly_chart(fig_obv, use_container_width=True)

with tabs[5]:  # Triple RSI
    st.subheader("Triple RSI")
    rsi7 = calculate_rsi(df, period=7)
    rsi_14 = calculate_rsi(df, period=Config.RSI_PERIOD_DEFAULT)
    rsi21 = calculate_rsi(df, period=Config.RSI_PERIOD_SLOW)
    fig_triple = go.Figure()
    fig_triple.add_trace(
        go.Scatter(
            x=df["date"],
            y=rsi7,
            mode="lines",
            name="RSI 7",
            line=dict(color="purple", width=1.5),
        )
    )
    fig_triple.add_trace(
        go.Scatter(
            x=df["date"],
            y=rsi_14,
            mode="lines",
            name="RSI 14",
            line=dict(color="blue", width=1.5),
        )
    )
    fig_triple.add_trace(
        go.Scatter(
            x=df["date"],
            y=rsi21,
            mode="lines",
            name="RSI 21",
            line=dict(color="orange", width=1.5),
        )
    )
    fig_triple.update_layout(
        yaxis=dict(title="Triple RSI"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_triple = add_crosshair_to_fig(fig_triple)
    st.plotly_chart(fig_triple, use_container_width=True)

with tabs[6]:  # Triple RSI (SMA)
    st.subheader("Triple RSI (SMA)")
    rsi7_sma = calculate_rsi(df, period=Config.RSI_PERIOD_FAST).rolling(window=7).mean()
    rsi_14_sma = calculate_rsi(df, period=Config.RSI_PERIOD_DEFAULT).rolling(window=7).mean()
    rsi21_sma = calculate_rsi(df, period=Config.RSI_PERIOD_SLOW).rolling(window=7).mean()
    fig_triple_sma = go.Figure()
    fig_triple_sma.add_trace(
        go.Scatter(
            x=df["date"],
            y=rsi7_sma,
            mode="lines",
            name="RSI 7 SMA",
            line=dict(color="purple", width=1.5, dash="dot"),
        )
    )
    fig_triple_sma.add_trace(
        go.Scatter(
            x=df["date"],
            y=rsi_14_sma,
            mode="lines",
            name="RSI 14 SMA",
            line=dict(color="blue", width=1.5, dash="dot"),
        )
    )
    fig_triple_sma.add_trace(
        go.Scatter(
            x=df["date"],
            y=rsi21_sma,
            mode="lines",
            name="RSI 21 SMA",
            line=dict(color="orange", width=1.5, dash="dot"),
        )
    )
    fig_triple_sma.update_layout(
        yaxis=dict(title="Triple RSI (SMA)"),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig_triple_sma = add_crosshair_to_fig(fig_triple_sma)
    st.plotly_chart(fig_triple_sma, use_container_width=True)

with tabs[7]:  # ATR
    st.subheader("ATR (Average True Range)")
    atr = calculate_atr(df, period=Config.ATR_PERIOD)
    fig_atr = go.Figure()
    fig_atr.add_trace(
        go.Scatter(
            x=df["date"],
            y=atr,
            mode="lines",
            name="ATR",
            line=dict(color="royalblue", width=2),
        )
    )
    fig_atr.update_layout(
        yaxis=dict(title="ATR"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_atr = add_crosshair_to_fig(fig_atr)
    st.plotly_chart(fig_atr, use_container_width=True)

with tabs[8]:  # CCI
    st.subheader("CCI (Commodity Channel Index)")
    cci = calculate_cci(df, period=20)
    fig_cci = go.Figure()
    fig_cci.add_trace(
        go.Scatter(
            x=df["date"],
            y=cci,
            mode="lines",
            name="CCI",
            line=dict(color="#ff9800", width=2),
        )
    )
    fig_cci.add_hline(y=100, line_dash="dash", line_color="gray")
    fig_cci.add_hline(y=-100, line_dash="dash", line_color="gray")
    fig_cci.update_layout(
        yaxis=dict(title="CCI"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_cci = add_crosshair_to_fig(fig_cci)
    st.plotly_chart(fig_cci, use_container_width=True)

with tabs[9]:  # ROC
    st.subheader("ROC (Rate of Change)")
    roc = calculate_roc(df, period=Config.ROC_PERIOD)
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=df["date"],
            y=roc,
            mode="lines",
            name="ROC",
            line=dict(color="#8bc34a", width=2),
        )
    )
    fig_roc.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_roc.update_layout(
        yaxis=dict(title="ROC"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_roc = add_crosshair_to_fig(fig_roc)
    st.plotly_chart(fig_roc, use_container_width=True)

with tabs[10]:  # Stochastic
    st.subheader("Stochastic Oscillator")
    k, d = calculate_stochastic(df, k_period=Config.STOCH_K_PERIOD, d_period=Config.STOCH_D_PERIOD)
    fig_stoch = go.Figure()
    fig_stoch.add_trace(
        go.Scatter(
            x=df["date"],
            y=k,
            mode="lines",
            name="%K",
            line=dict(color="#03a9f4", width=2),
        )
    )
    fig_stoch.add_trace(
        go.Scatter(
            x=df["date"],
            y=d,
            mode="lines",
            name="%D",
            line=dict(color="#e91e63", width=1, dash="dash"),
        )
    )
    fig_stoch.add_hline(y=80, line_dash="dash", line_color="gray")
    fig_stoch.add_hline(y=20, line_dash="dash", line_color="gray")
    fig_stoch.update_layout(
        yaxis=dict(title="Stochastic"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_stoch = add_crosshair_to_fig(fig_stoch)
    st.plotly_chart(fig_stoch, use_container_width=True)

with tabs[11]:  # Williams %R
    st.subheader("Williams %R")
    willr = calculate_williams_r(df, period=Config.WILLIAMS_R_PERIOD)
    fig_will = go.Figure()
    fig_will.add_trace(
        go.Scatter(
            x=df["date"],
            y=willr,
            mode="lines",
            name="Williams %R",
            line=dict(color="#9c27b0", width=2),
        )
    )
    fig_will.add_hline(y=-20, line_dash="dash", line_color="gray")
    fig_will.add_hline(y=-80, line_dash="dash", line_color="gray")
    fig_will.update_layout(
        yaxis=dict(title="Williams %R"), height=300, margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_will = add_crosshair_to_fig(fig_will)
    st.plotly_chart(fig_will, use_container_width=True)

with tabs[12]:  # AO
    st.subheader("Awesome Oscillator")
    ao = calculate_awesome_oscillator(df)
    colors = ["green" if val >= 0 else "red" for val in ao]
    fig_ao = go.Figure()
    fig_ao.add_trace(go.Bar(x=df["date"], y=ao, name="AO", marker_color=colors))
    fig_ao.update_layout(
        title=f"{symbol} Awesome Oscillator",
        xaxis_title="Tarih",
        yaxis_title="Değer",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_ao = add_crosshair_to_fig(fig_ao)
    st.plotly_chart(fig_ao, use_container_width=True)

with tabs[13]:  # StochRSI
    st.subheader("Stochastic RSI")
    stoch_k, stoch_d = calculate_stoch_rsi(df)
    fig_stoch_rsi = go.Figure()
    fig_stoch_rsi.add_trace(
        go.Scatter(x=df["date"], y=stoch_k, mode="lines", name="StochK")
    )
    fig_stoch_rsi.add_trace(
        go.Scatter(x=df["date"], y=stoch_d, mode="lines", name="StochD")
    )
    fig_stoch_rsi.update_layout(
        title=f"{symbol} Stochastic RSI",
        xaxis_title="Tarih",
        yaxis_title="Değer",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_stoch_rsi = add_crosshair_to_fig(fig_stoch_rsi)
    st.plotly_chart(fig_stoch_rsi, use_container_width=True)

# Grafik ana panelde

with st.container():
    with st.spinner(f"{symbol} verisi çekiliyor..."):
        limit = 250  # Daha uzun veri çekimi için
        df, _ = get_chart_data(symbol, interval, limit=limit)
        # 'date' sütunu yoksa güvenli şekilde oluştur
        if "date" not in df.columns:
            if "open_time" in df.columns:
                df["date"] = pd.to_datetime(df["open_time"], unit="ms")
            elif "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            else:
                st.error(
                    "Veride tarih bilgisi yok! Grafik çizilemiyor."
                )
                st.stop()
        else:
            df["date"] = (
                pd.to_datetime(df["date"], unit="s", errors="coerce")
                if np.issubdtype(df["date"].dtype, np.number)  # type: ignore
                else pd.to_datetime(df["date"], errors="coerce")
            )
        # UTC+3 (Europe/Istanbul) zaman dilimine çevir
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            if getattr(df["date"].dt, "tz", None) is None:
                df["date"] = (
                    df["date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Istanbul")
                )
            else:
                df["date"] = df["date"].dt.tz_convert("Europe/Istanbul")

    if not df.empty and {"open_time", "open", "high", "low", "close"}.issubset(
        df.columns
    ):
        # Zamanı UTC+3'e (Europe/Istanbul) çevir ve okunur hale getir (güvenli şekilde)
        if pd.api.types.is_datetime64_any_dtype(df["open_time"]):
            if getattr(df["open_time"].dt, "tz", None) is None:
                x_data = (
                    df["open_time"]
                    .dt.tz_localize("UTC")
                    .dt.tz_convert("Europe/Istanbul")
                    .dt.strftime("%Y-%m-%d %H:%M")
                    .tolist()
                )
            else:
                x_data = (
                    df["open_time"]
                    .dt.tz_convert("Europe/Istanbul")
                    .dt.strftime("%Y-%m-%d %H:%M")
                    .tolist()
                )
        else:
            x_data = (
                pd.to_datetime(df["open_time"], unit="ms")
                .dt.tz_localize("UTC")
                .dt.tz_convert("Europe/Istanbul")
                .dt.strftime("%Y-%m-%d %H:%M")
                .tolist()
            )
        y_data = [
            [float(row.open), float(row.close), float(row.low), float(row.high)]  # type: ignore
            for row in df.itertuples()
        ]

        # Plotly ile fiyat (candlestick) ve hacim overlay grafiği oluştur
        fig = go.Figure()

        # Fiyat (mum grafiği)
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Fiyat",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            )
        )

        # Hacim (barlar)
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Hacim",
                marker_color=[
                    "#26a69a" if float(c) >= float(o) else "#ef5350"
                    for o, c in zip(df["open"], df["close"])
                ],
                opacity=0.4,
                yaxis="y2",
            )
        )

        # EMA (21)
        ema = calculate_ema(df, period=Config.EMA_DEFAULT_PERIOD, price_col="close")
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=ema, name="EMA (21)", line=dict(color="orange", width=2)
            )
        )

        # SMA (21)
        sma = calculate_sma(df, period=Config.SMA_DEFAULT_PERIOD, price_col="close")
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=sma,
                name="SMA (21)",
                line=dict(color="blue", width=2, dash="dot"),
            )
        )

        # VWAP (tek trace, legendonly)
        vwap = calculate_vwap(df)
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=vwap,
                mode="lines",
                name="VWAP",
                line=dict(color="purple", dash="dash"),
                hovertemplate="%{x}<br>VWAP: %{y:.2f}<extra></extra>",
                showlegend=True,
                visible="legendonly",
            )
        )

        # Destek/direnç seviyelerini bul
        supports, resistances = find_support_resistance(df, order=10, tolerance=0.002)
        last_close = df["close"].iloc[-1]
        all_levels = sorted(set(supports + resistances))
        # Sadece toplu Scatter trace ile destek/direnç çizgileri (legend'dan aç/kapa yapılabilir)
        # Destek ve direnç seviyelerini ayır
        destek_levels = [lvl for lvl in all_levels if lvl < last_close]
        direnç_levels = [lvl for lvl in all_levels if lvl >= last_close]
        # Tüm destek ve direnç seviyelerini tek bir trace ile göster (legend: Destek/Direnç)
        all_sr_levels = destek_levels + direnç_levels
        if all_sr_levels:
            xs = []
            ys = []
            for lvl in all_sr_levels:
                xs += list(df["date"]) + [None]  # None ile çizgiler ayrılır
                ys += [lvl] * len(df["date"]) + [None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="Destek/Direnç",
                    line=dict(color="#888", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=True,
                    visible="legendonly",
                )
            )

        # Bollinger Bands - tek trace ile aç/kapa yapılabilir
        sma_bb = df["close"].rolling(window=20).mean()
        std_bb = df["close"].rolling(window=20).std()
        upper_band = sma_bb + 2 * std_bb
        lower_band = sma_bb - 2 * std_bb
        xs = list(df["date"]) + [None] + list(df["date"]) + [None] + list(df["date"])
        ys = list(upper_band) + [None] + list(lower_band) + [None] + list(sma_bb)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Bollinger Bands",
                line=dict(color="#90caf9", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=True,
                visible="legendonly",
            )
        )

        donchian_upper, donchian_lower = calculate_donchian_channel(df, period=20)
        xs = list(df["date"]) + [None] + list(df["date"])
        ys = list(donchian_upper) + [None] + list(donchian_lower)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Donchian Channel",
                line=dict(color="#1976d2", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=True,
                visible="legendonly",
            )
        )

        # Parabolic SAR overlay
        psar, trend = calculate_parabolic_sar(df, step=0.02, max_step=0.2)
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=psar,
                mode="markers",
                name="Parabolic SAR",
                marker=dict(color="cyan", size=4),
                hovertemplate="%{x}<br>Parabolic SAR: %{y:.2f}<extra></extra>",
            )
        )

        # VWAP (sabit parametreyle doğrudan ekleniyor)
        vwap = calculate_vwap(df)
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=vwap,
                mode="lines",
                name="VWAP",
                line=dict(color="purple", dash="dash"),
                hovertemplate="%{x}<br>VWAP: %{y:.2f}<extra></extra>",
            )
        )

        # Keltner Channel - tek trace, legendonly
        keltner_ema, keltner_upper, keltner_lower = calculate_keltner_channel(
            df, period=20, atr_mult=2
        )
        xs = list(df["date"]) + [None] + list(df["date"]) + [None] + list(df["date"])
        ys = (
            list(keltner_upper)
            + [None]
            + list(keltner_lower)
            + [None]
            + list(keltner_ema)
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Keltner Channel",
                line=dict(color="#cddc39", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=True,
                visible="legendonly",
            )
        )

        # İkinci y eksenini ekle
        fig.update_layout(
            yaxis=dict(title="Fiyat", side="left", showgrid=True),
            yaxis2=dict(
                title="Hacim",
                overlaying="y",
                side="right",
                showgrid=False,
                position=1.0,
            ),
            xaxis=dict(title="Tarih"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        # Crosshair (artı işareti) etkinleştir
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="solid",
            spikethickness=1,
            spikecolor="#aaa",
        )
        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="solid",
            spikethickness=1,
            spikecolor="#aaa",
        )
        fig.update_layout(
            hovermode="x",
            spikedistance=-1,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Seçilen coin için tüm göstergeleri hesapla ve tablo olarak göster
        indicator_results = {}
        if not df.empty:
            # Tablo için varsayılan periyotları tanımla
            rsi_period = 14
            macd_fast = 12
            macd_slow = 26
            macd_signal = 9

            try:
                indicator_results["RSI"] = calculate_rsi(df, period=rsi_period).iloc[-1]
            except Exception as e:
                indicator_results["RSI"] = np.nan
                print(f"[ERROR] Gösterge Tablosu - RSI Hesaplama Hatası: {e}")
            try:
                macd, signal, hist = calculate_macd(
                    df,
                    fast=macd_fast,
                    slow=macd_slow,
                    signal=macd_signal,
                    price_col="close",
                )
                indicator_results["MACD"] = macd.iloc[-1]
                indicator_results["MACD Signal"] = signal.iloc[-1]
                indicator_results["MACD Histogram"] = hist.iloc[-1]
            except Exception as e:
                indicator_results["MACD"] = indicator_results["MACD Signal"] = (
                    indicator_results["MACD Histogram"]
                ) = np.nan
                print(f"[ERROR] Gösterge Tablosu - MACD Hesaplama Hatası: {e}")
            try:
                adx, plus_di, minus_di = calculate_adx(df, adxlen=14, dilen=14)
                indicator_results["ADX"] = adx.iloc[-1]
                indicator_results["+DI"] = plus_di.iloc[-1]
                indicator_results["-DI"] = minus_di.iloc[-1]
            except Exception:
                indicator_results["ADX"] = indicator_results["+DI"] = indicator_results[
                    "-DI"
                ] = np.nan
            try:
                mfi = calculate_mfi(df, period=Config.MFI_PERIOD)
                indicator_results["MFI"] = mfi.iloc[-1]
            except Exception:
                indicator_results["MFI"] = np.nan
            try:
                obv = calculate_obv(df)
                indicator_results["OBV"] = obv.iloc[-1]
            except Exception:
                indicator_results["OBV"] = np.nan
            try:
                atr = calculate_atr(df, period=Config.ATR_PERIOD)
                indicator_results["ATR (14)"] = atr.iloc[-1]
            except Exception:
                indicator_results["ATR (14)"] = np.nan
            try:
                cci = calculate_cci(df, period=20)
                indicator_results["CCI (20)"] = cci.iloc[-1]
            except Exception:
                indicator_results["CCI (20)"] = np.nan
            try:
                roc = calculate_roc(df, period=Config.ROC_PERIOD)
                indicator_results["ROC (9)"] = roc.iloc[-1]
            except Exception:
                indicator_results["ROC (9)"] = np.nan
            try:
                percent_k, percent_d = calculate_stochastic(df, k_period=14, d_period=3)
                indicator_results["Stoch %K"] = percent_k.iloc[-1]
                indicator_results["Stoch %D"] = percent_d.iloc[-1]
            except Exception:
                indicator_results["Stoch %K"] = indicator_results["Stoch %D"] = np.nan
            try:
                willr = calculate_williams_r(df, period=Config.WILLIAMS_R_PERIOD)
                indicator_results["Williams %R"] = willr.iloc[-1]
            except Exception:
                indicator_results["Williams %R"] = np.nan
            try:
                psar, _ = calculate_parabolic_sar(df)
                indicator_results["Parabolic SAR"] = psar.iloc[-1]
            except Exception:
                indicator_results["Parabolic SAR"] = np.nan
            try:
                vwap = calculate_vwap(df)
                indicator_results["VWAP"] = vwap.iloc[-1]
            except Exception:
                indicator_results["VWAP"] = np.nan
            try:
                ema = calculate_ema(df, period=Config.EMA_DEFAULT_PERIOD, price_col="close")
                indicator_results["EMA (21)"] = ema.iloc[-1]
            except Exception:
                indicator_results["EMA (21)"] = np.nan
            try:
                sma = calculate_sma(df, period=Config.SMA_DEFAULT_PERIOD, price_col="close")
                indicator_results["SMA (21)"] = sma.iloc[-1]
            except Exception:
                indicator_results["SMA (21)"] = np.nan
            try:
                # Bollinger Bands
                bb_period = 20
                bb_std = 2
                sma_bb = calculate_sma(df, period=bb_period, price_col="close")
                std_bb = df["close"].rolling(window=bb_period).std()
                upper_bb = sma_bb + bb_std * std_bb
                lower_bb = sma_bb - bb_std * std_bb
                indicator_results["Bollinger Üst"] = upper_bb.iloc[-1]
                indicator_results["Bollinger Orta"] = sma_bb.iloc[-1]
                indicator_results["Bollinger Alt"] = lower_bb.iloc[-1]
            except Exception:
                indicator_results["Bollinger Üst"] = indicator_results[
                    "Bollinger Orta"
                ] = indicator_results["Bollinger Alt"] = np.nan
            try:
                # Donchian Channel
                donchian_upper, donchian_lower = calculate_donchian_channel(
                    df, period=20
                )
                indicator_results["Donchian Üst"] = donchian_upper.iloc[-1]
                indicator_results["Donchian Alt"] = donchian_lower.iloc[-1]
            except Exception:
                indicator_results["Donchian Üst"] = indicator_results[
                    "Donchian Alt"
                ] = np.nan
            try:
                # Keltner Channel
                ema_keltner, upper_keltner, lower_keltner = calculate_keltner_channel(
                    df, period=20, atr_mult=2
                )
                indicator_results["Keltner Üst"] = upper_keltner.iloc[-1]
                indicator_results["Keltner Orta"] = ema_keltner.iloc[-1]
                indicator_results["Keltner Alt"] = lower_keltner.iloc[-1]
            except Exception:
                indicator_results["Keltner Üst"] = indicator_results["Keltner Orta"] = (
                    indicator_results["Keltner Alt"]
                ) = np.nan
            st.markdown("### Seçilen Coin için Göstergeler Tablosu")
            st.dataframe(
                pd.DataFrame(indicator_results, index=[symbol]).T,
                use_container_width=True,
                height=600,
            )




