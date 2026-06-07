# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
import streamlit as st

st.set_page_config(page_title="TRader Panel", layout="wide")

# =============================================================================
# NAVIGATION MENU
# =============================================================================
from streamlit_option_menu import option_menu

# Ana menü
selected = option_menu(
    menu_title=None,
    options=["📊 Sinyal Analizi", "🧪 Backtest & Paper Trade", "📈 Grafik Analizi", "🏆 Signal Performance"],
    icons=["graph-up", "flask", "bar-chart", "trophy"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#02ab21"},
    },
)

# Başlık ayarlama
if selected == "📊 Sinyal Analizi":
    st.title("TRader Panel - Pes Eden Maldır !!!")
elif selected == "🧪 Backtest & Paper Trade":
    st.title("🧪 Backtest & Paper Trading Sistemi")
elif selected == "📈 Grafik Analizi":
    st.title("📈 Grafik ve İndikatör Analizi")
elif selected == "🏆 Signal Performance":
    st.title("🏆 Signal Performance KPI Dashboard")

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
@st.cache_data(ttl=10)  # 10 saniye cache (daha kısa)
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
        
        # Timestamp'leri datetime'a çevir (UTC olarak, sonra Türkiye saatine çevir)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True).dt.tz_convert('Europe/Istanbul')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True).dt.tz_convert('Europe/Istanbul')
        
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

# Backtest & Paper Trading
from backtest import BacktestEngine, PaperTrader, StrategyTester, PerformanceAnalyzer
from backtest_ui import render_backtest_tab

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
    """Kolon isimlerini Türkçe'ye çevir - basitleştirilmiş versiyon."""
    import pandas as pd
    
    if df.empty:
        return df
    
    try:
        # Temel kolon mapping
        column_mapping = {
            'symbol': 'Coin',
            'signal_type': 'Sinyal Türü',
            'indicators': 'indicators',
            'strength': 'Sinyal Gücü',
            'vpms_score': 'VPM Skoru',
            'timestamp': 'Tarih/Saat',
            'price': 'Fiyat ($)',
            'momentum': 'Momentum',
            'interval': 'Zaman Dilimi',
            'normalized_composite': 'Ratio',
            'alpha': 'Alpha',
            'beta': 'Beta',
            'zscore_ratio_percent': 'Z-Score %'
        }
        
        # Kolon adlarını değiştir
        df = df.rename(columns=column_mapping)
        
        # Sinyal tiplerini AL/SAT formatına çevir
        # ✅ DOĞRU: Long=Fiyat yükselecek → AL, Short=Fiyat düşecek → SAT
        if 'Sinyal Türü' in df.columns:
            df["Sinyal Türü"] = df["Sinyal Türü"].replace({"Long": "AL", "Short": "SAT"})
            # Sadece AL ve SAT sinyallerini tut
            df = df[df["Sinyal Türü"].isin(["AL", "SAT"])]
        
        return df
        
    except Exception as e:
        print(f"_rename_signal_columns hatası: {e}")
        return df



@st.cache_data(ttl=5)  # 5 saniye cache
def load_signals(hours=24, signal_types_filter=None, filter_type="EXACT", interval="1m"):
    """Cache-first stratejisi: Önce Redis, sonra veritabanı. MTF destekli."""
    try:
        # Sync version kullan - daha güvenli, MTF interval desteği ile
        signals = _load_signals_cache_first_sync(
            hours=hours, 
            signal_types_filter=signal_types_filter, 
            filter_type=filter_type,
            interval=interval
        )
        if not signals:
            return pd.DataFrame()
        
        df = pd.DataFrame(signals)
        # _rename_signal_columns zaten _load_signals_cache_first_sync içinde çağrılıyor
        # Tekrar çağırmaya gerek yok
        return df
    except Exception as e:
        print(f"Sinyal yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def _load_signals_cache_first_sync(hours=24, signal_types_filter=None, filter_type="EXACT", interval="1m"):
    """Cache-first stratejisi: Redis -> Database fallback (Sync version). MTF destekli."""
    import redis
    import json
    import psycopg2
    import psycopg2.extras
    from config import Config
    
    # 1. Redis cache'den sinyalleri dene (filtreleme parametrelerini dahil et)
    try:
        r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
        # Cache key'ine filtreleme parametrelerini ve interval'ı ekle
        filter_key = f"{filter_type}:{'-'.join(signal_types_filter) if signal_types_filter else 'ALL'}"
        cache_key = f"signals_cache:{hours}h:{filter_key}:{interval}"
        cached_data = r.get(cache_key)
        r.close()
        
        if cached_data:
            cached_signals = json.loads(cached_data)
            print(f"✅ Redis cache HIT: {len(cached_signals)} sinyal (filtered)")
            # Cache'den gelen veriyi _rename_signal_columns ile işle
            df_cached = pd.DataFrame(cached_signals)
            df_cached = _rename_signal_columns(df_cached, only_confirmed=True)
            return df_cached.to_dict('records')
        else:
            print("⚠️ Redis cache MISS - veritabanına gidiliyor (filtered query)")
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
        
        # Base query
        base_query = f"""
        SELECT 
            symbol, signal_type, indicators, strength, vpms_score, 
            timestamp, price, momentum, interval, normalized_composite,
            alpha, beta, zscore_ratio_percent
        FROM signals 
        WHERE status = 'active'
        AND timestamp >= NOW() - INTERVAL '{hours} hours'
        AND interval = '{interval}'
        """
        
        # İndikatör türü filtresi ekle
        if signal_types_filter and len(signal_types_filter) > 0:
            if filter_type == "C20MX_LIKE":
                # C20MX için LIKE operatörü kullan (tüm C20MX alt türlerini yakala)
                base_query += " AND indicators LIKE 'C20MX%'"
                query = base_query + " ORDER BY timestamp DESC LIMIT 1000"
                cursor.execute(query)
            elif filter_type == "RSI_LIKE":
                # RSI için LIKE operatörü kullan
                base_query += " AND indicators LIKE 'RSI_Cross%'"
                query = base_query + " ORDER BY timestamp DESC LIMIT 1000"
                cursor.execute(query)
            elif filter_type == "EXACT":
                # Exact match için IN clause (MA200_Cross için)
                indicator_list = "', '".join(signal_types_filter)
                base_query += f" AND indicators IN ('{indicator_list}')"
                query = base_query + " ORDER BY timestamp DESC LIMIT 1000"
                cursor.execute(query)
            else:
                # Tüm indikatörler
                query = base_query + " ORDER BY timestamp DESC LIMIT 1000"
                cursor.execute(query)
        else:
            query = base_query + " ORDER BY timestamp DESC LIMIT 1000"
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
            
            # Redis'e cache'le (5 dakika) - filtreleme parametreleriyle
            try:
                r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
                filter_key = f"{filter_type}:{'-'.join(signal_types_filter) if signal_types_filter else 'ALL'}"
                cache_key = f"signals_cache:{hours}h:{filter_key}"
                r.set(cache_key, json.dumps(processed_signals, default=str), ex=300)
                r.close()
                print(f"✅ {len(processed_signals)} sinyal yüklendi ve işlendi (sync SQL, filtered)")
            except Exception:
                pass  # Cache yazma hatası önemli değil
            
            return processed_signals
        
        return []
        
    except Exception as e:
        print(f"Veritabanı hatası: {e}")
        import traceback
        traceback.print_exc()
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
    # ✅ DOĞRU: Long=Fiyat yükselecek → AL, Short=Fiyat düşecek → SAT
    if 'Sinyal Türü' in df.columns:
        df["Sinyal Türü"] = df["Sinyal Türü"].replace({"Long": "AL", "Short": "SAT"})

        # Sadece AL ve SAT sinyallerini tut
        df = df[df["Sinyal Türü"].isin(["AL", "SAT"])]

    # "Sadece onaylı" filtresini uygula (eğer kolon varsa)
    if only_confirmed and 'Onaylı' in df.columns:
        df = df[df['Onaylı'] == True]
    elif only_confirmed and 'vpm_confirmed' in df.columns:
        # Alternatif: vpm_confirmed kolonu varsa onu kullan
        df = df[df['vpm_confirmed'] == True]

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


# =============================================================================
# MAIN CONTENT BASED ON SELECTED TAB
# =============================================================================

if selected == "📊 Sinyal Analizi":
    # =============================================================================
    # SINYAL ANALIZI SEKMESI
    # =============================================================================
    st.subheader("🎯 Aktif Sinyal Tablosu")
    
    # Sinyal filtreleri
    st.sidebar.markdown("### 🎯 Aktif Sinyal Filtreleri")
    only_confirmed = st.sidebar.checkbox("Sadece onaylı", value=True)
    time_filter = st.sidebar.selectbox(
        "Zaman Aralığı", 
        ["Son 1 saat", "Son 6 saat", "Son 12 saat", "Son 24 saat", "Son 7 gün", "Tüm Aktif Sinyaller"], 
        index=5  # Varsayılan: "Tüm Aktif Sinyaller"
    )
    
    # Timeframe seçici - MTF desteği
    st.sidebar.markdown("#### ⏰ Zaman Dilimi Seçimi")
    
    timeframe_options = {
        "📊 1 Dakika": "1m",
        "⏰ 5 Dakika": "5m", 
        "🕐 15 Dakika": "15m"
    }
    
    selected_timeframe_option = st.sidebar.selectbox(
        "Hangi zaman dilimini görmek istiyorsunuz?",
        list(timeframe_options.keys()),
        index=0,  # Varsayılan: 1m
        key="timeframe_filter",
        help="Multi-timeframe analizi için zaman dilimi seçin"
    )
    
    selected_timeframe = timeframe_options[selected_timeframe_option]
    
    # Sinyal türü filtreleri
    st.sidebar.markdown("#### 📊 Sinyal Türü Seçimi")
    
    # Gerçek veritabanı indikatörlerine göre filtreleme seçenekleri
    indicator_filter_options = {
        "🎯 Tüm İndikatörler": None,
        "⚡ C20MX Sinyalleri": "C20MX",
        "📈 RSI Cross Sinyalleri": "RSI_Cross", 
        "📊 MA200 Cross Sinyalleri": "MA200_Cross"
    }
    
    selected_indicator_option = st.sidebar.selectbox(
        "Hangi indikatör sinyallerini görmek istiyorsunuz?",
        list(indicator_filter_options.keys()),
        index=0,
        key="indicator_filter",
        help="İndikatör türüne göre sinyalleri filtreler"
    )
    
    # Seçilen indikatör türünü al
    selected_indicator_type = indicator_filter_options[selected_indicator_option]
    
    # Gerçek indikatör türüne göre filtreleme mantığı
    if selected_indicator_type == "C20MX":
        signal_filter_type = "C20MX_LIKE"  # C20MX:* için LIKE
        selected_signal_types = ["C20MX"]
    elif selected_indicator_type == "RSI_Cross":
        signal_filter_type = "RSI_LIKE"  # RSI_Cross(9,24) için LIKE
        selected_signal_types = ["RSI_Cross"]
    elif selected_indicator_type == "MA200_Cross":
        signal_filter_type = "EXACT"  # Tam eşleşme
        selected_signal_types = ["MA200_Cross"]
    else:
        signal_filter_type = "ALL"  # Tüm indikatörler
        selected_signal_types = []
    
    # Zaman filtresini hours'a çevir
    hours_map = {
        "Son 1 saat": 1, 
        "Son 6 saat": 6, 
        "Son 12 saat": 12, 
        "Son 24 saat": 24, 
        "Son 7 gün": 168, 
        "Tüm Aktif Sinyaller": 99999  # Çok büyük değer = sınırsız
    }
    selected_hours = hours_map[time_filter]
    
    # Veritabanından sinyalleri yükle
    st.toast("Sinyal verileri yükleniyor...", icon="📊")
    
    try:
        # Sinyal türü filtresini hazırla
        signal_filter = selected_signal_types if selected_signal_types else None
        
        # Sync sinyal yükleme fonksiyonu kullan - MTF destekli
        signals_data = load_signals(
            hours=selected_hours, 
            signal_types_filter=signal_filter, 
            filter_type=signal_filter_type,
            interval=selected_timeframe
        )
        
        if signals_data.empty:
            if selected_indicator_option != "🎯 Tüm İndikatörler":
                st.warning(f"🔍 {time_filter} içinde **{selected_indicator_option}** bulunamadı.")
            else:
                st.warning(f"🔍 {time_filter} içinde aktif sinyal bulunamadı.")
            
        else:
            # Filtreleme bilgisi göster - MTF bilgisi ile
            if selected_indicator_option != "🎯 Tüm İndikatörler":
                st.success(f"✅ {len(signals_data)} adet **{selected_indicator_option}** yüklendi ({selected_timeframe_option})")
            else:
                st.success(f"✅ {len(signals_data)} sinyal yüklendi ({selected_timeframe_option} - Tüm indikatörler)")
            # DataFrame zaten hazır
            df_signals = signals_data
            
            # Kolon adlarını Türkçe'ye çevir
            column_mapping = {
                'symbol': 'Coin',
                'timestamp': 'Tarih/Saat', 
                'signal_type': 'Sinyal Türü',
                'interval': 'Zaman Dilimi',
                'price': 'Fiyat',
                'vpms_score': 'VMP Skor',
                'vmp_confirmed': 'Onaylı',
                'strength': 'Güç',
                'status': 'Durum',
                # Finansal Ratio'lar
                'alpha': 'Alpha',
                'beta': 'Beta', 
                'sharpe_ratio': 'Sharpe',
                'sortino_ratio': 'Sortino',
                'calmar_ratio': 'Calmar',
                'omega_ratio': 'Omega',
                'treynor_ratio': 'Treynor',
                'information_ratio': 'Info Ratio',
                'zscore_ratio_percent': 'Z-Score %',
                'normalized_composite': 'Norm. Comp.',
                'normalized_price_change': 'Norm. Price',
                # Teknik İndikatörler
                'rsi': 'RSI',
                'macd': 'MACD',
                'momentum': 'Momentum',
                'atr': 'ATR',
                'adx': 'ADX'
            }
            
            # Mevcut kolonları yeniden adlandır
            for old_col, new_col in column_mapping.items():
                if old_col in df_signals.columns:
                    df_signals = df_signals.rename(columns={old_col: new_col})
            
            # Bar sayısı hesaplama (güvenli)
            if 'Tarih/Saat' in df_signals.columns and 'Zaman Dilimi' in df_signals.columns:
                try:
                    from utils.bar_counter import calculate_bars_since_signal
                    from datetime import datetime
                    
                    current_time = datetime.now()
                    
                    def safe_calculate_bars(row):
                        try:
                            return calculate_bars_since_signal(
                                row['Tarih/Saat'], 
                                row['Zaman Dilimi'], 
                                current_time
                            )
                        except Exception:
                            return 0
                    
                    df_signals['Bar Sayısı'] = df_signals.apply(safe_calculate_bars, axis=1)
                except Exception as e:
                    print(f"Bar sayısı hesaplama hatası: {e}")
                    df_signals['Bar Sayısı'] = 0
            
            # Sadece onaylı filtresi
            if only_confirmed and 'Onaylı' in df_signals.columns:
                df_signals = df_signals[df_signals['Onaylı'] == True]
            
            # Sinyal türüne göre ikon ekle
            def signal_icon(signal_type):
                signal_str = str(signal_type).upper()
                if any(word in signal_str for word in ["AL", "BUY", "LONG"]):
                    return "🟢 " + str(signal_type)
                elif any(word in signal_str for word in ["SAT", "SELL", "SHORT"]):
                    return "🔴 " + str(signal_type)
                else:
                    return str(signal_type)
            
            if 'Sinyal Türü' in df_signals.columns:
                df_signals["Sinyal Türü"] = df_signals["Sinyal Türü"].apply(signal_icon)
            
            # İndikatör türünü göstermek için indicators kolonunu ekle
            if 'indicators' in df_signals.columns:
                # İndikatör türünü daha okunabilir hale getir
                def format_indicator(indicator):
                    if pd.isna(indicator) or indicator == '':
                        return '📊 Bilinmeyen'  # Default
                    
                    indicator_str = str(indicator)
                    
                    # Gerçek indikatör türlerine göre formatlama
                    if 'C20MX' in indicator_str:
                        return '🎯 C20MX'
                    elif 'RSI_Cross' in indicator_str:
                        return '📈 RSI Cross'
                    elif 'MA200_Cross' in indicator_str:
                        return '📊 MA200 Cross'
                    else:
                        # Bilinmeyen indikatör - tam ismini göster (kısaltılmış)
                        return f'📊 {indicator_str[:20]}'
                
                df_signals['İndikatör'] = df_signals['indicators'].apply(format_indicator)
            
            # Tablolarda gösterilecek kolonları seç
            display_columns = [
                'Coin', 'Sinyal Türü', 'İndikatör', 'Tarih/Saat', 'Zaman Dilimi', 'Bar Sayısı',
                'Fiyat ($)', 'VPM Skoru', 'Ratio', 
                'Alpha', 'Beta', 'Z-Score %'
            ]
            
            # Sadece mevcut kolonları al
            available_columns = [col for col in display_columns if col in df_signals.columns]
            df_display = df_signals[available_columns].copy()
            
            # Sayısal değerleri formatla
            numeric_columns = ['Fiyat ($)', 'VPM Skoru', 'Ratio', 'Alpha', 'Beta', 'Z-Score %']
            for col in numeric_columns:
                if col in df_display.columns:
                    df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                    if col in ['Fiyat ($)']:
                        df_display[col] = df_display[col].round(4)
                    elif col in ['VPM Skoru', 'Alpha', 'Beta']:
                        df_display[col] = df_display[col].round(3)
                    elif col in ['Ratio']:
                        df_display[col] = df_display[col].round(3)
                    elif col in ['Z-Score %']:
                        df_display[col] = df_display[col].round(1)
            
            # İstatistikler
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam Sinyal", len(df_signals))
            with col2:
                # LONG sinyalleri say
                if 'Sinyal Türü' in df_signals.columns:
                    al_count = len(df_signals[df_signals["Sinyal Türü"].str.contains("🟢", na=False)])
                else:
                    al_count = len(df_signals[df_signals.get("signal_type", "").str.contains("LONG|Long", na=False, case=False)])
                st.metric("AL Sinyalleri", al_count)
            with col3:
                # SHORT sinyalleri say
                if 'Sinyal Türü' in df_signals.columns:
                    sat_count = len(df_signals[df_signals["Sinyal Türü"].str.contains("🔴", na=False)])
                else:
                    sat_count = len(df_signals[df_signals.get("signal_type", "").str.contains("SHORT|Short", na=False, case=False)])
                st.metric("SAT Sinyalleri", sat_count)
            with col4:
                if len(df_signals) > 0:
                    avg_bars = df_signals.get('Bar Sayısı', pd.Series([0])).mean()
                    st.metric("Ort. Bar Sayısı", f"{avg_bars:.1f}")
            
            # Sinyal tablolarını AL ve SAT olarak ayır (display DataFrame'den)
            if 'Sinyal Türü' in df_display.columns:
                # AL sinyalleri (🟢 içeren)
                df_al = df_display[df_display["Sinyal Türü"].str.contains("🟢", na=False)].reset_index(drop=True)
                # SAT sinyalleri (🔴 içeren)
                df_sat = df_display[df_display["Sinyal Türü"].str.contains("🔴", na=False)].reset_index(drop=True)
            else:
                # Fallback: signal_type kolonuna göre ayır (LONG/SHORT dahil)
                df_al = df_display[df_display.get("signal_type", "").str.contains("AL|BUY|LONG|Long", na=False, case=False)].reset_index(drop=True)
                df_sat = df_display[df_display.get("signal_type", "").str.contains("SAT|SELL|SHORT|Short", na=False, case=False)].reset_index(drop=True)
            
            # AgGrid import
            from st_aggrid import AgGrid, GridOptionsBuilder
            
            # AL Sinyalleri Tablosu
            st.markdown("### 🟢 AL Sinyalleri")
            if df_al.empty:
                st.info("AL sinyali bulunamadı.")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{len(df_al)} adet AL sinyali**")
                with col2:
                    if len(df_al) > 0 and 'Bar Sayısı' in df_al.columns:
                        avg_bars_al = df_al['Bar Sayısı'].mean()
                        st.metric("Ort. Bar", f"{avg_bars_al:.1f}")
                
                # AL tablosu için AgGrid
                gb_al = GridOptionsBuilder.from_dataframe(df_al)
                gb_al.configure_pagination(paginationAutoPageSize=True)
                gb_al.configure_default_column(editable=False, groupable=True)
                gb_al.configure_selection('single')
                
                # Kolon genişlikleri ve formatları
                gb_al.configure_column("Coin", width=90)
                gb_al.configure_column("Sinyal Türü", width=130)
                gb_al.configure_column("Tarih/Saat", width=150)
                gb_al.configure_column("Zaman Dilimi", width=100)
                gb_al.configure_column("Bar Sayısı", width=90)
                gb_al.configure_column("Fiyat ($)", width=100, type=["numericColumn", "numberColumnFilter"], precision=4)
                gb_al.configure_column("VPM Skoru", width=100, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_al.configure_column("Ratio", width=90, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_al.configure_column("Alpha", width=90, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_al.configure_column("Beta", width=90, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_al.configure_column("Z-Score %", width=100, type=["numericColumn", "numberColumnFilter"], precision=1)
                
                grid_options_al = gb_al.build()
                
                AgGrid(
                    df_al,
                    gridOptions=grid_options_al,
                    theme="streamlit",
                    enable_enterprise_modules=False,
                    fit_columns_on_grid_load=True,
                    height=300
                )
            
            st.markdown("---")  # Ayırıcı çizgi
            
            # SAT Sinyalleri Tablosu
            st.markdown("### 🔴 SAT Sinyalleri")
            if df_sat.empty:
                st.info("SAT sinyali bulunamadı.")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{len(df_sat)} adet SAT sinyali**")
                with col2:
                    if len(df_sat) > 0 and 'Bar Sayısı' in df_sat.columns:
                        avg_bars_sat = df_sat['Bar Sayısı'].mean()
                        st.metric("Ort. Bar", f"{avg_bars_sat:.1f}")
                
                # SAT tablosu için AgGrid
                gb_sat = GridOptionsBuilder.from_dataframe(df_sat)
                gb_sat.configure_pagination(paginationAutoPageSize=True)
                gb_sat.configure_default_column(editable=False, groupable=True)
                gb_sat.configure_selection('single')
                
                # Kolon genişlikleri ve formatları
                gb_sat.configure_column("Coin", width=90)
                gb_sat.configure_column("Sinyal Türü", width=130)
                gb_sat.configure_column("Tarih/Saat", width=150)
                gb_sat.configure_column("Zaman Dilimi", width=100)
                gb_sat.configure_column("Bar Sayısı", width=90)
                gb_sat.configure_column("Fiyat ($)", width=100, type=["numericColumn", "numberColumnFilter"], precision=4)
                gb_sat.configure_column("VPM Skoru", width=100, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_sat.configure_column("Ratio", width=90, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_sat.configure_column("Alpha", width=90, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_sat.configure_column("Beta", width=90, type=["numericColumn", "numberColumnFilter"], precision=3)
                gb_sat.configure_column("Z-Score %", width=100, type=["numericColumn", "numberColumnFilter"], precision=1)
                
                grid_options_sat = gb_sat.build()
                
                AgGrid(
                    df_sat,
                    gridOptions=grid_options_sat,
                    theme="streamlit",
                    enable_enterprise_modules=False,
                    fit_columns_on_grid_load=True,
                    height=300
                )
            
    except Exception as e:
        st.error(f"Sinyal verileri yüklenirken hata oluştu: {e}")
        st.info("Veritabanı bağlantısını kontrol edin.")

elif selected == "🧪 Backtest & Paper Trade":
    # =============================================================================
    # BACKTEST & PAPER TRADING SEKMESI
    # =============================================================================
    render_backtest_tab()

elif selected == "📈 Grafik Analizi":
    # =============================================================================
    # GRAFIK ANALIZI SEKMESI  
    # =============================================================================
    st.subheader("📈 Teknik Analiz ve Grafik İnceleme")
    
    # Kontrol paneli (sidebar)
    st.sidebar.markdown("### 📊 Grafik Ayarları")
    
    # Sembol seçimi
    try:
        symbols = get_symbols_cache_first()
        default_symbol = "BTCUSDT" if "BTCUSDT" in symbols else (symbols[0] if symbols else "BTCUSDT")
        selected_symbol = st.sidebar.selectbox("Sembol Seçiniz", symbols, index=symbols.index(default_symbol) if default_symbol in symbols else 0)
    except Exception as e:
        st.warning(f"Sembol listesi yüklenemedi: {e}")
        selected_symbol = st.sidebar.text_input("Sembol", value="BTCUSDT")
    
    # Zaman dilimi seçimi
    intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
    selected_interval = st.sidebar.selectbox("Zaman Dilimi", intervals, index=3)  # Default 1h
    
    # Veri miktarı
    data_limit = st.sidebar.slider("Veri Miktarı (Bar)", 50, 500, 200)
    
    # İndikatör seçenekleri
    st.sidebar.markdown("### 📈 Ana Grafik İndikatörleri")
    show_ema = st.sidebar.checkbox("EMA (21)", value=True)
    show_sma = st.sidebar.checkbox("SMA (21)", value=False)
    show_vwap = st.sidebar.checkbox("VWAP", value=False)
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)
    show_keltner = st.sidebar.checkbox("Keltner Channel", value=False)
    show_donchian = st.sidebar.checkbox("Donchian Channel", value=False)
    show_psar = st.sidebar.checkbox("Parabolic SAR", value=False)
    show_support_resistance = st.sidebar.checkbox("Support/Resistance", value=False)
    show_volume = st.sidebar.checkbox("Volume", value=True)
    
    st.sidebar.markdown("### 📊 Alt Panel İndikatörleri")
    show_rsi = st.sidebar.checkbox("RSI", value=False)
    show_macd = st.sidebar.checkbox("MACD", value=False)
    show_adx = st.sidebar.checkbox("ADX", value=False)
    show_mfi = st.sidebar.checkbox("MFI", value=False)
    show_obv = st.sidebar.checkbox("OBV", value=False)
    show_stochastic = st.sidebar.checkbox("Stochastic", value=False)
    show_williams = st.sidebar.checkbox("Williams %R", value=False)
    show_cci = st.sidebar.checkbox("CCI", value=False)
    show_roc = st.sidebar.checkbox("ROC", value=False)
    show_atr = st.sidebar.checkbox("ATR", value=False)
    
    tab_single, tab_mtf = st.tabs(["Tek Grafik", "MTF Analizi"])

    with tab_mtf:
        import redis as _redis_sync
        import pyarrow as _pa
        from io import StringIO as _StringIO

        def _get_mtf_df(symbol: str, tf: str, limit: int = 100):
            """Redis MTF cache → DB fallback sırasıyla dener."""
            try:
                _r = _redis_sync.Redis.from_url(Config.REDIS_URL, decode_responses=False)
                _data = _r.get(f"live_kline_data:{symbol}:{tf}")
                if _data:
                    if _data[:4] == b"ARDF":
                        _reader = _pa.ipc.open_stream(_data[4:])
                        _df = _reader.read_pandas()
                    else:
                        _df = pd.read_json(_StringIO(_data.decode("utf-8")), orient="split")
                    if "open_time" in _df.columns and pd.api.types.is_integer_dtype(_df["open_time"]):
                        _df["open_time"] = pd.to_datetime(_df["open_time"], unit="ms")
                    return _df.tail(limit)
            except Exception:
                pass
            return get_price_data_from_db(symbol, tf, limit)

        st_autorefresh(interval=60000, key="mtf_autorefresh")
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        tf_labels  = ["1m", "5m", "15m", "1H", "4H", "1D"]
        cols_row1 = st.columns(3)
        cols_row2 = st.columns(3)
        all_cols = cols_row1 + cols_row2
        for _i, (_tf, _label) in enumerate(zip(timeframes, tf_labels)):
            with all_cols[_i]:
                st.markdown(f"**{_label}** — {selected_symbol}")
                try:
                    _df = _get_mtf_df(selected_symbol, _tf, 100)
                    if _df is not None and not _df.empty:
                        _df = _df.sort_values("open_time").reset_index(drop=True)
                        _df["ema21"] = _df["close"].ewm(span=21, adjust=False).mean()
                        _fig = go.Figure()
                        _fig.add_trace(go.Candlestick(
                            x=_df["open_time"],
                            open=_df["open"], high=_df["high"],
                            low=_df["low"],  close=_df["close"],
                            name=_tf, showlegend=False,
                            increasing_line_color="#26a69a",
                            decreasing_line_color="#ef5350",
                        ))
                        _fig.add_trace(go.Scatter(
                            x=_df["open_time"], y=_df["ema21"],
                            line=dict(color="#ff9800", width=1),
                            name="EMA21", showlegend=False,
                        ))
                        _fig.update_layout(
                            height=280,
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis=dict(showticklabels=False, rangeslider_visible=False),
                            yaxis=dict(showticklabels=True, tickfont=dict(size=9)),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(_fig, use_container_width=True)
                    else:
                        st.info(f"{_tf}: Veri bekleniyor...")
                except Exception as _e:
                    st.warning(f"{_tf}: {_e}")

    with tab_single:

        # Grafik oluştur
        try:
            # Veri çekme
            st.toast(f"{selected_symbol} verisi çekiliyor...", icon="📊")

            # Gerçek veri çekme (eski_app.py'den entegre edildi)
            import plotly.graph_objects as go
            import numpy as np

            # Gerçek veri çekme fonksiyonu
            def get_chart_data(symbol: str, interval: str, limit: int):
                """Akıllı veri çekme: Önce database, sonra API fallback."""
                # Database'den veri çekmeyi dene
                try:
                    import psycopg2
                    import psycopg2.extras

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

                    if rows:
                        df = pd.DataFrame([dict(row) for row in rows])
                        # Veri tiplerini düzelt
                        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Timestamp'leri datetime'a çevir
                        if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
                            df['open_time'] = pd.to_datetime(df['open_time'])

                        # Sıralamayı düzelt (eski -> yeni)
                        df = df.sort_values('open_time').reset_index(drop=True)
                        df['date'] = df['open_time']

                        return df, "database"
                except Exception as e:
                    print(f"Database hatası: {e}")

                # API fallback
                try:
                    import requests

                    url = "https://api.binance.com/api/v3/klines"
                    params = {'symbol': symbol, 'interval': interval, 'limit': str(limit)}

                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    if data:
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
                        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True).dt.tz_convert('Europe/Istanbul')
                        df['date'] = df['open_time']

                        return df, "api"
                except Exception as e:
                    print(f"API hatası: {e}")

                return pd.DataFrame(), "none"

            # Gerçek veri çek
            df, data_source = get_chart_data(selected_symbol, selected_interval, data_limit)

            if df.empty:
                st.error("Veri alınamadı. Lütfen başka bir sembol deneyin.")
                st.stop()

            # Veri kaynağını göster
            if data_source == "database":
                st.success(f"✅ Database'den {len(df)} mum yüklendi (Hızlı & Real-time)")
            elif data_source == "api":
                st.warning(f"⚠️ API'den {len(df)} mum yüklendi (Yavaş - Database'de veri yok)")
            else:
                st.error("Veri kaynağı bulunamadı!")

            # Plotly ile profesyonel grafik oluştur
            fig = go.Figure()

            # Ana Candlestick grafiği
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

            # Volume (hacim) overlay
            if show_volume:
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
            if show_ema:
                ema = df['close'].ewm(span=21).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"], y=ema, name="EMA (21)", line=dict(color="orange", width=2)
                    )
                )

            # SMA (21)
            if show_sma:
                sma = df['close'].rolling(window=21).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=sma,
                        name="SMA (21)",
                        line=dict(color="blue", width=2, dash="dot"),
                    )
                )

            # VWAP (Volume Weighted Average Price)
            if show_vwap:
                try:
                    # Basit VWAP hesaplama
                    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                    fig.add_trace(
                        go.Scatter(
                            x=df["date"],
                            y=vwap,
                            mode="lines",
                            name="VWAP",
                            line=dict(color="purple", dash="dash"),
                            hovertemplate="%{x}<br>VWAP: %{y:.2f}<extra></extra>",
                            showlegend=True,
                        )
                    )
                except:
                    pass

            # Bollinger Bands
            if show_bollinger:
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
                    )
                )

            # Parabolic SAR
            if show_psar:
                try:
                    # Basit Parabolic SAR hesaplama (placeholder)
                    psar = df['close'] * 0.995  # Basit yaklaşım
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
                except:
                    pass

            # Keltner Channel
            if show_keltner:
                try:
                    # Basit Keltner Channel hesaplama
                    ema_20 = df['close'].ewm(span=20).mean()
                    atr_20 = df['high'].rolling(20).max() - df['low'].rolling(20).min()  # Basit ATR
                    keltner_upper = ema_20 + (atr_20 * 2)
                    keltner_lower = ema_20 - (atr_20 * 2)

                    xs = list(df["date"]) + [None] + list(df["date"]) + [None] + list(df["date"])
                    ys = list(keltner_upper) + [None] + list(keltner_lower) + [None] + list(ema_20)
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            name="Keltner Channel",
                            line=dict(color="#cddc39", width=1, dash="dot"),
                            hoverinfo="skip",
                            showlegend=True,
                        )
                    )
                except:
                    pass

            # Donchian Channel
            if show_donchian:
                try:
                    # Donchian Channel hesaplama
                    donchian_upper = df['high'].rolling(window=20).max()
                    donchian_lower = df['low'].rolling(window=20).min()

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
                        )
                    )
                except:
                    pass

            # Support/Resistance Seviyeleri
            if show_support_resistance:
                try:
                    # Basit support/resistance hesaplama
                    high_points = df['high'].rolling(window=10, center=True).max()
                    low_points = df['low'].rolling(window=10, center=True).min()

                    # Son fiyata göre seviyeler
                    last_close = df['close'].iloc[-1]
                    resistance_levels = high_points[high_points > last_close].dropna().unique()[-3:]
                    support_levels = low_points[low_points < last_close].dropna().unique()[-3:]

                    # Resistance çizgileri
                    for level in resistance_levels:
                        fig.add_hline(
                            y=level,
                            line_dash="dash",
                            line_color="red",
                            opacity=0.7,
                            annotation_text=f"R: {level:.2f}",
                            annotation_position="right"
                        )

                    # Support çizgileri
                    for level in support_levels:
                        fig.add_hline(
                            y=level,
                            line_dash="dash",
                            line_color="green",
                            opacity=0.7,
                            annotation_text=f"S: {level:.2f}",
                            annotation_position="right"
                        )
                except:
                    pass

            # Layout ayarları (TradingView tarzı)
            layout_updates = {
                "yaxis": dict(title="Fiyat", side="left", showgrid=True),
                "xaxis": dict(title="Tarih"),
                "legend": dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                "height": 600,
                "margin": dict(l=20, r=20, t=30, b=20),
                "title": f"{selected_symbol} - {selected_interval} Grafik Analizi",
                "xaxis_rangeslider_visible": False,
            }

            # Volume için ikinci y ekseni
            if show_volume:
                layout_updates["yaxis2"] = dict(
                    title="Hacim",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    position=1.0,
                )

            fig.update_layout(**layout_updates)

            # TradingView tarzı Crosshair (artı işareti) etkinleştir
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

            # Grafiği göster
            st.plotly_chart(fig, use_container_width=True)

            # İstatistikler
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Son Fiyat", f"{df['close'].iloc[-1]:.4f}")
            with col2:
                price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
                st.metric("Değişim", f"{price_change:.4f}", f"{(price_change/df['close'].iloc[-2]*100):.2f}%")
            with col3:
                st.metric("24h Yüksek", f"{df['high'].max():.4f}")
            with col4:
                st.metric("24h Düşük", f"{df['low'].min():.4f}")

            # Alt Panel İndikatörleri
            alt_panel_indicators = []

            # RSI
            if show_rsi:
                try:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))

                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=rsi, mode='lines', name='RSI',
                            line=dict(color='#ff9800', width=2)
                        )
                    )
                    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım (70)")
                    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım (30)")
                    rsi_fig.update_layout(
                        title="RSI (Relative Strength Index)",
                        yaxis=dict(title="RSI", range=[0, 100]),
                        height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("RSI", rsi_fig))
                except:
                    pass

            # MACD
            if show_macd:
                try:
                    # MACD hesaplama
                    ema_12 = df['close'].ewm(span=12).mean()
                    ema_26 = df['close'].ewm(span=26).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9).mean()
                    histogram = macd_line - signal_line

                    macd_fig = go.Figure()
                    macd_fig.add_trace(
                        go.Bar(
                            x=df['date'], y=histogram, name="Histogram",
                            marker_color=["#43a047" if v >= 0 else "#e53935" for v in histogram]
                        )
                    )
                    macd_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=macd_line, mode='lines', name='MACD',
                            line=dict(color='#1976d2', width=2)
                        )
                    )
                    macd_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=signal_line, mode='lines', name='Signal',
                            line=dict(color='#d32f2f', width=1, dash='dash')
                        )
                    )
                    macd_fig.update_layout(
                        title="MACD (Moving Average Convergence Divergence)",
                        yaxis=dict(title="MACD"), height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("MACD", macd_fig))
                except:
                    pass

            # ADX
            if show_adx:
                try:
                    # Basit ADX hesaplama (placeholder)
                    adx_values = pd.Series([25] * len(df))  # Placeholder

                    adx_fig = go.Figure()
                    adx_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=adx_values, mode='lines', name='ADX',
                            line=dict(color='#00bcd4', width=2)
                        )
                    )
                    adx_fig.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Eşiği (25)")
                    adx_fig.update_layout(
                        title="ADX (Average Directional Index)",
                        yaxis=dict(title="ADX"), height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("ADX", adx_fig))
                except:
                    pass

            # MFI
            if show_mfi:
                try:
                    # Basit MFI hesaplama
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    money_flow = typical_price * df['volume']
                    positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(14).sum()
                    negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(14).sum()
                    mfi = 100 - (100 / (1 + positive_flow / negative_flow))

                    mfi_fig = go.Figure()
                    mfi_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=mfi, mode='lines', name='MFI',
                            line=dict(color='#8bc34a', width=2)
                        )
                    )
                    mfi_fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Aşırı Alım (80)")
                    mfi_fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Aşırı Satım (20)")
                    mfi_fig.update_layout(
                        title="MFI (Money Flow Index)",
                        yaxis=dict(title="MFI", range=[0, 100]),
                        height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("MFI", mfi_fig))
                except:
                    pass

            # OBV
            if show_obv:
                try:
                    # OBV hesaplama
                    obv = pd.Series(index=df.index, dtype=float)
                    obv.iloc[0] = df['volume'].iloc[0]
                    for i in range(1, len(df)):
                        if df['close'].iloc[i] > df['close'].iloc[i-1]:
                            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                        else:
                            obv.iloc[i] = obv.iloc[i-1]

                    obv_fig = go.Figure()
                    obv_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=obv, mode='lines', name='OBV',
                            line=dict(color='#009688', width=2)
                        )
                    )
                    obv_fig.update_layout(
                        title="OBV (On-Balance Volume)",
                        yaxis=dict(title="OBV"), height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("OBV", obv_fig))
                except:
                    pass

            # Stochastic
            if show_stochastic:
                try:
                    # Stochastic hesaplama
                    lowest_low = df['low'].rolling(window=14).min()
                    highest_high = df['high'].rolling(window=14).max()
                    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
                    d_percent = k_percent.rolling(window=3).mean()

                    stoch_fig = go.Figure()
                    stoch_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=k_percent, mode='lines', name='%K',
                            line=dict(color='#03a9f4', width=2)
                        )
                    )
                    stoch_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=d_percent, mode='lines', name='%D',
                            line=dict(color='#e91e63', width=1, dash='dash')
                        )
                    )
                    stoch_fig.add_hline(y=80, line_dash="dash", line_color="red")
                    stoch_fig.add_hline(y=20, line_dash="dash", line_color="green")
                    stoch_fig.update_layout(
                        title="Stochastic Oscillator",
                        yaxis=dict(title="Stochastic", range=[0, 100]),
                        height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("Stochastic", stoch_fig))
                except:
                    pass

            # Williams %R
            if show_williams:
                try:
                    # Williams %R hesaplama
                    highest_high = df['high'].rolling(window=14).max()
                    lowest_low = df['low'].rolling(window=14).min()
                    williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

                    williams_fig = go.Figure()
                    williams_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=williams_r, mode='lines', name='Williams %R',
                            line=dict(color='#9c27b0', width=2)
                        )
                    )
                    williams_fig.add_hline(y=-20, line_dash="dash", line_color="red")
                    williams_fig.add_hline(y=-80, line_dash="dash", line_color="green")
                    williams_fig.update_layout(
                        title="Williams %R",
                        yaxis=dict(title="Williams %R", range=[-100, 0]),
                        height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("Williams %R", williams_fig))
                except:
                    pass

            # CCI
            if show_cci:
                try:
                    # CCI hesaplama
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    sma_tp = typical_price.rolling(window=20).mean()
                    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
                    cci = (typical_price - sma_tp) / (0.015 * mad)

                    cci_fig = go.Figure()
                    cci_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=cci, mode='lines', name='CCI',
                            line=dict(color='#ff9800', width=2)
                        )
                    )
                    cci_fig.add_hline(y=100, line_dash="dash", line_color="red")
                    cci_fig.add_hline(y=-100, line_dash="dash", line_color="green")
                    cci_fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
                    cci_fig.update_layout(
                        title="CCI (Commodity Channel Index)",
                        yaxis=dict(title="CCI"), height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("CCI", cci_fig))
                except:
                    pass

            # ROC
            if show_roc:
                try:
                    # ROC hesaplama
                    roc = ((df['close'] - df['close'].shift(9)) / df['close'].shift(9)) * 100

                    roc_fig = go.Figure()
                    roc_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=roc, mode='lines', name='ROC',
                            line=dict(color='#8bc34a', width=2)
                        )
                    )
                    roc_fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
                    roc_fig.update_layout(
                        title="ROC (Rate of Change)",
                        yaxis=dict(title="ROC %"), height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("ROC", roc_fig))
                except:
                    pass

            # ATR
            if show_atr:
                try:
                    # ATR hesaplama
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift())
                    low_close = np.abs(df['low'] - df['close'].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = true_range.rolling(window=14).mean()

                    atr_fig = go.Figure()
                    atr_fig.add_trace(
                        go.Scatter(
                            x=df['date'], y=atr, mode='lines', name='ATR',
                            line=dict(color='royalblue', width=2)
                        )
                    )
                    atr_fig.update_layout(
                        title="ATR (Average True Range)",
                        yaxis=dict(title="ATR"), height=300, margin=dict(l=20, r=20, t=30, b=20)
                    )
                    alt_panel_indicators.append(("ATR", atr_fig))
                except:
                    pass

            # Alt panel indikatörlerini göster
            if alt_panel_indicators:
                st.markdown("---")
                st.markdown("### 📊 Alt Panel İndikatörleri")

                for indicator_name, indicator_fig in alt_panel_indicators:
                    # Crosshair ekle
                    indicator_fig.update_xaxes(
                        showspikes=True, spikemode="across", spikesnap="cursor",
                        spikedash="solid", spikethickness=1, spikecolor="#aaa"
                    )
                    indicator_fig.update_yaxes(
                        showspikes=True, spikemode="across", spikesnap="cursor",
                        spikedash="solid", spikethickness=1, spikecolor="#aaa"
                    )
                    indicator_fig.update_layout(hovermode="x", spikedistance=-1)

                    st.plotly_chart(indicator_fig, use_container_width=True)


        except Exception as e:
            st.error(f"Grafik oluşturulurken hata: {e}")
            st.info("Veri çekme fonksiyonları henüz implement edilmemiş. Örnek veri gösteriliyor.")

# =============================================================================
# SIGNAL PERFORMANCE DASHBOARD
# =============================================================================
elif selected == "🏆 Signal Performance":
    from dashboard.signal_performance import render_signal_performance_dashboard
    render_signal_performance_dashboard()
