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
import asyncio
from datetime import datetime, timedelta

# =============================================================================
# ASYNC HELPER FOR STREAMLIT
# =============================================================================
def get_async_loop():
    """Get or create an asyncio event loop for the current thread."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def run_async_in_st(coro):
    """Run an async coroutine in Streamlit's sync context."""
    return get_async_loop().run_until_complete(coro)

# =============================================================================
# THIRD PARTY IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz
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

# Synchronous wrapper for async binance calls
def get_live_data_sync(symbol: str, interval: str, limit: int):
    """Tek kapı: Data Provider üzerinden OHLCV (Redis -> REST)."""
    async def _get_data():
        df = await fetch_ohlcv(symbol, interval, limit=limit, source='auto')
        return df if df is not None else pd.DataFrame()
    return run_async_in_st(_get_data())

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


@st.cache_resource(show_spinner=False)
def get_live_symbols():
    async def _get_symbols():
        # The manager should be initialized at app startup
        return await BinanceClientManager.get_top_volume_symbols_async(limit=250)
    return run_async_in_st(_get_symbols())

coin_list = get_live_symbols()


# === Otomatik Sinyal Tablosu (AgGrid) ===


async def load_signals(hours=24):
    """Veritabanından son sinyalleri yükler."""
    signals = await db_crud.get_recent_signals(hours=hours)
    if not signals:
        return pd.DataFrame()
    
    df = pd.DataFrame(signals)

    # Sütunları yeniden adlandır
    df = df.rename(columns={
        "symbol": "Coin",
        "signal_time": "Tarih/Saat",
        "signal_type": "Sinyal Türü",
        "interval": "Zaman Dilimi",
        "strength": "Sinyal Gücü",
        "price": "Fiyat ($)",
        "indicators": "Aktif İndikatörler",
        "alpha": "Alpha Katsayısı",
        "beta": "Beta Katsayısı", 
        "momentum": "Momentum",
        "rsi": "RSI Değeri",
        "macd": "MACD Değeri",
        "pullback_level": "Geri Çekilme Seviyesi",
        "atr": "ATR (Volatilite)",
        "adx": "ADX (Trend Gücü)",
        "plus_di": "+DI",
        "minus_di": "-DI",
        "sharpe_ratio": "Sharpe Oranı",
        "sortino_ratio": "Sortino Oranı",
        "calmar_ratio": "Calmar Oranı",
        "omega_ratio": "Omega Oranı",
        "treynor_ratio": "Treynor Oranı",
        "information_ratio": "Bilgi Oranı",
        "scaled_avg_normalized": "Normalize Ortalama",
        "normalized_composite": "Normalize Kompozit",
        "normalized_price_change": "Normalize Fiyat Değişimi",
        "perf_status": "Performans Durumu",
        "perf_next_candle_momentum_change_pct": "Sonraki Mum Momentum (%)",
        "perf_next_candle_volume_change_pct": "Sonraki Mum Hacim (%)",
        "perf_intra_candle_profit_pct": "Mum İçi Kar (%)",
        "perf_prev_to_signal_momentum_change_pct": "Önceki-Sinyal Momentum (%)",
        "perf_prev_to_signal_volume_change_pct": "Önceki-Sinyal Hacim (%)",
        "vpm_confirmed": "Onaylı",
        "vpms_score": "VPM Skoru",
        "mtf_score": "MTF Skoru",
        "vpms_mtf_score": "Birleşik Skor"
    })

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
        df["Tarih/Saat"] = pd.to_datetime(df["Tarih/Saat"])

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

async def get_active_symbols_from_redis():
    """Redis'te aktif olan sembolleri listeler."""
    try:
        import redis.asyncio as redis
        r = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
        keys = await r.keys('live_kline_data:*')
        await r.close()
        
        # Anahtar isimlerinden sembolleri çıkar
        symbols = [key.replace('live_kline_data:', '') for key in keys]
        return symbols[:50]  # İlk 50 sembol
    except Exception as e:
        return []


async def load_indicators():
    """Price_data tablosundan teknik indikatörleri yükle"""
    try:
        data = await db_crud.get_all_price_data_with_indicators()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Sütun isimlerini Türkçe'ye çevir
            df = df.rename(columns={
                "symbol": "Sembol",
                "close": "Fiyat",
                "ma200": "MA200",
                "rsi_14": "RSI",
                "macd": "MACD",
                "atr": "ATR",
                "momentum": "Momentum",
                "adx": "ADX",
                "plus_di": "+DI",
                "minus_di": "-DI",
                "timestamp": "Zaman"
            })
            
            # Sayısal değerleri formatla
            for col in ["Fiyat", "MA200", "RSI", "MACD", "ATR", "Momentum", "ADX", "+DI", "-DI"]:
                if col in df.columns:
                    df[col] = df[col].round(6)
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"İndikatör yükleme hatası: {str(e)}")
        return pd.DataFrame()


async def load_stats():
    """Async database'den sinyal istatistiklerini yükle"""
    try:
        stats = await db_crud.get_signal_stats()
        return stats
    except Exception as e:
        st.error(f"İstatistik yükleme hatası: {str(e)}")
        return {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'top_symbols': []
        }



# --- Database İstatistikleri Dashboard ---


st.subheader("Otomatik Sinyal Tablosu (Son 24 Saat)")

# Sadece onaylı sinyaller filtresi (AL/SAT tabloları için)
only_confirmed = st.sidebar.checkbox("Sadece onaylı", value=True)

# Veritabanını başlat (tabloların var olduğundan emin ol)
st.toast("Veritabanı kontrol ediliyor...", icon="🗄️")
run_async_in_st(init_db())

df_signals = run_async_in_st(load_signals(hours=24))
if df_signals.empty:
    st.info("Son 24 saatte geçerli sinyal bulunamadı.")
else:
    # "Sadece onaylı" filtresini uygula (varsa)
    if only_confirmed and 'Onaylı' in df_signals.columns:
        df_signals = df_signals[df_signals['Onaylı'] == True]

    # "Sinyal" sütununa göre renkli ikon ekle
    def signal_icon(signal_type):
        if "AL" in str(signal_type):
            return "🟢 " + str(signal_type)
        elif "SAT" in str(signal_type):
            return "🔴 " + str(signal_type)
        else:
            return str(signal_type)

    # Sinyal sütununda ikon göster
    if 'Sinyal Türü' in df_signals.columns:
        df_signals["Sinyal Türü"] = df_signals["Sinyal Türü"].apply(signal_icon)

        # AL ve SAT olarak ayır
        df_al = df_signals[df_signals["Sinyal Türü"].str.contains("AL")].reset_index(drop=True)
        df_sat = df_signals[df_signals["Sinyal Türü"].str.contains("SAT")].reset_index(drop=True)

    # AL sinyalleri tablosu
    st.markdown("### 🟢 AL Sinyalleri")
    if df_al.empty:
        st.info("AL sinyali bulunamadı.")
    else:
        gb_al = GridOptionsBuilder.from_dataframe(df_al)
        gb_al.configure_pagination(paginationAutoPageSize=True)
        gb_al.configure_default_column(editable=False, groupable=True)
        # Sadece geçerli sütunlar için ayar
        # Sadece geçerli sütunlar için ayar
        # Sadece geçerli sütunlar için ayar
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
    st.markdown("### 🔴 SAT Sinyalleri")
    if df_sat.empty:
        st.info("SAT sinyali bulunamadı.")
    else:
        gb_sat = GridOptionsBuilder.from_dataframe(df_sat)
        gb_sat.configure_pagination(paginationAutoPageSize=True)
        gb_sat.configure_default_column(editable=False, groupable=True)
        # Sadece geçerli sütunlar için ayar
        # Sadece geçerli sütunlar için ayar
        # Sadece geçerli sütunlar için ayar
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


# --- Sinyal Sütunlarını Dinamik Olarak Tespit Eden Fonksiyon ---
def get_signal_columns(df):
    # Sinyal sütunları: int veya float olup, ismi 'L', 'S', 'M', 'CROSS' içerenler
    exclude = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date",
        "symbol",
        "open_time",
        "timestamp",
    ]
    signal_cols = [
        col
        for col in df.columns
        if (
            (col not in exclude)
            and (any(x in col for x in ["L", "S", "M", "CROSS"]))
            and (df[col].dtype in ["int64", "float64"])
        )
    ]
    return signal_cols


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

symbols = get_live_symbols()
intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]

st.sidebar.title("Kontrol Paneli")
symbol = st.sidebar.selectbox("Sembol Seçiniz", symbols, index=symbols.index("ETHUSDT"))
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


# === VERİYİ HER ZAMAN EN BAŞTA ÇEK ===
with st.spinner(f"{symbol} verisi çekiliyor..."):
    limit = 250
    df = get_live_data_sync(symbol, interval, limit=limit)
    if df is None or df.empty:
        st.error("Veri alınamadı.")
        st.stop()
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
        df = get_live_data_sync(symbol, interval, limit=limit)
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




