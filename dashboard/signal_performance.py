"""
Signal Performance Dashboard with Interval Filter
==================================================
Sinyal performans KPI'larını görselleştirir - Interval filtresi ile.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional
import psycopg2
import psycopg2.extras
from config import Config


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def build_interval_filter(intervals: Optional[list]):
    """Interval filtresi için SQL WHERE clause oluştur."""
    if not intervals:
        return "", []
    
    placeholders = ','.join(['%s'] * len(intervals))
    return f" AND s.interval IN ({placeholders})", intervals


# =============================================================================
# DATABASE FONKSİYONLARI
# =============================================================================

@st.cache_data(ttl=60)
def get_performance_summary(days_back: int = 7, intervals: Optional[list] = None):
    """Genel performans özeti."""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        interval_filter, interval_params = build_interval_filter(intervals)
        params = [days_back] + interval_params
        
        query = f"""
            SELECT 
                COUNT(*) as total_signals,
                COUNT(CASE WHEN is_calculated THEN 1 END) as calculated_signals,
                ROUND(
                    COUNT(CASE WHEN return_t5_atr > 0 THEN 1 END)::numeric / 
                    NULLIF(COUNT(CASE WHEN is_calculated THEN 1 END), 0) * 100, 
                    2
                ) as hit_rate,
                ROUND(AVG(return_t5_atr), 4) as avg_return_atr,
                ROUND(AVG(return_t5_pct), 4) as avg_return_pct,
                ROUND(AVG(risk_reward), 4) as avg_risk_reward,
                ROUND(SUM(return_t5_atr), 4) as total_profit_atr
            FROM signal_performance sp
            JOIN signals s ON s.id = sp.signal_id
            WHERE s.timestamp >= NOW() - INTERVAL '%s days'
            {interval_filter};
        """
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        return dict(result) if result else {}
        
    finally:
        conn.close()


@st.cache_data(ttl=60)
def get_performance_by_signal_type(days_back: int = 7, intervals: Optional[list] = None):
    """Sinyal tipi bazında performans."""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        interval_filter, interval_params = build_interval_filter(intervals)
        params = [days_back] + interval_params
        
        query = f"""
            SELECT 
                s.signal_type,
                COUNT(*) as total,
                COUNT(CASE WHEN sp.is_calculated THEN 1 END) as calculated,
                ROUND(
                    COUNT(CASE WHEN sp.return_t5_atr > 0 THEN 1 END)::numeric / 
                    NULLIF(COUNT(CASE WHEN sp.is_calculated THEN 1 END), 0) * 100, 
                    2
                ) as hit_rate,
                ROUND(AVG(sp.return_t5_atr), 4) as avg_return_atr,
                ROUND(AVG(sp.return_t5_pct), 4) as avg_return_pct,
                ROUND(AVG(sp.risk_reward), 4) as avg_risk_reward
            FROM signal_performance sp
            JOIN signals s ON s.id = sp.signal_id
            WHERE s.timestamp >= NOW() - INTERVAL '%s days'
              AND sp.is_calculated = TRUE
              {interval_filter}
            GROUP BY s.signal_type
            ORDER BY hit_rate DESC NULLS LAST;
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    finally:
        conn.close()


@st.cache_data(ttl=60)
def get_performance_by_interval(days_back: int = 7):
    """Interval bazında performans - filtre yok (tüm interval'leri göster)."""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cursor.execute("""
            SELECT 
                s.interval,
                COUNT(*) as total,
                COUNT(CASE WHEN sp.is_calculated THEN 1 END) as calculated,
                ROUND(
                    COUNT(CASE WHEN sp.return_t5_atr > 0 THEN 1 END)::numeric / 
                    NULLIF(COUNT(CASE WHEN sp.is_calculated THEN 1 END), 0) * 100, 
                    2
                ) as hit_rate,
                ROUND(AVG(sp.return_t5_atr), 4) as avg_return_atr,
                ROUND(AVG(sp.return_t5_pct), 4) as avg_return_pct
            FROM signal_performance sp
            JOIN signals s ON s.id = sp.signal_id
            WHERE s.timestamp >= NOW() - INTERVAL '%s days'
              AND sp.is_calculated = TRUE
            GROUP BY s.interval
            ORDER BY hit_rate DESC NULLS LAST;
        """, (days_back,))
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    finally:
        conn.close()


@st.cache_data(ttl=60)
def get_performance_timeseries(days_back: int = 7, intervals: Optional[list] = None):
    """Günlük performans trendi."""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        interval_filter, interval_params = build_interval_filter(intervals)
        params = [days_back] + interval_params
        
        query = f"""
            SELECT 
                DATE(s.timestamp) as date,
                COUNT(*) as total_signals,
                COUNT(CASE WHEN sp.is_calculated THEN 1 END) as calculated,
                ROUND(AVG(sp.return_t5_atr), 4) as avg_return_atr,
                ROUND(SUM(sp.return_t5_atr), 4) as cumulative_return_atr,
                ROUND(
                    COUNT(CASE WHEN sp.return_t5_atr > 0 THEN 1 END)::numeric / 
                    NULLIF(COUNT(CASE WHEN sp.is_calculated THEN 1 END), 0) * 100, 
                    2
                ) as hit_rate
            FROM signal_performance sp
            JOIN signals s ON s.id = sp.signal_id
            WHERE s.timestamp >= NOW() - INTERVAL '%s days'
              AND sp.is_calculated = TRUE
              {interval_filter}
            GROUP BY DATE(s.timestamp)
            ORDER BY date ASC;
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    finally:
        conn.close()


@st.cache_data(ttl=60)
def get_top_performers(days_back: int = 7, limit: int = 10, intervals: Optional[list] = None):
    """En iyi performans gösteren sinyaller."""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        interval_filter, interval_params = build_interval_filter(intervals)
        params = [days_back] + interval_params + [limit]
        
        query = f"""
            SELECT 
                s.id,
                s.symbol,
                s.signal_type,
                s.indicators,
                s.vpms_score as vpm_score,
                s.interval,
                s.timestamp,
                sp.return_t5_atr,
                sp.return_t5_pct,
                sp.risk_reward,
                sp.mfe_atr,
                sp.mae_atr
            FROM signal_performance sp
            JOIN signals s ON s.id = sp.signal_id
            WHERE s.timestamp >= NOW() - INTERVAL '%s days'
              AND sp.is_calculated = TRUE
              {interval_filter}
            ORDER BY sp.return_t5_atr DESC
            LIMIT %s;
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    finally:
        conn.close()


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_kpi_cards(summary: dict):
    """KPI kartlarını render eder."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Toplam Sinyal",
            value=f"{summary.get('total_signals', 0):,}",
            delta=f"{summary.get('calculated_signals', 0)} hesaplandı"
        )
    
    with col2:
        hit_rate = summary.get('hit_rate', 0) or 0
        st.metric(
            label="🎯 Hit Rate",
            value=f"{hit_rate:.1f}%",
            delta="Kazanç oranı"
        )
    
    with col3:
        avg_return = summary.get('avg_return_atr', 0) or 0
        st.metric(
            label="📈 Ortalama Getiri",
            value=f"{avg_return:.2f} ATR",
            delta=f"{summary.get('avg_return_pct', 0) or 0:.2f}%"
        )
    
    with col4:
        rr = summary.get('avg_risk_reward', 0) or 0
        st.metric(
            label="⚖️ Risk/Reward",
            value=f"{rr:.2f}",
            delta="Ortalama"
        )


def render_performance_chart(timeseries_data: list):
    """Performans trend grafiği."""
    if not timeseries_data:
        st.info("Henüz veri yok")
        return
    
    df = pd.DataFrame(timeseries_data)
    
    fig = go.Figure()
    
    # Kümülatif getiri
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_return_atr'],
        name='Kümülatif Getiri (ATR)',
        line=dict(color='#02ab21', width=3),
        fill='tozeroy'
    ))
    
    # Hit rate (secondary axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['hit_rate'],
        name='Hit Rate (%)',
        line=dict(color='orange', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Performans Trendi',
        xaxis_title='Tarih',
        yaxis_title='Kümülatif Getiri (ATR)',
        yaxis2=dict(
            title='Hit Rate (%)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_breakdown_charts(by_type: list, by_interval: list):
    """Breakdown grafikleri."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sinyal Tipi Bazında")
        if by_type:
            df = pd.DataFrame(by_type)
            fig = px.bar(
                df,
                x='signal_type',
                y='hit_rate',
                color='avg_return_atr',
                text='total',
                title='Hit Rate by Signal Type',
                labels={'hit_rate': 'Hit Rate (%)', 'signal_type': 'Sinyal Tipi'},
                color_continuous_scale='RdYlGn'
            )
            fig.update_traces(texttemplate='%{text} sinyal', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Veri yok")
    
    with col2:
        st.subheader("Interval Bazında")
        if by_interval:
            df = pd.DataFrame(by_interval)
            fig = px.bar(
                df,
                x='interval',
                y='hit_rate',
                color='avg_return_atr',
                text='total',
                title='Hit Rate by Interval',
                labels={'hit_rate': 'Hit Rate (%)', 'interval': 'Interval'},
                color_continuous_scale='RdYlGn'
            )
            fig.update_traces(texttemplate='%{text} sinyal', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Veri yok")


def render_top_performers_table(top_performers: list):
    """En iyi performans tablosu."""
    if not top_performers:
        st.info("Henüz veri yok")
        return
    
    df = pd.DataFrame(top_performers)
    
    # Timestamp format
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Kolon sıralaması ve isimlendirme
    column_config = {
        'id': st.column_config.NumberColumn('ID', width='small'),
        'symbol': st.column_config.TextColumn('Sembol', width='small'),
        'signal_type': st.column_config.TextColumn('Sinyal Tipi', width='medium'),
        'indicators': st.column_config.TextColumn('📊 İndikatörler', width='large'),
        'vpm_score': st.column_config.NumberColumn('🎯 VPM', format='%.3f', width='small'),
        'interval': st.column_config.TextColumn('TF', width='small'),
        'timestamp': st.column_config.TextColumn('Zaman', width='medium'),
        'return_t5_atr': st.column_config.NumberColumn('T+5 (ATR)', format='%.2f', width='small'),
        'return_t5_pct': st.column_config.NumberColumn('T+5 (%)', format='%.2f%%', width='small'),
        'risk_reward': st.column_config.NumberColumn('R/R', format='%.2f', width='small'),
        'mfe_atr': st.column_config.NumberColumn('MFE', format='%.2f', width='small'),
        'mae_atr': st.column_config.NumberColumn('MAE', format='%.2f', width='small'),
    }
    
    # Kolon sıralaması
    column_order = ['id', 'symbol', 'signal_type', 'indicators', 'vpm_score', 'interval', 'timestamp', 
                    'return_t5_atr', 'return_t5_pct', 'risk_reward', 'mfe_atr', 'mae_atr']
    
    st.dataframe(
        df[column_order],
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=400
    )


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def render_signal_performance_dashboard():
    """Ana dashboard fonksiyonu."""
    
    st.header("📊 Signal Performance Dashboard")
    
    # Filtreler
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        days_back = st.slider(
            "Zaman Aralığı (Gün)",
            min_value=1,
            max_value=30,
            value=7,
            help="Son N günün verilerini göster"
        )
    
    with col2:
        # Interval filtresi
        selected_intervals = st.multiselect(
            "Interval Seç",
            options=["1m", "5m", "15m", "1h", "4h"],
            default=["5m", "15m"],
            help="Hangi timeframe'leri analiz etmek istiyorsun?"
        )
    
    with col3:
        auto_refresh = st.checkbox("Otomatik Yenile", value=False)
    
    with col4:
        if st.button("🔄 Yenile"):
            st.cache_data.clear()
            st.rerun()
    
    # Auto refresh
    if auto_refresh:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60000, key="performance_refresh")  # 60 saniye
    
    st.divider()
    
    # Interval filtresi (None = hepsi, [] = hiçbiri, list = seçilenler)
    interval_filter = selected_intervals if selected_intervals else None
    
    # KPI Kartları
    summary = get_performance_summary(days_back, interval_filter)
    render_kpi_cards(summary)
    
    st.divider()
    
    # Performans Trendi
    st.subheader("📈 Performans Trendi")
    timeseries = get_performance_timeseries(days_back, interval_filter)
    render_performance_chart(timeseries)
    
    st.divider()
    
    # Breakdown Grafikleri
    by_type = get_performance_by_signal_type(days_back, interval_filter)
    by_interval = get_performance_by_interval(days_back)  # Tüm interval'leri göster
    render_breakdown_charts(by_type, by_interval)
    
    st.divider()
    
    # Top Performers
    st.subheader("🏆 En İyi Performans Gösteren Sinyaller")
    top_performers = get_top_performers(days_back, limit=10, intervals=interval_filter)
    render_top_performers_table(top_performers)
    
    # Footer
    filter_info = f"Filtre: {', '.join(selected_intervals)}" if selected_intervals else "Filtre: Tüm interval'ler"
    st.caption(f"Son güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {filter_info}")
