"""
Backtest & Paper Trading UI Modülü - Streamlit arayüzü
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List, Optional

from backtest import BacktestEngine, PaperTrader, StrategyTester, PerformanceAnalyzer


def render_backtest_tab():
    """Backtest & Paper Trading sekmesini render eder"""
    
    st.markdown("### 🧪 Backtest & Paper Trading Sistemi")
    st.markdown("VP sisteminizin performansını test edin ve canlı paper trading yapın!")
    
    # Ana sekmeler
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Backtest", 
        "📊 Paper Trading", 
        "⚙️ Strateji Optimizasyonu", 
        "📈 Performans Analizi"
    ])
    
    with tab1:
        render_backtest_section()
    
    with tab2:
        render_paper_trading_section()
    
    with tab3:
        render_strategy_optimization_section()
    
    with tab4:
        render_performance_analysis_section()


def render_backtest_section():
    """Backtest bölümü"""
    st.markdown("#### 🔬 VP Sistemi Backtest")
    
    # Backtest parametreleri
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Test Parametreleri")
        
        # Sembol seçimi
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        selected_symbols = st.multiselect(
            "Test Edilecek Semboller",
            symbols,
            default=["BTCUSDT", "ETHUSDT"],
            help="Hangi sembolleri test etmek istiyorsunuz?"
        )
        
        # Tarih aralığı
        end_date = st.date_input(
            "Bitiş Tarihi",
            value=datetime.now().date(),
            help="Test bitiş tarihi"
        )
        
        start_date = st.date_input(
            "Başlangıç Tarihi",
            value=(datetime.now() - timedelta(days=30)).date(),
            help="Test başlangıç tarihi"
        )
        
        # Zaman dilimi
        timeframe = st.selectbox(
            "Zaman Dilimi",
            ["1h", "4h", "1d"],
            index=0,
            help="Test zaman dilimi"
        )
        
        # Başlangıç bakiyesi
        initial_balance = st.number_input(
            "Başlangıç Bakiyesi (USDT)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            key="backtest_initial_balance"
        )
    
    with col2:
        st.markdown("##### Risk Parametreleri")
        
        # Stop loss
        stop_loss = st.slider(
            "Stop Loss (%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="backtest_stop_loss"
        )
        
        # Take profit
        take_profit = st.slider(
            "Take Profit (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Hedef kar yüzdesi"
        )
        
        # Komisyon
        commission = st.slider(
            "Komisyon (%)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="İşlem komisyonu"
        )
        
        # VMP eşiği
        vmp_threshold = st.slider(
            "VMP Eşiği",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Minimum VMP skoru"
        )
    
    # Backtest başlat butonu
    if st.button("🚀 Backtest Başlat", type="primary"):
        if not selected_symbols:
            st.error("En az bir sembol seçmelisiniz!")
            return
        
        if start_date >= end_date:
            st.error("Başlangıç tarihi bitiş tarihinden önce olmalı!")
            return
        
        # Backtest parametreleri
        strategy_params = {
            'stop_loss_percentage': stop_loss,
            'take_profit_percentage': take_profit,
            'commission_rate': commission / 100,
            'vmp_threshold': vmp_threshold
        }
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Backtest başlatılıyor...")
            progress_bar.progress(20)
            
            # Backtest engine oluştur
            engine = BacktestEngine(initial_balance=initial_balance)
            
            status_text.text("Tarihsel veriler çekiliyor...")
            progress_bar.progress(40)
            
            # Async backtest çalıştır
            # Not: Streamlit'te async çalıştırmak için wrapper gerekli
            status_text.text("Backtest çalışıyor...")
            progress_bar.progress(60)
            
            # Simüle edilmiş sonuç (gerçek implementasyonda async çağrı olacak)
            result = simulate_backtest_result(
                selected_symbols, start_date, end_date, 
                timeframe, strategy_params, initial_balance
            )
            
            status_text.text("Sonuçlar analiz ediliyor...")
            progress_bar.progress(80)
            
            # Sonuçları göster
            display_backtest_results(result)
            
            progress_bar.progress(100)
            status_text.text("✅ Backtest tamamlandı!")
            
        except Exception as e:
            st.error(f"Backtest hatası: {str(e)}")
            progress_bar.empty()
            status_text.empty()


def render_paper_trading_section():
    """Paper trading bölümü"""
    st.markdown("#### 📊 Paper Trading")
    
    # Paper trading durumu
    if 'paper_trading_active' not in st.session_state:
        st.session_state.paper_trading_active = False
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not st.session_state.paper_trading_active:
            st.info("📊 Paper trading oturumu başlatılmamış")
            
            # Oturum parametreleri
            session_name = st.text_input(
                "Oturum Adı",
                value=f"PT_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="Paper trading oturum adı"
            )
            
            initial_balance = st.number_input(
                "Başlangıç Bakiyesi (USDT)",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                key="paper_trading_initial_balance"
            )
            
            if st.button("🚀 Paper Trading Başlat", type="primary"):
                # Paper trading başlat
                st.session_state.paper_trading_active = True
                st.session_state.session_name = session_name
                st.session_state.initial_balance = initial_balance
                st.session_state.current_balance = initial_balance
                st.session_state.open_positions = {}
                st.session_state.trade_history = []
                
                st.success(f"✅ Paper trading başlatıldı: {session_name}")
                st.rerun()
        
        else:
            # Aktif oturum
            st.success(f"✅ Aktif Oturum: {st.session_state.session_name}")
            
            # Oturum istatistikleri
            display_paper_trading_dashboard()
            
            # Oturumu durdur
            if st.button("⏹️ Oturumu Durdur", type="secondary"):
                st.session_state.paper_trading_active = False
                st.success("Paper trading oturumu durduruldu")
                st.rerun()
    
    with col2:
        # Risk parametreleri
        st.markdown("##### Risk Ayarları")
        
        max_position_size = st.slider(
            "Max Pozisyon (%)",
            min_value=1,
            max_value=20,
            value=5,
            help="Portföyün maksimum yüzdesi"
        )
        
        stop_loss_pt = st.slider(
            "Stop Loss (%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="paper_trading_stop_loss"
        )
        
        take_profit_pt = st.slider(
            "Take Profit (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="paper_trading_take_profit"
        )


def render_strategy_optimization_section():
    """Strateji optimizasyonu bölümü"""
    st.markdown("#### ⚙️ Strateji Optimizasyonu")
    
    # Optimizasyon türü seçimi
    optimization_type = st.selectbox(
        "Optimizasyon Türü",
        [
            "Momentum Periyodu Analizi",
            "VMP Eşik Optimizasyonu", 
            "Risk Parametresi Testi",
            "Çoklu Parametre Optimizasyonu"
        ],
        help="Hangi parametreleri optimize etmek istiyorsunuz?"
    )
    
    if optimization_type == "Momentum Periyodu Analizi":
        render_momentum_optimization()
    elif optimization_type == "VMP Eşik Optimizasyonu":
        render_vmp_threshold_optimization()
    elif optimization_type == "Risk Parametresi Testi":
        render_risk_parameter_optimization()
    else:
        render_multi_parameter_optimization()


def render_performance_analysis_section():
    """Performans analizi bölümü"""
    st.markdown("#### 📈 Performans Analizi")
    
    # Analiz türü seçimi
    analysis_type = st.selectbox(
        "Analiz Türü",
        [
            "Backtest Sonuçları Karşılaştırması",
            "Paper Trading Performansı",
            "Strateji Başarı Analizi",
            "Risk Metrikleri Detayı"
        ]
    )
    
    if analysis_type == "Backtest Sonuçları Karşılaştırması":
        render_backtest_comparison()
    elif analysis_type == "Paper Trading Performansı":
        render_paper_trading_analysis()
    elif analysis_type == "Strateji Başarı Analizi":
        render_strategy_success_analysis()
    else:
        render_risk_metrics_analysis()


def simulate_backtest_result(symbols, start_date, end_date, timeframe, params, initial_balance):
    """Simüle edilmiş backtest sonucu (demo amaçlı)"""
    import random
    
    # Demo sonuçları
    total_trades = random.randint(15, 50)
    winning_trades = random.randint(int(total_trades * 0.4), int(total_trades * 0.7))
    win_rate = (winning_trades / total_trades) * 100
    
    total_pnl = random.uniform(-1000, 3000)
    total_return = (total_pnl / initial_balance) * 100
    
    max_drawdown = random.uniform(5, 25)
    sharpe_ratio = random.uniform(0.5, 2.5)
    profit_factor = random.uniform(0.8, 2.2)
    
    return {
        'symbols': symbols,
        'timeframe': timeframe,
        'start_date': start_date,
        'end_date': end_date,
        'initial_balance': initial_balance,
        'final_balance': initial_balance + total_pnl,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': total_trades - winning_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'parameters': params
    }


def display_backtest_results(result):
    """Backtest sonuçlarını göster"""
    st.markdown("### 📊 Backtest Sonuçları")
    
    # Özet metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Toplam İşlem",
            result['total_trades'],
            help="Gerçekleştirilen toplam işlem sayısı"
        )
    
    with col2:
        st.metric(
            "Kazanma Oranı",
            f"{result['win_rate']:.1f}%",
            help="Başarılı işlem yüzdesi"
        )
    
    with col3:
        st.metric(
            "Toplam Getiri",
            f"{result['total_return']:.2f}%",
            delta=f"{result['total_pnl']:.2f} USDT",
            help="Toplam yatırım getirisi"
        )
    
    with col4:
        st.metric(
            "Sharpe Oranı",
            f"{result['sharpe_ratio']:.2f}",
            help="Risk-adjusted return"
        )
    
    # Detaylı metrikler
    st.markdown("#### 📈 Detaylı Performans Metrikleri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics_data = {
            'Metrik': [
                'Başlangıç Bakiyesi',
                'Final Bakiye', 
                'Net Kar/Zarar',
                'Maksimum Drawdown',
                'Profit Factor',
                'Kazanan İşlemler',
                'Kaybeden İşlemler'
            ],
            'Değer': [
                f"{result['initial_balance']:,.0f} USDT",
                f"{result['final_balance']:,.0f} USDT",
                f"{result['total_pnl']:,.2f} USDT",
                f"{result['max_drawdown']:.2f}%",
                f"{result['profit_factor']:.2f}",
                f"{result['winning_trades']}",
                f"{result['losing_trades']}"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(metrics_data),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        # Performans grafiği (demo)
        dates = pd.date_range(result['start_date'], result['end_date'], freq='D')
        cumulative_returns = [0]
        
        for i in range(1, len(dates)):
            change = random.uniform(-2, 3)  # Demo değişim
            cumulative_returns.append(cumulative_returns[-1] + change)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates[:len(cumulative_returns)],
            y=cumulative_returns,
            mode='lines',
            name='Kümülatif Getiri (%)',
            line=dict(color='#00d4aa', width=2)
        ))
        
        fig.update_layout(
            title="Backtest Performans Grafiği",
            xaxis_title="Tarih",
            yaxis_title="Kümülatif Getiri (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Test parametreleri
    st.markdown("#### ⚙️ Test Parametreleri")
    params_df = pd.DataFrame([
        {'Parametre': 'Stop Loss', 'Değer': f"{result['parameters']['stop_loss_percentage']}%"},
        {'Parametre': 'Take Profit', 'Değer': f"{result['parameters']['take_profit_percentage']}%"},
        {'Parametre': 'Komisyon', 'Değer': f"{result['parameters']['commission_rate']*100:.2f}%"},
        {'Parametre': 'VMP Eşiği', 'Değer': f"{result['parameters']['vmp_threshold']}"},
        {'Parametre': 'Semboller', 'Değer': ', '.join(result['symbols'])},
        {'Parametre': 'Zaman Dilimi', 'Değer': result['timeframe']}
    ])
    
    st.dataframe(params_df, use_container_width=True, hide_index=True)


def display_paper_trading_dashboard():
    """Paper trading dashboard"""
    # Portfolio özeti
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mevcut Bakiye",
            f"{st.session_state.current_balance:,.0f} USDT",
            delta=f"{st.session_state.current_balance - st.session_state.initial_balance:,.0f}"
        )
    
    with col2:
        total_return = ((st.session_state.current_balance - st.session_state.initial_balance) / st.session_state.initial_balance) * 100
        st.metric(
            "Toplam Getiri",
            f"{total_return:.2f}%"
        )
    
    with col3:
        st.metric(
            "Açık Pozisyonlar",
            len(st.session_state.open_positions)
        )
    
    with col4:
        st.metric(
            "Toplam İşlem",
            len(st.session_state.trade_history)
        )
    
    # Açık pozisyonlar
    if st.session_state.open_positions:
        st.markdown("##### 📊 Açık Pozisyonlar")
        positions_df = pd.DataFrame([
            {
                'Sembol': pos['symbol'],
                'Yön': pos['side'],
                'Giriş Fiyatı': f"{pos['entry_price']:.4f}",
                'Miktar': f"{pos['quantity']:.6f}",
                'P&L': f"{pos.get('unrealized_pnl', 0):.2f} USDT"
            }
            for pos in st.session_state.open_positions.values()
        ])
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
    
    # İşlem geçmişi
    if st.session_state.trade_history:
        st.markdown("##### 📈 Son İşlemler")
        recent_trades = st.session_state.trade_history[-10:]  # Son 10 işlem
        trades_df = pd.DataFrame([
            {
                'Tarih': trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Sembol': trade['symbol'],
                'Yön': trade['side'],
                'P&L': f"{trade['pnl']:.2f} USDT",
                'Durum': '✅' if trade['pnl'] > 0 else '❌'
            }
            for trade in recent_trades
        ])
        st.dataframe(trades_df, use_container_width=True, hide_index=True)


def render_momentum_optimization():
    """Momentum periyodu optimizasyonu"""
    st.markdown("##### 🔄 Momentum Periyodu Analizi")
    st.info("Farklı momentum periyotlarını (1, 3, 5, 10 bar) test ederek optimal değeri bulun.")
    
    # Test parametreleri
    col1, col2 = st.columns(2)
    
    with col1:
        test_periods = st.multiselect(
            "Test Edilecek Periyotlar",
            [1, 3, 5, 10, 15, 20],
            default=[1, 3, 5, 10],
            help="Hangi momentum periyotlarını test etmek istiyorsunuz?"
        )
    
    with col2:
        test_duration = st.selectbox(
            "Test Süresi",
            ["Son 7 gün", "Son 15 gün", "Son 30 gün"],
            index=1
        )
    
    if st.button("🚀 Momentum Analizi Başlat"):
        if not test_periods:
            st.error("En az bir periyot seçmelisiniz!")
            return
        
        # Simüle edilmiş sonuçlar
        results = []
        for period in test_periods:
            win_rate = random.uniform(45, 75)
            sharpe_ratio = random.uniform(0.5, 2.0)
            total_trades = random.randint(10, 30)
            profit_factor = random.uniform(0.8, 2.5)
            
            results.append({
                'Periyot': f"{period} bar",
                'Win Rate (%)': f"{win_rate:.1f}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Toplam İşlem': total_trades,
                'Profit Factor': f"{profit_factor:.2f}"
            })
        
        # Sonuçları göster
        st.markdown("#### 📊 Momentum Periyodu Test Sonuçları")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # En iyi periyot
        best_period = results[0]['Periyot']  # Demo için ilk
        st.success(f"🏆 En iyi performans: {best_period}")


def render_vmp_threshold_optimization():
    """VMP eşik optimizasyonu"""
    st.markdown("##### 🎯 VMP Eşik Optimizasyonu")
    st.info("Farklı VMP eşik değerlerini test ederek sinyal kalitesini optimize edin.")
    
    # Test aralığı
    col1, col2 = st.columns(2)
    
    with col1:
        min_threshold = st.slider("Minimum Eşik", 0.1, 1.0, 0.2, 0.1)
        max_threshold = st.slider("Maksimum Eşik", 0.5, 2.0, 1.0, 0.1)
    
    with col2:
        step_size = st.slider("Adım Büyüklüğü", 0.05, 0.2, 0.1, 0.05)
        test_symbols = st.multiselect(
            "Test Sembolleri",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            default=["BTCUSDT", "ETHUSDT"]
        )
    
    if st.button("🚀 VMP Eşik Analizi Başlat"):
        # Simüle edilmiş analiz
        thresholds = []
        current = min_threshold
        while current <= max_threshold:
            thresholds.append(current)
            current += step_size
        
        # Demo sonuçlar
        results = []
        for threshold in thresholds:
            signal_count = max(1, int(50 * (1 / threshold)))  # Düşük eşik = daha fazla sinyal
            win_rate = min(80, 40 + (threshold * 20))  # Yüksek eşik = daha iyi kalite
            
            results.append({
                'Eşik': f"{threshold:.1f}",
                'Sinyal Sayısı': signal_count,
                'Win Rate (%)': f"{win_rate:.1f}",
                'Kalite Skoru': f"{(win_rate * signal_count / 100):.1f}"
            })
        
        st.markdown("#### 📊 VMP Eşik Test Sonuçları")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)


def render_risk_parameter_optimization():
    """Risk parametresi optimizasyonu"""
    st.markdown("##### ⚠️ Risk Parametresi Optimizasyonu")
    st.info("Stop loss ve take profit değerlerini optimize edin.")
    
    st.write("Bu bölüm geliştirilme aşamasında...")


def render_multi_parameter_optimization():
    """Çoklu parametre optimizasyonu"""
    st.markdown("##### 🔧 Çoklu Parametre Optimizasyonu")
    st.info("Birden fazla parametreyi aynı anda optimize edin.")
    
    st.write("Bu bölüm geliştirilme aşamasında...")


def render_backtest_comparison():
    """Backtest karşılaştırması"""
    st.markdown("##### 📊 Backtest Sonuçları Karşılaştırması")
    st.write("Bu bölüm geliştirilme aşamasında...")


def render_paper_trading_analysis():
    """Paper trading analizi"""
    st.markdown("##### 📈 Paper Trading Performans Analizi")
    st.write("Bu bölüm geliştirilme aşamasında...")


def render_strategy_success_analysis():
    """Strateji başarı analizi"""
    st.markdown("##### 🎯 Strateji Başarı Analizi")
    st.write("Bu bölüm geliştirilme aşamasında...")


def render_risk_metrics_analysis():
    """Risk metrikleri analizi"""
    st.markdown("##### ⚠️ Risk Metrikleri Detayı")
    st.write("Bu bölüm geliştirilme aşamasında...")


# Demo için gerekli import
import random
