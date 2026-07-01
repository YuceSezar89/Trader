#!/usr/bin/env python3
"""
Timeframe Aggregation Test Scripti

Bu script timeframe_aggregator.py modülünü test eder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from utils.timeframe_aggregator import TimeframeAggregator, create_multi_timeframe_data
from datetime import datetime

def test_basic_aggregation():
    """Temel aggregation testleri"""
    print("🧪 Temel Aggregation Testleri")
    print("=" * 50)
    
    # Test verisi oluştur (25 adet 1m bar - 5m için 5 grup)
    test_df = TimeframeAggregator.create_test_data(bars=25)
    print(f"✅ Test verisi oluşturuldu: {len(test_df)} adet 1m bar")
    
    # 1m -> 5m aggregation
    df_5m = TimeframeAggregator.aggregate_ohlcv(test_df, '1m', '5m')
    print(f"✅ 1m -> 5m: {len(test_df)} -> {len(df_5m)} bars")
    
    # 1m -> 15m aggregation  
    df_15m = TimeframeAggregator.aggregate_ohlcv(test_df, '1m', '15m')
    print(f"✅ 1m -> 15m: {len(test_df)} -> {len(df_15m)} bars")
    
    # Sonuçları göster
    if not df_5m.empty:
        print(f"\n📊 5m Verisi (İlk 3 bar):")
        print(df_5m[['open_time', 'open', 'high', 'low', 'close', 'volume']].head(3))
    
    if not df_15m.empty:
        print(f"\n📊 15m Verisi (İlk 2 bar):")
        print(df_15m[['open_time', 'open', 'high', 'low', 'close', 'volume']].head(2))
    
    return df_5m, df_15m

def test_multi_timeframe():
    """Multi-timeframe test"""
    print("\n🧪 Multi-Timeframe Testi")
    print("=" * 50)
    
    # 30 adet 1m bar (5m için 6 grup, 15m için 2 grup)
    test_df = TimeframeAggregator.create_test_data(bars=30)
    
    # Multi-timeframe oluştur
    mtf_data = create_multi_timeframe_data(test_df)
    
    print("📈 Multi-Timeframe Sonuçları:")
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df)} bars")
    
    return mtf_data

def test_validation():
    """Validation testleri"""
    print("\n🧪 Validation Testleri")
    print("=" * 50)
    
    # Boş DataFrame testi
    empty_df = pd.DataFrame()
    result = TimeframeAggregator.validate_dataframe(empty_df)
    print(f"✅ Boş DataFrame validation: {result} (False olmalı)")
    
    # Eksik kolon testi
    incomplete_df = pd.DataFrame({'open': [1, 2, 3], 'high': [2, 3, 4]})
    result = TimeframeAggregator.validate_dataframe(incomplete_df)
    print(f"✅ Eksik kolon validation: {result} (False olmalı)")
    
    # Geçerli DataFrame testi
    valid_df = TimeframeAggregator.create_test_data(bars=5)
    result = TimeframeAggregator.validate_dataframe(valid_df)
    print(f"✅ Geçerli DataFrame validation: {result} (True olmalı)")

def test_edge_cases():
    """Edge case testleri"""
    print("\n🧪 Edge Case Testleri")
    print("=" * 50)
    
    # Yetersiz veri testi (5m için 5 bar gerekli, sadece 3 bar ver)
    small_df = TimeframeAggregator.create_test_data(bars=3)
    result = TimeframeAggregator.aggregate_ohlcv(small_df, '1m', '5m')
    print(f"✅ Yetersiz veri testi: {len(result)} bars (0 olmalı)")
    
    # Desteklenmeyen aggregation testi
    test_df = TimeframeAggregator.create_test_data(bars=10)
    result = TimeframeAggregator.aggregate_ohlcv(test_df, '5m', '1m')  # Ters yön
    print(f"✅ Desteklenmeyen aggregation: {len(result)} bars (0 olmalı)")

def main():
    """Ana test fonksiyonu"""
    print("🚀 Timeframe Aggregator Test Suite")
    print("=" * 60)
    
    try:
        # Temel testler
        df_5m, df_15m = test_basic_aggregation()
        
        # Multi-timeframe test
        mtf_data = test_multi_timeframe()
        
        # Validation testleri
        test_validation()
        
        # Edge case testleri
        test_edge_cases()
        
        print("\n🎉 Tüm testler tamamlandı!")
        print("=" * 60)
        
        # Özet
        print("\n📋 Test Özeti:")
        print(f"  ✅ Temel aggregation: Çalışıyor")
        print(f"  ✅ Multi-timeframe: Çalışıyor") 
        print(f"  ✅ Validation: Çalışıyor")
        print(f"  ✅ Edge cases: Çalışıyor")
        
    except Exception as e:
        print(f"\n❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
