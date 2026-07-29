-- ============================================================
-- Migration 024: trade_snapshots'a vp_score_real eklendi
--
-- trade_snapshot.py'deki vp_buy/vp_sell/vp_score her zaman geometrik
-- proxy hacimle hesaplanıyordu (compute_vp_score varsayılan
-- use_real_volume=False), sinyal anındaki (entry_features) dual-track
-- tasarımdan (proxy vp_score + gerçek vp_score_real, ayrı kolonlar)
-- farklıydı. Migration 023 ile CA view'lar artık buy_volume/sell_volume
-- taşıdığı için sürdürme fazında da gerçek hacimle hesaplama mümkün —
-- aynı deseni buraya taşıyoruz.
-- ============================================================

ALTER TABLE trade_snapshots ADD COLUMN IF NOT EXISTS vp_score_real DOUBLE PRECISION;
