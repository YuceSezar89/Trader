from __future__ import annotations

import json
from urllib.request import urlopen
from urllib.parse import urlencode
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd

from inceleme.pine_panel_equiv import compute_all_for_symbol

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

SYMBOLS: List[str] = [
    "BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XRPUSDT",
    "LTCUSDT","DOTUSDT","SOLUSDT","AVAXUSDT","TRXUSDT",
    "UNIUSDT","LINKUSDT","VETUSDT","XLMUSDT","NEARUSDT",
    "WIFUSDT","ZRXUSDT","ATOMUSDT","CAKEUSDT","KSMUSDT",
]

INTERVAL = "15m"
LIMIT = 500


def fetch_klines(symbol: str, interval: str = INTERVAL, limit: int = LIMIT) -> pd.DataFrame:
    params = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    with urlopen(f"{BINANCE_KLINES}?{params}") as resp:
        data = json.loads(resp.read().decode("utf-8"))
    cols = [
        'open_time','open','high','low','close','volume','close_time','quote_asset_volume',
        'trades','taker_buy_base','taker_buy_quote','ignore'
    ]
    df = pd.DataFrame(data, columns=cols)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dt'] = pd.to_datetime(df['open_time'], unit='ms', utc=True).dt.tz_convert('UTC')
    df.set_index('dt', inplace=True)
    return df


def main() -> None:
    rows: List[Dict] = []
    for sym in SYMBOLS:
        try:
            df = fetch_klines(sym)
            out = compute_all_for_symbol(df['close'], length_ema=200, r_period=14, rf=0.02)
            last = out.iloc[-1]
            rows.append({
                "symbol": sym,
                "time": out.index[-1].isoformat(),
                "price": float(out['close'].iloc[-1]),
                "ema": float(last['ema']) if pd.notna(last['ema']) else None,
                "ratio_percent": float(last['ratio_percent']) if pd.notna(last['ratio_percent']) else None,
                "mROC_long": float(last['mROC_long']) if pd.notna(last['mROC_long']) else None,
                "mROC_short": float(last['mROC_short']) if pd.notna(last['mROC_short']) else None,
                "msince_long": float(last['msince_long']) if pd.notna(last['msince_long']) else None,
                "msince_short": float(last['msince_short']) if pd.notna(last['msince_short']) else None,
                "r_score": float(last['r_score']) if pd.notna(last['r_score']) else None,
            })
        except Exception as e:
            rows.append({"symbol": sym, "error": str(e)})
    res_df = pd.DataFrame(rows)
    # Print compact table
    display_cols = ["symbol","price","ema","ratio_percent","mROC_long","mROC_short","msince_long","msince_short","r_score"]
    print(res_df[display_cols].to_string(index=False))
    # Save CSV
    out_path = "exports/pine_panel_15m.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
