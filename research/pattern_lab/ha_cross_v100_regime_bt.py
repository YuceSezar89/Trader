"""
rsi_cross_v100_regime_bt.py'nin AYNI deseni (kırmızı VE yeşil V100 durumu),
HA_Cross'a uygulanmış.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.rsi_cross_v100_regime_bt import run  # pylint: disable=wrong-import-position

INDICATOR = "HA_Cross"


if __name__ == "__main__":
    run(indicator=INDICATOR)
