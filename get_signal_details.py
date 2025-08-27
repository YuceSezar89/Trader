import asyncio
from sqlalchemy.future import select
from database.engine import async_session_maker as async_session
from database.models import Signal
import sys
import traceback

# Kullanıcı-dostu isimler (şema değişmeden)
ALIASES = {
    'perf_next_candle_momentum_change_pct': 'Sonraki Mum Momentum %',
    'perf_next_candle_volume_change_pct': 'Sonraki Mum Hacim %',
    'perf_intra_candle_profit_pct': 'Sinyal Mumu Max Kâr %',
    'perf_prev_to_signal_momentum_change_pct': 'Önceki→Sinyal Momentum %',
    'perf_prev_to_signal_volume_change_pct': 'Önceki→Sinyal Hacim %',
    'normalized_price_change': 'Normalize Fiyat Skoru',
}

async def get_signal_by_id(signal_id: int):
    """Fetches a signal from the database by its ID and prints its details."""
    try:
        async with async_session() as session:
            result = await session.execute(select(Signal).where(Signal.id == signal_id))
            signal = result.scalar_one_or_none()

            if signal:
                print(f"--- Sinyal Detayları (ID: {signal_id}) ---")
                details = signal.to_dict()
                for key, value in details.items():
                    label = ALIASES.get(key, key)
                    print(f"{label:<40}: {value}")
                print("---------------------------------")
            else:
                print(f"ID'si {signal_id} olan sinyal bulunamadı.")
    except Exception as e:
        print(f"Veritabanı işlemi sırasında bir hata oluştu: {e}")
        traceback.print_exc()

async def main():
    if len(sys.argv) > 1:
        try:
            signal_id_to_find = int(sys.argv[1])
            await get_signal_by_id(signal_id_to_find)
        except ValueError:
            print("Lütfen geçerli bir sinyal ID'si girin.")
    else:
        print("Kullanım: python get_signal_details.py <signal_id>")

if __name__ == "__main__":
    asyncio.run(main())

