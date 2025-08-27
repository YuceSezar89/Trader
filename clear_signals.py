import asyncio
import sys
import os

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.crud import clear_all_signals

async def main():
    deleted = await clear_all_signals()
    print(f"Signals tablosu temizlendi. Silinen kayıt sayısı: {deleted}")

if __name__ == "__main__":
    resp = input("DİKKAT: Bu işlem 'signals' tablosundaki TÜM kayıtları kalıcı olarak silecek. Devam? (evet/hayır): ")
    if resp.strip().lower() == "evet":
        asyncio.run(main())
    else:
        print("İşlem iptal edildi.")
