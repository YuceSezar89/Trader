import asyncio
import sys
import os

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.engine import init_db

async def main():
    print("Veritabanı tabloları siliniyor ve yeniden oluşturuluyor...")
    await init_db()
    print("Veritabanı başarıyla sıfırlandı.")

if __name__ == "__main__":
    # Kullanıcı onayı al
    response = input("DİKKAT: Bu işlem 'signals.db' dosyasındaki tüm verileri kalıcı olarak silecektir. Devam etmek istiyor musunuz? (evet/hayır): ")
    if response.lower() == 'evet':
        asyncio.run(main())
    else:
        print("İşlem iptal edildi.")
