"""
Telegram bildirimleri — Bot API'ye doğrudan HTTP isteği (ekstra kütüphane yok).

Trading akışını asla bloklamaz/çökertmez: hata sessizce loglanır.
"""
import logging

import aiohttp

from config import Config

logger = logging.getLogger(__name__)

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


async def send_telegram_message(text: str) -> None:
    """Yapılandırılmış Telegram kanalına mesaj gönderir. Hata durumunda sessizce loglar."""
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.debug("[Telegram] token/chat_id yapılandırılmamış, mesaj atlandı")
        return
    url = _API_URL.format(token=Config.TELEGRAM_BOT_TOKEN)
    payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": text}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("[Telegram] gönderim başarısız (%s): %s", resp.status, body)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("[Telegram] gönderim hatası: %s", exc)
