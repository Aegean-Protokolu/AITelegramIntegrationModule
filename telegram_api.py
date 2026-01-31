import requests

BASE_URL = "https://api.telegram.org/bot"

def get_updates(bot_token, offset=None, timeout=40):
    url = f"{BASE_URL}{bot_token}/getUpdates"

    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset

    response = requests.get(url, params=params, timeout=timeout + 5)
    response.raise_for_status()

    return response.json()

def send_message(bot_token, chat_id, text):
    url = f"{BASE_URL}{bot_token}/sendMessage"

    response = requests.post(url, json={
        "chat_id": chat_id,
        "text": text
    }, timeout=30)

    response.raise_for_status()

def send_long_message(bot_token, chat_id, text, limit=3072):
    """
    Splits long messages into chunks safe for Telegram (4096 char limit).
    """
    if not text:
        return

    for i in range(0, len(text), limit):
        chunk = text[i:i + limit]
        send_message(bot_token, chat_id, chunk)
