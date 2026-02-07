import requests
import os

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

def get_file(bot_token, file_id):
    """
    Get file metadata from Telegram.
    
    Returns:
        dict with 'file_path', 'file_size', etc.
    """
    url = f"{BASE_URL}{bot_token}/getFile"

    response = requests.get(url, params={"file_id": file_id}, timeout=30)
    response.raise_for_status()

    return response.json()['result']

def download_file(bot_token, file_path, save_path):
    """
    Download file from Telegram servers.
    
    Args:
        file_path: Telegram file_path from getFile
        save_path: Local path to save file
    
    Returns:
        True if successful
    """
    url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return True
