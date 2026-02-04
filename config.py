# ==============================
# Telegram Bot Configuration
# ==============================
# Get your bot token from @BotFather on Telegram
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

# Your Telegram user/chat ID
# Get it from: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
CHAT_ID = int("YOUR_CHAT_ID_HERE")

# ==============================
# LM Studio (BASE URL)
# ==============================
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

# ==============================
# Models
# ==============================
# Chat / reasoning model
MODEL_NAME = "qwen2.5-vl-7b-instruct-abliterated"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

# ===== Aliases expected by lm.py =====
CHAT_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/chat/completions"
CHAT_MODEL = MODEL_NAME
