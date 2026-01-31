import time
from time import time as now_time
from config import BOT_TOKEN
from telegram_api import get_updates, send_long_message
from lm import run_local_lm

chat_memory = {}
last_activity = {}

# 1 turn = user + assistant
MAX_TURNS = 18

INACTIVITY_RESET_MINUTES = 30
INACTIVITY_SECONDS = INACTIVITY_RESET_MINUTES * 60

# Manual reset keywords (LOWERCASE)
RESET_KEYWORDS = [
    "reset chat"
]

offset = None

print("AI Integration module has started. Waiting for messages...")

while True:
    try:
        updates = get_updates(BOT_TOKEN, offset)

        for update in updates.get("result", []):
            offset = update["update_id"] + 1

            message = update.get("message")
            if not message or "text" not in message:
                continue

            chat_id = message["chat"]["id"]
            text_clean = message["text"].strip()
            if not text_clean:
                continue

            print(f"📩 {chat_id}: {text_clean}")

            current_time = now_time()

            # ==============================
            # AUTO-RESET after inactivity
            # ==============================
            last_time = last_activity.get(chat_id)
            if last_time and (current_time - last_time) > INACTIVITY_SECONDS:
                print(f"⏳ Inactivity reset for chat {chat_id}")
                chat_memory[chat_id] = []

            last_activity[chat_id] = current_time

            # ==============================
            # MANUAL RESET
            # ==============================
            if text_clean.lower() in RESET_KEYWORDS:
                chat_memory[chat_id] = []
                send_long_message(
                    BOT_TOKEN,
                    chat_id,
                    "♻️ Chat memory has been reset."
                )
                continue  # 🚨 CRITICAL

            # ==============================
            # Load or initialize history
            # ==============================
            history = chat_memory.get(chat_id, [])

            # AUTO-RESET if too long
            if len(history) >= MAX_TURNS * 2:
                print(f"🔄 Length reset for chat {chat_id}")
                history = []
                chat_memory[chat_id] = []

            # Add user message
            history.append({
                "role": "user",
                "content": text_clean
            })

            # ==============================
            # Call local LM
            # ==============================
            reply = run_local_lm(history)

            # Save assistant reply
            history.append({
                "role": "assistant",
                "content": reply
            })

            chat_memory[chat_id] = history

            # Send reply
            send_long_message(BOT_TOKEN, chat_id, reply)

        time.sleep(1.5)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(8)
