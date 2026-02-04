import re
import time
from time import time as now_time
from typing import Dict, List

from config import BOT_TOKEN
from telegram_api import get_updates, send_long_message

from lm import run_local_lm, extract_memory_candidates, compute_embedding, SYSTEM_PROMPT
from memory import MemoryStore
from meta_memory import MetaMemoryStore
from reasoning import ReasoningStore
from memory_arbiter import MemoryArbiter
from memory_consolidator import MemoryConsolidator

# ==============================
# Runtime state
# ==============================

chat_memory: Dict[int, List[dict]] = {}
last_activity: Dict[int, float] = {}

MAX_TURNS = 18  # user + assistant

INACTIVITY_RESET_MINUTES = 30
INACTIVITY_SECONDS = INACTIVITY_RESET_MINUTES * 60

RESET_CHAT = {"/resetchat", "/chatreset", "/clearchat"}
RESET_MEMORY = {"/resetmemory", "/memoryreset", "/clearmemory"}
RESET_REASONING = {"/resetreasoning", "/reasoningreset", "/clearreasoning"}
RESET_META_MEMORY = {"/resetmetamemory", "/metamemoryreset", "/clearmetamemory"}
RESET_ALL = {"/resetall", "/clearall"}

# Type-specific memory removal commands
REMOVE_IDENTITY = {"/removeidentity", "/clearidentity", "/deleteidentity"}
REMOVE_FACT = {"/removefact", "/clearfact", "/deletefact", "/removefacts", "/clearfacts"}
REMOVE_PREFERENCE = {"/removepreference", "/clearpreference", "/deletepreference", "/removepreferences", "/clearpreferences"}
REMOVE_GOAL = {"/removegoal", "/cleargoal", "/deletegoal", "/removegoals", "/cleargoals"}
REMOVE_BELIEF = {"/removebelief", "/clearbelief", "/deletebelief", "/removebeliefs", "/clearbeliefs"}
REMOVE_PERMISSION = {"/removepermission", "/clearpermission", "/deletepermission", "/removepermissions", "/clearpermissions"}
REMOVE_RULE = {"/removerule", "/clearrule", "/deleterule", "/removerules", "/clearrules"}

offset = None

# Consolidation scheduling
last_consolidation_time = 0
CONSOLIDATION_INTERVAL_SECONDS = 600  # Run every 10 minutes (was 3600 = 1 hour)

# ==============================
# Core components
# ==============================

memory_store = MemoryStore(db_path="./data/memory.sqlite3")
meta_memory_store = MetaMemoryStore(db_path="./data/meta_memory.sqlite3")
reasoning_store = ReasoningStore(embed_fn=compute_embedding)
arbiter = MemoryArbiter(memory_store, meta_memory_store=meta_memory_store)
consolidator = MemoryConsolidator(memory_store, meta_memory_store=meta_memory_store, similarity_threshold=0.85)

# ==============================
# Initial memory consolidation
# ==============================

print("🧠 [Consolidator] Running initial consolidation on startup...")
stats = consolidator.consolidate(time_window_hours=None)
print(f"🧠 [Consolidator] Initial: Processed {stats['processed']}, Consolidated {stats['consolidated']}, Skipped {stats['skipped']}")

print("🤖 Bot started. Waiting for messages...")

# ==============================
# Main loop
# ==============================

while True:
    try:
        updates = get_updates(BOT_TOKEN, offset)

        for update in updates.get("result", []):
            offset = update["update_id"] + 1

            message = update.get("message")
            if not message or "text" not in message:
                continue

            chat_id = message["chat"]["id"]
            text = message["text"].strip()
            if not text:
                continue

            print(f"📩 {chat_id}: {text}")

            now = now_time()

            # ==============================
            # Periodic memory consolidation
            # ==============================

            # last_consolidation_time is module-level, can be updated directly
            if now - last_consolidation_time > CONSOLIDATION_INTERVAL_SECONDS:
                print("🧠 [Consolidator] Running periodic consolidation...")
                # Consolidate ALL memories (time_window_hours=None means all)
                stats = consolidator.consolidate(time_window_hours=None)
                print(f"🧠 [Consolidator] Processed: {stats['processed']}, Consolidated: {stats['consolidated']}, Skipped: {stats['skipped']}")
                last_consolidation_time = now

            # ==============================
            # Inactivity reset (chat only)
            # ==============================

            last = last_activity.get(chat_id)
            if last and (now - last) > INACTIVITY_SECONDS:
                chat_memory[chat_id] = []

            last_activity[chat_id] = now

            # Normalize command: lowercase and strip whitespace
            cmd = text.lower().strip()

            # ==============================
            # RESET COMMANDS
            # ==============================

            if cmd in RESET_CHAT:
                chat_memory[chat_id] = []
                send_long_message(BOT_TOKEN, chat_id, "♻️ Chat history cleared.")
                continue

            if cmd in RESET_MEMORY:
                memory_store.clear()
                send_long_message(BOT_TOKEN, chat_id, "🧠 Long-term memory wiped.")
                continue

            if cmd in RESET_REASONING:
                reasoning_store.clear()
                send_long_message(BOT_TOKEN, chat_id, "🧩 Reasoning buffer cleared.")
                continue

            if cmd in RESET_META_MEMORY:
                meta_memory_store.clear()
                send_long_message(BOT_TOKEN, chat_id, "🧠 Meta-memories cleared.")
                continue

            if cmd in RESET_ALL:
                chat_memory[chat_id] = []
                reasoning_store.clear()
                memory_store.clear()
                meta_memory_store.clear()
                send_long_message(
                    BOT_TOKEN,
                    chat_id,
                    "🔥 Full reset complete (chat + reasoning + memory + meta-memory)."
                )
                continue

            # ==============================
            # Manual consolidation command
            # ==============================

            if cmd in {"/consolidate", "/consolidatenow"}:
                print("🧠 [Consolidator] Manual consolidation triggered...")
                stats = consolidator.consolidate(time_window_hours=None)
                send_long_message(
                    BOT_TOKEN,
                    chat_id,
                    f"🧠 Consolidation complete: Processed {stats['processed']}, Consolidated {stats['consolidated']}, Skipped {stats['skipped']}."
                )
                continue

            # ==============================
            # TYPE-SPECIFIC MEMORY REMOVAL
            # ==============================

            if cmd in REMOVE_IDENTITY:
                count = memory_store.clear_by_type("IDENTITY")
                send_long_message(BOT_TOKEN, chat_id, f"🗑️ Removed {count} IDENTITY memories.")
                continue

            if cmd in REMOVE_FACT:
                count = memory_store.clear_by_type("FACT")
                send_long_message(BOT_TOKEN, chat_id, f"🗑️ Removed {count} FACT memories.")
                continue

            if cmd in REMOVE_PREFERENCE:
                count = memory_store.clear_by_type("PREFERENCE")
                send_long_message(BOT_TOKEN, chat_id, f"🗑️ Removed {count} PREFERENCE memories.")
                continue

            if cmd in REMOVE_GOAL:
                count = memory_store.clear_by_type("GOAL")
                send_long_message(BOT_TOKEN, chat_id, f"🗑️ Removed {count} GOAL memories.")
                continue

            if cmd in REMOVE_BELIEF:
                count = memory_store.clear_by_type("BELIEF")
                send_long_message(BOT_TOKEN, chat_id, f"🗑️ Removed {count} BELIEF memories.")
                continue

            if cmd in REMOVE_PERMISSION:
                count = memory_store.clear_by_type("PERMISSION")
                send_long_message(BOT_TOKEN, chat_id, f"🗑️ Removed {count} PERMISSION memories.")
                continue

            if cmd in REMOVE_RULE:
                count = memory_store.clear_by_type("RULE")
                send_long_message(BOT_TOKEN, chat_id, f"🗑️ Removed {count} RULE memories.")
                continue

            # ==============================
            # Show memories
            # ==============================

            if cmd == "/memories":
                items = memory_store.list_recent(limit=30)
                if not items:
                    send_long_message(BOT_TOKEN, chat_id, "🧠 No saved memories.")
                else:
                    # Sort by hierarchy: PERMISSION → RULE → IDENTITY → PREFERENCE → FACT → BELIEF
                    hierarchy = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "FACT", "BELIEF"]
                    type_emoji = {
                        "IDENTITY": "👤",
                        "FACT": "📌",
                        "PREFERENCE": "❤️",
                        "GOAL": "🎯",
                        "RULE": "⚖️",
                        "PERMISSION": "✅",
                        "BELIEF": "💭"
                    }

                    # Group by type and sort by hierarchy
                    grouped = {}
                    for (_id, mem_type, subject, text) in items:
                        if mem_type not in grouped:
                            grouped[mem_type] = []
                        grouped[mem_type].append((subject, text))

                    lines = []
                    # Standard hierarchy
                    hierarchy = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]
                    
                    for mem_type in hierarchy:
                        if mem_type in grouped:
                            emoji = type_emoji.get(mem_type, "💡")
                            lines.append(f"\n{emoji} {mem_type}:")
                            for subject, text in grouped[mem_type]:
                                lines.append(f"  - [{subject}] {text}")
                            del grouped[mem_type]

                    # Remaining types
                    for mem_type, remaining_items in grouped.items():
                        emoji = type_emoji.get(mem_type, "💡")
                        lines.append(f"\n{emoji} {mem_type}:")
                        for subject, text in remaining_items:
                            lines.append(f"  - [{subject}] {text}")

                    send_long_message(
                        BOT_TOKEN,
                        chat_id,
                        "🧠 Saved Memories :\n" + "\n".join(lines),
                    )
                continue

            # ==============================
            # Show meta-memories
            # ==============================

            if cmd in {"/metamemories", "/meta-memories"}:
                items = meta_memory_store.list_recent(limit=30)
                if not items:
                    send_long_message(BOT_TOKEN, chat_id, "🧠 No meta-memories.")
                else:
                    lines = []
                    for (_id, event_type, subject, text, created_at) in items:
                        # Format with event type emoji
                        event_emoji = {
                            "MEMORY_CREATED": "✨",
                            "VERSION_UPDATE": "🔄",
                            "CONFLICT_DETECTED": "⚠️",
                            "CONSOLIDATION": "🔗"
                        }.get(event_type, "🧠")

                        lines.append(f"{event_emoji} [{subject}] {text}")

                    send_long_message(
                        BOT_TOKEN,
                        chat_id,
                        "🧠 Meta-Memories (Reflections):\n" + "\n".join(lines),
                    )
                continue

            # ==============================
            # Load chat history
            # ==============================

            history = chat_memory.get(chat_id, [])
            if len(history) >= MAX_TURNS * 2:
                history = history[-(MAX_TURNS * 2):]

            history.append({"role": "user", "content": text})

            # ==============================
            # Prepare memory context
            # ==============================

            # Load ONLY latest memories (consolidated data)
            # Do NOT include old linked versions in default context
            # This prevents AI from pulling outdated data
            # For old versions: AI can explicitly query memory history if needed
            memory_items = memory_store.list_recent(limit=10)
            memory_context = ""
            if memory_items:
                memory_lines = [f"- [{subject}] {mem_text}" for (_id, _type, subject, mem_text) in memory_items]
                memory_context = "Known facts (current - consolidated):\n" + "\n".join(memory_lines) + "\n\n"

            # Optional: Include recent meta-memories for self-awareness
            # This allows the AI to know about its own changes
            meta_items = meta_memory_store.list_recent(limit=3)
            if meta_items:
                meta_lines = [f"- {text}" for (_id, _event, _subj, text, _time) in meta_items]
                memory_context += "Recent changes (self-reflection):\n" + "\n".join(meta_lines) + "\n\n"

            # ==============================
            # Call LM
            # ==============================

            # Enhance system prompt with memory context
            enhanced_system_prompt = SYSTEM_PROMPT
            if memory_context:
                enhanced_system_prompt = memory_context + SYSTEM_PROMPT

            # The LM is only used to respond; we will not store external facts.
            reply = run_local_lm(history, system_prompt=enhanced_system_prompt)
            history.append({"role": "assistant", "content": reply})
            chat_memory[chat_id] = history

            send_long_message(BOT_TOKEN, chat_id, reply)

            # ==============================
            # Memory candidate extraction
            # ==============================

            # ONLY extract candidates from LLM (no auto-saving user text)
            candidates = extract_memory_candidates(
                user_text=text,
                assistant_text=reply
            )

            print(f"🔍 [Debug] Extracted {len(candidates)} candidate(s): {candidates}")

            # Add source metadata and filter by confidence
            for c in candidates:
                c["source"] = "assistant"
                c["confidence"] = c.get("confidence", 0.9)  # default high confidence (0.9 for PERMISSION)

            print(f"🔍 [Debug] After metadata: {candidates}")

            # Filter: skip low-confidence or empty candidates
            candidates = [c for c in candidates if c.get("confidence", 0.5) > 0.4]

            print(f"🔍 [Debug] After confidence filter (>0.4): {candidates}")

            # ------------------------------
            # Optional improvement: skip if empty
            # ------------------------------
            if not candidates:
                print("🔍 [Debug] No candidates, skipping promotion")
                continue  # nothing to reason about, skip promotion

            # ==============================
            # Reasoning layer
            # ==============================

            reasoning_ids = []
            for c in candidates:
                rid = reasoning_store.add(
                    content=c["text"],
                    source=c.get("source", "user"),
                    confidence=c.get("confidence", 0.9),
                )
                reasoning_ids.append(rid)

            # ==============================
            # Arbiter promotion (safe)
            # ==============================

            promoted = 0
            for c in candidates:
                print(f"🔍 [Debug] Processing candidate: {c}")
                r = reasoning_store.search(c["text"], top_k=1)

                # SAFETY CHECK: skip if reasoning store returned empty
                if not r or len(r) == 0:
                    print(f"🔍 [Debug] No reasoning results found for: {c['text']}")
                    continue

                print(f"🔍 [Debug] Reasoning result: {r[0]}")

                mem_type = c.get("type", "FACT")
                subject = c.get("subject", "User")
                confidence = c.get("confidence", 0.85)

                print(f"🔍 [Debug] Calling arbiter with: type={mem_type}, subject={subject}, confidence={confidence}")

                mid = arbiter.consider(
                    text=r[0]["content"],
                    mem_type=mem_type,
                    subject=subject,
                    confidence=confidence,
                    source=r[0].get("source", "reasoning"),
                )

                print(f"🔍 [Debug] Arbiter returned mid={mid}")

                if mid is not None:
                    promoted += 1
                    print(f"✅ [Debug] Memory saved with ID: {mid}")
                else:
                    print(f"❌ [Debug] Memory was rejected by arbiter")

            if promoted:
                print(f"🧠 Promoted {promoted} memory item(s).")
            else:
                print(f"🔍 [Debug] No memories promoted in this round")

        time.sleep(1.5)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(8)
