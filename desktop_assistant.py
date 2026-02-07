"""
AI Desktop Assistant
A standalone desktop application with integrated chat, document management, and Telegram bridge
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import time
import json
import os
import re
import sys
from datetime import datetime
import shutil
from typing import Dict, List, Optional

# Import our existing modules
from document_store_faiss import FaissDocumentStore
from document_processor import DocumentProcessor
from lm import compute_embedding, run_local_lm, extract_memory_candidates, DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT
from telegram_api import send_message, send_long_message, get_updates, get_file, download_file
from event_bus import EventBus
from continuousobserver import ContinuousObserver

# Import Memory System
from memory import MemoryStore
from meta_memory import MetaMemoryStore
from reasoning import ReasoningStore
from memory_arbiter import MemoryArbiter
from memory_consolidator import MemoryConsolidator
from daydreaming import DAYDREAM_EXTRACTOR_PROMPT as DEFAULT_DAYDREAM_EXTRACTOR_PROMPT
from decider import Decider
from hod import Hod

from ui import DesktopAssistantUI

class TelegramBridge:
    """Handles communication with Telegram API"""

    def __init__(self, bot_token: str, chat_id: int):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.is_connected = False
        self.last_update_id = None

    def send_message(self, text: str) -> bool:
        """Send message to Telegram"""
        try:
            send_long_message(self.bot_token, self.chat_id, text)
            return True
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
            return False

    def get_messages(self) -> List[Dict]:
        """Get new messages from Telegram"""
        try:
            # Use offset + 1 to confirm processed messages to Telegram
            offset = self.last_update_id + 1 if self.last_update_id is not None else None
            updates = get_updates(self.bot_token, offset=offset)
            messages = []

            for update in updates.get("result", []):
                # Update last_update_id to the current update's ID
                self.last_update_id = update["update_id"]

                if "message" in update:
                    msg = update["message"]
                    
                    # Base message data
                    message_data = {
                        "id": update["update_id"],
                        "chat_id": msg.get("chat", {}).get("id"),
                        "from": msg.get("from", {}).get("first_name", "Unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "date": msg.get("date", int(time.time())), # Capture Telegram timestamp
                        "type": "unknown"
                    }

                    if "text" in msg:
                        message_data["type"] = "text"
                        message_data["text"] = msg["text"]
                    elif "document" in msg:
                        message_data["type"] = "document"
                        message_data["document"] = msg["document"]
                    elif "photo" in msg:
                        message_data["type"] = "photo"
                        # Telegram sends multiple sizes; take the last one (highest quality)
                        message_data["photo"] = msg["photo"][-1]
                        message_data["caption"] = msg.get("caption", "")
                    
                    if message_data["type"] != "unknown":
                        messages.append(message_data)
            return messages
        except Exception as e:
            print(f"❌ Telegram receive error: {e}")
            return []


class DesktopAssistantApp(DesktopAssistantUI):
    # Command sets
    RESET_CHAT = {"/resetchat", "/chatreset", "/clearchat"}
    RESET_MEMORY = {"/resetmemory", "/memoryreset", "/clearmemory"}
    RESET_REASONING = {"/resetreasoning", "/reasoningreset", "/clearreasoning"}
    RESET_META_MEMORY = {"/resetmetamemory", "/metamemoryreset", "/clearmetamemory"}
    RESET_ALL = {"/resetall", "/clearall"}

    REMOVE_IDENTITY = {"/removeidentity", "/clearidentity", "/deleteidentity"}
    REMOVE_FACT = {"/removefact", "/clearfact", "/deletefact", "/removefacts", "/clearfacts"}
    REMOVE_PREFERENCE = {"/removepreference", "/clearpreference", "/deletepreference", "/removepreferences", "/clearpreferences"}
    REMOVE_GOAL = {"/removegoal", "/cleargoal", "/deletegoal", "/removegoals", "/cleargoals"}
    REMOVE_BELIEF = {"/removebelief", "/clearbelief", "/deletebelief", "/removebeliefs", "/clearbeliefs"}
    REMOVE_PERMISSION = {"/removepermission", "/clearpermission", "/deletepermission", "/removepermissions", "/clearpermissions"}
    REMOVE_RULE = {"/removerule", "/clearrule", "/deleterule", "/removerules", "/clearrules"}

    DOCUMENT_LIST = {"/documents", "/docs", "/listdocs"}
    DOCUMENT_REMOVE = {"/removedoc", "/removedocument", "/deletedoc", "/deletedocument"}
    DOCUMENT_CONTENT = {"/doccontent", "/docsummarize", "/docpreview"}

    # Inactivity settings
    INACTIVITY_RESET_SECONDS = 30 * 60  # 30 minutes

    def __init__(self, root):
        self.root = root
        self.root.title("AI Desktop Assistant")
        self.root.geometry("1200x800")

        # Track the settings file path first
        self.settings_file_path = "./settings.json"

        # State
        self.settings = self.load_settings()
        self.telegram_bridge = None
        self.is_running = False
        self.observer = None
        self.connected = False
        self.is_showing_placeholder = False  # Track placeholder state
        self.last_activity: Dict[int, float] = {} # Track last activity per chat_id
        self.stop_processing_flag = False
        self.is_processing = False
        self.daydreamer = None
        self.decider = None
        self.hod = None
        self.start_time = time.time()
        self.processing_lock = threading.Lock()
        self.telegram_status_sent = False  # Track if status has been sent to avoid spam
        
        # Initialize chat mode based on settings
        initial_mode = self.settings.get("ai_mode", "Daydream")
        self.chat_mode_var = tk.BooleanVar(value=(initial_mode == "Chat"))
        self.daydream_cycle_count = 0
        self.pending_confirmation_command = None

        # Initialize ttkbootstrap style with loaded theme
        theme_map = {
            "Cosmo": "cosmo",
            "Cyborg": "cyborg",
            "Darkly": "darkly"
        }
        theme_to_apply = theme_map.get(self.settings.get("theme", "Darkly"), self.settings.get("theme", "darkly"))
        self.style = ttk.Style(theme=theme_to_apply)

        # Initialize bridge toggle
        self.telegram_bridge_enabled = tk.BooleanVar()

        self.setup_ui()
        self.load_settings_into_ui()
        
        # Redirect stdout/stderr to logs tab
        self.redirect_logging()

        # Initialize Brain (Memory & Documents) - Moved after UI setup to capture logs
        self.chat_memory = {}
        self.init_brain()
        self.document_processor = DocumentProcessor(embed_fn=self.get_embedding_fn())
        
        # Refresh documents list now that DB is initialized
        self.refresh_documents()
        self.refresh_database_view()

        # Start background processes (Consolidation)
        self.start_background_processes()

        # Initialize connection state based on settings
        if (self.settings.get("telegram_bridge_enabled", False) and
            self.settings.get("bot_token") and
            self.settings.get("chat_id")):
            self.telegram_bridge_enabled.set(True)
            # Attempt to connect if credentials are provided and bridge is enabled
            self.bot_token_var.set(self.settings.get("bot_token"))
            self.chat_id_var.set(self.settings.get("chat_id"))
            # Connect automatically if settings are valid
            self.connect()
        else:
            self.telegram_bridge_enabled.set(False)
            # Ensure we're disconnected
            self.disconnect()

    def get_embedding_fn(self):
        """Returns a lambda that uses current settings for embeddings"""
        return lambda text: compute_embedding(
            text, 
            base_url=self.settings.get("base_url"), 
            embedding_model=self.settings.get("embedding_model")
        )

    def init_brain(self):
        """Initialize the AI memory and reasoning components"""
        try:
            # Alias: "Da'at" | Function: Knowledge
            self.memory_store = MemoryStore(db_path="./data/memory.sqlite3")
            self.meta_memory_store = MetaMemoryStore(
                db_path="./data/meta_memory.sqlite3",
                embed_fn=self.get_embedding_fn()
            )
            self.document_store = FaissDocumentStore(
                db_path="./data/documents_faiss.sqlite3",
                embed_fn=self.get_embedding_fn()
            )
            self.reasoning_store = ReasoningStore(embed_fn=self.get_embedding_fn())
            self.arbiter = MemoryArbiter(self.memory_store, meta_memory_store=self.meta_memory_store, embed_fn=self.get_embedding_fn())
            self.consolidator = MemoryConsolidator(
                self.memory_store, 
                meta_memory_store=self.meta_memory_store, 
                document_store=self.document_store,
                consolidation_thresholds=self.settings.get("consolidation_thresholds"),
                max_inconclusive_attempts=int(self.settings.get("max_inconclusive_attempts", 10)),
                max_retrieval_failures=int(self.settings.get("max_retrieval_failures", 10))
            )
            
            # Initialize Event Bus
            self.event_bus = EventBus()
            
            # Initialize Daydreamer
            # Alias: "Chokhmah" | Function: Raw, Creative Output
            from daydreaming import Daydreamer
            self.daydreamer = Daydreamer(
                memory_store=self.memory_store,
                reasoning_store=self.reasoning_store,
                arbiter=self.arbiter,
                document_store=self.document_store,
                get_settings_fn=lambda: self.settings,
                log_fn=self.log_to_main,
                chat_fn=self.on_proactive_message,
                stop_check_fn=lambda: self.stop_processing_flag,
                status_fn=lambda msg: self.root.after(0, lambda: self.status_var.set(msg)),
                get_chat_history_fn=self.get_current_chat_history,
                event_bus=self.event_bus
            )

            # Helper for sync verification in loop (thread-safe UI updates)
            def verify_batch_sync():
                if hasattr(self, 'consolidator'):
                    self.root.after(0, lambda: self.status_var.set("Verifying sources (Decider)..."))
                    removed = self.consolidator.verify_sources(batch_size=50, stop_check_fn=lambda: self.stop_processing_flag)
                    if removed > 0:
                        self.root.after(0, self.refresh_database_view)
                        self.log_to_main(f"🧹 [Decider] Removed {removed} memories.")

            # Wrappers for Telegram notifications
            def start_daydream_wrapper():
                self.send_telegram_status("☁️ Model is processing memories (Daydreaming)...")
                try:
                    mode = "auto"
                    topic = None
                    if self.decider and hasattr(self.decider, 'daydream_mode'):
                        mode = self.decider.daydream_mode
                        topic = getattr(self.decider, 'daydream_topic', None)
                    self.daydreamer.perform_daydream(mode=mode, topic=topic)
                finally:
                    self.send_telegram_status("✅ Processing finished.")

            def verify_batch_wrapper():
                self.send_telegram_status("⚙️ Model is processing memories (Verification Batch)...")
                try:
                    verify_batch_sync()
                finally:
                    self.send_telegram_status("✅ Processing finished.")

            def verify_all_wrapper():
                self.send_telegram_status("⚙️ Model is processing memories (Full Verification)...")
                try:
                    if hasattr(self, 'consolidator'):
                        self.root.after(0, lambda: self.status_var.set("Verifying ALL sources..."))
                        
                        last_remaining = -1
                        stuck_count = 0
                        
                        while True:
                            if self.stop_processing_flag:
                                break
                            remaining = self.consolidator.get_unverified_count()
                            if remaining == 0:
                                break
                            
                            # Loop protection: Break if stuck on the same number of memories
                            if remaining == last_remaining:
                                stuck_count += 1
                                if stuck_count >= 5:
                                    self.log_to_main(f"⚠️ Verification loop stuck on {remaining} memories. Aborting.")
                                    break
                            else:
                                stuck_count = 0
                                last_remaining = remaining

                            self.consolidator.verify_sources(batch_size=50, stop_check_fn=lambda: self.stop_processing_flag)
                            self.root.after(0, self.refresh_database_view)
                        
                        if self.hod:
                            self.hod.perform_analysis("Full Verification")
                finally:
                    self.send_telegram_status("✅ Processing finished.")

            def start_loop_wrapper():
                self.send_telegram_status("🔄 Daydream loop enabled.")
                self.enable_daydream_loop()

            # Wrappers for Strategic Thinking (Goal Management & Document Access)
            def remove_goal_wrapper(target):
                """Allows Decider to remove completed or obsolete goals"""
                try:
                    # Fetch all memories to safely check type
                    all_memories = self.memory_store.list_recent(limit=None)
                    
                    target_id = None
                    try:
                        target_id = int(str(target).strip())
                    except ValueError:
                        pass
                        
                    found_items = []
                    if target_id is not None:
                        found_items = [m for m in all_memories if m[0] == target_id]
                    else:
                        # Search by text content
                        target_text = str(target).lower()
                        found_items = [m for m in all_memories if target_text in m[3].lower()]
                    
                    removed_count = 0
                    response_msgs = []
                    
                    for item in found_items:
                        # item structure: (id, type, subject, text, ...)
                        mem_id = item[0]
                        mem_type = item[1]
                        mem_text = item[3]
                        
                        if mem_type == "GOAL":
                            self.memory_store.soft_delete_entry(mem_id)
                            self.log_to_main(f"🗑️ [Decider] Marked GOAL as inactive: {mem_text}")
                            response_msgs.append(f"Removed '{mem_text}'")
                            removed_count += 1
                        else:
                            # Protect chat memories and other types
                            response_msgs.append(f"Skipped ID {mem_id} (Type: {mem_type})")
                            
                    if removed_count > 0:
                        return f"✅ Success. {', '.join(response_msgs)}"
                    elif response_msgs:
                        return f"⚠️ Failed. {', '.join(response_msgs)}"
                    else:
                        return "❌ No matching goals found."
                except Exception as e:
                    return f"❌ Error removing goal: {e}"

            def list_documents_wrapper():
                """Allows Decider to see available documents for daydreaming topics"""
                try:
                    docs = self.document_store.list_documents(limit=50)
                    if not docs:
                        return "No documents available."
                    
                    lines = ["Available Documents:"]
                    for doc in docs:
                        # doc: (id, filename, file_type, page_count, chunk_count, created_at)
                        lines.append(f"- ID {doc[0]}: {doc[1]} ({doc[4]} chunks)")
                    return "\n".join(lines)
                except Exception as e:
                    return f"Error listing documents: {e}"

            def read_document_wrapper(target):
                """Allows Decider to read a specific document"""
                try:
                    # Try ID first
                    doc_id = None
                    try:
                        doc_id = int(str(target).strip())
                    except:
                        pass
                    
                    docs = self.document_store.list_documents(limit=1000)
                    selected_doc = None
                    
                    if doc_id:
                        selected_doc = next((d for d in docs if d[0] == doc_id), None)
                    
                    if not selected_doc:
                        # Try filename match
                        target_lower = str(target).lower().strip()
                        selected_doc = next((d for d in docs if target_lower in d[1].lower()), None)
                        
                    if not selected_doc:
                        return f"❌ Document '{target}' not found."
                        
                    # Fetch chunks
                    doc_id = selected_doc[0]
                    filename = selected_doc[1]
                    chunks = self.document_store.get_document_chunks(doc_id)
                    
                    if not chunks:
                        return f"⚠️ Document '{filename}' is empty."
                        
                    # Return first 5 chunks (Overview)
                    preview = "\n\n".join([c['text'] for c in chunks[:5]])
                    return f"📄 Content of '{filename}' (First 5 chunks):\n{preview}"
                except Exception as e:
                    return f"❌ Error reading document: {e}"

            def search_memory_wrapper(query):
                """Allows Decider to actively search memories"""
                try:
                    emb = self.get_embedding_fn()(query)
                    results = self.memory_store.search(emb, limit=10)
                    if not results:
                        return "No matching memories found."
                    
                    lines = [f"Search results for '{query}':"]
                    for r in results:
                        # r: (id, type, subject, text, similarity)
                        lines.append(f"- [{r[1]}] {r[3]} (Sim: {r[4]:.2f})")
                    return "\n".join(lines)
                except Exception as e:
                    return f"❌ Error searching memory: {e}"

            def prune_memory_wrapper(target_id):
                """Allows Hod to mark memories for pruning (tagging only)"""
                try:
                    mid = int(str(target_id).strip())
                    mem = self.memory_store.get(mid)
                    if not mem:
                        return f"⚠️ Memory ID {mid} not found."
                    
                    # Check if already deleted
                    if mem.get('deleted', 0) == 1:
                        return f"ℹ️ Memory ID {mid} is already pruned."

                    # Protection check: Do not prune verified daydream memories
                    is_verified = mem.get('verified', 0) == 1
                    source = mem.get('source', '').lower()
                    mem_type = mem.get('type', '').upper()
                    verification_attempts = mem.get('verification_attempts') or 0

                    if is_verified and 'daydream' in source:
                        return f"⚠️ Cannot prune verified daydream memory (ID: {mid})."

                    # Protection: Do not prune daydream memories that haven't been verified yet
                    # EXCEPTION: Allow pruning BELIEFs, as Daydreamer refutation IS the verification for them.
                    if 'daydream' in source and verification_attempts == 0 and not is_verified and mem_type != "BELIEF":
                        return f"⚠️ Cannot prune unprocessed daydream memory (ID: {mid}). Let Verifier check it first (unless it's a refuted belief)."

                    # Protect Assistant Notes
                    if mem_type == "NOTE":
                        return f"⚠️ Cannot prune Assistant Note (ID: {mid})."

                    # Protect Chat Memories
                    if source == 'assistant':
                        return f"⚠️ Cannot prune Chat Memory (ID: {mid})."

                    # Tag instead of delete
                    if self.memory_store.set_flag(mid, "PRUNE_REQUESTED"):
                        return f"🏷️ Tagged memory ID {mid} for pruning (Decider must confirm)."
                    return f"⚠️ Failed to tag memory ID {mid}."
                except Exception as e:
                    return f"❌ Error tagging memory: {e}"

            def confirm_prune_wrapper(target_id):
                """Allows Decider to confirm and execute a prune request"""
                try:
                    mid = int(str(target_id).strip())
                    
                    # Safety: Verify flag exists
                    mem = self.memory_store.get(mid)
                    if not mem or mem.get('flags') != 'PRUNE_REQUESTED':
                        return f"⚠️ Memory ID {mid} is not flagged for pruning."

                    if self.memory_store.soft_delete_entry(mid):
                        return f"🗑️ Pruned memory ID {mid}."
                    return f"⚠️ Memory ID {mid} not found."
                except Exception as e:
                    return f"❌ Error confirming prune: {e}"

            def reject_prune_wrapper(target_id):
                """Allows Decider to reject a prune request (clear tag)"""
                try:
                    mid = int(str(target_id).strip())
                    if self.memory_store.set_flag(mid, None): # Clear flag
                        return f"✅ Rejected prune for memory ID {mid}."
                    return f"⚠️ Memory ID {mid} not found."
                except Exception as e:
                    return f"❌ Error rejecting prune: {e}"

            def summarize_wrapper():
                """Allows Decider to trigger Hod's summarization"""
                if self.hod:
                    self.hod.run_summarization()
                    return "🔮 Hod is summarizing the session."
                return "⚠️ Hod not initialized."

            def consolidate_summaries_wrapper():
                """Allows Decider to trigger Hod's summary consolidation"""
                if self.hod:
                    result = self.hod.consolidate_summaries()
                    return result
                return "⚠️ Hod not initialized."

            # Initialize Decider
            self.decider = Decider(
                get_settings_fn=lambda: self.settings,
                update_settings_fn=self.update_settings_from_decider,
                memory_store=self.memory_store,
                document_store=self.document_store,
                reasoning_store=self.reasoning_store,
                arbiter=self.arbiter,
                meta_memory_store=self.meta_memory_store,
                actions={
                    "start_daydream": start_daydream_wrapper,
                    "verify_batch": verify_batch_wrapper,
                    "verify_all": verify_all_wrapper,
                    "start_loop": start_loop_wrapper,
                    "stop_daydream": self.stop_daydream,
                    "run_observer": lambda: self.observer.perform_observation() if self.observer else None,
                    "run_hod": lambda: self.hod.perform_analysis("Decider Cycle") if self.hod else None,
                    "remove_goal": remove_goal_wrapper,
                    "list_documents": list_documents_wrapper,
                    "read_document": read_document_wrapper,
                    "search_memory": search_memory_wrapper,
                    "confirm_prune": confirm_prune_wrapper,
                    "reject_prune": reject_prune_wrapper,
                    "summarize": summarize_wrapper,
                    "consolidate_summaries": consolidate_summaries_wrapper
                },
                log_fn=self.log_to_main,
                chat_fn=self.on_proactive_message,
                get_chat_history_fn=self.get_current_chat_history,
                stop_check_fn=lambda: self.stop_processing_flag
            )

            # Initialize Continuous Observer (Netzach)
            self.observer = ContinuousObserver(
                memory_store=self.memory_store,
                reasoning_store=self.reasoning_store,
                meta_memory_store=self.meta_memory_store,
                get_settings_fn=lambda: self.settings,
                get_chat_history_fn=self.get_current_chat_history,
                manifest_fn=self.on_proactive_message,
                get_meta_memories_fn=lambda: self.meta_memory_store.list_recent(limit=10),
                get_main_logs_fn=self.get_recent_main_logs,
                get_doc_logs_fn=self.get_recent_doc_logs,
                get_status_fn=self.get_current_status_text,
                event_bus=self.event_bus,
                get_recent_docs_fn=lambda: self.document_store.list_documents(limit=5),
                log_fn=self.log_to_main,
                stop_check_fn=lambda: self.stop_processing_flag
            )
            
            # Initialize Hod (Reflective Analyst)
            self.hod = Hod(
                memory_store=self.memory_store,
                meta_memory_store=self.meta_memory_store,
                reasoning_store=self.reasoning_store,
                get_settings_fn=lambda: self.settings,
                get_main_logs_fn=self.get_recent_main_logs,
                get_doc_logs_fn=self.get_recent_doc_logs,
                log_fn=self.log_to_main,
                event_bus=self.event_bus,
                prune_memory_fn=prune_memory_wrapper
            )
            
            # --- Event Bus Wiring ---
            # 1. Netzach -> Decider (Wake up)
            self.event_bus.subscribe("DECIDER_WAKE", lambda e: self.decider.start_daydream())
            
            # 2. Netzach -> System (Param updates)
            def handle_param_update(event):
                data = event.data
                if "temperature_delta" in data:
                    self.decider.increase_temperature(data["temperature_delta"])
                if "temperature_decrease" in data:
                    self.decider.decrease_temperature(data["temperature_decrease"])
                if "tokens_delta" in data:
                    self.decider.increase_tokens(data["tokens_delta"])
                if "tokens_decrease" in data:
                    self.decider.decrease_tokens(data["tokens_decrease"])
            self.event_bus.subscribe("SYSTEM_PARAM_UPDATE", handle_param_update)
            
            # 3. Netzach -> Hod (Instruction)
            self.event_bus.subscribe("HOD_INSTRUCTION", lambda e: self.hod.receive_instruction(e.data) if self.hod else None)
            
            # 4. Netzach -> Hod (Summary Request)
            self.event_bus.subscribe("REQUEST_SUMMARY", lambda e: self.hod.run_summarization() if self.hod else None)
            
            # 5. Netzach -> Decider (Observation)
            self.event_bus.subscribe("DECIDER_OBSERVATION", lambda e: self.decider.receive_observation(e.data))

            # 6. Hod -> Netzach (Wake/Instruction)
            self.event_bus.subscribe("NETZACH_WAKE", lambda e: self.observer.perform_observation())
            
            def handle_netzach_instruction(event):
                msg = event.data
                if hasattr(self.meta_memory_store, 'add_event'):
                    self.meta_memory_store.add_event("HOD_MESSAGE", "Hod", f"Message to Netzach: {msg}")
                self.observer.perform_observation()
            self.event_bus.subscribe("NETZACH_INSTRUCTION", handle_netzach_instruction)
            
            # Trigger initial analysis to orient the system based on past sessions
            if self.hod:
                self.root.after(2000, lambda: threading.Thread(target=self.hod.perform_analysis, args=("System Startup",), daemon=True).start())
            
            print("🧠 Brain initialized successfully.")
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize AI components:\n{e}")

    def update_settings_from_decider(self, new_settings: Dict):
        """Callback for Decider to update settings and UI"""
        self.settings.update(new_settings)
        self.save_settings()
        # Update UI on main thread
        if hasattr(self, 'temperature_var'):
            self.root.after(0, lambda: self.temperature_var.set(new_settings.get("temperature", 0.7)))
        if hasattr(self, 'max_tokens_var'):
            self.root.after(0, lambda: self.max_tokens_var.set(new_settings.get("max_tokens", 800)))

    def enable_daydream_loop(self):
        """Allow the daydream loop to run by clearing stop flags and chat mode"""
        self.stop_processing_flag = False
        if self.chat_mode_var.get():
            self.chat_mode_var.set(False)
            self.on_chat_mode_toggle()

    def get_recent_main_logs(self) -> str:
        """Get last 15 lines of main logs for Netzach"""
        if hasattr(self, 'log_buffer'):
            full_text = "".join(self.log_buffer)
            lines = full_text.splitlines()
            return "\n".join(lines[-15:])
        
        if not hasattr(self, 'main_log_text'): return ""
        try:
            content = self.main_log_text.get("end-15l", "end-1c")
            return content.strip()
        except:
            return ""

    def get_recent_doc_logs(self) -> str:
        """Get last 10 lines of document logs for Netzach"""
        if hasattr(self, 'debug_log_buffer'):
            return "".join(self.debug_log_buffer[-10:])
            
        if not hasattr(self, 'debug_log'): return ""
        try:
            content = self.debug_log.get("end-10l", "end-1c")
            return content.strip()
        except:
            return ""

    def get_current_status_text(self) -> str:
        """Get current status bar text for Netzach"""
        return self.status_var.get()

    def get_current_chat_history(self) -> List[Dict]:
        """Helper for Daydreamer/Observer to see the current conversation"""
        chat_id = int(self.settings.get("chat_id", 0) or 0)
        return self.chat_memory.get(chat_id, [])

    def load_settings(self) -> Dict:
        """Load settings from file"""
        settings_file = self.settings_file_path
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                
                # Auto-repair: Fix memory extractor prompt if it contains the old "echoes" rule
                current_prompt = settings.get("memory_extractor_prompt", "")
                if "echoes of the other party" in current_prompt or "SELF-CONTAINED" not in current_prompt or "Assistant's suggestions" not in current_prompt:
                    print("🔧 Auto-repairing memory extractor prompt in settings...")
                    # Hardcoded correct prompt to ensure fix
                    settings["memory_extractor_prompt"] = "Extract durable memories from the conversation. Return ONLY a valid JSON array.\n\nMemory Types (Apply to BOTH User and Assistant):\n- IDENTITY: Names, roles, locations, RELIGION (e.g., 'Assistant religion is Hinduism', 'Assistant is a Buddhist')\n- PREFERENCE: Likes/dislikes (e.g., 'User loves coffee', 'Assistant prefers inner peace')\n- GOAL: Specific, actionable objectives (e.g., 'User wants to learn Python', 'Assistant plans to analyze X'). Do NOT extract general statements like 'Future research should...' as GOALs.\n- FACT: Objective truths (e.g., 'User is an engineer', 'Assistant can process data')\n- BELIEF: Opinions/convictions (e.g., 'User believes AI is good', 'Assistant believes in meditation')\n- PERMISSION: Explicit user grants (e.g., 'User allowed Assistant to hold opinions')\n- RULE: Behavior guidelines (e.g., 'Assistant should not use emojis')\n\nRules:\n1. Extract from BOTH User AND Assistant.\n2. Each object MUST have: \"type\", \"subject\" (User or Assistant), \"text\".\n3. Use DOUBLE QUOTES for all keys and string values.\n4. Max 5 memories, max 240 chars each.\n5. EXCLUDE: Pure greetings (e.g., 'Hi'), questions, and filler. DO NOT exclude facts stated during introductions (e.g., 'Hi, I'm X').\n6. EXCLUDE generic assistant politeness (e.g., 'Assistant goal is to help', 'I'm here to help', 'feel free to ask').\n7. EXCLUDE contextual/situational goals (e.g., 'help with X topic' where X is current conversation topic).\n8. ONLY extract ASSISTANT GOALS if they represent true self-chosen objectives or explicit commitments.\n9. DO NOT extract facts from the Assistant's text if it is merely recalling known info. ALWAYS extract new facts from the User's text.\n10. ATTRIBUTION RULE: If User says 'I am X', subject is User. If Assistant says 'I am X', subject is Assistant. NEVER attribute User statements to Assistant.\n11. CRITICAL: DO NOT attribute Assistant's suggestions, lists, or hypothetical topics to the User. Only record User interests if the USER explicitly stated them.\n12. MAKE MEMORIES SELF-CONTAINED: Replace pronouns like 'This', 'These', 'It' with specific nouns. Ensure the text makes sense without the surrounding context.\n13. If no new memories, return [].\n"
                    self.save_settings_to_file(settings)
                
                # Ensure defaults exist for Decider baselines to prevent drift
                if "default_temperature" not in settings:
                    settings["default_temperature"] = 0.7
                if "default_max_tokens" not in settings:
                    settings["default_max_tokens"] = 800
                
                return settings
        else:
            return {}

    def save_settings(self):
        """Save settings to file"""
        settings_file = self.settings_file_path
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, indent=2, ensure_ascii=False)
            
    def save_settings_to_file(self, settings_dict):
        """Helper to write settings dict to disk"""
        with open(self.settings_file_path, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=2, ensure_ascii=False)

    def on_proactive_message(self, sender, msg):
        """Handle proactive messages from Daydreamer or Observer (Netzach)"""
        # 1. Always log to AI Interactions (Netzach) window for transparency
        self.root.after(0, lambda: self.add_netzach_message(f"{sender}: {msg}"))

        # 2. Determine if it should appear in the Main Chat
        # Only explicit messages (SPEAK) should appear in main chat.
        # Thoughts, decisions, and daydreaming are internal.
        should_show_in_chat = False
        
        if sender == "Decider":
            # Filter out internal logs/thoughts. If it's a [SPEAK] message, it won't have these markers.
            internal_markers = ["🤔", "💭", "🤖", "🛠️", "📩", "⚠️", "✅", "Decision:", "Thought:"]
            if not any(marker in msg for marker in internal_markers):
                should_show_in_chat = True
        
        if should_show_in_chat:
            # Show in local UI
            self.root.after(0, lambda: self.add_chat_message("Assistant", msg, "incoming"))
            
            # Forward to Telegram
            if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
                 self.telegram_bridge.send_message(msg)

            # Update Chat Memory so the AI remembers its own proactive statement
            chat_id = int(self.settings.get("chat_id", 0) or 0)
            history = self.chat_memory.get(chat_id, [])
            history.append({"role": "assistant", "content": msg})
            if len(history) > 20:
                history = history[-20:]
            self.chat_memory[chat_id] = history

    def stop_processing(self):
        """Stop current AI generation"""
        print("🛑 Stop button clicked.")
        if self.is_processing:
            self.stop_processing_flag = True
            self.status_var.set("Stopping...")
            print("⏳ Sending stop signal to background process...")
        else:
            print("ℹ️ AI is currently idle.")

    def stop_daydream(self):
        """Stop daydreaming specifically"""
        print("🛑 Stop Daydream triggered.")
        self.stop_processing_flag = True
        
        if self.decider:
            self.decider.report_forced_stop()
            
        # Reset flag after a moment to allow Decider to pick up the "forced stop" state
        def reset_flag():
            time.sleep(1.5) 
            self.stop_processing_flag = False
            print("▶️ Decider ready for next turn (Cooldown active).")
            
        threading.Thread(target=reset_flag, daemon=True).start()

    def on_chat_mode_toggle(self):
        """Handle chat mode toggle"""
        if self.chat_mode_var.get():
            print("🔒 Chat Mode enabled. Telegram Bridge active.")
        else:
            print("🔓 Chat Mode disabled. Telegram Bridge paused.")

    def start_daydream(self):
        """Manually trigger a daydream cycle"""
        if self.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy processing a task.")
            return
            
        if self.daydreamer:
            def run_with_lock():
                with self.processing_lock:
                    self.is_processing = True
                    try:
                        self.daydreamer.perform_daydream()
                    finally:
                        self.is_processing = False
                        self.root.after(0, lambda: self.status_var.set("Ready"))
            threading.Thread(target=run_with_lock, daemon=True).start()
        else:
            messagebox.showerror("Error", "Daydreamer not initialized.")

    def verify_memory_sources(self):
        """Manually trigger memory source verification"""
        # Alias: "Binah" | Function: Reasoning and Logic
        if self.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy (e.g. Daydreaming). Please click 'Stop' or enable 'Chat Mode' first.")
            return
            
        if not hasattr(self, 'consolidator'):
            return

        def verify_thread():
            print("🧹 [Manual Verifier] Starting quick batch verification...")
            with self.processing_lock:
                self.is_processing = True
                self.root.after(0, lambda: self.status_var.set("Verifying memory sources..."))
                
                try:
                    # Use a smaller batch for quick verification
                    if self.stop_processing_flag:
                        return

                    removed = self.consolidator.verify_sources(batch_size=50, stop_check_fn=lambda: self.stop_processing_flag)
                    msg = f"Verification complete. Removed {removed} hallucinated memories."
                    print(f"🧹 [Manual Verifier] {msg}")
                    self.root.after(0, lambda: messagebox.showinfo("Verification Result", msg))
                    self.root.after(0, self.refresh_database_view)
                    if self.hod:
                        self.hod.perform_analysis("Verification Batch")
                except Exception as e:
                    print(f"Verification error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Verification failed: {e}"))
                finally:
                    self.is_processing = False
                    self.root.after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=verify_thread, daemon=True).start()

    def verify_all_memory_sources(self):
        """Loop verification until all memories are verified"""
        if self.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy. Please click 'Stop' first.")
            return
            
        if not hasattr(self, 'consolidator'):
            return

        def verify_all_thread():
            print("🧹 [Manual Verifier] Starting FULL verification loop...")
            with self.processing_lock:
                self.is_processing = True
                self.root.after(0, lambda: self.status_var.set("Verifying ALL sources..."))
                
                total_removed = 0
                last_remaining = -1
                stuck_count = 0
                
                try:
                    while True:
                        if self.stop_processing_flag:
                            print("🛑 Verification loop stopped by user.")
                            break
                        
                        # Check if anything left to verify
                        remaining = self.consolidator.get_unverified_count()
                        if remaining == 0:
                            print("✅ All cited memories verified.")
                            break
                        
                        # Loop protection
                        if remaining == last_remaining:
                            stuck_count += 1
                            if stuck_count >= 5:
                                print(f"⚠️ Verification loop stuck on {remaining} memories. Aborting.")
                                break
                        else:
                            stuck_count = 0
                            last_remaining = remaining
                        
                        self.root.after(0, lambda: self.status_var.set(f"Verifying... ({remaining} left)"))
                        
                        # Verify a batch
                        removed = self.consolidator.verify_sources(batch_size=10000, stop_check_fn=lambda: self.stop_processing_flag)
                        total_removed += removed
                        
                        # Refresh UI to show progress
                        self.root.after(0, self.refresh_database_view)
                        
                    msg = f"Full verification complete. Removed {total_removed} memories."
                    print(f"🧹 [Manual Verifier] {msg}")
                    self.root.after(0, lambda: messagebox.showinfo("Verification Result", msg))
                    if self.hod:
                        self.hod.perform_analysis("Full Verification")
                except Exception as e:
                    print(f"Verification error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Verification failed: {e}"))
                finally:
                    self.is_processing = False
                    self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=verify_all_thread, daemon=True).start()

    def toggle_connection(self):
        """Toggle connection to Telegram"""
        # Toggle the setting
        new_state = not self.telegram_bridge_enabled.get()
        self.telegram_bridge_enabled.set(new_state)
        # Save the new state
        self.settings["telegram_bridge_enabled"] = new_state
        self.save_settings()

        # Update connection based on new state
        if new_state:
            # Only connect if both credentials are provided
            bot_token = self.bot_token_var.get().strip()
            chat_id_str = self.chat_id_var.get().strip()
            if bot_token and chat_id_str:
                try:
                    int(chat_id_str)  # Validate chat ID is numeric
                    self.connect()
                except ValueError:
                    messagebox.showerror("Connection Error", "Chat ID must be a valid number")
                    self.telegram_bridge_enabled.set(False)
                    self.settings["telegram_bridge_enabled"] = False
                    self.save_settings()
            else:
                messagebox.showerror("Connection Error", "Please enter both Bot Token and Chat ID in Settings")
                self.telegram_bridge_enabled.set(False)
                self.settings["telegram_bridge_enabled"] = False
                self.save_settings()
        else:
            self.disconnect()

    def connect(self):
        """Connect to Telegram"""
        if self.is_connected():
            return  # Already connected

        bot_token = self.bot_token_var.get().strip()
        chat_id_str = self.chat_id_var.get().strip()

        if not bot_token or not chat_id_str:
            # Don't show error if called internally - just return
            return

        try:
            chat_id = int(chat_id_str)
            # Alias: "Yesod" | Function: Foundation, Transmission
            self.telegram_bridge = TelegramBridge(bot_token, chat_id)

            # Test connection
            if self.telegram_bridge.send_message("✅ Connected to Desktop Assistant"):
                self.connected = True
                self.connect_button.config(text="Connected", bootstyle="success")
                self.status_var.set("Connected to Telegram")

                # Start message polling
                threading.Thread(target=self.poll_telegram_messages, daemon=True).start()

            else:
                raise Exception("Failed to send test message")

        except Exception as e:
            # Only show error if this was a direct user action
            if not self.settings.get("telegram_bridge_enabled", False):
                # If bridge is disabled, don't show error
                pass
            else:
                messagebox.showerror("Connection Error", f"Failed to connect: {e}")
            self.disconnect()

    def disconnect(self):
        """Disconnect from Telegram"""
        self.connected = False
        self.telegram_bridge = None
        self.connect_button.config(text="Connect", bootstyle="secondary")
        self.status_var.set("Disconnected from Telegram")

    def send_telegram_status(self, message: str):
        """Send a status update to Telegram if connected"""
        if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
             # Suppress repetitive status messages until user interacts
             if self.telegram_status_sent:
                 return

             if self.telegram_bridge.send_message(message):
                 if "finished" in message.lower():
                     self.telegram_status_sent = True

    def is_connected(self):
        """Check if connected to Telegram"""
        return self.connected and self.telegram_bridge is not None

    def handle_command(self, text: str, chat_id: int) -> Optional[str]:
        """Process slash commands and return response if matched"""
        cmd_parts = text.strip().split()
        if not cmd_parts:
            return None
        
        cmd = cmd_parts[0].lower()

        # Confirmation handling
        if cmd == "/y":
            if self.pending_confirmation_command:
                pending_cmd = self.pending_confirmation_command
                self.pending_confirmation_command = None
                
                if pending_cmd in self.RESET_CHAT:
                    self.chat_memory[chat_id] = []
                    return "♻️ Chat history cleared."

                if pending_cmd in self.RESET_MEMORY:
                    self.memory_store.clear()
                    return "🧠 Long-term memory wiped."
                    
                if pending_cmd in self.RESET_REASONING:
                    self.reasoning_store.clear()
                    return "🧩 Reasoning buffer cleared."

                if pending_cmd in self.RESET_META_MEMORY:
                    self.meta_memory_store.clear()
                    return "🧠 Meta-memories cleared."

                if pending_cmd in self.RESET_ALL:
                    self.chat_memory[chat_id] = []
                    self.reasoning_store.clear()
                    self.memory_store.clear()
                    self.meta_memory_store.clear()
                    return "🔥 Full reset complete (chat + reasoning + memory + meta-memory)."
            else:
                return "ℹ️ No pending command to confirm."

        # Reset Commands (Initiate confirmation)
        if cmd in self.RESET_CHAT or cmd in self.RESET_MEMORY or cmd in self.RESET_REASONING or cmd in self.RESET_META_MEMORY or cmd in self.RESET_ALL:
            self.pending_confirmation_command = cmd
            return "⚠️ Are you sure? This action is irreversible. Type `/Y` to confirm."

        # Clear pending confirmation if another command is issued
        self.pending_confirmation_command = None

        # Consolidation
        if cmd in {"/consolidate", "/consolidatenow"}:
            def run_consolidation():
                self.log_to_main("🧠 Starting manual consolidation...")
                stats = self.consolidator.consolidate(time_window_hours=None)
                msg = f"🧠 Consolidation complete: Processed {stats['processed']}, Consolidated {stats['consolidated']}, Skipped {stats['skipped']}."
                self.log_to_main(msg)
                self.root.after(0, lambda: self.add_chat_message("System", msg, "incoming"))
                self.root.after(0, self.refresh_database_view)
            
            threading.Thread(target=run_consolidation, daemon=True).start()
            return "⏳ Consolidation started in background..."

        # Memory Removal
        if cmd in self.REMOVE_IDENTITY:
            count = self.memory_store.clear_by_type("IDENTITY")
            return f"🗑️ Removed {count} IDENTITY memories."
        
        if cmd in self.REMOVE_FACT:
            count = self.memory_store.clear_by_type("FACT")
            return f"🗑️ Removed {count} FACT memories."
            
        if cmd in self.REMOVE_PREFERENCE:
            count = self.memory_store.clear_by_type("PREFERENCE")
            return f"🗑️ Removed {count} PREFERENCE memories."
            
        if cmd in self.REMOVE_GOAL:
            count = self.memory_store.clear_by_type("GOAL")
            return f"🗑️ Removed {count} GOAL memories."
            
        if cmd in self.REMOVE_BELIEF:
            count = self.memory_store.clear_by_type("BELIEF")
            return f"🗑️ Removed {count} BELIEF memories."
            
        if cmd in self.REMOVE_PERMISSION:
            count = self.memory_store.clear_by_type("PERMISSION")
            return f"🗑️ Removed {count} PERMISSION memories."
            
        if cmd in self.REMOVE_RULE:
            count = self.memory_store.clear_by_type("RULE")
            return f"🗑️ Removed {count} RULE memories."

        # Document Management
        if cmd in self.DOCUMENT_LIST:
            docs = self.document_store.list_documents(limit=20)
            if not docs:
                return "📚 No documents in the database."
            
            lines = []
            for doc_id, filename, file_type, page_count, chunk_count, created_at in docs:
                date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                page_info = f", {page_count} pages" if page_count else ""
                lines.append(f"📄 {filename} ({file_type}{page_info}, {chunk_count} chunks) - {date_str}")
            
            return "📚 Document Database:\n" + "\n".join(lines)

        if cmd in self.DOCUMENT_REMOVE:
            # Extract filename
            match = re.search(r'"([^"]*)"', text)
            if match:
                doc_filename = match.group(1)
                # Find doc ID
                docs = self.document_store.list_documents(limit=1000)
                doc_id = next((d[0] for d in docs if d[1] == doc_filename), None)
                
                if doc_id:
                    if self.document_store.delete_document(doc_id):
                        # Refresh GUI if open
                        self.root.after(0, self.refresh_documents)
                        return f"🗑️ Successfully removed document: {doc_filename}"
                    else:
                        return f"❌ Could not remove document: {doc_filename}"
                else:
                    return f"❌ Document not found: {doc_filename}"
            else:
                return "🗑️ To remove a document, use: /RemoveDoc \"filename.pdf\"\nUse /Documents to see available documents."

        if cmd in self.DOCUMENT_CONTENT or any(text.lower().startswith(x) for x in self.DOCUMENT_CONTENT):
             # Extract filename
            match = re.search(r'"([^"]*)"', text)
            if match:
                doc_filename = match.group(1)
                # Find doc ID
                docs = self.document_store.list_documents(limit=1000)
                doc_id = next((d[0] for d in docs if d[1] == doc_filename), None)
                
                if doc_id:
                    chunks = self.document_store.get_document_chunks(doc_id)
                    if chunks:
                        preview = "\n\n".join([f"Chunk {c['chunk_index']+1}: {c['text'][:200]}..." for c in chunks[:3]])
                        return f"📖 Content preview for '{doc_filename}':\n\n{preview}"
                    return f"❌ No content found for: {doc_filename}"
                return f"❌ Document not found: {doc_filename}"
            else:
                return "📖 To view document content, use: /DocContent \"filename.pdf\"\nUse /Documents to see available documents."

        # Memories View
        if cmd == "/memories":
            items = self.memory_store.list_recent(limit=None)
            if not items:
                return "🧠 No saved memories."
            
            type_emoji = {
                "IDENTITY": "👤", "FACT": "📌", "PREFERENCE": "❤️", 
                "GOAL": "🎯", "RULE": "⚖️", "PERMISSION": "✅", "BELIEF": "💭"
            }
            
            grouped = {}
            for (_id, mem_type, subject, text) in items:
                _id, mem_type, subject, text = item[:4]
                grouped.setdefault(mem_type, []).append((subject, text))
            
            lines = []
            hierarchy = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]
            
            for mem_type in hierarchy:
                if mem_type in grouped:
                    emoji = type_emoji.get(mem_type, "💡")
                    lines.append(f"\n{emoji} {mem_type}:")
                    for subject, text in grouped[mem_type]:
                        lines.append(f"  - [{subject}] {text}")
                    del grouped[mem_type]
            
            for mem_type, remaining in grouped.items():
                emoji = type_emoji.get(mem_type, "💡")
                lines.append(f"\n{emoji} {mem_type}:")
                for subject, text in remaining:
                    lines.append(f"  - [{subject}] {text}")
            
            return "🧠 Saved Memories :\n" + "\n".join(lines)

        # Meta Memories View
        if cmd in {"/metamemories", "/meta-memories"}:
            items = self.meta_memory_store.list_recent(limit=30)
            if not items:
                return "🧠 No meta-memories."
            
            lines = []
            for (_id, event_type, subject, text, created_at) in items:
                event_emoji = {
                    "MEMORY_CREATED": "✨", "VERSION_UPDATE": "🔄",
                    "CONFLICT_DETECTED": "⚠️", "CONSOLIDATION": "🔗"
                }.get(event_type, "🧠")
                lines.append(f"{event_emoji} [{subject}] {text}")
            
            return "🧠 Meta-Memories (Reflections):\n" + "\n".join(lines)

        # Chat Memories View
        if cmd in {"/chatmemories", "/chatmemory"}:
            items = self.memory_store.list_recent(limit=None)
            if not items:
                return "🧠 No saved memories."
            
            # Filter out daydream memories
            chat_items = [item for item in items if len(item) >= 5 and item[4] != 'daydream']
            
            if not chat_items:
                return "🧠 No chat memories found."

            type_emoji = {
                "IDENTITY": "👤", "FACT": "📌", "PREFERENCE": "❤️", 
                "GOAL": "🎯", "RULE": "⚖️", "PERMISSION": "✅", "BELIEF": "💭"
            }
            
            grouped = {}
            for item in chat_items:
                _id, mem_type, subject, text = item[:4]
                grouped.setdefault(mem_type, []).append((subject, text))
            
            lines = []
            hierarchy = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]
            
            for mem_type in hierarchy:
                if mem_type in grouped:
                    emoji = type_emoji.get(mem_type, "💡")
                    lines.append(f"\n{emoji} {mem_type}:")
                    for subject, text in grouped[mem_type]:
                        lines.append(f"  - [{subject}] {text}")
                    del grouped[mem_type]
            
            for mem_type, remaining in grouped.items():
                emoji = type_emoji.get(mem_type, "💡")
                lines.append(f"\n{emoji} {mem_type}:")
                for subject, text in remaining:
                    lines.append(f"  - [{subject}] {text}")
            
            return "🧠 Chat Memories (No Daydreams):\n" + "\n".join(lines)

        # Assistant Notes (formerly Special Memories)
        if cmd in {"/note", "/notes", "/specialmemory"}:
            # If arguments provided, create note
            if len(cmd_parts) > 1:
                content = text[len(cmd_parts[0]):].strip()
                if self.decider:
                    self.decider.create_note(content)
                    return f"📝 Note created: {content}"
                else:
                    return "❌ Decider not initialized."
            
            # List notes
            items = self.memory_store.list_recent(limit=None)
            if not items:
                return "🧠 No saved memories."
            
            notes = [item for item in items if item[1] == "NOTE"]
            
            if not notes:
                return "📝 No assistant notes found."
            
            lines = []
            for item in notes:
                # item: (id, type, subject, text, source, verified)
                _id, mem_type, subject, text = item[:4]
                lines.append(f"📝 [ID:{_id}] {text}")
            
            return "📝 Assistant Notes:\n" + "\n".join(lines)

        if cmd in {"/clearnotes", "/clearspecialmemory"}:
            items = self.memory_store.list_recent(limit=None)
            count = 0
            for item in items:
                if item[1] == "NOTE":
                    if self.memory_store.delete_entry(item[0]):
                        count += 1
            return f"📝 Cleared {count} notes."

        # Remove Summaries
        if cmd in {"/removesummaries", "/clearsummaries", "/deletesummaries"}:
            if not self.meta_memory_store:
                return "❌ Meta-memory store not initialized."
            
            count_summary = self.meta_memory_store.delete_by_event_type("SESSION_SUMMARY")
            count_analysis = self.meta_memory_store.delete_by_event_type("HOD_ANALYSIS")
            total = count_summary + count_analysis
            
            # Refresh UI if needed
            self.root.after(0, self.refresh_database_view)
            
            return f"🗑️ Removed {total} summaries ({count_summary} session summaries, {count_analysis} Hod analyses)."

        # Consolidate Summaries
        if cmd in {"/consolidatesummaries", "/compresssummaries"}:
            if not self.hod:
                return "❌ Hod not initialized."
            
            result = self.hod.consolidate_summaries()
            self.root.after(0, self.refresh_database_view)
            return result

        # Status
        if cmd == "/status":
            status_msg = "📊 **System Status**\n\n"
            status_msg += f"🔌 Telegram Bridge: {'Connected' if self.is_connected() else 'Disconnected'}\n"
            
            cycle_limit = int(self.settings.get("daydream_cycle_limit", 15))
            cycle_info = f"(Cycle {self.daydream_cycle_count}/{cycle_limit})"
            
            status_msg += f"🤖 AI Mode: {'🔒 Chat Mode (Daydream Paused)' if self.chat_mode_var.get() else '☁️ Daydream Mode (Active)'} {cycle_info}\n"
            status_msg += f"⚙️ Processing: {'⏳ Busy' if self.is_processing else '✅ Idle'}\n"
            status_msg += f"📚 Knowledge Base: {self.document_store.get_total_documents()} files ({self.document_store.get_total_chunks()} chunks)\n"
            
            mem_items = self.memory_store.list_recent(limit=None)
            verified_count = sum(1 for item in mem_items if len(item) > 5 and item[5] == 1)
            status_msg += f"🧠 Memory: {len(mem_items)} active nodes ({verified_count} verified)\n"
            return status_msg

        # Memory Statistics
        if cmd in {"/memorystatistics", "/memorystats"}:
            items = self.memory_store.list_recent(limit=None)
            if not items: return "📊 Memory is empty."
            
            by_type = {}
            by_source = {}
            verified_count = 0
            for item in items:
                mtype, source, is_verified = item[1], item[4], (item[5] if len(item) > 5 else 0)
                by_type[mtype] = by_type.get(mtype, 0) + 1
                by_source[source] = by_source.get(source, 0) + 1
                if is_verified: verified_count += 1
            
            stats = f"📊 **Memory Statistics**\n\n**Total:** {len(items)}\n**Verified:** {verified_count} ({verified_count/len(items)*100:.1f}%)\n\n**By Type:**\n" + "\n".join([f"- {t}: {c}" for t, c in sorted(by_type.items(), key=lambda x: x[1], reverse=True)]) + "\n\n**By Source:**\n" + "\n".join([f"- {s}: {c}" for s, c in sorted(by_source.items(), key=lambda x: x[1], reverse=True)])
            return stats

        # Exit Chat Mode
        if cmd == "/exitchatmode":
            if self.chat_mode_var.get():
                self.chat_mode_var.set(False)
                self.on_chat_mode_toggle()
                return "🔓 Chat Mode disabled. Daydreaming will resume shortly."
            return "ℹ️ Chat Mode is already disabled."

        # Daydream Status
        if cmd in {"/daydreamstatus", "/ddstatus"}:
            cycle_limit = int(self.settings.get("daydream_cycle_limit", 15))
            status_msg = "☁️ **Daydream Status**\n\n"
            
            if self.chat_mode_var.get():
                status_msg += "🚫 State: Paused (Chat Mode Active)\n"
            elif not self.daydreamer:
                status_msg += "❌ State: Not Initialized\n"
            else:
                status_msg += f"✅ State: {'Processing' if self.is_processing else 'Active (Idle loop)'}\n"
                
            status_msg += f"🔄 Cycle Progress: {self.daydream_cycle_count} / {cycle_limit}\n"
            return status_msg

        # Verification
        if cmd in {"/verifysources", "/verify"}:
            self.root.after(1000, self.verify_memory_sources)
            return "🕵️ Source verification scheduled."

        if cmd in {"/verifyall", "/verifyallsources"}:
            self.root.after(1000, self.verify_all_memory_sources)
            return "🕵️ Full verification loop scheduled."

        if cmd == "/stop":
            self.stop_processing()
            return "🛑 All processing stopped."

        if cmd in {"/stopverifying", "/stopverify"}:
            self.stop_processing()
            return "🛑 Verification stopped."
            
        if cmd == "/terminate_desktop":
            self.root.after(1000, self.root.destroy)
            return "👋 Shutting down desktop assistant..."

        # Decider Commands
        if cmd == "/decider":
            if len(cmd_parts) < 2:
                return "🤖 Decider Usage: /decider [up|down|daydream|verify|verifyall|loop|stopdaydream]"
            
            action = cmd_parts[1].lower()
            if not self.decider:
                return "❌ Decider not initialized."

            if action == "up":
                self.decider.increase_temperature()
                return "🌡️ Temperature increased."
            elif action == "down":
                self.decider.decrease_temperature()
                return "🌡️ Temperature decreased."
            elif action == "daydream":
                self.decider.start_daydream()
                return "☁️ Daydream triggered."
            elif action == "verify":
                self.decider.start_verification_batch()
                return "🕵️ Verification triggered."
            elif action == "verifyall":
                self.decider.verify_all()
                return "🕵️ Full verification triggered."
            elif action == "loop":
                self.decider.start_daydream_loop()
                return "🔄 Daydream loop enabled."
            elif action == "stopdaydream":
                self.decider.stop_daydream()
                return "🛑 Daydream stopped."
            else:
                return f"❌ Unknown decider action: {action}"

        # List Commands
        if cmd in {"/listcommands", "/help", "/commands"}:
            return (
                "🛠️ **Command List**\n\n"
                "**System:**\n"
                "• `/Status` - Show system state\n"
                "• `/DaydreamStatus` - Show daydream cycle info\n"
                "• `/ExitChatMode` - Resume daydreaming\n\n"
                "• `/Disrupt` - Interrupt current loop (Telegram only)\n"
                "• `/Stop` - Stop ALL processing (Chat, Docs, Verify)\n"
                "• `/StopVerifying` - Stop verification loop\n"
                "• `/Terminate_Desktop` - Close application\n\n"
                
                "**Memory:**\n"
                "• `/Memories` - Show all memories\n"
                "• `/ChatMemories` - Show chat memories\n"
                "• `/MetaMemories` - Show memory logs\n"
                "• `/MemoryStats` - Show memory counts\n"
                "• `/Consolidate` - Merge duplicates\n"
                "• `/SpecialMemories` - Show special memories\n"
                "• `/SpecialMemory [text]` - Add special memory\n"
                "• `/ClearSpecialMemory` - Clear all special memories\n"
                "• `/Verify` - Verify sources (batch)\n"
                "• `/VerifyAll` - Verify all sources\n\n"
                
                "**Docs:**\n"
                "• `/Documents` - List files\n"
                "• `/DocContent \"file\"` - Read file\n"
                "• `/RemoveDoc \"file\"` - Delete file\n\n"
                
                "**Cleanup:**\n"
                "• `/ResetChat` - Clear chat\n"
                "• `/ResetMemory` - Wipe DB\n"
                "• `/Remove[Type]` - Delete type (e.g. /RemoveGoal)"
            )

        return None

    def send_message(self, event=None):
        """Send message to both local chat and Telegram"""
        message = self.message_entry.get().strip()
        if not message:
            return

        # Interrupt background tasks if running (Local Disrupt)
        if self.is_processing:
             self.stop_processing_flag = True
             if self.decider:
                 self.decider.report_forced_stop()
             print("🛑 Local user message triggered disruption.")

        # Add to local chat UI immediately
        self.add_chat_message("You", message, "outgoing")
        self.message_entry.delete(0, tk.END)

    def send_image(self):
        """Select and send an image to the AI"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return

        # Get optional caption from entry
        caption = self.message_entry.get().strip()
        if not caption:
            caption = "Analyze this image."
        
        # Clear entry
        self.message_entry.delete(0, tk.END)

        # Interrupt background tasks if running (Local Disrupt)
        if self.is_processing:
             self.stop_processing_flag = True
             if self.decider:
                 self.decider.report_forced_stop()
             print("🛑 Local user image triggered disruption.")

        # Create a temp copy to avoid deleting the user's original file
        temp_dir = "./data/temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}_{filename}")
        shutil.copy2(file_path, temp_path)

        # Add to UI
        self.add_chat_message("You", f"{filename}\n{caption}", "outgoing", image_path=temp_path, trigger_processing=False)

        # Process in background
        threading.Thread(
            target=self.process_message_thread,
            args=(caption, True, None, temp_path),
            daemon=True
        ).start()

    def process_message_thread(self, user_text: str, is_local: bool, telegram_chat_id=None, image_path: str = None):
        """
        Core AI Logic: RAG -> LLM -> Memory Extraction -> Response
        Runs in a separate thread.
        """
        # Check for non-locking commands (read-only) to avoid waiting for processing lock
        cmd = user_text.strip().split()[0].lower() if user_text.strip() else ""
        NON_LOCKING_COMMANDS = {
            "/status", "/daydreamstatus", "/ddstatus", 
            "/memories", "/chatmemories", "/chatmemory", "/metamemories", "/meta-memories", 
            "/memorystats", "/memorystatistics", 
            "/documents", "/docs", "/listdocs", 
            "/listcommands", "/help", "/commands",
            "/specialmemories"
        }
        
        if cmd in NON_LOCKING_COMMANDS:
            try:
                chat_id = telegram_chat_id if telegram_chat_id else int(self.settings.get("chat_id", 0) or 0)
                response = self.handle_command(user_text.strip(), chat_id)
                if response:
                    self.root.after(0, lambda: self.add_chat_message("System", response, "incoming"))
                    if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
                        if telegram_chat_id:
                            self.telegram_bridge.send_message(response)
                        elif is_local:
                            self.telegram_bridge.send_message(f"Desktop Command: {user_text}")
                            self.telegram_bridge.send_message(response)
            except Exception as e:
                print(f"Error executing non-locking command: {e}")
            return

        # Acquire lock to ensure mutual exclusion with Daydream/Verification loops
        with self.processing_lock:
            self.is_processing = True
            self.stop_processing_flag = False
            
            try:
                # Determine Chat ID (Local uses 0 or configured ID, Telegram uses actual ID)
                chat_id = telegram_chat_id if telegram_chat_id else int(self.settings.get("chat_id", 0) or 0)
                
                # Inactivity Reset Check
                now = time.time()
                last = self.last_activity.get(chat_id)
                if last and (now - last) > self.INACTIVITY_RESET_SECONDS:
                    self.chat_memory[chat_id] = []
                    print(f"♻️ Chat history cleared due to inactivity for chat {chat_id}")
                self.last_activity[chat_id] = now

                if self.stop_processing_flag:
                    return

                # Check for commands
                if user_text.strip().startswith("/"):
                    response = self.handle_command(user_text.strip(), chat_id)
                    if response:
                        # Send response to UI
                        self.root.after(0, lambda: self.add_chat_message("System", response, "incoming"))
                        
                        # Send to Telegram if applicable
                        if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
                            if telegram_chat_id:
                                self.telegram_bridge.send_message(response)
                            elif is_local:
                                self.telegram_bridge.send_message(f"Desktop Command: {user_text}")
                                self.telegram_bridge.send_message(response)
                        return

                if self.stop_processing_flag:
                    return

                # 1. Prepare Context (Chat History)
                history = self.chat_memory.get(chat_id, [])
                history.append({"role": "user", "content": user_text})
                
                # Limit history
                if len(history) > 20: 
                    history = history[-20:]

                if self.stop_processing_flag:
                    return

                # Delegate core logic to Decider
                reply = self.decider.process_chat_message(
                    user_text=user_text,
                    history=history,
                    status_callback=lambda msg: self.root.after(0, lambda: self.status_var.set(msg)),
                    image_path=image_path
                )

                if self.stop_processing_flag:
                    return

                # Update History
                history.append({"role": "assistant", "content": reply})
                self.chat_memory[chat_id] = history

                # Update UI (Thread-safe)
                self.root.after(0, lambda: self.add_chat_message("Assistant", reply, "incoming"))

                # Now that the user has their reply, let the Decider think about what's next.
                if self.decider:
                    self.decider.run_post_chat_decision_cycle()

                # Send to Telegram if applicable
                if self.is_connected() and self.settings.get("telegram_bridge_enabled", False) and self.chat_mode_var.get():
                    # If local user typed it, send to Telegram
                    if is_local:
                        self.telegram_bridge.send_message(f"Desktop: {user_text}") # Optional: echo user text
                        self.telegram_bridge.send_message(reply)
                    # If it came from Telegram, just send the reply
                    elif telegram_chat_id:
                        self.telegram_bridge.send_message(reply)

            except Exception as e:
                error_msg = str(e)
                print(f"Error processing message: {error_msg}")
                self.root.after(0, lambda: self.add_chat_message("System", f"Error: {error_msg}", "incoming"))
            finally:
                # Do not delete image_path here, as UI needs it for display/click
                self.is_processing = False
                self.stop_processing_flag = False
                self.root.after(0, lambda: self.status_var.set("Ready"))

    def handle_telegram_document(self, msg: Dict):
        """Handle document upload from Telegram"""
        try:
            file_info = msg["document"]
            file_id = file_info["file_id"]
            file_name = file_info.get("file_name", "unknown_file")
            file_size = file_info.get("file_size", 0)
            chat_id = msg["chat_id"]
            bot_token = self.settings.get("bot_token")

            # Check supported types
            if not file_name.lower().endswith(('.pdf', '.docx')):
                self.telegram_bridge.send_message(f"⚠️ Unsupported file type: {file_name}. Please send PDF or DOCX.")
                return

            self.telegram_bridge.send_message(f"📄 Received {file_name}, processing...")

            # Get file path from Telegram
            file_data = get_file(bot_token, file_id)
            telegram_file_path = file_data["file_path"]

            # Download
            local_dir = "./data/uploaded_docs"
            os.makedirs(local_dir, exist_ok=True)
            local_file_path = os.path.join(local_dir, file_name)
            
            download_file(bot_token, telegram_file_path, local_file_path)

            # Check duplicates
            file_hash = self.document_store.compute_file_hash(local_file_path)
            if self.document_store.document_exists(file_hash):
                self.telegram_bridge.send_message(f"⚠️ Document '{file_name}' already exists in database. Skipping...")
                os.remove(local_file_path)
                return

            # Process
            chunks, page_count, file_type = self.document_processor.process_document(local_file_path)

            # Add to store
            self.document_store.add_document(
                file_hash=file_hash,
                filename=file_name,
                file_type=file_type,
                file_size=file_size,
                page_count=page_count,
                chunks=chunks,
                upload_source="telegram"
            )

            self.telegram_bridge.send_message(f"✅ Successfully added '{file_name}' to database ({len(chunks)} chunks).")
            
            # Cleanup
            os.remove(local_file_path)
            
            # Refresh GUI if needed
            self.root.after(0, self.refresh_documents)

        except Exception as e:
            print(f"Error handling Telegram document: {e}")
            if self.telegram_bridge:
                self.telegram_bridge.send_message(f"❌ Error processing document: {str(e)}")

    def handle_disrupt_command(self, chat_id):
        """Handle /disrupt command from Telegram to stop processing immediately"""
        print("🛑 Disrupt command received from Telegram.")
        if self.telegram_bridge:
            self.telegram_bridge.send_message("🛑 Disrupting current process...")
        
        self.stop_processing_flag = True
        
        if self.decider:
            self.decider.report_forced_stop()
            
        def reset_flag():
            time.sleep(1.5) 
            self.stop_processing_flag = False
            print("▶️ Decider ready for next turn (Cooldown active).")
            if self.telegram_bridge:
                self.telegram_bridge.send_message("▶️ Process disrupted. Decider is in cooldown.")
            
        threading.Thread(target=reset_flag, daemon=True).start()

    def poll_telegram_messages(self):
        """Poll for new messages from Telegram"""
        while self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
            try:
                messages = self.telegram_bridge.get_messages()
                for msg in messages:
                    # Ignore old messages (prevent death loop from stuck commands)
                    if msg.get("date", 0) < self.start_time:
                        print(f"⚠️ Ignoring old message from {msg.get('from')}: {msg.get('text', 'doc')}")
                        continue

                    if msg["type"] == "text":
                        # Reset status suppression on interaction
                        self.telegram_status_sent = False

                        # Check for disrupt command OR implicit disrupt on any message
                        text_content = msg.get("text", "").strip().lower()
                        is_explicit_disrupt = text_content == "/disrupt"
                        
                        if is_explicit_disrupt or self.is_processing:
                            self.handle_disrupt_command(msg["chat_id"])
                            if is_explicit_disrupt:
                                continue

                        # Show in UI
                        self.root.after(0, lambda m=msg: self.add_chat_message(m["from"], m["text"], "incoming"))
                        # Process logic
                        threading.Thread(
                            target=self.process_message_thread, 
                            args=(msg["text"], False, msg["chat_id"]), # Use actual chat_id from msg
                            daemon=True
                        ).start()
                    elif msg["type"] == "document":
                        # Handle document in background
                        threading.Thread(
                            target=self.handle_telegram_document,
                            args=(msg,),
                            daemon=True
                        ).start()
                    elif msg["type"] == "photo":
                        # Handle photo
                        try:
                            # Interrupt if processing (Implicit Disrupt)
                            if self.is_processing:
                                self.handle_disrupt_command(msg["chat_id"])

                            file_id = msg["photo"]["file_id"]
                            caption = msg.get("caption", "") or "Analyze this image."
                            
                            # Download to temp
                            temp_path = f"./data/temp_img_{file_id}.jpg"
                            file_data = get_file(self.settings.get("bot_token"), file_id)
                            download_file(self.settings.get("bot_token"), file_data["file_path"], temp_path)
                            
                            self.root.after(0, lambda m=msg, c=caption, p=temp_path: self.add_chat_message(m["from"], c, "incoming", image_path=p))
                            
                            threading.Thread(
                                target=self.process_message_thread,
                                args=(caption, False, msg["chat_id"], temp_path),
                                daemon=True
                            ).start()
                        except Exception as e:
                            print(f"Error handling photo: {e}")

                time.sleep(0.01)  # Poll every 0.01 seconds
            except Exception as e:
                print(f"Error polling messages: {e}")
                time.sleep(1)  # Wait longer on error

    def upload_documents(self):
        """Upload documents via GUI"""
        file_paths = filedialog.askopenfilenames(
            title="Select PDF or DOCX files",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("DOCX files", "*.docx"),
                ("All supported", "*.pdf *.docx")
            ]
        )

        if not file_paths:
            return

        def upload_thread():
            success_count = 0
            total_files = len(file_paths)

            # Only log if debug_log has been initialized
            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Starting upload of {total_files} document(s)")

            for i, file_path in enumerate(file_paths):
                try:
                    filename = os.path.basename(file_path)
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Processing ({i+1}/{total_files}): {filename}")

                    # Check for duplicates
                    file_hash = self.document_store.compute_file_hash(file_path)
                    if self.document_store.document_exists(file_hash):
                        if hasattr(self, 'debug_log'):
                            self.log_debug_message(f"Skipping duplicate: {filename}")
                        continue

                    # Process document
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Extracting text from: {filename}")
                    chunks, page_count, file_type = self.document_processor.process_document(file_path)
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Successfully extracted {len(chunks)} chunks from {filename} ({page_count} pages)")

                    # Add to store
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Adding document to store: {filename}")
                    self.document_store.add_document(
                        file_hash=file_hash,
                        filename=filename,
                        file_type=file_type,
                        file_size=os.path.getsize(file_path),
                        page_count=page_count,
                        chunks=chunks,
                        upload_source="desktop_gui"
                    )
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Successfully added: {filename}")

                    success_count += 1

                except Exception as e:
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                    print(f"Error processing {file_path}: {e}")

            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Upload complete: {success_count}/{total_files} documents processed successfully")

            # Update UI in main thread
            self.root.after(0, lambda: self.refresh_documents())  # This will update original_docs
            self.root.after(0, lambda: self.status_var.set(f"Uploaded {success_count} documents"))

        threading.Thread(target=upload_thread, daemon=True).start()

    def start_background_processes(self):
        """Start background processes"""
        # Start memory consolidation loop
        threading.Thread(target=self.consolidation_loop, daemon=True).start()
        # Start daydreaming loop
        threading.Thread(target=self.daydream_loop, daemon=True).start()

    def consolidation_loop(self):
        """Periodic memory consolidation"""
        # Initial delay to ensure startup logs are visible
        time.sleep(1)
        
        while True:
            try:
                if hasattr(self, 'consolidator'):
                    stats = self.consolidator.consolidate(time_window_hours=None)
                    if stats['processed'] > 0:
                        print(f"🧠 [Consolidator] Processed: {stats['processed']}, Consolidated: {stats['consolidated']}, Skipped: {stats['skipped']}")
                        if self.hod:
                            self.hod.perform_analysis("Consolidation")
                
                # Prune old operational meta-memories (keep last 3 days of logs)
                if hasattr(self, 'meta_memory_store'):
                    # 3 days = 259200 seconds
                    pruned_count = self.meta_memory_store.prune_events(max_age_seconds=259200)
                    if pruned_count > 0:
                        print(f"🧹 [Meta-Memory] Pruned {pruned_count} old operational events.")

            except Exception as e:
                print(f"Consolidation/Cleanup error: {e}")
            
            time.sleep(600) # 10 minutes

    def daydream_loop(self):
        """Continuous daydreaming loop"""
        time.sleep(2)  # Initial buffer
        
        while True:
            try:
                # Check stop flag
                if self.stop_processing_flag:
                    time.sleep(0.01)
                    continue

                # Check if Decider has work to do
                has_work = False
                if self.decider and self.decider.current_task != "wait":
                    has_work = True
                
                # Manage UI state based on work
                # Ensure chat input is enabled (allow chatting while daydreaming)
                self.root.after(0, lambda: self.toggle_chat_input(True))

                # If has work, try to acquire lock and run
                if not self.is_processing:
                    if has_work:
                        if self.processing_lock.acquire(blocking=False):
                            try:
                                # Double check inside lock
                                if self.is_processing: continue
                                
                                if self.decider:
                                    self.is_processing = True
                                    
                                    # Update status for UI
                                    task = self.decider.current_task.capitalize()
                                    remaining = self.decider.cycles_remaining
                                    status_msg = f"Active: {task} ({remaining} left)"
                                    self.root.after(0, lambda: self.status_var.set(status_msg))
                                    
                                    # Decider rules the loop
                                    self.decider.run_cycle()
                            finally:
                                self.is_processing = False
                                self.processing_lock.release()
                                self.root.after(0, lambda: self.status_var.set("Ready"))
                        else:
                            time.sleep(0.1)
                    else:
                        # No work, just sleep
                        # Run observer if idle to maintain "always working" state
                        if self.observer:
                            self.observer.perform_observation()
                        time.sleep(1.0)
                else:
                    time.sleep(0.01)
                        
            except Exception as e:
                print(f"Daydream loop error: {e}")
                time.sleep(1)

def main():
    root = tk.Tk()
    app = DesktopAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()