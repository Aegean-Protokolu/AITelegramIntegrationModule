"""
UI Module for AI Telegram Desktop Assistant
Contains the DesktopAssistantUI mixin and UI-related helpers.
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import sys
import os
import json
import re
import threading
from datetime import datetime

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow not installed. Images will not be displayed. Run: pip install Pillow")
    Image = None

# Import prompts
from lm import DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT
from daydreaming import DAYDREAM_EXTRACTOR_PROMPT as DEFAULT_DAYDREAM_EXTRACTOR_PROMPT


class StdoutRedirector:
    def __init__(self, app, original_stream):
        self.app = app
        self.original_stream = original_stream

    def write(self, string):
        if self.original_stream:
            try:
                self.original_stream.write(string)
            except:
                pass
        self.app.log_to_main(string)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except:
                pass


class DesktopAssistantUI:
    """UI Mixin for DesktopAssistantApp"""

    def setup_ui(self):
        """Setup the main UI"""
        # Initialize buffers for thread-safe logging
        self.log_buffer = []
        self.debug_log_buffer = []

        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Chat tab
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="💬 Chat")

        # Logs tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📝 Logs")

        # Database tab
        self.database_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.database_frame, text="🗄️ Memory Database")

        # Documents tab
        self.docs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.docs_frame, text="📚 Documents")

        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="⚙️ Settings")

        # Help tab
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text="❓ Help")

        # About tab
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="ℹ️ About")

        self.setup_chat_tab()
        self.setup_logs_tab()
        self.setup_database_tab()
        self.setup_documents_tab()
        self.setup_settings_tab()
        self.setup_help_tab()
        self.setup_about_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, bootstyle="secondary")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_dark_theme(self):
        """Setup dark theme"""
        self.style = ttk.Style()
        if self.settings.get("theme", "dark") == "dark":
            # Configure dark theme
            self.style.theme_use("clam")
            self.style.configure(".", background="#2b2b2b", foreground="white")
            self.style.configure("TFrame", background="#2b2b2b")
            self.style.configure("TLabel", background="#2b2b2b", foreground="white")
            self.style.configure("TButton", background="#3a3a3a", foreground="white")
            self.style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b")
            self.style.map("TButton", background=[("active", "#4a4a4a")])

    def setup_chat_tab(self):
        """Setup chat interface"""
        # Determine font for emoji support
        chat_font = ("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 10)
        entry_font = ("Segoe UI Emoji", 11) if os.name == 'nt' else ("Arial", 11)

        # Create PanedWindow for split view
        self.chat_paned = ttk.Panedwindow(self.chat_frame, orient=tk.VERTICAL)
        self.chat_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Top: Assistant Chat ---
        assistant_frame = ttk.Frame(self.chat_paned)
        self.chat_paned.add(assistant_frame, weight=3)

        self.chat_history = scrolledtext.ScrolledText(
            assistant_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="white",
            font=chat_font
        )
        self.chat_history.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Bottom: Netzach Observations ---
        netzach_container = ttk.Frame(self.chat_paned)
        self.chat_paned.add(netzach_container, weight=1)

        ttk.Label(netzach_container, text="👁️ AI Interactions", font=("Arial", 9, "bold"),
                  bootstyle="info").pack(anchor=tk.W)

        self.netzach_history = scrolledtext.ScrolledText(
            netzach_container,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#1a1a2e",  # Slightly bluer/darker for distinction
            fg="#a0a0ff",  # Soft blue text
            font=("Consolas", 10),
            height=5
        )
        self.netzach_history.pack(fill=tk.BOTH, expand=True)

        # Input frame
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Configure large emoji style
        self.style.configure("Big.Link.TButton", font=("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 15))

        # Image button
        image_btn = ttk.Button(input_frame, text="📷", command=self.send_image, style="Big.Link.TButton")
        image_btn.pack(side=tk.LEFT, padx=(0, 2))

        # Message entry
        self.message_entry = ttk.Entry(input_frame, font=entry_font)
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind("<Return>", self.send_message)

        # Buttons ordered Right-to-Left to appear Left-to-Right:
        # STOP ALL <- STOP DAYDREAM <- DAYDREAM <- CHAT MODE <- CONNECT <- SEND

        # Stop button (STOP ALL)
        stop_button = ttk.Button(input_frame, text="Stop", command=self.stop_processing, bootstyle="danger")
        stop_button.pack(side=tk.RIGHT)

        # Daydream Stop button (STOP DAYDREAM)
        daydream_stop_button = ttk.Button(input_frame, text="Stop Daydream", command=self.stop_daydream, bootstyle="danger-outline")
        daydream_stop_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Daydream button (DAYDREAM)
        daydream_button = ttk.Button(input_frame, text="Daydream", command=self.start_daydream, bootstyle="info")
        daydream_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Chat Mode button (CHAT MODE)
        self.chat_mode_btn = ttk.Checkbutton(
            input_frame,
            text="Chat Mode",
            variable=self.chat_mode_var,
            bootstyle="warning-toolbutton",
            command=self.on_chat_mode_toggle
        )
        self.chat_mode_btn.pack(side=tk.RIGHT, padx=(0, 5))

        # Connect/Disconnect button (CONNECT)
        self.connect_button = ttk.Button(input_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Send button (SEND)
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(0, 5))

    def add_netzach_message(self, message):
        """Add message to Netzach's observation window"""
        self.netzach_history.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] ✨ {message}\n\n"

        self.netzach_history.insert(tk.END, formatted_msg)
        # Limit size
        if int(self.netzach_history.index('end-1c').split('.')[0]) > 100:
            self.netzach_history.delete("1.0", "2.0")

        self.netzach_history.see(tk.END)
        self.netzach_history.config(state=tk.DISABLED)

    def toggle_chat_input(self, enabled: bool):
        """Enable or disable chat input widgets"""
        state = tk.NORMAL if enabled else tk.DISABLED
        if hasattr(self, 'message_entry'):
            self.message_entry.config(state=state)
        if hasattr(self, 'send_button'):
            self.send_button.config(state=state)

    def setup_logs_tab(self):
        """Setup logs interface"""
        # Controls frame
        controls_frame = ttk.Frame(self.logs_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Clear button
        clear_button = ttk.Button(controls_frame, text="Clear Logs", command=self.clear_main_log, bootstyle="secondary")
        clear_button.pack(side=tk.RIGHT)

        # Log text area
        self.main_log_text = scrolledtext.ScrolledText(
            self.logs_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4"
        )
        self.main_log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def redirect_logging(self):
        """Redirect stdout and stderr to the logs tab"""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        sys.stdout = StdoutRedirector(self, self.original_stdout)
        sys.stderr = StdoutRedirector(self, self.original_stderr)

    def log_to_main(self, message):
        """Thread-safe logging to main log widget"""
        # Buffer for thread-safe access
        if not hasattr(self, 'log_buffer'): self.log_buffer = []
        self.log_buffer.append(message)
        if len(self.log_buffer) > 5000:
            self.log_buffer = self.log_buffer[-4000:]

        if hasattr(self, 'main_log_text'):
            self.root.after(0, lambda: self._log_to_main_safe(message))

    def _log_to_main_safe(self, message):
        """Internal method to update log widget"""
        try:
            self.main_log_text.config(state=tk.NORMAL)
            self.main_log_text.insert(tk.END, message)

            # Limit log size to prevent lag (keep last 2000 lines)
            num_lines = int(self.main_log_text.index('end-1c').split('.')[0])
            if num_lines > 2000:
                self.main_log_text.delete("1.0", f"{num_lines - 2000 + 1}.0")

            self.main_log_text.see(tk.END)
            self.main_log_text.config(state=tk.DISABLED)
        except Exception:
            pass

    def clear_main_log(self):
        """Clear the main log"""
        self.main_log_text.config(state=tk.NORMAL)
        self.main_log_text.delete(1.0, tk.END)
        self.main_log_text.config(state=tk.DISABLED)

    def setup_database_tab(self):
        """Setup database viewer interface"""
        # Database Notebook (Memories vs Meta-Memories)
        db_notebook = ttk.Notebook(self.database_frame)
        db_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Refresh button (Placed at top right, inline with tabs)
        refresh_button = ttk.Button(self.database_frame, text="🔄 Refresh", command=self.refresh_database_view,
                                    bootstyle="secondary")
        refresh_button.place(relx=1.0, x=-5, y=2, anchor="ne")

        # Compress Summaries button (Placed to the left of Refresh)
        compress_button = ttk.Button(self.database_frame, text="🗜️ Compress", command=self.compress_summaries,
                                   bootstyle="info-outline")
        compress_button.place(relx=1.0, x=-95, y=2, anchor="ne")

        # Export Summaries button (Placed to the left of Refresh)
        export_button = ttk.Button(self.database_frame, text="💾 Export Summaries", command=self.export_summaries,
                                   bootstyle="info")
        export_button.place(relx=1.0, x=-185, y=2, anchor="ne")

        # Verify All button (Placed to the left of Export)
        verify_all_button = ttk.Button(self.database_frame, text="🧹 Verify All", command=self.verify_all_memory_sources,
                                       bootstyle="warning")
        verify_all_button.place(relx=1.0, x=-325, y=2, anchor="ne")

        # Verify Batch button (Placed to the left of Verify All)
        verify_button = ttk.Button(self.database_frame, text="🧹 Verify Sources", command=self.verify_memory_sources,
                                   bootstyle="warning")
        verify_button.place(relx=1.0, x=-425, y=2, anchor="ne")

        # Stop Verification button (Placed to the left of Verify Sources)
        stop_verify_button = ttk.Button(self.database_frame, text="🛑 Stop", command=self.stop_processing,
                                        bootstyle="danger")
        stop_verify_button.place(relx=1.0, x=-535, y=2, anchor="ne")

        # Stats Label (Verified / Total)
        # Moved to notebook header (tab bar) via place() and event binding
        self.memory_stats_var = tk.StringVar(value="Verified: 0 / 0")
        self.stats_label = ttk.Label(self.notebook, textvariable=self.memory_stats_var, bootstyle="info")
        
        # Bind tab change to show/hide stats
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        # Trigger once to set initial state
        self.root.after(100, lambda: self._on_tab_changed(None))

        # Summaries Tab
        summaries_frame = ttk.Frame(db_notebook)
        db_notebook.add(summaries_frame, text="Summaries")
        self.summaries_scrollable_frame = self._setup_scrollable_frame(summaries_frame)

        # Chat Memories Tab
        chat_memories_frame = ttk.Frame(db_notebook)
        db_notebook.add(chat_memories_frame, text="Chat Memories")
        self.chat_memories_scrollable_frame = self._setup_scrollable_frame(chat_memories_frame)

        # Daydream Memories Tab
        daydream_memories_frame = ttk.Frame(db_notebook)
        db_notebook.add(daydream_memories_frame, text="Daydream Memories")
        self.daydream_memories_scrollable_frame = self._setup_scrollable_frame(daydream_memories_frame)

        # Assistant Notes Tab
        notes_frame = ttk.Frame(db_notebook)
        db_notebook.add(notes_frame, text="Assistant Notes")
        self.notes_scrollable_frame = self._setup_scrollable_frame(notes_frame)

        # Meta-Memories Tab
        meta_frame = ttk.Frame(db_notebook)
        db_notebook.add(meta_frame, text="Meta-Memories")

        self.meta_memories_text = scrolledtext.ScrolledText(
            meta_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 10),
            bg="#1e1e1e",
            fg="white"
        )
        self.meta_memories_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _on_tab_changed(self, event):
        """Handle tab change to show/hide stats label in the header"""
        try:
            if not hasattr(self, 'stats_label'): return
            
            # Get index of the currently selected tab
            current_idx = self.notebook.index("current")
            db_idx = self.notebook.index(self.database_frame)
            
            if current_idx == db_idx:
                self.stats_label.place(relx=1.0, x=-5, y=2, anchor="ne")
            else:
                self.stats_label.place_forget()
        except Exception:
            pass

    def _setup_scrollable_frame(self, parent_frame):
        """Helper to setup a scrollable frame structure"""
        canvas = tk.Canvas(parent_frame, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel support
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Attach handler to frame for child widgets to use
        scrollable_frame.on_mousewheel = _on_mousewheel

        return scrollable_frame

    def refresh_database_view(self):
        """Refresh the database views"""
        if not hasattr(self, 'memory_store') or not hasattr(self, 'meta_memory_store'):
            return

        # Update status immediately
        if hasattr(self, 'status_var'):
            self.status_var.set("Refreshing database...")

        def fetch_data():
            try:
                # Fetch data in background thread
                mem_items = self.memory_store.list_recent(limit=None)
                
                # Calculate stats
                total_count = len(mem_items)
                verified_count = sum(1 for item in mem_items if len(item) > 5 and item[5] == 1)
                unverified_beliefs = sum(1 for item in mem_items if item[1] == 'BELIEF' and (len(item) <= 5 or item[5] == 0))
                unverified_facts = sum(1 for item in mem_items if item[1] == 'FACT' and (len(item) <= 5 or item[5] == 0))
                stats_text = f"Total: {total_count} | Verified: {verified_count} | Unverified Beliefs: {unverified_beliefs} | Unverified Facts: {unverified_facts}"

                # Fetch Summaries
                summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=10)
                analyses = self.meta_memory_store.get_by_event_type("HOD_ANALYSIS", limit=10)
                summary_items = summaries + analyses
                summary_items.sort(key=lambda x: x['created_at'], reverse=True)

                # Fetch Meta-Memories
                meta_items = self.meta_memory_store.list_recent(limit=75)

                # Schedule UI update on main thread
                self.root.after(0, lambda: self._update_database_ui(mem_items, stats_text, summary_items, meta_items))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Refresh Error", f"Failed to refresh data: {e}"))
                self.root.after(0, lambda: self.status_var.set("Refresh failed"))

        # Start background thread
        threading.Thread(target=fetch_data, daemon=True).start()

    def _update_database_ui(self, mem_items, stats_text, summary_items, meta_items):
        """Update UI elements with fetched data (Main Thread)"""
        try:
            # 1. Clear existing widgets
            for widget in self.summaries_scrollable_frame.winfo_children(): widget.destroy()
            for widget in self.chat_memories_scrollable_frame.winfo_children(): widget.destroy()
            for widget in self.daydream_memories_scrollable_frame.winfo_children(): widget.destroy()
            for widget in self.notes_scrollable_frame.winfo_children(): widget.destroy()

            # 2. Update Stats
            self.memory_stats_var.set(stats_text)

            # 3. Populate Memories
            chat_items = []
            daydream_items = []
            note_items = []

            for item in mem_items:
                if item[1] == "NOTE":
                    note_items.append(item)
                elif len(item) >= 5 and item[4] == 'daydream':
                    daydream_items.append(item)
                else:
                    chat_items.append(item)

            self._populate_memory_tab(self.chat_memories_scrollable_frame, chat_items)
            self._populate_memory_tab(self.daydream_memories_scrollable_frame, daydream_items)
            self._populate_memory_tab(self.notes_scrollable_frame, note_items)

            # 4. Populate Summaries
            if not summary_items:
                lbl = ttk.Label(self.summaries_scrollable_frame, text="📜 No session summaries found.", padding=20)
                lbl.pack(anchor="center")
            else:
                for item in summary_items:
                    self._create_summary_card(self.summaries_scrollable_frame, item)

            # 5. Populate Meta-Memories
            self.meta_memories_text.config(state=tk.NORMAL)
            self.meta_memories_text.delete(1.0, tk.END)
            
            if not meta_items:
                self.meta_memories_text.insert(tk.END, "🧠 No meta-memories.")
            else:
                for (_id, event_type, subject, text, created_at) in meta_items:
                    event_emoji = {
                        "MEMORY_CREATED": "✨", "VERSION_UPDATE": "🔄",
                        "CONFLICT_DETECTED": "⚠️", "CONSOLIDATION": "🔗"
                    }.get(event_type, "🧠")

                    try:
                        date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = str(created_at)

                    self.meta_memories_text.insert(tk.END, f"[{date_str}] {event_emoji} [{subject}] {text}\n")
            
            self.meta_memories_text.config(state=tk.DISABLED)
            self.status_var.set("Ready")
        except Exception as e:
            print(f"UI Update Error: {e}")

    def export_summaries(self):
        """Export session summaries to a text file"""
        if not hasattr(self, 'meta_memory_store'):
            messagebox.showerror("Error", "Meta-memory store not initialized.")
            return

        try:
            # Fetch all summaries
            summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1000)
            analyses = self.meta_memory_store.get_by_event_type("HOD_ANALYSIS", limit=1000)
            
            all_items = summaries + analyses
            all_items.sort(key=lambda x: x['created_at'], reverse=True)

            if not all_items:
                messagebox.showinfo("Export", "No summaries to export.")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Summaries"
            )

            if not file_path:
                return

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("SESSION SUMMARIES & ANALYSES\n")
                f.write("============================\n\n")
                
                for item in all_items:
                    date_str = datetime.fromtimestamp(item['created_at']).strftime("%Y-%m-%d %H:%M:%S")
                    type_str = "SUMMARY" if item.get('event_type') == "SESSION_SUMMARY" else "ANALYSIS"
                    subject = item.get('subject', 'Unknown')
                    
                    f.write(f"[{date_str}] [{type_str}] [{subject}]\n")
                    f.write(f"{item['text']}\n")
                    f.write("-" * 50 + "\n\n")

            messagebox.showinfo("Export", f"Successfully exported {len(all_items)} items to {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export summaries: {e}")

    def compress_summaries(self):
        """Trigger summary consolidation"""
        if hasattr(self, 'hod') and self.hod:
            # Run in background to avoid freezing UI
            import threading
            def run_compression():
                result = self.hod.consolidate_summaries()
                self.root.after(0, lambda: messagebox.showinfo("Compression", result))
                self.root.after(0, self.refresh_database_view)
            threading.Thread(target=run_compression, daemon=True).start()
        else:
            messagebox.showerror("Error", "Hod not initialized.")

    def _create_summary_card(self, parent, item):
        """Create a card for a session summary"""
        date_str = datetime.fromtimestamp(item['created_at']).strftime("%Y-%m-%d %H:%M")
        
        # Different style for Hod Analysis
        style_color = "info" if item.get('event_type') == "SESSION_SUMMARY" else "secondary"
        prefix = "📅" if item.get('event_type') == "SESSION_SUMMARY" else "🔮"
        
        text = f"{prefix} {date_str} - {item['subject']}"
        
        # Robust LabelFrame creation handling different ttk/ttkbootstrap versions
        try:
            # Try bootstyle first (standard for ttkbootstrap)
            card = ttk.LabelFrame(parent, text=text, bootstyle=style_color)
        except Exception:
            try:
                # Try style (standard for ttk)
                style_name = f"{style_color.title()}.TLabelframe"
                card = ttk.LabelFrame(parent, text=text, style=style_name)
            except Exception:
                # Fallback to no style (standard tk or broken ttk)
                card = ttk.LabelFrame(parent, text=text)

        card.pack(fill=tk.X, pady=5, padx=5)
        
        lbl = ttk.Label(card, text=item['text'], wraplength=780, justify=tk.LEFT)
        lbl.pack(fill=tk.X, padx=5, pady=5)

    def _populate_memory_tab(self, parent_frame, items):
        """Populate a memory tab with sections"""
        if not items:
            lbl = ttk.Label(parent_frame, text="🧠 No saved memories.", padding=20)
            lbl.pack(anchor="center")
            return

        type_emoji = {
            "IDENTITY": "👤", "FACT": "📌", "PREFERENCE": "❤️",
            "GOAL": "🎯", "RULE": "⚖️", "PERMISSION": "✅", "BELIEF": "💭",
            "NOTE": "📝"
        }

        grouped = {}
        for item in items:
            # item: (id, type, subject, text, source)
            _id, mem_type, subject, text = item[:4]
            grouped.setdefault(mem_type, []).append((_id, subject, text))

        hierarchy = ["NOTE", "PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]

        for mem_type in hierarchy:
            if mem_type in grouped:
                all_items = grouped[mem_type]
                total_count = len(all_items)
                # Limit to latest 50 items per type to prevent lag
                self._create_memory_section(parent_frame, mem_type, all_items[:50], type_emoji, total_count)
                del grouped[mem_type]

        for mem_type, remaining in grouped.items():
            total_count = len(remaining)
            # Limit to latest 50 items per type
            self._create_memory_section(parent_frame, mem_type, remaining[:50], type_emoji, total_count)

    def on_memory_right_click(self, event, widget):
        """Handle right-click on memory text widget"""
        try:
            index = widget.index(f"@{event.x},{event.y}")
            tags = widget.tag_names(index)
            
            mem_id = None
            for tag in tags:
                if tag.startswith("mem_"):
                    try:
                        mem_id = int(tag.split("_")[1])
                        break
                    except:
                        pass
            
            if mem_id:
                menu = tk.Menu(self.root, tearoff=0)
                menu.add_command(label=f"❌ Delete Memory ID: {mem_id}", command=lambda: self.delete_memory_action(mem_id))
                menu.tk_popup(event.x_root, event.y_root)
        except Exception as e:
            print(f"Right-click error: {e}")

    def delete_memory_action(self, mem_id):
        """Delete a memory by ID and refresh view"""
        if not hasattr(self, 'memory_store'):
            return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete memory ID {mem_id}?"):
            try:
                if self.memory_store.delete_entry(mem_id):
                    self.status_var.set(f"Deleted memory {mem_id}")
                    self.refresh_database_view()
                else:
                    messagebox.showerror("Error", f"Failed to delete memory ID {mem_id}")
            except Exception as e:
                messagebox.showerror("Error", f"Error deleting memory: {e}")

    def _create_memory_section(self, parent, mem_type, items, type_emoji, total_count=None):
        """Helper to create a collapsible section for a memory type"""
        emoji = type_emoji.get(mem_type, "💡")
        count = len(items)
        if total_count is None:
            total_count = count

        if count < total_count:
            title = f"{emoji} {mem_type} ({count}/{total_count})"
        else:
            title = f"{emoji} {mem_type} ({total_count})"

        # Container for the whole section
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, expand=False, padx=5, pady=2)

        # Content container (holds canvas + scrollbar)
        content_container = ttk.Frame(container)

        # Toggle state
        is_open = tk.BooleanVar(value=True)

        def toggle():
            if is_open.get():
                content_container.pack_forget()
                is_open.set(False)
                toggle_btn.configure(text=f"▶ {title}")
            else:
                content_container.pack(fill=tk.X, expand=False, padx=10, pady=5)
                is_open.set(True)
                toggle_btn.configure(text=f"▼ {title}")

        # Header Button
        toggle_btn = ttk.Button(
            container,
            text=f"▼ {title}",
            command=toggle,
            bootstyle="secondary-outline",
            cursor="hand2"
        )
        toggle_btn.pack(fill=tk.X)

        # Pack content initially
        content_container.pack(fill=tk.X, expand=True, padx=10, pady=5)

        # Use a Text widget for performance (instead of hundreds of Frames)
        # Calculate height: approx 3 lines per item, max 20 lines
        widget_height = min(len(items) * 3, 20)
        if widget_height < 3: widget_height = 3

        text_widget = scrolledtext.ScrolledText(
            content_container,
            wrap=tk.WORD,
            height=widget_height,
            font=("Segoe UI", 9) if os.name == 'nt' else ("Arial", 9),
            bg="#2b2b2b",
            fg="white",
            borderwidth=0,
            highlightthickness=0
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags
        text_widget.tag_config("header", foreground="#61afef", font=("Segoe UI", 9, "bold"))
        text_widget.tag_config("content", foreground="#dcdcdc")
        text_widget.tag_config("separator", foreground="#4e4e4e")

        for _id, subject, text in items:
            header = f"[ID:{_id}] [{subject}]\n"
            content = f"{text}\n"
            sep = "-" * 80 + "\n"
            
            # Add unique tag for ID
            mem_tag = f"mem_{_id}"
            
            text_widget.insert(tk.END, header, ("header", mem_tag))
            text_widget.insert(tk.END, content, ("content", mem_tag))
            text_widget.insert(tk.END, sep, ("separator", mem_tag))

        text_widget.config(state=tk.DISABLED)
        
        # Bind right-click
        text_widget.bind("<Button-3>", lambda e: self.on_memory_right_click(e, text_widget))

    def setup_documents_tab(self):
        """Setup documents interface"""
        # Upload and search frame
        controls_frame = ttk.Frame(self.docs_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Upload button
        upload_button = ttk.Button(controls_frame, text="Upload Documents", command=self.upload_documents,
                                   bootstyle="secondary")
        upload_button.pack(side=tk.LEFT, padx=(0, 5))

        # Stop Processing button
        stop_docs_button = ttk.Button(controls_frame, text="Stop Processing", command=self.stop_processing,
                                      bootstyle="danger")
        stop_docs_button.pack(side=tk.LEFT, padx=(0, 5))

        # Refresh button
        refresh_button = ttk.Button(controls_frame, text="Refresh", command=self.refresh_documents,
                                    bootstyle="secondary")
        refresh_button.pack(side=tk.LEFT, padx=(0, 5))

        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(controls_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.RIGHT, padx=(0, 5))
        self.search_entry.bind('<KeyRelease>', self.filter_documents)
        self.search_entry.bind('<FocusIn>', self.on_search_focus_in)
        self.search_entry.bind('<FocusOut>', self.on_search_focus_out)

        # Set placeholder text
        self.set_placeholder_text()

        # Clear search button
        clear_search_button = ttk.Button(controls_frame, text="Clear", command=self.clear_search, bootstyle="secondary")
        clear_search_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Documents treeview
        docs_tree_frame = ttk.Frame(self.docs_frame)
        docs_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("ID", "Filename", "Type", "Pages", "Chunks", "Date Added")
        self.docs_tree = ttk.Treeview(docs_tree_frame, columns=columns, show="headings", height=15)

        # Define headings with click events for sorting and arrows
        for col in columns:
            self.docs_tree.heading(col, text=f"{col} ↕", command=lambda c=col: self.sort_column(c))
            self.docs_tree.column(col, width=100)

        docs_scrollbar = ttk.Scrollbar(docs_tree_frame, orient=tk.VERTICAL, command=self.docs_tree.yview)
        self.docs_tree.configure(yscrollcommand=docs_scrollbar.set)

        self.docs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        docs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Results label
        self.results_var = tk.StringVar(value="Showing all documents")
        self.results_label = ttk.Label(self.docs_frame, textvariable=self.results_var, bootstyle="secondary")
        self.results_label.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Buttons frame
        docs_buttons_frame = ttk.Frame(self.docs_frame)
        docs_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        delete_button = ttk.Button(docs_buttons_frame, text="Delete Selected", command=self.delete_selected_document,
                                   bootstyle="secondary")
        delete_button.pack(side=tk.LEFT)

        delete_all_button = ttk.Button(docs_buttons_frame, text="Delete All", command=self.delete_all_documents,
                                       bootstyle="secondary")
        delete_all_button.pack(side=tk.LEFT, padx=(5, 0))

        check_integrity_button = ttk.Button(docs_buttons_frame, text="Check Integrity",
                                            command=self.check_document_integrity, bootstyle="warning")
        check_integrity_button.pack(side=tk.LEFT, padx=(5, 0))

        # Clear log button on the right side
        clear_log_button = ttk.Button(docs_buttons_frame, text="Clear Log", command=self.clear_debug_log,
                                      bootstyle="secondary")
        clear_log_button.pack(side=tk.RIGHT)

        # Store original documents list
        self.original_docs = []
        # Track sort direction for each column
        self.sort_directions = {col: False for col in columns}  # False = ascending, True = descending

        # Initialize debug log first
        self.setup_debug_log()

    def setup_debug_log(self):
        """Setup debug log frame in the documents tab"""
        # Create debug log frame using regular Frame instead of LabelFrame to avoid ttkbootstrap issues
        debug_frame = ttk.Frame(self.docs_frame)
        debug_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add a label for the log section
        log_label = ttk.Label(debug_frame, text="Document Processing Log", font=("Arial", 10, "bold"))
        log_label.pack(anchor=tk.W, padx=5, pady=(5, 0))

        # Create text widget for logs - smaller height
        self.debug_log = tk.Text(debug_frame, height=6, state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(debug_frame, orient=tk.VERTICAL, command=self.debug_log.yview)
        self.debug_log.configure(yscrollcommand=scrollbar.set)

        self.debug_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0, 5))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 5))

    def log_debug_message(self, message):
        """Log a debug message to the debug log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        
        # Buffer for thread-safe access
        if not hasattr(self, 'debug_log_buffer'): self.debug_log_buffer = []
        self.debug_log_buffer.append(formatted)
        if len(self.debug_log_buffer) > 1000:
            self.debug_log_buffer = self.debug_log_buffer[-1000:]

        if hasattr(self, 'debug_log'):
            self.root.after(0, lambda: self._log_debug_safe(formatted))

    def _log_debug_safe(self, formatted_message):
        try:
            self.debug_log.config(state=tk.NORMAL)
            self.debug_log.insert(tk.END, formatted_message)

            # Limit log size to prevent lag (keep last 500 lines)
            num_lines = int(self.debug_log.index('end-1c').split('.')[0])
            if num_lines > 500:
                self.debug_log.delete("1.0", f"{num_lines - 500 + 1}.0")

            self.debug_log.see(tk.END)  # Auto-scroll to the end
            self.debug_log.config(state=tk.DISABLED)
        except Exception:
            pass

    def clear_debug_log(self):
        """Clear the debug log"""
        self.debug_log.config(state=tk.NORMAL)
        self.debug_log.delete(1.0, tk.END)
        self.debug_log.config(state=tk.DISABLED)

    def setup_settings_tab(self):
        """Setup settings interface"""
        # Buttons frame - Pack at bottom first to ensure visibility
        buttons_frame = ttk.Frame(self.settings_frame)
        buttons_frame.pack(side=tk.BOTTOM, pady=10)

        # Save button
        save_settings_button = ttk.Button(buttons_frame, text="Save Settings", command=self.save_settings_from_ui,
                                          bootstyle="primary")
        save_settings_button.pack(side=tk.LEFT, padx=5)

        # Load button - opens file dialog
        load_settings_button = ttk.Button(buttons_frame, text="Load Settings",
                                          command=self.load_settings_from_file_dialog, bootstyle="secondary")
        load_settings_button.pack(side=tk.LEFT, padx=5)

        settings_notebook = ttk.Notebook(self.settings_frame)
        settings_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Model settings
        model_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(model_frame, text=" Model")

        # URL settings box
        url_box = ttk.LabelFrame(model_frame, text="API URLs")
        url_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(url_box, text="Base URL:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.base_url_var = tk.StringVar(value="http://127.0.0.1:1234/v1")
        base_url_entry = ttk.Entry(url_box, textvariable=self.base_url_var, width=50)
        base_url_entry.grid(row=0, column=1, padx=5, pady=5)

        # Model settings box (removed padding as it causes TclError)
        model_box = ttk.LabelFrame(model_frame, text="Model Names")
        model_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(model_box, text="Chat Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.chat_model_var = tk.StringVar(value="qwen2.5-vl-7b-instruct-abliterated")
        chat_model_entry = ttk.Entry(model_box, textvariable=self.chat_model_var, width=50)
        chat_model_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(model_box, text="Embedding Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.embedding_model_var = tk.StringVar(value="text-embedding-nomic-embed-text-v1.5")
        embedding_model_entry = ttk.Entry(model_box, textvariable=self.embedding_model_var, width=50)
        embedding_model_entry.grid(row=1, column=1, padx=5, pady=5)

        # Generation settings tab
        gen_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(gen_frame, text=" Generation")

        gen_box = ttk.LabelFrame(gen_frame, text="Generation Parameters")
        gen_box.pack(fill=tk.X, padx=5, pady=5)
        gen_box.columnconfigure(1, weight=1)

        # Temperature
        ttk.Label(gen_box, text="Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.temperature_var = tk.DoubleVar(value=0.7)
        temp_scale = ttk.Scale(gen_box, from_=0.0, to=2.0, variable=self.temperature_var,
                               command=lambda v: self.temperature_var.set(round(float(v), 2)))
        temp_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        temp_entry = ttk.Entry(gen_box, textvariable=self.temperature_var, width=10)
        temp_entry.grid(row=0, column=2, padx=5, pady=5)

        # Top P
        ttk.Label(gen_box, text="Top P:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.top_p_var = tk.DoubleVar(value=0.94)
        top_p_scale = ttk.Scale(gen_box, from_=0.0, to=1.0, variable=self.top_p_var,
                                command=lambda v: self.top_p_var.set(round(float(v), 2)))
        top_p_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        top_p_entry = ttk.Entry(gen_box, textvariable=self.top_p_var, width=10)
        top_p_entry.grid(row=1, column=2, padx=5, pady=5)

        # Max Tokens
        ttk.Label(gen_box, text="Max Tokens:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_tokens_var = tk.IntVar(value=800)
        max_tokens_entry = ttk.Entry(gen_box, textvariable=self.max_tokens_var, width=10)
        max_tokens_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Auto-Adjust Step
        ttk.Label(gen_box, text="Auto-Adjust Step:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.temperature_step_var = tk.DoubleVar(value=0.20)
        step_scale = ttk.Scale(gen_box, from_=0.01, to=0.50, variable=self.temperature_step_var,
                               command=lambda v: self.temperature_step_var.set(round(float(v), 2)))
        step_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        step_entry = ttk.Entry(gen_box, textvariable=self.temperature_step_var, width=10)
        step_entry.grid(row=3, column=2, padx=5, pady=5)

        # Prompts settings tab
        prompts_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(prompts_frame, text=" Prompts")

        prompts_box = ttk.LabelFrame(prompts_frame, text="Prompts")
        prompts_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        prompts_box.columnconfigure(0, weight=1)
        prompts_box.rowconfigure(1, weight=1)
        prompts_box.rowconfigure(3, weight=1)

        ttk.Label(prompts_box, text="System Prompt:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.system_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=10, width=60)
        self.system_prompt_text.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(prompts_box, text="Memory Extractor Prompt:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.memory_extractor_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=15, width=60)
        self.memory_extractor_prompt_text.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(prompts_box, text="Daydream Extractor Prompt:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.daydream_extractor_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=15, width=60)
        self.daydream_extractor_prompt_text.grid(row=5, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Memory settings tab
        memory_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(memory_frame, text=" Memory")

        # Cycles & Limits
        limits_box = ttk.LabelFrame(memory_frame, text="Cycles & Limits")
        limits_box.pack(fill=tk.NONE, anchor=tk.W, padx=5, pady=5)

        ttk.Label(limits_box, text="Daydream Cycles (Before Verification):").grid(row=0, column=0, sticky=tk.W, padx=5,
                                                                                  pady=5)
        self.daydream_cycle_limit_var = tk.IntVar(value=15)
        ttk.Entry(limits_box, textvariable=self.daydream_cycle_limit_var, width=10).grid(row=0, column=1, padx=5,
                                                                                         pady=5)

        ttk.Label(limits_box, text="Inconclusive Deletion Limit:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_inconclusive_attempts_var = tk.IntVar(value=10)
        ttk.Entry(limits_box, textvariable=self.max_inconclusive_attempts_var, width=10).grid(row=1, column=1, padx=5,
                                                                                              pady=5)

        ttk.Label(limits_box, text="Retrieval Failure Deletion Limit:").grid(row=2, column=0, sticky=tk.W, padx=5,
                                                                             pady=5)
        self.max_retrieval_failures_var = tk.IntVar(value=10)
        ttk.Entry(limits_box, textvariable=self.max_retrieval_failures_var, width=10).grid(row=2, column=1, padx=5,
                                                                                           pady=5)

        # Consolidation Thresholds
        thresholds_box = ttk.LabelFrame(memory_frame, text="Consolidation Thresholds (0.0 - 1.0)")
        thresholds_box.pack(fill=tk.NONE, anchor=tk.W, padx=5, pady=5)

        self.threshold_vars = {}
        # Hierarchy order: PERMISSION -> RULE -> IDENTITY -> PREFERENCE -> GOAL -> FACT -> BELIEF
        types = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]

        # Create single column of entries
        for i, t in enumerate(types):
            ttk.Label(thresholds_box, text=f"{t}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.DoubleVar(value=0.9)
            self.threshold_vars[t] = var
            ttk.Entry(thresholds_box, textvariable=var, width=8).grid(row=i, column=1, padx=5, pady=5)

        # General settings tab
        general_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(general_frame, text=" General")

        # Startup settings
        startup_box = ttk.LabelFrame(general_frame, text="Startup Settings")
        startup_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(startup_box, text="Initial AI Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.ai_mode_var = tk.StringVar(value="Daydream")
        ai_mode_combo = ttk.Combobox(startup_box, textvariable=self.ai_mode_var, values=["Chat", "Daydream"],
                                     state="readonly", width=15)
        ai_mode_combo.grid(row=0, column=1, padx=5, pady=5)

        # Bridges settings
        bridges_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(bridges_frame, text=" Bridges")

        # Telegram bridge settings box
        telegram_box = ttk.LabelFrame(bridges_frame, text="Telegram Bridge Settings")
        telegram_box.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)

        # Telegram settings inside the box
        ttk.Label(telegram_box, text="Bot Token:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.bot_token_var = tk.StringVar()
        bot_token_entry = ttk.Entry(telegram_box, textvariable=self.bot_token_var, width=50)
        bot_token_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(telegram_box, text="Chat ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.chat_id_var = tk.StringVar()
        chat_id_entry = ttk.Entry(telegram_box, textvariable=self.chat_id_var, width=50)
        chat_id_entry.grid(row=1, column=1, padx=5, pady=5)

        # Telegram bridge toggle
        self.telegram_bridge_enabled = tk.BooleanVar()
        telegram_toggle = ttk.Checkbutton(
            telegram_box,
            text="Enable Telegram Bridge",
            variable=self.telegram_bridge_enabled,
            bootstyle="round-toggle"
        )
        telegram_toggle.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        # Appearance settings
        appearance_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(appearance_frame, text="Appearance")

        ttk.Label(appearance_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.theme_var = tk.StringVar(value="Darkly")
        theme_combo = ttk.Combobox(appearance_frame, textvariable=self.theme_var, values=[
            "Cosmo", "Cyborg", "Darkly"
        ])
        theme_combo.grid(row=0, column=1, padx=5, pady=5)

    def setup_help_tab(self):
        """Setup help interface"""
        help_text = scrolledtext.ScrolledText(self.help_frame, wrap=tk.WORD, state=tk.NORMAL,
                                              font=("Segoe UI", 10) if os.name == 'nt' else ("Arial", 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Define tags for formatting
        help_text.tag_config("h1", font=("Segoe UI", 16, "bold"), foreground="#61afef")
        help_text.tag_config("h2", font=("Segoe UI", 13, "bold"), foreground="#98c379")
        help_text.tag_config("h3", font=("Segoe UI", 11, "bold"), foreground="#e5c07b")
        help_text.tag_config("code", font=("Consolas", 9), background="#2c313a", foreground="#e06c75")
        help_text.tag_config("bold", font=("Segoe UI", 10, "bold"))
        help_text.tag_config("normal", font=("Segoe UI", 10))
        help_text.tag_config("quote", font=("Segoe UI", 10, "italic"), foreground="#c678dd")

        self.load_markdown_to_widget(help_text, "help.md")
        help_text.config(state=tk.DISABLED)

    def setup_about_tab(self):
        """Setup about interface"""
        about_text = scrolledtext.ScrolledText(self.about_frame, wrap=tk.WORD, state=tk.NORMAL)
        about_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tags
        about_text.tag_config("h1", font=("Segoe UI", 18, "bold"), foreground="#61afef", justify="center")
        about_text.tag_config("h2", font=("Segoe UI", 14, "bold"), foreground="#98c379")
        about_text.tag_config("h3", font=("Segoe UI", 12, "bold"), foreground="#e5c07b")
        about_text.tag_config("bold", font=("Segoe UI", 11, "bold"))
        about_text.tag_config("normal", font=("Segoe UI", 11))
        about_text.tag_config("quote", font=("Segoe UI", 11, "italic"), foreground="#c678dd", justify="center")
        about_text.tag_config("code", font=("Consolas", 9), background="#2c313a", foreground="#e06c75")

        self.load_markdown_to_widget(about_text, "about.md")
        about_text.config(state=tk.DISABLED)

    def load_markdown_to_widget(self, widget, filepath):
        """
        Reads a markdown file and inserts it into a Tkinter Text widget with formatting.
        Supports: # Headers, **bold**, `code`, > quotes.
        """
        if not os.path.exists(filepath):
            widget.insert(tk.END, f"File not found: {filepath}\n", "normal")
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.rstrip()
                tag = "normal"
                
                # Determine block-level tag
                if line.startswith("# "):
                    tag = "h1"
                    line = line[2:]
                elif line.startswith("## "):
                    tag = "h2"
                    line = line[3:]
                elif line.startswith("### "):
                    tag = "h3"
                    line = line[4:]
                elif line.startswith("> "):
                    tag = "quote"
                    line = line[2:]
                
                # Parse inline formatting (bold and code)
                # Regex splits by **bold** and `code`
                tokens = re.split(r'(\*\*.*?\*\*|`.*?`)', line)
                
                for token in tokens:
                    if not token: continue
                    
                    local_tag = tag
                    text_to_insert = token
                    
                    if token.startswith("**") and token.endswith("**"):
                        local_tag = "bold" if tag == "normal" else tag # Keep header font if header
                        text_to_insert = token[2:-2]
                    elif token.startswith("`") and token.endswith("`"):
                        local_tag = "code"
                        text_to_insert = token[1:-1]
                    
                    widget.insert(tk.END, text_to_insert, (tag, local_tag) if tag == "normal" else (tag,))
                
                widget.insert(tk.END, "\n")
        except Exception as e:
            widget.insert(tk.END, f"Error loading {filepath}: {e}\n", "normal")

    def load_settings_into_ui(self):
        """Load settings into UI fields"""
        self.bot_token_var.set(self.settings.get("bot_token", ""))
        self.chat_id_var.set(str(self.settings.get("chat_id", "")))
        self.theme_var.set(self.settings.get("theme", "Darkly"))
        self.telegram_bridge_enabled.set(self.settings.get("telegram_bridge_enabled", False))
        self.base_url_var.set(self.settings.get("base_url", "http://127.0.0.1:1234/v1"))
        self.chat_model_var.set(self.settings.get("chat_model", "qwen2.5-vl-7b-instruct-abliterated"))
        self.embedding_model_var.set(self.settings.get("embedding_model", "text-embedding-nomic-embed-text-v1.5"))
        self.temperature_var.set(self.settings.get("temperature", 0.7))
        self.top_p_var.set(self.settings.get("top_p", 0.94))
        self.max_tokens_var.set(self.settings.get("max_tokens", 800))
        self.temperature_step_var.set(self.settings.get("temperature_step", 0.20))

        self.system_prompt_text.delete(1.0, tk.END)
        self.system_prompt_text.insert(tk.END, self.settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT))

        self.memory_extractor_prompt_text.delete(1.0, tk.END)
        self.memory_extractor_prompt_text.insert(tk.END, self.settings.get("memory_extractor_prompt",
                                                                           DEFAULT_MEMORY_EXTRACTOR_PROMPT))

        self.daydream_extractor_prompt_text.delete(1.0, tk.END)
        self.daydream_extractor_prompt_text.insert(tk.END, self.settings.get("daydream_extractor_prompt",
                                                                             DEFAULT_DAYDREAM_EXTRACTOR_PROMPT))

        self.ai_mode_var.set(self.settings.get("ai_mode", "Daydream"))

        # Memory settings
        self.daydream_cycle_limit_var.set(self.settings.get("daydream_cycle_limit", 15))
        self.max_inconclusive_attempts_var.set(self.settings.get("max_inconclusive_attempts", 10))
        self.max_retrieval_failures_var.set(self.settings.get("max_retrieval_failures", 10))

        thresholds = self.settings.get("consolidation_thresholds", {})
        default_thresholds = {"GOAL": 0.88, "IDENTITY": 0.87, "BELIEF": 0.87, "PERMISSION": 0.87, "FACT": 0.93,
                              "PREFERENCE": 0.93, "RULE": 0.93}
        for t, var in self.threshold_vars.items():
            var.set(thresholds.get(t, default_thresholds.get(t, 0.9)))

    def load_settings_from_file_dialog(self):
        """Load settings from a selected file via dialog"""
        file_path = filedialog.askopenfilename(
            title="Select settings file",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ],
            initialfile="settings.json"
        )

        if not file_path:
            return  # User cancelled the dialog

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)

            # Update current settings with loaded ones
            self.settings.update(loaded_settings)

            # Update the settings file path to the loaded file
            self.settings_file_path = file_path

            # Update UI with new settings
            self.load_settings_into_ui()

            # Apply the loaded theme
            theme_map = {
                "Cosmo": "cosmo",
                "Cyborg": "cyborg",
                "Darkly": "darkly"
            }
            theme_to_apply = theme_map.get(self.settings.get("theme", "Darkly"), self.settings.get("theme", "darkly"))
            self.style.theme_use(theme_to_apply)

            # Update connection state based on loaded settings
            if (self.settings.get("telegram_bridge_enabled", False) and
                    self.settings.get("bot_token") and
                    self.settings.get("chat_id")):
                self.telegram_bridge_enabled.set(True)
                self.connect()
            else:
                self.telegram_bridge_enabled.set(False)
                self.disconnect()

            messagebox.showinfo("Settings", f"Settings loaded successfully from {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings from {file_path}:\n{str(e)}")

    def save_settings_from_ui(self):
        """Save settings from UI fields"""
        self.settings["bot_token"] = self.bot_token_var.get()
        self.settings["chat_id"] = self.chat_id_var.get()
        self.settings["theme"] = self.theme_var.get().lower()  # Convert to lowercase for ttkbootstrap
        self.settings["telegram_bridge_enabled"] = self.telegram_bridge_enabled.get()
        self.settings["base_url"] = self.base_url_var.get()
        self.settings["chat_model"] = self.chat_model_var.get()
        self.settings["embedding_model"] = self.embedding_model_var.get()
        self.settings["temperature"] = self.temperature_var.get()
        self.settings["top_p"] = self.top_p_var.get()
        self.settings["max_tokens"] = self.max_tokens_var.get()
        self.settings["temperature_step"] = self.temperature_step_var.get()
        self.settings["system_prompt"] = self.system_prompt_text.get(1.0, tk.END).strip()
        self.settings["memory_extractor_prompt"] = self.memory_extractor_prompt_text.get(1.0, tk.END).strip()
        self.settings["daydream_extractor_prompt"] = self.daydream_extractor_prompt_text.get(1.0, tk.END).strip()
        self.settings["ai_mode"] = self.ai_mode_var.get()

        # Memory settings
        self.settings["daydream_cycle_limit"] = self.daydream_cycle_limit_var.get()
        self.settings["max_inconclusive_attempts"] = self.max_inconclusive_attempts_var.get()
        self.settings["max_retrieval_failures"] = self.max_retrieval_failures_var.get()

        thresholds = {t: var.get() for t, var in self.threshold_vars.items()}
        self.settings["consolidation_thresholds"] = thresholds

        self.save_settings()

        # Apply new theme (convert capitalized name to lowercase)
        theme_map = {
            "Cosmo": "cosmo",
            "Cyborg": "cyborg",
            "Darkly": "darkly"
        }
        theme_to_apply = theme_map.get(self.settings["theme"].capitalize(), self.settings["theme"])
        self.style.theme_use(theme_to_apply)

        # Only attempt connection if bridge is enabled and both credentials are provided
        if (self.settings["telegram_bridge_enabled"] and
                self.settings["bot_token"] and
                self.settings["chat_id"]):
            self.connect()
        elif not self.settings["telegram_bridge_enabled"]:
            # If bridge is disabled, make sure we're disconnected
            self.disconnect()

        messagebox.showinfo("Settings", "Settings saved successfully!")

    def add_chat_message(self, sender, message, message_type="incoming", image_path=None, trigger_processing=True):
        """Add message to chat history"""
        self.chat_history.config(state=tk.NORMAL)

        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Insert Header
        self.chat_history.insert(tk.END, f"[{timestamp}] {sender}:\n", message_type)

        # Insert Image if present
        if image_path and os.path.exists(image_path) and Image:
            try:
                # Load and resize to fixed thumbnail
                img = Image.open(image_path)
                thumbnail_size = (250, 250) # Uniform size box
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Keep reference to prevent GC
                if not hasattr(self, 'image_references'):
                    self.image_references = []
                self.image_references.append(photo)
                
                # Insert image
                self.chat_history.image_create(tk.END, image=photo, padx=5, pady=5)
                
                # Bind click event to open original
                img_tag = f"img_{len(self.image_references)}"
                self.chat_history.tag_add(img_tag, "end-1c")
                self.chat_history.tag_bind(img_tag, "<Button-1>", lambda e, p=image_path: self.open_image(p))
                self.chat_history.tag_bind(img_tag, "<Enter>", lambda e: self.chat_history.config(cursor="hand2"))
                self.chat_history.tag_bind(img_tag, "<Leave>", lambda e: self.chat_history.config(cursor=""))
                
                self.chat_history.insert(tk.END, "\n")
            except Exception as e:
                print(f"Error displaying image: {e}")
                self.chat_history.insert(tk.END, f"[Image Error: {e}]\n", message_type)

        # Insert Message Text
        self.chat_history.insert(tk.END, f"{message}\n\n", message_type)

        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

        # If this was an outgoing message (from local user), process it
        if message_type == "outgoing" and trigger_processing:
            # Run processing in background thread to avoid freezing GUI
            import threading
            threading.Thread(target=self.process_message_thread, args=(message, True)).start()

    def open_image(self, path):
        """Open image in default viewer"""
        try:
            if os.name == 'nt':
                os.startfile(path)
            else:
                # Cross platform fallback using PIL
                if Image:
                    img = Image.open(path)
                    img.show()
        except Exception as e:
            print(f"Error opening image: {e}")

    def clear_search(self):
        """Clear the search field"""
        self.search_var.set("")
        self.search_entry.delete(0, tk.END)  # Actually clear the entry field
        self.set_placeholder_text()  # Set placeholder text
        self.filter_documents()
        self.search_entry.focus_set()  # Set focus back to search box

    def set_placeholder_text(self):
        """Set placeholder text in search box"""
        self.search_entry.delete(0, tk.END)  # Clear any existing text
        self.search_entry.insert(0, "Search documents...")
        self.is_showing_placeholder = True  # Mark that we're showing placeholder

    def on_search_focus_in(self, event):
        """Handle focus in for search entry"""
        if self.is_showing_placeholder:
            self.search_entry.delete(0, tk.END)
            self.is_showing_placeholder = False  # Mark that we're no longer showing placeholder

    def on_search_focus_out(self, event):
        """Handle focus out for search entry"""
        if not self.search_var.get():  # If search box is empty
            self.set_placeholder_text()

    def sort_column(self, col):
        """Sort the treeview by the given column with toggle functionality"""
        # Toggle sort direction for this column
        self.sort_directions[col] = not self.sort_directions[col]
        is_descending = self.sort_directions[col]

        # Update heading text to show sort direction
        arrow = " ↓" if is_descending else " ↑"
        for c in self.sort_directions.keys():
            current_text = self.docs_tree.heading(c)['text']
            # Remove any existing arrows
            clean_text = current_text.split()[0]  # Get just the column name
            if c == col:
                # This is the column being sorted, show the current sort direction
                self.docs_tree.heading(c, text=f"{clean_text}{arrow}")
            else:
                # Other columns show the default arrow
                self.docs_tree.heading(c, text=f"{clean_text} ↕")

        # Get all items
        items = [(self.docs_tree.set(k, col), k) for k in self.docs_tree.get_children('')]

        # Determine if we're sorting numbers or text
        is_numeric = col in ['ID', 'Pages', 'Chunks']

        # Sort based on column type
        if is_numeric:
            # Convert to integer for comparison, handle 'N/A' values
            def sort_key(item):
                val = item[0]
                if val == 'N/A' or val == '':
                    return float('-inf') if is_descending else float('inf')  # Put N/A at the end
                try:
                    return int(val)
                except ValueError:
                    try:
                        return float(val)
                    except ValueError:
                        return val  # Fallback to string comparison if conversion fails

            items.sort(key=sort_key, reverse=is_descending)
        else:
            # For text columns, sort alphabetically (case-insensitive)
            items.sort(key=lambda x: x[0].lower(), reverse=is_descending)

        # Rearrange items in sorted order
        for index, (val, k) in enumerate(items):
            self.docs_tree.move(k, '', index)

        # Log sort action
        if hasattr(self, 'debug_log'):
            sort_order = "descending" if is_descending else "ascending"
            self.log_debug_message(f"Sorted by {col} ({sort_order})")

    def refresh_documents(self):
        """Refresh documents list"""
        if not hasattr(self, 'document_store'):
            return
        # Get all documents from store
        all_docs = self.document_store.list_documents(limit=1000)

        # Store original documents
        self.original_docs = all_docs

        # Clear current tree
        for item in self.docs_tree.get_children():
            self.docs_tree.delete(item)

        # Add documents to tree
        for doc in all_docs:
            doc_id, filename, file_type, page_count, chunk_count, created_at = doc
            date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            page_info = str(page_count) if page_count else "N/A"

            self.docs_tree.insert("", "end", values=(
                doc_id,
                filename,
                file_type.upper(),
                page_info,
                chunk_count,
                date_str
            ))

        # Reset sort directions and heading arrows
        columns = ("ID", "Filename", "Type", "Pages", "Chunks", "Date Added")
        for col in columns:
            self.sort_directions[col] = False
            self.docs_tree.heading(col, text=f"{col} ↕")

        # Update results label
        self.results_var.set(f"Showing all {len(all_docs)} documents")

        # Only log if debug_log has been initialized
        if hasattr(self, 'debug_log'):
            self.log_debug_message(f"Refreshed document list: {len(all_docs)} documents loaded")

    def filter_documents(self, event=None):
        """Filter documents based on search term"""
        # Check if we're currently showing the placeholder
        if self.is_showing_placeholder:
            search_term = ""
        else:
            search_term = self.search_var.get().strip().lower()

        # Clear current tree
        for item in self.docs_tree.get_children():
            self.docs_tree.delete(item)

        if not search_term:
            # No search term, show all documents
            filtered_docs = self.original_docs
            self.results_var.set(f"Showing all {len(filtered_docs)} documents")
        else:
            # Filter documents based on search term
            filtered_docs = []
            for doc in self.original_docs:
                doc_id, filename, file_type, page_count, chunk_count, created_at = doc
                # Check if search term is in filename, type, or other fields
                if (search_term in filename.lower() or
                        search_term in file_type.lower() or
                        search_term in str(page_count) or
                        search_term in str(chunk_count)):
                    filtered_docs.append(doc)

            self.results_var.set(f"Showing {len(filtered_docs)} of {len(self.original_docs)} documents")

        # Add filtered documents to tree
        for doc in filtered_docs:
            doc_id, filename, file_type, page_count, chunk_count, created_at = doc
            date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            page_info = str(page_count) if page_count else "N/A"

            self.docs_tree.insert("", "end", values=(
                doc_id,
                filename,
                file_type.upper(),
                page_info,
                chunk_count,
                date_str
            ))

        # After filtering, if there was a sort applied, reapply it
        # Find which column is currently sorted (has an arrow other than ↕)
        for col in self.sort_directions.keys():
            current_text = self.docs_tree.heading(col)['text']
            if "↓" in current_text or "↑" in current_text:
                # Reapply the sort to the filtered results
                self.sort_column(col)
                break

        # Log filter action
        if hasattr(self, 'debug_log'):
            if search_term:
                self.log_debug_message(f"Filtered documents with search term: '{search_term}'")
            else:
                self.log_debug_message("Cleared document filter, showing all documents")

    def delete_selected_document(self):
        """Delete selected document"""
        selected = self.docs_tree.selection()
        if not selected:
            messagebox.showwarning("Delete", "Please select a document to delete")
            return

        values = self.docs_tree.item(selected[0], "values")
        doc_id = int(values[0])

        if messagebox.askyesno("Confirm Delete", f"Delete document {values[1]}?"):
            success = self.document_store.delete_document(doc_id)
            if success:
                if hasattr(self, 'debug_log'):
                    self.log_debug_message(f"Deleted document: {values[1]} (ID: {doc_id})")
                self.refresh_documents()  # This will update original_docs
                self.status_var.set("Document deleted")

    def delete_all_documents(self):
        """Delete all documents"""
        doc_count = len(self.docs_tree.get_children())
        if doc_count == 0:
            messagebox.showinfo("Delete All", "No documents to delete")
            return

        if messagebox.askyesno("Confirm Delete All", f"Delete all {doc_count} documents?"):
            # Get all doc IDs first
            docs = self.document_store.list_documents(limit=1000)
            deleted_count = 0
            for doc in docs:
                if self.document_store.delete_document(doc[0]):
                    deleted_count += 1

            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Deleted all {deleted_count} documents")
            self.refresh_documents()  # This will update original_docs
            self.status_var.set(f"Deleted {deleted_count} documents")

    def check_document_integrity(self):
        """Check for broken or incomplete documents"""
        if not hasattr(self.document_store, 'find_broken_documents'):
            messagebox.showinfo("Info", "Integrity check not supported by current document store.")
            return

        broken_docs = self.document_store.find_broken_documents()

        if not broken_docs:
            messagebox.showinfo("Integrity Check", "✅ No broken documents found.\nDatabase integrity is good.")
            return

        msg = f"⚠️ Found {len(broken_docs)} broken document(s):\n\n"
        for doc in broken_docs[:10]:
            msg += f"• {doc['filename']} (ID: {doc['id']})\n  Issue: {doc['issue']}\n"

        if len(broken_docs) > 10:
            msg += f"\n...and {len(broken_docs) - 10} more."

        msg += "\nDo you want to delete these broken documents?"

        if messagebox.askyesno("Integrity Check", msg):
            deleted_count = 0
            for doc in broken_docs:
                if self.document_store.delete_document(doc['id']):
                    deleted_count += 1

            self.refresh_documents()
            messagebox.showinfo("Cleanup", f"🗑️ Deleted {deleted_count} broken documents.")