import os
import sqlite3
import time
import json
from typing import List, Tuple, Optional


class MetaMemoryStore:
    """
    Meta-Memory Store: Tracks changes and reflections about memories.

    This is separate from the main memory store to keep meta-cognition
    distinct from actual memories.

    Meta-memories enable:
    - Self-reflection ("I used to be called Ada")
    - Temporal reasoning ("My name changed 3 times this week")
    - Change tracking ("User renamed me on Feb 4th")
    """

    def __init__(self, db_path: str = "./data/meta_memory.sqlite3"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS meta_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,      -- VERSION_UPDATE, CONFLICT_DETECTED, etc.
                    memory_type TEXT NOT NULL,     -- IDENTITY, FACT, PREFERENCE, etc.
                    subject TEXT NOT NULL,         -- User | Assistant
                    text TEXT NOT NULL,            -- Human-readable description
                    old_id INTEGER,                -- Reference to old memory
                    new_id INTEGER,                -- Reference to new memory
                    old_value TEXT,                -- Old value (extracted)
                    new_value TEXT,                -- New value (extracted)
                    metadata TEXT,                 -- Additional JSON metadata
                    created_at INTEGER NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_event_type ON meta_memories(event_type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_subject ON meta_memories(subject);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_created ON meta_memories(created_at);")

    def add_meta_memory(
        self,
        event_type: str,
        memory_type: str,
        subject: str,
        text: str,
        old_id: Optional[int] = None,
        new_id: Optional[int] = None,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Add a meta-memory about a memory change or event.

        Args:
            event_type: Type of event (VERSION_UPDATE, CONFLICT_DETECTED, etc.)
            memory_type: Type of memory affected (IDENTITY, FACT, etc.)
            subject: Who the memory is about (User, Assistant)
            text: Human-readable description
            old_id: ID of old memory (if applicable)
            new_id: ID of new memory (if applicable)
            old_value: Old value (extracted from memory text)
            new_value: New value (extracted from memory text)
            metadata: Additional structured data

        Returns:
            ID of created meta-memory
        """
        metadata_json = json.dumps(metadata) if metadata else None

        with self._connect() as con:
            cur = con.execute("""
                INSERT INTO meta_memories (
                    event_type,
                    memory_type,
                    subject,
                    text,
                    old_id,
                    new_id,
                    old_value,
                    new_value,
                    metadata,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_type.upper(),
                memory_type.upper(),
                subject,
                text.strip(),
                old_id,
                new_id,
                old_value,
                new_value,
                metadata_json,
                int(time.time()),
            ))
            return cur.lastrowid

    def list_recent(self, limit: int = 30) -> List[Tuple[int, str, str, str, str]]:
        """
        Get recent meta-memories.

        Returns: List of (id, event_type, subject, text, created_at)
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, event_type, subject, text, created_at
                FROM meta_memories
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows

    def get_by_subject(self, subject: str, limit: int = 30) -> List[dict]:
        """
        Get meta-memories for a specific subject (User or Assistant).
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, event_type, memory_type, subject, text,
                       old_id, new_id, old_value, new_value, metadata, created_at
                FROM meta_memories
                WHERE subject = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (subject, limit)).fetchall()

        result = []
        for r in rows:
            result.append({
                'id': r[0],
                'event_type': r[1],
                'memory_type': r[2],
                'subject': r[3],
                'text': r[4],
                'old_id': r[5],
                'new_id': r[6],
                'old_value': r[7],
                'new_value': r[8],
                'metadata': json.loads(r[9]) if r[9] else None,
                'created_at': r[10],
            })
        return result

    def get_by_event_type(self, event_type: str, limit: int = 30) -> List[dict]:
        """
        Get meta-memories by event type (e.g., VERSION_UPDATE).
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, event_type, memory_type, subject, text,
                       old_id, new_id, old_value, new_value, metadata, created_at
                FROM meta_memories
                WHERE event_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (event_type.upper(), limit)).fetchall()

        result = []
        for r in rows:
            result.append({
                'id': r[0],
                'event_type': r[1],
                'memory_type': r[2],
                'subject': r[3],
                'text': r[4],
                'old_id': r[5],
                'new_id': r[6],
                'old_value': r[7],
                'new_value': r[8],
                'metadata': json.loads(r[9]) if r[9] else None,
                'created_at': r[10],
            })
        return result

    def clear(self):
        """DANGEROUS: Clears all meta-memories."""
        with self._connect() as con:
            con.execute("DELETE FROM meta_memories")
