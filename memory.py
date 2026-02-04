import os
import sqlite3
import time
import json
import hashlib
from typing import List, Dict, Optional, Tuple


class MemoryStore:
    """
    Immutable, append-only memory ledger.

    Responsibilities:
    - Store memory items with versioning
    - Maintain identities for duplicate/version tracking
    - Support basic conflict detection
    """

    def __init__(self, db_path: str = "./data/memory.sqlite3"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    # --------------------------
    # Internal DB
    # --------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL,
                    parent_id INTEGER,
                    type TEXT NOT NULL,        -- FACT | PREFERENCE | GOAL | RULE | PERMISSION | IDENTITY
                    subject TEXT NOT NULL,     -- User | Assistant
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    conflict_with TEXT,
                    created_at INTEGER NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_identity ON memories(identity);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_subject ON memories(subject);")

    # --------------------------
    # Identity
    # --------------------------

    def compute_identity(self, text: str, mem_type: str = None) -> str:
        """
        Deterministic identity for a memory item.
        
        - IDENTITY type: Uses broad patterns (e.g., "User name is") to force versioning.
        - Other types: Uses full normalized text to allow multiple distinct items.
        """
        text_lower = " ".join(text.lower().strip().split())

        if mem_type and mem_type.upper() == "IDENTITY":
            # Patterns for identifying unique "slots" in identity
            patterns = [
                ("name is", "name is"),
                ("lives in", "lives in"),
                ("works at", "works at"),
                ("occupation is", "occupation is"),
                ("is currently a", "is currently a"),
                ("is now called", "name is"),
                ("is known as", "name is"),
                ("identity is", "identity is"),
            ]

            for trigger, norm in patterns:
                if trigger in text_lower:
                    parts = text_lower.split(trigger)
                    identity_base = parts[0] + norm
                    return hashlib.sha256(identity_base.encode("utf-8")).hexdigest()

        # Default: use full normalized text
        return hashlib.sha256(text_lower.encode("utf-8")).hexdigest()

    def exists_identity(self, identity: str) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM memories WHERE identity = ? LIMIT 1",
                (identity,),
            ).fetchone()
        return row is not None

    def get_by_identity(self, identity: str) -> List[Dict]:
        """
        Returns all versions of a claim, ordered chronologically.
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, parent_id, type, subject, text, confidence, source, conflict_with, created_at
                FROM memories
                WHERE identity = ?
                ORDER BY created_at ASC
            """, (identity,)).fetchall()

        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "parent_id": r[1],
                "type": r[2],
                "subject": r[3],
                "text": r[4],
                "confidence": r[5],
                "source": r[6],
                "conflict_with": json.loads(r[7] or "[]"),
                "created_at": r[8],
            })
        return result

    # --------------------------
    # Add memory (append-only)
    # --------------------------

    def add_entry(
        self,
        *,
        identity: str,
        text: str,
        mem_type: str,
        subject: str = "User",
        confidence: float,
        source: str,
        parent_id: Optional[int] = None,
        conflicts: Optional[List[int]] = None,
        created_at: Optional[int] = None,
    ) -> int:
        """
        Append a new memory event.

        identity MUST be provided
        subject should be 'User' or 'Assistant' (default: 'User')
        parent_id enables version chaining
        """
        if not identity:
            raise ValueError("identity must be explicitly provided")

        conflicts_json = json.dumps(conflicts or [])
        timestamp = created_at if created_at is not None else int(time.time())

        with self._connect() as con:
            cur = con.execute("""
                INSERT INTO memories (
                    identity,
                    parent_id,
                    type,
                    subject,
                    text,
                    confidence,
                    source,
                    conflict_with,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                identity,
                parent_id,
                mem_type.upper(),
                subject,
                text.strip()[:500],
                float(confidence),
                source,
                conflicts_json,
                timestamp,
            ))
            return cur.lastrowid

    # --------------------------
    # Query helpers
    # --------------------------

    def list_recent(self, limit: int = 30) -> List[Tuple[int, str, str, str]]:
        """
        Get recent memories, excluding old superseded versions.

        A memory is hidden if:
        1. It has a parent_id set (meaning it was superseded/consolidated).
        2. There's a newer memory with the EXACT same identity.
        3. There's a newer memory that explicitly points to it as a parent (supersedes it).
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT m.id, m.type, m.subject, m.text
                FROM memories m
                WHERE m.parent_id IS NULL
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE (newer.identity = m.identity OR newer.parent_id = m.id)
                    AND newer.created_at > m.created_at
                )
                ORDER BY m.created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows

    def get_memory_history(self, identity: str) -> List[Dict]:
        """
        Get full version history for a memory via parent_id chain.

        Use this to retrieve old consolidated/linked versions.
        LLM can call this to access previous versions of a memory
        (e.g., old names, previous preferences, etc.)

        Args:
            identity: The identity hash (e.g., "assistant name is")

        Returns: List of all versions in order (oldest → newest)
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, parent_id, type, subject, text, confidence, created_at
                FROM memories
                WHERE identity = ?
                ORDER BY created_at ASC
            """, (identity,)).fetchall()

        versions = []
        for r in rows:
            versions.append({
                'id': r[0],
                'parent_id': r[1],
                'type': r[2],
                'subject': r[3],
                'text': r[4],
                'confidence': r[5],
                'created_at': r[6],
            })
        return versions

    def find_conflicts_exact(self, text: str) -> List[Dict]:
        """
        Very conservative conflict detection.
        Only checks explicit negation overlap.
        """
        lowered = text.lower()
        negated = any(w in lowered for w in (" not ", " never ", " no "))
        if not negated:
            return []

        with self._connect() as con:
            rows = con.execute("SELECT id, type, text FROM memories").fetchall()

        conflicts = []
        for r in rows:
            if r[2].lower() in lowered or lowered in r[2].lower():
                conflicts.append({
                    "id": r[0],
                    "type": r[1],
                    "text": r[2],
                })
        return conflicts

    # --------------------------
    # Dangerous operations
    # --------------------------

    def clear(self):
        """DANGEROUS: Clears the entire memory ledger."""
        with self._connect() as con:
            con.execute("DELETE FROM memories")

    def clear_by_type(self, mem_type: str) -> int:
        """
        DANGEROUS: Clears all memories of a specific type.
        
        Args:
            mem_type: Memory type to clear (FACT, PREFERENCE, GOAL, etc.)
        
        Returns:
            Number of memories deleted
        """
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM memories WHERE type = ?",
                (mem_type.upper(),)
            )
            con.commit()
            return cur.rowcount
