import os
import sqlite3
import time
import json
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


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

    def __init__(self, db_path: str = "./data/meta_memory.sqlite3", embed_fn=None):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.embed_fn = embed_fn
        self._init_db()
        
        self.faiss_index = None
        self.meta_id_mapping = []
        if FAISS_AVAILABLE:
            self._build_faiss_index()

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
                    created_at INTEGER NOT NULL,
                    embedding TEXT
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_event_type ON meta_memories(event_type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_subject ON meta_memories(subject);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_created ON meta_memories(created_at);")

            # Migration: Add embedding column if it doesn't exist
            try:
                con.execute("ALTER TABLE meta_memories ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass

            # Migration: Unify 'Decider' subject to 'Assistant'
            con.execute("UPDATE meta_memories SET subject = 'Assistant' WHERE subject = 'Decider'")

    def _build_faiss_index(self):
        """Build FAISS index from active meta-memories."""
        try:
            with self._connect() as con:
                rows = con.execute("SELECT id, embedding FROM meta_memories WHERE embedding IS NOT NULL").fetchall()
            
            embeddings = []
            ids = []
            
            for r in rows:
                if r[1]:
                    emb = np.array(json.loads(r[1]), dtype='float32')
                    embeddings.append(emb)
                    ids.append(r[0])
            
            if embeddings:
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                embeddings_matrix = np.array(embeddings)
                faiss.normalize_L2(embeddings_matrix)
                self.faiss_index.add(embeddings_matrix)
                self.meta_id_mapping = ids
                print(f"🧠 [Meta-Memory] FAISS index built with {len(ids)} records.")
        except Exception as e:
            print(f"⚠️ Failed to build FAISS index for meta-memory: {e}")
            self.faiss_index = None

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
        
        # Generate embedding
        embedding = None
        embedding_json = None
        if self.embed_fn:
            try:
                embedding = self.embed_fn(text)
                embedding_json = json.dumps(embedding.tolist())
            except Exception as e:
                print(f"⚠️ Failed to generate embedding for meta-memory: {e}")

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
                    created_at,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                embedding_json
            ))
            row_id = cur.lastrowid
            
            # Update FAISS
            if FAISS_AVAILABLE and embedding is not None:
                if self.faiss_index is None:
                    dimension = len(embedding)
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    self.meta_id_mapping = []
                
                emb_np = embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(emb_np)
                self.faiss_index.add(emb_np)
                self.meta_id_mapping.append(row_id)

            return row_id

    def add_event(self, event_type: str, subject: str, text: str) -> int:
        """
        Helper to log system events (Netzach, Hod, Decider actions) to meta-memory.
        Wraps add_meta_memory with a default memory_type.
        """
        return self.add_meta_memory(
            event_type=event_type,
            memory_type="SYSTEM_EVENT",
            subject=subject,
            text=text
        )

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

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict]:
        """Semantic search for meta-memories."""
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                q_emb = query_embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(q_emb)
                
                search_k = min(limit * 5, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(q_emb, search_k)
                
                candidate_ids = []
                candidate_scores = {}
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.meta_id_mapping):
                        mid = self.meta_id_mapping[idx]
                        candidate_ids.append(mid)
                        candidate_scores[mid] = float(scores[0][i])
                
                if not candidate_ids:
                    return []

                placeholders = ','.join(['?'] * len(candidate_ids))
                with self._connect() as con:
                    rows = con.execute(f"""
                        SELECT id, event_type, subject, text, created_at
                        FROM meta_memories
                        WHERE id IN ({placeholders})
                    """, candidate_ids).fetchall()
                
                results = []
                for r in rows:
                    mid = r[0]
                    if mid in candidate_scores:
                        results.append({
                            'id': mid,
                            'event_type': r[1],
                            'subject': r[2],
                            'text': r[3],
                            'created_at': r[4],
                            'similarity': candidate_scores[mid]
                        })
                
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:limit]
            except Exception as e:
                print(f"⚠️ FAISS search failed for meta-memory: {e}")
        
        return []

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

    def delete_by_event_type(self, event_type: str) -> int:
        """Delete meta-memories by event type."""
        with self._connect() as con:
            cur = con.execute("DELETE FROM meta_memories WHERE event_type = ?", (event_type.upper(),))
            con.commit()
            count = cur.rowcount
        
        if count > 0 and FAISS_AVAILABLE:
            # Rebuild index to remove deleted items
            self._build_faiss_index()
            
        return count

    def delete_entries(self, ids: List[int]) -> int:
        """Delete specific meta-memories by ID."""
        if not ids:
            return 0
        
        with self._connect() as con:
            placeholders = ','.join(['?'] * len(ids))
            cur = con.execute(f"DELETE FROM meta_memories WHERE id IN ({placeholders})", ids)
            con.commit()
            count = cur.rowcount
        
        if count > 0 and FAISS_AVAILABLE:
            self._build_faiss_index()
            
        return count

    def clear(self):
        """DANGEROUS: Clears all meta-memories."""
        with self._connect() as con:
            con.execute("DELETE FROM meta_memories")
        
        self.faiss_index = None
        self.meta_id_mapping = []

    def prune_events(self, max_age_seconds: int = 259200, event_types: List[str] = None) -> int:
        """
        Prune old meta-memories of specific types.
        Default: Prune system operational logs older than 3 days (259200 seconds).
        """
        if event_types is None:
            # Default noisy types to clean up
            event_types = [
                "DECIDER_ACTION", 
                "NETZACH_ACTION", 
                "HOD_INSTRUCTION", 
                "DECIDER_OBSERVATION_RECEIVED", 
                "TOOL_EXECUTION",
                "NETZACH_INFO",
                "NETZACH_INSTRUCTION",
                "STRATEGIC_THOUGHT",
                "CHAIN_OF_THOUGHT"
            ]
            
        cutoff = int(time.time()) - max_age_seconds
        
        if not event_types:
            return 0
            
        placeholders = ','.join(['?'] * len(event_types))
        params = [cutoff] + event_types
        
        with self._connect() as con:
            cur = con.execute(f"""
                DELETE FROM meta_memories 
                WHERE created_at < ? AND event_type IN ({placeholders})
            """, params)
            con.commit()
            count = cur.rowcount
        
        if count > 0 and FAISS_AVAILABLE:
            # Rebuild index since we removed items to keep IDs in sync
            self._build_faiss_index()
            
        return count
