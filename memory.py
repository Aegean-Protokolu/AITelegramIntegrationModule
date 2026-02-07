import os
import sqlite3
import time
import json
import hashlib
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


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
        
        self.faiss_index = None
        self.memory_id_mapping = []  # Maps FAISS index ID -> SQLite ID
        if FAISS_AVAILABLE:
            self._build_faiss_index()

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
                    created_at INTEGER NOT NULL,
                    embedding TEXT,
                    verified INTEGER DEFAULT 0
                )
            """)
            
            # Migration: Add deleted column
            try:
                con.execute("ALTER TABLE memories ADD COLUMN deleted INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            con.execute("CREATE INDEX IF NOT EXISTS idx_identity ON memories(identity);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_subject ON memories(subject);")

            # Migration: Add embedding column if it doesn't exist (for existing DBs)
            try:
                con.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass
            
            # Migration: Add verified column if it doesn't exist
            try:
                con.execute("ALTER TABLE memories ADD COLUMN verified INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            # Migration: Add verification_attempts column if it doesn't exist
            try:
                con.execute("ALTER TABLE memories ADD COLUMN verification_attempts INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            # Migration: Add flags column
            try:
                con.execute("ALTER TABLE memories ADD COLUMN flags TEXT")
            except sqlite3.OperationalError:
                pass

            # New table for tracking consolidation history to avoid re-checking pairs
            con.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_history (
                    id_a INTEGER NOT NULL,
                    id_b INTEGER NOT NULL,
                    similarity REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (id_a, id_b)
                )
            """)
            
            # Migration: Unify 'Decider' subject to 'Assistant'
            con.execute("UPDATE memories SET subject = 'Assistant' WHERE subject = 'Decider'")

    def _build_faiss_index(self):
        """
        Build FAISS index from active memories in DB.
        This runs on startup to cache embeddings.
        """
        try:
            # Fetch all active memories with embeddings
            # Use cursor to fetch in batches to avoid OOM
            with self._connect() as con:
                cur = con.execute("""
                    SELECT id, embedding FROM memories 
                    WHERE parent_id IS NULL 
                    AND deleted = 0
                    AND embedding IS NOT NULL
                """)
            
                batch_size = 5000
                total_loaded = 0
                
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                        
                    embeddings = []
                    ids = []
                    
                    for r in rows:
                        if r[1]:
                            try:
                                emb = np.array(json.loads(r[1]), dtype='float32')
                                embeddings.append(emb)
                                ids.append(r[0])
                            except:
                                continue
                    
                    if embeddings:
                        if self.faiss_index is None:
                            dimension = len(embeddings[0])
                            self.faiss_index = faiss.IndexFlatIP(dimension)
                            self.memory_id_mapping = []
                        
                        # Check dimension consistency
                        if len(embeddings[0]) == self.faiss_index.d:
                            embeddings_matrix = np.array(embeddings)
                            faiss.normalize_L2(embeddings_matrix)
                            self.faiss_index.add(embeddings_matrix)
                            self.memory_id_mapping.extend(ids)
                            total_loaded += len(ids)

                if total_loaded > 0:
                    print(f"🧠 [Memory] FAISS index built with {total_loaded} active memories.")
                    
        except Exception as e:
            print(f"⚠️ Failed to build FAISS index for memory: {e}")
            self.faiss_index = None

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

        # Normalize pronouns for identity consistency
        # This ensures "Your name is X" and "Assistant name is X" map to the same identity slot
        text_lower = text_lower.replace("your name", "assistant name")
        text_lower = text_lower.replace("my name", "user name")

        # Patterns for identifying unique "slots" in identity
        # We check these REGARDLESS of mem_type to catch "FACTS" that are actually identities
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

    def get(self, memory_id: int) -> Optional[Dict]:
        """Retrieve a specific memory by ID."""
        with self._connect() as con:
            row = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at, verified, flags, verification_attempts
                FROM memories
                WHERE id = ?
            """, (memory_id,)).fetchone()
        
        if not row:
            return None
            
        return {
            "id": row[0], "identity": row[1], "parent_id": row[2],
            "type": row[3], "subject": row[4], "text": row[5],
            "confidence": row[6], "source": row[7], "created_at": row[8],
            "verified": row[9], "flags": row[10], "verification_attempts": row[11]
        }

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
        embedding: Optional[np.ndarray] = None,
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
        embedding_json = json.dumps(embedding.tolist()) if embedding is not None else None

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
                    created_at,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                identity,
                parent_id,
                mem_type.upper(),
                subject,
                text.strip()[:1000],
                float(confidence),
                source,
                conflicts_json,
                timestamp,
                embedding_json,
            ))
            row_id = cur.lastrowid

            # Update FAISS if available
            if FAISS_AVAILABLE and embedding is not None:
                if self.faiss_index is None:
                    dimension = len(embedding)
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    self.memory_id_mapping = []

                emb_np = embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(emb_np)
                self.faiss_index.add(emb_np)
                self.memory_id_mapping.append(row_id)

            return row_id

    # --------------------------
    # Query helpers
    # --------------------------

    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about active memories."""
        stats = {}
        with self._connect() as con:
            # Count active goals
            row = con.execute("SELECT COUNT(*) FROM memories WHERE type = 'GOAL' AND parent_id IS NULL AND deleted = 0").fetchone()
            stats['active_goals'] = row[0] if row else 0
            
            # Count unverified beliefs
            row = con.execute("SELECT COUNT(*) FROM memories WHERE type = 'BELIEF' AND verified = 0 AND parent_id IS NULL AND deleted = 0 AND source = 'daydream' AND (text LIKE '%[Source:%' OR text LIKE '%[Supported by%')").fetchone()
            stats['unverified_beliefs'] = row[0] if row else 0
            
            # Count unverified facts
            row = con.execute("SELECT COUNT(*) FROM memories WHERE type = 'FACT' AND verified = 0 AND parent_id IS NULL AND deleted = 0 AND source = 'daydream' AND (text LIKE '%[Source:%' OR text LIKE '%[Supported by%')").fetchone()
            stats['unverified_facts'] = row[0] if row else 0
            
        return stats

    def list_recent(self, limit: Optional[int] = 30) -> List[Tuple[int, str, str, str, str, int]]:
        """
        Get recent memories, excluding old superseded versions.

        A memory is hidden if:
        1. It has a parent_id set (meaning it was superseded/consolidated).
        2. There's a newer memory with the EXACT same identity.
        3. There's a newer memory that explicitly points to it as a parent (supersedes it).
        """
        with self._connect() as con:
            query = """
                SELECT m.id, m.type, m.subject, m.text, m.source, m.verified, m.flags
                FROM memories m
                WHERE m.parent_id IS NULL
                AND m.deleted = 0
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE (newer.identity = m.identity OR newer.parent_id = m.id)
                    AND newer.created_at > m.created_at
                )
                ORDER BY m.created_at DESC
            """
            if limit is not None:
                query += " LIMIT ?"
                rows = con.execute(query, (limit,)).fetchall()
            else:
                rows = con.execute(query).fetchall()
        return rows

    def get_active_by_type(self, mem_type: str) -> List[Tuple[int, str, str]]:
        """Get all active memories of a specific type (subject, text)."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT m.id, m.subject, m.text
                FROM memories m
                WHERE m.type = ?
                AND m.parent_id IS NULL
                AND m.deleted = 0
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE (newer.identity = m.identity OR newer.parent_id = m.id)
                    AND newer.created_at > m.created_at
                )
            """, (mem_type.upper(),)).fetchall()
        return rows

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> List[Tuple[int, str, str, str, float]]:
        """
        Semantic search for memories using cosine similarity.
        Returns: List of (id, type, subject, text, similarity)
        """
        # 1. Fast Path: Use FAISS if available
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                q_emb = query_embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(q_emb)
                
                # Search more candidates than needed to account for filtered/inactive ones
                search_k = min(limit * 10, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(q_emb, search_k)
                
                candidate_ids = []
                candidate_scores = {}
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.memory_id_mapping):
                        mem_id = self.memory_id_mapping[idx]
                        candidate_ids.append(mem_id)
                        candidate_scores[mem_id] = float(scores[0][i])
                
                if not candidate_ids:
                    return []

                # Verify candidates are still active in DB
                placeholders = ','.join(['?'] * len(candidate_ids))
                with self._connect() as con:
                    rows = con.execute(f"""
                        SELECT m.id, m.type, m.subject, m.text
                        FROM memories m
                        WHERE m.id IN ({placeholders})
                        AND m.parent_id IS NULL
                        AND m.deleted = 0
                    """, candidate_ids).fetchall()
                
                results = []
                for r in rows:
                    mid = r[0]
                    if mid in candidate_scores:
                        results.append((mid, r[1], r[2], r[3], candidate_scores[mid]))
                
                results.sort(key=lambda x: x[4], reverse=True)
                return results[:limit]

            except Exception as e:
                print(f"⚠️ FAISS search failed, falling back to SQL: {e}")

        # 2. Slow Path: Linear Scan (Fallback)
        # Fetch all active memories that have embeddings
        with self._connect() as con:
            rows = con.execute("""
                SELECT m.id, m.type, m.subject, m.text, m.embedding
                FROM memories m
                WHERE m.parent_id IS NULL
                AND m.deleted = 0
                AND m.embedding IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE (newer.identity = m.identity OR newer.parent_id = m.id)
                    AND newer.created_at > m.created_at
                )
            """).fetchall()

        results = []
        # Calculate similarity in Python (efficient enough for < 10k memories)
        q_norm = np.linalg.norm(query_embedding)

        for r in rows:
            if not r[4]: continue
            mem_emb = np.array(json.loads(r[4]))
            # Check dimensions to prevent shape mismatch errors
            if query_embedding.shape != mem_emb.shape:
                continue
            m_norm = np.linalg.norm(mem_emb)
            if m_norm > 0 and q_norm > 0:
                sim = np.dot(query_embedding, mem_emb) / (q_norm * m_norm)
                results.append((r[0], r[1], r[2], r[3], float(sim)))

        results.sort(key=lambda x: x[4], reverse=True)
        return results[:limit]

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

    def delete_entry(self, memory_id: int) -> bool:
        """
        DANGEROUS: Delete a specific memory entry by ID.
        Used for removing hallucinations or corrupted data.
        """
        with self._connect() as con:
            cur = con.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            con.commit()
            return cur.rowcount > 0

    def soft_delete_entry(self, memory_id: int) -> bool:
        """
        Soft delete a memory entry by marking it as deleted.
        Preserves the record but hides it from active queries.
        """
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET deleted = 1 WHERE id = ?", (memory_id,))
            con.commit()
            return cur.rowcount > 0

    def set_flag(self, memory_id: int, flag: Optional[str]) -> bool:
        """Set or clear a flag on a memory entry."""
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET flags = ? WHERE id = ?", (flag, memory_id))
            con.commit()
            return cur.rowcount > 0

    def mark_verified(self, memory_id: int) -> None:
        """Mark a memory as verified against source."""
        with self._connect() as con:
            con.execute("UPDATE memories SET verified = 1 WHERE id = ?", (memory_id,))
            con.commit()

    def increment_verification_attempts(self, memory_id: int) -> int:
        """
        Increment the verification attempts counter for a memory.
        Returns the new count.
        """
        with self._connect() as con:
            con.execute("UPDATE memories SET verification_attempts = COALESCE(verification_attempts, 0) + 1 WHERE id = ?", (memory_id,))
            con.commit()
            row = con.execute("SELECT verification_attempts FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return row[0] if row else 0

    def get_comparison_similarity(self, id1: int, id2: int) -> Optional[float]:
        """Check if two memories have been compared before."""
        if id1 > id2:
            id1, id2 = id2, id1
        with self._connect() as con:
            row = con.execute(
                "SELECT similarity FROM consolidation_history WHERE id_a = ? AND id_b = ?",
                (id1, id2)
            ).fetchone()
        return row[0] if row else None

    def record_comparison(self, id1: int, id2: int, similarity: float) -> None:
        """Record that two memories have been compared."""
        if id1 > id2:
            id1, id2 = id2, id1
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO consolidation_history (id_a, id_b, similarity, created_at) VALUES (?, ?, ?, ?)",
                (id1, id2, float(similarity), int(time.time()))
            )
            con.commit()
