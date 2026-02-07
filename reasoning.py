import os
import sqlite3
import time
import json
from typing import List, Optional, Dict, Callable

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class ReasoningStore:
    """
    Non-authoritative semantic reasoning store.

    - Uses embeddings (injected, not owned)
    - Allows fuzzy similarity
    - Stores hypotheses, interpretations, temporary conclusions
    - NEVER treated as truth
    - Supports Time-To-Live (TTL) for transient thoughts
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        db_path: str = "./data/reasoning.sqlite3",
    ):
        self.embed_fn = embed_fn
        db_dir = os.path.dirname(db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = db_path
        self._init_db()
        
        self.faiss_index = None
        self.reasoning_id_mapping = []
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
                CREATE TABLE IF NOT EXISTS reasoning_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_reasoning_created "
                "ON reasoning_nodes(created_at);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_reasoning_expires "
                "ON reasoning_nodes(expires_at);"
            )

    def _build_faiss_index(self):
        """Build FAISS index from active reasoning nodes."""
        try:
            # Determine target dimension from current embedding function
            # This prevents crashes if the DB contains mixed-dimension vectors (e.g. after switching models)
            try:
                dummy_emb = self.embed_fn("test")
                target_dim = dummy_emb.shape[0]
            except:
                target_dim = None

            now = int(time.time())
            with self._connect() as con:
                rows = con.execute("""
                    SELECT id, embedding FROM reasoning_nodes 
                    WHERE (expires_at IS NULL OR expires_at > ?)
                    AND embedding IS NOT NULL
                """, (now,)).fetchall()
            
            embeddings = []
            ids = []
            
            for r in rows:
                if r[1]:
                    emb = np.array(json.loads(r[1]), dtype='float32')
                    
                    # Skip embeddings that don't match the current model's dimension
                    if target_dim and emb.shape[0] != target_dim:
                        continue
                    # Also ensure consistency within the list if target_dim failed
                    if embeddings and emb.shape != embeddings[0].shape:
                        continue
                    embeddings.append(emb)
                    ids.append(r[0])
            
            if embeddings:
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                embeddings_matrix = np.array(embeddings)
                faiss.normalize_L2(embeddings_matrix)
                self.faiss_index.add(embeddings_matrix)
                self.reasoning_id_mapping = ids
                # print(f"🧠 [Reasoning] FAISS index built with {len(ids)} active nodes.")
            else:
                self.faiss_index = None
                self.reasoning_id_mapping = []
                
        except Exception as e:
            print(f"⚠️ Failed to build FAISS index for reasoning: {e}")
            self.faiss_index = None

    # --------------------------
    # Add reasoning nodes
    # --------------------------
    def add(
        self,
        content: str,
        source: str = "inference",
        confidence: float = 0.5,
        ttl_seconds: Optional[int] = 3600,
    ) -> int:
        now = int(time.time())
        expires_at = now + ttl_seconds if ttl_seconds else None

        emb = self.embed_fn(content)
        emb_json = json.dumps(emb.astype(float).tolist())

        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO reasoning_nodes
                (content, embedding, source, confidence, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (content, emb_json, source, confidence, now, expires_at),
            )
            row_id = cur.lastrowid
            
            # Update FAISS
            if FAISS_AVAILABLE and self.faiss_index is not None:
                emb_np = emb.reshape(1, -1).astype('float32')
                faiss.normalize_L2(emb_np)
                self.faiss_index.add(emb_np)
                self.reasoning_id_mapping.append(row_id)
            elif FAISS_AVAILABLE and self.faiss_index is None:
                 self._build_faiss_index()

            return int(row_id)

    def list_recent(self, limit: int = 10) -> List[Dict]:
        """Get most recent reasoning nodes."""
        now = int(time.time())
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, content, source, confidence, created_at
                FROM reasoning_nodes
                WHERE expires_at IS NULL OR expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (now, limit)).fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "content": r[1],
                "source": r[2],
                "confidence": r[3],
                "created_at": r[4]
            })
        return results

    # --------------------------
    # Retrieve a reasoning node by ID
    # --------------------------
    def get(self, node_id: int) -> Optional[Dict]:
        with self._connect() as con:
            row = con.execute(
                "SELECT id, content, embedding, source, confidence, created_at, expires_at "
                "FROM reasoning_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "content": row[1],
                "embedding": np.array(json.loads(row[2]), dtype=float),
                "source": row[3],
                "confidence": row[4],
                "created_at": row[5],
                "expires_at": row[6],
            }

    # --------------------------
    # Semantic retrieval
    # --------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query or not query.strip():
            return []
            
        q_emb = self.embed_fn(query)
        now = int(time.time())

        # 1. Fast Path: FAISS
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                q_emb_np = q_emb.reshape(1, -1).astype('float32')
                faiss.normalize_L2(q_emb_np)
                
                search_k = min(top_k * 5, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(q_emb_np, search_k)
                
                candidate_ids = []
                candidate_scores = {}
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.reasoning_id_mapping):
                        rid = self.reasoning_id_mapping[idx]
                        candidate_ids.append(rid)
                        candidate_scores[rid] = float(scores[0][i])
                
                if not candidate_ids:
                    return []

                placeholders = ','.join(['?'] * len(candidate_ids))
                with self._connect() as con:
                    rows = con.execute(f"""
                        SELECT id, content, source, confidence
                        FROM reasoning_nodes
                        WHERE id IN ({placeholders})
                        AND (expires_at IS NULL OR expires_at > ?)
                    """, (*candidate_ids, now)).fetchall()
                
                results = []
                for r in rows:
                    rid = r[0]
                    if rid in candidate_scores:
                        results.append({
                            "id": rid,
                            "content": r[1],
                            "similarity": candidate_scores[rid],
                            "source": r[2],
                            "confidence": r[3],
                        })
                
                results.sort(key=lambda x: x["similarity"], reverse=True)
                return results[:top_k]
            except Exception as e:
                print(f"⚠️ FAISS search failed for reasoning: {e}")

        # 2. Slow Path: Linear Scan (Numpy)
        results = []
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT id, content, embedding, source, confidence
                FROM reasoning_nodes
                WHERE expires_at IS NULL OR expires_at > ?
                """,
                (now,),
            ).fetchall()

        q_norm = np.linalg.norm(q_emb)
        for r in rows:
            emb = np.array(json.loads(r[2]), dtype=float)
            # Check dimensions to prevent shape mismatch errors
            if q_emb.shape != emb.shape:
                continue
            # Cosine similarity
            dot = np.dot(q_emb, emb)
            norm = np.linalg.norm(emb)
            if norm > 0 and q_norm > 0:
                sim = float(dot / (q_norm * norm))
                results.append({
                    "id": r[0],
                    "content": r[1],
                    "similarity": sim,
                    "source": r[3],
                    "confidence": r[4],
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # --------------------------
    # Housekeeping
    # --------------------------
    def prune(self) -> int:
        now = int(time.time())
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM reasoning_nodes "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            count = cur.rowcount
        
        if count > 0 and FAISS_AVAILABLE:
            # Rebuild index to remove expired items
            self._build_faiss_index()
            
        return count

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM reasoning_nodes")
        
        self.faiss_index = None
        self.reasoning_id_mapping = []
