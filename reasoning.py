import os
import sqlite3
import time
import json
from typing import List, Optional, Dict, Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ReasoningStore:
    """
    Non-authoritative semantic reasoning store.

    - Uses embeddings (injected, not owned)
    - Allows fuzzy similarity
    - Stores hypotheses, interpretations, temporary conclusions
    - NEVER treated as truth
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
            return int(cur.lastrowid)

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
        q_emb = self.embed_fn(query)
        now = int(time.time())

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

        for r in rows:
            emb = np.array(json.loads(r[2]), dtype=float)
            sim = float(cosine_similarity([q_emb], [emb])[0][0])
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
            return cur.rowcount

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM reasoning_nodes")
