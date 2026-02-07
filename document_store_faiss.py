"""
FAISS-Enhanced Document Store

This version uses FAISS for fast vector similarity search
while maintaining SQLite for metadata storage.

Performance improvement: ~10-50x faster for 10,000+ chunks
"""

import os
import sqlite3
import time
import hashlib
import json
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-cpu")


class FaissDocumentStore:
    """
    FAISS-enhanced document store with fast vector search.
    
    Uses:
    - SQLite for metadata (filenames, page numbers, etc.)
    - FAISS for vector embeddings (fast similarity search)
    """

    def __init__(self, db_path: str = "./data/documents_faiss.sqlite3", embed_fn=None):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for this document store. Install with: pip install faiss-cpu")
            
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.faiss_index = None
        self.chunk_id_mapping = []  # Maps FAISS index to chunk IDs
        self._init_db()
        self._load_faiss_index()
        self._sync_faiss_index()

    # --------------------------
    # Database Initialization
    # --------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            # Documents table (metadata)
            con.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER NOT NULL,
                    upload_source TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)

            # Chunks table (text segments with metadata)
            con.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER,
                    created_at INTEGER NOT NULL,
                    embedding TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            con.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)")
            
            # Migration: Add embedding column if it doesn't exist
            try:
                con.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass

    # --------------------------
    # FAISS Integration
    # --------------------------

    def _load_faiss_index(self):
        """Load or create FAISS index"""
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        
        if os.path.exists(index_path):
            try:
                # Load existing index
                self.faiss_index = faiss.read_index(index_path)
                # Rebuild chunk ID mapping
                self._rebuild_chunk_mapping()
            except Exception as e:
                print(f"⚠️ FAISS index file is corrupted or incompatible: {e}")
                print("🔧 Deleting corrupted index and creating a fresh one...")
                try:
                    os.remove(index_path)
                except:
                    pass
                self._create_empty_index()
        else:
            self._create_empty_index()
            
    def _create_empty_index(self):
        """Helper to create a new empty index"""
        dimension = self._detect_embedding_dimension()
        print(f"🔧 Creating FAISS index with dimension: {dimension}")
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        self.chunk_id_mapping = []
            
    def _detect_embedding_dimension(self):
        """Detect the embedding dimension from the model"""
        try:
            # Try to get dimension from existing chunks
            with self._connect() as con:
                sample_chunk = con.execute("""
                    SELECT text FROM chunks LIMIT 1
                """).fetchone()
                
                if sample_chunk:
                    # Regenerate embedding for sample text
                    if self.embed_fn:
                        sample_embedding = self.embed_fn(sample_chunk[0])
                    else:
                        from lm import compute_embedding
                        sample_embedding = compute_embedding(sample_chunk[0])
                    return len(sample_embedding)
        except:
            pass
        
        # Fallback: create test embedding to detect dimension
        try:
            test_embedding = self.embed_fn("test") if self.embed_fn else np.random.rand(1536)
            return len(test_embedding)
        except Exception as e:
            print(f"⚠️ Could not detect embedding dimension: {e}")
            print("🔧 Using default dimension 1536")
            return 1536  # Default fallback
            
    def _rebuild_chunk_mapping(self):
        """Rebuild the mapping from FAISS index to chunk IDs"""
        with self._connect() as con:
            rows = con.execute("""
                SELECT id FROM chunks ORDER BY id
            """).fetchall()
            self.chunk_id_mapping = [row[0] for row in rows]

    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        faiss.write_index(self.faiss_index, index_path)

    def _add_embeddings_to_faiss(self, embeddings: List[np.ndarray], chunk_ids: List[int]):
        """Add embeddings to FAISS index"""
        if not embeddings:
            return
            
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize for inner product search
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        start_id = self.faiss_index.ntotal
        self.faiss_index.add(embeddings_array)
        
        # Update mapping
        for i, chunk_id in enumerate(chunk_ids):
            self.chunk_id_mapping.append(chunk_id)
        
        # Save index
        self._save_faiss_index()

    def _sync_faiss_index(self):
        """Ensure FAISS index is in sync with SQLite chunks."""
        try:
            with self._connect() as con:
                count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            
            # If DB has data but FAISS is empty or mismatched, rebuild
            if count > 0 and (self.faiss_index.ntotal == 0 or self.faiss_index.ntotal != count):
                print(f"⚠️ FAISS index out of sync (Index: {self.faiss_index.ntotal}, DB: {count}). Rebuilding...")
                self._rebuild_index()
        except Exception as e:
            print(f"⚠️ Error syncing FAISS index: {e}")

    def _rebuild_index(self):
        """Rebuild FAISS index from SQLite chunks (re-embedding if necessary)."""
        # Reset index first
        dimension = 1536 # Default, will be updated by first batch if possible
        self.faiss_index = None 
        self.chunk_id_mapping = []

        with self._connect() as con:
            cur = con.execute("SELECT id, text, embedding FROM chunks ORDER BY id")
        
            batch_size = 1000
            total_processed = 0

            print(f"🔄 Rebuilding index (batch size: {batch_size})...")
            
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                
                batch_embeddings = []
                batch_ids = []
                batch_updates = []

                for r in rows:
                    chunk_id = r[0]
                    text = r[1]
                    emb_json = r[2]
                    
                    emb = None
                    if emb_json:
                        try:
                            emb = np.array(json.loads(emb_json), dtype='float32')
                        except:
                            pass
                    
                    if emb is None:
                        # Re-compute if missing (Legacy data recovery)
                        if self.embed_fn:
                            try:
                                emb = self.embed_fn(text).astype('float32')
                                batch_updates.append((json.dumps(emb.tolist()), chunk_id))
                            except Exception as e:
                                print(f"❌ Failed to compute embedding for chunk {chunk_id}: {e}")
                                continue

                    if emb is not None:
                        batch_embeddings.append(emb)
                        batch_ids.append(chunk_id)
                
                # Initialize index on first batch if needed
                if self.faiss_index is None and batch_embeddings:
                    dimension = len(batch_embeddings[0])
                    self.faiss_index = faiss.IndexFlatIP(dimension)

                # Add batch to FAISS immediately to free memory
                if batch_embeddings and self.faiss_index:
                    self._add_embeddings_to_faiss(batch_embeddings, batch_ids)

                # Save recovered embeddings to DB immediately
                if batch_updates:
                    with self._connect() as update_con:
                        update_con.executemany("UPDATE chunks SET embedding = ? WHERE id = ?", batch_updates)
                        update_con.commit()

                total_processed += len(rows)
                # print(f"   Processed {total_processed} chunks...")
            
            print(f"✅ FAISS index rebuilt successfully ({total_processed} chunks).")

    # --------------------------
    # Document Management
    # --------------------------

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def document_exists(self, file_hash: str) -> bool:
        """Check if document already exists in database."""
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM documents WHERE file_hash = ? LIMIT 1",
                (file_hash,)
            ).fetchone()
        return row is not None

    def add_document(
        self,
        file_hash: str,
        filename: str,
        file_type: str,
        file_size: int,
        page_count: Optional[int],
        chunks: List[Dict],  # [{'text': str, 'embedding': np.ndarray, 'page_number': int}, ...]
        upload_source: str = "telegram"
    ) -> int:
        """
        Add document and its chunks to database with FAISS indexing.
        """
        timestamp = int(time.time())
        chunk_embeddings = []
        chunk_ids = []

        with self._connect() as con:
            # Insert document metadata
            cur = con.execute("""
                INSERT INTO documents (
                    file_hash, filename, file_type, file_size, 
                    page_count, chunk_count, upload_source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_hash,
                filename,
                file_type,
                file_size,
                page_count,
                len(chunks),
                upload_source,
                timestamp
            ))
            document_id = cur.lastrowid

            # Insert chunks and collect embeddings
            for idx, chunk in enumerate(chunks):
                con.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, text,
                        page_number, created_at, embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    idx,
                    chunk['text'],
                    chunk.get('page_number'),
                    timestamp,
                    json.dumps(chunk['embedding'].tolist())
                ))
                
                # Store embedding for FAISS
                chunk_embeddings.append(chunk['embedding'])
                # Get the chunk ID (we'll need to query it after commit)
                chunk_ids.append(None)  # Will be filled after commit

            con.commit()

            # Get actual chunk IDs
            chunk_rows = con.execute("""
                SELECT id FROM chunks 
                WHERE document_id = ? 
                ORDER BY chunk_index
            """, (document_id,)).fetchall()
            
            chunk_ids = [row[0] for row in chunk_rows]

        # Add embeddings to FAISS
        self._add_embeddings_to_faiss(chunk_embeddings, chunk_ids)
        
        # Save FAISS index immediately
        self._save_faiss_index()
        
        return document_id

    # --------------------------
    # Fast Semantic Search
    # --------------------------

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Fast semantic search using FAISS.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            document_id: Optionally filter by specific document
        
        Returns:
            List of chunks with similarity scores
        """
        # OPTIMIZATION: Local Search (Document-Specific)
        # If we know the document, search ONLY its chunks via SQLite + Numpy
        # This avoids the issue where relevant chunks are pushed out of the top-k by other documents
        if document_id is not None:
            with self._connect() as con:
                # Fetch all chunks for this document that have embeddings
                rows = con.execute("""
                    SELECT id, chunk_index, text, page_number, embedding 
                    FROM chunks 
                    WHERE document_id = ? AND embedding IS NOT NULL
                """, (document_id,)).fetchall()
            
            if not rows:
                return []

            results = []
            q_norm = np.linalg.norm(query_embedding)
            
            for r in rows:
                emb = np.array(json.loads(r[4]), dtype='float32')
                emb_norm = np.linalg.norm(emb)
                if q_norm > 0 and emb_norm > 0:
                    sim = np.dot(query_embedding, emb) / (q_norm * emb_norm)
                    results.append({
                        'chunk_id': r[0],
                        'document_id': document_id,
                        'chunk_index': r[1],
                        'text': r[2],
                        'page_number': r[3],
                        'filename': '', # Caller knows filename
                        'similarity': float(sim)
                    })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]

        # STANDARD: Global Search via FAISS
        if self.faiss_index.ntotal == 0:
            return []

        # Normalize query embedding
        query_array = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_array)

        # Search using FAISS
        scores, indices = self.faiss_index.search(query_array, min(top_k * 2, self.faiss_index.ntotal))
        
        # Get chunk IDs from mapping
        results = []
        valid_results = 0
        
        with self._connect() as con:
            for i in range(len(indices[0])):
                if valid_results >= top_k:
                    break
                    
                faiss_idx = indices[0][i]
                if faiss_idx >= len(self.chunk_id_mapping):
                    continue
                    
                chunk_id = self.chunk_id_mapping[faiss_idx]
                score = float(scores[0][i])
                
                # Get chunk details
                row = con.execute("""
                    SELECT c.chunk_index, c.text, c.page_number, d.filename, d.id
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id = ?
                """, (chunk_id,)).fetchone()
                
                if not row:
                    continue
                    
                # Apply document filter if specified
                if document_id and row[4] != document_id:
                    continue
                
                results.append({
                    'chunk_id': chunk_id,
                    'document_id': row[4],
                    'chunk_index': row[0],
                    'text': row[1],
                    'page_number': row[2],
                    'filename': row[3],
                    'similarity': score
                })
                valid_results += 1

        return results

    # --------------------------
    # Document Queries
    # --------------------------

    def search_filenames(self, query: str, limit: int = 5) -> List[str]:
        """Search for filenames containing query terms."""
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return []
            
        with self._connect() as con:
            rows = con.execute("SELECT filename FROM documents").fetchall()
            
        matches = []
        for (filename,) in rows:
            fn_lower = filename.lower()
            score = sum(1 for t in terms if t in fn_lower)
            if score > 0:
                matches.append((score, filename))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    def get_document_by_filename(self, filename: str) -> Optional[int]:
        """Get document ID by exact filename."""
        with self._connect() as con:
            row = con.execute("SELECT id FROM documents WHERE filename = ?", (filename,)).fetchone()
        return row[0] if row else None

    def list_documents(self, limit: int = 50) -> List[Tuple]:
        """List all documents."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, filename, file_type, page_count, chunk_count, created_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows

    def get_document_chunks(self, document_id: int, include_embeddings: bool = False) -> List[Dict]:
        """Get all chunks for a specific document."""
        query = "SELECT chunk_index, text, page_number"
        if include_embeddings:
            query += ", embedding"

        query += " FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC"

        with self._connect() as con:
            rows = con.execute(query, (document_id,)).fetchall()

        results = []
        for r in rows:
            item = {
                'chunk_index': r[0],
                'text': r[1],
                'page_number': r[2]
            }
            if include_embeddings and r[3]:
                try:
                    item['embedding'] = np.array(json.loads(r[3]), dtype='float32')
                except:
                    pass
            results.append(item)

        return results

    def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks (and remove from FAISS)."""
        with self._connect() as con:
            # Check if document exists
            exists = con.execute(
                "SELECT 1 FROM documents WHERE id = ?",
                (document_id,)
            ).fetchone()

            if not exists:
                return False

            # Get chunk IDs to remove from FAISS
            chunk_rows = con.execute("""
                SELECT id FROM chunks WHERE document_id = ?
            """, (document_id,)).fetchall()
            
            chunk_ids_to_remove = [row[0] for row in chunk_rows]

            # Delete from SQLite
            con.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            con.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            con.commit()

        # Remove from FAISS index
        self._remove_chunks_from_faiss(chunk_ids_to_remove)
        
        return True

    def find_broken_documents(self) -> List[Dict]:
        """Find documents with integrity issues (0 chunks, count mismatch, missing embeddings)."""
        broken = []
        with self._connect() as con:
            # Check 1: Metadata mismatch or empty chunks
            rows = con.execute("""
                SELECT d.id, d.filename, d.chunk_count, COUNT(c.id) as actual_chunks
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id
                HAVING d.chunk_count != actual_chunks OR d.chunk_count = 0
            """).fetchall()
            
            for r in rows:
                issue = "No chunks found" if r[2] == 0 else f"Chunk count mismatch (Meta: {r[2]}, Actual: {r[3]})"
                broken.append({'id': r[0], 'filename': r[1], 'issue': issue})

            # Check 2: Missing embeddings
            rows_emb = con.execute("""
                SELECT d.id, d.filename, COUNT(c.id)
                FROM documents d
                JOIN chunks c ON d.id = c.document_id
                WHERE c.embedding IS NULL
                GROUP BY d.id
            """).fetchall()

            for r in rows_emb:
                if not any(b['id'] == r[0] for b in broken):
                    broken.append({'id': r[0], 'filename': r[1], 'issue': f"Missing embeddings for {r[2]} chunks"})
        
        return broken

    def _remove_chunks_from_faiss(self, chunk_ids: List[int]):
        """Remove specific chunks from FAISS index."""
        # Rebuild index from DB (which now excludes the deleted document)
        self._rebuild_index()

    # --------------------------
    # Utilities
    # --------------------------

    def get_total_documents(self) -> int:
        """Get total number of documents."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def get_total_chunks(self) -> int:
        """Get total number of chunks across all documents."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def get_search_stats(self) -> Dict:
        """Get FAISS search statistics."""
        return {
            'total_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'dimension': self.faiss_index.d if self.faiss_index else 0,
            'index_type': str(type(self.faiss_index).__name__) if self.faiss_index else 'None'
        }

    def clear(self):
        """DANGEROUS: Clear all documents and chunks."""
        with self._connect() as con:
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM documents")
            con.commit()
        
        # Clear FAISS index
        dimension = 1536  # Adjust as needed
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.chunk_id_mapping = []
        self._save_faiss_index()
