"""
Memory Consolidator: Smart consolidation WITHOUT data loss.

Strategy:
- Group memories by identity pattern
- IDENTITY type: Version chaining (old → new, new supersedes)
  Example: "AI Companion" → "Ada" (clear version update)

- Other types: Smart linking (KEEP BOTH memories)
  If similarity ≥ threshold: Link them via parent_id
  But NEVER delete: Append-only, no data loss
  Goal: Prevent duplication while keeping all unique information

Result: Compressed database with zero data loss, full audit trail via parent_id chain

Meta-Memory: Automatically creates meta-memories when consolidation happens
- Tracks what changed, when, and why
- Stored in separate meta_memory database
- Enables self-reflection and temporal reasoning
"""

import time
import json
from typing import List, Dict, Optional, Callable
from datetime import datetime
import re
import random

from memory import MemoryStore
from meta_memory import MetaMemoryStore
from lm import compute_embedding, run_local_lm
import numpy as np


class MemoryConsolidator:
    """Consolidates similar/duplicate memories to keep memory store clean."""

    def __init__(self, memory_store: MemoryStore, meta_memory_store: Optional[MetaMemoryStore] = None, document_store=None, 
                 consolidation_thresholds: Optional[Dict[str, float]] = None,
                 max_inconclusive_attempts: int = 8,
                 max_retrieval_failures: int = 8,
                 log_fn: Callable[[str], None] = print):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.document_store = document_store
        self.log = log_fn
        
        self.consolidation_thresholds = consolidation_thresholds or {
            "GOAL": 0.88,
            "IDENTITY": 0.87,
            "BELIEF": 0.87,
            "PERMISSION": 0.87,
            "FACT": 0.93,
            "PREFERENCE": 0.93,
            "RULE": 0.93
        }
        self.max_inconclusive_attempts = max_inconclusive_attempts
        self.max_retrieval_failures = max_retrieval_failures

    def consolidate(self, time_window_hours: Optional[int] = None) -> Dict[str, int]:
        """
        Smart consolidation: Detect & link duplicates WITHOUT data loss.

        Process:
        1. Group memories by identity pattern (For forced replacements like Name/Location)
        2. Consolidate across the whole set for semantic overlaps (e.g. "loves coffee" vs "prefers coffee")
        """
        stats = {'processed': 0, 'consolidated': 0, 'skipped': 0}

        # Get memories based on time window
        if time_window_hours is not None:
            cutoff_time = int(time.time()) - (time_window_hours * 3600)
            recent_memories = self._get_memories_after(cutoff_time)
        else:
            recent_memories = self._get_all_memories()

        if not recent_memories:
            self.log("🧠 [Consolidator] No memories to consolidate")
            return stats

        # Sort all memories chronologically
        recent_memories = sorted(recent_memories, key=lambda m: m['created_at'])

        # --- Phase 1: Identity-based Forced Replacements (Name, Location etc) ---
        # Recompute identity hashes for IDENTITY types to catch legacy data with old patterns
        groups_by_identity = {}
        for mem in recent_memories:
            # For IDENTITY types, recompute the identity hash using current patterns
            if mem['type'].upper() == 'IDENTITY':
                identity = self.memory_store.compute_identity(mem['text'], mem['type'])
            else:
                identity = mem['identity']
            
            if identity not in groups_by_identity:
                groups_by_identity[identity] = []
            groups_by_identity[identity].append(mem)

        identity_consolidated_ids = set()
        for identity, group in groups_by_identity.items():
            if len(group) > 1 and group[-1]['type'].upper() == 'IDENTITY':
                for i in range(len(group) - 1):
                    old, new = group[i], group[i+1]
                    if old['subject'] == new['subject'] and old['type'] == new['type']:
                        if self._mark_consolidated(old['id'], new['id']):
                            stats['consolidated'] += 1
                            identity_consolidated_ids.add(old['id'])
                            if self.meta_memory_store:
                                self._create_meta_memory(old, new, "VERSION_UPDATE")

        # --- Phase 2: Universal Semantic Overlap (Refinements/Duplicates) ---
        groups_by_type = {}
        for mem in recent_memories:
            if mem['id'] in identity_consolidated_ids: continue
            if mem['type'] == 'NOTE': continue  # Skip Assistant Notes
            key = (mem['subject'], mem['type'])
            if key not in groups_by_type: groups_by_type[key] = []
            groups_by_type[key].append(mem)

        for (subj, mtype), group in groups_by_type.items():
            stats['processed'] += 1
            if len(group) < 2: continue

            # Prepare data for vectorized calculation
            valid_items = []
            embeddings = []
            
            for mem in group:
                # Skip if already consolidated or if it already has a parent
                if mem['id'] in identity_consolidated_ids or mem['parent_id'] is not None:
                    continue
                
                emb = mem.get('embedding')
                if emb is None:
                    emb = compute_embedding(mem['text'])
                    # Self-healing: Save this embedding so we don't compute it again
                    self._update_embedding(mem['id'], emb)
                
                if emb is not None:
                    valid_items.append(mem)
                    embeddings.append(np.array(emb))
            
            if len(embeddings) < 2:
                continue

            # Vectorized Cosine Similarity: Sim = (A . B.T) / (|A| * |B|)
            # 1. Stack embeddings into matrix (N x D)
            matrix = np.stack(embeddings)
            
            # 2. Normalize rows (L2 norm)
            norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix_normalized = matrix / (norm + 1e-10) # Avoid div by zero
            
            # 3. Compute Similarity Matrix (N x N)
            sim_matrix = np.dot(matrix_normalized, matrix_normalized.T)
            
            # 4. Iterate upper triangle
            count = len(valid_items)
            for r in range(count):
                old = valid_items[r]
                if old['id'] in identity_consolidated_ids: continue

                # Determine threshold
                is_identity_like = "name is" in old['text'].lower() or "lives in" in old['text'].lower()
                lookup_type = old['type']
                if is_identity_like:
                    lookup_type = 'IDENTITY'
                threshold = self.consolidation_thresholds.get(lookup_type, 0.93)

                for c in range(r + 1, count):
                    new = valid_items[c]
                    if new['id'] in identity_consolidated_ids: continue

                    similarity = float(sim_matrix[r, c])

                    # Check cache for previous rejections to avoid re-logging
                    # (Optional optimization, but matrix calc is fast enough now)
                    # We still check it to respect historical "distinct" decisions if logic changes
                    cached_sim = self.memory_store.get_comparison_similarity(old['id'], new['id'])
                    if cached_sim is not None and cached_sim < threshold and similarity < threshold:
                        continue

                    # Special case: IDENTITY substring containment
                    # If one identity is a substring expansion of another, consolidate
                    is_substring_expansion = False
                    if old['type'] == 'IDENTITY' and old['subject'] == new['subject']:
                        # Extract the actual values (after "User is", "Assistant is", etc.)
                        old_value = self._extract_value_from_text(old['text'])
                        new_value = self._extract_value_from_text(new['text'])
                        
                        old_normalized = old_value.lower().strip()
                        new_normalized = new_value.lower().strip()
                        
                        # Check if one is contained in the other (but not identical)
                        if old_normalized != new_normalized:
                            if old_normalized in new_normalized or new_normalized in old_normalized:
                                is_substring_expansion = True
                                self.log(f"      🔍 Substring match: '{old_value[:50]}' ⊂ '{new_value[:50]}'")
                    
                    # Also check BELIEF substring (e.g., "believes in Islam" vs "believes in Islam and...")
                    if old['type'] == 'BELIEF' and old['subject'] == new['subject']:
                        old_value = self._extract_value_from_text(old['text'])
                        new_value = self._extract_value_from_text(new['text'])
                        
                        old_normalized = old_value.lower().strip()
                        new_normalized = new_value.lower().strip()
                        
                        if old_normalized != new_normalized:
                            if old_normalized in new_normalized or new_normalized in old_normalized:
                                is_substring_expansion = True
                                self.log(f"      🔍 Substring match: '{old_value[:50]}' ⊂ '{new_value[:50]}'")
                    
                    # Debug output
                    if old['type'] in ('IDENTITY', 'BELIEF', 'PERMISSION') and similarity > 0.85:
                        self.log(f"      🔍 Comparing {old['type']}: similarity={similarity:.3f}, threshold={threshold:.3f}, substring={is_substring_expansion}")
                        self.log(f"         OLD: {old['text'][:60]}")
                        self.log(f"         NEW: {new['text'][:60]}")
                    
                    if similarity >= threshold or is_substring_expansion:
                        if self._mark_consolidated(old['id'], new['id']):
                            stats['consolidated'] += 1
                            identity_consolidated_ids.add(old['id'])
                            if self.meta_memory_store:
                                self._create_meta_memory(old, new, "SIMILARITY_LINK")
                            break
                    else:
                        # Record comparison to avoid re-checking next time
                        # Only record if similarity is high enough (> 0.8) to be worth caching/logging
                        if similarity > 0.8:
                            self.memory_store.record_comparison(old['id'], new['id'], similarity)
        return stats

    def verify_sources(self, batch_size: int = 5, stop_check_fn: Optional[Callable[[], bool]] = None) -> int:
        """
        Verify memories against their cited sources.
        Removes memories that are not supported by the source document.
        Also removes memories that reference documents without citation.
        """
        if not self.document_store:
            return 0
            
        # Get all active memories
        all_memories = self._get_all_memories()
        
        removed_count = 0

        # 1. Cleanup: Remove memories that refer to documents but lack citation
        # Includes BELIEF here because uncited beliefs about documents are also invalid
        cleanup_types = {"GOAL", "FACT", "BELIEF"}
        uncited_candidates = [
            m for m in all_memories
            if "[Source:" not in m['text'] and "[Supported by" not in m['text'] and m['type'] in cleanup_types and m.get('source') == 'daydream'
        ]
        
        triggers = ["the document", "this document", "the study", "this study", "the research", "this research", 
                    "the pdf", "this pdf", "the findings", "these findings", "the article", "this article", "the text"]
        
        for mem in uncited_candidates:
            if stop_check_fn and stop_check_fn():
                break
                
            lower_text = mem['text'].lower()
            if any(t in lower_text for t in triggers):
                self.log(f"❌ [Verifier] Removing uncited document reference ID {mem['id']}: '{mem['text'][:50]}...'")
                if self.memory_store.delete_entry(mem['id']):
                    removed_count += 1
                    if self.meta_memory_store:
                        self.meta_memory_store.add_meta_memory("CORRECTION", mem['type'], mem['subject'], f"Deleted uncited document reference: '{mem['text']}'", old_id=mem['id'], metadata={'reason': 'uncited_document_reference'})

        # 2. Verify cited memories
        # Filter for those with a source citation AND allowed types
        # Including BELIEF to ensure groundedness (beliefs with sources must be supported by them)
        target_types = {"GOAL", "FACT", "BELIEF"}
        candidates = [
            m for m in all_memories 
            if ("[Source:" in m['text'] or "[Supported by" in m['text']) and m['type'] in target_types and not m.get('verified') and m.get('source') == 'daydream'
        ]
        
        if not candidates and removed_count == 0:
            return removed_count
            
        # Pick a random batch to verify (spreads load over time)
        batch = random.sample(candidates, min(len(candidates), batch_size))
        
        self.log(f"🧹 [Verifier] Found {len(candidates)} unverified candidates. Checking {len(batch)} of them...")

        # Optimization: Sort by source to maximize cache hits
        batch.sort(key=lambda m: re.search(r"\[(?:Source|Supported by): (.*?)\]", m['text']).group(1) if re.search(r"\[(?:Source|Supported by): (.*?)\]", m['text']) else "")
        
        current_doc_cache = {'filename': None, 'id': None, 'chunk_map': None}

        for mem in batch:
            if stop_check_fn and stop_check_fn():
                self.log("🛑 [Verifier] Verification stopped by user.")
                break

            # Double-check verification status (in case of race conditions)
            with self.memory_store._connect() as con:
                row = con.execute("SELECT verified FROM memories WHERE id = ?", (mem['id'],)).fetchone()
                if row and row[0] == 1:
                    self.log(f"⏩ [Verifier] Memory {mem['id']} already verified. Skipping.")
                    continue

            # Extract filename
            match = re.search(r"\[(?:Source|Supported by): (.*?)\]", mem['text'])
            if not match:
                continue
                
            filename = match.group(1)
            
            self.log(f"🔍 [Verifier] Checking Memory {mem['id']} against source '{filename}'...")
            
            # Optimization: Cache document chunks to avoid repeated DB fetches
            if current_doc_cache['filename'] != filename:
                doc_id = self.document_store.get_document_by_filename(filename)
                if not doc_id:
                    current_doc_cache = {'filename': None, 'id': None, 'chunk_map': None}
                    continue
                
                # Fetch chunks for context reconstruction
                all_chunks = self.document_store.get_document_chunks(doc_id, include_embeddings=False)
                chunk_map = {c['chunk_index']: c['text'] for c in all_chunks}

                current_doc_cache = {
                    'filename': filename,
                    'id': doc_id,
                    'chunk_map': chunk_map
                }
            
            if not current_doc_cache['id']:
                continue
                
            # Clean text for search (remove source tag)
            clean_text = mem['text'].replace(match.group(0), "").strip()
            if not clean_text:
                continue
                
            # Search chunks in that document using the memory's content
            query_emb = compute_embedding(clean_text)

            # Use optimized local search in document store
            search_results = self.document_store.search_chunks(query_emb, top_k=5, document_id=current_doc_cache['id'])
            
            # Fallback: Cross-Lingual / Keyword Search Generation
            # If results are poor (e.g. top similarity < 0.45), try to generate a better query
            # This handles the English Memory -> Turkish Doc gap
            if not search_results or (search_results and search_results[0]['similarity'] < 0.45):
                self.log(f"⚠️ [Verifier] Low similarity ({search_results[0]['similarity'] if search_results else 0:.2f}) for direct search. Attempting cross-lingual query generation...")
                
                gen_prompt = (
                    f"I need to find evidence for this claim in a document named '{filename}'.\n"
                    f"Claim: {clean_text}\n"
                    "The document might be in a different language (e.g., Turkish, Spanish) or use different terminology.\n"
                    "Generate a search query (keywords or translated sentence) that is most likely to match the raw text in the document.\n"
                    "Output ONLY the search query text."
                )
                
                better_query = run_local_lm([{"role": "user", "content": gen_prompt}], temperature=0.1, max_tokens=100).strip()
                
                if better_query and better_query != clean_text:
                    self.log(f"🔍 [Verifier] Retrying search with generated query: '{better_query}'")
                    query_emb_2 = compute_embedding(better_query)
                    search_results_2 = self.document_store.search_chunks(query_emb_2, top_k=5, document_id=current_doc_cache['id'])
                    
                    if search_results_2:
                        search_results = search_results_2 # Prefer the generated query results

            if not search_results:
                self.log(f"⚠️ [Verifier] No relevant text chunks found in '{filename}' for Memory {mem['id']}.")
                
                # Increment attempts and check limit
                attempts = self.memory_store.increment_verification_attempts(mem['id'])
                if attempts >= self.max_retrieval_failures:
                    self.log(f"❌ [Verifier] Memory {mem['id']} deleted after {self.max_retrieval_failures} failed retrieval attempts.")
                    if self.memory_store.delete_entry(mem['id']):
                        removed_count += 1
                        if self.meta_memory_store:
                            self.meta_memory_store.add_meta_memory("CORRECTION", mem['type'], mem['subject'], f"Deleted memory after {self.max_retrieval_failures} failed retrievals: '{clean_text}'", old_id=mem['id'], metadata={'reason': 'retrieval_failure_limit_reached', 'attempts': attempts})
                continue

            # Context Expansion: Get surrounding chunks to "read" the document section
            chunk_map = current_doc_cache['chunk_map']
            
            relevant_indices = set()
            MAX_CHARS = 4000  # Reduced limit to avoid 400 Bad Request
            current_chars = 0

            for res in search_results:
                # Stop adding chunks if we exceed the limit
                if current_chars >= MAX_CHARS:
                    break

                idx = res['chunk_index']
                # Add window of +/- 1 chunk to provide context
                for i in range(idx - 1, idx + 2):
                    if i in chunk_map:
                        if i not in relevant_indices:
                            # Check length before adding
                            chunk_len = len(chunk_map[i])
                            if current_chars + chunk_len < MAX_CHARS:
                                relevant_indices.add(i)
                                current_chars += chunk_len
            
            sorted_indices = sorted(list(relevant_indices))
            
            # Reconstruct text flow with separators for non-contiguous sections
            context_parts = []
            last_idx = -999
            for idx in sorted_indices:
                if idx > last_idx + 1:
                    context_parts.append("\n... [Skipped sections] ...\n")
                context_parts.append(f"[Section {idx}] {chunk_map[idx]}")
                last_idx = idx
                
            context_text = "\n".join(context_parts)
            
            # Verify with LLM
            prompt = (
                "You are a fact-checker verifying if a memory is supported by a source document.\n"
                "Note: The Document Excerpts and the Memory Claim might be in DIFFERENT LANGUAGES.\n"
                "Verify based on meaning, not just keyword matching.\n\n"
                f"Excerpts from '{filename}':\n{context_text}\n\n"
                f"Memory Claim: {clean_text}\n\n"
                "Task: Analyze if the Memory Claim is supported by the text.\n"
                "1. Briefly analyze the relationship between the text and the claim.\n"
                "2. Allow for reasonable inference, synthesis, and summarization. It does not need to be verbatim.\n"
                "3. Only reject if the claim clearly contradicts the text or is completely unrelated/hallucinated.\n\n"
                "Output format:\n"
                "Reasoning: <short analysis>\n"
                "Verdict: VALID or INVALID"
            )
            
            response = run_local_lm([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=150)
            
            # Robust parsing: look for keywords even if formatting is messy
            if "Verdict: INVALID" in response or "Verdict:INVALID" in response:
                self.log(f"❌ [Verifier] Memory {mem['id']} rejected. Claim: '{clean_text[:50]}...' not found in '{filename}'")
                if self.memory_store.delete_entry(mem['id']):
                    removed_count += 1
                    if self.meta_memory_store:
                        self.meta_memory_store.add_meta_memory("CORRECTION", mem['type'], mem['subject'], f"Deleted hallucinated memory about {filename}: '{clean_text}'", old_id=mem['id'], metadata={'reason': 'invalid_verdict', 'llm_response': response})
            elif "Verdict: VALID" in response or "Verdict:VALID" in response:
                self.log(f"✅ [Verifier] Memory {mem['id']} confirmed VALID.")
                self.memory_store.mark_verified(mem['id'])
            else:
                self.log(f"❓ [Verifier] Inconclusive verdict for Memory {mem['id']}. Response: {response[:50]}...")
                
                # Increment attempts and check limit
                attempts = self.memory_store.increment_verification_attempts(mem['id'])
                if attempts >= self.max_inconclusive_attempts:
                    self.log(f"❌ [Verifier] Memory {mem['id']} deleted after {self.max_inconclusive_attempts} inconclusive verification attempts.")
                    if self.memory_store.delete_entry(mem['id']):
                        removed_count += 1
                        if self.meta_memory_store:
                            self.meta_memory_store.add_meta_memory("CORRECTION", mem['type'], mem['subject'], f"Deleted memory after {self.max_inconclusive_attempts} inconclusive verifications: '{clean_text}'", old_id=mem['id'], metadata={'reason': 'inconclusive_limit_reached', 'attempts': attempts})

        return removed_count

    def get_unverified_count(self) -> int:
        """Get number of unverified memories that require verification."""
        target_types = ("GOAL", "FACT", "BELIEF")
        placeholders = ','.join(['?'] * len(target_types))
        with self.memory_store._connect() as con:
            row = con.execute(f"""
                SELECT COUNT(*) FROM memories 
                WHERE verified = 0 
                AND (text LIKE '%[Source:%' OR text LIKE '%[Supported by%')
                AND type IN ({placeholders})
                AND parent_id IS NULL
                AND source = 'daydream'
            """, target_types).fetchone()
        return row[0] if row else 0

    def _consolidate_group(self, group: List[Dict]) -> int:
        """DEPRECATED: Logic moved to main consolidate function."""
        return 0

    def _get_memories_after(self, cutoff_time: int) -> List[Dict]:
        """Get all memories created after cutoff_time."""
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at, embedding, verified
                FROM memories
                WHERE created_at > ?
                ORDER BY created_at DESC
            """, (cutoff_time,)).fetchall()

        memories = []
        for r in rows:
            memories.append({
                'id': r[0],
                'identity': r[1],
                'parent_id': r[2],
                'type': r[3],
                'subject': r[4],
                'text': r[5],
                'confidence': r[6],
                'source': r[7],
                'created_at': r[8],
                'embedding': json.loads(r[9]) if r[9] else None,
                'verified': r[10] if len(r) > 10 else 0,
            })
        return memories

    def _get_all_memories(self) -> List[Dict]:
        """Get ALL memories regardless of age."""
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at, embedding, verified
                FROM memories
                ORDER BY created_at DESC
            """).fetchall()

        memories = []
        for r in rows:
            memories.append({
                'id': r[0],
                'identity': r[1],
                'parent_id': r[2],
                'type': r[3],
                'subject': r[4],
                'text': r[5],
                'confidence': r[6],
                'source': r[7],
                'created_at': r[8],
                'embedding': json.loads(r[9]) if r[9] else None,
                'verified': r[10] if len(r) > 10 else 0,
            })
        return memories

    def _mark_consolidated(self, old_id: int, new_id: int) -> bool:
        """
        Mark old_id as consolidated into new_id.

        Adds a "consolidated_into" relationship without deleting.
        (Append-only: add a record instead of modifying)

        Returns:
            True if consolidation was performed, False if already consolidated
        """
        # For now: just add a parent_id relationship if not already set
        # In future: could add a consolidation_log table
        with self.memory_store._connect() as con:
            # Check if already has parent
            existing = con.execute(
                "SELECT parent_id FROM memories WHERE id = ?",
                (old_id,)
            ).fetchone()

            # Update parent_id to point to the latest version
            # This creates a version chain: old -> new
            if existing:
                current_parent = existing[0]
                # Only update if not already pointing to this new_id
                if current_parent != new_id:
                    con.execute(
                        "UPDATE memories SET parent_id = ? WHERE id = ?",
                        (new_id, old_id)
                    )
                    con.commit()
                    self.log(f"      🔗 Updated parent_id: ID {old_id} now points to ID {new_id}")
                    return True
                else:
                    # Already consolidated
                    return False
            return False

    def _update_embedding(self, memory_id: int, embedding: np.ndarray) -> None:
        """Update the embedding for a specific memory ID."""
        try:
            embedding_json = json.dumps(embedding.tolist())
            with self.memory_store._connect() as con:
                con.execute("UPDATE memories SET embedding = ? WHERE id = ?", (embedding_json, memory_id))
                con.commit()
        except Exception as e:
            self.log(f"⚠️ Failed to update embedding for memory {memory_id}: {e}")

    def _create_meta_memory(self, old_mem: Dict, new_mem: Dict, event_type: str) -> None:
        """
        Create a meta-memory about a memory change.

        This enables self-reflection and temporal reasoning.

        Examples:
        - VERSION_UPDATE (IDENTITY): "Assistant name changed from Ada to Lara on 2026-02-04 at 11:37"
        - SIMILARITY_LINK (PREFERENCE): "User added similar preference: 'loves tea' (related to 'loves coffee') on 2026-02-04 14:30"
        - SIMILARITY_LINK (GOAL): "User added similar goal: 'learn Rust' (related to 'learn Python') on 2026-02-04 15:00"

        Args:
            old_mem: The old memory version
            new_mem: The new memory version
            event_type: Type of event (VERSION_UPDATE, SIMILARITY_LINK, CONFLICT_DETECTED, etc.)
        """
        # Extract the actual values from the memory text
        old_value = self._extract_value_from_text(old_mem['text'])
        new_value = self._extract_value_from_text(new_mem['text'])

        # Skip redundant meta-memories where value hasn't actually changed
        # (e.g., during pointer-only updates or deduplication)
        if old_value == new_value and event_type in ("VERSION_UPDATE", "SIMILARITY_LINK"):
            return

        # Format timestamp
        timestamp = datetime.fromtimestamp(new_mem['created_at']).strftime("%Y-%m-%d %H:%M")

        # Create human-readable meta-memory text based on event type
        if event_type == "VERSION_UPDATE":
            # VERSION_UPDATE: True replacement (IDENTITY types)
            if old_mem['type'] == 'IDENTITY':
                # For identity changes, be specific about what changed
                if 'name is' in old_mem['text'].lower():
                    meta_text = f"{old_mem['subject']} name changed from {old_value} to {new_value} on {timestamp}"
                elif 'lives in' in old_mem['text'].lower():
                    meta_text = f"{old_mem['subject']} location changed from {old_value} to {new_value} on {timestamp}"
                else:
                    meta_text = f"{old_mem['subject']} {old_mem['type'].lower()} updated from '{old_value}' to '{new_value}' on {timestamp}"
            else:
                meta_text = f"{old_mem['subject']} {old_mem['type'].lower()} updated from '{old_value}' to '{new_value}' on {timestamp}"

        elif event_type == "SIMILARITY_LINK":
            # SIMILARITY_LINK: Additive/related (PREFERENCE, GOAL, FACT, etc.)
            # Use language that suggests addition, not replacement
            mem_type_lower = old_mem['type'].lower()

            if old_mem['type'] == 'PREFERENCE':
                meta_text = f"{old_mem['subject']} added similar preference: '{new_value}' (related to '{old_value}') on {timestamp}"
            elif old_mem['type'] == 'GOAL':
                meta_text = f"{old_mem['subject']} added similar goal: '{new_value}' (related to '{old_value}') on {timestamp}"
            elif old_mem['type'] == 'FACT':
                meta_text = f"{old_mem['subject']} added similar fact: '{new_value}' (related to '{old_value}') on {timestamp}"
            elif old_mem['type'] == 'RULE':
                meta_text = f"{old_mem['subject']} added similar rule: '{new_value}' (related to '{old_value}') on {timestamp}"
            elif old_mem['type'] == 'PERMISSION':
                meta_text = f"{old_mem['subject']} granted similar permission: '{new_value}' (related to '{old_value}') on {timestamp}"
            else:
                meta_text = f"{old_mem['subject']} added similar {mem_type_lower}: '{new_value}' (related to '{old_value}') on {timestamp}"

        else:
            # Fallback for other event types
            meta_text = f"{old_mem['subject']} {old_mem['type'].lower()} event: '{old_value}' → '{new_value}' on {timestamp}"

        # Create metadata with structured information
        metadata = {
            "timestamp": new_mem['created_at'],
            "is_replacement": event_type == "VERSION_UPDATE",
            "is_additive": event_type == "SIMILARITY_LINK"
        }

        # Save the meta-memory
        self.meta_memory_store.add_meta_memory(
            event_type=event_type,
            memory_type=old_mem['type'],
            subject=old_mem['subject'],
            text=meta_text,
            old_id=old_mem['id'],
            new_id=new_mem['id'],
            old_value=old_value,
            new_value=new_value,
            metadata=metadata
        )

        self.log(f"      🧠 Meta-memory created: {meta_text}")

    @staticmethod
    def _extract_value_from_text(text: str) -> str:
        """
        Extract the actual value from memory text.
        Example: "Assistant name is Ada" → "Ada"
        Example: "User lives in Van, Türkiye" → "Van, Türkiye"
        """
        text = text.strip()

        # Common patterns
        patterns = [
            " is ",
            " lives in ",
            " works at ",
            " wants to ",
            " prefers ",
        ]

        for pattern in patterns:
            if pattern in text.lower():
                parts = text.split(pattern, 1)
                if len(parts) == 2:
                    return parts[1].strip()

        # Fallback: return the whole text
        return text

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
