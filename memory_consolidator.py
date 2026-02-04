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
from typing import List, Dict, Optional
from datetime import datetime

from memory import MemoryStore
from meta_memory import MetaMemoryStore
from lm import compute_embedding
import numpy as np


class MemoryConsolidator:
    """Consolidates similar/duplicate memories to keep memory store clean."""

    def __init__(self, memory_store: MemoryStore, meta_memory_store: Optional[MetaMemoryStore] = None, similarity_threshold: float = 0.85):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.similarity_threshold = similarity_threshold

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
            print("🧠 [Consolidator] No memories to consolidate")
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
            key = (mem['subject'], mem['type'])
            if key not in groups_by_type: groups_by_type[key] = []
            groups_by_type[key].append(mem)

        for (subj, mtype), group in groups_by_type.items():
            stats['processed'] += 1
            if len(group) < 2: continue

            for i in range(len(group)):
                old = group[i]
                # Skip if already consolidated or if it already has a parent (avoid flattening)
                if old['id'] in identity_consolidated_ids or old['parent_id'] is not None: 
                    continue
                
                old_emb = None
                for j in range(i + 1, len(group)):
                    new = group[j]
                    # Skip if already consolidated or if it already has a parent
                    if new['id'] in identity_consolidated_ids: 
                        continue

                    if old_emb is None: old_emb = compute_embedding(old['text'])
                    new_emb = compute_embedding(new['text'])
                    similarity = self._cosine_similarity(old_emb, new_emb)
                    
                    # Dynamic threshold based on memory type
                    # GOAL types: more aggressive consolidation (generic "help with X" statements)
                    # IDENTITY/BELIEF: moderate consolidation (handle duplicates/expansions)
                    # Others: conservative (preserve unique information)
                    if old['type'] == 'GOAL':
                        threshold = 0.88
                    elif old['type'] in ('IDENTITY', 'BELIEF', 'PERMISSION'):
                        threshold = 0.87  # Lowered to catch 0.88-0.89 similarities
                    else:
                        threshold = 0.93  # Conservative for FACT/PREFERENCE/RULE
                    
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
                                print(f"      🔍 Substring match: '{old_value[:50]}' ⊂ '{new_value[:50]}'")
                    
                    # Also check BELIEF substring (e.g., "believes in Islam" vs "believes in Islam and...")
                    if old['type'] == 'BELIEF' and old['subject'] == new['subject']:
                        old_value = self._extract_value_from_text(old['text'])
                        new_value = self._extract_value_from_text(new['text'])
                        
                        old_normalized = old_value.lower().strip()
                        new_normalized = new_value.lower().strip()
                        
                        if old_normalized != new_normalized:
                            if old_normalized in new_normalized or new_normalized in old_normalized:
                                is_substring_expansion = True
                                print(f"      🔍 Substring match: '{old_value[:50]}' ⊂ '{new_value[:50]}'")
                    
                    # Debug output
                    if old['type'] in ('IDENTITY', 'BELIEF', 'PERMISSION') and similarity > 0.85:
                        print(f"      🔍 Comparing {old['type']}: similarity={similarity:.3f}, threshold={threshold:.3f}, substring={is_substring_expansion}")
                        print(f"         OLD: {old['text'][:60]}")
                        print(f"         NEW: {new['text'][:60]}")
                    
                    if similarity >= threshold or is_substring_expansion:
                        if self._mark_consolidated(old['id'], new['id']):
                            stats['consolidated'] += 1
                            identity_consolidated_ids.add(old['id'])
                            if self.meta_memory_store:
                                self._create_meta_memory(old, new, "SIMILARITY_LINK")
                            break
        return stats

    def _consolidate_group(self, group: List[Dict]) -> int:
        """DEPRECATED: Logic moved to main consolidate function."""
        return 0

    def _get_memories_after(self, cutoff_time: int) -> List[Dict]:
        """Get all memories created after cutoff_time."""
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at
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
            })
        return memories

    def _get_all_memories(self) -> List[Dict]:
        """Get ALL memories regardless of age."""
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at
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
                    print(f"      🔗 Updated parent_id: ID {old_id} now points to ID {new_id}")
                    return True
                else:
                    # Already consolidated
                    return False
            return False

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

        print(f"      🧠 Meta-memory created: {meta_text}")

    @staticmethod
    def _extract_value_from_text(text: str) -> str:
        """
        Extract the actual value from memory text.
        Example: "Assistant name is Ada" → "Ada"
        Example: "User lives in Türkiye" → "Türkiye"
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
