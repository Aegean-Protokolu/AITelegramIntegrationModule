from typing import Optional, Callable

from memory import MemoryStore
from meta_memory import MetaMemoryStore


TYPE_PRECEDENCE = {
    "PERMISSION": 0,  # Highest priority: only user can grant
    "RULE": 1,        # Rules from/for assistant
    "IDENTITY": 2,    # Who is user/assistant
    "PREFERENCE": 3,  # Likes/dislikes
    "GOAL": 4,        # Aims/desires
    "FACT": 5,        # Objective truths
    "BELIEF": 6,      # Opinions/convictions (lowest priority)
    "NOTE": 1,        # Assistant Notes (High priority, internal)
}

CONFIDENCE_MIN = {
    "PERMISSION": 0.85,  # Very high: explicit user permission (only user can grant)
    "RULE": 0.9,         # Very high: guidelines for assistant behavior
    "IDENTITY": 0.8,     # High: identity claims (who are you)
    "PREFERENCE": 0.6,   # Medium-high: preferences/likes/dislikes
    "GOAL": 0.7,         # High: goals/desires
    "FACT": 0.7,         # High: factual assertions
    "BELIEF": 0.5,       # Medium: beliefs/opinions/convictions
    "NOTE": 0.9,         # Very high: manual notes
}


class MemoryArbiter:
    """
    Autonomous bridge between ReasoningStore and MemoryStore.

    - Does NOT reason
    - Does NOT decide truth
    - Only enforces admission rules
    """

    def __init__(self, memory_store: MemoryStore, meta_memory_store: Optional[MetaMemoryStore] = None, embed_fn: Optional[Callable] = None, log_fn: Callable[[str], None] = print):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.embed_fn = embed_fn
        self.log = log_fn

    # --------------------------
    # Public API
    # --------------------------
    def consider(
        self,
        text: str,
        mem_type: str,
        confidence: float,
        subject: str = "User",
        source: str = "reasoning"
    ) -> Optional[int]:
        """
        Decide whether to promote reasoning into memory.

        Returns memory_id if stored, None otherwise.
        """

        mem_type = mem_type.upper()
        self.log(f"🔍 [Arbiter] Processing: type={mem_type}, subject={subject}, confidence={confidence}, text={text[:50]}")

        if mem_type not in TYPE_PRECEDENCE:
            self.log(f"❌ [Arbiter] Type '{mem_type}' not in precedence table")
            return None

        # 1️⃣ Confidence gate
        min_conf = CONFIDENCE_MIN[mem_type]
        if confidence < min_conf:
            self.log(f"❌ [Arbiter] Confidence gate failed: {confidence} < {min_conf} (required for {mem_type})")
            return None

        self.log(f"✅ [Arbiter] Passed confidence gate: {confidence} >= {min_conf}")

        # 2️⃣ Identity + version chaining
        identity = self.memory_store.compute_identity(text, mem_type=mem_type)
        previous_versions = self.memory_store.get_by_identity(identity)
        parent_id = previous_versions[-1]["id"] if previous_versions else None

        # 2.5️⃣ Exact duplicate guard (same text as latest version)
        if previous_versions:
            last_text = previous_versions[-1]["text"].strip()
            if last_text == text.strip():
                self.log(f"❌ [Arbiter] Exact duplicate detected (ID: {previous_versions[-1]['id']})")
                return None

        self.log(f"✅ [Arbiter] No duplicates. Previous versions: {len(previous_versions)}, parent_id: {parent_id}")

        # 3️⃣ Conflict detection (exact, conservative)
        
        # 2.6️⃣ Cross-Subject Identity Conflict Guard
        # Prevent Assistant from claiming User's name/identity and vice versa
        # Check this for IDENTITY type OR if the text looks like an identity claim
        extracted_val = self._extract_value(text)
        if extracted_val and (mem_type == "IDENTITY" or "name is" in text.lower()):
            # Check against all active identities (and FACTs that act like identities)
            active_identities = self.memory_store.get_active_by_type("IDENTITY")
            active_facts = self.memory_store.get_active_by_type("FACT")
            
            for _, subj, txt in (active_identities + active_facts):
                # If subject is different (e.g. User vs Assistant)
                if subj.lower() != subject.lower():
                    existing_val = self._extract_value(txt)
                    if existing_val and existing_val.lower() == extracted_val.lower():
                        self.log(f"❌ [Arbiter] Identity conflict: '{extracted_val}' is already assigned to {subj}")
                        return None

        conflicts = self.memory_store.find_conflicts_exact(text)
        self.log(f"🔍 [Arbiter] Found {len(conflicts)} conflicts")

        # 4️⃣ Precedence dampening
        adjusted_confidence = confidence
        for c in conflicts:
            if TYPE_PRECEDENCE[c["type"]] > TYPE_PRECEDENCE[mem_type]:
                adjusted_confidence *= 0.5
                self.log(f"⚠️ [Arbiter] Dampening confidence due to conflict with {c['type']}: {confidence} -> {adjusted_confidence}")

        # 5️⃣ Append-only write (versioned)
        self.log(f"✅ [Arbiter] Saving memory with adjusted_confidence={adjusted_confidence}")
        
        # Use consistent timestamp for both memory and meta-memory
        # Generate embedding if function provided
        embedding = None
        if self.embed_fn:
            embedding = self.embed_fn(text)

        import time
        created_at = int(time.time())
        from datetime import datetime
        timestamp = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")

        memory_id = self.memory_store.add_entry(
            text=text,
            mem_type=mem_type,
            subject=subject,
            confidence=adjusted_confidence,
            source=source,
            identity=identity,
            parent_id=parent_id,
            conflicts=[c["id"] for c in conflicts],
            created_at=created_at,
            embedding=embedding,
        )

        # 6️⃣ Create meta-memory about this new memory creation
        if self.meta_memory_store and memory_id:
            # Extract the current value
            value = self._extract_value(text)
            
            # Check for previous version to show change
            old_value = None
            if previous_versions:
                old_value = self._extract_value(previous_versions[-1]["text"])

            # Create human-readable meta-memory
            if mem_type == 'IDENTITY':
                if 'name is' in text.lower():
                    if old_value:
                        meta_text = f"{subject} name changed from {old_value} to {value} on {timestamp}"
                    else:
                        meta_text = f"{subject} name set to {value} on {timestamp}"
                elif 'lives in' in text.lower():
                    if old_value:
                        meta_text = f"{subject} location changed from {old_value} to {value} on {timestamp}"
                    else:
                        meta_text = f"{subject} location set to {value} on {timestamp}"
                else:
                    if old_value:
                        meta_text = f"{subject} {mem_type.lower()} updated: '{old_value}' -> '{value}' on {timestamp}"
                    else:
                        meta_text = f"{subject} {mem_type.lower()} recorded: '{value}' on {timestamp}"
            elif old_value:
                meta_text = f"{subject} {mem_type.lower()} updated: '{old_value}' -> '{value}' on {timestamp}"
            else:
                # Default for new recordings
                type_labels = {
                    'PREFERENCE': 'preference',
                    'GOAL': 'goal',
                    'FACT': 'fact',
                    'RULE': 'rule',
                    'PERMISSION': 'permission',
                    'BELIEF': 'belief'
                }
                label = type_labels.get(mem_type, mem_type.lower())
                meta_text = f"{subject} {label} recorded: '{value}' on {timestamp}"

            self.meta_memory_store.add_meta_memory(
                event_type="VERSION_UPDATE" if old_value else "MEMORY_CREATED",
                memory_type=mem_type,
                subject=subject,
                text=meta_text,
                old_id=previous_versions[-1]["id"] if previous_versions else None,
                new_id=memory_id,
                old_value=old_value,
                new_value=value,
                metadata={"confidence": adjusted_confidence, "source": source}
            )
            self.log(f"      🧠 Meta-memory: {meta_text}")

        return memory_id

    @staticmethod
    def _extract_value(text: str) -> str:
        """Extract the value from memory text."""
        text = text.strip()
        patterns = [" is ", " lives in ", " works at ", " wants to ", " prefers ", " loves ", " allowed ", " granted "]
        for pattern in patterns:
            if pattern in text.lower():
                parts = text.split(pattern, 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return text
