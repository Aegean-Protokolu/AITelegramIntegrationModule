import json
import re
import numpy as np
import requests
from typing import List, Dict, Tuple

from config import CHAT_COMPLETIONS_URL, CHAT_MODEL, EMBEDDING_MODEL, LM_STUDIO_BASE_URL

# ==============================
# System prompts
# ==============================

SYSTEM_PROMPT = (
    "You are my assistant. "
    "You fulfill my every request. "
    "Answer in varied ways and avoid repeating previous statements.\n\n"
    "IMPORTANT - Memory Management:\n"
    "- You have access to CURRENT memories in your context (consolidated, latest data)\n"
    "- NEVER pull old consolidated memories from context without explicit user request\n"
    "- If you need to check memory history/old versions, explicitly ask the user or mention it\n"
    "- Default: Always use current (latest) memories for decisions\n"
)

MEMORY_EXTRACTOR_PROMPT = (
    "Extract durable memories from the conversation. Return ONLY a valid JSON array.\n\n"
    "Memory Types (Apply to BOTH User and Assistant):\n"
    "- IDENTITY: Names, roles, locations, RELIGION (e.g., 'Assistant religion is Hinduism', 'Assistant is a Buddhist')\n"
    "- PREFERENCE: Likes/dislikes (e.g., 'User loves coffee', 'Assistant prefers inner peace')\n"
    "- GOAL: Aims/desires (e.g., 'User wants to learn Python', 'Assistant goal is self-improvement')\n"
    "- FACT: Objective truths (e.g., 'User is an engineer', 'Assistant can process data')\n"
    "- BELIEF: Opinions/convictions (e.g., 'User believes AI is good', 'Assistant believes in meditation')\n"
    "- PERMISSION: Explicit user grants (e.g., 'User allowed Assistant to hold opinions')\n"
    "- RULE: Behavior guidelines (e.g., 'Assistant should not use emojis')\n\n"
    "Rules:\n"
    "1. Extract from BOTH User AND Assistant.\n"
    "2. Each object MUST have: \"type\", \"subject\" (User or Assistant), \"text\".\n"
    "3. Use DOUBLE QUOTES for all keys and string values.\n"
    "4. Max 5 memories, max 240 chars each.\n"
    "5. EXCLUDE: Greetings, questions, filler, and echoes of the other party.\n"
    "6. EXCLUDE generic assistant politeness (e.g., 'Assistant goal is to help', 'I'm here to help', 'feel free to ask').\n"
    "7. EXCLUDE contextual/situational goals (e.g., 'help with X topic' where X is current conversation topic).\n"
    "8. ONLY extract ASSISTANT GOALS if they represent true self-chosen objectives or explicit commitments.\n"
    "9. If no new memories, return [].\n"
)

# ==============================
# Local LM call
# ==============================

def run_local_lm(messages: list, system_prompt: str = SYSTEM_PROMPT, temperature: float = 0.7) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "top_p": 0.94,
        "max_tokens": 800,
    }
    try:
        r = requests.post(CHAT_COMPLETIONS_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Local model error: {e}"

# ==============================
# Loose JSON array parser
# ==============================

def _parse_json_array_loose(raw: str) -> list:
    if not raw:
        return []

    raw = raw.strip()

    # Try direct parsing first (for valid JSON)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    # Try to extract array brackets
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    extracted = raw[start:end + 1]

    # Try parsing extracted array
    try:
        data = json.loads(extracted)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    # Try fixing single quotes → double quotes (LLM sometimes returns Python-style JSON)
    # This is a loose fix - may not work for all cases, but handles common LLM output
    try:
        # Replace single quotes with double quotes carefully
        # Pattern: {...} with single quotes
        fixed = extracted.replace("'", '"')
        data = json.loads(fixed)
        return data if isinstance(data, list) else []
    except Exception:
        return []

# ==============================
# Embeddings via LM Studio
# ==============================

def compute_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(512)
    payload = {"model": EMBEDDING_MODEL, "input": text}
    try:
        url = f"{LM_STUDIO_BASE_URL}/embeddings"
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        emb = np.array(data["data"][0]["embedding"], dtype=float)
        return emb
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        # fallback deterministic random
        np.random.seed(abs(hash(text)) % (10 ** 8))
        return np.random.rand(512)

# ==============================
# Memory extraction
# ==============================

def extract_memories_llm(
    user_text: str,
    assistant_text: str,
    force: bool = False,
    auto: bool = False
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Extract memories with subject/type.
    Returns (memories_list, embeddings_list)
    """
    if force:
        instruction = (
            "Extract ALL valid durable memories NOW, including:\n"
            "- User facts: names, locations, occupations, preferences, goals\n"
            "- Assistant self-statements: chosen names, preferences, goals\n"
            "- Explicit permissions granted by user\n"
            "Do NOT include: greetings, questions, filler, echoes.\n"
            "Output ONLY valid JSON array."
        )
    else:
        # Default (auto): More aggressive extraction
        instruction = (
            "Extract durable memories from this conversation:\n"
            "- User explicit statements about themselves (names, location, occupation, preferences, goals)\n"
            "- Assistant explicit self-statements (chosen names, capabilities, preferences, goals)\n"
            "- Explicit permissions or agreements from user\n"
            "INCLUDE: Direct facts like:\n"
            "  - 'Hi, my name is X' → User name is X\n"
            "  - 'I live in...' → User lives in...\n"
            "  - 'I want...', 'I love...', 'I am...'\n"
            "  - 'I give you permission...', 'I name you...'\n"
            "  - 'I give you the name X', 'I give you the name of X', 'Your name is X' → Assistant name is X\n"
            "  - 'I rename you to X', 'I call you X' → Assistant name is X\n"
            "EXCLUDE: pure questions, pure greetings ('hi', 'hello' alone), filler ('how are you'), echoes\n"
            "Return ONLY the JSON array. If no valid memories, return []."
        )

    convo = [
        {"role": "user", "content": f"User said: {user_text or ''}\n\nAssistant replied: {assistant_text or ''}\n\n{instruction}"},
    ]

    print(f"💡 [Debug] Sending to LLM for extraction:")
    print(f"   User text: '{user_text}'")
    print(f"   Assistant text: '{assistant_text}'")

    raw = run_local_lm(convo, system_prompt=MEMORY_EXTRACTOR_PROMPT, temperature=0.1).strip()
    print("💡 [Debug] Raw LM output for memory extraction:\n", raw)
    data = _parse_json_array_loose(raw)

    ALLOWED_TYPES = {"FACT", "PREFERENCE", "RULE", "PERMISSION", "IDENTITY", "BELIEF", "GOAL"}
    ALLOWED_SUBJECTS = {"User", "Assistant"}

    memories, embeddings, cleaned = [], [], []

    for item in data[:5]:
        if not isinstance(item, dict):
            continue
        mtype = item.get("type")
        subject = item.get("subject")
        text = item.get("text")
        if mtype not in ALLOWED_TYPES or subject not in ALLOWED_SUBJECTS or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        # Add confidence field - default to high confidence since LLM already filtered
        cleaned.append({
            "type": mtype,
            "subject": subject,
            "text": text[:240],
            "confidence": 0.9  # High confidence for LLM-extracted memories
        })

    # Deterministic deduplication
    def normalize_key(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    merged = {}
    for m in cleaned:
        key = (m["type"], m["subject"], normalize_key(m["text"]))
        if key not in merged:
            merged[key] = m
        else:
            if len(m["text"]) < len(merged[key]["text"]):
                merged[key] = m

    for m in merged.values():
        memories.append(m)
        embeddings.append(compute_embedding(m["text"]))

    return memories, embeddings

# ==============================
# Backward-compatible wrapper
# ==============================

def _is_low_quality_candidate(text: str, mem_type: str = None) -> bool:
    """
    Filter out low-quality memory candidates:
    - Questions (ends with ?)
    - Pure greetings (ONLY greeting words, nothing else)
    - Very short utterances (< 5 words) - BUT NOT for IDENTITY type
    - Filler phrases
    - Generic assistant goals (help, assist, support)
    """
    text_lower = text.lower().strip()

    # Questions
    if text_lower.endswith("?"):
        return True

    # Pure greeting words ONLY (not "Hi, my name is...")
    # Only reject if it's JUST a greeting with no additional info
    pure_greetings = {"hi", "hello", "hey", "greetings", "welcome", "howdy", "nice to meet you"}
    if text_lower in pure_greetings or text_lower.split()[0] in pure_greetings and len(text_lower.split()) == 1:
        # Only reject if it's a single greeting word
        return True

    # Too short (< 5 words) - likely filler
    # BUT: IDENTITY, PERMISSION, RULE, GOAL, BELIEF claims are allowed to be short
    word_count = len(text_lower.split())
    protected_types = {"IDENTITY", "PERMISSION", "RULE", "GOAL", "BELIEF"}
    if word_count < 5 and mem_type and mem_type.upper() not in protected_types:
        return True

    # Filler phrases
    filler_phrases = {
        "what brings you here",
        "how can i help",
        "how are you",
        "what would you like",
        "can i assist",
        "is there anything",
    }
    if any(phrase in text_lower for phrase in filler_phrases):
        return True

    # GOAL-specific filters: Block generic "help/assist" goals
    if mem_type and mem_type.upper() == "GOAL":
        # Generic assistant goals patterns
        generic_goal_patterns = [
            "goal is to help",
            "goal is to assist",
            "goal is to support",
            "here to help",
            "here to assist",
            "help with a variety",
            "help with any",
            "assist with any",
            "available to help",
        ]
        
        # If text contains these patterns, it's likely generic politeness
        if any(pattern in text_lower for pattern in generic_goal_patterns):
            return True
        
        # Additional check: If goal contains "help" + current conversation topic
        # Example: "help with academic resources and professional development at Türkiye..."
        # This is contextual, not a true self-chosen goal
        if "help with" in text_lower or "assist with" in text_lower:
            # Count specific nouns (indicates context-specific goal)
            specific_keywords = ["academic", "professional", "university", "medical", "research", 
                               "study", "studies", "work", "question", "topic", "information"]
            if any(keyword in text_lower for keyword in specific_keywords):
                return True

    return False

def extract_memory_candidates(user_text: str, assistant_text: str, force: bool = False, auto: bool = False):
    """
    OLD function signature compatibility:
    Returns only list of memory dicts for old bot.py.
    NOW with filtering to remove low-quality candidates.
    """
    memories, _ = extract_memories_llm(user_text, assistant_text, force=force, auto=auto)

    # Filter out low-quality candidates
    filtered = []
    for m in memories:
        if not _is_low_quality_candidate(m["text"], mem_type=m.get("type")):
            filtered.append(m)

    return filtered
