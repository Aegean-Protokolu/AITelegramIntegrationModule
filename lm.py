import json
import ast
import re
import numpy as np
import requests
import os
import base64
from typing import List, Dict, Tuple, Callable, Optional

# ==============================
# Configuration Defaults
# ==============================

def _load_settings():
    if os.path.exists("settings.json"):
        try:
            with open("settings.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

_settings = _load_settings()

LM_STUDIO_BASE_URL = _settings.get("base_url", "http://127.0.0.1:1234/v1")
CHAT_MODEL = _settings.get("chat_model", "qwen2.5-vl-7b-instruct-abliterated")
EMBEDDING_MODEL = _settings.get("embedding_model", "text-embedding-nomic-embed-text-v1.5")
CHAT_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/chat/completions"

# ==============================
# System prompts
# ==============================

SYSTEM_PROMPT = _settings.get("system_prompt", "")
MEMORY_EXTRACTOR_PROMPT = _settings.get("memory_extractor_prompt", "")

# Aliases for external usage (e.g. GUI defaults)
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_MEMORY_EXTRACTOR_PROMPT = MEMORY_EXTRACTOR_PROMPT

# ==============================
# Local LM call
# ==============================

def run_local_lm(
    messages: list, 
    system_prompt: str = None, 
    temperature: float = None, 
    top_p: float = None, 
    max_tokens: int = None,
    base_url: str = None, 
    chat_model: str = None,
    stop_check_fn: Optional[Callable[[], bool]] = None,
    images: List[str] = None
) -> str:
    # Optimization: Only load settings from disk if arguments are missing
    if any(param is None for param in [system_prompt, temperature, top_p, max_tokens, base_url, chat_model]):
        settings = _load_settings()
    else:
        settings = {}

    # Resolve parameters: Argument > Settings.json > Global Default
    if system_prompt is None:
        system_prompt = settings.get("system_prompt", SYSTEM_PROMPT)
    if temperature is None:
        temperature = settings.get("temperature", 0.7)
    if top_p is None:
        top_p = settings.get("top_p", 0.94)
    if max_tokens is None:
        max_tokens = settings.get("max_tokens", 800)
    if base_url is None:
        base_url = settings.get("base_url", LM_STUDIO_BASE_URL)
    if chat_model is None:
        chat_model = settings.get("chat_model", CHAT_MODEL)

    # Handle Vision Payload
    final_messages = [{"role": "system", "content": system_prompt}]
    
    if images:
        # Construct multi-modal user message
        content_payload = []
        # Add text from the last user message if it exists
        last_text = messages[-1]['content'] if messages else "Analyze this image."
        content_payload.append({"type": "text", "text": last_text})
        
        for img_path in images:
            if os.path.exists(img_path):
                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        
        # Replace the last message with the multi-modal one
        final_messages.extend(messages[:-1])
        final_messages.append({"role": "user", "content": content_payload})
    else:
        final_messages.extend(messages)

    payload = {
        "model": chat_model,
        "messages": final_messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    # Enable streaming if we have a stop check function
    if stop_check_fn:
        payload["stream"] = True

    try:
        chat_completions_url = f"{base_url}/chat/completions"
        
        if stop_check_fn:
            full_content = ""
            with requests.post(chat_completions_url, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if stop_check_fn():
                        return full_content + " [Interrupted]"
                    
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                delta = data_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                            except:
                                pass
            return full_content
        else:
            # Standard non-streaming request
            r = requests.post(chat_completions_url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
            
    except Exception as e:
        # Debugging for context length issues
        if "400" in str(e) and not images: # Don't auto-retry vision requests yet
            total_chars = sum(len(m.get("content", "")) for m in messages) + len(system_prompt)
            print(f"⚠️ [LM Error] 400 Bad Request. Approx Prompt Length: {total_chars} chars. Reduce context.")
            
            # Auto-Retry Strategy: Prune oldest messages
            if len(messages) > 1:
                print("🔄 Auto-retrying with pruned context...")
                return run_local_lm(messages[1:], system_prompt, temperature, top_p, max_tokens, base_url, chat_model, stop_check_fn)
                
        return f"⚠️ Local model error: {e}"

# ==============================
# Loose JSON array parser
# ==============================

def _parse_json_array_loose(raw: str) -> list:
    if not raw:
        return []

    raw = raw.strip()

    # Strip markdown code blocks if present (e.g. ```json ... ```)
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
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

    # Try parsing as Python literal (handles single quotes, None, True/False)
    try:
        data = ast.literal_eval(extracted)
        return data if isinstance(data, list) else []
    except Exception:
        return []

# ==============================
# Embeddings via LM Studio
# ==============================

def compute_embedding(text: str, base_url: str = None, embedding_model: str = None) -> np.ndarray:
    # Optimization: Only load settings from disk if arguments are missing
    if base_url is None or embedding_model is None:
        settings = _load_settings()
    else:
        settings = {}
    
    if base_url is None:
        base_url = settings.get("base_url", LM_STUDIO_BASE_URL)
    if embedding_model is None:
        embedding_model = settings.get("embedding_model", EMBEDDING_MODEL)

    if not text.strip():
        return np.zeros(768)
    payload = {"model": embedding_model, "input": text}
    try:
        url = f"{base_url}/embeddings"
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        emb = np.array(data["data"][0]["embedding"], dtype=float)
        return emb
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        # Return zero vector to prevent random similarity matches in vector DB
        return np.zeros(768)

# ==============================
# Memory extraction
# ==============================

def extract_memories_llm(
    user_text: str,
    assistant_text: str,
    force: bool = False,
    auto: bool = False,
    base_url: str = LM_STUDIO_BASE_URL,
    chat_model: str = CHAT_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
    memory_extractor_prompt: str = MEMORY_EXTRACTOR_PROMPT,
    custom_instruction: str = None,
    stop_check_fn: Optional[Callable[[], bool]] = None
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Extract memories with subject/type.
    Returns (memories_list, embeddings_list)
    """
    if custom_instruction:
        instruction = custom_instruction
    elif force:
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
            "EXCLUDE: pure questions, pure greetings ('hi', 'hello' alone), filler ('how are you'). DO NOT exclude facts just because they were repeated.\n"
            "CRITICAL: DO NOT attribute Assistant's suggestions, lists, or hypothetical topics to the User. Only record User interests if the USER explicitly stated them.\n"
            "Return ONLY the JSON array. If no valid memories, return []."
        )

    convo = [
        {"role": "user", "content": f"User said: {user_text or ''}\n\nAssistant replied: {assistant_text or ''}\n\n{instruction}"},
    ]

    # print(f"💡 [Debug] Sending to LLM for extraction:")
    # print(f"   User text: '{user_text}'")
    # print(f"   Assistant text: '{assistant_text}'")

    raw = run_local_lm(convo, system_prompt=memory_extractor_prompt, temperature=0.1, base_url=base_url, chat_model=chat_model, stop_check_fn=stop_check_fn).strip()
    # print("💡 [Debug] Raw LM output for memory extraction:\n", raw)
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
            "text": text[:1000],
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
        embeddings.append(compute_embedding(m["text"], base_url=base_url, embedding_model=embedding_model))

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
    protected_types = {"IDENTITY", "PERMISSION", "RULE", "GOAL", "BELIEF", "PREFERENCE"}
    
    # Allow FACT if it contains "name is" or other identity markers (prevents filtering "My name is X")
    is_identity_fact = "name is" in text_lower or "lives in" in text_lower or "works at" in text_lower or "i am" in text_lower or "user is" in text_lower
    
    if word_count < 3 and mem_type and mem_type.upper() not in protected_types and not is_identity_fact:
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
        # Example: "help with academic resources and professional development at Van..."
        # This is contextual, not a true self-chosen goal
        if "help with" in text_lower or "assist with" in text_lower:
            # Count specific nouns (indicates context-specific goal)
            specific_keywords = ["academic", "professional", "university", "medical", "research", 
                               "study", "studies", "work", "question", "topic", "information"]
            if any(keyword in text_lower for keyword in specific_keywords):
                return True

        # Filter out passive research recommendations from documents (often misclassified as GOALs)
        # e.g. "Future investigations should focus on...", "Further research is needed..."
        passive_research_patterns = [
            "future investigation", "future research", "further research", 
            "further investigation", "further studies", "additional studies",
            "comprehensive education", "therapeutic approaches need",
            "there is a need", "this finding may offer", "needs to be", "should focus on"
        ]
        # Only filter if it doesn't explicitly mention the assistant/I doing it
        if any(text_lower.startswith(p) for p in passive_research_patterns) and "assistant" not in text_lower and " i " not in text_lower:
            return True
            
        # Catch "The goal is to..." when it refers to a study's goal, not the assistant's
        if text_lower.startswith("the goal is to") and "assistant" not in text_lower:
                return True

    return False

def extract_memory_candidates(
    user_text: str, 
    assistant_text: str, 
    force: bool = False, 
    auto: bool = False,
    base_url: str = LM_STUDIO_BASE_URL,
    chat_model: str = CHAT_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
    memory_extractor_prompt: str = MEMORY_EXTRACTOR_PROMPT,
    custom_instruction: str = None,
    stop_check_fn: Optional[Callable[[], bool]] = None
):
    """
    OLD function signature compatibility:
    Returns only list of memory dicts for old bot.py.
    NOW with filtering to remove low-quality candidates.
    """
    memories, _ = extract_memories_llm(user_text, assistant_text, force=force, auto=auto, base_url=base_url, chat_model=chat_model, embedding_model=embedding_model, memory_extractor_prompt=memory_extractor_prompt, custom_instruction=custom_instruction, stop_check_fn=stop_check_fn)

    # Filter out low-quality candidates
    filtered = []
    for m in memories:
        if not _is_low_quality_candidate(m["text"], mem_type=m.get("type")):
            filtered.append(m)

    return filtered
