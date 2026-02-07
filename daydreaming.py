import time
import random
import re
from typing import Dict, Callable, Optional, List

from lm import run_local_lm, extract_memory_candidates, DEFAULT_MEMORY_EXTRACTOR_PROMPT, _parse_json_array_loose, compute_embedding
from event_bus import EventBus

# Specialized prompt for extracting insights from internal monologue
DAYDREAM_EXTRACTOR_PROMPT = (
    "Extract insights, goals, facts, and preferences from the Assistant's internal monologue. "
    "Return ONLY a valid JSON array.\n\n"
    "Memory Types:\n"
    "- GOAL: Specific, actionable objectives for the Assistant (e.g., 'Assistant plans to cross-reference X with Y'). Do NOT extract general statements like 'Future research should...' as GOALs; classify them as BELIEFS or FACTS instead.\n"
    "- FACT: Objective truths derived from documents or reasoning\n"
    "- BELIEF: Opinions, convictions, hypotheses, or research insights\n"
    "- PREFERENCE: Personal likes/dislikes ONLY (e.g., 'Assistant enjoys sci-fi'). DO NOT use for research suggestions, hypotheses, or document relevance.\n\n"
    "Rules:\n"
    "1. Extract from the Assistant's text.\n"
    "2. Each object MUST have: \"type\", \"subject\" (must be 'Assistant'), \"text\".\n"
    "3. Use DOUBLE QUOTES for all keys and string values.\n"
    "4. Max 5 memories.\n"
    "5. MAKE MEMORIES SELF-CONTAINED: Replace pronouns like 'This', 'These', 'It' with specific nouns. Ensure the text makes sense without the surrounding context.\n"
    "6. Return ONLY the JSON array. If no new memories, return [].\n"
)

DAYDREAM_INSTRUCTION = (
    "Analyze the Internal Monologue above. "
    "Extract key insights as FACT, BELIEF, GOAL, or PREFERENCE memories for the Assistant. "
    "Format as JSON objects with keys: 'type', 'subject' (must be 'Assistant'), 'text'. "
    "Ensure the text includes the source document filename if mentioned. "
    "CRITICAL: Replace pronouns (e.g., 'This', 'These', 'It') with specific nouns to make the memory self-contained. "
    "Return ONLY a valid JSON array. Do not invent sources."
)

class Daydreamer:
    """
    Autonomous thought generator.
    
    Operates when the system is idle to:
    1. Review current goals and recent memories.
    2. Generate new insights or connections.
    3. Feed these insights back into the memory system.
    """
    def __init__(
        self, 
        memory_store, 
        reasoning_store, 
        arbiter, 
        document_store,
        get_settings_fn: Callable[[], Dict],
        log_fn: Callable[[str], None] = print,
        chat_fn: Optional[Callable[[str, str], None]] = None,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        status_fn: Optional[Callable[[str], None]] = None,
        get_chat_history_fn: Optional[Callable[[], List[Dict]]] = None,
        event_bus: Optional[EventBus] = None
    ):
        self.memory_store = memory_store
        self.reasoning_store = reasoning_store
        self.arbiter = arbiter
        self.document_store = document_store
        self.get_settings = get_settings_fn
        self.log = log_fn
        self.chat_fn = chat_fn
        self.stop_check = stop_check_fn or (lambda: False)
        self.status_fn = status_fn
        self.get_chat_history = get_chat_history_fn or (lambda: [])
        self.event_bus = event_bus
        self.last_daydream_time = 0

    def perform_daydream(self, mode: str = "auto", topic: Optional[str] = None):
        """
        Autonomous thought generation. mode: 'auto', 'read', 'insight'
        topic: Optional subject to focus on (e.g. "Neurology")
        """
        msg = "☁️ Starting daydreaming cycle..."
        if topic: msg += f" (Focus: {topic})"
        self.log(msg)
        
        if self.event_bus:
            self.event_bus.publish("DAYDREAM_START", {"mode": mode, "topic": topic}, source="Daydreamer")
        
        if self.stop_check():
            return

        try:
            settings = self.get_settings()
            reading_filename = None
            
            # 1. Gather Context
            recent_memories = self.memory_store.list_recent(limit=10)
            goals = self.memory_store.get_active_by_type("GOAL")

            if len(goals) > 10:
                goals = random.sample(goals, 10)

            context = "Current Knowledge State:\n"
            if goals:
                context += "Active Goals:\n" + "\n".join([f"- {t}" for _, s, t in goals]) + "\n"
            
            if recent_memories:
                context += "Recent Memories:\n" + "\n".join([f"- [{m[1]}] {m[3][:150]}..." for m in recent_memories]) + "\n"
            
            # --- Belief Analysis Injection ---
            # Allow INSIGHT mode to target a topic (specific belief) or random if no topic
            # AUTO mode only triggers random belief analysis occasionally
            analyzing_belief = False
            should_analyze_belief = (mode == "insight") or (mode == "auto" and not topic and random.random() < 0.25)

            if should_analyze_belief:
                # Only analyze beliefs generated by daydreaming to avoid pruning chat memories
                all_mems = self.memory_store.list_recent(limit=None)
                beliefs = [(m[0], m[2], m[3]) for m in all_mems if m[1] == 'BELIEF' and m[4] == 'daydream']
                if beliefs:
                    selected_belief = None
                    
                    # If topic provided in INSIGHT mode, try to find matching belief
                    if topic and mode == "insight":
                        matches = [b for b in beliefs if topic.lower() in b[2].lower()]
                        if matches:
                            selected_belief = random.choice(matches)
                            self.log(f"☁️ Daydreaming focus: Targeted belief analysis on '{topic}'")
                    
                    # Fallback to random if no topic or no match
                    if not selected_belief:
                        selected_belief = random.choice(beliefs)

                    belief_id, subj, belief_text = selected_belief
                    analyzing_belief = True
                    self.log(f"☁️ Daydreaming focus: Analyzing belief '{belief_text[:50]}...'")
                    
                    # Find related facts to support/contradict
                    emb = compute_embedding(belief_text, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
                    related_items = self.memory_store.search(emb, limit=5)
                    
                    # NEW: Search documents for raw evidence
                    related_docs = []
                    if self.document_store:
                        # Search for chunks relevant to the belief
                        related_docs = self.document_store.search_chunks(emb, top_k=3)
                        if related_docs:
                            self.log(f"☁️ Found {len(related_docs)} document chunks related to belief.")
                    
                    context += f"\n--- BELIEF UNDER REVIEW ---\n[ID: {belief_id}] [{subj}] {belief_text}\n"
                    
                    if related_docs:
                        context += "Relevant Document Evidence:\n"
                        for d in related_docs:
                             context += f"- [Doc: {d['filename']}] {d['text'][:300]}...\n"
                    
                    context += "Relevant Facts/Memories:\n"
                    for r in related_items:
                        # r: (id, type, subject, text, similarity)
                        if r[3] != belief_text: # Don't list the belief itself as evidence
                            context += f"- [{r[1]}] {r[3]}\n"
                    context += "---------------------------\n"

            # Decision Phase
            if mode == "read":
                response = "[READ_RANDOM]"
            elif mode == "insight" or analyzing_belief:
                decision_prompt = (
                    "You are the AI Assistant reflecting on your internal state. "
                    "Review the provided Context (Goals and Memories). "
                )
                
                if analyzing_belief:
                    decision_prompt += (
                        "You are critically analyzing a specific BELIEF. "
                        "Compare it against the Relevant Facts and Document Evidence. "
                        "1. Is the belief supported by facts? If so, RESTATE the belief with the citation appended (e.g., '... [Supported by <filename>]').\n"
                        "2. Is it contradicted? If so, explicitly state 'Refuting Belief [ID: <id>]' and generate a corrected FACT.\n"
                        "   CRITICAL: If you generate a corrected FACT, you MUST append '[Source: <filename>]' using the filenames from the Document Evidence.\n"
                        "3. Is it ambiguous? Generate a GOAL to research it.\n"
                        "Output ONLY the resulting insight/thought."
                    )
                else:
                    decision_prompt += (
                        "Generate a new insight, hypothesis, or goal refinement now based ONLY on the provided memories. "
                        "Output ONLY the thought."
                    )

                messages = [{"role": "user", "content": context}]
                response = run_local_lm(
                    messages, system_prompt=decision_prompt, temperature=0.8, max_tokens=300, base_url=settings.get("base_url"), chat_model=settings.get("chat_model"),
                    stop_check_fn=self.stop_check
                )
            elif topic:
                # If a topic is explicitly set, default to reading about it
                response = "[READ_RANDOM]"
            elif not goals and not recent_memories:
                # Bootstrap: If nothing in memory, try to read
                response = "[READ_RANDOM]"
            else:
                decision_prompt = (
                    "You are the AI Assistant reflecting on your internal state. "
                    "Review the provided Context (Goals and Memories). "
                    "You have access to a library of documents. "
                    "Decide whether to reflect on existing knowledge or read a new document for inspiration. "
                    "To read a random document, output: [READ_RANDOM] "
                    "To reflect on current memory, generate a new insight, hypothesis, or goal refinement now based ONLY on the provided memories. "
                    "Output ONLY the thought or the command [READ_RANDOM]. Do NOT output [REFLECT]."
                )
                
                messages = [{"role": "user", "content": context}]
                
                response = run_local_lm(
                    messages,
                    system_prompt=decision_prompt,
                    temperature=0.8,
                    max_tokens=300,
                    base_url=settings.get("base_url"),
                    chat_model=settings.get("chat_model"),
                    stop_check_fn=self.stop_check
                )

            if self.stop_check():
                return

            thought = response.strip()
            
            # Check for LLM error before processing
            if thought.startswith("⚠️"):
                self.log(f"❌ Daydream generation failed: {thought}")
                return
            
            if thought.strip().upper() == "[REFLECT]":
                self.log("☁️ AI chose to reflect (explicit tag). Generating insight...")
                # Fallback: If LLM outputted [REFLECT] despite instructions, force a reflection generation
                reflection_prompt = (
                    "You have chosen to reflect on your internal state. "
                    "Review the Context (Goals and Memories). "
                    "Generate a new insight, hypothesis, or goal refinement now based ONLY on the provided memories. "
                    "Output ONLY the thought."
                )
                messages = [{"role": "user", "content": context}]
                thought = run_local_lm(
                    messages, 
                    system_prompt=reflection_prompt, 
                    temperature=0.8, 
                    max_tokens=300, 
                    base_url=settings.get("base_url"), 
                    chat_model=settings.get("chat_model"),
                    stop_check_fn=self.stop_check
                ).strip()
            
            # Execution Phase
            if "READ_RANDOM" in response:
                self.log("☁️ AI decided to read a document.")
                if self.event_bus:
                    self.event_bus.publish("DAYDREAM_ACTION", "Reading document...", source="Daydreamer")
                
                if self.document_store:
                    docs = self.document_store.list_documents(limit=100)
                    if docs:
                        selected_doc = None
                        
                        # 1. Try to find document by Topic
                        if topic:
                            # A. Filename match
                            topic_lower = topic.lower()
                            matches = [d for d in docs if topic_lower in d[1].lower()]
                            if matches:
                                selected_doc = random.choice(matches)
                                self.log(f"☁️ Found document matching topic '{topic}': {selected_doc[1]}")
                            else:
                                # B. Semantic search
                                try:
                                    emb = compute_embedding(topic, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
                                    # Search chunks to find relevant documents
                                    chunk_results = self.document_store.search_chunks(emb, top_k=3)
                                    if chunk_results:
                                        found_filenames = list(set([c['filename'] for c in chunk_results]))
                                        relevant_docs = [d for d in docs if d[1] in found_filenames]
                                        if relevant_docs:
                                            selected_doc = random.choice(relevant_docs)
                                            self.log(f"☁️ Found document semantically related to '{topic}': {selected_doc[1]}")
                                except Exception as e:
                                    self.log(f"⚠️ Topic search failed: {e}")

                        if not selected_doc:
                            selected_doc = random.choice(docs)
                            
                        doc_id, filename = selected_doc[0], selected_doc[1]
                        reading_filename = filename
                        
                        if self.event_bus:
                            self.event_bus.publish("DAYDREAM_READ", {"filename": filename}, source="Daydreamer")

                        # Get chunks
                        chunks = self.document_store.get_document_chunks(doc_id)
                        if chunks:
                            # Read a random sequential section (up to 3 chunks)
                            start_idx = random.randint(0, max(0, len(chunks) - 3))
                            reading_chunks = chunks[start_idx : start_idx + 3]
                            
                            # ISOLATION FIX: Reset context to ONLY the document to prevent hallucinated connections
                            context = f"Reading Document '{filename}':\n"
                            for c in reading_chunks:
                                context += f"- {c['text'][:400]}\n"
                            
                            # Generate thought based on document
                            doc_instruction = f"IMPORTANT: You are reading '{reading_filename}'. When referring to it, start with 'According to {reading_filename}...' or similar."
                            
                            daydream_prompt = (
                                "You are the AI Assistant reading a document from your library. "
                                "Review the Document Excerpts below. "
                                "Generate a new insight, hypothesis, refinement of a goal, or a personal preference based on this document. "
                                "If you find interesting information in the documents, create a new GOAL to study it further, extract a FACT, or form a PREFERENCE. "
                                "Focus ONLY on the document content. Do NOT connect to external topics unless they are general knowledge. "
                                f"{doc_instruction} "
                                "Do NOT repeat known facts. "
                                "Output ONLY the new thought/insight."
                            )
                            
                            messages = [{"role": "user", "content": context}]
                            
                            thought = run_local_lm(
                                messages,
                                system_prompt=daydream_prompt,
                                temperature=0.8,
                                max_tokens=300,
                                base_url=settings.get("base_url"),
                                chat_model=settings.get("chat_model"),
                                stop_check_fn=self.stop_check
                            )
                        else:
                            thought = "I tried to read a document but it was empty."
                    else:
                        thought = "I wanted to read, but the library is empty."
                else:
                    thought = "I wanted to read, but I have no document store."
            
            if self.stop_check():
                return

            # Safety check: If thought is still the command (execution failed or skipped), abort
            if "READ_RANDOM" in thought and len(thought) < 50:
                return

            # UX Improvement: Format JSON output for display if applicable
            display_thought = thought
            parsed_candidates = _parse_json_array_loose(thought)
            if parsed_candidates and isinstance(parsed_candidates, list) and len(parsed_candidates) > 0 and isinstance(parsed_candidates[0], dict):
                # Check if it looks like our memory structure
                if "type" in parsed_candidates[0] and "text" in parsed_candidates[0]:
                    lines = []
                    for item in parsed_candidates:
                        lines.append(f"• [{item.get('type', '?')}] {item.get('text', '')}")
                    display_thought = "\n".join(lines)

            self.log(f"☁️ Daydream thought: {display_thought}")
            
            if self.event_bus:
                self.event_bus.publish("DAYDREAM_THOUGHT", thought, source="Daydreamer")

            # Capture raw thought for Hod to analyze (e.g. for refutations)
            self.reasoning_store.add(content=f"Daydream Stream: {thought}", source="daydream_raw", confidence=1.0, ttl_seconds=3600)
            
            if self.chat_fn:
                self.chat_fn("Daydream", display_thought)
            
            # Pre-processing: Mask refuted beliefs to prevent re-extraction
            # Matches: "Refuting Belief [ID: 123]: <content> <newline/Revised>"
            # We replace the content with a placeholder so the extractor doesn't see it as a valid belief.
            extraction_text = thought
            if "Refuting Belief" in thought:
                extraction_text = re.sub(
                    r"(Refuting Belief \[ID: \d+\]:)(.*?)(?=\n|Revised Fact|$)", 
                    r"\1 [CONTENT REDACTED TO PREVENT RE-MEMORIZATION]", 
                    thought, 
                    flags=re.DOTALL | re.IGNORECASE
                )
                self.log(f"🛡️ Masked refuted belief text for extraction.")

            # 3. Process through pipeline (Extractor -> Reasoning -> Arbiter -> Memory)
            # Pass the belief text being analyzed so we can filter it out if the LLM repeats it
            ignored_text = None
            if analyzing_belief:
                ignored_text = belief_text

            self._process_thought(extraction_text, settings, reading_filename, enforce_citation=analyzing_belief, ignored_text=ignored_text)
            
            self.last_daydream_time = time.time()
            
        except Exception as e:
            self.log(f"❌ Daydream error: {e}")

    def _process_thought(self, thought: str, settings: Dict, source_filename: Optional[str] = None, enforce_citation: bool = False, ignored_text: Optional[str] = None):
        """Feed the thought back into the memory system."""
        try:
            if self.stop_check():
                return
            
            # Optimization: Try parsing as JSON first (for [READ_RANDOM] path)
            candidates = _parse_json_array_loose(thought)
            
            # Validate candidates structure
            valid_json = False
            if candidates and isinstance(candidates, list) and isinstance(candidates[0], dict):
                if "type" in candidates[0] and "text" in candidates[0]:
                    valid_json = True
            
            if not valid_json:
                # Fallback to extraction LLM call (for Reflection path or failed JSON)
                if self.status_fn:
                    self.status_fn("Extracting memories...")
                
                # Dynamic instruction to force citation
                instruction = DAYDREAM_INSTRUCTION
                if source_filename:
                    instruction += f" If the text is derived from '{source_filename}', append '[Source: {source_filename}]' if not already present. If the text is about a different topic or document, DO NOT append this source."

                candidates = extract_memory_candidates(
                    user_text="Internal Monologue",
                    assistant_text=thought,
                    base_url=settings.get("base_url"),
                    chat_model=settings.get("chat_model"),
                    embedding_model=settings.get("embedding_model"),
                    memory_extractor_prompt=settings.get("daydream_extractor_prompt", DAYDREAM_EXTRACTOR_PROMPT),
                    custom_instruction=instruction,
                    stop_check_fn=self.stop_check
                )

            if not candidates:
                self.log("☁️ Daydream extraction found no candidates.")
                return

            promoted = 0
            for c in candidates:
                # Heuristic: If source is known but not mentioned, append it.
                if source_filename and source_filename not in c["text"]:
                    # Automatically append source for content-derived types
                    if c.get("type") in ("FACT", "BELIEF", "PREFERENCE"):
                        c["text"] += f" [Source: {source_filename}]"
                    else:
                        # For GOALs or others, check triggers to avoid misattributing generic goals
                        lower_text = c["text"].lower()
                        # Check for generic references that imply a source
                        triggers = ["the document", "this document", "the study", "this study", "the research", "this research", 
                                    "the pdf", "this pdf", "the findings", "these findings", "this information", "this insight", 
                                    "these results", "the results", "the analysis", "this finding", "this paper", "the paper",
                                    "hypothesis", "hypotheses", "the article", "this article", "the text"]
                        if any(t in lower_text for t in triggers):
                            c["text"] += f" [Source: {source_filename}]"

                if self.stop_check():
                    return

                # Filter out meaningless facts that are just source citations
                if (c["text"].strip().startswith("[Source:") or c["text"].strip().startswith("[Supported by")) and len(c["text"].strip()) < 150:
                    continue

                # Enforce citation for facts generated during belief analysis
                if enforce_citation and c.get("type") == "FACT" and "[Source:" not in c["text"] and "[Supported by" not in c["text"]:
                    self.log(f"⚠️ Dropping uncited FACT from belief analysis: {c['text'][:50]}...")
                    continue

                # Prevent re-memorizing the belief we are currently analyzing
                if ignored_text and c.get("type") in ("BELIEF", "FACT"):
                    # Filter out meta-statements about the belief
                    if c["text"].lower().strip().startswith("the belief that"):
                        self.log(f"🛡️ Skipping meta-statement about belief: {c['text'][:50]}...")
                        continue

                    # Allow if it has a citation (improvement)
                    has_citation = "[Source:" in c["text"] or "[Supported by" in c["text"]

                    # Simple containment check or similarity check
                    # Check overlap. If it's a FACT/BELIEF that is just the ignored text (even with citation if it's a direct copy), we might want to be careful.
                    # But generally we allow cited versions. We block uncited duplicates.
                    if not has_citation and (ignored_text in c["text"] or c["text"] in ignored_text or len(set(c["text"].split()) & set(ignored_text.split())) / len(set(ignored_text.split())) > 0.8):
                        self.log(f"🛡️ Skipping re-memorization of analyzed belief: {c['text'][:50]}...")
                        continue

                # Reasoning layer
                self.reasoning_store.add(content=c["text"], source="daydream", confidence=0.9)

                # Arbiter promotion
                mid = self.arbiter.consider(
                    text=c["text"],
                    mem_type=c.get("type", "FACT"),
                    subject=c.get("subject", "User"),
                    confidence=0.85,
                    source="daydream"
                )
                
                if mid is not None:
                    promoted += 1
                    self.log(f"✅ [Daydream] Memory saved with ID: {mid}")
            
            if promoted:
                self.log(f"🧠 Promoted {promoted} memory item(s) from daydream.")

        except Exception as e:
            self.log(f"Daydream processing error: {e}")