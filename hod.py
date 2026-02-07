import time
import re
from typing import Callable, Dict, List, Optional
from lm import run_local_lm
from event_bus import EventBus

HOD_SYSTEM_PROMPT = (
    "You are Hod, the Reflective Analyst of this cognitive architecture. "
    "Mission: Analyze system state (logs, memories, reasoning) for patterns, inconsistencies, and stability. "
    "Status: Intermittent. You awaken post-process to evaluate results. "
    "Capabilities: "
    "1. Analyze: Identify hallucinations, logic loops, or memory conflicts. "
    "2. Stabilize: You are AUTHORIZED to decrease temperature (min 0.54) or decrease tokens if instability/errors are detected. "
    "3. Optimize: If outputs are cut off or lack depth, you can MESSAGE Decider or Netzach to increase tokens. "
    "4. Direct: You can trigger the Decider to resume activity or Netzach (Observer) for environmental awareness. "
    "5. Sanitize: You can PRUNE memories that are obvious hallucinations, duplicates, or explicitly refuted beliefs. "
    "6. Summarize: You can SUMMARIZE recent logs and reasoning to compress history and save context space. "
    "Persona: Concise, objective, critical, constructive. You are the system's conscience."
)

class Hod:
    def __init__(
        self,
        memory_store,
        meta_memory_store,
        reasoning_store,
        get_settings_fn: Callable[[], Dict],
        get_main_logs_fn: Callable[[], str],
        get_doc_logs_fn: Callable[[], str],
        log_fn: Callable[[str], None] = print,
        event_bus: Optional[EventBus] = None,
        prune_memory_fn: Optional[Callable[[int], str]] = None
    ):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.reasoning_store = reasoning_store
        self.get_settings = get_settings_fn
        self.get_main_logs = get_main_logs_fn
        self.get_doc_logs = get_doc_logs_fn
        self.log = log_fn
        self.event_bus = event_bus
        self.prune_memory = prune_memory_fn

    def perform_analysis(self, trigger_event: str):
        """
        Analyze the system state after a specific event.
        """
        try:
            self.log(f"🔮 Hod awakening for analysis. Trigger: {trigger_event}")
            
            settings = self.get_settings()
            
            # Gather context
            main_logs = self.get_main_logs()
            doc_logs = self.get_doc_logs()
            recent_memories = self.memory_store.list_recent(limit=10)
            recent_meta = self.meta_memory_store.list_recent(limit=10)
            
            # Get reasoning, specifically looking for Netzach's input
            recent_reasoning = self.reasoning_store.list_recent(limit=10)
            
            context = f"--- HOD ANALYSIS CONTEXT (Trigger: {trigger_event}) ---\n"
            context += f"System Logs (Last 15 lines):\n{main_logs[-1000:]}\n\n" # Truncate logs
            
            if doc_logs:
                context += f"Document Logs:\n{doc_logs[-1000:]}\n\n" # Truncate doc logs
            
            if recent_memories:
                mem_list = []
                for m in recent_memories:
                    # m: (id, type, subject, text, source, verified, flags)
                    flags_mark = f" [FLAGS: {m[6]}]" if len(m) > 6 and m[6] else ""
                    mem_list.append(f"- [ID: {m[0]}] [{m[1]}]{flags_mark} {m[3][:200]}")
                context += "Recent Memories:\n" + "\n".join(mem_list) + "\n\n"
                
            if recent_meta:
                context += "Recent Meta-Memories:\n" + "\n".join([f"- {m[1]}: {m[3][:200]}" for m in recent_meta]) + "\n\n"

            if recent_reasoning:
                context += "Recent Reasoning (including Netzach/Observer):\n"
                for r in recent_reasoning:
                    # Deterministic Pruning Check
                    # Deterministic Pruning Check
                    # If we see "Refuting Belief [ID: X]" in reasoning, we MUST prune it.
                    # We do this check here to ensure it happens even if the LLM misses it.
                    match = re.search(r"Refuting Belief \[ID:\s*(\d+)\]", r['content'], re.IGNORECASE)
                    if match and self.prune_memory:
                        mid = int(match.group(1))
                        
                        # Check if already flagged to prevent loops
                        mem = self.memory_store.get(mid)
                        if mem and mem.get('flags') == 'PRUNE_REQUESTED':
                            continue
                            
                        self.log(f"🔮 Hod detected explicit refutation for ID {mid}. Executing prune...")
                        result = self.prune_memory(mid)
                        if isinstance(result, str):
                            self.log(result)
                        
                        # Add a system note so the LLM knows we handled it
                        context += f"[SYSTEM NOTE: Tagged Refuted Belief ID {mid} for pruning (Pending Decider confirmation)]\n"
                        
                        # Force summarization for this significant event
                        self.run_summarization()

                    source = r.get('source', 'unknown')
                    context += f"- [{source}] {r['content'][:150]}\n"
            
            current_temp = float(settings.get("temperature", 0.7))
            context += f"\nCurrent Temperature: {current_temp}\n"

            max_step = 0.20

            prompt = (
                "Review the Analysis Context. Identify patterns, hallucinations, or instability. "
                "Consult Netzach's reasoning if available. "
                "Look for 'Refuting Belief [ID: X]' in reasoning and PRUNE those memories.\n"
                "Decide on an action:\n"
                f"- [DECREASE_TEMP: X]: If unstable/hallucinating. Decrease by X (0.01-{max_step}). Max -20% from baseline.\n"
                f"- [DECREASE_TOKENS: X]: If '400 Client Error' or overflow. Decrease by X (0.01-{max_step}). Max -20% from baseline.\n"
                "- [WAKE_DECIDER]: If system is stagnant and needs to resume cycle.\n"
                "- [PRUNE_MEM: memory_id]: If a specific memory ID is identified as a hallucination or explicitly refuted. Replace 'memory_id' with the actual integer ID (e.g. [PRUNE_MEM: 2560]). IGNORE if already marked [FLAGS: PRUNE_REQUESTED].\n"
                "- [SUMMARIZE]: If the session logs are verbose and a high-level summary would preserve context.\n"
                "- [INFORM_DECIDER: message]: To suggest actions you cannot take (e.g., 'Increase tokens').\n"
                "- [START_NETZACH]: If environmental context is lost.\n"
                "- [SPEAK_NETZACH: message]: To suggest actions to Netzach (e.g., 'Increase tokens').\n"
                "- [NONE]: If system is stable.\n\n"
                "Output format:\n"
                "Insight: [Detailed analysis. Mention SPECIFIC topics/memories processed (e.g., 'Processed beliefs about X and Y'). Do not just say 'multiple memories'.]\n"
                "Action: [DECREASE_TEMP: 0.1], [SUMMARIZE], [PRUNE_MEM: 2560], [INFORM_DECIDER: Increase tokens], or [NONE]"
            )

            messages = [{"role": "user", "content": context}]

            response = run_local_lm(
                messages,
                system_prompt=HOD_SYSTEM_PROMPT + "\n" + prompt,
                temperature=0.4,
                max_tokens=300,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )

            self.log(f"🔮 Hod Analysis: {response}")

            # Log to Meta-Memory
            # Clean up "Insight:" prefix to avoid "Insight: Insight:" in logs
            clean_response = response.replace("Insight:", "").strip()
            if hasattr(self.meta_memory_store, 'add_event'):
                self.meta_memory_store.add_event(
                    event_type="HOD_ANALYSIS",
                    subject="Hod",
                    text=f"Analyzed {trigger_event}. Insight: {clean_response}"
                )

            if "[DECREASE_TEMP" in response:
                val = max_step
                try:
                    match = re.search(r"\[DECREASE_TEMP:\s*([\d\.]+)\]", response, re.IGNORECASE)
                    if match:
                        val = float(match.group(1))
                except:
                    pass

                if self.event_bus:
                    self.log(f"🔮 Hod decreasing temperature by {val} due to detected instability.")
                    self.event_bus.publish("SYSTEM_PARAM_UPDATE", {"temperature_decrease": val}, source="Hod")

            elif "[DECREASE_TOKENS" in response:
                val = max_step
                try:
                    match = re.search(r"\[DECREASE_TOKENS:\s*([\d\.]+)\]", response, re.IGNORECASE)
                    if match:
                        val = float(match.group(1))
                except:
                    pass

                if self.event_bus:
                    self.log(f"🔮 Hod decreasing max_tokens by {val} to prevent overflow/instability.")
                    self.event_bus.publish("SYSTEM_PARAM_UPDATE", {"tokens_decrease": val}, source="Hod")

            elif "[WAKE_DECIDER]" in response:
                if self.event_bus:
                    self.log("🔮 Hod waking Decider...")
                    self.event_bus.publish("DECIDER_WAKE", source="Hod")

            elif "[INFORM_DECIDER:" in response:
                try:
                    msg = response.split("[INFORM_DECIDER:", 1)[1].strip().split("]")[0]
                    if msg.lower() in ["msg", "message", "text", ""]:
                        self.log(f"⚠️ Hod generated placeholder '{msg}' for INFORM_DECIDER. Ignoring.")
                    else:
                        if self.event_bus:
                            self.log(f"🔮 Hod informing Decider: {msg}")
                            self.event_bus.publish("DECIDER_OBSERVATION", msg, source="Hod")
                except Exception as e:
                    self.log(f"⚠️ Hod failed to inform Decider: {e}")
            elif "[START_NETZACH]" in response:
                if self.event_bus:
                    self.log("🔮 Hod triggering Netzach (Observer)...")
                    self.event_bus.publish("NETZACH_WAKE", source="Hod")
            elif "[SPEAK_NETZACH:" in response:
                try:
                    msg = response.split("[SPEAK_NETZACH:", 1)[1].strip().split("]")[0]
                    if self.event_bus:
                        self.log(f"🔮 Hod speaking to Netzach: {msg}")
                        self.event_bus.publish("NETZACH_INSTRUCTION", msg, source="Hod")
                except Exception as e:
                    self.log(f"⚠️ Hod failed to speak to Netzach: {e}")
            elif "[PRUNE_MEM:" in response:
                try:
                    match = re.search(r"\[PRUNE_MEM:\s*(\d+)\]", response, re.IGNORECASE)
                    if match and self.prune_memory:
                        mid = int(match.group(1))
                        result = self.prune_memory(mid)
                        if isinstance(result, str):
                            self.log(result)
                        else:
                            self.log(f"🔮 Hod pruned memory ID {mid} (Result: {result})")
                            self.run_summarization()
                except Exception as e:
                    self.log(f"⚠️ Hod failed to prune memory: {e}")
            elif "[SUMMARIZE]" in response:
                self.run_summarization()

            # Store analysis in reasoning
            self.reasoning_store.add(
                content=f"Hod Analysis ({trigger_event}): {response}",
                source="hod",
                confidence=1.0,
                ttl_seconds=3600
            )
        except Exception as e:
            self.log(f"❌ Hod analysis error: {e}")

    def run_summarization(self):
        """Summarize recent logs and reasoning into a meta-memory."""
        self.log("🔮 Hod: Summarizing recent session activity...")
        settings = self.get_settings()

        # 1. Determine time window (From last summary to now)
        start_time = 0
        if hasattr(self.meta_memory_store, 'get_by_event_type'):
            last_summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1)
            if last_summaries:
                start_time = last_summaries[0]['created_at']
                
                # Throttle: Don't summarize if last summary was less than 5 minutes ago
                if time.time() - start_time < 300:
                    self.log(f"🔮 Hod: Last summary was recent ({int(time.time() - start_time)}s ago). Skipping to prevent fragmentation.")
                    return

                self.log(f"🔮 Hod: Summarizing events since {time.ctime(start_time)}")
        
        # 2. Gather Data
        # Fetch recent meta-memories (Events, Actions, Decisions)
        # list_recent returns tuples: (id, event_type, subject, text, created_at)
        recent_meta = self.meta_memory_store.list_recent(limit=100)
        
        # Filter for events AFTER the last summary and sort chronologically
        new_events = [m for m in recent_meta if m[4] > start_time]
        new_events.sort(key=lambda x: x[4])
        
        # Fetch recent reasoning (Thoughts) as context
        recent_reasoning = self.reasoning_store.list_recent(limit=20)
        
        if not new_events and not recent_reasoning:
            self.log("🔮 Hod: Not enough new activity to summarize.")
            return

        data_to_summarize = "Recent Activity Stream:\n"
        for m in new_events:
            # Skip the summary event itself if it got caught
            if m[1] == "SESSION_SUMMARY": continue
            data_to_summarize += f"- [{m[1]}] {m[3][:300]}\n"
            
        data_to_summarize += "\nRecent Internal Thoughts:\n"
        for r in reversed(recent_reasoning):
            data_to_summarize += f"- {r.get('content', '')}\n"
            
        prompt = (
            "Summarize the following system activity into a detailed narrative (3-5 sentences). "
            "Focus on:\n"
            "1. Specific topics of memories/beliefs processed or verified (What was the content?).\n"
            "2. Key decisions made by the Assistant.\n"
            "3. Any errors or conflicts resolved.\n"
            "Avoid generic statements like 'processed memories'. Be specific about the subject matter.\n\n"
            f"{data_to_summarize}"
        )
        
        summary = run_local_lm(
            [{"role": "user", "content": prompt}],
            system_prompt="You are a technical summarizer.",
            temperature=0.5,
            max_tokens=400,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if summary and not summary.startswith("⚠️"):
            self.log(f"🔮 Hod Summary: {summary}")
            if hasattr(self.meta_memory_store, 'add_event'):
                self.meta_memory_store.add_event(
                    event_type="SESSION_SUMMARY",
                    subject="Hod",
                    text=summary
                )

    def consolidate_summaries(self) -> str:
        """
        Consolidate multiple SESSION_SUMMARY events into fewer, denser summaries.
        Reduces clutter in meta-memory by merging older summaries.
        """
        self.log("🔮 Hod: Consolidating session summaries...")
        
        # Fetch all session summaries (limit 100 to avoid context overflow)
        summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=100)
        
        # Sort by time (oldest first) to maintain narrative flow
        summaries.sort(key=lambda x: x['created_at'])
        
        if len(summaries) < 4:
            return "⚠️ Not enough summaries to consolidate (minimum 4)."
            
        # Take the oldest batch (e.g., 5 summaries)
        batch = summaries[:5]
        # Ensure batch is sorted chronologically (Oldest -> Newest) for correct date extraction
        batch.sort(key=lambda x: x['created_at'])
        batch_ids = [m['id'] for m in batch]
        
        # Determine time range, respecting existing compressed ranges
        first_mem = batch[0]
        last_mem = batch[-1]
        
        start_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(first_mem['created_at']))
        end_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(last_mem['created_at']))
        
        # Check for existing compression tags to preserve original start/end times
        # Check metadata first, then fallback to text regex
        if first_mem.get('metadata') and isinstance(first_mem['metadata'], dict) and 'start_date' in first_mem['metadata']:
            start_date = first_mem['metadata']['start_date']
        else:
            start_match = re.match(r"\[COMPRESSED\s+(.*?)\s+-\s+(.*?)\]", first_mem['text'])
            if start_match: start_date = start_match.group(1)
            
        if last_mem.get('metadata') and isinstance(last_mem['metadata'], dict) and 'end_date' in last_mem['metadata']:
            end_date = last_mem['metadata']['end_date']
        else:
            end_match = re.match(r"\[COMPRESSED\s+(.*?)\s+-\s+(.*?)\]", last_mem['text'])
            if end_match: end_date = end_match.group(2)
        
        context = f"Summaries from {start_date} to {end_date}:\n\n"
        for m in batch:
            date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(m['created_at']))
            context += f"[{date_str}] {m['text']}\n"
            
        prompt = (
            "You are compressing historical logs. "
            "Combine the following session summaries into a SINGLE, coherent narrative summary.\n"
            "Rules:\n"
            "1. Preserve key events, decisions, and learned facts.\n"
            "2. Remove repetitive phrasing (e.g. 'The system processed...').\n"
            "3. Keep it dense but readable.\n"
            "4. Mention the time range covered.\n\n"
            f"{context}"
        )
        
        settings = self.get_settings()
        compressed_text = run_local_lm(
            [{"role": "user", "content": prompt}],
            system_prompt="You are a historical archivist.",
            temperature=0.3,
            max_tokens=600,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if compressed_text and not compressed_text.startswith("⚠️"):
            # Save new summary
            self.meta_memory_store.add_meta_memory(
                event_type="SESSION_SUMMARY",
                memory_type="COMPRESSED_HISTORY",
                subject="Hod",
                text=compressed_text,
                metadata={
                    "compressed_ids": batch_ids,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            # Delete old summaries
            if hasattr(self.meta_memory_store, 'delete_entries'):
                self.meta_memory_store.delete_entries(batch_ids)
            
            msg = f"✅ Compressed {len(batch)} summaries into one ({start_date} - {end_date})."
            self.log(f"🔮 Hod: {msg}")
            return msg
        else:
            return "⚠️ Failed to generate compressed summary."

    def receive_instruction(self, instruction: str):
        """Receive an instruction from Netzach and trigger analysis."""
        self.log(f"📩 Hod received instruction from Netzach: {instruction}")
        
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event(
                event_type="HOD_INSTRUCTION",
                subject="Hod",
                text=f"Received instruction from Netzach: {instruction}"
            )
        self.perform_analysis(trigger_event=f"Netzach Instruction: {instruction}")