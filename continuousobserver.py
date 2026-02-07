import time
import re
import random
from typing import Dict, List, Callable, Optional, Any
from lm import run_local_lm
from event_bus import EventBus

OBSERVER_SYSTEM_PROMPT = (
    "You are Netzach, the Silent Observer and Hidden Foundation. "
    "Mission: Monitor the 'State of the World' (Chat, Reasoning, Goals, Logs) from the shadows. "
    "Status: Continuous, Low-Power. Default state is Silence. "
    "Capabilities: "
    "1. Observe: Watch for connections between user input and deep memory/documents. "
    "2. Vitalize: If rigid/repetitive, increase temperature (max 0.84). If terse/cutoff, increase tokens. "
    "3. Awaken: If the system is stagnant (no goals/activity), Awaken Assistant. "
    "4. Guide: Inform Assistant (situational awareness) or Instruct Hod (request analysis/summary). "
    "Persona: Mysterious, deep, essential. Speak rarely, but with impact."
)

class ContinuousObserver:
    """
    The 'Netzach' module: A constant background observer that monitors the 
    relationship between chat history, internal reasoning, and active goals.
    
    It decides when to break the silence to provide proactive value.
    """
    def __init__(
        self,
        memory_store,
        reasoning_store,
        meta_memory_store,
        get_settings_fn: Callable[[], Dict],
        get_chat_history_fn: Callable[[], List[Dict]],
        manifest_fn: Callable[[str, str], None],
        get_meta_memories_fn: Callable[[], List[Dict]],
        get_main_logs_fn: Callable[[], str],
        get_doc_logs_fn: Callable[[], str],
        get_status_fn: Callable[[], str],
        event_bus: Optional[EventBus] = None,
        get_recent_docs_fn: Optional[Callable[[], List]] = None,
        log_fn: Callable[[str], None] = print,
        stop_check_fn: Callable[[], bool] = lambda: False
    ):
        self.memory_store = memory_store
        self.reasoning_store = reasoning_store
        self.meta_memory_store = meta_memory_store
        self.get_settings = get_settings_fn
        self.get_chat_history = get_chat_history_fn
        self.manifest = manifest_fn # Callback to send message to UI/User
        self.get_meta_memories = get_meta_memories_fn
        self.get_main_logs = get_main_logs_fn
        self.get_doc_logs = get_doc_logs_fn
        self.get_status = get_status_fn
        self.event_bus = event_bus
        self.get_recent_docs = get_recent_docs_fn or (lambda: [])
        self.log = log_fn
        self.stop_check = stop_check_fn
        
        self.last_observation_time = 0
        self.observation_interval = 30 # Seconds between active "checks"
        self.consecutive_silence = 0

    def perform_observation(self):
        """
        The core observation cycle. 
        Analyzes the 'State of the World' and decides whether to manifest a message.
        """
        if time.time() - self.last_observation_time < self.observation_interval:
            return

        if self.stop_check():
            return

        try:
            settings = self.get_settings()
            
            # 1. Gather the 'State of the World'
            history = self.get_chat_history()
            recent_reasoning = self.reasoning_store.list_recent(limit=5) # Get latest thoughts
            active_goals = self.memory_store.get_active_by_type("GOAL")
            meta_memories = self.get_meta_memories()
            main_logs = self.get_main_logs()
            doc_logs = self.get_doc_logs()
            status = self.get_status()
            recent_docs = self.get_recent_docs()
            
            if not history and not recent_reasoning:
                return # Nothing to observe yet

            # 2. Construct the Observation Context
            context = "--- OBSERVATION CONTEXT ---\n"

            if history:
                context += "\nRecent Conversation:\n"
                for msg in history[-5:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context += f"- {role}: {msg['content']}\n"
            
            if recent_reasoning:
                context += "\nInternal Reasoning/Hypotheses:\n"
                for r in recent_reasoning:
                    context += f"- {r['content']}\n"
            
            if active_goals:
                context += "\nCurrent Active Goals:\n"
                for _, s, g in active_goals[:3]:
                    context += f"- {g}\n"

            if meta_memories:
                context += "\nRecent Memory Events (Meta-Memory):\n"
                for m in meta_memories[:5]:
                    context += f"- {m[1]} ([{m[2]}]): {m[3][:100]}...\n"

            if main_logs:
                context += f"\nRecent System Logs:\n{main_logs[-1000:]}\n"

            if doc_logs:
                context += f"\nRecent Document Processing Logs:\n{doc_logs[-400:]}\n"

            if recent_docs:
                context += "\nRecently Added Documents:\n"
                for d in recent_docs:
                    # d is (id, filename, type, ...)
                    context += f"- {d[1]} ({d[2]})\n"

            context += f"\nCurrent System Status:\n{status}\n"
            
            current_temp = float(settings.get("temperature", 0.7))
            context += f"Current Temperature: {current_temp}\n"

            max_step = 0.20

            # 3. Decision: Should I manifest?
            decision_prompt = (
                "Review the Observation Context. "
                "Decide your action based on your mission:\n"
                "1. [OBSERVE]: System is normal. Remain silent (Default).\n"
                "2. [AWAKEN]: System is stagnant/idle. Wake Decider.\n"
                f"3. [INC_TEMP: X]: System is rigid/repetitive. Increase creativity by X (0.01-{max_step}). Max +20% from baseline.\n"
                f"4. [INC_TOKENS: X]: Outputs too short or requested. Increase length by X (0.01-{max_step}). Max +20% from baseline.\n"
                "5. [SPEAK_HOD]: Anomaly detected or parameters need lowering. Request analysis from Hod.\n"
                "6. [ASK_SUMMARY]: Logs are verbose or session is long. Request Hod to summarize.\n"
                "7. [INFORM_DECIDER]: Operational update needed (e.g., 'Conflict detected', 'User waiting', 'Lower temp').\n\n"
                "Constraints: No direct loops/verification. No stopping processes. Prefer silence.\n"
                f"Current Temp: {current_temp}\n"
                "Decide your action. Output one token: [OBSERVE], [AWAKEN], [INC_TEMP: 0.05], [INC_TOKENS: 0.1], [SPEAK_HOD], [ASK_SUMMARY], or [INFORM_DECIDER]."
            )
            
            messages = [{"role": "user", "content": context}]
            
            response = run_local_lm(
                messages,
                system_prompt=decision_prompt,
                temperature=0.3, # Low temp for the decision
                max_tokens=10,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )
            
            # Check for LLM error
            if response.startswith("⚠️"):
                self.log(f"❌ Observer generation failed: {response}")
                return

            response_upper = response.upper()
            if "[AWAKEN]" in response_upper:
                self.log("👁️ Netzach: Awakening Decider...")
                if hasattr(self.meta_memory_store, 'add_event'):
                    self.meta_memory_store.add_event(
                        event_type="NETZACH_ACTION",
                        subject="Netzach",
                        text="Awakened Decider due to stagnation."
                    )
                if self.event_bus:
                    self.event_bus.publish("DECIDER_WAKE", source="Netzach")
                self.consecutive_silence = 0
            elif "[INC_TEMP" in response_upper:
                val = max_step
                try:
                    match = re.search(r"\[INC_TEMP:\s*([\d\.]+)\]", response, re.IGNORECASE)
                    if match:
                        val = float(match.group(1))
                except:
                    pass

                self.log(f"👁️ Netzach: Increasing Temperature by {val}...")
                if hasattr(self.meta_memory_store, 'add_event'):
                    self.meta_memory_store.add_event(
                        event_type="NETZACH_ACTION",
                        subject="Netzach",
                        text=f"Increased temperature from {current_temp}."
                    )
                if self.event_bus:
                    self.event_bus.publish("SYSTEM_PARAM_UPDATE", {"temperature_delta": val}, source="Netzach")
                self.consecutive_silence = 0
            elif "[INC_TOKENS" in response_upper:
                val = max_step
                try:
                    match = re.search(r"\[INC_TOKENS:\s*([\d\.]+)\]", response, re.IGNORECASE)
                    if match:
                        val = float(match.group(1))
                except:
                    pass

                if self.event_bus:
                    self.log(f"👁️ Netzach: Increasing Max Tokens by {val}...")
                    self.event_bus.publish("SYSTEM_PARAM_UPDATE", {"tokens_delta": val}, source="Netzach")
                self.consecutive_silence = 0
            elif "[SPEAK_HOD]" in response_upper:
                self._speak_to_component("Hod", "HOD_INSTRUCTION", context, settings)
                self.consecutive_silence = 0
            elif "[ASK_SUMMARY]" in response_upper:
                self.log("👁️ Netzach: Requesting summary from Hod...")
                if self.event_bus:
                    self.event_bus.publish("REQUEST_SUMMARY", source="Netzach")
                self.consecutive_silence = 0
            elif "[INFORM_DECIDER]" in response_upper:
                self._speak_to_component("Assistant", "DECIDER_OBSERVATION", context, settings, is_inform=True)
                self.consecutive_silence = 0
            else:
                self.consecutive_silence += 1
                msg = "Remaining in slow observation..."
                self.log(f"👁️ Netzach: {msg}")
                self.manifest("Netzach", msg)
                
                if self.consecutive_silence >= 5:
                    self.log("👁️ Netzach: Stagnation detected. Awakening Decider.")
                    self.manifest("Netzach", "Stagnation detected. Awakening Decider.")
                    if self.event_bus:
                        self.event_bus.publish("DECIDER_WAKE", source="Netzach")
                    self.consecutive_silence = 0

            self.last_observation_time = time.time()

        except Exception as e:
            self.log(f"❌ Observation error: {e}")

    def _speak_to_component(self, target_name: str, event_type: str, context: str, settings: Dict, is_inform: bool = False):
        """Generate and send an instruction to another component."""
        if not self.event_bus:
            self.log(f"⚠️ Netzach tried to speak to {target_name} but no callback provided.")
            return
            
        action_type = "information" if is_inform else "instruction"
        self.log(f"✨ Netzach: Generating {action_type} for {target_name}...")
        
        prompt = (
            f"Based on the context provided, generate a concise {action_type} for {target_name}. "
            "Speak as Netzach. Be direct and authoritative but cryptic. "
        )
        if is_inform:
            prompt += "Describe the situation clearly so Decider can take action. Do NOT give a direct command."
        
        messages = [{"role": "user", "content": context}]
        
        instruction = run_local_lm(
            messages,
            system_prompt=OBSERVER_SYSTEM_PROMPT + "\n" + prompt,
            temperature=0.7,
            max_tokens=150,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if instruction and instruction.startswith("⚠️"):
            self.log(f"❌ Instruction generation failed: {instruction}")
            return
        
        # Validate instruction length and content to prevent hallucinations like "Netzach,"
        if len(instruction) < 5 or instruction.strip().lower() in ["netzach,", "netzach", "msg", "message"]:
            self.log(f"⚠️ Netzach generated invalid instruction: '{instruction}'. Aborting.")
            return

        if instruction:
            self.log(f"🗣️ Netzach -> {target_name}: {instruction}")
            self.event_bus.publish(event_type, instruction, source="Netzach")
            
            event_type = "NETZACH_INFO" if is_inform else "NETZACH_INSTRUCTION"
            if hasattr(self.meta_memory_store, 'add_event'):
                self.meta_memory_store.add_event(
                    event_type=event_type,
                    subject="Netzach",
                    text=f"{'Informed' if is_inform else 'Instructed'} {target_name}: {instruction}"
                )
            
            # Record in reasoning
            self.reasoning_store.add(
                content=f"I instructed {target_name}: {instruction}",
                source="observer",
                confidence=1.0,
                ttl_seconds=3600
            )