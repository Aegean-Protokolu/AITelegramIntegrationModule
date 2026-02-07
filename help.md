# User Guide & Command Reference

Welcome to your AI Desktop Assistant. This is not just a chatbot; it is a cognitive architecture designed for long-term memory, autonomous research, and self-correction.

## 1. Core Architecture
The system is composed of several specialized agents working in concert:

*   **Decider (The Executive):** Controls the cognitive loop. It decides whether to chat, daydream, verify facts, or use tools. It manages system parameters like temperature and token limits.
*   **Netzach (The Observer):** A background process that monitors the conversation and internal state. It detects stagnation, loops, or the need for parameter adjustments and 'nudges' the Decider.
*   **Hod (The Analyst):** Runs post-process analysis. It critiques the Decider's actions, summarizes sessions, and identifies hallucinations or memory conflicts.
*   **Chokhmah (The Daydreamer):** Generates new insights by connecting existing memories or reading documents when the system is idle.
*   **Yesod (The Bridge):** Handles communication with the outside world (Telegram).

## 2. Modes of Operation
*   **Chat Mode:** Direct interaction. The AI prioritizes responding to you. Telegram bridge is active. Daydreaming is paused.
*   **Daydream Mode:** Autonomous background processing. The AI reads documents, consolidates memories, and thinks about active goals. Chat is available but secondary.

## 3. Memory System
Memories are stored in a vector database (FAISS + SQLite). They have types:

*   **IDENTITY:** Facts about you or the AI (e.g., 'User is a developer').
*   **FACT:** Objective truths derived from documents or conversation.
*   **GOAL:** Objectives the AI is working towards.
*   **BELIEF:** Hypotheses or opinions (subject to verification).
*   **NOTE:** Manual notes you explicitly save.

### Verification & Consolidation
The system automatically **consolidates** duplicate memories and **verifies** facts against source documents to prevent hallucinations.

## 4. Command Reference

### System Control
*   `/status` - Show current system state and activity.
*   `/stop` - Immediately halt current processing (Daydream/Verification).
*   `/terminate_desktop` - Close the application.
*   `/exitchatmode` - Disable Chat Mode and resume Daydreaming.

### Memory Management
*   `/memories` - List all active memories.
*   `/chatmemories` - List memories derived from chat (excludes daydreams).
*   `/metamemories` - Show the log of memory changes (Meta-Cognition).
*   `/memorystats` - Show statistics (verified counts, types).
*   `/consolidate` - Force a consolidation cycle to merge duplicates.
*   `/verify` - Run a batch verification of unverified memories.
*   `/verifyall` - Force verification of ALL memories against sources.
*   `/note [text]` - Save a permanent note manually.
*   `/notes` - List all manual notes.
*   `/clear_mem [ID]` - Delete a specific memory by ID.

### Document Management
*   `/documents` - List all uploaded documents.
*   `/doccontent "filename"` - Preview the content of a document.
*   `/removedoc "filename"` - Delete a document and its chunks.

### Decider & Automation
*   `/decider loop` - Start the autonomous Daydream Loop.
*   `/decider daydream` - Trigger a single Daydream cycle.
*   `/decider verify` - Trigger a verification batch.
*   `/decider up/down` - Manually adjust temperature.

### Cleanup & Reset (Use with Caution)
*   `/resetchat` - Clear the current conversation window history.
*   `/removesummaries` - Delete all session summaries.
*   `/resetmemory` - WIPE ALL MEMORIES (Irreversible).
*   `/resetall` - FACTORY RESET (Wipes everything).
*   `/remove[type]` - Remove all memories of a type (e.g., /RemoveGoal, /RemoveFact).

## 5. Natural Language Capabilities
You can ask the Decider to perform tasks directly in chat:

*   **Research:** "Learn about [Topic]" or "Expand knowledge about [Topic]" -> Starts a research loop.
*   **Analysis:** "Think about [Topic]" -> Starts a Chain of Thought analysis.
*   **Verification:** "Verify facts" or "Check sources" -> Triggers verification.
*   **Tools:** "Calculate 5*5", "Roll dice", "System info", "What time is it".

## 6. Troubleshooting
*   **If the AI is stuck repeating itself:** The Observer (Netzach) should catch this and increase temperature. You can also type `/decider up`.
*   **If the AI hallucinates:** Run `/verify` to check facts against documents.
*   **If processing is too slow:** Check `/status`. If a loop is running, use `/stop`.