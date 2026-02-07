# AI Telegram Desktop Assistant

A standalone cognitive architecture desktop application that combines long-term memory, autonomous research (daydreaming), and multi-modal interaction via a local UI and Telegram. Designed to run locally with LM Studio.

## 🧠 Core Architecture

This project implements a multi-agent cognitive architecture where specialized modules work in concert to maintain state, pursue goals, and learn continuously. It is designed to be more than a chatbot; it is a persistent digital assistant.

### The Agents
1.  **Decider (`decider.py`)**: The Executive Function.
    - Controls the main cognitive loop.
    - Determines the next action: Chat, Daydream, Verify, or Tool Use.
    - Dynamically adjusts system parameters (Temperature, Max Tokens) based on context.
    - **Tools**: Can autonomously use Calculator, Clock, Dice, and System Info.
    
2.  **Netzach (`continuousobserver.py`)**: The Observer.
    - Runs in the background, monitoring logs and chat history.
    - Detects stagnation, repetitive loops, or "boredom".
    - Injects observations to "wake up" the Decider or suggest parameter changes.
    - **Capabilities**: Can increase temperature if output is rigid, or request Hod to summarize if logs get too long.
    
3.  **Hod (`hod.py`)**: The Analyst.
    - Performs post-process reflection.
    - Analyzes logs to identify hallucinations or conflicts.
    - Summarizes sessions into Meta-Memory to preserve context.
    - Can request the Decider to prune invalid memories.
    - **Self-Correction**: Can autonomously lower temperature or max tokens if instability is detected.
    
4.  **Chokhmah (`daydreaming.py`)**: The Daydreamer.
    - Operates during idle time (when not chatting).
    - Reads documents from the local library to generate new facts and goals.
    - Synthesizes existing memories to generate new insights.
    - **Modes**: `Auto` (Random), `Read` (Document focus), `Insight` (Memory synthesis).
    
5.  **Yesod (`telegram_api.py`)**: The Bridge.
    - Manages the connection to the outside world via Telegram.
    - Handles message polling and file downloads.

## 📂 Project Structure & File Descriptions

### Core Application
-   **`desktop_assistant.py`**: The main entry point. Initializes the UI (Tkinter), starts background threads (Daydreamer, Consolidator, Telegram Polling), and coordinates the components.
-   **`ui.py`**: Contains the `DesktopAssistantUI` mixin class. Handles all Tkinter widget setup, layout, and event binding to keep the main logic clean.
-   **`settings.json`**: Configuration file storing API keys, model settings, thresholds, and prompts.

### Cognitive Modules
-   **`decider.py`**: Implements the `Decider` class. Contains logic for task switching, tool execution, and strategic thinking chains.
-   **`continuousobserver.py`**: Implements `Netzach`. Uses a separate LLM call to monitor the "State of the World" and publish events to the `EventBus`.
-   **`hod.py`**: Implements `Hod`. Analyzes system logs and memories to ensure stability and coherence.
-   **`daydreaming.py`**: Implements `Daydreamer`. Handles the autonomous research loop and insight generation.

### Memory System
-   **`memory.py`**: The primary `MemoryStore`. Manages the SQLite database for long-term memories (`memories` table) and the FAISS vector index for semantic search.
-   **`meta_memory.py`**: The `MetaMemoryStore`. Tracks events *about* memories (creation, updates, conflicts) to enable self-reflection.
-   **`reasoning.py`**: The `ReasoningStore`. A transient buffer for working memory, hypotheses, and tool outputs. Items here expire after a TTL (Time-To-Live).
-   **`memory_arbiter.py`**: The `MemoryArbiter`. A gatekeeper that evaluates new information from the Reasoning layer against confidence thresholds and precedence rules before promoting it to Long-Term Memory.
-   **`memory_consolidator.py`**: The `MemoryConsolidator`. A background process that finds duplicate or related memories and links them (versioning) to prevent database bloat.

### Document Handling (RAG)
-   **`document_store_faiss.py`**: Manages the storage of document chunks and their embeddings. Uses FAISS for fast retrieval.
-   **`document_processor.py`**: Handles file ingestion. Extracts text from PDF (via PyMuPDF) and DOCX, cleans it, and splits it into semantic chunks with overlap.

### Infrastructure
-   **`lm.py`**: The interface for the Local LLM (LM Studio). Handles API requests, vision payloads, and memory extraction prompts.
-   **`telegram_api.py`**: Low-level wrapper for Telegram Bot API requests.
-   **`event_bus.py`**: A simple Publish/Subscribe system that allows agents (Netzach, Decider, Hod) to communicate asynchronously without tight coupling.

## ✨ Key Features

### 1. Multi-Layered Memory
-   **Chat Memory**: Short-term rolling context (last 20 messages).
-   **Long-Term Memory**: Permanent storage for Facts, Goals, and Identity.
-   **Meta-Memory**: Logs of *why* and *when* memories changed.
-   **Reasoning Store**: Temporary storage for thoughts and tool outputs (TTL-based).

### 2. Autonomous Daydreaming
When not chatting, the AI enters "Daydream Mode". It will:
-   Select a random document or memory.
-   Generate new insights or hypotheses.
-   Verify existing beliefs against documents.
-   Consolidate duplicate information.

### 3. Self-Correction & Verification
The system includes a **Verifier** loop. It periodically checks "BELIEF" and "FACT" memories against source documents. If a memory is unsupported or hallucinated, it is removed, and a correction is logged in Meta-Memory.

### 4. Multi-Modal Support
You can send images via the Desktop UI or Telegram. The system uses vision-capable models (like `qwen2.5-vl`) to analyze them.

### 5. Tools
The Decider can autonomously execute:
-   **Calculator**: For math operations.
-   **Clock**: To check current time.
-   **Dice**: For RNG.
-   **System Info**: To check host machine stats.

## 🛠️ Installation

### Prerequisites
1.  **Python 3.10+** recommended.
2.  **LM Studio** (or compatible OpenAI-API local server):
    - **Chat Model**: Load a smart instruction-tuned model (e.g., `Qwen2.5-VL-7B-Instruct`).
    - **Embedding Model**: Load a text embedding model (e.g., `Nomic-Embed-Text-v1.5`).
    - **Server**: Start the server on port `1234`.
3.  **Telegram Bot** (Optional):
    - Get a token from @BotFather.
    - Get your Chat ID (the bot prints this in logs when you message it).

### Setup
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install requests numpy faiss-cpu ttkbootstrap PyMuPDF python-docx Pillow pyinstaller
    ```
3.  Run the application:
    ```bash
    python desktop_assistant.py
    ```

## ⚙️ Configuration

Configuration is managed via `settings.json` or the **Settings Tab** in the UI.

| Category | Setting | Description | Default |
| :--- | :--- | :--- | :--- |
| **API** | `bot_token` | Telegram Bot Token | "" |
| | `chat_id` | Your Telegram User ID | "" |
| | `base_url` | LLM Server URL | `http://127.0.0.1:1234/v1` |
| **Models** | `chat_model` | Model identifier string | `qwen2.5-vl...` |
| | `embedding_model` | Embedding identifier string | `nomic-embed...` |
| **System** | `ai_mode` | Startup mode ("Chat" or "Daydream") | "Daydream" |
| | `telegram_bridge_enabled` | Enable/Disable Telegram | `true` |
| | `theme` | UI Theme (Darkly, Cosmo, Cyborg) | "Darkly" |
| **Generation** | `temperature` | Creativity (0.0 - 2.0) | 0.7 |
| | `max_tokens` | Response length limit | 800 |
| | `system_prompt` | Core personality instructions | (See file) |
| **Memory** | `daydream_cycle_limit` | Cycles before pausing loop | 10 |
| | `consolidation_thresholds` | Similarity required to merge | (Dict) |

## 📦 Building an Executable (.exe)

To create a standalone executable:

1.  `pip install pyinstaller`
2.  `pyinstaller --noconsole --onefile --name "AI_Assistant" desktop_assistant.py`
3.  Copy `dist/AI_Assistant.exe` to your project folder.
4.  Ensure `settings.json` and the `data/` folder are next to the `.exe`.

## 🎮 Usage Guide

### Modes
-   **Chat Mode**: The AI pauses background research to respond instantly to you.
-   **Daydream Mode**: The AI runs background loops. It might be reading or thinking. It will still reply to chat, but might finish its current thought first.

### Commands
You can control the system via natural language or slash commands.

#### Natural Language
-   "Research [topic]" -> Starts a Daydream loop on that topic.
-   "Think about [topic]" -> Starts a Chain of Thought analysis.
-   "Verify your facts" -> Runs the Verifier.
-   "Summarize this session" -> Runs Hod's summarization.
-   "Calculate 5 * 5" -> Uses Calculator tool.
-   "Roll a dice" -> Uses Dice tool.

#### Slash Commands
**System Control**
-   `/status`: View system state.
-   `/stop`: Halt current processing.
-   `/terminate_desktop`: Close the application.
-   `/exitchatmode`: Disable Chat Mode and resume Daydreaming.
-   `/decider [action]`: Manual control (e.g., `/decider up`, `/decider loop`).

**Memory Management**
-   `/memories`: List active memories.
-   `/chatmemories`: List memories derived from chat only.
-   `/metamemories`: Show memory logs (Meta-Cognition).
-   `/memorystats`: Show statistics (verified counts, types).
-   `/note [text]`: Save a permanent note manually.
-   `/notes`: List all manual notes.
-   `/consolidate`: Force memory consolidation.
-   `/verify`: Run verification batch.
-   `/verifyall`: Force verification of ALL memories against sources.

**Documents**
-   `/documents`: List uploaded files.
-   `/doccontent "filename"`: Preview document content.
-   `/removedoc "filename"`: Delete a document.

**Maintenance**
-   `/removesummaries`: Delete all session summaries.
-   `/consolidatesummaries`: Merge multiple summaries into one.
-   `/resetall`: **Factory Reset** (Wipes DB).
-   `/resetchat`: Clear current chat window history.

## 🖥️ UI Features

-   **Chat Tab**: Main interface. Includes "Netzach Observations" panel at the bottom.
-   **Logs Tab**: Real-time system logs.
-   **Memory Database Tab**:
    -   View Memories, Summaries, and Meta-Memories.
    -   **Export Summaries**: Save session history to text file.
    -   **Compress**: Manually trigger summary consolidation.
-   **Documents Tab**:
    -   Upload PDF/DOCX.
    -   Search/Filter documents.
    -   **Check Integrity**: Find and fix broken document chunks.

## 📄 License
MIT