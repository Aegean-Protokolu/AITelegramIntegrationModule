# 🤖 AI Telegram Integration Module 2.0

<div align="center">

  <img src="banner.png" width="1280" alt="AI Telegram Integration Module - Banner">

  <br />

</div>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

An advanced Telegram bot with **multi-layered memory architecture**, connecting a **local language model (LM Studio)** with intelligent memory management, reasoning, and consolidation systems.

---

## ⚡ Features

<details>
<summary>Click to expand Features</summary>

### 1. **Multi-Layered Memory System**
   - **Chat Memory**: Rolling conversation history (18 turns)
   - **Long-Term Memory**: Persistent SQLite-based memory store with versioning
   - **Reasoning Layer**: Semantic embedding-based hypothesis storage
   - **Meta-Memory**: Self-reflective memory tracking (tracks changes, consolidations)
   
### 2. **Memory Types (Hierarchical)**
   - **PERMISSION**: Explicit user grants (highest priority)
   - **RULE**: Behavior guidelines for assistant
   - **IDENTITY**: Names, roles, locations, religion
   - **PREFERENCE**: Likes/dislikes
   - **GOAL**: Aims/desires
   - **FACT**: Objective truths
   - **BELIEF**: Opinions/convictions (lowest priority)

### 3. **Intelligent Memory Management**
   - **Automatic Extraction**: LLM extracts durable memories from conversations
   - **Confidence Gating**: Type-specific confidence thresholds prevent noise
   - **Conflict Detection**: Detects and tracks contradictions
   - **Version Chaining**: Links memory updates via parent_id (append-only ledger)
   - **Smart Consolidation**: 
     - Automatic deduplication via semantic similarity
     - Substring detection for expansions (e.g., "Software Engineer" → "Software Engineer at Tech Company")
     - Type-specific thresholds (GOAL: 0.88, IDENTITY/BELIEF: 0.87, others: 0.93)
     - Periodic consolidation (every 10 minutes) + startup consolidation

### 4. **Advanced Command System (Slash Commands)**
   - **General Resets**: `/ResetChat`, `/ResetMemory`, `/ResetReasoning`, `/ResetMetaMemory`, `/ResetAll`
   - **Type-Specific Removal**: `/RemoveGoal`, `/RemoveIdentity`, `/RemoveFact`, `/RemoveBelief`, `/RemovePreference`, `/RemovePermission`, `/RemoveRule`
   - **Inspection**: `/Memories`, `/MetaMemories`
   - **Maintenance**: `/Consolidate`

### 5. **Memory Arbiter**
   - Autonomous gatekeeper between reasoning and long-term memory
   - Enforces admission rules (confidence thresholds, precedence)
   - Creates meta-memories for transparency
   - Prevents duplicate storage

### 6. **Local LM Integration**
   - Uses LM Studio models via local API (`v1/chat/completions`)
   - Embedding support for semantic similarity
   - Rolling context support with configurable turns
   - Memory-enhanced system prompts

### 7. **Safety Features**
   - Inactivity-based chat reset (30 minutes)
   - Generic assistant goal filtering (blocks "I'm here to help" pollution)
   - Low-quality candidate filtering (blocks greetings, questions, filler)
   - Append-only memory ledger (no data loss)

</details>

---

## 🏗 Architecture

<details>
<summary>Click to expand Architecture</summary>

### **Memory Flow**
```
User Message
    ↓
[Chat Memory] (rolling history)
    ↓
[LM Studio] (response generation)
    ↓
[Memory Extraction] (LLM-based candidate detection)
    ↓
[Reasoning Store] (semantic embeddings, TTL=1h)
    ↓
[Memory Arbiter] (confidence gating, conflict detection)
    ↓
[Long-Term Memory] (SQLite, versioned, append-only)
    ↓
[Memory Consolidator] (deduplication, similarity linking)
    ↓
[Meta-Memory] (change tracking, self-reflection)
```

### **File Structure**
- **`bot.py`**: Main loop, command handling, orchestration
- **`telegram_api.py`**: Telegram API wrapper (polling, message sending)
- **`lm.py`**: LM Studio integration, memory extraction, embeddings
- **`memory.py`**: Long-term memory store (SQLite, versioning)
- **`reasoning.py`**: Semantic reasoning store (embeddings, TTL)
- **`memory_arbiter.py`**: Admission control, conflict detection
- **`memory_consolidator.py`**: Deduplication, similarity linking
- **`meta_memory.py`**: Meta-cognition, change tracking
- **`config.py`**: Configuration (tokens, URLs, models)

</details>

---

## 🛠 Installation

<details>
<summary>Click to expand Installation</summary>

### 1. **Clone the repository**

```bash
git clone https://github.com/Aegean-E/AITelegramIntegrationModule.git
cd AITelegramIntegrationModule
```

### 2. **Install Python dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**
- `requests` (Telegram API, LM Studio)
- `numpy` (embeddings)
- `scikit-learn` (cosine similarity)

### 3. **Create a Telegram Bot and Get User ID**

```
3.1 - Open Telegram and search for @BotFather
3.2 - Send the command: /newbot
3.3 - Follow the prompts:
      - Give your bot a name
      - Give your bot a username (must end with `bot`)
3.4 - Save your Bot Token (e.g., 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ)
3.5 - Send a message to your new bot
3.6 - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
3.7 - Find your user ID in the response (under "chat": {"id": ...})
```

### 4. **Configure Your Bot (`config.py`)**

```python
# Telegram bot token from BotFather
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Your own chat/user ID (optional, can restrict bot to yourself)
CHAT_ID = int("YOUR_TELEGRAM_CHAT_ID")

# Local LM Studio API endpoint
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234"
CHAT_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"

# Model names (loaded in LM Studio)
CHAT_MODEL = "qwen2.5-vl-7b-instruct-abliterated"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"
```

### 5. **Run Your Local Server on LM Studio**

```
5.1 - Open LM Studio on your computer
5.2 - Load your desired chat model (e.g., Qwen 2.5 VL 7B)
5.3 - Load your embedding model (e.g., Nomic Embed Text v1.5)
5.4 - Start the local server (default: http://127.0.0.1:1234)
```

### 6. **Start the Bot**

**Option A: Command Line**
```bash
python bot.py
```

**Option B: Batch File (Windows)**
```bash
# Edit startbot.bat with your paths
# Then double-click startbot.bat
```

</details>

---

## 📖 Usage

<details>
<summary>Click to expand Usage</summary>

### **Basic Conversation**
Just message your bot naturally. It will:
- Remember facts about you (name, location, profession)
- Track preferences and beliefs
- Store permissions and rules
- Auto-consolidate duplicates

### **Slash Commands**

#### **General Management**
- `/ResetChat` - Clear conversation history
- `/ResetMemory` - Wipe long-term memory
- `/ResetReasoning` - Clear reasoning buffer
- `/ResetMetaMemory` - Clear meta-memories
- `/ResetAll` - Full reset (chat + memory + reasoning + meta)

#### **Type-Specific Removal**
- `/RemoveGoal` - Remove all GOAL memories
- `/RemoveIdentity` - Remove all IDENTITY memories
- `/RemoveFact` - Remove all FACT memories
- `/RemoveBelief` - Remove all BELIEF memories
- `/RemovePreference` - Remove all PREFERENCE memories
- `/RemovePermission` - Remove all PERMISSION memories
- `/RemoveRule` - Remove all RULE memories

#### **Inspection**
- `/Memories` - View saved memories (hierarchical display)
- `/MetaMemories` - View meta-memories (change history)

#### **Maintenance**
- `/Consolidate` - Manually trigger memory consolidation

### **Example Workflow**
```
You: My name is x
Bot: [Stores IDENTITY: User name is X]

You: I'm a worker at X.
Bot: [Stores IDENTITY: User is a worker at X.]

You: /Memories
Bot: Shows consolidated memories (no duplicates)
```

</details>

---

## 🧠 Memory System Details

<details>
<summary>Click to expand Memory System</summary>

### **Confidence Thresholds (Admission Gates)**
| Type | Minimum Confidence |
|------|-------------------|
| PERMISSION | 0.85 |
| RULE | 0.90 |
| IDENTITY | 0.80 |
| PREFERENCE | 0.60 |
| GOAL | 0.70 |
| FACT | 0.70 |
| BELIEF | 0.50 |

### **Consolidation Thresholds (Similarity)**
| Type | Threshold | Notes |
|------|-----------|-------|
| GOAL | 0.88 | Aggressive (filters generic "help" statements) |
| IDENTITY | 0.87 | Moderate (handles duplicates/expansions) |
| BELIEF | 0.87 | Moderate (handles duplicates/expansions) |
| PERMISSION | 0.87 | Moderate (handles similar grants) |
| FACT | 0.93 | Conservative (preserves unique info) |
| PREFERENCE | 0.93 | Conservative (preserves unique info) |
| RULE | 0.93 | Conservative (preserves unique info) |

### **Memory Extraction Filters**
**Excluded from extraction:**
- Pure greetings ("hi", "hello")
- Questions (ends with ?)
- Filler phrases ("how can I help", "what brings you here")
- Generic assistant goals ("I'm here to help", "feel free to ask")
- Contextual goals (e.g., "help with [current topic]")

### **Meta-Memory Events**
- `MEMORY_CREATED`: New memory stored
- `VERSION_UPDATE`: Memory updated (IDENTITY types)
- `SIMILARITY_LINK`: Similar memory linked (other types)
- `CONFLICT_DETECTED`: Contradiction found

</details>

---

## 🔧 Configuration

<details>
<summary>Click to expand Configuration</summary>

### **Tunable Parameters (bot.py)**
```python
MAX_TURNS = 18                          # Chat history turns
INACTIVITY_RESET_MINUTES = 30           # Auto-reset after inactivity
CONSOLIDATION_INTERVAL_SECONDS = 600    # Auto-consolidation interval (10 min)
```

### **Memory Thresholds (memory_arbiter.py)**
```python
CONFIDENCE_MIN = {
    "PERMISSION": 0.85,
    "RULE": 0.90,
    "IDENTITY": 0.80,
    # ... etc
}
```

### **Consolidation Thresholds (memory_consolidator.py)**
```python
# Adjust in consolidate() method:
threshold = 0.88 if type == 'GOAL' else 0.87 if type in (...) else 0.93
```

</details>

---

## 🚀 Advanced Features

<details>
<summary>Click to expand Advanced Features</summary>

### **Version Chaining**
Every memory update creates a new version linked via `parent_id`:
```
Memory ID 1: "Assistant name is Ada"
    ↓ (parent_id = 1)
Memory ID 5: "Assistant name is Lara"
```

Query old versions: `memory_store.get_memory_history(identity)`

### **Meta-Memory Self-Reflection**
The bot can reflect on its own changes:
```
Bot: "I used to be called Ada, but you renamed me to Lara on 2026-02-04 14:24"
```

### **Conflict Detection**
Detects contradictions (e.g., "User loves coffee" vs "User never drinks coffee")

### **Append-Only Ledger**
No memories are deleted - old versions are hidden but preserved for audit trails

</details>

---

## 📊 Database Schema

<details>
<summary>Click to expand Database Schema</summary>

### **memories table (memory.sqlite3)**
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    identity TEXT NOT NULL,        -- SHA256 hash for deduplication
    parent_id INTEGER,              -- Links to previous version
    type TEXT NOT NULL,             -- FACT | PREFERENCE | GOAL | etc.
    subject TEXT NOT NULL,          -- User | Assistant
    text TEXT NOT NULL,             -- Memory content
    confidence REAL NOT NULL,       -- Admission confidence
    source TEXT NOT NULL,           -- reasoning | user | assistant
    conflict_with TEXT,             -- JSON array of conflicting IDs
    created_at INTEGER NOT NULL     -- Unix timestamp
);
```

### **meta_memories table (meta_memory.sqlite3)**
```sql
CREATE TABLE meta_memories (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,      -- MEMORY_CREATED | VERSION_UPDATE | etc.
    memory_type TEXT NOT NULL,     -- IDENTITY | FACT | etc.
    subject TEXT NOT NULL,         -- User | Assistant
    text TEXT NOT NULL,            -- Human-readable description
    old_id INTEGER,                -- Reference to old memory
    new_id INTEGER,                -- Reference to new memory
    old_value TEXT,                -- Old value (extracted)
    new_value TEXT,                -- New value (extracted)
    metadata TEXT,                 -- JSON metadata
    created_at INTEGER NOT NULL    -- Unix timestamp
);
```

### **reasoning_nodes table (reasoning.sqlite3)**
```sql
CREATE TABLE reasoning_nodes (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,       -- JSON array of floats
    source TEXT,
    confidence REAL DEFAULT 0.5,
    created_at INTEGER NOT NULL,
    expires_at INTEGER             -- TTL support
);
```

</details>

---

## 🐛 Troubleshooting

<details>
<summary>Click to expand Troubleshooting</summary>

### **Bot Not Responding**
- Check LM Studio server is running (`http://127.0.0.1:1234`)
- Verify `BOT_TOKEN` in `config.py`
- Check console for errors

### **Memory Not Saving**
- Check confidence thresholds (may be too high)
- Look for "Arbiter" debug output in console
- Verify LLM is extracting candidates (check "Debug" logs)

### **Too Many Duplicate Memories**
- Run `/Consolidate` manually
- Lower consolidation interval in `bot.py`
- Check similarity thresholds in `memory_consolidator.py`

### **Generic Goals Polluting Memory**
- The system now filters these automatically
- Run `/RemoveGoal` to clean existing pollution
- Check `_is_low_quality_candidate()` in `lm.py`

</details>

---

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LM Studio](https://lmstudio.ai/) - Local LLM runtime
- [Telegram Bot API](https://core.telegram.org/bots/api) - Bot platform
- Qwen, Nomic Embed - Open-source models

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
