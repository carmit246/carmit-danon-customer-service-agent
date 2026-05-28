# Customer Service Data Analyst Agent

A LangGraph-based ReAct agent that answers questions about the
[Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

**Authors:** Carmit Danon

---

## Architecture Overview

### Model Choice

| Role | Model | Reason |
|---|---|---|
| ReAct agent (main) | `Qwen/Qwen3-235B-A22B` | State-of-the-art reasoning, strong native tool-use / function-calling support, excellent instruction following. |
| Query router | `Qwen/Qwen3-32B` | Lighter and faster for the simple classification task; 3-class output (structured / unstructured / out_of_scope) needs no heavy reasoning. |

Both models are accessed exclusively via the **Nebius Token Factory** API.

### Graph Topology

```
START
  │
  ▼
router  ─── out_of_scope ──► END  (polite decline)
  │
  └── in_scope ──► agent ◄──────────────────────┐
                     │                            │
                     ├── tool_calls ──► tools ───┘
                     ├── done ──────────────────► END
                     └── max_iterations ──► fallback ──► END
```

**Nodes:**

| Node | Description |
|---|---|
| `router` | Classifies the query as *structured*, *unstructured*, or *out_of_scope* using a dedicated smaller LLM with `with_structured_output`. |
| `agent` | One LLM reasoning step; selects and calls tools (ReAct pattern). Uses a tailored system prompt depending on query type. |
| `tools` | LangGraph `ToolNode` — executes all tool calls returned by the agent and appends `ToolMessage` results to the state. |
| `fallback` | Fires when `iterations ≥ MAX_ITERATIONS` (12); emits a graceful message instead of spinning forever. |

### Tools

| Tool | Description |
|---|---|
| `get_categories` | List all unique top-level categories. |
| `get_intents` | List intents, optionally filtered to one category. |
| `count_records` | Count records matching category / intent filters. |
| `show_examples` | Return N sample records; supports keyword search. |
| `get_intent_distribution` | Count + percentage breakdown of intents within a category. |
| `get_sample_for_summary` | Retrieve a representative text sample for qualitative analysis. |

---

## Requirements

- **Python 3.10+**
- A [Nebius Token Factory](https://studio.nebius.com/) API key
- Internet access on first run (to download the dataset from HuggingFace, ~50 MB)

**Key dependencies** (see `requirements.txt` for pinned versions):

| Package | Purpose |
|---|---|
| `langgraph` | ReAct graph runtime |
| `langgraph-checkpoint-sqlite` | SQLite-backed persistent checkpointer |
| `langchain` / `langchain-openai` | LLM client & tool abstractions |
| `datasets` | HuggingFace dataset download & caching |
| `pandas` | In-memory dataset querying |
| `pydantic` | Tool input schemas |
| `rich` | Formatted CLI output |
| `python-dotenv` | `.env` file support |

---

## Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/<your-handle>/carmit-danon-customer-service-agent.git
cd carmit-danon-customer-service-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
# then edit .env and fill in your Nebius Token Factory key
```

`.env` (minimum):
```
NEBIUS_API_KEY=your-nebius-api-key-here
```

The dataset (~26 K records) is downloaded automatically from HuggingFace on first run
and cached locally by the `datasets` library.

---

## Running the CLI

```bash
python3 main.py                        # default session
python3 main.py --session my_session   # named, persistent session
python3 main.py --debug                # verbose reasoning logs
```

### Persistent sessions (Task 2a)

Conversation history is stored in `data/checkpoints.db` (SQLite).  Passing the
same `--session` name resumes the conversation exactly where you left off, even
after restarting the process:

```bash
# First run
python3 main.py --session alice
You ▶ Show me 3 examples from the REFUND category.
You ▶ exit

# Later — history is restored automatically
python3 main.py --session alice
You ▶ Show me 3 more.    ← agent remembers the previous results
```

### User profile (Task 2b)

After each turn the agent extracts key facts about the user (name, frequent
topics, preferences) and saves them to `data/profiles/<session_id>.json`.
The profile is injected into every system prompt so the agent can answer:

```
You ▶ My name is Alice and I'm mainly interested in refund data.
You ▶ What do you remember about me?
```

Profile files are human-readable JSON and persist across restarts independently
of the conversation checkpoint.

### Example CLI session

```
You ▶ What categories exist in the dataset?
You ▶ How many refund requests did we get?
You ▶ Show me 5 examples from the SHIPPING category.
You ▶ Show me 5 more.                            ← follow-up using history
You ▶ What is the distribution of intents in the ACCOUNT category?
You ▶ Summarize how agents respond to complaint intents.
You ▶ What do you remember about me?             ← answered from user profile
You ▶ What should I query next?                  ← recommendation (Bonus B)
You ▶ Who won the 2024 Champions League?         ← gracefully declined
```

The CLI prints:
- Query classification (structured / unstructured)
- Reasoning steps: every tool call and its result
- Final answer in a highlighted panel

---

## MCP Server (Task 3)

### Starting the server

```bash
# stdio transport (default — works with Claude Desktop and the MCP CLI)
python3 mcp_server.py

# HTTP / SSE transport — useful for programmatic clients
python3 mcp_server.py --sse
python3 mcp_server.py --sse --port 9000
```

The server exposes all 6 dataset tools:
`get_categories`, `get_intents`, `count_records`, `show_examples`,
`get_intent_distribution`, `get_sample_for_summary`.

### Connecting a client

**Interactive browser (MCP Inspector):**
```bash
npx @modelcontextprotocol/inspector python3 mcp_server.py
```

**Python async client (stdio):**
```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("python3 mcp_server.py") as client:
        # List available tools
        tools = await client.list_tools()
        print([t.name for t in tools])

        # Call a tool
        result = await client.call_tool("get_categories", {})
        print(result)

        result = await client.call_tool(
            "count_records", {"category": "REFUND"}
        )
        print(result)

asyncio.run(main())
```

**Python async client (SSE — requires server running with `--sse`):**
```python
async with Client("http://localhost:8000/sse") as client:
    result = await client.call_tool("get_categories", {})
    print(result)
```

---

## Streamlit UI (Bonus A)

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

Features:
- **Chat interface** with full session history rendered on every page load.
- **Reasoning steps** shown in collapsible expanders for each assistant turn.
- **Sidebar** with session ID input, live user profile display, and a clear button.
- **Query recommender** (Bonus B) built in — ask "What should I query next?".

---

## Query Recommender (Bonus B)

Works in both the CLI and the Streamlit UI.

```
You ▶ What should I query next?
  💡 Suggested: "Show me the distribution of intents in the REFUND category."
     Reply yes to run it, describe changes to refine it, or ask something else.

You ▶ I'd rather see examples instead.
  💡 Revised: "Show me 5 examples from the REFUND category."

You ▶ yes
  → Executing: Show me 5 examples from the REFUND category.
  [agent runs the query and displays results]
```

The recommender reads the full conversation history and user profile to pick
a contextually relevant next query.  All logic lives in `app/recommender.py`
and is shared between the CLI and the Streamlit UI.

---

## Project Structure

```
carmit-danon-customer-service-agent/
├── main.py              ← CLI (--session, --debug, Bonus B recommender)
├── mcp_server.py        ← FastMCP server (Task 3, --sse flag)
├── streamlit_app.py     ← Streamlit chat UI (Bonus A + Bonus B)
├── requirements.txt
├── .env.example
├── README.md
├── data/                ← runtime data (git-ignored)
│   ├── checkpoints.db   ← SQLite conversation history (all sessions)
│   └── profiles/        ← per-session user profile JSON files
└── app/
    ├── __init__.py
    ├── data_loader.py   ← loads & caches the Bitext dataset
    ├── tools.py         ← 6 LangChain tools with Pydantic schemas
    ├── router.py        ← query classifier (structured/unstructured/out_of_scope)
    ├── agent.py         ← LangGraph ReAct graph + AgentState
    ├── memory.py        ← SqliteSaver checkpointer factory (Task 2a)
    ├── profile.py       ← ProfileManager — distilled user facts (Task 2b)
    └── recommender.py   ← query recommendation engine (Bonus B)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEBIUS_API_KEY` | — | **Required.** Your Nebius Token Factory API key. |
| `NEBIUS_BASE_URL` | `https://api.studio.nebius.com/v1/` | Nebius API base URL. |
| `MAIN_MODEL` | `Qwen/Qwen3-235B-A22B` | Model for the ReAct reasoning loop. |
| `ROUTER_MODEL` | `Qwen/Qwen3-32B` | Model for query classification. |
