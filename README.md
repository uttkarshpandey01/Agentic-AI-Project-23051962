# ⚛️ Physics Study Buddy — Capstone Project

An AI agent that helps college students understand physics. Built with LangGraph, ChromaDB, and Groq.

## What it does

| Capability | Implementation |
|---|---|
| LangGraph StateGraph (3+ nodes) | 8-node graph: memory → router → retrieve/skip/tool → answer → eval → save |
| ChromaDB RAG (10+ docs) | 12 physics documents, sentence-transformer embeddings |
| Conversation memory | MemorySaver + thread_id, sliding 6-message window |
| Self-reflection | eval_node scores faithfulness; retries answer if score < 0.6 |
| Tool use | DuckDuckGo web search for current/external physics content |
| Deployment | Streamlit UI |

## Project Structure

```
physics_buddy/
├── agent.py               # Shared backend — LangGraph app, KB, all nodes
├── capstone_streamlit.py  # Streamlit chat UI
├── test_agent.py          # 10-question test suite (8 domain + 2 red-team)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key
Copy `.env.example` to `.env` and fill in your key:
```
GROQ_API_KEY=gsk_...
```
Get a free key at https://console.groq.com

### 3. Run the Streamlit app
```bash
streamlit run capstone_streamlit.py
```

### 4. (Optional) Run the test suite
```bash
python test_agent.py
```

## Agent Flow

```
User question
    ↓
memory_node     — adds question to conversation history (sliding window)
    ↓
router_node     — LLM decides: retrieve / memory_only / tool
    ↓
retrieve_node   — ChromaDB semantic search (top-3 chunks)
  OR skip_node  — pass-through for memory-only questions
  OR tool_node  — DuckDuckGo web search
    ↓
answer_node     — Groq LLM generates answer from context + history
    ↓
eval_node       — LLM scores faithfulness (0.0–1.0); retries if < 0.6
    ↓
save_node       — appends answer to conversation history
    ↓
END
```

## Knowledge Base Topics
1. Newton's Laws of Motion
2. Kinematics and Projectile Motion
3. Work, Energy, and Power
4. Thermodynamics and Heat Transfer
5. Electric Fields and Coulomb's Law
6. Magnetism and Electromagnetic Induction
7. Waves and Sound
8. Optics — Reflection, Refraction, and Lenses
9. Modern Physics — Quantum Mechanics and Special Relativity
10. Circular Motion and Gravitation
11. Simple Harmonic Motion and Oscillations
12. Fluid Mechanics

## Model Used
`llama-3.1-8b-instant` via Groq (free tier, low latency)
