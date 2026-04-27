# Adaptive HybridRAG: Enhancing Graph-Vector Retrieval with LLM Self-Assessment and Dynamic Web Fallback

> Knowledge-Graph + Vector retrieval with LLM self-assessment and dynamic web fallback — substantially reducing hallucinations and graceful-degradation failures compared to single-mode RAG.

---

## Novelty

This implementation transforms HybridRAG into an autonomous researcher. By integrating LLM self-assessment and web-search fallback, the system dynamically identifies knowledge gaps in its internal database and fetches live external data, effectively eliminating hallucinations and dead-ends when local data lacks sufficient context.

---

## Proposed Tech Stack

- LLM Engine: Groq API (Free Tier) or Ollama (Llama-3)
- Embedding Model: HuggingFace (all-MiniLM-L6-v2)
- Vector DB: ChromaDB (Local)
- Graph DB: Neo4j Desktop (Local)
- Web Search: DuckDuckGo Search (LangChain Tool)
- Orchestration: LangGraph

---

## Proposed Problem Being Faced

**The Limitation of VectorRAG:** Dense vector retrieval searches for "strings, not things." While it is good at finding semantically similar text, it struggles severely with complex reasoning, multi-hop questions, and extracting explicit relationships between entities (e.g., financial data or complex company networks).

**Hallucinations & Missing Context:** Because standard RAG lacks structured factual boundaries, the LLM frequently hallucinates or gives factually incorrect information when asked domain-specific, complex questions. Furthermore, if the internal database doesn't have the answer, the system fails entirely.

---

## Novel Changes Proposed by Us

**Graph + Vector Fusion:** Moving away from single-dimensional retrieval. We are implementing a dual-retrieval system that cross-references dense vectors with structured knowledge triples.

**Adaptive Fallback:** Implementing an intelligent routing agent that can decide when to use local data vs. when to search the web, preventing the "I don't know" dead-ends common in standard chatbots.

---

## Proposed Solution (Our Novelty)

**The HybridRAG Approach:** We propose replacing standard RAG with a Hybrid retrieval system that integrates Knowledge Graphs (GraphRAG) with Vector Search (VectorRAG).

**How it resolves the issue:** Knowledge Graphs store data as strict, factual relationships (Node -> Edge -> Node). By combining structured Graph retrieval with flexible Vector retrieval, we get the best of both worlds: factual accuracy and semantic context. Additionally, we will introduce a web-search fallback if the internal knowledge base lacks the answer.

---

## What It Does

An offline ingestion pipeline turns a document corpus into **two parallel indexes**: a ChromaDB dense-vector store and a Neo4j knowledge graph (entities + typed relations extracted by an LLM). At query time a **LangGraph state machine** runs both retrievers in parallel, fuses their results, and passes the combined context to a **grader LLM** that decides whether the context is sufficient. If sufficient → generate. If poor → rewrite and retry. If off-topic or retries exhausted → escalate to **DuckDuckGo web search**. A final **hallucination check** verifies the answer is grounded in the retrieved context before returning.

---

## Architecture

```
  raw docs ──► loader ──► chunker ─┬──► embedder ──► ChromaDB (vectors)
                                   └──► LLMGraphTransformer ──► Neo4j (graph)

  query ──► query_analyzer ──► hybrid_retriever ──► relevance_grader
                                      ▲                    │
                                      │         sufficient │ poor │ off_topic
                                      │                    ▼      ▼
                                      │              generator  rewriter ──► (loop ≤ 3)
                                      │                    │
                                      └────── web_searcher ┘
                                                     │
                                         hallucination_checker ──► END
```

**Tech stack:**

| Component | Library |
|---|---|
| LLM (generation) | Groq `llama-3.3-70b-versatile` |
| LLM (grader / rewriter) | Groq `llama-3.1-8b-instant` |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB | ChromaDB (local, persistent) |
| Graph DB | Neo4j (Aura cloud or Desktop) |
| Web search | DuckDuckGo via LangChain |
| Orchestration | LangGraph |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## Prerequisites

- Python ≥ 3.10
- A free [Groq API key](https://console.groq.com)
- A Neo4j instance — either [Neo4j Aura Free](https://neo4j.com/cloud/platform/aura-graph-database/) (cloud) or [Neo4j Desktop](https://neo4j.com/download/) (local)

---

## Quickstart

### 1 — Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd NLP_Adap_RAG

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```dotenv
# Required
GROQ_API_KEY=gsk_...

# Neo4j Aura (cloud)
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=<instance-id>
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=<instance-id>

# Neo4j Desktop (local) — use instead of the above
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=your-password
# NEO4J_DATABASE=neo4j

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=adaptive-hybridrag
```

### 4 — Add your documents

Copy PDF, TXT, or HTML files into `data/raw/`:

```bash
mkdir -p data/raw
cp /path/to/your/documents/*.pdf data/raw/
```

### 5 — Ingest documents

```bash
# Ingest into both ChromaDB (vector) and Neo4j (graph)
python -m scripts.ingest data/raw

# Vector store only (no Neo4j required)
python -m scripts.ingest data/raw --vector-only

# Graph store only (vector already populated)
python -m scripts.ingest data/raw --graph-only

# Reduce batch size if Groq rate-limits during graph extraction
python -m scripts.ingest data/raw --batch-size 5
```

---

## Running the Application

### Option A — Streamlit UI (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The sidebar lets you toggle the agent route trace and retrieved context panels.

### Option B — CLI (one-shot)

```bash
python -m scripts.ask "What were Apple's revenues in FY2023?"

# Show retrieved graph facts and vector passages
python -m scripts.ask "What were Apple's revenues in FY2023?" --verbose
```

### Option C — CLI (interactive REPL)

```bash
python -m scripts.ask
```

Type your questions at the `>` prompt. Type `exit` or press `Ctrl-C` to quit.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite (30 tests) covers retrieval fusion logic, grader routing, and the end-to-end LangGraph workflow using lightweight mocks — no API keys or running databases required.

---

## Running the Evaluation Harness

Requires a populated vector store and Neo4j graph (run ingestion first).

```bash
# Full evaluation (15 Q/A pairs across single-hop, multi-hop, out-of-scope)
python -m eval.run_eval

# Filter to specific question types
python -m eval.run_eval --types single_hop multi_hop

# Custom Q/A set and output path
python -m eval.run_eval --qa-path eval/qa_set.jsonl --output eval/results.json
```

**Target metrics:**

| Metric | Target |
|---|---|
| Faithfulness | ≥ 0.85 |
| Answer relevancy | ≥ 0.80 |
| Context precision | ≥ 0.75 |
| Context recall | ≥ 0.75 |

---

## Configuration

All tuneable parameters live in `config.yaml` — no code changes needed:

```yaml
llm:
  provider: groq                        # groq | ollama
  generation_model: llama-3.3-70b-versatile
  fast_model: llama-3.1-8b-instant

vector_store:
  top_k: 6                              # documents retrieved per query

agent:
  max_iterations: 3                     # loop guard

web_search:
  max_results: 3
```

---

## Project Structure

```
NLP_Adap_RAG/
├── app.py                     # Streamlit UI
├── config.yaml                # all tuneable parameters
├── requirements.txt
├── .env.example               # credential template
├── data/
│   └── raw/                   # drop your PDFs / TXT here
├── src/
│   ├── config.py              # Pydantic AppConfig (reads config.yaml + .env)
│   ├── llm/provider.py        # Groq + Ollama abstraction
│   ├── ingestion/             # loader, chunker, vector_indexer, graph_indexer
│   ├── retrieval/             # vector, graph, web, hybrid (RRF fusion)
│   ├── graph_nodes/           # query_analyzer, grader, rewriter, generator, hallucination_checker
│   ├── agent/
│   │   ├── state.py           # GraphState TypedDict
│   │   └── workflow.py        # LangGraph assembly
│   └── utils/                 # prompts, logging
├── scripts/
│   ├── ingest.py              # CLI ingestion
│   └── ask.py                 # CLI query
├── tests/                     # 30 unit tests (no API keys required)
└── eval/
    ├── qa_set.jsonl           # 15 benchmark Q/A pairs
    └── run_eval.py            # RAGAS scoring harness
```

---

## Non-Goals

- No multi-user auth or hosted deployment — local single-user only.
- No non-English support.
- No real-time graph updates — ingestion is a batch job.
- No fine-tuning; off-the-shelf models only.

---

## References

| Resource | What it supplies |
|---|---|
| [HybridRAG paper (arXiv:2408.04948)](https://arxiv.org/abs/2408.04948) | Core approach: graph + vector fusion for financial documents |
| [WeKnow-RAG paper (arXiv:2408.07611)](https://arxiv.org/abs/2408.07611) | Web fallback + adaptive routing design |
| [LangGraph Agentic RAG example](https://github.com/langchain-ai/langgraph) | Reference implementation of the self-assessment grader loop |
| [HybridRAG-Bench](https://github.com/junhongmit/HybridRAG-Bench) | Benchmarking methodology |
| [NetApp hybrid-rag-graph-with-ai-governance](https://github.com/NetApp/hybrid-rag-graph-with-ai-governance) | Repo built around the HybridRAG arXiv:2408.04948 paper concepts |
| [tomasonjo/blogs](https://github.com/tomasonjo/blogs) | LangChain + Neo4j GraphRAG templates |
