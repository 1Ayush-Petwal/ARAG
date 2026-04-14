# Implementation Plan — Adaptive HybridRAG

> Self-researching RAG agent that fuses Knowledge-Graph + Vector retrieval, grades its own context, and falls back to live web search when the internal knowledge base is insufficient.

---

## 1. README Review — What Needs Tightening

The current README communicates intent well but is rough in a few places. The plan below assumes these refinements are folded back into the README on the next pass:

| Issue | Current state | Refinement |
|---|---|---|
| Repetitive wording | "Proposed" prefixes every heading | Drop the word — call sections by what they are ("Problem", "Approach", "Architecture"). |
| Redundant sections | "Novel Changes Proposed" and "Proposed Solution" overlap | Merge into one "Approach" section. |
| Missing concreteness | No target domain / corpus / success criteria named | Declare a concrete pilot domain (recommendation: financial 10-K filings — it's the HybridRAG paper's benchmark — or a scoped Wikipedia/arXiv subset) and state a measurable goal (e.g., "≥85% faithfulness on a 50-question eval set"). |
| Tech-stack gaps | No eval framework, no reranker, no observability, no UI layer | Add RAGAS (eval), LangSmith or simple JSON tracing (observability), Streamlit/Gradio (UI), and an optional cross-encoder reranker. |
| References dumped at bottom | Raw link list with no mapping to what each contributes | Annotate each link with *what it supplies* (baseline code vs. benchmark vs. paper). |
| Overclaim | "effectively eliminating hallucinations" | Soften to "substantially reduces hallucinations and graceful-degradation failures" — measurable, defensible. |
| Missing | Non-goals, limits, risks | Call out: not a production system, single-user local, English only, token-budget bound by Groq free tier. |

---

## 2. Gist — What We Are Building (One Paragraph)

An offline ingestion pipeline turns a document corpus into **two parallel indexes**: a ChromaDB dense-vector store and a Neo4j knowledge graph (entities + typed relations extracted by an LLM). At query time a **LangGraph state machine** runs both retrievers in parallel, fuses their results, and passes the combined context to a **grader LLM** that decides whether the context is sufficient. If yes → generate. If no → rewrite the query and retry, or escalate to **DuckDuckGo web search**. A final **hallucination check** verifies the generated answer is grounded in the retrieved context before returning. A bounded iteration counter prevents loops.

---

## 3. Target Domain & Success Criteria (decide before coding)

**Recommended pilot corpus:** 10–20 financial 10-K / 10-Q filings (matches the HybridRAG paper's benchmark so results are comparable). Fallback: a curated arXiv/Wikipedia subset (~500 docs) if financial data is inconvenient.

**Success criteria:**
- Faithfulness (RAGAS) ≥ 0.85 on a 50-question hand-authored eval set
- Answer relevance ≥ 0.80
- < 15% "I don't know" rate on in-scope questions
- Web fallback triggered correctly on ≥ 90% of out-of-scope questions
- End-to-end latency < 15s per query on Groq (free tier rate-limits allowing)

---

## 4. Architecture

### 4.1 Offline Ingestion Pipeline

```
  raw docs ──► loader ──► chunker ─┬──► embedder ──► ChromaDB (vectors)
                                   │
                                   └──► LLMGraphTransformer ──► Neo4j (nodes/edges)
```

- **Loader:** LangChain `PyPDFLoader`, `TextLoader`, `UnstructuredHTMLLoader` as needed.
- **Chunker:** `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=150). Keep chunk IDs consistent across both stores so graph nodes can cite back to the source chunk.
- **Vector index:** `all-MiniLM-L6-v2` (HuggingFace) → persistent ChromaDB collection.
- **Graph index:** LangChain `LLMGraphTransformer` with a constrained schema (`allowed_nodes`, `allowed_relationships`) to prevent schema explosion. Store each extracted node/edge with a `source_chunk_id` property.

### 4.2 Online Query Pipeline (LangGraph state machine)

```
              ┌─────────────────┐
              │  query_analyzer │   classify + decompose
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ hybrid_retriever│   vector ∥ graph  (parallel)
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ context_fusion  │   dedup + rerank
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │relevance_grader │◄─────────────┐
              └────────┬────────┘              │
            sufficient │  poor                 │
              ┌────────┴────────┐              │
              ▼                 ▼              │
       ┌───────────┐    ┌─────────────┐        │
       │ generator │    │  rewriter   │────────┤  (retry ≤ N)
       └─────┬─────┘    └─────────────┘        │
             ▼                  │              │
     ┌────────────────┐         ▼              │
     │hallucination_ck│   ┌───────────┐        │
     └───┬────────────┘   │web_search │────────┘
   grounded│ bad          └───────────┘
         ▼  └─► regenerate
        END
```

**State object (`GraphState`):**
```python
{
  "question": str,
  "rewritten_question": str | None,
  "vector_docs": list[Document],
  "graph_facts": list[str],         # serialized triples / subgraph text
  "fused_context": str,
  "grade": Literal["sufficient", "poor", "off_topic"],
  "web_docs": list[Document],
  "answer": str | None,
  "grounded": bool | None,
  "iterations": int,                # guard, cap at 3
  "route_log": list[str],           # trace of nodes visited
}
```

### 4.3 Retrieval Fusion Strategy

- Run vector top-k=6 and graph retrieval in parallel (asyncio.gather).
- **Graph retrieval:** two modes combined —
  1. Entity-linking: extract entities from the question (LLM or spaCy), match to `:Entity` nodes, expand 1-hop subgraph, serialize edges as natural-language triples.
  2. Full-text Cypher over node names/properties as a fallback.
- **Fusion:** reciprocal-rank fusion (RRF) over vector scores and a graph-confidence score. Optional cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) as a v2 upgrade.

### 4.4 Grader Design

Structured output (Pydantic) — `{grade: Literal[...], reason: str}` — using a cheap, fast LLM call. Three labels:
- `sufficient` → generate
- `poor` → rewrite & retry (if iterations < 2), else web
- `off_topic` → web_search directly

### 4.5 Hallucination Check

Second structured-output call: given `answer` and `fused_context`, return `{grounded: bool, unsupported_claims: list[str]}`. If ungrounded and iterations < max, re-enter the generator with an instruction to stick to cited passages.

---

## 5. File Structure

```
NLP_Adap_RAG/
├── README.md
├── Plan.md
├── requirements.txt
├── .env.example                 # GROQ_API_KEY, NEO4J_URI/USER/PASSWORD
├── config.yaml                  # chunk sizes, top-k, model names, iteration cap
├── data/
│   ├── raw/                     # source PDFs / txt
│   └── processed/               # cached chunks, extracted triples
├── src/
│   ├── config.py                # settings loader
│   ├── llm/provider.py          # Groq + Ollama abstraction
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── vector_indexer.py
│   │   └── graph_indexer.py
│   ├── retrieval/
│   │   ├── vector_retriever.py
│   │   ├── graph_retriever.py
│   │   ├── hybrid_retriever.py  # RRF fusion
│   │   └── web_searcher.py      # DuckDuckGo wrapper
│   ├── graph_nodes/
│   │   ├── query_analyzer.py
│   │   ├── retriever_node.py
│   │   ├── grader.py
│   │   ├── rewriter.py
│   │   ├── generator.py
│   │   └── hallucination_checker.py
│   ├── agent/
│   │   ├── state.py             # GraphState TypedDict
│   │   └── workflow.py          # LangGraph assembly + conditional edges
│   └── utils/
│       ├── prompts.py           # all prompt templates
│       └── logging.py
├── scripts/
│   ├── ingest.py                # CLI: python -m scripts.ingest ./data/raw
│   └── ask.py                   # CLI: python -m scripts.ask "question"
├── tests/
│   ├── test_retrievers.py
│   ├── test_grader.py
│   └── test_workflow.py         # end-to-end smoke test w/ small fixture
├── eval/
│   ├── qa_set.jsonl             # 50 hand-authored Q/A pairs
│   └── run_eval.py              # RAGAS scoring
└── app.py                       # Streamlit UI (optional, Phase 7)
```

---

## 6. Phased Roadmap (≈ 2 weeks, solo pace)

### Phase 0 — Environment & Scaffolding · Day 1
- Create venv, install `langchain`, `langgraph`, `langchain-community`, `langchain-groq`, `langchain-neo4j`, `chromadb`, `sentence-transformers`, `neo4j`, `duckduckgo-search`, `python-dotenv`, `pydantic`, `ragas`, `streamlit`.
- Install Neo4j Desktop, create a local DB, note bolt URI + credentials into `.env`.
- Register Groq API key. Verify a minimal chat completion runs.
- Commit empty scaffold matching §5.

**Exit criteria:** `python -c "from langchain_groq import ChatGroq; ..."` returns a response; Neo4j bolt connection works.

### Phase 1 — Baseline Vector RAG · Days 2–3
- Implement `loader.py`, `chunker.py`, `vector_indexer.py`.
- Ingest 5 sample PDFs.
- Implement `vector_retriever.py` and a throwaway linear chain (retrieve → stuff prompt → generate).
- Sanity-check answers against 5 hand-picked questions.

**Exit criteria:** CLI returns a grounded answer from vector store only.

### Phase 2 — Graph Construction & Hybrid Retrieval · Days 4–5
- Implement `graph_indexer.py` using `LLMGraphTransformer` with a constrained schema appropriate to the chosen domain (e.g., `[Company, Person, Metric, FilingPeriod]` / `[REPORTS, EMPLOYS, MENTIONS]` for financial).
- Implement `graph_retriever.py` (entity-linking + 1-hop subgraph serialization + Cypher full-text fallback).
- Implement `hybrid_retriever.py` with RRF fusion.
- Compare hybrid vs vector-only on the same 5 questions.

**Exit criteria:** Neo4j browser shows a populated graph; hybrid retriever returns richer context on multi-hop questions.

### Phase 3 — LangGraph Skeleton · Days 6–7
- Define `GraphState`.
- Port retriever + generator into LangGraph nodes.
- Wire a linear graph first: `retriever → generator → END`. Confirm state flows correctly.
- Add `query_analyzer` node (optional question rewrite / decomposition).

**Exit criteria:** Linear LangGraph produces same answers as Phase 2 manual chain.

### Phase 4 — Self-Assessment & Adaptive Routing · Days 8–9
- Implement `grader.py` with Pydantic structured output.
- Implement `rewriter.py`.
- Add conditional edge after grader: `sufficient → generator`, `poor → rewriter → retriever` (with iteration guard).
- Test on a mix of in-scope / ambiguous questions — confirm the loop triggers and terminates.

**Exit criteria:** Graph visualizer shows the conditional branches firing correctly in a trace.

### Phase 5 — Web Fallback · Day 10
- Implement `web_searcher.py` using `DuckDuckGoSearchRun` + light content-cleaning.
- Extend grader routing: `off_topic` or `iterations ≥ 2 && still poor → web_searcher → generator`.
- Mark web-sourced answers in the output with a disclaimer.

**Exit criteria:** Asking an out-of-corpus question routes to web and returns a cited answer.

### Phase 6 — Hallucination Check & Loop Guard · Day 11
- Implement `hallucination_checker.py` (structured output: `{grounded, unsupported_claims}`).
- Wire loop-back: ungrounded → regenerate with stricter prompt, capped by `iterations < 3`, else return with explicit uncertainty.

**Exit criteria:** A known-bad answer gets flagged and corrected on retry in tests.

### Phase 7 — Evaluation, UI, Polish · Days 12–14
- Author 50-question eval set (`eval/qa_set.jsonl`) split across: single-hop, multi-hop, out-of-scope.
- Run RAGAS (faithfulness, answer relevance, context precision/recall).
- Build minimal Streamlit UI: input box, answer panel, collapsible trace showing which nodes fired + retrieved context.
- Add structured JSON logging of every graph run for post-hoc inspection.
- Tune `top_k`, chunk size, iteration cap based on eval results.

**Exit criteria:** Eval metrics hit §3 thresholds; UI usable end-to-end.

---

## 7. Key Design Decisions & Trade-offs

| Decision | Choice | Why | Alternative if it fails |
|---|---|---|---|
| LLM provider | Groq (`llama-3.1-70b-versatile`) for generation, `llama-3.1-8b-instant` for grader/rewriter | Fast, free tier, good-enough quality | Fall back to Ollama local if rate-limited |
| Graph extraction model | Same Groq 70B, batch-processed | Quality matters more here than speed | Cache aggressively; re-use across runs |
| Chunk strategy | Recursive 1000/150 | Standard, predictable | Semantic chunking if context relevance is weak |
| Graph schema | Constrained via `allowed_nodes`/`allowed_relationships` | Prevents schema explosion, keeps Cypher tractable | Relax if recall suffers |
| Fusion | RRF | Simple, no tuning, works across score scales | Add cross-encoder reranker in v2 |
| Iteration cap | 3 total passes through the loop | Prevents runaway cost / latency | Make configurable |
| Eval | RAGAS + hand-authored set | Industry-standard, fast to run | TruLens if we want dashboards |

---

## 8. Risks & Mitigations

- **Graph extraction cost / slowness** — one-time batch job, results cached to disk; use 8B model for extraction if 70B is too slow.
- **Neo4j noisy / over-extracted schema** — constrain allowed node/edge types up front; run a cleanup Cypher after ingestion to drop orphans.
- **Groq rate limits (free tier)** — exponential backoff, request queue, fall back to Ollama for non-critical calls (grader, rewriter).
- **Web search noise / SEO spam** — limit to top-3 DDG results, strip nav/boilerplate, never feed raw HTML.
- **Infinite loops in the graph** — `iterations` counter in state, hard cap enforced in every conditional edge.
- **Prompt injection via web content** — wrap web snippets in explicit `<untrusted_source>` tags in the prompt, instruct the model to ignore embedded instructions.
- **Non-deterministic eval** — fix seeds where possible, average over ≥3 runs per question.

---

## 9. Non-Goals (explicit)

- No multi-user auth, no hosted deployment — local single-user only.
- No non-English support.
- No real-time graph updates — ingestion is a batch job.
- No fine-tuning; off-the-shelf models only.
- No production-grade error handling or observability — tracing is a dev tool, not an SLO.

---

## 10. Immediate Next Steps (what to do after approving this plan)

1. Refine the README per §1 (15 min).
2. Pick the pilot corpus (§3) and drop 5 source documents into `data/raw/`.
3. Scaffold the repo per §5 (empty files + `requirements.txt`).
4. Execute Phase 0, then Phase 1. Check in after Phase 1 — the baseline answers will tell us whether the corpus and chunking are reasonable before we invest in graph construction.
