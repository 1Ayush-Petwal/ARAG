# Adaptive HybridRAG: Enhancing Graph-Vector Retrieval with LLM Self-Assessment and Dynamic Web Fallback

## Novelty

This implementation transforms HybridRAG into an autonomous researcher. By integrating LLM self-assessment and web-search fallback, the system dynamically identifies knowledge gaps in its internal database and fetches live external data, effectively eliminating hallucinations and dead-ends when local data lacks sufficient context.


## Proposed Tech Stack:
- LLM Engine: Groq API (Free Tier) or Ollama (Llama-3)

- Embedding Model: HuggingFace (all-MiniLM-L6-v2)

- Vector DB: ChromaDB (Local)

- Graph DB: Neo4j Desktop (Local)

- Web Search: DuckDuckGo Search (LangChain Tool)

- Orchestration: LangGraph

## Proposed Problem Being Faced

The Limitation of VectorRAG: Dense vector retrieval searches for "strings, not things." While it is good at finding semantically similar text, it struggles severely with complex reasoning, multi-hop questions, and extracting explicit relationships between entities (e.g., financial data or complex company networks).

Hallucinations & Missing Context: Because standard RAG lacks structured factual boundaries, the LLM frequently hallucinates or gives factually incorrect information when asked domain-specific, complex questions. Furthermore, if the internal database doesn't have the answer, the system fails entirely.

## Novel Changes Proposed by Us

Graph + Vector Fusion: Moving away from single-dimensional retrieval. We are implementing a dual-retrieval system that cross-references dense vectors with structured knowledge triples.

Adaptive Fallback: Implementing an intelligent routing agent that can decide when to use local data vs. when to search the web, preventing the "I don't know" dead-ends common in standard chatbots.

## Proposed Solution (Our Novelty)

The HybridRAG Approach: We propose replacing standard RAG with a Hybrid retrieval system that integrates Knowledge Graphs (GraphRAG) with Vector Search (VectorRAG).

How it resolves the issue: Knowledge Graphs store data as strict, factual relationships (Node -> Edge -> Node). By combining structured Graph retrieval with flexible Vector retrieval, we get the best of both worlds: factual accuracy and semantic context. Additionally, we will introduce a web-search fallback if the internal knowledge base lacks the answer.

## Referenced Resources: 

Novelty Implementation (Web Fallback & LLM Assessment)
The creators of LangChain have recently released official boilerplate code for the WeKnow-RAG concepts under their LangGraph framework:

- https://github.com/langchain-ai/langgraph (Agentic RAG Example): If you look in the LangGraph examples repository (/examples/rag/langgraph_agentic_rag.ipynb), they provide the exact code for the "Self-Assessment Grader." It uses a computational graph to route the LLM: if the retrieved context is bad (Graded "No"), it triggers an external web search tool and rewrites the query.

- https://github.com/sarabesh/HybridRAG: This repository perfectly implements the baseline you need. It combines vector search (embeddings) with graph search (structured knowledge graphs) to retrieve context before passing it to the LLM.

- https://github.com/junhongmit/HybridRAG-Bench

- https://github.com/NetApp/hybrid-rag-graph-with-ai-governance(This is a real repo built around the exact HybridRAG arXiv:2408.04948 paper concepts)

- https://github.com/tomasonjo/blogs ( Contains great LangChain + Neo4j GraphRAG templates )

- (WeKnow-RAG: An Adaptive Approach for Retrieval-Augmented Generation Integrating Web Search and Knowledge Graphs) (https://arxiv.org/abs/2408.07611)

- (HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction)(https://arxiv.org/abs/2408.04948)
