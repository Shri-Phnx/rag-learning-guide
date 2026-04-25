# 🧠 RAG Learning Guide

> Interactive visual guide to **Retrieval-Augmented Generation** — built from whiteboard notes.

**Live demo:** https://shri-phnx.github.io/rag-learning-guide/

---

## What's Inside

| Tab | Content |
|-----|---------|
| 🗺️ **Pipeline** | End-to-end RAG flow — Indexing (DL → TS → Embedding → Vector Store) + Query (Retriever → SP+R+UQ → LLM → G). Tap any stage for details. |
| 📖 **Context** | What RAG is, why it exists, the core insight, when NOT to use it, common pitfalls, RAGAS evaluation metrics, and real-world use cases. |
| 💻 **Code** | Syntax-highlighted Python/LangChain code for every single stage, plus a full end-to-end Streamlit pipeline. Copy button included. |
| ⚡ **Tech Stack** | 5 layers compared (UI, Orchestration, Embedding, Vector DB, LLM) with effort ratings, recommendations, and a RAG vs Fine-tuning decision guide. |

---

## Tech Stack Used in the Guide

- **UI:** Streamlit · Gradio · Flask · FastAPI
- **Orchestration:** LangChain (LC) · LlamaIndex · Haystack
- **Embedding:** OpenAI Ada-002 · HuggingFace BGE · Cohere
- **Vector DB:** FAISS · Chroma · PgVector · Pinecone
- **LLM:** GPT-4 · Claude 3.5 · Gemini 1.5 · LLaMA 3

---

## Run Locally

```bash
npm install
npm run dev
```

Open http://localhost:5173/rag-learning-guide/

## The Augmented Prompt Formula

```
SP  +  R  +  UQ  →  LLM  →  G
│       │       │              │
System  Retrieved  User        Grounded
Prompt  Chunks     Query       Answer
```

---

*Built from whiteboard notes — my understanding of how RAG works and how to implement it.*
