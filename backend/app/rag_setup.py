# app/rag_setup.py

import os
import json
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# LangChain core (modern import paths)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Vector store + embeddings (compatible imports)
try:
    from langchain_community.vectorstores import Chroma
except Exception as e:
    raise ImportError(f"Failed to import Chroma vectorstore: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError(f"Failed to import HuggingFaceEmbeddings: {e}")

# Google Gemini LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as e:
    raise ImportError(f"Failed to import ChatGoogleGenerativeAI: {e}")


# ===============================================================
# 🔧 Environment & Constants
# ===============================================================
load_dotenv()

DB_DIRECTORY = os.getenv("DB_DIRECTORY", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dk_bhave_scholarship")

# Embeddings used during ingestion must match here
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Device: "cpu", "cuda", "mps", or "auto"
DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").lower()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
LLM_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0"))
RAG_MAX_OUTPUT_TOKENS = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "768"))

# Retrieval params (via env to tune without code changes)
RAG_SEARCH_TYPE = os.getenv("RAG_SEARCH_TYPE", "mmr").lower()  # "mmr", "similarity", "similarity_score_threshold"
RAG_K = int(os.getenv("RAG_K", "6"))
RAG_FETCH_K = int(os.getenv("RAG_FETCH_K", "40"))     # only for MMR
RAG_LAMBDA = float(os.getenv("RAG_LAMBDA", "0.7"))    # only for MMR
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))  # used if search_type supports it

# Trim each retrieved chunk to keep prompt tight
CONTEXT_MAX_CHARS_PER_DOC = int(os.getenv("CONTEXT_MAX_CHARS_PER_DOC", "1200"))


# ===============================================================
# 🧩 Helpers
# ===============================================================
def _detect_device(config_value: str) -> str:
    """Return 'cpu', 'cuda', or 'mps' based on env and availability."""
    if config_value != "auto":
        return config_value

    try:
        import torch  # type: ignore
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def _first_non_empty(*vals):
    for v in vals:
        if v:
            return v
    return None


def _clip(s: str, max_chars: int) -> str:
    if not s or len(s) <= max_chars:
        return s or ""
    return s[:max_chars].rstrip() + " …"


def format_docs(docs) -> str:
    """
    Format retrieved documents into a readable context block.
    Includes best-effort source and page_type metadata, and clips size per doc.
    """
    formatted = []
    for i, doc in enumerate(docs, start=1):
        md = getattr(doc, "metadata", {}) or {}
        source = _first_non_empty(md.get("source"), md.get("url"), md.get("page_url"), "Unknown")
        page_type = md.get("page_type", "Unknown")
        content = _clip((doc.page_content or "").strip(), CONTEXT_MAX_CHARS_PER_DOC)
        formatted.append(
            f"--- Document {i} ---\n"
            f"[Type: {page_type}] [Source: {source}]\n"
            f"{content}\n"
        )
    return "\n".join(formatted)


def citations_from_docs(docs) -> List[Dict[str, Any]]:
    """
    Build a structured list of sources from the retrieved docs, aligned to the
    order used in the context block (doc_id starts at 1).
    """
    items: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs, start=1):
        md = getattr(doc, "metadata", {}) or {}
        source = _first_non_empty(md.get("source"), md.get("url"), md.get("page_url"), "Unknown")
        items.append({
            "doc_id": i,
            "source": source,
            "page_type": md.get("page_type", "Unknown"),
            "title": md.get("title") or md.get("page_title") or None,
            "content_preview": _clip((doc.page_content or "").replace("\n", " "), 240),
        })
    return items


def _maybe_warn_embedding_mismatch(vs: Chroma) -> None:
    """
    If collection metadata has an embedding model name different from runtime,
    print a friendly warning. Works with typical Chroma clients; ignored if unavailable.
    """
    try:
        col = getattr(vs, "_collection", None)
        meta = getattr(col, "metadata", None)
        if isinstance(meta, dict):
            ingested_model = meta.get("embedding_model") or meta.get("embedding") or meta.get("model")
            if ingested_model and str(ingested_model) != EMBEDDING_MODEL:
                print(f"⚠️ Embedding model mismatch: DB='{ingested_model}', runtime='{EMBEDDING_MODEL}'.")
                print("   Consider re-ingesting or aligning models to avoid degraded recall.")
    except Exception:
        pass


def _count_chroma_vectors(vs: Chroma) -> int:
    """
    Robust vector count. Prefers public APIs, falls back to private collection.
    """
    # Try public pagination (if available in your LangChain version)
    try:
        total = 0
        page = 0
        page_size = 10_000
        while True:
            out = vs.get(ids=None, where=None, limit=page_size, include=[], offset=page * page_size)
            ids = out.get("ids", [])
            total += len(ids)
            if len(ids) < page_size:
                break
            page += 1
        return total
    except Exception:
        pass

    # Fallback to underlying chromadb collection
    try:
        return vs._collection.count()
    except Exception:
        return -1  # unknown


# ===============================================================
# 🧱 Vector Store & Embeddings
# ===============================================================
def _load_embeddings() -> HuggingFaceEmbeddings:
    device = _detect_device(DEVICE)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,  # must match ingestion
            "batch_size": 32 if device == "cpu" else 64,
        },
    )


def _load_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    # Embedding kwarg name differs across versions; embedding_function is broadly compatible
    return Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ===============================================================
# 💬 LLM Loader (Gemini with fallback)
# ===============================================================
def _load_llm(preferred_model: str) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in your .env file!")

    def _build(model_name: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=RAG_TEMPERATURE,
            max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
            convert_system_message_to_human=True,
        )

    try:
        return _build(preferred_model)
    except Exception as e:
        if preferred_model == DEFAULT_GEMINI_MODEL:
            raise
        print(f"⚠️ Failed to load '{preferred_model}': {e}")
        print(f"   Falling back to '{DEFAULT_GEMINI_MODEL}'")
        return _build(DEFAULT_GEMINI_MODEL)


# ===============================================================
# ⚙️ RAG Chain Builder
# ===============================================================
def _build_retriever(vector_store: Chroma):
    """
    Build a retriever using env-configured strategy:
    - "mmr": diverse + relevant
    - "similarity": top-k cosine similarity
    - "similarity_score_threshold": filter by score_threshold (if supported)
    """
    st = RAG_SEARCH_TYPE
    if st not in {"mmr", "similarity", "similarity_score_threshold"}:
        print(f"⚠️ Unknown RAG_SEARCH_TYPE='{st}', defaulting to 'mmr'.")
        st = "mmr"

    if st == "mmr":
        kwargs = {"k": RAG_K, "fetch_k": RAG_FETCH_K, "lambda_mult": RAG_LAMBDA}
    elif st == "similarity":
        kwargs = {"k": RAG_K}
    else:  # similarity_score_threshold
        kwargs = {"k": RAG_K, "score_threshold": RAG_SCORE_THRESHOLD}

    retriever = vector_store.as_retriever(search_type=st, search_kwargs=kwargs)
    desc = f"mode={st}, k={kwargs.get('k')}"
    if st == "mmr":
        desc += f", fetch_k={kwargs.get('fetch_k')}, λ={kwargs.get('lambda_mult')}"
    if st == "similarity_score_threshold":
        desc += f", score_threshold={kwargs.get('score_threshold')}"
    print(f"  ✓ Retriever configured ({desc})")
    return retriever


def create_rag_chain():
    """
    Builds and returns a complete RAG pipeline with:
    - Chroma retriever (configurable via env)
    - Gemini LLM
    - Context-only prompt to minimize hallucinations
    - Structured output: {"answer": str, "sources": List[...]}
    """
    print("🔄 Initializing RAG pipeline...")

    # 1) Embeddings
    print(f"  ➤ Loading embedding model: {EMBEDDING_MODEL} (device={_detect_device(DEVICE)})")
    embeddings = _load_embeddings()

    # 2) Vector Store
    print(f"  ➤ Loading vector database from: {DB_DIRECTORY} (collection={COLLECTION_NAME})")
    try:
        vector_store = _load_vector_store(embeddings)
        _maybe_warn_embedding_mismatch(vector_store)
        count = _count_chroma_vectors(vector_store)
        if count < 0:
            print("  ⚠️ Could not determine vector count.")
        else:
            print(f"  ✓ Loaded vector store with {count} vectors.")
        if count == 0:
            raise ValueError("Vector store is empty. Run your ingestion first.")
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")

    # 3) Retriever
    retriever = _build_retriever(vector_store)

    # 4) LLM
    print(f"  ➤ Loading Gemini model: {LLM_MODEL}")
    llm = _load_llm(LLM_MODEL)
    print("  ✓ Gemini model initialized.")

    # 5) Prompt
    template = """You are a helpful and strictly factual assistant for the D. K. Bhave Scholarship at Savitribai Phule Pune University.

Use ONLY the context below to answer the user's question.

Rules:
1) Use only the provided context — do not add outside information.
2) If the answer is not in the context, reply exactly:
   "I don’t have that specific information in the provided documents. Please check the official website or contact the scholarship office."
3) Do NOT include citations in the answer text; they will be provided separately.
4) Be concise and clear.

Context:
{context}

User Question:
{question}

Answer (based only on the above context):
"""
    prompt = PromptTemplate.from_template(template)

    # 6) Chain (LCEL) - return both answer and sources

    # Step A: retrieve docs + keep them
    retrieval = {"docs": retriever, "question": RunnablePassthrough()}

    # Step B: build prompt inputs from retrieved docs
    to_prompt_inputs = RunnableLambda(lambda d: {
        "context": format_docs(d["docs"]),
        "question": d["question"],
    })

    # Step C: generate answer
    answer_chain = to_prompt_inputs | prompt | llm | StrOutputParser()

    # Step D: extract structured sources from the same docs
    sources_chain = RunnableLambda(lambda d: citations_from_docs(d["docs"]))

    # Final: run answer and sources in parallel
    rag_chain = retrieval | RunnableParallel(answer=answer_chain, sources=sources_chain)

    print("✅ RAG chain initialized successfully.")
    return rag_chain


# ===============================================================
# 🔍 Helper: Retrieve Sources for a Query
# ===============================================================
def get_sources_from_query(query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant documents for a query (for debugging or display).
    Note: The score returned by Chroma is a distance; lower is better.
    """
    embeddings = _load_embeddings()
    vector_store = _load_vector_store(embeddings)

    results = vector_store.similarity_search_with_score(query, k=k)
    formatted_sources = []
    for doc, distance in results:
        md = doc.metadata or {}
        formatted_sources.append({
            "distance": float(distance),  # lower is better (distance = 1 - cosine_similarity if normalized)
            "source": _first_non_empty(md.get("source"), md.get("url"), md.get("page_url"), "Unknown"),
            "page_type": md.get("page_type", "Unknown"),
            "content_preview": _clip((doc.page_content or "").replace("\n", " "), 240),
        })
    return formatted_sources


# ===============================================================
# 🧠 Debug / Test Mode
# ===============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Testing RAG Chain")
    print("=" * 60)

    try:
        rag_chain = create_rag_chain()

        test_queries = [
            "What is the D.K. Bhave scholarship?",
            "What are the eligibility criteria?",
            "What is the application deadline?",
            "Which documents are required for application?",
        ]

        interactive = os.getenv("INTERACTIVE_TEST", "false").lower() in {"1", "true", "yes"}

        for q in test_queries:
            print(f"\n{'=' * 60}")
            print(f"❓ Question: {q}")
            print(f"{'=' * 60}")
            out = rag_chain.invoke(q)
            if isinstance(out, dict):
                print(f"\n🧠 Answer:\n{out['answer']}\n")
                print("📚 Sources used (in prompt order):")
                for s in out["sources"]:
                    print(f" - Doc {s['doc_id']}: {s['source']} [{s['page_type']}]")
            else:
                # Fallback if you change the chain to only return a string
                print(f"\n🧠 Answer:\n{out}\n")

            print(f"{'-' * 60}")
            print("🔎 (Debug) Top similar sources:")
            debug_sources = get_sources_from_query(q, k=max(RAG_K, 6))
            for i, s in enumerate(debug_sources, start=1):
                print(f"{i}. Distance: {s['distance']:.4f}")
                print(f"   Type: {s['page_type']}")
                print(f"   Source: {s['source']}")
                print(f"   Preview: {s['content_preview']}")
            print("-" * 60)

            if interactive:
                input("\nPress Enter for next question...")

    except Exception as e:
        import traceback
        print(f"\n❌ RAG test failed: {e}")
        traceback.print_exc()