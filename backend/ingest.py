#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pickle
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Ensure both imports are available for pickle compatibility
try:
    from langchain_core.documents import Document as LCDocument  # newer
except Exception:
    LCDocument = None

try:
    from langchain.docstore.document import Document as LegacyDocument  # older
except Exception:
    LegacyDocument = None

# Prefer newer Document class for constructing new docs
DocumentClass = LCDocument or LegacyDocument
if DocumentClass is None:
    raise ImportError("Could not import a compatible LangChain Document class.")

# Vector store and embeddings
try:
    from langchain_community.vectorstores import Chroma
except Exception as e:
    raise ImportError(f"Failed to import Chroma vectorstore from langchain_community: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError(f"Failed to import HuggingFaceEmbeddings from langchain_community: {e}")

# Optional: turn off tokenizer parallelism noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =========================
# Defaults / Configuration
# =========================
DEFAULT_INPUT = Path("document_chunks.pkl")
DEFAULT_DB_DIR = Path("chroma_db")
DEFAULT_COLLECTION = "dk_bhave_scholarship"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 100


# =========================
# Utilities
# =========================
def detect_device() -> str:
    try:
        import torch  # type: ignore
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def infer_id(doc, idx: int) -> str:
    m = getattr(doc, "metadata", {}) or {}
    if "id" in m and m["id"]:
        return str(m["id"])
    if "chunk_id" in m and m["chunk_id"]:
        return str(m["chunk_id"])
    # fallback: hash of source + offsets + idx
    source = str(m.get("source", ""))
    start = str(m.get("start_char", ""))
    end = str(m.get("end_char", ""))
    base = f"{source}::{start}-{end}::{idx}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()  # stable but compact


def is_document(obj) -> bool:
    # Best-effort check for LangChain Document-like object
    if LCDocument and isinstance(obj, LCDocument):
        return True
    if LegacyDocument and isinstance(obj, LegacyDocument):
        return True
    # Fallback: duck typing
    return hasattr(obj, "page_content") and hasattr(obj, "metadata")


def to_document(obj) -> "DocumentClass":
    # If already a Document, return; else construct from dict
    if is_document(obj):
        return obj
    if isinstance(obj, dict) and "page_content" in obj and "metadata" in obj:
        return DocumentClass(page_content=obj["page_content"], metadata=obj.get("metadata") or {})
    raise ValueError("Unsupported chunk format; expected LangChain Document or dict with page_content/metadata.")


# =========================
# Loading chunks
# =========================
def load_chunks(input_path: Path) -> List["DocumentClass"]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    ext = input_path.suffix.lower()
    chunks: List["DocumentClass"] = []

    if ext == ".pkl" or ext == ".pickle":
        # Load pickle (ensure both Document modules are importable for compatibility)
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            chunks = [to_document(d) for d in data]
        else:
            raise ValueError(f"Unexpected pickle content type: {type(data)}")
    elif ext == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(to_document(obj))
    else:
        raise ValueError("Unsupported input format. Use .pkl/.pickle or .jsonl")

    # Ensure metadata dict present
    for d in chunks:
        if d.metadata is None:
            d.metadata = {}

    return chunks


# =========================
# Embeddings
# =========================
def initialize_embeddings(model_name: str, device: Optional[str] = None, batch_size: int = 32) -> HuggingFaceEmbeddings:
    device = device or detect_device()
    print(f"🔄 Initializing embedding model: {model_name} (device={device}, batch={batch_size})")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
        )
        print("✓ Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load embedding model '{model_name}': {e}")


# =========================
# Chroma helpers
# =========================
def backup_existing_db(db_dir: Path) -> None:
    if db_dir.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"{db_dir}_backup_{ts}")
        try:
            shutil.copytree(db_dir, backup_dir)
            print(f"✓ Backup created at: {backup_dir}")
        except Exception as e:
            print(f"⚠ Backup failed: {e}")


def clear_existing_db(db_dir: Path, force: bool = False) -> bool:
    if not db_dir.exists():
        return True

    if not force:
        resp = input(f"\n⚠ Database '{db_dir}' exists. Recreate it? (yes/no): ").strip().lower()
        if resp != "yes":
            print("❌ Ingestion cancelled.")
            return False

    backup_existing_db(db_dir)
    try:
        shutil.rmtree(db_dir)
        print(f"✓ Cleared old database: {db_dir}")
        return True
    except Exception as e:
        print(f"❌ Error deleting database: {e}")
        return False


def create_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: Path,
    collection_metadata: Optional[dict] = None,
):
    # Connect to (or create) a persistent collection
    # Compatibility: some versions expect embedding or embedding_function
    kwargs = dict(collection_name=collection_name, persist_directory=str(persist_directory))
    try:
        vs = Chroma(embedding_function=embeddings, collection_metadata=collection_metadata, **kwargs)
    except TypeError:
        # Fallback older signature
        vs = Chroma(embedding=embeddings, collection_metadata=collection_metadata, **kwargs)
    return vs


def from_documents_compat(
    documents: List["DocumentClass"],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: Path,
    collection_metadata: Optional[dict] = None,
):
    kwargs = dict(
        documents=documents,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    try:
        return Chroma.from_documents(embedding=embeddings, collection_metadata=collection_metadata, **kwargs)
    except TypeError:
        return Chroma.from_documents(embedding_function=embeddings, collection_metadata=collection_metadata, **kwargs)


def get_existing_ids(vs, candidate_ids: List[str]) -> set:
    """
    Return the subset of candidate_ids that already exist in the Chroma collection.
    Uses underlying _collection.get when available.
    """
    existing = set()
    try:
        # Access the underlying chromadb Collection (not part of public API but widely used)
        col = getattr(vs, "_collection", None)
        if col is not None:
            # get returns only existing ids
            out = col.get(ids=candidate_ids, include=[])
            for i in out.get("ids", []):
                if i is not None:
                    existing.add(i)
    except Exception:
        # Fallback: assume none exist (worst-case re-adding, but add_documents handles duplicates by error)
        pass
    return existing


# =========================
# Verification / Stats
# =========================
def verify_chunks(chunks: List["DocumentClass"]) -> None:
    print("\n" + "=" * 60)
    print("📊 Chunk Dataset Summary")
    print("=" * 60)

    total = len(chunks)
    total_chars = sum(len(d.page_content or "") for d in chunks)
    avg_size = total_chars / total if total else 0

    print(f"Total chunks: {total}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average chunk size: {avg_size:.1f} characters")

    # Page type distribution
    page_types = Counter((d.metadata or {}).get("page_type", "Unknown") for d in chunks)
    print("\nChunks by page type:")
    for ptype, count in page_types.most_common():
        print(f"  - {ptype}: {count}")

    # Top sources
    sources = Counter((d.metadata or {}).get("source", "unknown") for d in chunks)
    top_sources = sources.most_common(5)
    print("\nTop sources:")
    for src, cnt in top_sources:
        print(f"  - {src} → {cnt} chunks")

    # Duplicate chunk_id check
    ids = [infer_id(d, i) for i, d in enumerate(chunks)]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        print(f"\n⚠ Detected {len(dupes)} duplicate IDs (by metadata). Will disambiguate at add-time.")
    else:
        print("\n✓ No duplicate IDs detected.")


# =========================
# Ingestion
# =========================
def ingest_in_batches(
    chunks: List["DocumentClass"],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: Path,
    batch_size: int = 100,
    dataset_tag: Optional[str] = None,
    incremental: bool = False,
):
    total = len(chunks)
    print(f"\n🧩 Ingesting {total} chunks in batches of {batch_size}...")

    # Attach IDs (and ensure they exist in metadata)
    ids: List[str] = []
    prepared_docs: List["DocumentClass"] = []
    seen = set()

    for i, d in enumerate(chunks):
        doc_id = infer_id(d, i)
        # Ensure id in metadata for traceability
        d.metadata = dict(d.metadata or {})
        d.metadata["id"] = doc_id
        # NOTE: Keep chunk_id if present (more human-readable)
        if "chunk_id" not in d.metadata and "chunk_index" in d.metadata:
            d.metadata["chunk_id"] = f"{d.metadata.get('page_name','page')}__{d.metadata['chunk_index']}"
        # Avoid duplicates in-memory
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ids.append(doc_id)
        prepared_docs.append(d)

    # Collection metadata (helps track provenance)
    collection_metadata = {
        "source": "rag_ingestion",
        "embedding_model": getattr(embeddings, "model_name", "huggingface"),
        "dataset_tag": dataset_tag or "",
        "created_at": datetime.utcnow().isoformat(),
    }

    # Create or connect to collection
    vs = create_vectorstore(
      embeddings=embeddings,
      collection_name=collection_name,
      persist_directory=persist_directory,
      collection_metadata=collection_metadata,
    )

    # If incremental, skip docs with existing IDs
    if incremental:
        existing = get_existing_ids(vs, ids)
        if existing:
            print(f"ℹ Skipping {len(existing)} existing chunks (incremental mode).")
        id_to_doc = {doc.metadata["id"]: doc for doc in prepared_docs}
        prepared_docs = [id_to_doc[i] for i in ids if i not in existing]
        ids = [i for i in ids if i not in existing]

    if not prepared_docs:
        print("✓ Nothing new to ingest.")
        return vs

    # Ingest in batches
    for start in range(0, len(prepared_docs), batch_size):
        end = start + batch_size
        batch_docs = prepared_docs[start:end]
        batch_ids = ids[start:end]
        batch_num = start // batch_size + 1
        total_batches = (len(prepared_docs) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} → {len(batch_docs)} chunks")

        # Add docs with IDs
        try:
            vs.add_documents(batch_docs, ids=batch_ids)
        except TypeError:
            # Older API might not accept ids kw; try without
            vs.add_documents(batch_docs)

        # Persist per batch to avoid data loss on interruption
        try:
            vs.persist()
        except Exception as e:
            print(f"⚠ Warning: Persistence issue after batch {batch_num}: {e}")

    print("✓ All chunks ingested successfully.")
    return vs


# =========================
# Verification queries
# =========================
def run_test_queries(db, queries: Optional[List[str]] = None, k: int = 3) -> None:
    print("\n" + "=" * 60)
    print("🔍 Verification Queries")
    print("=" * 60)

    queries = queries or [
        "What is D.K. Bhave Scholarship?",
        "Eligibility requirements for the scholarship",
        "When is the last date to apply?",
        "How to apply online for D.K. Bhave?",
    ]

    for q in queries:
        print(f"\n❓ Query: {q}")
        try:
            results = db.similarity_search_with_score(q, k=k)
            if not results:
                print("   ⚠ No results found.")
                continue

            for i, (doc, score) in enumerate(results, 1):
                # Chroma returns distance (cosine distance if normalized embeddings)
                similarity = 1 - float(score)
                print(f"\n   {i}. Similarity: {similarity:.3f} (distance={score:.3f})")
                print(f"      Source: {doc.metadata.get('source')}")
                print(f"      Type: {doc.metadata.get('page_type')}")
                print(f"      ID: {doc.metadata.get('id')}")
                snippet = (doc.page_content or "")[:220].replace("\n", " ")
                print(f"      Preview: {snippet}...")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")


# =========================
# Main pipeline
# =========================
def ingest_data(
    input_path: Path,
    db_directory: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    force_recreate: bool = False,
    incremental: bool = False,
    skip_queries: bool = False,
    limit: Optional[int] = None,
):
    print("=" * 60)
    print("🚀 RAG Vector Store Ingestion Pipeline")
    print("=" * 60)

    try:
        chunks = load_chunks(input_path)
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return

    if limit:
        chunks = chunks[:limit]
    print(f"✓ Loaded {len(chunks)} chunks from {input_path}")

    verify_chunks(chunks)

    # If not incremental, we may rebuild DB
    if not incremental:
        if not clear_existing_db(db_directory, force=force_recreate):
            return
    else:
        print("ℹ Incremental mode: existing DB will be reused; only new chunks will be added.")

    # Initialize embedding model
    device = detect_device()
    embed_batch = 64 if device in ("cuda", "mps") else 32
    try:
        embeddings = initialize_embeddings(model_name=model_name, device=device, batch_size=embed_batch)
    except Exception as e:
        print(e)
        return

    # Ingest
    dataset_tag = ""
    try:
        if input_path.exists():
            dataset_tag = f"{input_path.name}:{file_sha256(input_path)[:8]}"
    except Exception:
        pass

    db = ingest_in_batches(
        chunks=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=db_directory,
        batch_size=batch_size,
        dataset_tag=dataset_tag,
        incremental=incremental,
    )

    # Persist database (final)
    try:
        db.persist()
        print(f"💾 Database persisted at: {db_directory}")
    except Exception as e:
        print(f"⚠ Warning: Final persistence issue: {e}")

    # Verify by querying
    if not skip_queries:
        run_test_queries(db)

    # Final summary
    print("\n" + "=" * 60)
    print("✅ Ingestion Complete!")
    print("=" * 60)
    print(f"Database: {db_directory.resolve()}")
    print(f"Collection: {collection_name}")
    print(f"Embedding Model: {model_name}")
    print(f"Device: {device}")
    print("=" * 60)


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Ingest chunked documents into ChromaDB")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help="Input chunks file (.pkl or .jsonl)")
    p.add_argument("-d", "--db", type=Path, default=DEFAULT_DB_DIR, help="Chroma persist directory")
    p.add_argument("-c", "--collection", type=str, default=DEFAULT_COLLECTION, help="Chroma collection name")
    p.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help="HuggingFace embedding model")
    p.add_argument("-b", "--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for ingestion")
    p.add_argument("-f", "--force", action="store_true", help="Force recreate database without prompt")
    p.add_argument("--incremental", action="store_true", help="Add only new chunks (by ID) to existing DB")
    p.add_argument("--skip-queries", action="store_true", help="Skip verification queries after ingestion")
    p.add_argument("--limit", type=int, default=None, help="Limit number of chunks ingested (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_data(
        input_path=args.input,
        db_directory=args.db,
        collection_name=args.collection,
        model_name=args.model,
        batch_size=args.batch,
        force_recreate=args.force,
        incremental=args.incremental,
        skip_queries=args.skip_queries,
        limit=args.limit,
    )