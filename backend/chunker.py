#!/usr/bin/env python3
import argparse
import html
import json
import logging
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

# --- LangChain compatibility imports ---
try:
    # Newer LangChain
    from langchain_core.documents import Document
except Exception:
    # Older LangChain
    from langchain.docstore.document import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Optional tokenizer for token-based chunking ---
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def token_len(s: str) -> int:
        return len(_ENC.encode(s))

    TOKENIZER_NAME = "tiktoken/cl100k_base"
except Exception:
    _ENC = None

    def token_len(s: str) -> int:
        # Heuristic: ~4 chars per token for English-like text
        return max(1, int(len(s) / 4))

    TOKENIZER_NAME = "heuristic(len/4)"

# --- Logging ---
logger = logging.getLogger("rag_chunker")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    input_file: Path = Path("scraped_data_for_rag.json")
    output_pickle: Path = Path("document_chunks.pkl")
    output_jsonl: Optional[Path] = Path("document_chunks.jsonl")
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_content_chars: int = 80
    min_chunk_chars: int = 60
    use_token_split: bool = False
    keep_separator: bool = True
    prepend_context_header: bool = True
    preview: int = 3  # number of sample chunks to preview

# ============================================================
# Utilities
# ============================================================
NAV_PHRASES_RE = re.compile(
    r"\b(click here|read more|learn more|back to top|skip to content|faq|frequently asked questions)\b",
    flags=re.I
)

PUNCT_TRANSLATE = str.maketrans({
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-", "‐": "-",
    "\u00A0": " ",  # non-breaking space
})

ZERO_WIDTHS = dict.fromkeys(map(ord, ["\u200b", "\u200c", "\u200d", "\ufeff"]), None)


def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize scraped text for embedding."""
    if not text:
        return ""

    # Decode entities and normalize unicode
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(PUNCT_TRANSLATE)
    text = text.translate(ZERO_WIDTHS)

    # Standardize newlines and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove navigation filler phrases (keep actual content)
    text = NAV_PHRASES_RE.sub("", text)

    # Collapse excessive punctuation (keep ellipses)
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)

    # Light spacing normalization after punctuation when smashed together
    text = re.sub(r"([.!?;,:])([A-Za-z])", r"\1 \2", text)

    # Trim and final whitespace normalization
    text = text.strip()
    return text


def derive_page_name(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url)
    path = p.path or ""
    parts = [seg for seg in path.split("/") if seg]
    if not parts:
        return "home"
    name = parts[-1]
    if "." in name:
        name = name.split(".")[0]
    return name or "page"


def infer_page_type(url: str, title: str = "") -> str:
    u = (url or "").lower()
    t = (title or "").lower()
    checks = [
        ("faq", "FAQ"),
        ("contact", "Contact"),
        ("about", "About"),
        ("apply", "Application"),
        ("pricing", "Pricing"),
        ("docs", "Docs"),
        ("documentation", "Docs"),
        ("blog", "Blog"),
        ("support", "Support"),
        ("help", "Support"),
        ("terms", "Legal"),
        ("privacy", "Legal"),
        ("legal", "Legal"),
        ("careers", "Careers"),
        ("jobs", "Careers"),
        ("index", "Main"),
        ("home", "Main"),
        ("default", "Main"),
    ]
    for needle, label in checks:
        if needle in u or needle in t:
            return label
    return "General"


def enhance_metadata(item: Dict[str, Any], cleaned_text: Optional[str] = None) -> Dict[str, Any]:
    src_meta = item.get("metadata", {}) or {}
    url = item.get("url") or item.get("source") or src_meta.get("url", "")
    title = item.get("title", "") or src_meta.get("title", "")

    parsed = urlparse(url) if url else None
    domain = parsed.netloc if parsed else ""
    path = parsed.path if parsed else ""

    metadata = {
        "source": url,
        "domain": domain,
        "path": path,
        "title": title,
        "description": src_meta.get("description", ""),
        "keywords": src_meta.get("keywords", ""),
        "language": src_meta.get("language", ""),
        "word_count": item.get("word_count", 0),
        "page_name": derive_page_name(url) if url else "",
        "page_type": infer_page_type(url, title),
    }

    # Take accurate word count if cleaned text provided
    if cleaned_text:
        metadata["word_count"] = len(cleaned_text.split())

    return metadata


def build_context_header(metadata: Dict[str, Any]) -> str:
    """Create a small header for each document chunk."""
    bits = []
    if metadata.get("title"):
        bits.append(f"[Page: {metadata['title']}]")
    if metadata.get("page_name"):
        bits.append(f"[Section: {metadata['page_name']}]")
    if metadata.get("page_type"):
        bits.append(f"[Type: {metadata['page_type']}]")
    if metadata.get("source"):
        bits.append(f"[Source: {metadata['source']}]")
    return "\n".join(bits)


def load_scraped_pages(path: Path) -> List[Dict[str, Any]]:
    """Load your scraped JSON and return a list of page dicts."""
    if not path.exists():
        logger.error(f"❌ Error: {path} not found.")
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"❌ Failed to parse JSON: {e}")
        return []

    if isinstance(data, dict) and "pages" in data:
        pages = data["pages"]
    elif isinstance(data, list):
        pages = data
    else:
        logger.error("❌ Unable to find a 'pages' array in the JSON.")
        return []

    if not isinstance(pages, list) or not pages:
        logger.error("❌ No pages found in the JSON.")
        return []

    logger.info(f"✓ Loaded {len(pages)} pages from {path}")
    return pages


def quality_gate(text: str, min_chars: int) -> bool:
    """Filter out tiny or low-signal chunks (heuristic)."""
    if len(text) < min_chars:
        return False
    # Drop extremely non-alphanumeric heavy chunks (like nav, tables, junk)
    alnum_ratio = sum(c.isalnum() for c in text) / max(1, len(text))
    if alnum_ratio < 0.20:
        return False
    return True


def get_text_splitter(cfg: Config) -> RecursiveCharacterTextSplitter:
    length_fn: Callable[[str], int] = token_len if cfg.use_token_split else len
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=length_fn,
        keep_separator=cfg.keep_separator,
        add_start_index=True,  # gives 'start_index' in chunk metadata
        separators=[
            "\n\n", "\n",
            ". ", "? ", "! ", "; ", ": ",
            "— ", " - ", ", ",
            " ", ""
        ],
    )


def annotate_and_filter_chunks(
    chunks: List[Document],
    cfg: Config
) -> List[Document]:
    """Add rich metadata to chunks and filter low-quality ones."""
    counters: Dict[str, int] = {}
    seen_texts: set = set()
    refined: List[Document] = []

    for ch in chunks:
        text = ch.page_content.strip()
        meta = ch.metadata or {}

        # Ensure we have a stable per-page key
        page_uid = meta.get("source") or meta.get("title") or meta.get("page_name") or "page"
        counters.setdefault(page_uid, 0)

        if not quality_gate(text, cfg.min_chunk_chars):
            continue

        # Deduplicate exact chunk texts (rare but happens)
        key = (page_uid, text)
        if key in seen_texts:
            continue
        seen_texts.add(key)

        # Increment index per page and annotate
        counters[page_uid] += 1
        chunk_index = counters[page_uid]

        start_char = meta.get("start_index", None)
        end_char = start_char + len(text) if isinstance(start_char, int) else None

        ch.metadata.update({
            "chunk_index": chunk_index,
            "chunk_id": f"{meta.get('page_name', 'page')}__{chunk_index}",
            "chunk_length_chars": len(text),
            "chunk_length_tokens": token_len(text),
            "tokenizer": TOKENIZER_NAME,
            "start_char": start_char,
            "end_char": end_char,
        })
        refined.append(ch)

    return refined


def save_pickle(chunks: List[Document], path: Path) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"✓ Saved {len(chunks)} chunks → {path}")
    except Exception as e:
        logger.error(f"❌ Error saving pickle file: {e}")


def save_jsonl(chunks: List[Document], path: Path) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            for ch in chunks:
                obj = {"page_content": ch.page_content, "metadata": ch.metadata}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        logger.info(f"✓ Saved {len(chunks)} chunks → {path}")
    except Exception as e:
        logger.error(f"❌ Error saving JSONL file: {e}")


def preview_chunks(chunks: List[Document], n: int = 3) -> None:
    print("\nSample chunks:")
    for i, ch in enumerate(chunks[:n], 1):
        snippet = ch.page_content[:200].replace("\n", " ")
        ptype = ch.metadata.get("page_type", "N/A")
        print(f"  {i}. [{ptype}] {snippet}...")


def build_documents(pages: List[Dict[str, Any]], cfg: Config) -> List[Document]:
    documents: List[Document] = []
    for item in pages:
        raw_text = item.get("content") or item.get("text") or item.get("body") or ""
        text = clean_and_normalize_text(raw_text)

        if len(text) < cfg.min_content_chars:
            continue  # skip trivial content

        meta = enhance_metadata(item, cleaned_text=text)
        content = text
        if cfg.prepend_context_header:
            header = build_context_header(meta)
            if header:
                meta["context_header"] = header  # keep in metadata for reference
                content = f"{header}\n\n{text}"

        documents.append(Document(page_content=content, metadata=meta))
    return documents


def run(cfg: Config) -> None:
    print("=" * 60)
    print(" RAG Document Chunking Pipeline ")
    print("=" * 60)

    pages = load_scraped_pages(cfg.input_file)
    if not pages:
        print("=" * 60)
        return

    documents = build_documents(pages, cfg)
    if not documents:
        logger.error("❌ No valid documents after cleaning.")
        print("=" * 60)
        return

    logger.info(f"✓ Prepared {len(documents)} cleaned documents.")
    splitter = get_text_splitter(cfg)
    chunks = splitter.split_documents(documents)
    refined_chunks = annotate_and_filter_chunks(chunks, cfg)

    if not refined_chunks:
        logger.error("❌ No quality chunks created — try lowering filters.")
        print("=" * 60)
        return

    avg_len = sum(len(c.page_content) for c in refined_chunks) / len(refined_chunks)
    logger.info(f"\n✓ Created {len(refined_chunks)} chunks (avg length: {avg_len:.0f} chars)")
    logger.info(f"✓ Splitter length function: {'tokens' if cfg.use_token_split else 'chars'} ({TOKENIZER_NAME})\n")

    preview_chunks(refined_chunks, cfg.preview)

    # Save outputs
    if cfg.output_pickle:
        save_pickle(refined_chunks, cfg.output_pickle)
    if cfg.output_jsonl:
        save_jsonl(refined_chunks, cfg.output_jsonl)

    logger.info("✓ Ready for embedding and vector store creation.")
    print("=" * 60)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="RAG Document Chunking Pipeline")
    parser.add_argument("-i", "--input", type=Path, default=Path("scraped_data_for_rag.json"), help="Input JSON file")
    parser.add_argument("-p", "--out-pkl", type=Path, default=Path("document_chunks.pkl"), help="Output pickle file")
    parser.add_argument("-j", "--out-jsonl", type=Path, default=Path("document_chunks.jsonl"), help="Output JSONL file")
    parser.add_argument("--size", type=int, default=500, help="Chunk size (chars or tokens)")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap (chars or tokens)")
    parser.add_argument("--token-split", action="store_true", help="Use token-based splitting (requires tiktoken for accuracy)")
    parser.add_argument("--min-doc-chars", type=int, default=80, help="Minimum cleaned document length to include")
    parser.add_argument("--min-chunk-chars", type=int, default=60, help="Minimum chunk length to include")
    parser.add_argument("--no-header", action="store_true", help="Do not prepend context header to content")
    parser.add_argument("--preview", type=int, default=3, help="Number of sample chunks to preview")
    args = parser.parse_args()

    return Config(
        input_file=args.input,
        output_pickle=args.out_pkl,
        output_jsonl=args.out_jsonl,
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        min_content_chars=args.min_doc_chars,
        min_chunk_chars=args.min_chunk_chars,
        use_token_split=bool(args.token_split),
        keep_separator=True,
        prepend_context_header=not args.no_header,
        preview=args.preview,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)