"""07 Reliability: manual retrieval over an embedded Chroma index.

Local-only retrieval: chromadb PersistentClient + its built-in DefaultEmbeddingFunction
(all-MiniLM-L6-v2 through onnxruntime). If that model cannot be loaded (offline demo, CI)
the module falls back to a deterministic hashed bag-of-words embedding so retrieval keeps
working with no network. Set THOR_RAG_EMBEDDING=hashed to force the fallback (tests do).

Passages returned here are the ONLY manual text the explanation may cite (CLAUDE.md 7).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from apps.api.schemas import Explanation, ManualPassage

from .explain import FEATURE_PHRASES

log = logging.getLogger(__name__)

CHUNK_CHARS = 600
HASHED_DIM = 256
_EMBEDDING_ENV = "THOR_RAG_EMBEDDING"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


# --------------------------------------------------------------------------------------
# Embedding functions
# --------------------------------------------------------------------------------------


def _hashed_vector(text: str, dim: int = HASHED_DIM) -> list[float]:
    """Deterministic hashed bag-of-words (unigrams + bigrams), L2-normalised."""
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for g in grams:
        idx = int(hashlib.sha1(g.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


try:
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class HashedEmbeddingFunction(EmbeddingFunction[Documents]):
        """Offline fallback embedding: sha1-hashed bag-of-words, 256 dims, unit length."""

        def __init__(self, dim: int = HASHED_DIM) -> None:
            self.dim = dim

        def __call__(self, input: Documents) -> Embeddings:
            return [_hashed_vector(t, self.dim) for t in input]

        @staticmethod
        def name() -> str:
            return "thor_hashed_bow"

        def get_config(self) -> dict[str, Any]:
            return {"dim": self.dim}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> HashedEmbeddingFunction:
            return HashedEmbeddingFunction(int(config.get("dim", HASHED_DIM)))

        def default_space(self) -> str:
            return "cosine"

except Exception as _import_error:  # pragma: no cover - chromadb missing entirely
    HashedEmbeddingFunction = None  # type: ignore[assignment,misc]
    log.warning("chromadb unavailable: %s", _import_error)


_EF_CACHE: dict[str, Any] = {}


def _embedding_function() -> Any:
    """Resolve the embedding function once per process.

    THOR_RAG_EMBEDDING=hashed forces the offline fallback. Otherwise chromadb's
    DefaultEmbeddingFunction is constructed and probed with one string; any failure
    (no model, no network) logs a warning and returns the hashed fallback.
    """
    mode = os.environ.get(_EMBEDDING_ENV, "default").strip().lower()
    if mode in _EF_CACHE:
        return _EF_CACHE[mode]
    if mode == "hashed":
        ef = HashedEmbeddingFunction()
    else:
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            ef = DefaultEmbeddingFunction()
            ef(["probe"])
        except Exception as e:  # noqa: BLE001 - offline or model download failure
            log.warning("DefaultEmbeddingFunction unavailable (%s); using hashed fallback", e)
            ef = HashedEmbeddingFunction()
    _EF_CACHE[mode] = ef
    return ef


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


def _split_long(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split text into pieces of at most ~limit chars at paragraph, then sentence, boundaries."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > limit:
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                if len(current) + len(sentence) + 1 > limit and current:
                    pieces.append(current.strip())
                    current = ""
                current = f"{current} {sentence}".strip()
            continue
        if len(current) + len(para) + 2 > limit and current:
            pieces.append(current.strip())
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_markdown(text: str, source: str) -> list[dict[str, Any]]:
    """Chunk a Markdown document by ## / ### headings into ~600-char pieces.

    Input: file text and its file name. Output: list of {"text", "metadata": {source,
    section}} where section is the nearest ## or ### heading (deeper headings stay inline
    as text so a whole numbered section, e.g. "4.2 Drive-end bearing wear", shares one
    section label). Text before the first ## heading is labelled with the document title.
    """
    chunks: list[dict[str, Any]] = []
    title = source
    section = source
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        for piece in _split_long(body):
            chunks.append(
                {"text": f"{section}\n{piece}", "metadata": {"source": source, "section": section}}
            )

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title = heading
                if section == source:
                    section = title
                continue
            if level in (2, 3):
                flush()
                section = heading
                continue
            buffer.append(heading)
            continue
        buffer.append(line)
    flush()
    return chunks


def chunk_pdf(path: Path) -> list[dict[str, Any]]:
    """Chunk a PDF page by page (~600 chars each) with metadata {source, section, page}."""
    from pypdf import PdfReader

    chunks: list[dict[str, Any]] = []
    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        first_line = text.splitlines()[0].strip()[:80] or f"page {page_no}"
        for piece in _split_long(text):
            chunks.append(
                {
                    "text": piece,
                    "metadata": {"source": path.name, "section": first_line, "page": page_no},
                }
            )
    return chunks


def load_manual_chunks(manuals_dir: Path) -> list[dict[str, Any]]:
    """Load and chunk every *.md / *.txt / *.pdf under manuals_dir (sorted, deterministic)."""
    chunks: list[dict[str, Any]] = []
    for path in sorted(Path(manuals_dir).glob("*")):
        suffix = path.suffix.lower()
        try:
            if suffix in (".md", ".txt"):
                chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
            elif suffix == ".pdf":
                chunks.extend(chunk_pdf(path))
        except Exception as e:  # noqa: BLE001 - one bad file must not block indexing
            log.warning("skipping manual %s: %s", path.name, e)
    return chunks


# --------------------------------------------------------------------------------------
# Chroma access
# --------------------------------------------------------------------------------------


def _client(chroma_path: Path) -> Any:
    import chromadb

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def _create_collection(client: Any, name: str, ef: Any) -> Any:
    try:
        return client.get_or_create_collection(
            name, embedding_function=ef, configuration={"hnsw": {"space": "cosine"}}
        )
    except Exception:  # noqa: BLE001 - older chromadb signature
        return client.get_or_create_collection(
            name, embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )


def _open_collection(client: Any, name: str) -> Any | None:
    """Open an existing collection with the configured embedding function; on a config
    mismatch (index built under the other embedding) try the alternative; None if absent."""
    ef = _embedding_function()
    candidates = [ef]
    if HashedEmbeddingFunction is not None and not isinstance(ef, HashedEmbeddingFunction):
        candidates.append(HashedEmbeddingFunction())
    for candidate in candidates:
        try:
            return client.get_collection(name, embedding_function=candidate)
        except Exception as e:  # noqa: BLE001
            log.debug("get_collection(%s) with %s failed: %s", name, type(candidate).__name__, e)
    return None


# --------------------------------------------------------------------------------------
# Public tool functions
# --------------------------------------------------------------------------------------


def build_index(manuals_dir: Path, chroma_path: Path, collection: str = "manuals") -> int:
    """(Re)build the manual index.

    Inputs: directory of *.md / *.txt / *.pdf manuals, Chroma persistence path, collection
    name. The collection is dropped and rebuilt so the index always mirrors the directory.
    Output: number of chunks indexed (0 if the directory has no readable manuals).
    """
    chunks = load_manual_chunks(Path(manuals_dir))
    client = _client(Path(chroma_path))
    try:
        client.delete_collection(collection)
    except Exception:  # noqa: BLE001 - did not exist
        pass
    col = _create_collection(client, collection, _embedding_function())
    if not chunks:
        log.warning("no manual chunks found in %s", manuals_dir)
        return 0
    batch = 64
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        col.upsert(
            ids=[f"{c['metadata']['source']}::{start + i}" for i, c in enumerate(part)],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
        )
    log.info("indexed %d manual chunks from %s into %s", len(chunks), manuals_dir, chroma_path)
    return len(chunks)


def retrieve_manual_context(
    query: str, chroma_path: Path, k: int = 3, collection: str = "manuals"
) -> list[ManualPassage]:
    """Retrieve the k most relevant manual passages for a query.

    Inputs: free-text query (normally from query_from_explanation), Chroma path, k,
    collection name. Output: ManualPassage list ordered best first with score = 1 - cosine
    distance clamped to [0, 1]; empty if the index is missing or empty. Never raises on a
    retrieval problem (logs and returns []) because manual context enriches the
    explanation but is never load-bearing for the recommendation.
    """
    try:
        client = _client(Path(chroma_path))
        col = _open_collection(client, collection)
        if col is None or col.count() == 0:
            # Self-heal: build the index from the configured manuals directory once.
            from apps.api.settings import get_settings

            manuals_dir = Path(get_settings().manuals_dir)
            log.warning(
                "manual collection %s missing/empty at %s; building from %s",
                collection,
                chroma_path,
                manuals_dir,
            )
            build_index(manuals_dir, Path(chroma_path), collection=collection)
            col = _open_collection(client, collection)
            if col is None:
                return []
        n = col.count()
        if n == 0:
            return []
        res = col.query(
            query_texts=[query],
            n_results=max(1, min(max(5 * k, 20), n)),  # over-fetch, keep one hit per section
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:  # noqa: BLE001 - retrieval must never crash the pipeline
        log.warning("manual retrieval failed: %s", e)
        return []
    passages: list[ManualPassage] = []
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    dists = res.get("distances") or [[]]
    seen: set[tuple[str, str]] = set()
    for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
        meta = meta or {}
        key = (str(meta.get("source", "unknown")), str(meta.get("section", "")))
        if key in seen:
            continue  # results are distance-ordered, so the first hit per section is the best
        seen.add(key)
        if len(passages) >= k:
            break
        page = meta.get("page")
        passages.append(
            ManualPassage(
                source=str(meta.get("source", "unknown")),
                section=str(meta.get("section", "")),
                page=int(page) if page is not None else None,
                text=str(doc),
                score=float(min(1.0, max(0.0, 1.0 - float(dist)))),
            )
        )
    return passages


def query_from_explanation(exp: Explanation) -> str:
    """Deterministic retrieval query built from the top SHAP features.

    Input: Explanation. Output: a fixed-format string naming the drive-end bearing fault
    context, the plain-language names of up to three risk-raising features (falling back
    to the top features by |shap| if none raise risk), and the words "signature
    recommended actions". Same Explanation -> same string, so retrieval is reproducible.
    """
    raising = [f for f in exp.top_features if f.direction == "raises_risk"]
    chosen = (raising or exp.top_features)[:3]
    phrases = [FEATURE_PHRASES.get(f.feature, f.feature.replace("_", " ")) for f in chosen]
    if not phrases:
        phrases = ["vibration RMS", "bearing temperature"]
    return (
        "induction motor drive-end bearing wear: "
        + ", ".join(phrases)
        + " signature recommended actions"
    )
