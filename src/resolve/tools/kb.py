"""The two retrieval tools as LangChain tools. The descriptions are what steers the model."""
from __future__ import annotations

from langchain_core.tools import tool

from resolve.kb import search as kb_search

MAX_DOC_CHARS = 6000


@tool
def search_kb(query: str, k: int = 5) -> list[dict]:
    """Search the Quillstack help center for chunks that answer a question. Use it first, with the
    customer's own words or an exact error code (QS-1042), and again with different words if the
    first results do not answer. Returns chunks with doc_id, chunk_id, title, space, text, score."""
    return kb_search.search_kb(query, k=int(k))


@tool
def read_doc(doc_id: str) -> dict:
    """Read one whole help-center article by its doc_id (for example kb-0015) when a chunk from
    search_kb looks right but is cut off, or when you need the full procedure before answering."""
    doc = kb_search.read_doc(doc_id)
    if "body" in doc and len(doc["body"]) > MAX_DOC_CHARS:
        doc["body"] = doc["body"][:MAX_DOC_CHARS] + "\n[truncated]"
    return doc


KB_TOOLS = [search_kb, read_doc]
