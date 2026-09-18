"""Split an article into chunks that follow its headings.

A chunk is one heading's section (the intro before the first heading counts as a
section too). Sections longer than `max_words` are split on paragraph boundaries
with a small overlap, and every chunk carries the article title and its heading at
the top so it makes sense on its own when retrieved.
"""
from __future__ import annotations

import re

HEADING = re.compile(r"^(#{2,3})\s+(.*)$")
SKIP_HEADINGS = {"related", "related articles", "see also"}
MIN_WORDS = 15


def _sections(body: str) -> list[tuple[str, str]]:
    """Yield (heading, text) pairs in document order. The first heading may be ''."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_code = False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = HEADING.match(line) if not in_code else None
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    out = []
    for h, lines_ in sections:
        text = "\n".join(lines_).strip()
        # Skip link lists and near-empty sections: they retrieve well and answer nothing.
        if not text or h.strip().lower() in SKIP_HEADINGS or len(text.split()) < MIN_WORDS:
            continue
        out.append((h, text))
    return out


def _split_long(text: str, max_words: int, overlap_words: int) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, current, count = [], [], 0
    for p in paragraphs:
        n = len(p.split())
        if current and count + n > max_words:
            pieces.append("\n\n".join(current))
            tail = " ".join("\n\n".join(current).split()[-overlap_words:]) if overlap_words else ""
            current, count = ([tail] if tail else []), len(tail.split())
        current.append(p)
        count += n
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_article(article: dict, max_words: int = 220, overlap_words: int = 30) -> list[dict]:
    chunks = []
    for heading, text in _sections(article["body"]):
        for piece in _split_long(text, max_words, overlap_words):
            header = article["title"] + (f" > {heading}" if heading else "")
            chunks.append({
                "chunk_id": f"{article['id']}#{len(chunks):02d}",
                "doc_id": article["id"],
                "title": article["title"],
                "heading": heading,
                "space": article["space"],
                "labels": article["labels"],
                "text": f"{header}\n\n{piece}",
            })
    return chunks


def chunk_all(articles: list[dict], **kw) -> list[dict]:
    out = []
    for a in articles:
        out.extend(chunk_article(a, **kw))
    return out
