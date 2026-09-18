"""Load the Quillstack knowledge base from data/kb/{public,internal}/*.md.

Each article starts with a front-matter block between two `---` lines:

    ---
    id: kb-0044
    title: Create an API key
    space: public
    labels: [api, keys]
    updated: 2026-05-20
    version: 5
    ---

followed by a markdown body. `load_articles()` returns one dict per article with
those fields plus `body` and `path`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

KB_DIR = Path(os.environ.get("KB_DIR", "data/kb"))
FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.S)


def parse_front_matter(meta_text: str) -> dict:
    """YAML when it parses; otherwise one `key: value` per line. Titles such as
    'Quillstack plans compared: Free, Team, and Business' are not valid YAML unquoted, and
    a `---` inside a body must not be taken for the closing fence."""
    try:
        meta = yaml.safe_load(meta_text)
        if isinstance(meta, dict):
            return meta
    except yaml.YAMLError:
        pass
    meta: dict = {}
    for line in meta_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value.strip("'\"")
    return meta


def parse_article(text: str, path: Path) -> dict:
    m = FRONT_MATTER.match(text)
    if not m:
        raise ValueError(f"{path}: missing front matter")
    meta_text, body = m.group(1), text[m.end():]
    meta = parse_front_matter(meta_text)
    for key in ("id", "title", "space"):
        if key not in meta:
            raise ValueError(f"{path}: front matter lacks {key!r}")
    meta["labels"] = list(meta.get("labels") or [])
    meta["updated"] = str(meta.get("updated", ""))
    meta["version"] = int(meta.get("version", 1))
    meta["body"] = body.strip()
    meta["path"] = str(path)
    return meta


def load_articles(kb_dir: Path | str = KB_DIR) -> list[dict]:
    kb_dir = Path(kb_dir)
    articles = []
    for space in ("public", "internal"):
        for path in sorted((kb_dir / space).glob("*.md")):
            article = parse_article(path.read_text(encoding="utf-8"), path)
            if article["space"] != space:
                raise ValueError(f"{path}: space {article['space']!r} does not match folder {space!r}")
            articles.append(article)
    return articles


def article_by_id(articles: list[dict]) -> dict[str, dict]:
    return {a["id"]: a for a in articles}
