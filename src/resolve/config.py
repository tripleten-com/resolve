"""Load the YAML files under configs/ once, with paths resolved from the repository root.

    from resolve.config import ROOT, graph_config, tools_config, guards_config, judge_config
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("RESOLVE_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "configs"
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing config {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def graph_config() -> dict:
    return load_yaml("graph.yaml")


def tools_config() -> dict:
    return load_yaml("tools.yaml")


def guards_config() -> dict:
    return load_yaml("guards.yaml")


def judge_config() -> dict:
    return load_yaml("judge.yaml")


def model_for(role: str) -> str:
    """Resolve a role in graph.yaml (classify, specialists, review) to a model name from .env."""
    from resolve.llm import MODEL, MODEL_REVIEW
    env_name = graph_config().get("models", {}).get(role, "LLM_MODEL")
    return {"LLM_MODEL": MODEL, "LLM_MODEL_REVIEW": MODEL_REVIEW}.get(env_name) or os.environ.get(env_name, MODEL)
