"""Shared test setup: src on the path, heuristics-only detection so tests do not load the HF model."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RESOLVE_INJECTION_CLASSIFIER", "0")
