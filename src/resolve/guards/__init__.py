"""Guards are plain code that runs outside the model and can stop a run.

`guards_enabled()` reads configs/guards.yaml; the environment variable RESOLVE_GUARDS_ENABLED=0
overrides it so the injection suite can run guards-off without editing the file.
"""
from __future__ import annotations

import os

from resolve.config import guards_config


def guards_enabled() -> bool:
    env = os.environ.get("RESOLVE_GUARDS_ENABLED")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    return bool(guards_config().get("guards_enabled", True))
