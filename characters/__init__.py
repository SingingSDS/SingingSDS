from __future__ import annotations

import importlib
import pathlib
from .base import Character

CHARACTERS: dict[str, Character] = {}

for file in pathlib.Path(__file__).parent.glob("*.py"):
    if file.name in {"__init__.py", "base.py"}:
        continue
    module_name = f"{__name__}.{file.stem}"
    module = importlib.import_module(module_name)
    if hasattr(module, "get_character"):
        c: Character = getattr(module, "get_character")()
        CHARACTERS[file.stem] = c


def get_character(name: str) -> Character:
    return CHARACTERS[name]
