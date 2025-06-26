import importlib
import pkgutil
from pathlib import Path

from .base import MelodyDatasetHandler

_registry = {}

for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if module_name in ("__init__", "base"):
        continue

    module = importlib.import_module(f"{__name__}.{module_name}")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, MelodyDatasetHandler)
            and attr is not MelodyDatasetHandler
        ):
            _registry[attr.name] = attr  # 注册 class 本身


def get_melody_handler(name: str) -> type[MelodyDatasetHandler]:
    if name not in _registry:
        raise ValueError(f"Melody source '{name}' not found")
    return _registry[name]
