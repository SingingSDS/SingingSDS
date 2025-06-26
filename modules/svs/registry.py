from .base import AbstractSVSModel

SVS_MODEL_REGISTRY = {}


def register_svs_model(prefix: str):
    def wrapper(cls):
        assert issubclass(cls, AbstractSVSModel), f"{cls} must inherit AbstractSVSModel"
        SVS_MODEL_REGISTRY[prefix] = cls
        return cls

    return wrapper


def get_svs_model(model_id: str, device="cpu", **kwargs) -> AbstractSVSModel:
    for prefix, cls in SVS_MODEL_REGISTRY.items():
        if model_id.startswith(prefix):
            return cls(model_id, device=device, **kwargs)
    raise ValueError(f"No SVS wrapper found for model: {model_id}")
