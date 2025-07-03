from .base import AbstractASRModel

ASR_MODEL_REGISTRY = {}


def register_asr_model(prefix: str):
    def wrapper(cls):
        assert issubclass(cls, AbstractASRModel), f"{cls} must inherit AbstractASRModel"
        ASR_MODEL_REGISTRY[prefix] = cls
        return cls

    return wrapper


def get_asr_model(model_id: str, device="cpu", **kwargs) -> AbstractASRModel:
    for prefix, cls in ASR_MODEL_REGISTRY.items():
        if model_id.startswith(prefix):
            return cls(model_id, device=device, **kwargs)
    raise ValueError(f"No ASR wrapper found for model: {model_id}")
