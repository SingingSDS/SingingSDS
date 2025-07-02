from .base import AbstractLLMModel

LLM_MODEL_REGISTRY = {}


def register_llm_model(prefix: str):
    def wrapper(cls):
        assert issubclass(cls, AbstractLLMModel), f"{cls} must inherit AbstractLLMModel"
        LLM_MODEL_REGISTRY[prefix] = cls
        return cls

    return wrapper


def get_llm_model(model_id: str, device="cpu", **kwargs) -> AbstractLLMModel:
    for prefix, cls in LLM_MODEL_REGISTRY.items():
        if model_id.startswith(prefix):
            return cls(model_id, device=device, **kwargs)
    raise ValueError(f"No LLM wrapper found for model: {model_id}")
