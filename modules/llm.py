import os
from abc import ABC, abstractmethod

from transformers import pipeline

LLM_MODEL_REGISTRY = {}
hf_token = os.getenv("HF_TOKEN")


class AbstractLLMModel(ABC):
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        print(f"Loading LLM model {model_id}...")
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass


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


@register_llm_model("google/gemma")
@register_llm_model("tii/")  # e.g., Falcon
@register_llm_model("meta-llama")
class HFTextGenerationLLM(AbstractLLMModel):
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        model_kwargs = kwargs.setdefault("model_kwargs", {})
        model_kwargs["cache_dir"] = cache_dir
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device=0 if device == "cuda" else -1,
            return_full_text=False,
            token=hf_token,
            **kwargs,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        outputs = self.pipe(prompt, **kwargs)
        return outputs[0]["generated_text"] if outputs else ""
