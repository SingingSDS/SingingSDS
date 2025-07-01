from abc import ABC, abstractmethod


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
