import os

from transformers import pipeline

from .base import AbstractLLMModel
from .registry import register_llm_model

hf_token = os.getenv("HF_TOKEN")


@register_llm_model("openai-community/")
@register_llm_model("google/gemma-")
@register_llm_model("meta-llama/Llama-")
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
            trust_remote_code=True,
            **kwargs,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        outputs = self.pipe(prompt, **kwargs)
        return outputs[0]["generated_text"] if outputs else ""
