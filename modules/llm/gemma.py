import os
from typing import Optional

from transformers import pipeline

from .base import AbstractLLMModel
from .registry import register_llm_model

hf_token = os.getenv("HF_TOKEN")


@register_llm_model("google/gemma-2-")
class GemmaLLM(AbstractLLMModel):
    def __init__(
        self, model_id: str, device: str = "auto", cache_dir: str = "cache", **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        model_kwargs = kwargs.setdefault("model_kwargs", {})
        model_kwargs["cache_dir"] = cache_dir
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map=device,
            return_full_text=False,
            token=hf_token,
            trust_remote_code=True,
            **kwargs,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_new_tokens=50, **kwargs) -> str:
        if not system_prompt:
            system_prompt = ""
        formatted_prompt = f"{system_prompt}\n\n现在，有人对你说：{prompt}\n\n你这样回答："
        outputs = self.pipe(formatted_prompt, max_new_tokens=max_new_tokens, **kwargs)
        return outputs[0]["generated_text"] if outputs else ""
