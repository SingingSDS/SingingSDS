import os
from typing import Optional

from transformers import pipeline

from .base import AbstractLLMModel
from .registry import register_llm_model

hf_token = os.getenv("HF_TOKEN")


@register_llm_model("meta-llama/Llama-")
class LlamaLLM(AbstractLLMModel):
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

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[
            str
        ] = "You are a pirate chatbot who always responds in pirate speak!",
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        outputs = self.pipe(messages, max_new_tokens=max_new_tokens, **kwargs)
        return outputs[0]["generated_text"]
