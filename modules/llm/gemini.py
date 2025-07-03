import os
from typing import Optional

from google import genai
from google.genai import types

from .base import AbstractLLMModel
from .registry import register_llm_model


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@register_llm_model("gemini-2.5-flash")
class GeminiLLM(AbstractLLMModel):
    def __init__(
        self, model_id: str, device: str = "auto", cache_dir: str = "cache", **kwargs
    ):
        if not GOOGLE_API_KEY:
            raise ValueError(
                "Please set the GOOGLE_API_KEY environment variable to use Gemini."
            )
        super().__init__(model_id=model_id, **kwargs)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 1024,
        **kwargs,
    ) -> str:
        generation_config_dict = {
            "max_output_tokens": max_output_tokens,
            **kwargs,
        }
        if system_prompt:
            generation_config_dict["system_instruction"] = system_prompt
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(**generation_config_dict),
        )
        if response.text:
            return response.text
        else:
            print(
                f"No response from Gemini. May need to increase max_new_tokens. Current max_new_tokens: {max_new_tokens}"
            )
            return ""
