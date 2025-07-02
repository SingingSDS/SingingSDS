import os

os.environ["XDG_CACHE_HOME"] = "./.cache" # must be set before importing google.generativeai

import google.generativeai as genai

from .base import AbstractLLMModel
from .registry import register_llm_model


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@register_llm_model("gemini-2.5-flash")
class GeminiLLM(AbstractLLMModel):
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        if os.environ.get("XDG_CACHE_HOME") != cache_dir:
            raise RuntimeError(
                f"XDG_CACHE_HOME must be set to '{cache_dir}' before importing this module."
            )
        if not GOOGLE_API_KEY:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable to use Gemini.")
        super().__init__(model_id=model_id, **kwargs)
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_id)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.model.generate_content(prompt, **kwargs)
        return response.text
