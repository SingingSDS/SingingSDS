import google.generativeai as genai
from abc import ABC, abstractmethod
from .base import AbstractLLMModel
from .registry import register_llm_model



GEMINI_TOKEN = os.getenv("GEMINI_API_KEY")


@register_llm_model("gemini-2.5-flash")
class GeminiModel(AbstractLLMModel):
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        super().__init__(model_id=model_id, **kwargs)
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_id)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.model.generate_content(prompt, **kwargs)
        return response.text

