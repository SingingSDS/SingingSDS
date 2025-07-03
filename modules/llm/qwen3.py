# Ref: https://qwenlm.github.io/blog/qwen3/

from typing import Optional

from .base import AbstractLLMModel
from .registry import register_llm_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@register_llm_model("Qwen/Qwen3-")
class Qwen3LLM(AbstractLLMModel):
    def __init__(
        self, model_id: str, device: str = "auto", cache_dir: str = "cache", **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device, torch_dtype="auto", cache_dir=cache_dir
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        # parse thinking content
        if enable_thinking:
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            output_ids = output_ids[index:]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
