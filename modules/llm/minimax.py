# Ref: https://github.com/MiniMax-AI/MiniMax-01

from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    QuantoConfig,
)

from .base import AbstractLLMModel
from .registry import register_llm_model


@register_llm_model("MiniMaxAI/MiniMax-Text-01")
class MiniMaxLLM(AbstractLLMModel):
    def __init__(
        self, model_id: str, device: str = "cuda", cache_dir: str = "cache", **kwargs
    ):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("MiniMax model only supports CUDA device")
            super().__init__(model_id, device, cache_dir, **kwargs)

            # load hf config
            hf_config = AutoConfig.from_pretrained(
                "MiniMaxAI/MiniMax-Text-01", trust_remote_code=True, cache_dir=cache_dir,
            )

            # quantization config, int8 is recommended
            quantization_config = QuantoConfig(
                weights="int8",
                modules_to_not_convert=[
                    "lm_head",
                    "embed_tokens",
                ]
                + [
                    f"model.layers.{i}.coefficient"
                    for i in range(hf_config.num_hidden_layers)
                ]
                + [
                    f"model.layers.{i}.block_sparse_moe.gate"
                    for i in range(hf_config.num_hidden_layers)
                ],
            )

            # assume 8 GPUs
            world_size = torch.cuda.device_count()
            layers_per_device = hf_config.num_hidden_layers // world_size
            # set device map
            device_map = {
                "model.embed_tokens": "cuda:0",
                "model.norm": f"cuda:{world_size - 1}",
                "lm_head": f"cuda:{world_size - 1}",
            }
            for i in range(world_size):
                for j in range(layers_per_device):
                    device_map[f"model.layers.{i * layers_per_device + j}"] = f"cuda:{i}"

            # load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "MiniMaxAI/MiniMax-Text-01", cache_dir=cache_dir
            )

            # load bfloat16 model, move to device, and apply quantization
            self.quantized_model = AutoModelForCausalLM.from_pretrained(
                "MiniMaxAI/MiniMax-Text-01",
                torch_dtype="bfloat16",
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                offload_buffers=True,
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"Failed to load MiniMax model: {e}")
            breakpoint()
            raise e

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[
            str
        ] = "You are a helpful assistant created by MiniMax based on MiniMax-Text-01 model.",
        max_new_tokens: int = 20,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        messages.append({"role": "user", "content": [
                        {"type": "text", "text": prompt}]})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # tokenize and move to device
        model_inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=200020,
            use_cache=True,
        )
        generated_ids = self.quantized_model.generate(
            **model_inputs, generation_config=generation_config
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response
