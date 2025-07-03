from characters import get_character
from modules.llm import get_llm_model
from time import time

if __name__ == "__main__":
    supported_llms = [
        # "MiniMaxAI/MiniMax-Text-01",
        # "Qwen/Qwen3-8B",
        # "Qwen/Qwen3-30B-A3B",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "tiiuae/Falcon-H1-1B-Base",
        # "tiiuae/Falcon-H1-3B-Instruct",
        # "google/gemma-2-2b",
        # "gemini-2.5-flash",
    ]
    character_prompt = get_character("Yaoyin").prompt
    for model_id in supported_llms:
        try:
            print(f"Loading model: {model_id}")
            llm = get_llm_model(model_id, cache_dir="./.cache")
            prompt = "你好，今天你心情怎么样？"
            start_time = time()
            result = llm.generate(prompt, system_prompt=character_prompt)
            end_time = time()
            print(f"[{model_id}] LLM inference time: {end_time - start_time:.2f} seconds")
            print(f"[{model_id}] LLM inference result:", result)
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            breakpoint()
            continue
