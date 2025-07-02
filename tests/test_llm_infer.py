from modules.llm import get_llm_model

if __name__ == "__main__":
    supported_llms = [
        # "MiniMaxAI/MiniMax-M1-80k", #-》load with custom code
        # "Qwen/Qwen-1_8B",
        # "meta-llama/Llama-3.1-8B-Instruct", # pending for approval
        # "tiiuae/Falcon-H1-1B-Base",
        # "tiiuae/Falcon-H1-3B-Instruct",
        # "tencent/Hunyuan-A13B-Instruct", # -> load with custom code
        # "deepseek-ai/DeepSeek-R1-0528",
        # "openai-community/gpt2-xl",
        # "google/gemma-2-2b",
    ]
    for model_id in supported_llms:
        try:
            print(f"Loading model: {model_id}")
            llm = get_llm_model(model_id, cache_dir="./.cache")
            prompt = "你好，今天你心情怎么样？"
            result = llm.generate(prompt)
            print(f"=================")
            print(f"[{model_id}] LLM inference result:", result)
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            breakpoint()
            continue
