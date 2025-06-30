import torch
import time
import librosa

from modules.asr import get_asr_model
from modules.llm import get_llm_model
from modules.svs import get_svs_model
from evaluation.svs_eval import load_evaluators, run_evaluation
from modules.melody import MelodyController
from modules.utils.text_normalize import clean_llm_output


class SingingDialoguePipeline:
    def __init__(self, config: dict):
        if "device" in config:
            self.device = config["device"]
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = config["cache_dir"]
        self.asr = get_asr_model(
            config["asr_model"], device=self.device, cache_dir=self.cache_dir
        )
        self.llm = get_llm_model(
            config["llm_model"], device=self.device, cache_dir=self.cache_dir
        )
        self.svs = get_svs_model(
            config["svs_model"], device=self.device, cache_dir=self.cache_dir
        )
        self.melody_controller = MelodyController(
            config["melody_source"], self.cache_dir
        )
        self.track_latency = config.get("track_latency", False)
        self.evaluators = load_evaluators(config.get("evaluators", {}).get("svs", []))

    def set_asr_model(self, asr_model: str):
        self.asr = get_asr_model(
            asr_model, device=self.device, cache_dir=self.cache_dir
        )

    def set_llm_model(self, llm_model: str):
        self.llm = get_llm_model(
            llm_model, device=self.device, cache_dir=self.cache_dir
        )

    def set_svs_model(self, svs_model: str):
        self.svs = get_svs_model(
            svs_model, device=self.device, cache_dir=self.cache_dir
        )

    def set_melody_controller(self, melody_source: str):
        self.melody_controller = MelodyController(melody_source, self.cache_dir)

    def run(
        self,
        audio_path,
        language,
        prompt_template,
        speaker,
        max_new_tokens=100,
    ):
        if self.track_latency:
            asr_start_time = time.time()
        audio_array, audio_sample_rate = librosa.load(audio_path, sr=16000)
        asr_result = self.asr.transcribe(
            audio_array, audio_sample_rate=audio_sample_rate, language=language
        )
        if self.track_latency:
            asr_end_time = time.time()
            asr_latency = asr_end_time - asr_start_time
        melody_prompt = self.melody_controller.get_melody_constraints()
        prompt = prompt_template.format(melody_prompt, asr_result)
        if self.track_latency:
            llm_start_time = time.time()
        output = self.llm.generate(prompt, max_new_tokens=max_new_tokens)
        if self.track_latency:
            llm_end_time = time.time()
            llm_latency = llm_end_time - llm_start_time
        print(f"llm output: {output}确认一下是不是不含prompt的")
        llm_response = clean_llm_output(output, language=language)
        score = self.melody_controller.generate_score(llm_response, language)
        if self.track_latency:
            svs_start_time = time.time()
        singing_audio, sample_rate = self.svs.synthesize(
            score, language=language, speaker=speaker
        )
        if self.track_latency:
            svs_end_time = time.time()
            svs_latency = svs_end_time - svs_start_time
        results = {
            "asr_text": asr_result,
            "llm_text": llm_response,
            "svs_audio": (singing_audio, sample_rate),
        }
        if self.track_latency:
            results["metrics"].update({
                "asr_latency": asr_latency,
                "llm_latency": llm_latency,
                "svs_latency": svs_latency,
            })
        return results

    def evaluate(self, audio_path):
        return run_evaluation(audio_path, self.evaluators)
