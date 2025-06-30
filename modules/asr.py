import os
from abc import ABC, abstractmethod

import librosa
import numpy as np
from transformers import pipeline

ASR_MODEL_REGISTRY = {}
hf_token = os.getenv("HF_TOKEN")


class AbstractASRModel(ABC):
    @abstractmethod
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, audio_sample_rate: int, **kwargs) -> str:
        pass


def register_asr_model(prefix):
    def wrapper(cls):
        assert issubclass(cls, AbstractASRModel), f"{cls} must inherit AbstractASRModel"
        ASR_MODEL_REGISTRY[prefix] = cls
        return cls

    return wrapper


def get_asr_model(model_id: str, device="cpu", **kwargs) -> AbstractASRModel:
    for prefix, cls in ASR_MODEL_REGISTRY.items():
        if model_id.startswith(prefix):
            return cls(model_id, device=device, **kwargs)
    raise ValueError(f"No ASR wrapper found for model: {model_id}")


@register_asr_model("openai/whisper")
class WhisperASR(AbstractASRModel):
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        model_kwargs = kwargs.setdefault("model_kwargs", {})
        model_kwargs["cache_dir"] = cache_dir
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
            token=hf_token,
            **kwargs,
        )

    def transcribe(self, audio: np.ndarray, audio_sample_rate: int, language: str, **kwargs) -> str:
        if audio_sample_rate != 16000:
            try:
                audio, _ = librosa.resample(audio, orig_sr=audio_sample_rate, target_sr=16000)
            except Exception as e:
                breakpoint()
                print(f"Error resampling audio: {e}")
                audio = librosa.resample(audio, orig_sr=audio_sample_rate, target_sr=16000)
        return self.pipe(audio, generate_kwargs={"language": language}).get("text", "")
