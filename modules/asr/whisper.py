import os
from typing import Optional

import librosa
import numpy as np
from transformers.pipelines import pipeline

from .base import AbstractASRModel
from .registry import register_asr_model

hf_token = os.getenv("HF_TOKEN")


@register_asr_model("openai/whisper")
class WhisperASR(AbstractASRModel):
    def __init__(
        self, model_id: str, device: str = "auto", cache_dir: str = "cache", **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)
        model_kwargs = kwargs.setdefault("model_kwargs", {})
        model_kwargs["cache_dir"] = cache_dir
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device_map=device,
            token=hf_token,
            **kwargs,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        audio_sample_rate: int,
        language: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio using Whisper model

        Args:
            audio: Audio numpy array
            audio_sample_rate: Sample rate of the audio
            language: Language hint (optional)

        Returns:
            Transcribed text as string
        """
        try:
            # Resample to 16kHz if needed
            if audio_sample_rate != 16000:
                audio = librosa.resample(
                    audio, orig_sr=audio_sample_rate, target_sr=16000
                )

            # Generate transcription
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language

            result = self.pipe(
                audio,
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
                **kwargs,
            )

            # Extract text from result
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            elif isinstance(result, list) and len(result) > 0:
                # Handle list of results
                first_result = result[0]
                if isinstance(first_result, dict):
                    return first_result.get("text", str(first_result))
                else:
                    return str(first_result)
            else:
                return str(result)

        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
            return ""
