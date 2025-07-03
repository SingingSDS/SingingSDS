import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf

try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None

from .base import AbstractASRModel
from .registry import register_asr_model


@register_asr_model("funasr/paraformer-zh")
class ParaformerASR(AbstractASRModel):
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        super().__init__(model_id, device, cache_dir, **kwargs)

        if AutoModel is None:
            raise ImportError(
                "funasr is not installed. Please install it with: pip3 install -U funasr"
            )

        model_name = model_id.replace("funasr/", "")
        language = model_name.split("-")[1]
        if language == "zh":
            self.language = "mandarin"
        elif language == "en":
            self.language = "english"
        else:
            raise ValueError(
                f"Language cannot be determined. {model_id} is not supported"
            )

        try:
            original_cache_dir = os.getenv("MODELSCOPE_CACHE")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["MODELSCOPE_CACHE"] = cache_dir
            self.model = AutoModel(
                model=model_name,
                model_revision="v2.0.4",
                vad_model="fsmn-vad",
                vad_model_revision="v2.0.4",
                punc_model="ct-punc-c",
                punc_model_revision="v2.0.4",
                device=device,
            )
            if original_cache_dir:
                os.environ["MODELSCOPE_CACHE"] = original_cache_dir
            else:
                del os.environ["MODELSCOPE_CACHE"]

        except Exception as e:
            raise ValueError(f"Error loading Paraformer model: {e}")

    def transcribe(
        self,
        audio: np.ndarray,
        audio_sample_rate: int,
        language: Optional[str] = None,
        **kwargs,
    ) -> str:
        if language and language != self.language:
            raise ValueError(
                f"Paraformer model {self.model_id} only supports {self.language} language, but {language} was requested"
            )

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, audio_sample_rate)
                temp_file = f.name

            result = self.model.generate(input=temp_file, batch_size_s=300, **kwargs)

            os.unlink(temp_file)

            print(f"Transcription result: {result}, type: {type(result)}")

            return result[0]["text"]
        except Exception as e:
            raise ValueError(f"Error during transcription: {e}")
