from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractASRModel(ABC):
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ):
        print(f"Loading ASR model {model_id}...")
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        audio_sample_rate: int,
        language: Optional[str] = None,
        **kwargs,
    ) -> str:
        pass
