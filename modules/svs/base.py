from abc import ABC, abstractmethod

import numpy as np


class AbstractSVSModel(ABC):
    @abstractmethod
    def __init__(
        self, model_id: str, device: str = "cpu", cache_dir: str = "cache", **kwargs
    ): ...

    @abstractmethod
    def synthesize(
        self,
        score: list[tuple[float, float, str, int]],
        language: str,
        speaker: str,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize singing audio from music score.
        """
        pass
