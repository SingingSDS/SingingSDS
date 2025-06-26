from abc import ABC, abstractmethod


class MelodyDatasetHandler(ABC):
    name: str

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_song_ids(self) -> list[str]:
        pass

    @abstractmethod
    def get_phrase_length(self, song_id):
        pass

    @abstractmethod
    def iter_song_phrases(self, song_id):
        pass
