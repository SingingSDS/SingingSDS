from dataclasses import dataclass


@dataclass
class Character:
    name: str
    image_path: str
    default_timbre: str
    prompt: str
