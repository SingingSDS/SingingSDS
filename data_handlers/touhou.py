from .base import MelodyDatasetHandler


class Touhou(MelodyDatasetHandler):
    name = "touhou"

    def __init__(self, melody_type, *args, **kwargs):
        if melody_type != "note":
            raise ValueError(
                f"Touhou dataset only contains note annotations. {melody_type} is not supported."
            )

        import json

        with open("data/touhou/note_data.json", "r", encoding="utf-8") as f:
            song_db = json.load(f)
        song_db = {song["name"]: song for song in song_db}
        self.song_db = song_db

    def get_song_ids(self):
        return list(self.song_db.keys())

    def get_phrase_length(self, song_id):
        # touhou score does not have phrase segmentation
        return None

    def iter_song_phrases(self, song_id):
        song = self.song_db[song_id]
        song = {
            "tempo": song["tempo"],
            "note_start_times": [n[0] * (100 / song["tempo"]) for n in song["score"]],
            "note_end_times": [n[1] * (100 / song["tempo"]) for n in song["score"]],
            "note_lyrics": ["" for n in song["score"]],
            "note_midi": [n[2] for n in song["score"]],
        }
        # touhou score does not have phrase segmentation
        yield song
