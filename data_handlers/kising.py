from .base import MelodyDatasetHandler


class KiSing(MelodyDatasetHandler):
    name = "kising"

    def __init__(self, melody_type, cache_dir, *args, **kwargs):
        # melody_type: support alignment type for "sample" melody source
        import json

        from datasets import load_dataset

        song_db = load_dataset(
            "espnet/kising_score_segments", cache_dir=cache_dir, split="train"
        ).to_pandas()
        song_db.set_index("segment_id", inplace=True)
        assert (
            song_db.index.is_unique
        ), "KiSing score segments should have unique segment_id."
        if melody_type == "lyric":
            with open("data/kising/song2word_lengths.json", "r") as f:
                song2word_lengths = json.load(f)
        elif melody_type == "note":
            with open("data/kising/song2note_lengths.json", "r") as f:
                song2word_lengths = json.load(f)
        self.song_db = song_db
        self.song2word_lengths = song2word_lengths

    def get_song_ids(self):
        return list(self.song2word_lengths.keys())

    def get_phrase_length(self, song_id):
        return self.song2word_lengths[song_id]

    def iter_song_phrases(self, song_id):
        segment_id = 1
        while f"{song_id}_{segment_id:03d}" in self.song_db.index:
            segment = self.song_db.loc[f"{song_id}_{segment_id:03d}"].to_dict()
            segment["note_lyrics"] = [
                lyric.strip("<>") if lyric in ["<AP>", "<SP>"] else lyric
                for lyric in segment["note_lyrics"]
            ]
            yield segment
            segment_id += 1
