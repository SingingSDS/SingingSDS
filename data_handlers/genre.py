from .base import MelodyDatasetHandler


class Genre(MelodyDatasetHandler):
    name = "genre"

    def __init__(self, melody_type, *args, **kwargs):
        import json

        with open("data/genre/word_data.json", "r", encoding="utf-8") as f:
            song_db = json.load(f)
        song_db = {song["id"]: song for song in song_db} # id as major
        self.song_db = song_db

    def get_song_ids(self):
        return list(self.song_db.keys())

    def get_style_keywords(self, song_id):
        genre = self.song_db[song_id]["genre"]
        super_genre = self.song_db[song_id]["super-genre"]
        gender = self.song_db[song_id]["gender"]
        return (genre, super_genre, gender)

    def get_phrase_length(self, song_id):
        # Return the number of lyrics (excluding SP/AP) in each phrase of the song
        song = self.song_db[song_id]
        note_lyrics = song.get("note_lyrics", [])
        
        phrase_lengths = []
        for phrase in note_lyrics:
            count = sum(1 for word in phrase if word not in ("SP", "AP"))
            phrase_lengths.append(count)

        return phrase_lengths

    def iter_song_phrases(self, song_id):
        segment_id = 1
        song = self.song_db[song_id]
        for phrase_score, phrase_lyrics in zip(song["score"], song["note_lyrics"]):
            segment = {
                "note_start_times": [n[0] for n in phrase_score],
                "note_end_times": [n[1] for n in phrase_score],
                "note_lyrics": [character for character in phrase_lyrics],
                "note_midi": [n[2] for n in phrase_score],
            }
            yield segment
            segment_id += 1
