from datasets import load_dataset, concatenate_datasets

ds = load_dataset("espnet/ace-kising-segments", cache_dir="cache")

combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

# 2. filter rows by singer: baber
combined = combined.filter(lambda x: x["singer"] == "barber")

# 3. create a new column, which counts the nonzero numbers in the list in the note_midi column
combined = combined.map(
    lambda x: {"note_midi_length": len([n for n in x["note_midi"] if n != 0])}
)

# 4. sort by segment_id
combined = combined.sort("segment_id")

# 5. iterate over rows
prev_songid = None
prev_song_segment_id = None
song2note_lengths = {}
for row in combined:
    # segment_id: kising_barber_{songid}_{song_segment_id}
    _, _, songid, song_segment_id = row["segment_id"].split("_")
    if prev_songid != songid:
        if prev_songid is not None:
            assert (
                song_segment_id == "001"
            ), f"prev_songid: {prev_songid}, songid: {songid}, song_segment_id: {song_segment_id}"
        song2note_lengths[f"kising_{songid}"] = [row["note_midi_length"]]
    else:
        assert (
            int(song_segment_id) >= int(prev_song_segment_id) + 1
        ), f"prev_song_segment_id: {prev_song_segment_id}, song_segment_id: {song_segment_id}"
        song2note_lengths[f"kising_{songid}"].append(row["note_midi_length"])
    prev_songid = songid
    prev_song_segment_id = song_segment_id

# 6. write to json
import json

with open("song2note_lengths.json", "w") as f:
    json.dump(song2note_lengths, f, indent=4)

# 7. convert to pandas DataFrame
import pandas as pd

df = pd.DataFrame.from_dict(combined)
df = df.drop(columns=["audio", "singer"])
df["segment_id"] = df["segment_id"].str.replace("kising_barber_", "kising_")
# export to csv
df.to_csv("song_db.csv", index=False)

# 8. push score segments to hub
# remove audio and singer columns
combined = combined.remove_columns(["audio", "singer"])
# replace kising_barber_ with kising_
combined = combined.map(
    lambda x: {"segment_id": x["segment_id"].replace("kising_barber_", "kising_")}
)
# upload to hub
combined.push_to_hub("jhansss/kising_score_segments")
