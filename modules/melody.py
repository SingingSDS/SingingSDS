import random
from typing import Iterator

from data_handlers import get_melody_handler

from .utils.g2p import preprocess_text


class MelodyController:
    def __init__(self, melody_source_id: str, cache_dir: str):
        self.melody_source_id = melody_source_id
        self.song_id = None

        # load song database if needed
        parts = self.melody_source_id.split("-")
        self.mode = parts[0]
        self.align_type = parts[1]
        dataset_name = parts[-1]
        if dataset_name == "none":
            self.database = None
        else:
            handler_cls = get_melody_handler(dataset_name)
            self.database = handler_cls(self.align_type, cache_dir)

    def get_melody_constraints(self, max_num_phrases: int = 5) -> str:
        """Return a lyric-format prompt based on melody structure."""
        if self.mode == "gen":
            return ""

        elif self.mode == "sample":
            assert self.database is not None, "Song database is not loaded."
            self.song_id = random.choice(self.database.get_song_ids())
            self.reference_song = self.database.iter_song_phrases(self.song_id)
            phrase_length = self.database.get_phrase_length(self.song_id)

            if not phrase_length:
                return ""

            prompt = (
                "\n请按照歌词格式回答我的问题，每句需遵循以下字数规则："
                + "".join(
                    [
                        f"\n第{i}句：{c}个字"
                        for i, c in enumerate(phrase_length[:max_num_phrases], 1)
                    ]
                )
                + "\n如果没有足够的信息回答，请使用最少的句子，不要重复、不要扩展、不要加入无关内容。\n"
            )
            return prompt

        else:
            raise ValueError(f"Unsupported melody mode: {self.mode}")

    def generate_score(
        self, lyrics: str, language: str
    ) -> list[tuple[float, float, str, int]]:
        """
        lyrics: [lyric, ...]
        returns: [(start, end, lyric, pitch), ...]
        """
        text_list = preprocess_text(lyrics, language)
        if self.mode == "gen" and self.align_type == "random":
            return self._generate_random_score(text_list)

        elif self.mode == "sample":
            if not self.reference_song:
                raise RuntimeError(
                    "Must call get_melody_constraints() before generate_score() in sample mode."
                )
            return self._align_text_to_score(
                text_list, self.reference_song, self.align_type
            )

        else:
            raise ValueError(f"Unsupported melody_source_id: {self.melody_source_id}")

    def _generate_random_score(self, text_list: list[str]):
        st = 0
        score = []
        for lyric in text_list:
            pitch = random.randint(57, 69)
            duration = round(random.uniform(0.1, 0.5), 4)
            ed = st + duration
            score.append((st, ed, lyric, pitch))
            st = ed
        return score

    def _align_text_to_score(
        self,
        text_list: list[str],
        song_phrase_iterator: Iterator[dict],
        align_type: str,
    ):
        score = []
        text_idx = 0

        while text_idx < len(text_list):
            reference = next(song_phrase_iterator)
            for st, ed, ref_lyric, pitch in zip(
                reference["note_start_times"],
                reference["note_end_times"],
                reference["note_lyrics"],
                reference["note_midi"],
            ):
                assert ref_lyric not in [
                    "<AP>",
                    "<SP>",
                ], f"Proccessed {self.melody_source_id} score segments should not contain <AP> or <SP>."  # TODO: remove in PR, only for debug
                if pitch == 0:
                    score.append((st, ed, ref_lyric, pitch))
                elif ref_lyric in ["-", "——"] and align_type == "lyric":
                    score.append((st, ed, "-", pitch))
                else:
                    score.append((st, ed, text_list[text_idx], pitch))
                    text_idx += 1
                    if text_idx >= len(text_list):
                        break
        return score
