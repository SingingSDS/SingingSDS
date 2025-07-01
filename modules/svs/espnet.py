from typing import Callable

import numpy as np

from modules.utils.g2p import (
    kana_to_phonemes_openjtalk,
    pinyin_to_phonemes_ace,
    pinyin_to_phonemes_opencpop,
)

from .base import AbstractSVSModel
from .registry import register_svs_model


@register_svs_model("espnet/")
class ESPNetSVS(AbstractSVSModel):
    def __init__(self, model_id: str, device="cpu", cache_dir="cache", **kwargs):
        from espnet2.bin.svs_inference import SingingGenerate
        from espnet_model_zoo.downloader import ModelDownloader

        print(f"Downloading {model_id} to {cache_dir}") # TODO: should improve log code
        downloaded = ModelDownloader(cache_dir).download_and_unpack(model_id)
        print(f"Downloaded {model_id} to {cache_dir}") # TODO: should improve log code
        self.model = SingingGenerate(
            train_config=downloaded["train_config"],
            model_file=downloaded["model_file"],
            device=device,
        )
        self.model_id = model_id
        self.output_sample_rate = self.model.fs
        self.phoneme_mappers = self._build_phoneme_mappers()

    def _build_phoneme_mappers(self) -> dict[str, Callable[[str], list[str]]]:
        if self.model_id == "espnet/aceopencpop_svs_visinger2_40singer_pretrain":
            phoneme_mappers = {
                "mandarin": pinyin_to_phonemes_opencpop,
            }
        elif self.model_id == "espnet/mixdata_svs_visinger2_spkemb_lang_pretrained":

            def mandarin_mapper(pinyin: str) -> list[str]:
                phns = pinyin_to_phonemes_ace(pinyin)
                return [phn + "@zh" for phn in phns]

            def japanese_mapper(kana: str) -> list[str]:
                phones = kana_to_phonemes_openjtalk(kana)
                return [phn + "@jp" for phn in phones]

            phoneme_mappers = {
                "mandarin": mandarin_mapper,
                "japanese": japanese_mapper,
            }
        else:
            phoneme_mappers = {}
        return phoneme_mappers

    def _preprocess(self, score: list[tuple[float, float, str, int] | tuple[float, float, str, float]], language: str):
        if language not in self.phoneme_mappers:
            raise ValueError(f"Unsupported language: {language} for {self.model_id}")
        phoneme_mapper = self.phoneme_mappers[language]

        # text to phoneme
        notes = []
        phns = []
        pre_phn = None
        for st, ed, text, pitch in score:
            assert text not in [
                "<AP>",
                "<SP>",
            ], f"Proccessed score segments should not contain <AP> or <SP>. {score}"  # TODO: remove in PR, only for debug
            if text == "AP" or text == "SP":
                lyric_units = [text]
                phn_units = [text]
            elif text == "-" or text == "——":
                lyric_units = [text]
                if pre_phn is None:
                    raise ValueError(
                        f"Text `{text}` cannot be recognized by {self.model_id}. Lyrics cannot start with a lyric continuation symbol `-` or `——`"
                    )
                phn_units = [pre_phn]
            else:
                try:
                    lyric_units = phoneme_mapper(text)
                except ValueError as e:
                    raise ValueError(
                        f"Text `{text}` cannot be recognized by {self.model_id}"
                    ) from e
                phn_units = lyric_units
            notes.append((st, ed, "".join(lyric_units), pitch, "_".join(phn_units)))
            phns.extend(phn_units)
            pre_phn = phn_units[-1]

        batch = {
            "score": {
                "tempo": 120,  # does not affect svs result, as note durations are in time unit
                "notes": notes,
            },
            "text": " ".join(phns),
        }
        return batch

    def synthesize(
        self, score: list[tuple[float, float, str, float] | tuple[float, float, str, int]], language: str, speaker: str, **kwargs
    ):
        batch = self._preprocess(score, language)
        if self.model_id == "espnet/aceopencpop_svs_visinger2_40singer_pretrain":
            sid = np.array([int(speaker)])
            output_dict = self.model(batch, sids=sid)
        elif self.model_id == "espnet/mixdata_svs_visinger2_spkemb_lang_pretrained":
            langs = {
                "zh": 2,
                "jp": 1,
            }
            if language not in langs:
                raise ValueError(
                    f"Unsupported language: {language} for {self.model_id}"
                )
            lid = np.array([langs[language]])
            spk_embed = np.load(speaker)
            output_dict = self.model(batch, lids=lid, spembs=spk_embed)
        else:
            raise NotImplementedError(f"Model {self.model_id} not supported")
        wav_info = output_dict["wav"].cpu().numpy()
        return wav_info, self.output_sample_rate
