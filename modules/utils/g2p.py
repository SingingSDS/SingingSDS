import json
import re
import warnings
from pathlib import Path

from kanjiconv import KanjiConv
from pypinyin import lazy_pinyin

from .resources.pinyin_dict import PINYIN_DICT

kanji_to_kana = KanjiConv()

yoon_map = {
    "ぁ": "あ",
    "ぃ": "い",
    "ぅ": "う",
    "ぇ": "え",
    "ぉ": "お",
    "ゃ": "や",
    "ゅ": "ゆ",
    "ょ": "よ",
    "ゎ": "わ",
}

# ACE_phonemes
with open(Path(__file__).parent / "resources" / "all_plans.json", "r") as f:
    ace_phonemes_all_plans = json.load(f)
for plan in ace_phonemes_all_plans["plans"]:
    if plan["language"] == "zh":
        ace_phonemes_zh_plan = plan
        break


def preprocess_text(text: str, language: str) -> list[str]:
    text = text.replace(" ", "")
    if language == "mandarin":
        text_list = to_pinyin(text)
    elif language == "japanese":
        text_list = to_kana(text)
    else:
        raise ValueError(f"Other languages are not supported")
    return text_list


def to_pinyin(text: str) -> list[str]:
    pinyin_list = lazy_pinyin(text)
    text_list = []
    for text in pinyin_list:
        if text[0] == "S" or text[0] == "A" or text[0] == "-":
            sp_strs = re.findall(r"-|AP|SP", text)
            for phn in sp_strs:
                text_list.append(phn)
        else:
            text_list.append(text)
    return text_list


def replace_chouonpu(hiragana_text: str) -> str:
    """process「ー」since the previous packages didn't support"""
    vowels = {
        "あ": "あ",
        "い": "い",
        "う": "う",
        "え": "え",
        "お": "う",
        "か": "あ",
        "き": "い",
        "く": "う",
        "け": "え",
        "こ": "う",
        "さ": "あ",
        "し": "い",
        "す": "う",
        "せ": "え",
        "そ": "う",
        "た": "あ",
        "ち": "い",
        "つ": "う",
        "て": "え",
        "と": "う",
        "な": "あ",
        "に": "い",
        "ぬ": "う",
        "ね": "え",
        "の": "う",
        "は": "あ",
        "ひ": "い",
        "ふ": "う",
        "へ": "え",
        "ほ": "う",
        "ま": "あ",
        "み": "い",
        "む": "う",
        "め": "え",
        "も": "う",
        "や": "あ",
        "ゆ": "う",
        "よ": "う",
        "ら": "あ",
        "り": "い",
        "る": "う",
        "れ": "え",
        "ろ": "う",
        "わ": "あ",
        "を": "う",
    }
    new_text = []
    for i, char in enumerate(hiragana_text):
        if char == "ー" and i > 0:
            prev_char = new_text[-1]
            if prev_char in yoon_map:
                prev_char = yoon_map[prev_char]
            new_text.append(vowels.get(prev_char, prev_char))
        else:
            new_text.append(char)
    return "".join(new_text)


def to_kana(text: str) -> list[str]:
    hiragana_text = kanji_to_kana.to_hiragana(text.replace(" ", ""))
    hiragana_text_wl = replace_chouonpu(hiragana_text).split(" ")
    final_ls = []
    for subword in hiragana_text_wl:
        sl_prev = 0
        for i in range(len(subword) - 1):
            if sl_prev >= len(subword) - 1:
                break
            sl = sl_prev + 1
            if subword[sl] in yoon_map:
                final_ls.append(subword[sl_prev : sl + 1])
                sl_prev += 2
            else:
                final_ls.append(subword[sl_prev])
                sl_prev += 1
        final_ls.append(subword[sl_prev])
    return final_ls


def kana_to_phonemes_openjtalk(kana: str) -> list[str]:
    import pyopenjtalk

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # add space between each character
        kana = " ".join(list(kana))
        # phones is a str object separated by space
        phones = pyopenjtalk.g2p(kana, kana=False)
        if len(w) > 0:
            for warning in w:
                if "No phoneme" in str(warning.message):
                    raise ValueError(f"No phoneme found for {kana}. {warning.message}")
    phones = phones.split(" ")
    return phones


def pinyin_to_phonemes_opencpop(pinyin: str) -> list[str]:
    pinyin = pinyin.lower()
    if pinyin in ace_phonemes_zh_plan["dict"]:
        phns = ace_phonemes_zh_plan["dict"][pinyin]
        return phns
    elif pinyin in ace_phonemes_zh_plan["syllable_alias"]:
        phns = ace_phonemes_zh_plan["dict"][
            ace_phonemes_zh_plan["syllable_alias"][pinyin]
        ]
        return phns
    else:
        raise ValueError(f"{pinyin} not registered in Opencpop phoneme dict")


def pinyin_to_phonemes_ace(pinyin: str) -> list[str]:
    pinyin = pinyin.lower()
    if pinyin in PINYIN_DICT:
        phns = PINYIN_DICT[pinyin]
        return phns
    else:
        raise ValueError(f"{pinyin} not registered in ACE phoneme dict")
