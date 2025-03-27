import os
import json
import warnings
from typing import List
import re

import pyopenjtalk
from resource.pinyin_dict import PINYIN_DICT
from pypinyin import lazy_pinyin

def preprocess_input(src_str, seg_syb=" "):
    src_str = src_str.replace("\n", seg_syb)
    src_str = src_str.replace(" ", seg_syb)
    return src_str


def postprocess_phn(phns, model_name, lang):
    if "Chinese" in model_name:
        return phns
    return [phn + "@" + lang for phn in phns]


def pyopenjtalk_g2p(text) -> List[str]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # phones is a str object separated by space
        phones = pyopenjtalk.g2p(text, kana=False)
        if len(w) > 0:
            for warning in w:
                if "No phoneme" in str(warning.message):
                    return False
 
    phones = phones.split(" ")
    return phones


def split_pinyin_ace(pinyin: str, zh_plan: dict) -> tuple[str]:
    # load pinyin dict from local/pinyin.dict
    pinyin = pinyin.lower()
    if pinyin in zh_plan["dict"]:
        return zh_plan["dict"][pinyin]
    elif pinyin in zh_plan["syllable_alias"]:
        return zh_plan["dict"][zh_plan["syllable_alias"][pinyin]]
    else:
        return False


def split_pinyin_py(pinyin: str) -> tuple[str]:
    pinyin = pinyin.lower()
    if pinyin in PINYIN_DICT:
        return PINYIN_DICT[pinyin]
    else:
        return False


def get_tokenizer(model, lang):
    if lang == "zh":
        if "Chinese" in model:
            print("hello")
            return lambda text: split_pinyin_py(text)
        else:
            with open(os.path.join("resource/all_plans.json"), "r") as f:
                all_plan_dict = json.load(f)
            for plan in all_plan_dict["plans"]:
                if plan["language"] == "zh":
                    zh_plan = plan
            return lambda text: split_pinyin_ace(text, zh_plan)
    elif lang == "jp":
        return pyopenjtalk_g2p


def get_pinyin(texts):
    pinyin_list = lazy_pinyin(texts)
    text_list = []
    for text in pinyin_list:
        if text[0] == "S" or text[0] == "A" or text[0] == '-':
            sp_strs = re.findall(r'-|AP|SP', text)
            for phn in sp_strs:
                text_list.append(phn)
        else:
            text_list.append(text)
    return text_list


def load_pitch_dict(file_path = "resource/midi-note.scp"):
    pitch_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            pitch_dict[items[0]] = int(items[1])
            pitch_dict[items[1]] = int(items[1])
    return pitch_dict

