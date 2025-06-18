import os
import json
import warnings
from typing import List
import re

from resource.pinyin_dict import PINYIN_DICT
from pypinyin import pinyin, Style
from zhconv import convert


def preprocess_input(src_str, seg_syb=" "):
    src_str = src_str.replace("\n", seg_syb)
    src_str = src_str.replace(" ", seg_syb)
    return src_str


def postprocess_phn(phns, model_name, lang):
    if model_name == "espnet/aceopencpop_svs_visinger2_40singer_pretrain":
        return phns
    return [phn + "@" + lang for phn in phns]


def pyopenjtalk_g2p(text) -> List[str]:
    import pyopenjtalk
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # add space between each character
        text = " ".join(list(text))
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
    if model == "espnet/aceopencpop_svs_visinger2_40singer_pretrain":
        if lang == "zh":
            return lambda text: split_pinyin_py(text)
        else:
            raise ValueError(f"Only support Chinese language for {model}")
    elif model == "espnet/mixdata_svs_visinger2_spkemb_lang_pretrained":
        if lang == "zh":
            with open(os.path.join("resource/all_plans.json"), "r") as f:
                all_plan_dict = json.load(f)
            for plan in all_plan_dict["plans"]:
                if plan["language"] == "zh":
                    zh_plan = plan
            return lambda text: split_pinyin_ace(text, zh_plan)
        elif lang == "jp":
            return pyopenjtalk_g2p
        else:
            raise ValueError(f"Only support Chinese and Japanese language for {model}")
    else:
        raise ValueError(f"Only support espnet/aceopencpop_svs_visinger2_40singer_pretrain and espnet/mixdata_svs_visinger2_spkemb_lang_pretrained for now")


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


def is_special(block):
    return any(token in block for token in ['-', 'AP', 'SP'])


def get_pinyin(texts):
    texts = preprocess_input(texts, seg_syb="")
    blocks = re.compile(r'[\u4e00-\u9fff]|[^\u4e00-\u9fff]+').findall(texts) 

    characters = [block for block in blocks if is_chinese(block)] 
    chinese_text = ''.join(characters)
    chinese_text = convert(chinese_text, 'zh-cn')
    
    chinese_pinyin = pinyin(chinese_text, style=Style.NORMAL)
    chinese_pinyin = [item[0] for item in chinese_pinyin]
    
    text_list = []
    pinyin_idx = 0
    for block in blocks:
        if is_chinese(block):
            text_list.append(chinese_pinyin[pinyin_idx])
            pinyin_idx += 1
        else:
            if is_special(block):
                specials = re.compile(r"-|AP|SP").findall(block)
                text_list.extend(specials)
            else:
                text_list.append(block)
    
    return text_list
