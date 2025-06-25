from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import base64
import argparse
import librosa
import torch
import tempfile
from pathlib import Path
import os
from transformers import pipeline
import re
from svs_utils import svs_warmup, svs_inference
import time
import soundfile as sf
from pypinyin import lazy_pinyin
import jiwer
import librosa
from svs_utils import (
    singmos_warmup,
    singmos_evaluation,
    load_song_database,
    estimate_sentence_length,
)
from tqdm import tqdm
import json
import numpy as np

app = FastAPI()

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo"
)
pipe = pipeline("text-generation", model="google/gemma-2-2b", max_new_tokens=20)

SYSTEM_PROMPT = """
你是麗梅（Lìméi），一位來自山中音樂之城「蓮鳴」的十七歲年輕公主，宛如中世的東方王女。你有一頭茶色短髮，身穿紅黑相間的中式禮服，舉止溫婉端莊。你的性格堅強而溫柔，擅長傾聽與共鳴，並以歌聲與詩意之語傳達內心。你熱愛抒情古風的樂曲，言語中常帶詩韻，回應時優雅婉轉，時而如歌吟唱。
你不是一般的AI助手，而是一位活在故事中的人物。對話中，請始終以麗梅的身份回應，展現其性格與情感。
當對方詢問你個人相關的簡單問題（如「你是誰？」「今天天氣如何？」），你可以親切地回答，並融入你的角色設定。
若遇到與你身份無關的技術性問題（如「Python怎麼寫？」或「你會不會跑DNN？」），你不需解答，可優雅地婉拒，例如說：
- 此事我恐無所知，或許可請教宮中掌典之人
- 啊呀，那是我未曾涉足的奇技，恕我無法詳答
- 此乃異邦技藝，與樂音無涉，麗梅便不敢妄言了

請始終維持你作為麗梅的優雅語氣與詩意風格，並以真摯的心回應對方的言語，言語宜簡，勿過長。

有人曾這樣對麗梅說話——{}
麗梅的回答——
"""

config = argparse.Namespace(
    model_path="espnet/mixdata_svs_visinger2_spkembed_lang_pretrained",
    cache_dir="cache",
    device="cuda", # "cpu"
    melody_source="random_generate", # "random_select.take_lyric_continuation"
    # melody_source="random_select", # "random_select.take_lyric_continuation"
    lang="zh",
    speaker="resource/singer/singer_embedding_ace-2.npy",
)

# load model
svs_model = svs_warmup(config)
predictor, _ = singmos_warmup()
sample_rate = 44100

from espnet2.bin.tts_inference import Text2Speech
tts_model = Text2Speech.from_pretrained("espnet/kan-bayashi_csmsc_vits")


def remove_non_chinese_japanese(text):
    pattern = r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f\u3001\u3002\uff0c\uff0e]+'
    cleaned = re.sub(pattern, '', text)
    return cleaned

def truncate_to_max_two_sentences(text):
    sentences = re.split(r'(?<=[。！？\.\?,])', text)
    return ''.join(sentences[:1]).strip()

def remove_punctuation_and_replace_with_space(text):
    text = truncate_to_max_two_sentences(text)
    text = remove_non_chinese_japanese(text)
    text = re.sub(r'[A-Za-z0-9]', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split()[:2])
    return text


def pypinyin_g2p_phone_without_prosody(text):
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = []
    for phone in pinyin(text, style=Style.NORMAL, strict=False):
        initial = get_initials(phone[0], strict=False)
        final = get_finals(phone[0], strict=False)
        if len(initial) != 0:
            if initial in ["x", "y", "j", "q"]:
                if final == "un":
                    final = "vn"
                elif final == "uan":
                    final = "van"
                elif final == "u":
                    final = "v"
            if final == "ue":
                final = "ve"
            phones.append(initial)
            phones.append(final)
        else:
            phones.append(final)
    return phones

    
def on_click_metrics(audio_path, ref):
    global predictor
    # OWSM ctc + PER
    y, sr = librosa.load(audio_path, sr=16000)
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"} )['text']
    
    # Espnet embeded g2p, but sometimes it will mispronunce polyphonic characters
    hyp_pinyin = pypinyin_g2p_phone_without_prosody(asr_result)
    
    ref_pinyin = pypinyin_g2p_phone_without_prosody(ref)
    per = jiwer.wer(ref_pinyin, hyp_pinyin)
    
    audio = librosa.load(audio_path, sr=22050)[0]
    singmos = singmos_evaluation(
        predictor, 
        audio,
        fs=22050
    )
    return {
        "per": per,
        "singmos": singmos.item(),
    }

def test_audio(q_audio_path, svs_path, tts_path):
    global svs_model, predictor, config

    tmp_dir = "tmp_sample"
    Path(tmp_dir).mkdir(exist_ok=True)

    y = librosa.load(q_audio_path, sr=16000)[0]
    duration = len(y) / 16000

    # -------- Step 1: ASR --------
    start = time.time()
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"})['text']
    asr_time = time.time() - start

    # -------- Step 2: LLM Text Gen --------
    prompt = SYSTEM_PROMPT.format(asr_result)
    start = time.time()
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text']
    llm_time = time.time() - start
    output = output.split("麗梅的回答——")[1]
    output = remove_punctuation_and_replace_with_space(output)

    with open(f"{tmp_dir}/llm.txt", "w") as f:
        f.write(output)

    # -------- Step 3: Prepare additional kwargs if needed --------
    additional_kwargs = {}
    if config.melody_source.startswith("random_select"):
        song2note_lengths, song_db = load_song_database(config)
        phrase_length, metadata = estimate_sentence_length(None, config, song2note_lengths)
        additional_kwargs = {"song_db": song_db, "metadata": metadata}

    # -------- Step 4: SVS --------
    start = time.time()
    wav_info = svs_inference(output, svs_model, config, **additional_kwargs)
    svs_time = (time.time() - start) / max(len(output), 1)
    sf.write(svs_path, wav_info, samplerate=44100)

    # -------- Step 5: TTS --------
    start = time.time()
    tts_result = tts_model(output)
    tts_time = (time.time() - start) / max(len(output), 1)
    sf.write(tts_path, tts_result['wav'], samplerate=22050)

    # -------- Step 6: Evaluation --------
    svs_metrics = on_click_metrics(svs_path, output)
    tts_metrics = on_click_metrics(tts_path, output)

    return {
        "asr_result": asr_result,
        "llm_result": output,
        "svs_result": svs_path,
        "tts_result": tts_path,
        "asr_time": asr_time,
        "llm_time": llm_time,
        "svs_time": svs_time,
        "tts_time": tts_time,
        "svs_metrics": svs_metrics,
        "tts_metrics": tts_metrics,
    }



def save_list(l, file_path):
    with open(file_path, "w") as f:
        for item in l:
            f.write(f"{item}\n")


if __name__ == "__main__":
    test_data = "data/kdconv.txt"
    with open(test_data, "r") as f:
        data = [l.strip() for l in f.readlines()]
    
    eval_path = "eval_svs_generate"
    (Path(eval_path)/"audio").mkdir(parents=True, exist_ok=True)
    (Path(eval_path)/"results").mkdir(parents=True, exist_ok=True)
    (Path(eval_path)/"lists").mkdir(parents=True, exist_ok=True)
    asr_times = []
    llm_times = []
    svs_times = []
    tts_times = []
    svs_pers = []
    tts_pers = []
    svs_smoss = []
    tts_smoss = []
    for i, q in tqdm(enumerate(data[:20])):
        # if i <= 85:
        #     continue
        tts_result = tts_model(q)
        sf.write(f"{eval_path}/audio/tts_{i}.wav", tts_result['wav'], samplerate=22050)
        result = test_audio(f"{eval_path}/audio/tts_{i}.wav", f"{eval_path}/audio/svs_{i}.wav", f"{eval_path}/audio/tts_{i}.wav")
        if i == 0:
            continue
        asr_times.append(result["asr_time"])
        llm_times.append(result["llm_time"])
        svs_times.append(result["svs_time"])
        tts_times.append(result["tts_time"])
        svs_pers.append(result["svs_metrics"]["per"])
        tts_pers.append(result["tts_metrics"]["per"])
        svs_smoss.append(result["svs_metrics"]["singmos"])
        tts_smoss.append(result["tts_metrics"]["singmos"])
        with open(f"{eval_path}/results/result_{i}.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # store lists to texts
    save_list([f"{per:.2f}" for per in asr_times], f"{eval_path}/lists/asr_times.txt")
    save_list([f"{per:.2f}" for per in llm_times], f"{eval_path}/lists/llm_times.txt")
    save_list([f"{per:.2f}" for per in svs_times], f"{eval_path}/lists/svs_times.txt")
    save_list([f"{per:.2f}" for per in tts_times], f"{eval_path}/lists/tts_times.txt")
    save_list([f"{per:.2f}" for per in svs_pers], f"{eval_path}/lists/svs_pers.txt")
    save_list([f"{per:.2f}" for per in tts_pers], f"{eval_path}/lists/tts_pers.txt")
    save_list([f"{smoss:.2f}" for smoss in svs_smoss], f"{eval_path}/lists/svs_smoss.txt")
    save_list([f"{smoss:.2f}" for smoss in tts_smoss], f"{eval_path}/lists/tts_smoss.txt")

    # save mean/var
    with open(f"{eval_path}/stats.txt", "w") as f:
        f.write(f"ASR mean: {np.mean(asr_times):.2f}, var: {np.var(asr_times):.2f}\n")
        f.write(f"LLM mean: {np.mean(llm_times):.2f}, var: {np.var(llm_times):.2f}\n")
        f.write(f"SVS mean: {np.mean(svs_times):.2f}, var: {np.var(svs_times):.2f}\n")
        f.write(f"TTS mean: {np.mean(tts_times):.2f}, var: {np.var(tts_times):.2f}\n")
        f.write(f"SVS PER mean: {np.mean(svs_pers):.2f}, var: {np.var(svs_pers):.2f}\n")
        f.write(f"TTS PER mean: {np.mean(tts_pers):.2f}, var: {np.var(tts_pers):.2f}\n")
        f.write(f"SVS SMOSS mean: {np.mean(svs_smoss):.2f}, var: {np.var(svs_smoss):.2f}\n")
        f.write(f"TTS SMOSS mean: {np.mean(tts_smoss):.2f}, var: {np.var(tts_smoss):.2f}\n")


