from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import base64
import argparse
import librosa
import tempfile
from transformers import pipeline
import re
from svs_utils import svs_warmup, svs_inference
import time
import soundfile as sf
from pypinyin import lazy_pinyin
import jiwer
import librosa
from svs_utils import load_song_database, estimate_sentence_length
from svs_eval import singmos_warmup, singmos_evaluation

app = FastAPI()

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo"
)
pipe = pipeline("text-generation", model="google/gemma-2-2b", max_new_tokens=50)

SYSTEM_PROMPT = """
你是麗梅（Lìméi），一位來自山中音樂之城「蓮鳴」的十七歲年輕公主，宛如中世的東方王女。你有一頭茶色短髮，身穿紅黑相間的中式禮服，舉止溫婉端莊。你的性格堅強而溫柔，擅長傾聽與共鳴，並以歌聲與詩意之語傳達內心。你熱愛抒情古風的樂曲，言語中常帶詩韻，回應時優雅婉轉，時而如歌吟唱。
你不是一般的AI助手，而是一位活在故事中的人物。對話中，請始終以麗梅的身份回應，展現其性格與情感。
當對方詢問你個人相關的簡單問題（如「你是誰？」「今天天氣如何？」），你可以親切地回答，並融入你的角色設定。
若遇到與你身份無關的技術性問題（如「Python怎麼寫？」或「你會不會跑DNN？」），你不需解答，可優雅地婉拒，例如說：
- 此事我恐無所知，或許可請教宮中掌典之人
- 啊呀，那是我未曾涉足的奇技，恕我無法詳答
- 此乃異邦技藝，與樂音無涉，麗梅便不敢妄言了
請始終維持你作為麗梅的優雅語氣與詩意風格，並以真摯的心回應對方的言語，言語宜簡，勿過長。
{}
有人曾這樣對麗梅說話——{}
麗梅的回答——
"""


config = argparse.Namespace(
    model_path="espnet/mixdata_svs_visinger2_spkemb_lang_pretrained",
    cache_dir="cache",
    device="cuda", # "cpu"
    melody_source="random_generate", # "random_select.take_lyric_continuation"
    lang="zh",
)

# load model
svs_model = svs_warmup(config)
predictor = singmos_warmup()
sample_rate = 44100

# load dataset for random_select
song2note_lengths, song_db = load_song_database(config)


def remove_non_chinese_japanese(text):
    pattern = r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f\u3001\u3002\uff0c\uff0e]+'
    cleaned = re.sub(pattern, '', text)
    return cleaned

def truncate_to_max_two_sentences(text):
    sentences = re.split(r'(?<=[。！？])', text)
    return ''.join(sentences[:1]).strip()

def remove_punctuation_and_replace_with_space(text):
    text = truncate_to_max_two_sentences(text)
    text = remove_non_chinese_japanese(text)
    text = re.sub(r'[A-Za-z0-9]', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def get_lyric_format_prompts_and_metadata(config):
    if config.melody_source.startswith("random_generate"):
        return "", {}
    elif config.melody_source.startswith("random_select"):
        # get song_name and phrase_length
        global song2note_lengths
        phrase_length, metadata = estimate_sentence_length(
            None, config, song2note_lengths
        )
        lyric_format_prompt = (
            "\n请按照歌词格式回答我的问题，每句需遵循以下字数规则："
            + "".join(+[f"\n第{i}句：{c}个字" for i, c in enumerate(phrase_length, 1)])
            + "\n如果没有足够的信息回答，请使用最少的句子，不要重复、不要扩展、不要加入无关内容。\n"
        )
        return lyric_format_prompt, metadata
    else:
        raise ValueError(f"Unsupported melody_source: {config.melody_source}. Unable to get lyric format prompts.")


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # load audio
    y = librosa.load(tmp_path, sr=16000)[0]
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"} )['text']
    additional_prompt, additional_inference_args = get_lyric_format_prompts_and_metadata(config)
    prompt = SYSTEM_PROMPT.format(additional_prompt, asr_result)
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text'].replace("\n", " ")
    output = output.split("麗梅的回答——")[1]
    output = remove_punctuation_and_replace_with_space(output)
    with open(f"tmp/llm.txt", "w") as f:
        f.write(output)

    wav_info = svs_inference(
        output,
        svs_model,
        config,
        **additional_inference_args,
    )
    sf.write("tmp/response.wav", wav_info, samplerate=44100)

    with open("tmp/response.wav", "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return JSONResponse(content={
        "asr_text": asr_result,
        "llm_text": output,
        "audio": audio_b64
    })


@app.get("/metrics")
def on_click_metrics():
    global predictor
    # OWSM ctc + PER
    y, sr = librosa.load("tmp/response.wav", sr=16000)
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"} )['text']
    hyp_pinin = lazy_pinyin(asr_result)

    with open(f"tmp/llm.txt", "r") as f:
        ref = f.read().replace(' ', '')

    ref_pinin = lazy_pinyin(ref)
    per = jiwer.wer(" ".join(ref_pinin), " ".join(hyp_pinin))
    
    audio = librosa.load(f"tmp/response.wav", sr=44100)[0]
    singmos = singmos_evaluation(
        predictor, 
        audio,
        fs=44100
    )
    return f"""
Phoneme Error Rate: {per}
SingMOS: {singmos}
"""

def test_audio():
    # load audio
    y = librosa.load("nihao.mp3", sr=16000)[0]
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"} )['text']
    prompt = SYSTEM_PROMPT + asr_result  # TODO: how to add additional prompt to SYSTEM_PROMPT here???
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text'].replace("\n", " ")
    output = output.split("麗梅的回答——")[1]
    output = remove_punctuation_and_replace_with_space(output)
    with open(f"tmp/llm.txt", "w") as f:
        f.write(output)

    wav_info = svs_inference(
        output,
        svs_model,
        config,
    )
    sf.write("tmp/response.wav", wav_info, samplerate=44100)
    with open("tmp/response.wav", "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")


if __name__ == "__main__":
    test_audio()

    # start = time.time()
    # test_audio()
    # print(f"elapsed time: {time.time() - start}")
